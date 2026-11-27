"""Finetune the HRNet field detectors (kp net, then line net) from PnLCalib's soccer weights on the hand labels.

  python pipeline/train_field.py prepare --labels answers/field_labels.json --frames results/label_frames --out RUN
  python pipeline/train_field.py train --kind kp   --out RUN --weights WEIGHTS_DIR [--epochs 60 ...]
  python pipeline/train_field.py train --kind line --out RUN --weights WEIGHTS_DIR

prepare: validate labels (labels_io), per-game joint solve (game_solve: lens k1, hash, outlier clicks), corrected
docs + split (field_data.split) -> RUN/prepared.json, RUN/game_solve.json.
train: checkpoints RUN/<kind>_last.pt every epoch (model, optimiser, scaler, schedule, RNG epoch) and
RUN/<kind>_best.pt; rerunning the same command resumes from <kind>_last.pt. Stages 1-2 are frozen for the first
--freeze_epochs epochs. Best epoch: kp = lowest validation yard score (mean over val frames of min(frame median
yd error, 10), failures count 10); line = lowest validation loss.
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels_io as LI
import game_solve as GS
import field_targets as FT
import field_data as FD
import field_model as FM
import field_calib as FC


def prepare(labels, frames, out, use_paint=True, fit_width=False, holdout_game=FD.HOLDOUT_GAME, dev_game=FD.DEV_GAME):
    os.makedirs(out, exist_ok=True)
    docs, rep = LI.load_valid(labels)
    if not docs:
        raise SystemExit("no valid labels")
    missing = [d["frame"] for d in docs if not os.path.exists(os.path.join(frames, d["frame"] + ".jpg"))]
    if missing:
        print(f"WARNING: {len(missing)} labelled frames have no jpg in {frames}; dropped: {missing[:5]}")
        docs = [d for d in docs if d["frame"] not in set(missing)]
    solved = GS.solve_all(docs, frames_dir=frames, use_paint=use_paint, fit_width=fit_width)
    fixed = GS.apply(docs, solved)
    FT.annotate_games(fixed)
    dropped = sorted(set(d["frame"] for d in docs) - set(d["frame"] for d in fixed))
    tr, va, te = FD.split(fixed, holdout_game, dev_game)
    json.dump({"docs": fixed, "train": [d["frame"] for d in tr], "val": [d["frame"] for d in va],
               "test": [d["frame"] for d in te], "dropped_by_game_solve": dropped, "label_report": rep},
              open(os.path.join(out, "prepared.json"), "w"))
    json.dump(solved, open(os.path.join(out, "game_solve.json"), "w"), indent=1)
    print(f"prepared: {len(fixed)} frames ({len(dropped)} dropped by game_solve flags) -> train {len(tr)} val {len(va)} test {len(te)}")
    for g, r in solved.items():
        print(f"  {g}: k1 {r['k1']:+.4f}+-{r['k1_se']:.4f}  hash {r['hash']}  click rms {r['before']['click_rms_px']:.2f}"
              f" -> {r['after']['click_rms_px']:.2f}px  flagged {len(r['flagged_frames'])}")
    return out


def load_prepared(run):
    P = json.load(open(os.path.join(run, "prepared.json")))
    by = {d["frame"]: d for d in P["docs"]}
    return [by[f] for f in P["train"]], [by[f] for f in P["val"]], [by[f] for f in P["test"]]


def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in ("image", "kp_xy", "kp_mask", "ln_xy", "ln_mask", "border")}
    out["index"] = [x["index"] for x in b]
    return out


@torch.no_grad()
def evaluate_kp(net, docs, frames, device, thresh=0.25, max_frames=None, in_wh=(FT.IN_W, FT.IN_H)):
    """Validation/test yard metric for the kp net alone (no line net): detect -> calibrate -> yard errors."""
    net.eval()
    per = {}
    for d in docs[:max_frames]:
        im = FD.load_image(frames, d["frame"])
        x = FD.to_input(im, in_wh)[None].to(device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            kl = net(x)[0].float()
        W, hashD, fb, Hl, k1 = FT.doc_geometry(d)
        kpm, _ = FT.doc_masks(d)
        p = FM.kp_probs(kl[None], torch.from_numpy(kpm)[None].to(kl.device))[0].cpu().numpy()
        kp = FT.decode_kp(p, kpm, thresh=thresh)
        H, info = FC.calibrate(kp, W, hashD, fb, k1=k1, allowed={n for n, m in zip(FT.KP_NAMES, kpm) if m})
        per[d["frame"]] = None if H is None else FC.yard_errors(Hl, k1, H, k1, W)
    return per


def train(args):
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = load_prepared(args.out)
    if args.limit:
        tr, va = tr[:args.limit], va[:max(1, args.limit // 2)]
    if not tr:
        raise SystemExit("no training frames (all rejected or flagged?): see game_solve.json")
    print(f"device {dev}  kind {args.kind}  train {len(tr)} val {len(va)} (test {len(te)} untouched)")
    in_wh = tuple(int(v) for v in args.size.split("x"))
    assert in_wh[1] % 4 == 0 and (in_wh[1] // 4) % 2 == 0 and in_wh[0] % 8 == 0, "size: W % 8 == 0 and H % 8 == 0"
    ds = FD.FieldDataset(tr, args.frames, train=True, seed=args.seed, in_wh=in_wh)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                                     drop_last=len(tr) >= args.bs, pin_memory=dev == "cuda",
                                     persistent_workers=args.workers > 0)
    vds = FD.FieldDataset(va, args.frames, train=False, in_wh=in_wh)
    if not va:
        print("WARNING: empty validation split; the best epoch is picked by TRAIN loss")
    vdl = torch.utils.data.DataLoader(vds, batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate)
    w = os.path.join(args.weights, "SV_kp" if args.kind == "kp" else "SV_lines") if args.weights else None
    net = FM.build(args.kind, w).to(dev)
    head = [p for n, p in net.named_parameters() if n.startswith("head.0.3.")]
    body = [p for n, p in net.named_parameters() if not n.startswith("head.0.3.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr_head}], weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(dl) / args.accum))
    total = steps_per_epoch * args.epochs; warm = steps_per_epoch

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))) * 0.98 + 0.02
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    last = os.path.join(args.out, f"{args.kind}_last.pt"); best_p = os.path.join(args.out, f"{args.kind}_best.pt")
    start, best = 0, float("inf")
    if os.path.exists(last) and not args.restart:
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from {last}: next epoch {start}, best so far {best:.4f}")
    logf = open(os.path.join(args.out, f"{args.kind}_log.jsonl"), "a")
    lossf = FM.kp_loss if args.kind == "kp" else FM.line_loss
    for ep in range(start, args.epochs):
        frozen = ep < args.freeze_epochs
        FM.set_frozen(net, frozen); FM.train_mode(net)
        ds.epoch = ep
        t0 = time.time(); tot = n = 0
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl):
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            kp_t, ln_t = FM.build_targets(b)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                out = net(b["image"])
            loss = lossf(out, kp_t, b["kp_mask"]) if args.kind == "kp" else lossf(out, ln_t, b["ln_mask"])
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0 or it + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            tot += float(loss.detach()); n += 1
            if args.max_steps and it + 1 >= args.max_steps:
                break
        # validation
        net.eval(); vl = vn = 0
        with torch.no_grad():
            for b in vdl:
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                kp_t, ln_t = FM.build_targets(b)
                with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                    out = net(b["image"])
                vl += float(lossf(out, kp_t, b["kp_mask"]) if args.kind == "kp" else lossf(out, ln_t, b["ln_mask"])); vn += 1
        rec = {"epoch": ep, "frozen": frozen, "train_loss": tot / max(1, n), "val_loss": vl / max(1, vn),
               "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        if args.kind == "kp" and va:
            per = evaluate_kp(net, va, args.frames, dev, args.kp_thresh, in_wh=in_wh)
            S = FC.summarize(per, len(va))
            rec["val_yd"] = S
            rec["score"] = float(np.mean([min(np.median(v), 10.0) if v is not None and len(v) else 10.0 for v in per.values()]))
        else:
            rec["score"] = rec["val_loss"] if va else rec["train_loss"]
        is_best = rec["score"] < best
        best = min(best, rec["score"])
        ck = {"kind": args.kind, "model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": best, "kp_names": FT.KP_NAMES, "line_names": FT.LINE_NAMES,
              "kp_thresh": args.kp_thresh, "in_wh": in_wh, "args": vars(args)}
        torch.save(ck, last + ".tmp"); os.replace(last + ".tmp", last)        # atomic: a disconnect can't corrupt it
        if is_best:
            torch.save({k: ck[k] for k in ("kind", "model", "epoch", "best", "kp_names", "line_names", "kp_thresh", "in_wh", "args")},
                       best_p + ".tmp")
            os.replace(best_p + ".tmp", best_p)
        rec["best"] = is_best
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        vy = rec.get("val_yd", {})
        print(f"epoch {ep:3d} {'F' if frozen else ' '} train {rec['train_loss']:.5f} val {rec['val_loss']:.5f}"
              + (f"  val yd med {vy.get('points_median_yd')} p90 {vy.get('points_p90_yd')} ok {vy.get('calibrated')}/{vy.get('frames')}" if vy else "")
              + f"  score {rec['score']:.4f}{' *' if is_best else ''}  {rec['sec']}s", flush=True)
    return best_p


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--labels", required=True); p.add_argument("--frames", required=True); p.add_argument("--out", required=True)
    p.add_argument("--no_paint", action="store_true"); p.add_argument("--fit_width", action="store_true")
    p.add_argument("--holdout_game", default=FD.HOLDOUT_GAME); p.add_argument("--dev_game", default=FD.DEV_GAME)
    t = sub.add_parser("train")
    t.add_argument("--kind", choices=["kp", "line"], required=True)
    t.add_argument("--out", required=True); t.add_argument("--frames", default="results/label_frames")
    t.add_argument("--weights", default=None, help="dir with SV_kp / SV_lines (soccer); omit = random init (tests)")
    t.add_argument("--epochs", type=int, default=60); t.add_argument("--freeze_epochs", type=int, default=8)
    t.add_argument("--bs", type=int, default=2); t.add_argument("--accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--lr_head", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=2); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--kp_thresh", type=float, default=0.25)
    t.add_argument("--size", default=f"{FT.IN_W}x{FT.IN_H}", help="network input WxH (keep 960x540; smaller only for CPU smoke tests)")
    t.add_argument("--limit", type=int, default=0, help="use only N train frames (smoke tests)")
    t.add_argument("--max_steps", type=int, default=0, help="steps per epoch cap (smoke tests)")
    t.add_argument("--restart", action="store_true", help="ignore an existing <kind>_last.pt")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.labels, a.frames, a.out, use_paint=not a.no_paint, fit_width=a.fit_width,
                holdout_game=a.holdout_game, dev_game=a.dev_game)
    else:
        train(a)


if __name__ == "__main__":
    main()
