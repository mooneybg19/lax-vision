"""Held-out evaluation in YARDS: detect -> calibrate -> compare with the label's mapping.

  python pipeline/eval_field.py --run RUN --frames results/label_frames [--split test] [--old_unet models_cal/field_kp.pt]

For every frame of the split (test = the whole held-out game + ~15% of the dev game, never trained or validated on):
image grid points every 16 px whose LABEL field position (label H from game_solve + the game's lens k1) is on the
field (0..110 x 0..W) are mapped to the field with the predicted calibration too; the error is the distance in
yards. Reported: success rate (calibration returned an H that passed the sanity checks), median / p90 over all
points of calibrated frames, the median / p90 of per-frame medians, and the same with failed frames counted as
infinite error. Configurations:
  A  kp net only, field type known, lens unknown (k1 = 0)      <- the engine's situation for a known field
  B  kp + line nets, field type known, k1 = 0
  C  kp + line nets, field type auto (tries 60 / 53.33 lacrosse, football HS / college)
  D  kp + line nets, field type known, the game's k1 known (upper bound for a per-game lens fit)
  old_unet  the earlier synthetic-data UNet + field_cal (60-yd world), if --old_unet is given
"""
import argparse, json, os, sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels_io as LI
import field_targets as FT
import field_calib as FC
import field_data as FD
import field_infer as FI
from synth_cam import distort


def legacy_unet_detect(frame_bgr, model_path, thresh=0.35):
    """The previous synthetic-data UNet detector (train_field_kp.UNet) -> {lax_field name: (u, v)} (frame px)."""
    from train_field_kp import UNet
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    net = UNet(len(ck["names"])); net.load_state_dict(ck["model"]); net.eval()
    H, W = frame_bgr.shape[:2]
    im = cv2.resize(frame_bgr.astype(np.float32) / 255.0, (224, 128)).transpose(2, 0, 1)
    with torch.no_grad():
        hm = net(torch.from_numpy(im[None]))[0].numpy()
    out = {}
    for c, nm in enumerate(ck["names"]):
        j = int(hm[c].argmax()); v, u = divmod(j, hm[c].shape[1])
        if hm[c][v, u] >= thresh:
            out[nm] = (u * W / hm[c].shape[1], v * H / hm[c].shape[0])
    return out


def draw(img, H, k1, W, col, th=2):
    for a, b in list(FT.line_world(W, 17.78).values())[:13]:
        t = np.linspace(0, 1, 80)[:, None]
        P = np.array(a, float) + t * (np.array(b, float) - np.array(a, float))
        q, w = LI.proj(H, P)
        q = distort(q, k1)
        ok = (w > 0) & (np.abs(q[:, 0]) < 5000) & (np.abs(q[:, 1]) < 5000)
        pts = q[ok]
        if len(pts) > 1:
            cv2.polylines(img, [np.round(pts * 4).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--frames", default="results/label_frames")
    ap.add_argument("--split", default="test"); ap.add_argument("--kp", default=None); ap.add_argument("--line", default=None)
    ap.add_argument("--old_unet", default=None); ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    P = json.load(open(os.path.join(a.run, "prepared.json")))
    by = {d["frame"]: d for d in P["docs"]}
    docs = [by[f] for f in P[a.split]][: a.limit or None]
    kp_path = a.kp or os.path.join(a.run, "kp_best.pt")
    line_path = a.line or os.path.join(a.run, "line_best.pt")
    det = FI.FieldDetector(kp_path, line_path if os.path.exists(line_path) else None)
    has_lines = det.line_net is not None
    out_dir = os.path.join(a.run, "eval"); os.makedirs(out_dir, exist_ok=True)
    configs = ["A"] + (["B", "C", "D"] if has_lines else ["C_kp", "D_kp"])
    per = {c: {} for c in configs}
    if a.old_unet:
        per["old_unet"] = {}
    info_log = {}
    for i, d in enumerate(docs):
        im = FD.load_image(a.frames, d["frame"])
        W, hashD, fb, Hl, k1l = FT.doc_geometry(d)
        lp = d.get("lax_painted", True)
        kl, ln = det.raw(im)
        kpm, _ = FT.masks(fb, lp)
        allowed = {n for n, m in zip(FT.KP_NAMES, kpm) if m}
        kp, lines, border = det.decode(kl, ln, fb, lp)
        res = {}
        res["A"] = FC.calibrate(kp, W, hashD, fb, k1=0.0, allowed=allowed)
        if has_lines:
            res["B"] = FC.calibrate(kp, W, hashD, fb, lines=lines, border=border, k1=0.0, allowed=allowed)
            res["D"] = FC.calibrate(kp, W, hashD, fb, lines=lines, border=border, k1=k1l, allowed=allowed)
            Hc, ic = det.calibrate(im, field="auto", k1=0.0)
            res["C"] = (Hc, ic)
        else:
            res["D_kp"] = FC.calibrate(kp, W, hashD, fb, k1=k1l, allowed=allowed)
            res["C_kp"] = det.calibrate(im, field="auto", k1=0.0, use_lines=False)
        info_log[d["frame"]] = {}
        for c, (H, info) in res.items():
            k1p = k1l if c.startswith("D") else 0.0
            Wp = info.get("width", W)
            per[c][d["frame"]] = None if H is None else FC.yard_errors(Hl, k1l, H, k1p, W)
            info_log[d["frame"]][c] = {k: v for k, v in info.items() if not k.startswith("_")}
        if a.old_unet:
            import field_cal as OLD
            import lax_field as LF
            kpo = legacy_unet_detect(im, a.old_unet)
            names = [n for n in kpo if n in LF.KEYPOINTS]
            Ho = None
            if len(names) >= 4:
                Ho, _ = OLD.homography_from_points([LF.KEYPOINTS[n] for n in names], [kpo[n] for n in names])
            per["old_unet"][d["frame"]] = None if Ho is None else FC.yard_errors(Hl, k1l, Ho, 0.0, W)
        if i < a.overlays:
            ov = im.copy()
            draw(ov, Hl, k1l, W, (0, 255, 0), 3)
            best_c = "B" if has_lines else "A"
            if res[best_c][0] is not None:
                draw(ov, res[best_c][0], 0.0, W, (255, 0, 255), 2)
            for n, (u, v, s) in kp.items():
                cv2.circle(ov, (int(round(u)), int(round(v))), 4, (0, 255, 255), 1)
            e = per[best_c][d["frame"]]
            txt = f"{d['frame']}  {best_c}: " + ("FAILED " + str(res[best_c][1].get("reason")) if e is None else f"median {np.median(e):.2f} yd  p90 {np.percentile(e, 90):.2f} yd")
            cv2.putText(ov, txt, (10, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, f"{d['frame']}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    report = {"split": a.split, "n_frames": len(docs), "kp_model": kp_path, "line_model": line_path if has_lines else None,
              "overall": {c: FC.summarize(v, len(docs)) for c, v in per.items()}, "by_game": {}}
    games = sorted({d["game"] for d in docs})
    for g in games:
        fs = {d["frame"] for d in docs if d["game"] == g}
        report["by_game"][g] = {c: FC.summarize({k: v for k, v in pv.items() if k in fs}, len(fs)) for c, pv in per.items()}
    report["per_frame"] = {c: {k: (None if v is None else float(np.median(v))) for k, v in pv.items()} for c, pv in per.items()}
    report["calib_info"] = info_log
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=float)
    lines = [f"# Field calibration, held-out {a.split} split ({len(docs)} frames)", "",
             "| config | frames | calibrated | points median yd | points p90 yd | per-frame median (median) | per-frame median (p90) | all frames, fail=inf: median / p90 |",
             "|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "-" if x is None else ("inf" if x == float("inf") else f"{x:.2f}")
    for c, S in report["overall"].items():
        lines.append(f"| {c} | {S['frames']} | {S['calibrated']} ({100 * S['success_rate']:.0f}%) | {fmt(S['points_median_yd'])} | "
                     f"{fmt(S['points_p90_yd'])} | {fmt(S['frame_median_of_medians_yd'])} | {fmt(S['frame_p90_of_medians_yd'])} | "
                     f"{fmt(S['all_frames_median_yd_fail_inf'])} / {fmt(S['all_frames_p90_yd_fail_inf'])} |")
    for g in games:
        lines += ["", f"## {g}", "", "| config | calibrated | points median yd | points p90 yd |", "|---|---|---|---|"]
        for c, S in report["by_game"][g].items():
            lines.append(f"| {c} | {S['calibrated']}/{S['frames']} | {fmt(S['points_median_yd'])} | {fmt(S['points_p90_yd'])} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "report.md"), "w").write(md)
    print(md)
    return report


if __name__ == "__main__":
    main()
