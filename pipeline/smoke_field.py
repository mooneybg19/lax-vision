"""CPU smoke test of the whole field-training path with FAKE labels (plumbing only, says nothing about accuracy).

Takes a few real frames per game, invents a random sideline camera for each, paints the field lines of that camera
onto a copy of the frame (so game_solve's paint step has something to snap to), writes labeler-format documents,
then runs exactly what the notebook runs: prepare -> train kp (resume check) -> train line -> eval -> field_infer.

  python pipeline/smoke_field.py --work /tmp/smoke --weights /path/with/SV_kp_and_SV_lines [--per_game 5]
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels_io as LI
import synth_cam as SC
import train_field as TF
import eval_field as EF
import field_infer as FI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = {"scituate-hanover-2024": (60.0, False, None), "anne-arundel-carroll-2021": (60.0, False, None),
         "stillwater-bsm-2025": (LI.FB_WIDTH, True, 17.78), "jesuit-westview": (LI.FB_WIDTH, True, 17.78)}


def make_fake(work, per_game, seed=0):
    rng = np.random.default_rng(seed)
    fdir = os.path.join(work, "frames"); os.makedirs(fdir, exist_ok=True)
    src = os.path.join(ROOT, "results", "label_frames")
    docs = []
    for g, (W, fb, h) in GAMES.items():
        frames = sorted(f[:-4] for f in os.listdir(src) if f.startswith(g + "_t"))
        k1 = rng.uniform(-0.05, 0.05)
        n = 0
        for fr in frames:
            if n >= per_game:
                break
            H = SC.camera_H(rng, W)
            d = SC.fake_label(fr, g, H, W, fb, h or 17.78, rng, k1=k1, n_max=7)
            if d is None:
                continue
            base = cv2.imread(os.path.join(src, fr + ".jpg"))
            cv2.imwrite(os.path.join(fdir, fr + ".jpg"), SC.render(H, W, fb, h or 17.78, k1, rng, base=base))
            docs.append(d); n += 1
        docs.append({"frame": frames[-1], "game": g, "skip": True, "at": 0})       # a skipped frame, as exported
    p = os.path.join(work, "field_labels.json")
    json.dump(docs, open(p, "w"))
    return p, fdir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/field_smoke"); ap.add_argument("--weights", default=None)
    ap.add_argument("--per_game", type=int, default=5)
    a = ap.parse_args(argv)
    if os.path.exists(a.work):
        shutil.rmtree(a.work)
    os.makedirs(a.work)
    t0 = time.time()
    labels, fdir = make_fake(a.work, a.per_game)
    run = os.path.join(a.work, "run")
    TF.main(["prepare", "--labels", labels, "--frames", fdir, "--out", run])
    # 480x272 input: the CPU here has 7 GB, HRNet-w48 training at 960x540 needs more. Same code path otherwise.
    common = ["--out", run, "--frames", fdir, "--bs", "1", "--accum", "1", "--workers", "0", "--limit", "6", "--max_steps", "2",
              "--size", "480x272"]
    if a.weights:
        common += ["--weights", a.weights]
    TF.main(["train", "--kind", "kp", "--epochs", "1", "--freeze_epochs", "1"] + common)
    assert os.path.exists(os.path.join(run, "kp_last.pt")) and os.path.exists(os.path.join(run, "kp_best.pt"))
    TF.main(["train", "--kind", "kp", "--epochs", "2", "--freeze_epochs", "1"] + common)     # must RESUME at epoch 1
    log = [json.loads(l) for l in open(os.path.join(run, "kp_log.jsonl"))]
    assert [r["epoch"] for r in log] == [0, 1] and log[0]["frozen"] and not log[1]["frozen"], log
    TF.main(["train", "--kind", "line", "--epochs", "1", "--freeze_epochs", "1"] + common)
    rep = EF.main(["--run", run, "--frames", fdir, "--limit", "3", "--overlays", "2",
                   "--old_unet", os.path.join(ROOT, "models_cal", "field_kp.pt")])
    assert rep["n_frames"] == 3 and set(rep["overall"]) >= {"A", "B", "C", "D", "old_unet"}
    im = cv2.imread(os.path.join(fdir, os.listdir(fdir)[0]))
    H, info = FI.keyframe_h(im, os.path.join(run, "kp_best.pt"))
    print("keyframe_h on an untrained model:", "None," if H is None else "H,", info)
    print(f"SMOKE OK in {time.time() - t0:.0f}s (fake labels: the numbers above mean nothing)")


if __name__ == "__main__":
    main()
