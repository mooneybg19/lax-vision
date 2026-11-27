"""Evaluate the trained detector + calibrator on held-out synthetic cameras (unseen seed).

Reports: landmark detection rate and pixel error vs ground truth, and the end-to-end field error (where a known
field point lands after detect -> calibrate), which is the number that matters for the engine.
"""
import sys
import numpy as np
import cv2
import lax_field as LF
import field_cal as FC
import gen_synth_field as G
from eval_field import legacy_unet_detect   # the old UNet detector (field_infer now serves the HRNet nets)

model = sys.argv[1] if len(sys.argv) > 1 else "models_cal/field_kp.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = np.random.default_rng(999)                 # unseen seed

det_px, field_err, n_land, ok = [], [], [], 0
for _ in range(N):
    img, hm, uv, vis, Himg = G.render(rng)
    kp = legacy_unet_detect(img, model)
    n_land.append(len(kp))
    # detector pixel error vs ground-truth projected landmarks
    for nm, (u, v) in kp.items():
        gi = LF.KP_NAMES.index(nm)
        if vis[gi]:
            det_px.append(float(np.hypot(u - uv[gi, 0], v - uv[gi, 1])))
    H_eng, info = FC.calibrate(kp, (img.shape[1], img.shape[0]), goal_in_view_x=15.0, near_sideline_is_y0=True)
    if H_eng is None:
        continue
    ok += 1
    # end-to-end: left goal centre (15,30) -> image -> engine field; truth is (0, 26.67)
    P = Himg @ np.array([15.0, 30.0, 1.0]); q = P[:2] / P[2]
    if 0 <= q[0] < img.shape[1] and 0 <= q[1] < img.shape[0]:
        p = H_eng @ np.array([q[0], q[1], 1.0]); p = p[:2] / p[2]
        field_err.append(float(np.hypot(p[0], p[1] - FC.GOAL_Y_ENG)))

def pct(a, p):
    return round(float(np.percentile(a, p)), 2) if len(a) else None

print(f"frames: {N}   calibrated: {ok} ({100*ok/N:.0f}%)")
print(f"landmarks detected / frame: median {int(np.median(n_land))}")
print(f"detector pixel error: median {pct(det_px,50)}  p90 {pct(det_px,90)}")
print(f"end-to-end field error (goal centre): median {pct(field_err,50)} yd  p90 {pct(field_err,90)} yd  (engine now ~10 yd)")
