import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import labels_io as LI
import synth_cam as SC
import field_targets as FT
import field_calib as FC
from test_field_targets import doc_for


def test_metric_is_zero_for_identical_and_scales_with_shift():
    d, H = doc_for(70, fb=False, k1=0.03)
    e = FC.yard_errors(H, 0.03, H, 0.03, 60.0)
    assert len(e) > 100 and e.max() < 1e-6
    S = np.array([[1, 0, 1.0], [0, 1, 0], [0, 0, 1]])           # field shifted 1 yd in x
    e = FC.yard_errors(H, 0.03, H @ S, 0.03, 60.0)
    assert np.allclose(e[np.isfinite(e)], 1.0, atol=1e-6)


def test_calibrate_from_noisy_keypoints_with_outliers_and_lines():
    rng = np.random.default_rng(0)
    res, fails = [], 0
    for s in range(40):
        fb = s % 2 == 0
        d, H = doc_for(100 + s, fb=fb, k1=0.0)
        t = FT.sparse_targets(d)
        kp = {}
        for c, n in enumerate(FT.KP_NAMES):
            p = t["kp_px"][c]
            if np.isfinite(p[0]):
                kp[n] = (p[0] + rng.normal(0, 1.5), p[1] + rng.normal(0, 1.5), 0.9)
        names = list(kp)
        for n in names[:2] if len(names) >= 8 else []:            # 2 gross outliers (wrong-place detections)
            kp[n] = (kp[n][0] + rng.uniform(40, 120), kp[n][1] + rng.uniform(-80, 80), 0.6)
        ex = FT.line_extremities(d)
        lines = {FT.LINE_NAMES[ci]: (p0 + rng.normal(0, 1.5, 2), p1 + rng.normal(0, 1.5, 2), 0.9) for ci, (p0, p1, _) in ex.items()}
        W, hashD, _, _, _ = FT.doc_geometry(d)
        Hp, info = FC.calibrate(kp, W, hashD, fb, lines=lines)
        if Hp is None:
            fails += 1; continue
        e = FC.yard_errors(H, 0.0, Hp, 0.0, W)
        res.append(np.median(e))
    res = np.array(res)
    print(f"calibrated {len(res)}/40, median of frame medians {np.median(res):.3f} yd, p90 {np.percentile(res, 90):.3f}")
    assert fails <= 4 and np.median(res) < 0.3 and np.percentile(res, 90) < 1.0


def test_calibrate_rejects_garbage():
    rng = np.random.default_rng(1)
    kp = {n: (rng.uniform(0, 1280), rng.uniform(0, 720), 0.5) for n in FT.KP_NAMES[:12]}
    Hp, info = FC.calibrate(kp, 60.0, 17.78, False)
    assert Hp is None, info
