"""With PERFECT network outputs (the rendered targets themselves), decode -> calibrate -> yard metric must give
~zero error: this pins the coordinate conventions (index px, 1280->480 output scaling, masked softmax, flips)."""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import field_targets as FT
import field_model as FM
import field_calib as FC
from field_infer import FieldDetector
from test_field_targets import doc_for


class Oracle(FieldDetector):
    def __init__(self):
        self.thresh = 0.25; self.device = "cpu"


def test_perfect_heatmaps_give_near_zero_yards():
    det = Oracle()
    meds, n_fail = [], 0
    for s in range(16):
        fb = s % 2 == 1
        d, H = doc_for(200 + s, fb=fb, k1=0.03 * (s % 3 - 1))
        r = FT.render(d)
        kl = torch.from_numpy(np.log(r["kp"] + 1e-4))            # logits whose softmax ~ the targets
        ln = torch.from_numpy(r["ln"])
        kp, lines, border = det.decode(kl, ln, fb)
        W, hashD, _, Hl, k1 = FT.doc_geometry(d)
        for k1p in (k1, 0.0):
            Hp, info = FC.calibrate(kp, W, hashD, fb, lines=lines, border=border, k1=k1p)
            if Hp is None:
                n_fail += 1; continue
            e = FC.yard_errors(Hl, k1, Hp, k1p, W)
            if k1p == k1:
                meds.append(np.median(e))
    meds = np.array(meds)
    print(f"oracle: {len(meds)} frames, median {np.median(meds):.4f} yd, max {meds.max():.4f} yd, fails {n_fail}")
    assert n_fail <= 2 and np.median(meds) < 0.05 and meds.max() < 0.3


def test_to_engine_matches_old_field_cal_on_60yd_world():
    import field_cal as OLD
    from field_infer import to_engine
    _, H = doc_for(300, fb=False, k1=0.0)
    for gx in (15.0, 95.0):
        for near in (True, False):
            a = OLD.to_engine_frame(H, (1280, 720), gx, near); b = to_engine(H, 60.0, gx, near)
            assert np.allclose(a / a[2, 2], b / b[2, 2], atol=1e-9)
