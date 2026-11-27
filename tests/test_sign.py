"""Cameras whose world origin (near-left corner) is BEHIND the camera plane: H[2,2] < 0 physically. Everything that
tests 'w > 0' (visibility, the metric, targets) must still work."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import labels_io as LI
import synth_cam as SC
import field_targets as FT
import field_calib as FC
import game_solve as GS


def right_end_camera():
    cam = np.array([95.0, -8.0, 12.0]); target = np.array([105.0, 40.0, 0.0])
    fwd = target - cam; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd]); f = 1300.0
    K = np.array([[f, 0, 639.5], [0, f, 359.5], [0, 0, 1.0]])
    H = K @ np.c_[R[:, 0], R[:, 1], -R @ cam]
    return H / abs(H[2, 2])


def test_origin_behind_camera():
    H = right_end_camera()
    assert H[2, 2] < 0                                        # the near-left corner is behind the camera plane
    rng = np.random.default_rng(0)
    d = SC.fake_label("g_t1", "g", H, 60.0, False, 17.78, rng, k1=0.0, n_max=8)
    assert d is not None and len(d["pts"]) >= 5
    # the labeler's fitH convention (w > 0 at the first clicked point) gives H[2,2] = -1 here
    assert np.asarray(d["H"]).reshape(3, 3)[2, 2] < 0
    assert LI.validate(d)[0] == []
    Hc = LI.H_of(d)
    assert np.all(LI.proj(Hc, [d["world"][n] for n in d["pts"]])[1] > 0)
    # a sign-flipped copy of the same H is the same mapping and must canonicalise identically
    d2 = dict(d, H=(-np.asarray(d["H"])).tolist())
    assert np.allclose(LI.H_of(d2), Hc)
    d["H_u"] = (H / abs(H[2, 2])).ravel().tolist(); d["k1"] = 0.0
    t = FT.sparse_targets(d)
    assert np.isfinite(t["kp_px"][:, 0]).sum() >= 5
    e = FC.yard_errors(H, 0.0, -H, 0.0, 60.0)                 # same mapping, opposite scale sign
    assert len(e) > 100 and np.all(np.isfinite(e)) and e.max() < 1e-6
    kp = {n: (t["kp_px"][c][0], t["kp_px"][c][1], 0.9) for c, n in enumerate(FT.KP_NAMES) if np.isfinite(t["kp_px"][c][0])}
    Hp, info = FC.calibrate(kp, 60.0, 17.78, False)
    assert Hp is not None, info
    assert np.median(FC.yard_errors(H, 0.0, Hp, 0.0, 60.0)) < 0.05
    others = []
    while len(others) < 10:
        o = SC.fake_label(f"g_t{len(others) + 2}", "g", SC.camera_H(rng, 60.0), 60.0, False, 17.78, rng, n_max=8)
        if o is not None:
            others.append(o)
    r = GS.GameSolver([d] + others, verbose=False).solve()
    Hs = np.asarray(r["frames"]["g_t1"]["H"]).reshape(3, 3)
    assert np.median(FC.yard_errors(H, 0.0, Hs, r["k1"], 60.0)) < 0.3
