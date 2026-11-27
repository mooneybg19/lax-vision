"""game_solve on synthetic games with known truth: a shared lens k1, a mislabeled hash distance, click noise and one
injected bad click. The solver must recover k1 and the hash, flag the bad click, and reduce the field error."""
import os, sys, tempfile
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import labels_io as LI
import synth_cam as SC
import game_solve as GS


def make_game(game, n, W, fb, h_true, h_lab, k1, seed, outdir=None):
    rng = np.random.default_rng(seed); docs, truth = [], {}
    i = 0
    while len(docs) < n:
        H = SC.camera_H(rng, W); i += 1
        d = SC.fake_label(f"{game}_t{i:05d}", game, H, W, fb, h_true, rng, k1=k1, click_sd=1.0,
                          n_max=int(rng.integers(5, 9)), hash_labeled=h_lab)
        if d is None:
            continue
        docs.append(d); truth[d["frame"]] = H
        if outdir:
            cv2.imwrite(os.path.join(outdir, d["frame"] + ".jpg"), SC.render(H, W, fb, h_true or 17.78, k1, rng),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    bad_frame = docs[3]["frame"]; bad_name = next(iter(docs[3]["pts"]))
    docs[3]["pts"][bad_name][0] += 15.0
    return docs, truth, (bad_frame, bad_name)


def field_err(Hu, k1, Ht, k1t, W):
    g = np.array([[u, v] for u in np.linspace(20, 1260, 32) for v in np.linspace(20, 700, 18)])
    wt = LI.proj(np.linalg.inv(Ht), SC.undistort(g, k1t))[0]
    ins = (wt[:, 0] >= 0) & (wt[:, 0] <= 110) & (wt[:, 1] >= 0) & (wt[:, 1] <= W)
    we = LI.proj(np.linalg.inv(Hu), SC.undistort(g, k1))[0]
    return np.hypot(*(we - wt)[ins].T)


def check(docs, truth, bad, k1t, W, r, max_med, max_p90):
    eb = np.concatenate([field_err(LI.H_of(d), 0.0, truth[d["frame"]], k1t, W) for d in docs])
    ea = np.concatenate([field_err(np.array(r["frames"][d["frame"]]["H"]).reshape(3, 3), r["k1"], truth[d["frame"]], k1t, W) for d in docs])
    print(f"k1 {r['k1']:.4f}+-{r['k1_se']:.4f} (true {k1t}); field err label H median {np.median(eb):.3f} p90 "
          f"{np.percentile(eb, 90):.3f} -> solved median {np.median(ea):.3f} p90 {np.percentile(ea, 90):.3f}")
    assert bad[1] in r["outlier_clicks"].get(bad[0], []), r["outlier_clicks"]
    assert np.median(ea) < max_med and np.percentile(ea, 90) < max_p90
    assert np.median(ea) < 0.8 * np.median(eb) and np.percentile(ea, 90) < 0.8 * np.percentile(eb, 90)


def test_football_clicks_only_recovers_k1_and_hash():
    k1t = -0.06
    docs, truth, bad = make_game("gfb", 30, LI.FB_WIDTH, True, 20.0, 17.78, k1t, 1)
    r = GS.GameSolver(docs, use_paint=False, verbose=False).solve()
    assert r["hash_snapped"] and r["hash"] == 20.0 and abs(r["hash_est"] - 20.0) < 0.1, r.get("hash_est")
    assert abs(r["k1"] - k1t) < max(3 * r["k1_se"], 0.015), (r["k1"], r["k1_se"])
    check(docs, truth, bad, k1t, LI.FB_WIDTH, r, 0.2, 0.8)


def test_lacrosse_with_paint_recovers_k1():
    k1t = 0.05
    with tempfile.TemporaryDirectory() as td:
        docs, truth, bad = make_game("glax", 15, 60.0, False, None, None, k1t, 2, outdir=td)
        r = GS.GameSolver(docs, frames_dir=td, use_paint=True, verbose=False).solve()
    assert r["n_paint_frames"] == 15
    assert abs(r["k1"] - k1t) < 0.01, (r["k1"], r["k1_se"])
    check(docs, truth, bad, k1t, 60.0, r, 0.15, 0.6)


def test_apply_and_consistency_check():
    docs, truth, bad = make_game("gx", 8, 60.0, False, None, None, 0.0, 5)
    for d in docs:
        if d["frame"] != bad[0]:
            assert LI.validate(d)[0] == []
    assert any("stated rms" in e for e in LI.validate(docs[3])[0])     # the injected click breaks the stated fit
    r = GS.solve_all(docs, verbose=False)
    out = GS.apply(docs, r)
    assert all("H_u" in d and "k1" in d for d in out)
    # a game whose frames disagree on the width must be refused, not silently mixed
    docs[1] = dict(docs[1], width=53.33)
    try:
        GS.GameSolver(docs, verbose=False); assert False
    except ValueError:
        pass
