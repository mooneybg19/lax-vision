import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import labels_io as LI
import synth_cam as SC
import field_targets as FT


def doc_for(seed, fb, k1=0.04, W=None, hashD=17.78):
    rng = np.random.default_rng(seed)
    W = W or (LI.FB_WIDTH if fb else 60.0)
    while True:
        H = SC.camera_H(rng, W)
        d = SC.fake_label(f"g_t{seed}", "g", H, W, fb, hashD, rng, k1=k1)
        if d is not None:
            break
    d["H_u"] = (H / H[2, 2]).ravel().tolist(); d["k1"] = k1; d["hash_used"] = hashD if fb else None; d["width_used"] = W
    return d, H


def test_channels_and_flip_pairs():
    assert FT.N_KP == 35 + 78 and len(set(FT.KP_NAMES)) == FT.N_KP
    assert FT.N_LINE == 13 + 18 + 2
    P = {FT.KP_NAMES[i]: FT.KP_NAMES[j] for i, j in enumerate(FT.KP_FLIP)}
    for a, b in [("restrain_L_near", "restrain_R_near"), ("restrain_L_far", "restrain_R_far"), ("corner_0_0", "corner_110_0"),
                 ("corner_0_60", "corner_110_60"), ("boxL_end_far", "boxR_end_far"), ("wing_near_L", "wing_near_R"),
                 ("wing_far_L", "wing_far_R"), ("creaseL_front", "creaseR_front"), ("creaseL_back", "creaseR_back"),
                 ("goalL_near", "goalR_near"), ("fb_L10_hnear", "fb_R10_hnear"), ("fb_L0_far", "fb_R0_far"),
                 ("fb_L30_hfar", "fb_R30_hfar")]:
        assert P[a] == b and P[b] == a
    for s in ("center_near", "center_far", "faceoff_X", "fb_50_hnear", "fb_50_hfar"):
        assert P[s] == s
    # near/far never swap
    for a, b in P.items():
        assert ("near" in a) == ("near" in b) and ("far" in a) == ("far" in b), (a, b)
    L = {FT.LINE_NAMES[i]: FT.LINE_NAMES[j] for i, j in enumerate(FT.LINE_FLIP)}
    assert L["restrain_L"] == "restrain_R" and L["boxL_far"] == "boxR_far" and L["yard_10"] == "yard_100"
    assert L["sideline_near"] == "sideline_near" and L["hash_far"] == "hash_far" and L["wing_near"] == "wing_near"


def test_masks():
    d, _ = doc_for(1, fb=False)
    t = FT.render(d)
    assert not t["kp_mask"][:FT.N_KP][FT.KP_IS_FB].any() and t["kp_mask"][FT.N_KP]
    assert t["kp"][:FT.N_KP][FT.KP_IS_FB].max() == 0 and t["ln"][:FT.N_LINE][FT.LINE_IS_FB].max() == 0
    d, _ = doc_for(2, fb=True)
    t = FT.render(d)                        # lax_painted defaults to True
    assert t["kp_mask"].all() and t["ln_mask"].all()
    assert t["kp"][:FT.N_KP][FT.KP_IS_FB].max() > 0.5
    # football game whose labels never click lacrosse-only paint: those channels are masked, shared ones kept
    d2 = dict(d, pts={k: v for k, v in d["pts"].items() if k.startswith("fb_")})
    d2["pts"].setdefault("fb_L10_hnear", [1, 1])
    FT.annotate_games([d2])
    assert d2["lax_painted"] is False
    t = FT.render(d2)
    i = FT.KP_NAMES.index
    assert not t["kp_mask"][i("boxL_near")] and not t["kp_mask"][i("faceoff_X")] and not t["kp_mask"][i("creaseR_front")]
    assert t["kp_mask"][i("restrain_L_near")] and t["kp_mask"][i("center_far")] and t["kp_mask"][i("fb_L10_hnear")]
    assert t["kp"][i("boxL_near")].max() == 0
    j = FT.LINE_NAMES.index
    assert not t["ln_mask"][j("wing_near")] and t["ln_mask"][j("restrain_R")] and t["ln_mask"][j("sideline_far")]
    d3 = dict(d); d3["pts"] = dict(d["pts"], creaseL_front=[5, 5]); FT.annotate_games([d3]); assert d3["lax_painted"] is True


def test_render_decode_roundtrip():
    errs = []
    for s in range(6):
        d, H = doc_for(10 + s, fb=s % 2 == 0)
        t = FT.render(d)
        det = FT.decode_kp(t["kp"], t["kp_mask"], thresh=0.5)
        for c, n in enumerate(FT.KP_NAMES):
            p = t["kp_px"][c]
            if not np.isfinite(p[0]):
                assert n not in det
                continue
            # the projection: world -> H -> lens, independent recomputation
            W, hashD, fb, _, k1 = FT.doc_geometry(d)
            q = SC.distort(LI.proj(H, [LI.landmarks(W, True, hashD)[n][:2]])[0], k1)[0]
            assert np.allclose(p, q, atol=1e-6)
            if n in det:       # (overlapping Gaussians of two nearby landmarks can hide one peak)
                errs.append(np.hypot(det[n][0] - p[0], det[n][1] - p[1]))
    errs = np.array(errs)
    print(f"render->decode error (image px): median {np.median(errs):.3f} max {errs.max():.3f} n {len(errs)}")
    assert len(errs) > 40 and np.median(errs) < 0.3 and np.percentile(errs, 95) < 1.0


def test_flip_equals_mirrored_camera():
    for s, fb in ((3, False), (4, True)):
        d, H = doc_for(30 + s, fb=fb)
        Fh = np.array([[-1, 0, FT.IMG_W - 1], [0, 1, 0], [0, 0, 1.0]]); M = np.array([[-1, 0, 110], [0, 1, 0], [0, 0, 1.0]])
        d2 = dict(d); d2["H_u"] = (Fh @ H @ M).ravel().tolist()
        a = FT.render(d, flip=True); b = FT.render(d2)
        assert np.allclose(a["kp_px"], b["kp_px"], atol=1e-6, equal_nan=True)
        assert np.abs(a["kp"] - b["kp"]).max() < 1e-5
        assert np.abs(a["ln"][:FT.N_LINE] - b["ln"][:FT.N_LINE]).max() < 1e-4
        assert (a["kp_mask"] == b["kp_mask"]).all() and (a["ln_mask"] == b["ln_mask"]).all()
        # all-lines channel is drawn by rasterising: allow a few edge pixels to differ
        assert np.mean(np.abs(a["ln"][FT.N_LINE] - b["ln"][FT.N_LINE]) > 0.2) < 0.01


def test_line_extremities_decode():
    d, H = doc_for(50, fb=True)
    t = FT.render(d)
    ex = FT.line_extremities(d)
    dec = FT.decode_lines(t["ln"], t["ln_mask"], thresh=0.5)
    n = 0
    for ci, (p0, p1, _) in ex.items():
        nm = FT.LINE_NAMES[ci]
        if nm not in dec:
            continue
        a, b = np.array(dec[nm][0]), np.array(dec[nm][1])
        e = min(max(np.hypot(*(a - p0)), np.hypot(*(b - p1))), max(np.hypot(*(a - p1)), np.hypot(*(b - p0))))
        assert e < 3.0, (nm, e)
        n += 1
    assert n >= 3
