"""The Python landmark/line/fitH port must match the labeler's JavaScript exactly (run in node)."""
import json, os, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import labels_io as LI


def js_source():
    html = open(os.path.join(ROOT, "tools/labeler/index.html")).read()
    a = html.index("const FB_N"); b = html.index("// ---------- paint:")
    return html[a:b]


RANDOM_FITS = []
_rng = np.random.default_rng(3)
for _ in range(25):
    n = int(_rng.integers(4, 9))
    w = _rng.uniform([0, 0], [110, 60], (n, 2))
    H = np.array([[12, -3, 200], [0.5, -6, 650], [0.002, -0.01, 1]]) + _rng.normal(0, [[1, .5, 20], [.2, .5, 20], [2e-4, 1e-3, 0]])
    q = (H @ np.c_[w, np.ones(n)].T).T
    RANDOM_FITS.append((w.tolist(), (q[:, :2] / q[:, 2:3] + _rng.normal(0, 1.5, (n, 2))).tolist()))


def run_js():
    cases = [(W, fb, h) for W in (60, 53.33, 55.5) for fb in (False, True) for h in (17.78, 20, 18.3)]
    script = js_source() + "\n" + f"""
const cases={json.dumps(cases)}, fits={json.dumps(RANDOM_FITS)}; const out={{lm:[],lines:[],yl:[],fits:[]}};
for(const [W,fb,h] of cases){{hashD=h;out.lm.push(Object.entries(landmarks(W,fb)).map(([k,v])=>[k,v[0],v[1],v[2],v[3]||0]));out.lines.push(fieldLines(W));out.yl.push(yardLines(W))}}
for(const [w,i] of fits){{const H=fitH(w,i);out.fits.push(H.slice(0,3).map(r=>r.slice()))}}
console.log(JSON.stringify({{cases,out}}));
"""
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def js():
    return run_js()


def test_landmarks_exact(js):
    for (W, fb, h), lm in zip(js["cases"], js["out"]["lm"]):
        py = LI.landmarks(W, fb, h)
        assert [r[0] for r in lm] == list(py.keys()), (W, fb, h)       # same names, same order
        fbn = set(LI.fb_points(W, h))
        for name, x, y, label, isfb in lm:
            px, py_, pl = py[name]
            assert px == x and py_ == y, (name, W, fb, h, (x, y), (px, py_))   # bit-exact float equality
            assert pl == label
            assert bool(isfb) == (name in fbn)


def test_lines_exact(js):
    for (W, fb, h), fl, yl in zip(js["cases"], js["out"]["lines"], js["out"]["yl"]):
        assert LI.field_lines(W) == fl
        assert LI.yard_lines(W) == yl


def test_fith_matches(js):
    for (w, i), Hj in zip(RANDOM_FITS, js["out"]["fits"]):
        Hp = LI.fit_h(w, i); Hj = np.array(Hj)
        g = np.array([[x, y] for x in range(0, 111, 10) for y in range(0, 61, 10)], float)
        d = np.abs(LI.proj(Hp, g)[0] - LI.proj(Hj, g)[0]).max()
        assert d < 1e-5, d


def synth_doc(W=60.0, fb=False, h=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    H = np.array([[11.0, -4.0, 80.0], [0.4, -7.5, 690.0], [0.0004, -0.009, 1.0]])
    L = LI.landmarks(W, fb, h or LI.HASH_HS)
    names = [n for n in L if 0 < LI.proj(H, [L[n][:2]])[0][0, 0] < 1280 and 0 < LI.proj(H, [L[n][:2]])[0][0, 1] < 720][:7]
    pts = {n: (LI.proj(H, [L[n][:2]])[0][0] + rng.normal(0, noise, 2)).round(1).tolist() for n in names}
    off = {n: float(np.hypot(*(LI.proj(H, [L[n][:2]])[0][0] - pts[n]))) for n in names}
    return {"frame": "g_t1", "game": "g", "skip": False, "img": [1280, 720], "pts": pts,
            "world": {n: [round(L[n][0], 3), round(L[n][1], 3)] for n in names}, "field": "football" if fb else "lacrosse",
            "hash": h, "width": W, "H": H.ravel().tolist(), "H_clicks": LI.fit_h([L[n][:2] for n in names], [pts[n] for n in names]).ravel().tolist(),
            "quality": "paint", "click_off": {k: round(v, 1) for k, v in off.items()}}


def test_validate_good_and_bad():
    d = synth_doc(noise=1.0)
    assert len(d["pts"]) >= 5
    e, w = LI.validate(d); assert e == [], e
    d2 = synth_doc(W=53.33, fb=True, h=20.0, noise=1.0); d2["frame"] = "g_t2"
    e, w = LI.validate(d2); assert e == [], e
    bad = synth_doc(noise=1.0); k = next(iter(bad["world"])); bad["world"][k][1] += 0.5
    assert any("world coords" in x for x in LI.validate(bad)[0])
    bad = synth_doc(noise=1.0); k = next(iter(bad["click_off"])); bad["click_off"][k] += 3
    assert any("click_off" in x for x in LI.validate(bad)[0])
    bad = synth_doc(noise=1.0); bad["H"] = [0] * 9
    assert LI.validate(bad)[0]
    bad = synth_doc(noise=1.0); bad["pts"]["fb_L10_near"] = [1, 1]; bad["world"]["fb_L10_near"] = [15, 0]
    assert any("unknown landmark" in x for x in LI.validate(bad)[0])
    good, rep = LI.load_valid([d, d2, {"frame": "x", "game": "g", "skip": True}], verbose=False)
    assert len(good) == 2 and rep["skipped"] == 1
