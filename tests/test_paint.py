"""paint.py must reproduce the labeler's paintField / paintMatch (JS, run in node) on real frames."""
import json, os, subprocess, sys, tempfile
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import paint as PT
import labels_io as LI


def js_block():
    html = open(os.path.join(ROOT, "tools/labeler/index.html")).read()
    return html[html.index("const FB_N"):html.index("// ---------- state ----------")]


FRAMES = ["scituate-hanover-2024_t00310", "jesuit-westview_t00408"]
H_TEST = [[11.0, -4.0, 80.0], [0.4, -7.5, 690.0], [0.0004, -0.009, 1.0]]


def test_paint_field_and_match_match_js():
    for fr in FRAMES:
        img = cv2.imread(os.path.join(ROOT, "results/label_frames", fr + ".jpg"))
        assert img is not None
        h, w = img.shape[:2]
        rgba = np.dstack([img[..., 2], img[..., 1], img[..., 0], np.full((h, w), 255, np.uint8)])
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "im.raw"); rgba.tofile(raw)
            onf = os.path.join(td, "on.raw"); dtf = os.path.join(td, "dt.raw")
            script = js_block() + f"""
const fs=require('fs');const buf=fs.readFileSync({json.dumps(raw)});const im={{width:{w},height:{h},data:new Uint8Array(buf)}};
const P=paintField(im);fs.writeFileSync({json.dumps(onf)},Buffer.from(P.on.buffer));fs.writeFileSync({json.dumps(dtf)},Buffer.from(P.dt.buffer));
const H={json.dumps(H_TEST)};const res={{}};
for(const [fb,W,hd] of [[false,60,17.78],[false,53.33,17.78],[true,53.33,17.78],[true,53.33,20]]){{fbMode=fb;hashD=hd;res[fb+'_'+W+'_'+hd]=paintMatch(H,P,W)}}
console.log(JSON.stringify(res));
"""
            r = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
            res = json.loads(r.stdout)
            on_js = np.fromfile(onf, np.uint8).reshape(h, w).astype(bool)
            dt_js = np.fromfile(dtf, np.float32).reshape(h, w)
        P = PT.paint_field(img)
        assert (P["on"] == on_js).all(), (fr, int((P["on"] != on_js).sum()))
        assert np.abs(P["dt"] - dt_js).max() < 1e-3
        for key, v in res.items():
            fb, W, hd = key.split("_"); fb = fb == "true"; W = float(W); hd = float(hd)
            p = PT.paint_match(np.array(H_TEST), P, W, fb, hd)
            if v is None:
                assert p is None; continue
            assert p["n"] == v["n"], (key, p, v)
            assert abs(p["frac"] - v["frac"]) < 1e-9 and abs(p["back"] - v["back"]) < 1e-9, (key, p, v)
