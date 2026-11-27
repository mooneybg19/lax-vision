"""Training targets for the field detector, rendered from (game_solve-corrected) labels.

Image coordinates everywhere in training/inference: "index" convention, pixel i has its centre at coordinate i
(OpenCV / numpy; also what the labeler's paint snap used). The labeler's raw clicks are canvas coordinates
(pixel centre at i + 0.5); game_solve converts them (CLICK_TO_INDEX).

Keypoint channels: ONE fixed list = the union of the lacrosse landmarks and the football-line landmarks, in the
labeler's order, plus a final background channel (NBJW layout). Landmarks that do not exist on a field type are
MASKED (excluded from the softmax and the loss), never trained as negatives.

Line channels (NBJW line-net layout): one channel per painted line with a Gaussian at each of the line's two
visible extremities, plus a final channel with Gaussians along every visible line. Lacrosse lines always; on
football-lined fields also the yard lines (except 35/55/75, which ARE the restraining/centre lines) and the two
hash rows. Again masked when the field type does not have them.

Horizontal flip: image x -> (w-1) - x mirrors the field x -> 110 - x, so Left<->Right names swap
(restrain_L <-> restrain_R, boxL <-> boxR, wing_*_L <-> wing_*_R, creaseL <-> creaseR, fb_L<N> <-> fb_R<N>,
corner_0_* <-> corner_110_*); near/far (y) never change. The permutation is DERIVED from the world coordinates
(and checked to be an involution) rather than typed by hand.
"""
import numpy as np
import cv2
from scipy import ndimage

import labels_io as LI
from synth_cam import distort, distort_valid

IMG_W, IMG_H = 1280, 720
IN_W, IN_H = 960, 540            # network input (NBJW size)
OUT_W, OUT_H = 480, 270          # network output (HRNet head at 1/2 of the input)
SIGMA = 2.0                      # Gaussian sigma in output pixels (NBJW: 2 at 480x270)
CLICK_TO_INDEX = -0.5            # labeler canvas coords -> index coords

# ------------------------------------------------------------------ channel lists
_LAX = list(LI.lax_points(60.0).keys())
_FB = list(LI.fb_points(LI.FB_WIDTH, LI.HASH_HS).keys())
KP_NAMES = _LAX + _FB
KP_IS_FB = np.array([n in set(_FB) for n in KP_NAMES])
N_KP = len(KP_NAMES)             # background is channel N_KP

LAX_LINES = ["sideline_near", "sideline_far", "endline_L", "endline_R", "center", "restrain_L", "restrain_R",
             "boxL_near", "boxL_far", "boxR_near", "boxR_far", "wing_near", "wing_far"]
YARD_X = [x for x in range(5, 106, 5) if x not in (35, 55, 75)]
FB_LINES = [f"yard_{x}" for x in YARD_X] + ["hash_near", "hash_far"]
LINE_NAMES = LAX_LINES + FB_LINES
LINE_IS_FB = np.array([n in set(FB_LINES) for n in LINE_NAMES])
N_LINE = len(LINE_NAMES)         # the all-lines channel is N_LINE


def line_world(W, hashD):
    """{name: [[x0,y0],[x1,y1]]} in yards for every line channel (football lines included; mask decides)."""
    c = W / 2; lo, hi = c - 20, c + 20
    d = {"sideline_near": [[0, 0], [110, 0]], "sideline_far": [[0, W], [110, W]],
         "endline_L": [[0, 0], [0, W]], "endline_R": [[110, 0], [110, W]], "center": [[55, 0], [55, W]],
         "restrain_L": [[35, 0], [35, W]], "restrain_R": [[75, 0], [75, W]],
         "boxL_near": [[0, lo], [35, lo]], "boxL_far": [[0, hi], [35, hi]],
         "boxR_near": [[75, lo], [110, lo]], "boxR_far": [[75, hi], [110, hi]],
         "wing_near": [[45, lo], [65, lo]], "wing_far": [[45, hi], [65, hi]]}
    for x in YARD_X:
        d[f"yard_{x}"] = [[x, 0], [x, W]]
    d["hash_near"] = [[5, hashD], [105, hashD]]; d["hash_far"] = [[5, W - hashD], [105, W - hashD]]
    return d


def _flip_perm(names, world_fn):
    perm = []
    for W, h in ((60.0, 17.78), (LI.FB_WIDTH, 17.78), (LI.FB_WIDTH, 20.0)):
        wd = world_fn(W, h)
        p = []
        for n in names:
            x, y = wd[n]
            cands = [m for m in names if abs(wd[m][0] - (110 - x)) < 1e-9 and abs(wd[m][1] - y) < 1e-9]
            assert len(cands) == 1, (n, cands)
            p.append(names.index(cands[0]))
        perm.append(p)
    assert all(q == perm[0] for q in perm)
    p = np.array(perm[0])
    assert (p[p] == np.arange(len(p))).all()           # involution
    return p


KP_FLIP = _flip_perm(KP_NAMES, lambda W, h: {n: v[:2] for n, v in LI.landmarks(W, True, h).items()})


def _line_mid(W, h):
    out = {}
    for n, ((x0, y0), (x1, y1)) in line_world(W, h).items():
        out[n] = ((x0 + x1) / 2, (y0 + y1) / 2 + 1e-3 * (y1 - y0) ** 2)   # midpoint + orientation tag
    return out


LINE_FLIP = _flip_perm(LINE_NAMES, _line_mid)


# ------------------------------------------------------------------ geometry of a training document
def doc_geometry(doc):
    """(W, hashD, fb, H (index convention, undistorted), k1) for a label doc, corrected by game_solve if it was run."""
    W, hashD, fb = LI.doc_geometry(doc)
    W = float(doc.get("width_used", W)); hashD = float(doc.get("hash_used") or hashD)
    if "H_u" in doc:
        H = np.asarray(doc["H_u"], float).reshape(3, 3); k1 = float(doc.get("k1", 0.0))
    else:  # raw label: H from the labeler (paint-snapped => index convention), no lens
        H = LI.H_of(doc); k1 = 0.0
    return W, hashD, fb, H, k1


def masks(fb):
    kp = np.ones(N_KP + 1, bool); ln = np.ones(N_LINE + 1, bool)
    if not fb:
        kp[:N_KP] = ~KP_IS_FB; ln[:N_LINE] = ~LINE_IS_FB
    return kp, ln


def project(H, k1, X, img_wh=(IMG_W, IMG_H)):
    """World (N,2) -> image index coords (N,2) through H and the lens; NaN where invalid (behind / lens fold)."""
    q, w = LI.proj(H, X)
    ok = (w > 1e-6) & distort_valid(q, k1, img_wh)
    q = distort(q, k1, img_wh)
    q[~ok] = np.nan
    return q


def to_out(q, img_wh=(IMG_W, IMG_H), out_wh=(OUT_W, OUT_H)):
    s = np.array([out_wh[0] / img_wh[0], out_wh[1] / img_wh[1]])
    return (np.asarray(q, float) + 0.5) * s - 0.5


def from_out(q, img_wh=(IMG_W, IMG_H), out_wh=(OUT_W, OUT_H)):
    s = np.array([out_wh[0] / img_wh[0], out_wh[1] / img_wh[1]])
    return (np.asarray(q, float) + 0.5) / s - 0.5


def inside(q, wh):
    return np.isfinite(q[:, 0]) & (q[:, 0] >= -0.5) & (q[:, 0] <= wh[0] - 0.5) & (q[:, 1] >= -0.5) & (q[:, 1] <= wh[1] - 0.5)


def keypoints_px(doc, img_wh=(IMG_W, IMG_H)):
    """(N_KP,2) image coords of every channel's landmark (NaN if absent / not in the image), and the kp mask."""
    W, hashD, fb, H, k1 = doc_geometry(doc)
    L = LI.landmarks(W, True, hashD)
    X = np.array([L[n][:2] for n in KP_NAMES], float)
    q = project(H, k1, X, img_wh)
    kpm, _ = masks(fb)
    q[~kpm[:N_KP]] = np.nan
    q[~inside(q, img_wh)] = np.nan
    return q, kpm


def line_extremities(doc, img_wh=(IMG_W, IMG_H), n=600):
    """{channel index: (p0, p1, polyline)} of each visible line in image coords: the first and last visible points
    along the line (refined by bisection at the image border), and the visible polyline for the all-lines channel."""
    W, hashD, fb, H, k1 = doc_geometry(doc)
    _, lnm = masks(fb)
    out = {}
    for ci, name in enumerate(LINE_NAMES):
        if not lnm[ci]:
            continue
        a, b = np.array(line_world(W, hashD)[name], float)
        t = np.linspace(0, 1, n)
        q = project(H, k1, a + t[:, None] * (b - a), img_wh)
        vis = inside(q, img_wh)
        if vis.sum() < 2:
            continue
        idx = np.flatnonzero(vis)

        def edge(i_in, i_out):
            lo, hi = t[i_in], t[i_out]
            for _ in range(40):
                m = (lo + hi) / 2
                if inside(project(H, k1, (a + m * (b - a))[None], img_wh), img_wh)[0]:
                    lo = m
                else:
                    hi = m
            return project(H, k1, (a + lo * (b - a))[None], img_wh)[0]
        p0 = edge(idx[0], idx[0] - 1) if idx[0] > 0 else q[idx[0]]
        p1 = edge(idx[-1], idx[-1] + 1) if idx[-1] < n - 1 else q[idx[-1]]
        if np.hypot(*(p1 - p0)) < 4:          # a sliver of a line: too short to be a usable target
            continue
        out[ci] = (p0, p1, q[vis])
    return out


def _gauss(hm, p, sigma):
    x, y = p
    r = int(np.ceil(3 * sigma))
    x0, x1 = max(0, int(np.floor(x)) - r), min(hm.shape[1], int(np.floor(x)) + r + 2)
    y0, y1 = max(0, int(np.floor(y)) - r), min(hm.shape[0], int(np.floor(y)) + r + 2)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1)[None, :]; ys = np.arange(y0, y1)[:, None]
    g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def render(doc, flip=False, img_wh=(IMG_W, IMG_H), out_wh=(OUT_W, OUT_H), sigma=SIGMA):
    """Targets for one doc. Returns dict with
       kp  (N_KP+1, h, w) float32: Gaussians at visible landmarks, last channel = background (1 - sum, clipped)
       kp_mask (N_KP+1,) bool,
       ln  (N_LINE+1, h, w) float32: Gaussians at each visible line's two extremities; last = along all lines
       ln_mask (N_LINE+1,) bool,
       kp_px (N_KP,2) landmark image coords used (NaN = not rendered), after the flip if flip."""
    q, kpm = keypoints_px(doc, img_wh)
    exts = line_extremities(doc, img_wh)
    lnm = masks(doc_geometry(doc)[2])[1]
    if flip:
        def F(p):
            p = np.array(p, float, copy=True); p[..., 0] = (img_wh[0] - 1) - p[..., 0]; return p
        q = F(q)[KP_FLIP]
        kpm = np.r_[kpm[:N_KP][KP_FLIP], kpm[N_KP]]
        exts = {int(LINE_FLIP[ci]): (F(p0), F(p1), F(pl)) for ci, (p0, p1, pl) in exts.items()}
        lnm = np.r_[lnm[:N_LINE][LINE_FLIP], lnm[N_LINE]]
    w, h = out_wh
    kp = np.zeros((N_KP + 1, h, w), np.float32)
    qo = to_out(q, img_wh, out_wh)
    for c in range(N_KP):
        if np.isfinite(qo[c, 0]):
            _gauss(kp[c], qo[c], sigma)
    kp[N_KP] = np.clip(1 - kp[:N_KP].sum(0), 0, 1)
    ln = np.zeros((N_LINE + 1, h, w), np.float32)
    canvas = np.zeros((h, w), np.uint8)
    for ci, (p0, p1, pl) in exts.items():
        _gauss(ln[ci], to_out(p0, img_wh, out_wh), sigma)
        _gauss(ln[ci], to_out(p1, img_wh, out_wh), sigma)
        po = to_out(pl, img_wh, out_wh)
        cv2.polylines(canvas, [np.round(po * 16).astype(np.int32).reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8, shift=4)
    if canvas.any():
        d = ndimage.distance_transform_edt(canvas == 0)
        ln[N_LINE] = np.exp(-d ** 2 / (2 * sigma ** 2)) * (d <= 3 * sigma)
    return {"kp": kp, "kp_mask": kpm, "ln": ln, "ln_mask": lnm, "kp_px": q}


# ------------------------------------------------------------------ decoding (inverse of render)
def decode_kp(hm, mask=None, thresh=0.1, img_wh=(IMG_W, IMG_H)):
    """Peak + 3x3 quadratic sub-pixel refinement per channel. hm: (C>=N_KP, h, w) probabilities.
    Returns {name: (u, v, score)} in image index coords."""
    hm = np.asarray(hm)
    h, w = hm.shape[-2:]
    out = {}
    for c in range(N_KP):
        if mask is not None and not mask[c]:
            continue
        m = hm[c]
        j = int(np.argmax(m)); y, x = divmod(j, w); s = float(m[y, x])
        if s < thresh:
            continue
        dx = dy = 0.0
        if 0 < x < w - 1:
            l, r = m[y, x - 1], m[y, x + 1]; den = l - 2 * s + r
            dx = 0.5 * (l - r) / den if den < 0 else 0.0
        if 0 < y < h - 1:
            u_, d_ = m[y - 1, x], m[y + 1, x]; den = u_ - 2 * s + d_
            dy = 0.5 * (u_ - d_) / den if den < 0 else 0.0
        p = from_out(np.array([x + np.clip(dx, -.5, .5), y + np.clip(dy, -.5, .5)]), img_wh, (w, h))
        out[KP_NAMES[c]] = (float(p[0]), float(p[1]), s)
    return out


def decode_lines(hm, mask=None, thresh=0.3, min_sep=6, img_wh=(IMG_W, IMG_H)):
    """Two strongest separated peaks per line channel -> {line name: ((u0,v0),(u1,v1), score)} in image coords."""
    hm = np.asarray(hm)
    h, w = hm.shape[-2:]
    out = {}
    for c in range(N_LINE):
        if mask is not None and not mask[c]:
            continue
        m = hm[c]
        mx = ndimage.maximum_filter(m, size=5)
        ys, xs = np.nonzero((m == mx) & (m >= thresh))
        if len(ys) < 2:
            continue
        order = np.argsort(-m[ys, xs])
        pts = []
        for k in order:
            p = np.array([xs[k], ys[k]], float)
            if all(np.hypot(*(p - q)) >= min_sep for q in pts):
                pts.append(p)
            if len(pts) == 2:
                break
        if len(pts) < 2:
            continue
        P = from_out(np.array(pts), img_wh, (w, h))
        out[LINE_NAMES[c]] = (tuple(P[0]), tuple(P[1]), float(min(m[int(pts[0][1]), int(pts[0][0])], m[int(pts[1][1]), int(pts[1][0])])))
    return out


def overlay(img_bgr, t, path=None):
    """Draw targets over a frame (for looking at them): landmark heat (red), all-lines channel (cyan), line
    extremities (yellow), each landmark labelled."""
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    kp = cv2.resize(t["kp"][:N_KP].max(0), (w, h), interpolation=cv2.INTER_LINEAR)
    ln = cv2.resize(t["ln"][N_LINE], (w, h), interpolation=cv2.INTER_LINEAR)
    ex = cv2.resize(t["ln"][:N_LINE].max(0), (w, h), interpolation=cv2.INTER_LINEAR)
    f = out.astype(np.float32)
    f = f * (1 - 0.55 * ln[..., None]) + 0.55 * ln[..., None] * np.array([255, 255, 0], np.float32)
    f = f * (1 - ex[..., None]) + ex[..., None] * np.array([0, 255, 255], np.float32)
    f = f * (1 - kp[..., None]) + kp[..., None] * np.array([0, 0, 255], np.float32)
    out = np.clip(f, 0, 255).astype(np.uint8)
    for c, p in enumerate(t["kp_px"]):
        if np.isfinite(p[0]):
            cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 2, (255, 255, 255), -1)
            cv2.putText(out, KP_NAMES[c], (int(p[0]) + 4, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    if path:
        cv2.imwrite(path, out)
    return out
