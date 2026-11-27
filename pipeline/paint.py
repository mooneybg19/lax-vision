"""Port of the labeler's paint detector (tools/labeler/index.html: paintField, dtAt, paintSamples, visibleSamples,
matchAt, paintMatch). Painted lines = thin bright ridges on darker turf; the distance transform to the ridge centres
is what the labeler snapped H to. tests/test_paint.py checks the port against the JS in node on real frames.

`dist` (optional) is a callable mapping undistorted image points (N,2) -> distorted (N,2) so the same match can be
scored under a lens model (game_solve); None reproduces the labeler exactly.
"""
import numpy as np
from scipy import ndimage

import labels_io as LI

MATCH_PX, TRUNC, EDGE = 2.5, 6.0, 6


def paint_field(img_bgr):
    """{W, H, on (bool HxW), dt (float32 HxW)} exactly as the labeler's paintField."""
    d = img_bgr.astype(np.float64)
    g = (0.3 * d[..., 2] + 0.59 * d[..., 1] + 0.11 * d[..., 0]).astype(np.float32)
    H, W = g.shape
    I = np.zeros((H + 1, W + 1), np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(g.astype(np.float64), axis=1), axis=0)
    R = 9
    ys = np.arange(H); xs = np.arange(W)
    y0 = np.maximum(0, ys - R); y1 = np.minimum(H, ys + R + 1)
    x0 = np.maximum(0, xs - R); x1 = np.minimum(W, xs + R + 1)
    S = I[y1][:, x1] - I[y0][:, x1] - I[y1][:, x0] + I[y0][:, x0]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    rid = (g.astype(np.float64) - S / area).astype(np.float32)
    on = np.zeros((H, W), bool)
    v = rid[3:H - 3, 3:W - 3]
    hx = np.ones_like(v, bool); hy = np.ones_like(v, bool)
    for k in (1, 2, 3):
        hx &= ~((rid[3:H - 3, 3 - k:W - 3 - k] > v) | (rid[3:H - 3, 3 + k:W - 3 + k] > v))
        hy &= ~((rid[3 - k:H - 3 - k, 3:W - 3] > v) | (rid[3 + k:H - 3 + k, 3:W - 3] > v))
    on[3:H - 3, 3:W - 3] = (v >= 14) & (hx | hy)
    if on.any():
        dt = ndimage.distance_transform_edt(~on).astype(np.float32)
    else:
        dt = np.full((H, W), 1e5, np.float32)
    return {"W": W, "H": H, "on": on, "dt": dt}


def ridge_points(img_bgr, P=None, smooth=1.5, thresh=10.0):
    """Sub-pixel paint-ridge centres (N,2) and a KD-tree over them, index coordinates. Same ridge strength as the
    labeler (brightness minus the 19x19 local mean), but smoothed (Gaussian, `smooth` px) before taking local maxima,
    so a flat-topped painted line several pixels wide gives ONE ridge at its centre, then moved to the vertex of a
    parabola across the line (horizontal or vertical, whichever is sharper). The labeler's whole-pixel ridge is
    ~0.85 px from the true centre on clean 5-px synthetic lines; this measure is what game_solve uses."""
    from scipy.spatial import cKDTree
    d = img_bgr.astype(np.float64)
    g = (0.3 * d[..., 2] + 0.59 * d[..., 1] + 0.11 * d[..., 0]).astype(np.float32)
    H, W = g.shape
    I = np.zeros((H + 1, W + 1), np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(g.astype(np.float64), axis=1), axis=0)
    R = 9
    ys = np.arange(H); xs = np.arange(W)
    y0 = np.maximum(0, ys - R); y1 = np.minimum(H, ys + R + 1)
    x0 = np.maximum(0, xs - R); x1 = np.minimum(W, xs + R + 1)
    S = I[y1][:, x1] - I[y0][:, x1] - I[y1][:, x0] + I[y0][:, x0]
    rid = g.astype(np.float64) - S / ((y1 - y0)[:, None] * (x1 - x0)[None, :])
    rid = ndimage.gaussian_filter(rid, smooth)
    c = rid[2:-2, 2:-2]
    hx = (c >= rid[2:-2, 1:-3]) & (c > rid[2:-2, 3:-1]) & (c >= rid[2:-2, :-4]) & (c > rid[2:-2, 4:])
    hy = (c >= rid[1:-3, 2:-2]) & (c > rid[3:-1, 2:-2]) & (c >= rid[:-4, 2:-2]) & (c > rid[4:, 2:-2])
    on = np.zeros_like(rid, bool); on[2:-2, 2:-2] = (c >= thresh) & (hx | hy)
    yy, xx = np.nonzero(on)
    v = rid[yy, xx]
    l, r = rid[yy, xx - 1], rid[yy, xx + 1]; u, dn = rid[yy - 1, xx], rid[yy + 1, xx]
    cx = l - 2 * v + r; cy = u - 2 * v + dn                 # curvature (negative at a maximum)
    hxm = hx[yy - 2, xx - 2] & (cx < 0); hym = hy[yy - 2, xx - 2] & (cy < 0)
    use_x = hxm & (~hym | (cx <= cy))
    use_y = hym & ~use_x
    px = xx.astype(float); py = yy.astype(float)
    px[use_x] += np.clip(0.5 * (l - r)[use_x] / cx[use_x], -0.5, 0.5)
    py[use_y] += np.clip(0.5 * (u - dn)[use_y] / cy[use_y], -0.5, 0.5)
    pts = np.c_[px, py][use_x | use_y]
    return pts, (cKDTree(pts) if len(pts) else None)


def dist_sub(tree, q, trunc=TRUNC):
    """Distance (px) from each q to the nearest sub-pixel ridge point, capped at trunc; NaN q -> trunc."""
    out = np.full(len(q), float(trunc))
    ok = np.isfinite(q).all(1)
    if tree is None or not ok.any():
        return out
    d, _ = tree.query(q[ok], distance_upper_bound=trunc)
    out[ok] = np.minimum(d, trunc)
    return out


def dt_at(P, q):
    """Bilinear distance at points q (N,2); NaN where the labeler's dtAt returns null."""
    q = np.atleast_2d(q)
    u, v = q[:, 0], q[:, 1]
    x = np.floor(u); y = np.floor(v)
    ok = np.isfinite(u) & np.isfinite(v) & (x >= 0) & (y >= 0) & (x < P["W"] - 1) & (y < P["H"] - 1)
    out = np.full(len(q), np.nan)
    xi = x[ok].astype(int); yi = y[ok].astype(int); a = u[ok] - xi; b = v[ok] - yi; D = P["dt"]
    out[ok] = (D[yi, xi] * (1 - a) + D[yi, xi + 1] * a) * (1 - b) + (D[yi + 1, xi] * (1 - a) + D[yi + 1, xi + 1] * a) * b
    return out


def paint_samples(W, fb, hashD):
    """(N,4) [x, y, nx, ny] world samples along the lines the field actually has painted (labeler paintSamples).
    Column 4 (returned separately) flags samples whose position depends on the hash distance."""
    out = []; is_hash = []

    def seg(a, b, step):
        L = np.hypot(b[0] - a[0], b[1] - a[1]); n = max(2, int(np.ceil(L / step)))
        nx = -(b[1] - a[1]) / L; ny = (b[0] - a[0]) / L
        for i in range(n + 1):
            out.append([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, nx, ny]); is_hash.append(False)

    if fb:
        for a, b in LI.yard_lines(W):
            seg(a, b, .5)
        for X in range(5, 106):
            out.append([X, hashD, 0, 1]); is_hash.append(True)
            out.append([X, W - hashD, 0, 1]); is_hash.append(True)
    else:
        for a, b in LI.field_lines(W):
            seg(a, b, .5)
        for gx in (15, 95):
            for i in range(72):
                t = i / 72 * 6.2832
                out.append([gx + 3 * np.cos(t), W / 2 + 3 * np.sin(t), np.cos(t), np.sin(t)]); is_hash.append(False)
    return np.array(out, float), np.array(is_hash, bool)


def project(H, P, dist=None):
    q, w = LI.proj(H, P)
    q = np.where((w > 1e-6)[:, None], q, np.nan)
    if dist is not None:
        q = dist(q)
    return q


def visible(H, S, P, dist=None):
    q = project(H, S[:, :2], dist)
    m = np.isfinite(q[:, 0]) & (q[:, 0] >= EDGE) & (q[:, 1] >= EDGE) & (q[:, 0] <= P["W"] - EDGE) & (q[:, 1] <= P["H"] - EDGE)
    return m


def match_at(H, P, S, off, dist=None):
    q = project(H, S[:, :2] + off * S[:, 2:4], dist)
    d = dt_at(P, q)
    d = d[np.isfinite(d)]
    return float(np.mean(d <= MATCH_PX)) if len(d) else 0.0


def paint_match(H, P, W, fb, hashD, dist=None):
    S, _ = paint_samples(W, fb, hashD)
    S = S[visible(H, S, P, dist)]
    if len(S) < 40:
        return None
    frac = match_at(H, P, S, 0, dist)
    back = float(np.mean([match_at(H, P, S, o, dist) for o in (2.5, -2.5, 1.5, -1.5)]))
    return {"frac": frac, "back": back, "n": int(len(S))}
