"""Synthetic sideline cameras, painted-field renders and fake label documents, for TESTS and PLUMBING ONLY.

Nothing here is valid for measuring real accuracy: the images are drawn lines on noise (or real frames with
made-up labels), so a model or solver that does well here has only shown that the code is wired correctly.
"""
import numpy as np
import cv2

import labels_io as LI

IMG_W, IMG_H = 1280, 720


def distort(q, k1, size=(IMG_W, IMG_H)):
    """One-parameter radial lens model shared with game_solve: p_d = c + (p_u - c)(1 + k1 r^2), r = |p_u - c| / max(w,h),
    c = the image centre in index coordinates ((w-1)/2, (h-1)/2), so the model is symmetric under a horizontal flip."""
    q = np.asarray(q, float)
    if not k1:
        return q
    c = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2]); S = float(max(size))
    d = q - c; r2 = (d ** 2).sum(-1, keepdims=True) / S ** 2
    return c + d * (1 + k1 * r2)


def distort_valid(q, k1, size=(IMG_W, IMG_H), margin=0.5):
    """Points (undistorted) where the lens model is used: within `margin` image sizes of the frame, and where the
    radial map is still monotonic (1 + 3 k1 r^2 > 0.2). Outside, a barrel model folds far points back into view."""
    q = np.asarray(q, float)
    c = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2]); S = float(max(size))
    r2 = ((q - c) ** 2).sum(-1) / S ** 2
    inside = (q[..., 0] > -margin * size[0]) & (q[..., 0] < (1 + margin) * size[0]) & \
             (q[..., 1] > -margin * size[1]) & (q[..., 1] < (1 + margin) * size[1])
    return inside & (1 + 3 * k1 * r2 > 0.2)


def undistort(q, k1, size=(IMG_W, IMG_H), iters=30):
    """Inverse of distort (Newton on the radius). Exact to ~1e-9 px for |k1| < 1 inside the image."""
    q = np.asarray(q, float)
    if not k1:
        return q
    c = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2]); S = float(max(size))
    d = q - c; rd = np.sqrt((d ** 2).sum(-1)) / S
    ru = rd.copy()
    for _ in range(iters):
        f = ru * (1 + k1 * ru ** 2) - rd; fp = 1 + 3 * k1 * ru ** 2
        ru = ru - f / fp
    scale = np.where(rd > 0, ru / np.maximum(rd, 1e-300), 1.0)
    return c + d * scale[..., None]


def camera_H(rng, W=60.0, f_range=(900, 1800)):
    """A random elevated near-sideline camera looking across the field; returns world(yd)->image homography."""
    for _ in range(1000):
        cam = np.array([rng.uniform(25, 85), rng.uniform(-25, -5), rng.uniform(8, 20)])
        target = np.array([rng.uniform(15, 95), rng.uniform(0.25, 0.65) * W, 0.0])
        fwd = target - cam; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        roll = rng.normal(0, 0.01)
        R = np.stack([right, -up, fwd])
        Rr = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rr @ R
        f = rng.uniform(*f_range)
        K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1.0]])
        t = -R @ cam
        H = K @ np.c_[R[:, 0], R[:, 1], t]
        H = H / H[2, 2]
        # keep cameras that see a decent piece of the field
        g = np.array([[x, y] for x in np.linspace(0, 110, 23) for y in np.linspace(0, W, 13)])
        q, w = LI.proj(H, g)
        inside = (w > 0) & (q[:, 0] > 0) & (q[:, 0] < IMG_W) & (q[:, 1] > 0) & (q[:, 1] < IMG_H)
        if inside.mean() > 0.12:
            return H
    raise RuntimeError("no camera")


def render(H, W, fb, hashD, k1=0.0, rng=None, base=None, line_px=3):
    """Draw the painted field (lacrosse lines; plus yard lines and hash ticks when fb) through H and the lens.
    base: optional real frame to draw on (plumbing), else noisy turf."""
    rng = rng or np.random.default_rng(0)
    if base is None:
        img = np.empty((IMG_H, IMG_W, 3), np.float32)
        img[:] = (40, 110, 50)
        img += rng.normal(0, 6, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = base.copy()

    def poly(P, col=(235, 235, 235), th=line_px):
        q, w = LI.proj(H, P)
        ok = (w > 1e-3) & distort_valid(q, k1)
        q = distort(q, k1)
        seg = []
        for qi, o in zip(q, ok):
            if o and abs(qi[0]) < 5 * IMG_W and abs(qi[1]) < 5 * IMG_H:
                seg.append(qi)
            else:
                if len(seg) > 1:
                    cv2.polylines(img, [np.round(np.array(seg) * 4).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=2)
                seg = []
        if len(seg) > 1:
            cv2.polylines(img, [np.round(np.array(seg) * 4).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=2)

    def seg(a, b):
        t = np.linspace(0, 1, max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.25)))[:, None]
        poly(np.array(a, float) + t * (np.array(b, float) - np.array(a, float)))

    for a, b in LI.field_lines(W):
        seg(a, b)
    for gx in (15, 95):
        a = np.linspace(0, 2 * np.pi, 97)
        poly(np.c_[gx + 3 * np.cos(a), W / 2 + 3 * np.sin(a)])
    if fb:
        for a, b in LI.yard_lines(W):
            seg(a, b)
        for X in range(5, 106):
            for y in (hashD, W - hashD):
                seg([X, y - 0.33], [X, y + 0.33])
    return img


def fake_label(frame, game, H, W, fb, hashD_true, rng, k1=0.0, click_sd=1.0, n_max=8, hash_labeled=None,
               width_labeled=None):
    """A label document as the labeler would save it (quality 'points', H = click fit), from a known camera.
    hash_labeled / width_labeled: what the labeler was told (may differ from the truth, to test game_solve)."""
    L_true = LI.landmarks(W, fb, hashD_true)
    names = []
    for n, v in L_true.items():
        q, w = LI.proj(H, [v[:2]])
        valid = distort_valid(q, k1)[0]
        q = distort(q, k1)[0]
        if valid and w[0] > 0 and 20 < q[0] < IMG_W - 20 and 20 < q[1] < IMG_H - 20:
            names.append(n)
    rng.shuffle(names)
    names = names[:n_max]
    if len(names) < 4:
        return None
    pts = {}
    for n in names:   # clicks are stored in the labeler's canvas convention (index + 0.5)
        q = distort(LI.proj(H, [L_true[n][:2]])[0], k1)[0] + rng.normal(0, click_sd, 2) + 0.5
        pts[n] = [round(float(q[0]), 1), round(float(q[1]), 1)]
    Wl = width_labeled or W
    hl = hash_labeled if hash_labeled is not None else hashD_true
    L_lab = LI.landmarks(Wl, fb, hl)
    wl = [[round(L_lab[n][0], 3), round(L_lab[n][1], 3)] for n in names]
    Hc = LI.fit_h([L_lab[n][:2] for n in names], [pts[n] for n in names])    # canvas convention, like the labeler
    q, _ = LI.proj(Hc, [L_lab[n][:2] for n in names])
    rms = float(np.sqrt(np.mean(np.sum((q - np.array([pts[n] for n in names])) ** 2, 1))))
    return {"frame": frame, "game": game, "skip": False, "img": [IMG_W, IMG_H], "pts": pts,
            "world": dict(zip(names, wl)), "field": "football" if fb else "lacrosse", "hash": hl if fb else None,
            "width": Wl, "widthKnown": True, "rms": round(rms, 2), "H": Hc.ravel().tolist(), "H_clicks": Hc.ravel().tolist(),
            "quality": "points", "snapped": False, "click_off": {}, "click_off_yd": {}, "synthetic": True}
