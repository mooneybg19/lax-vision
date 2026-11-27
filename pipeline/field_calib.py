"""Robust field calibration from the detectors' output, and the yard-error metric.

calibrate(): named keypoints (+ optional line extremities, + optional all-lines map) -> world(yd)->image H.
  1. RANSAC homography on the keypoints (cv2.RANSAC), threshold in image px.
  2. Robust (soft-L1) refinement of the 8 H parameters on the inlier keypoints plus, when given, every detected
     line extremity as a point-on-line constraint (its distance to the projected world line) and the all-lines
     heatmap as a truncated distance field (like the labeler's paint snap).
  3. Sanity checks; success needs >= MIN_INLIERS keypoint inliers (or 4 plus >= 2 agreeing lines).
Image coordinates are index coordinates (pixel centre at i). If the game's lens k1 is known, detections are
undistorted first and the returned H is the undistorted one.

yard_errors(): the evaluation metric (see docs/field_training.md).
"""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy import ndimage

import labels_io as LI
import field_targets as FT
from synth_cam import distort, undistort

MIN_INLIERS = 5


def _proj(H, X):
    q, w = LI.proj(H, X)
    return q, w


def world_kp(W, hashD):
    L = LI.landmarks(W, True, hashD)
    return {n: np.array(L[n][:2], float) for n in FT.KP_NAMES}


def calibrate(kp, W, hashD, fb, lines=None, border=None, k1=0.0, ransac_px=8.0, min_inliers=MIN_INLIERS,
              allowed=None, img_wh=(FT.IMG_W, FT.IMG_H), refine=True):
    """kp {name: (u, v, score)}; lines {line name: ((u0,v0), (u1,v1), score)}; border (h,w) all-lines probability
    map at the network output resolution or None. allowed: set of keypoint names valid for this field (default:
    all lacrosse names, plus football names if fb). Returns (H or None, info dict)."""
    WK = world_kp(W, hashD)
    if allowed is None:
        allowed = {n for n, f in zip(FT.KP_NAMES, FT.KP_IS_FB) if fb or not f}
    names = [n for n in kp if n in allowed]
    info = {"n_kp": len(names)}
    if len(names) < 4:
        info["reason"] = f"only {len(names)} keypoints"; return None, info
    X = np.array([WK[n] for n in names]); s = np.array([kp[n][2] for n in names], float)
    U = undistort(np.array([kp[n][:2] for n in names], float), k1, img_wh)
    method = cv2.RANSAC          # (USAC_MAGSAC returned None on clean data in OpenCV 5.0 here)
    H, m = cv2.findHomography(X.astype(np.float64), U.astype(np.float64), method, ransac_px, maxIters=10000, confidence=0.9999)
    if H is None or m is None:
        info["reason"] = "RANSAC failed"; return None, info
    inl = m.ravel().astype(bool)
    # line constraints in world: a line = (a, b) world endpoints -> homogeneous world line
    Lw = FT.line_world(W, hashD)
    line_obs = []
    if lines:
        for nm, (p0, p1, sc) in lines.items():
            if nm not in Lw or (FT.LINE_IS_FB[FT.LINE_NAMES.index(nm)] and not fb):
                continue
            a, b = np.array(Lw[nm][0], float), np.array(Lw[nm][1], float)
            lw = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
            for p in (p0, p1):
                line_obs.append((lw, undistort(np.array(p, float)[None], k1, img_wh)[0]))
    dtmap = None
    if border is not None:
        on = border > 0.5
        if on.sum() > 50:
            hh, ww = border.shape
            dtmap = (ndimage.distance_transform_edt(~on).astype(np.float32), hh, ww)

    def line_res(Hm, obs):
        r = []
        Hinv_T = np.linalg.inv(Hm).T
        for lw, p in obs:
            li = Hinv_T @ lw; li = li / np.hypot(li[0], li[1])
            r.append(li @ np.array([p[0], p[1], 1.0]))
        return np.array(r)

    def border_samples(Hm):
        segs = [Lw[n] for n, f in zip(FT.LINE_NAMES, FT.LINE_IS_FB) if fb or not f]
        P = []
        for a, b in segs:
            t = np.linspace(0, 1, 120)[:, None]
            P.append(np.array(a, float) + t * (np.array(b, float) - np.array(a, float)))
        P = np.concatenate(P)
        q, w = _proj(Hm, P)
        ok = (w > 0) & (q[:, 0] > 2) & (q[:, 0] < img_wh[0] - 3) & (q[:, 1] > 2) & (q[:, 1] < img_wh[1] - 3)
        return P[ok]

    Hn = LI.canon(H, X[inl])
    if refine:
        Xi, Ui, si = X[inl], U[inl], np.sqrt(np.clip(s[inl], 0.05, 1))
        # keep only line observations consistent with the RANSAC fit (within 3 x threshold)
        if line_obs:
            lr = np.abs(line_res(Hn, line_obs))
            line_obs = [o for o, e in zip(line_obs, lr) if e < 3 * ransac_px]
        BS = border_samples(Hn) if dtmap is not None else None
        T = np.array([[1 / 640, 0, -1.0], [0, 1 / 640, -360 / 640], [0, 0, 1.0]]); Ti = np.linalg.inv(T)

        def Hof(d):
            return Ti @ (np.eye(3) + np.append(d, 0).reshape(3, 3)) @ T @ Hn

        def res(d):
            Hm = Hof(d)
            q, w = _proj(Hm, Xi)
            r = [((q - Ui) * si[:, None]).ravel()]
            if line_obs:
                r.append(line_res(Hm, line_obs) * 0.7)
            if BS is not None and len(BS):
                qb, wb = _proj(Hm, BS)
                qb = distort(qb, k1, img_wh)
                qo = FT.to_out(qb, img_wh, (dtmap[2], dtmap[1]))
                x, y = qo[:, 0], qo[:, 1]
                ok = (x >= 0) & (y >= 0) & (x < dtmap[2] - 1) & (y < dtmap[1] - 1) & (wb > 0)
                dd = np.full(len(qb), 4.0)
                xi, yi = np.floor(x[ok]).astype(int), np.floor(y[ok]).astype(int); ax, ay = x[ok] - xi, y[ok] - yi
                D = dtmap[0]
                dd[ok] = (D[yi, xi] * (1 - ax) + D[yi, xi + 1] * ax) * (1 - ay) + (D[yi + 1, xi] * (1 - ax) + D[yi + 1, xi + 1] * ax) * ay
                scale = img_wh[0] / dtmap[2]
                r.append(np.minimum(dd, 4.0) * scale * 0.15)
            return np.concatenate(r)
        try:
            out = least_squares(res, np.zeros(8), loss="soft_l1", f_scale=3.0, max_nfev=200)
            Hr = LI.canon(Hof(out.x), Xi)
            # never accept a refinement that makes the keypoints much worse
            e0 = np.median(np.hypot(*(_proj(Hn, Xi)[0] - Ui).T)); e1 = np.median(np.hypot(*(_proj(Hr, Xi)[0] - Ui).T))
            if np.all(np.isfinite(Hr)) and e1 <= max(2 * e0, e0 + 2):
                Hn = Hr
        except Exception as ex:  # keep the RANSAC fit
            info["refine_error"] = str(ex)
    q, w = _proj(Hn, X)
    err = np.hypot(*(q - U).T)
    inl = err < ransac_px
    n_lines_ok = int(np.sum(np.abs(line_res(Hn, line_obs)) < ransac_px)) // 2 if line_obs else 0
    info.update({"inliers": int(inl.sum()), "reproj_px_median": float(np.median(err[inl])) if inl.any() else None,
                 "n_line_obs": len(line_obs), "lines_ok": n_lines_ok})
    if not (inl.sum() >= min_inliers or (inl.sum() >= 4 and n_lines_ok >= 2)):
        info["reason"] = f"{int(inl.sum())} inliers"; return None, info
    # sanity: inlier spread not degenerate; camera-side sideline below the far one; image centre maps near the field
    Xi = X[inl]
    ev = np.linalg.eigvalsh(np.cov(Xi.T)) if len(Xi) > 2 else np.zeros(2)
    if np.sqrt(max(ev[0], 0)) < 1.0:          # smallest spread (yd) of the inliers, as the labeler's spread() check
        info["reason"] = "inliers (nearly) collinear"; return None, info
    qn, wn = _proj(Hn, np.array([[55.0, 0.0], [55.0, W]]))
    if np.all(wn > 0) and not qn[0, 1] > qn[1, 1]:
        info["reason"] = "near sideline above far sideline"; return None, info
    c = LI.proj(np.linalg.inv(Hn), [[img_wh[0] / 2, img_wh[1] / 2]])[0][0]
    if not (-40 < c[0] < 150 and -40 < c[1] < W + 40):
        info["reason"] = "image centre maps off the field"; return None, info
    return Hn, info


# ------------------------------------------------------------------ the metric
def field_grid(img_wh=(FT.IMG_W, FT.IMG_H), step=16):
    xs = np.arange(step / 2, img_wh[0], step); ys = np.arange(step / 2, img_wh[1], step)
    return np.array([[x, y] for y in ys for x in xs], float)


def to_field(H, k1, pts, img_wh=(FT.IMG_W, FT.IMG_H)):
    """Image index coords -> field yards under (H undistorted world->image, k1). NaN where the ray misses the
    ground in front of the camera."""
    U = undistort(pts, k1, img_wh)
    # sign safety net: the bottom of a sideline frame always looks at ground in front of the camera
    bottom = undistort(np.array([[x, img_wh[1] - 20.0] for x in np.linspace(100, img_wh[0] - 100, 9)]), k1, img_wh)
    H = LI.canon(H, LI.proj(np.linalg.inv(H), bottom)[0])
    Hi = np.linalg.inv(H)
    q, w = LI.proj(Hi, U)
    # a point is in front of the camera iff projecting it back gives positive w
    _, w2 = LI.proj(H, q)
    q[~(w2 > 0)] = np.nan
    return q


def yard_errors(H_label, k1_label, H_pred, k1_pred, W, img_wh=(FT.IMG_W, FT.IMG_H), step=16):
    """Errors (yd) of the predicted mapping at the image grid points whose LABEL position is on the field
    (0..110 x 0..W). Returns array (one value per grid point; inf where the prediction misses the ground)."""
    g = field_grid(img_wh, step)
    wl = to_field(H_label, k1_label, g, img_wh)
    on = np.isfinite(wl[:, 0]) & (wl[:, 0] >= 0) & (wl[:, 0] <= 110) & (wl[:, 1] >= 0) & (wl[:, 1] <= W)
    wp = to_field(H_pred, k1_pred, g[on], img_wh)
    e = np.hypot(*(wp - wl[on]).T)
    e[~np.isfinite(e)] = np.inf
    return e


def summarize(per_frame, n_frames):
    """per_frame: {frame: errors array or None (calibration failed)}. Returns the reported numbers."""
    ok = {k: v for k, v in per_frame.items() if v is not None and len(v)}
    allp = np.concatenate(list(ok.values())) if ok else np.array([])
    fm = np.array([np.median(v) for v in ok.values()]) if ok else np.array([])
    fail_inf = np.r_[fm, np.full(n_frames - len(ok), np.inf)]
    def pc(a, p):   # 'higher' keeps infinite (failed) entries well-defined instead of inf - inf = nan
        return float(np.percentile(a, p, method="higher")) if len(a) else None
    return {"frames": n_frames, "calibrated": len(ok), "success_rate": len(ok) / max(1, n_frames),
            "points_median_yd": pc(allp, 50), "points_p90_yd": pc(allp, 90),
            "frame_median_of_medians_yd": pc(fm, 50), "frame_p90_of_medians_yd": pc(fm, 90),
            "all_frames_median_yd_fail_inf": pc(fail_inf, 50), "all_frames_p90_yd_fail_inf": pc(fail_inf, 90),
            "frames_under_1yd": int(np.sum(fm < 1.0)), "frames_under_2yd": int(np.sum(fm < 2.0))}
