"""PnLCalib-style field calibration for lacrosse (drop-in alternative to field_geom.keyframe_h).

Given named lacrosse landmarks located in the image (from the learned HRNet keypoint detector in pnl/, trained
on synthetic + auto-labelled frames), this computes the field<->image mapping as a camera, not a flat per-frame
homography, and returns the image->field mapping in the ENGINE's convention so the rest of the engine is unchanged:
    X = yards from the in-view goal line toward the field interior, Y = yards from the near sideline (goal at Y=GOAL_Y_ENG).

The planar ground-plane calibration here is robust (RANSAC) and line-refined; the full camera (shared focal across a
shot) is layered on top by the engine's existing shot chaining. This module has no soccer assumptions; it takes the
lacrosse world model from lax_field.
"""
import numpy as np
import cv2
import lax_field as LF

GOAL_Y_ENG = 160 / 3.0 / 2        # 26.67, matches field_geom's near-sideline convention downstream


def homography_from_points(world_xy, img_xy, ransac_px=6.0):
    """world (yd, absolute) -> image homography, robust. Needs >=4 correspondences. Returns (H_w2i, inliers) or (None,None)."""
    if len(world_xy) < 4:
        return None, None
    W = np.asarray(world_xy, np.float32)
    I = np.asarray(img_xy, np.float32)
    H, mask = cv2.findHomography(W, I, cv2.RANSAC, ransac_px, maxIters=2000, confidence=0.999)
    if H is None or mask is None or int(mask.sum()) < 4:
        return None, None
    return H, mask.ravel().astype(bool)


def refine_with_lines(H_w2i, line_img_pts):
    """Nonlinear refinement of the world->image homography against sampled points on detected field lines
    (each pt is (world_xy, image_xy)); least-squares on reprojection. Returns refined H."""
    from scipy.optimize import least_squares
    if not line_img_pts:
        return H_w2i
    Wp = np.array([w for w, _ in line_img_pts], float)
    Ip = np.array([i for _, i in line_img_pts], float)

    def resid(h):
        Hm = np.append(h, 1.0).reshape(3, 3)
        P = (Hm @ np.c_[Wp, np.ones(len(Wp))].T).T
        proj = P[:, :2] / P[:, 2:3]
        return (proj - Ip).ravel()

    h0 = (H_w2i / H_w2i[2, 2]).ravel()[:8]
    try:
        r = least_squares(resid, h0, method="lm", max_nfev=200)
        return np.append(r.x, 1.0).reshape(3, 3)
    except Exception:
        return H_w2i


def to_engine_frame(H_w2i, image_wh, goal_in_view_x, near_sideline_is_y0):
    """Convert an absolute-world(yd)->image homography into the engine's image->field homography:
    origin at the in-view goal line, +X toward interior, Y from the near sideline, goal at Y=GOAL_Y_ENG.
    goal_in_view_x is 15 or 95 (which goal); near_sideline_is_y0 says whether world y=0 is the near sideline."""
    H_i2w = np.linalg.inv(H_w2i)
    # map absolute world (x:0..110, y:0..60) -> engine (X from goal line, Y from near sideline 0..53.33 centered 26.67)
    if goal_in_view_x < LF.LENGTH / 2:            # left goal at x=15: interior is +x
        a, bx = 1.0, -goal_in_view_x
    else:                                         # right goal at x=95: interior is -x
        a, bx = -1.0, goal_in_view_x
    if near_sideline_is_y0:
        cy = 1.0; dy = GOAL_Y_ENG - LF.GOAL_Y     # world y=30 -> engine Y=26.67
    else:
        cy = -1.0; dy = GOAL_Y_ENG + LF.GOAL_Y
    M = np.array([[a, 0, bx], [0, cy, dy], [0, 0, 1.0]])   # world(abs) -> engine field
    H_i2eng = M @ H_i2w
    return H_i2eng / H_i2eng[2, 2]


def calibrate(kp_img_by_name, image_wh, line_img_pts=None, goal_in_view_x=15.0, near_sideline_is_y0=True):
    """Main entry. kp_img_by_name: {landmark_name: (u,v)} from the detector. Returns (H_image->engineField, info)."""
    names = [n for n in kp_img_by_name if n in LF.KEYPOINTS]
    if len(names) < 4:
        return None, {"reason": f"only {len(names)} known landmarks"}
    Wxy = [LF.KEYPOINTS[n] for n in names]
    Ixy = [kp_img_by_name[n] for n in names]
    H_w2i, inl = homography_from_points(Wxy, Ixy)
    if H_w2i is None:
        return None, {"reason": "homography RANSAC failed", "landmarks": len(names)}
    H_w2i = refine_with_lines(H_w2i, line_img_pts or [])
    H_eng = to_engine_frame(H_w2i, image_wh, goal_in_view_x, near_sideline_is_y0)
    # quality: reprojection rms of inlier landmarks (px)
    P = (H_w2i @ np.c_[np.array(Wxy), np.ones(len(Wxy))].T).T
    proj = P[:, :2] / P[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum((proj - np.array(Ixy)) ** 2, 1)[inl])))
    return H_eng, {"landmarks": len(names), "inliers": int(inl.sum()), "reproj_rms_px": round(rms, 2)}


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    # synthetic elevated-sideline camera: project the lacrosse world, recover it, check field accuracy.
    rng = np.random.default_rng(0)
    W, Hh = 1280, 720
    worst = 0.0
    for trial in range(200):
        # camera above the near sideline, looking across the field
        cam = np.array([rng.uniform(20, 90), rng.uniform(-25, -8), rng.uniform(4, 12)])   # yd, up z
        look = np.array([rng.uniform(40, 70), 30, 0]) - cam
        f = rng.uniform(900, 1700)
        fwd = look / np.linalg.norm(look)
        right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, -up, fwd])
        K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])

        def project(pt):
            c = R @ (np.array([pt[0], pt[1], 0.0]) - cam)
            if c[2] <= 0.1:
                return None
            p = K @ c
            return p[:2] / p[2]

        kp = {}
        for nm, (wx, wy) in LF.KEYPOINTS.items():
            q = project((wx, wy))
            if q is not None and -50 < q[0] < W + 50 and -50 < q[1] < Hh + 50:
                kp[nm] = (float(q[0]) + rng.normal(0, 1.5), float(q[1]) + rng.normal(0, 1.5))
        if len(kp) < 6:
            continue
        H_eng, info = calibrate(kp, (W, Hh), goal_in_view_x=15.0, near_sideline_is_y0=True)
        if H_eng is None:
            continue
        # check a known world point maps to the right engine-field location
        test_world = (15.0, 30.0)                 # left goal center -> engine X=0, Y=26.67
        q = project(test_world)
        if q is None:
            continue
        p = H_eng @ np.array([q[0], q[1], 1.0]); p = p[:2] / p[2]
        err = np.hypot(p[0] - 0.0, p[1] - GOAL_Y_ENG)
        worst = max(worst, err)
    print(f"self-test worst goal-center field error over trials: {worst:.2f} yd (want < ~1)")
