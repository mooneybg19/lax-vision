"""v2 field calibration: put every frame of a settled window on a real overhead field, in yards.

Shared turf fields carry the football grid (a painted line across the field every 5 yards, sidelines
53 1/3 yards apart), and the lacrosse goal sits on the field's center. That grid is the most reliable
landmark set on high-school sideline film, so it is the calibration target:

  1. White painted lines are found in each frame (turf only, players masked out).
  2. The camera-motion chain from the v1 tracks (`H_to_prev`) puts every frame of the window on the
     first frame's picture (the "canvas"). The camera sits on a tripod, so it only pans/tilts/zooms; one
     camera model per window then explains every frame.
  3. On the canvas, the family of lines meeting at one vanishing point = the across-field yard lines.
     Their order (with missing lines filled in by a 1-D projective fit) gives X = 5 yd x index.
  4. A pinhole camera (focal length, pan, tilt, roll, position) is fitted by least squares to:
       - the yard lines (X = 5k),
       - the near sideline (Y = 0) when the thick white border is in view,
       - the goal's base (the goal mouth is centered: Y = 26.67),
       - players' heights (a weak prior, about 1.9 yd tall, to pin the zoom).
     A camera model, not a free homography, is what keeps across-field distances honest from one
     sideline angle (the free homography has no way to know how wide the field is).
  5. Pass 2 fixes the camera's distance from the sideline and its height to the game-wide median (it's the
     same physical camera all game) and refits zoom/pan/tilt per window.

Homographies follow roboflow/sports' ViewTransformer convention (image points -> field points).
Outputs results/<game>-field-v2/mNN.json: per-frame image->field homography, fit quality, goal position.
Nothing is tuned to one game: the only constants are field dimensions and an average player height.
"""
import argparse, json, os, subprocess, sys
import cv2
import numpy as np
from scipy.optimize import least_squares

FIELD_W = 160 / 3.0      # football field width in yards (53 1/3); NCAA/NFHS allow lacrosse widths 53 1/3 - 60
STEP = 5.0
PLAYER_H = 1.9           # yards, helmet to cleats in a ready stance
GOAL_Y = FIELD_W / 2

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("--clips", required=True, help="folder with mNN.mp4 + moments.json (clip_start per moment)")
ap.add_argument("--tracks", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--only", default="")
ap.add_argument("--debug", default="")
a = ap.parse_args()
R = "results"
TR = a.tracks or f"{R}/{a.game}-tracks-v1"
OUT = a.out or f"{R}/{a.game}-field-v2"
os.makedirs(OUT, exist_ok=True)
CLIPM = {m["id"]: m for m in json.load(open(os.path.join(a.clips, "moments.json")))}


# ---------------------------------------------------------------- image -> painted line segments
def turf_mask(hsv):
    g = ((hsv[..., 0] > 28) & (hsv[..., 0] < 95) & (hsv[..., 1] > 50)).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return g


def find_segments(im, boxes):
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.int16)
    blur = cv2.GaussianBlur(v, (0, 0), 9).astype(np.int16)
    turf = turf_mask(hsv)
    white = ((v - blur > 22) & (hsv[..., 1] < 85) & (turf > 0)).astype(np.uint8) * 255
    for x1, y1, x2, y2 in boxes:                       # players and their sticks are not lines
        pw, ph = (x2 - x1) * 0.35, (y2 - y1) * 0.12
        white[int(max(0, y1 - ph)):int(min(H, y2 + ph)), int(max(0, x1 - pw)):int(min(W, x2 + pw))] = 0
    L = cv2.HoughLinesP(white, 1, np.pi / 720, 45, minLineLength=W // 16, maxLineGap=18)
    segs = [] if L is None else [list(map(float, s)) for s in L.reshape(-1, 4)]
    # near sideline: the thick white border along the bottom of the field. Its field-side (upper) edge.
    thick = ((v > 175) & (hsv[..., 1] < 60)).astype(np.uint8)
    thick = cv2.morphologyEx(thick, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(thick)
    side = None
    best = 0
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if w > 0.35 * W and area > 0.012 * W * H and y + h > 0.45 * H and area > best:
            cols = np.arange(x, x + w, 8)
            tops = []
            for c in cols:
                ys = np.nonzero(lab[:, c] == j)[0]
                if len(ys):
                    tops.append((c, ys.min()))
            if len(tops) > 20:
                P = np.array(tops, float)
                # robust line fit of the top edge
                best_in, best_l = None, None
                rng = np.random.default_rng(0)
                for _ in range(60):
                    i1, i2 = rng.choice(len(P), 2, replace=False)
                    if abs(P[i1, 0] - P[i2, 0]) < 30:
                        continue
                    m = (P[i2, 1] - P[i1, 1]) / (P[i2, 0] - P[i1, 0])
                    r = np.abs(P[:, 1] - (P[i1, 1] + m * (P[:, 0] - P[i1, 0])))
                    inl = r < 3
                    if best_in is None or inl.sum() > best_in.sum():
                        best_in = inl
                if best_in is not None and best_in.sum() > 0.6 * len(P):
                    Q = P[best_in]
                    m, c0 = np.polyfit(Q[:, 0], Q[:, 1], 1)
                    side = [float(Q[:, 0].min()), float(m * Q[:, 0].min() + c0), float(Q[:, 0].max()), float(m * Q[:, 0].max() + c0)]
                    best = area
    return segs, side


# ---------------------------------------------------------------- geometry helpers
def hpt(Hm, x, y):
    p = Hm @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def seg_line(s):
    p1 = np.array([s[0], s[1], 1.0]); p2 = np.array([s[2], s[3], 1.0])
    l = np.cross(p1, p2)
    return l / np.hypot(l[0], l[1])


def vp_family(segs):
    """Vanishing point shared by the most distinct long lines (the yard lines)."""
    if len(segs) < 3:
        return None, []
    S = np.array(segs)
    lens = np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])
    order = np.argsort(-lens)[:40]
    best = (0, None, None)
    dirs = np.stack([S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]], 1) / lens[:, None]
    mids = (S[:, :2] + S[:, 2:]) / 2
    for ii in range(len(order)):
        for jj in range(ii + 1, len(order)):
            i, j = order[ii], order[jj]
            vp = np.cross(seg_line(S[i]), seg_line(S[j]))
            if abs(vp[2]) > 1e-9:
                vpe = vp[:2] / vp[2]
                to = vpe[None, :] - mids
                tn = to / np.maximum(np.linalg.norm(to, axis=1, keepdims=True), 1e-9)
                cos = np.abs((tn * dirs).sum(1))
            else:                                   # parallel on the canvas: VP at infinity
                d = vp[:2] / np.hypot(vp[0], vp[1])
                cos = np.abs(dirs @ d)
            ok = cos > np.cos(np.radians(1.2))
            if ok.sum() < 3:
                continue
            # count distinct lines: cluster by perpendicular offset to the mean direction
            idx = np.nonzero(ok)[0]
            md = dirs[idx].copy(); md[md[:, 0] < 0] *= -1
            mdir = md.mean(0); mdir /= np.linalg.norm(mdir)
            nr = np.array([-mdir[1], mdir[0]])
            offs = np.sort(mids[idx] @ nr)
            n_lines = 1 + int((np.diff(offs) > 9.0).sum())
            score = n_lines * 1000 + lens[ok].sum() / 100
            if score > best[0]:
                best = (score, vp, ok)
    if best[1] is None:
        return None, []
    return best[1], list(np.nonzero(best[2])[0])


def group_lines(segs, idx, vp, ref_pt):
    """Group family segments into distinct lines; position s = where each crosses the transversal through
    ref_pt perpendicular to the family's mean direction."""
    S = np.array(segs)[idx]
    mids = (S[:, :2] + S[:, 2:]) / 2
    d = S[:, 2:] - S[:, :2]
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    d[d[:, 0] < 0] *= -1
    md = d.mean(0); md /= np.linalg.norm(md)
    nrm = np.array([-md[1], md[0]])
    # intersection of each segment's infinite line with the transversal ref_pt + s*nrm
    s_list = []
    for k in range(len(S)):
        l = seg_line(S[k])
        den = l[0] * nrm[0] + l[1] * nrm[1]
        if abs(den) < 1e-6:
            s_list.append(np.nan); continue
        s = -(l[0] * ref_pt[0] + l[1] * ref_pt[1] + l[2]) / den
        s_list.append(s)
    s_arr = np.array(s_list)
    order = np.argsort(s_arr)
    groups, cur = [], []
    for k in order:
        if np.isnan(s_arr[k]):
            continue
        if cur and s_arr[k] - s_arr[cur[-1]] > 7.0:
            groups.append(cur); cur = []
        cur.append(k)
    if cur:
        groups.append(cur)
    lines = []
    for g in groups:
        Lsum = sum(np.hypot(*(S[k, 2:] - S[k, :2])) for k in g)
        if Lsum < 60:
            continue
        lines.append(dict(s=float(np.median(s_arr[g])), segs=[S[k].tolist() for k in g], length=float(Lsum)))
    return lines, nrm


def assign_indices(svals):
    """Integer index per line (5-yd steps) by RANSAC on a 1-D projective map k -> s. Lines that don't fit the
    equal-spacing pattern (numbers, stray marks) are left out (index None). Returns (indices, inlier rms)."""
    s = np.array(svals, float)
    n = len(s)
    if n < 3:
        return None, 1e9

    def solve(k, sv):
        A = np.stack([k, np.ones_like(k), -k * sv], 1)
        p, *_ = np.linalg.lstsq(A, sv, rcond=None)
        return p

    def fwd(p, k):
        return (p[0] * k + p[1]) / (p[2] * k + 1)

    def inv(p, sv):
        return (sv - p[1]) / (p[0] - p[2] * sv)

    best = (0, 1e9, None)
    import itertools
    idx = range(n)
    for i, j, l in itertools.combinations(idx, 3):
        for d1 in (1, 2, 3):
            for d2 in (1, 2, 3):
                k3 = np.array([0.0, d1, d1 + d2])
                p = solve(k3, s[[i, j, l]])
                if not np.all(np.isfinite(p)) or abs(p[0] - p[2] * s[i]) < 1e-9:
                    continue
                kh = inv(p, s)
                kr = np.round(kh)
                pred = fwd(p, kr)
                spacing = np.abs(fwd(p, kr + 0.5) - fwd(p, kr - 0.5))
                ok = (np.abs(pred - s) < np.maximum(3.0, 0.06 * spacing)) & np.isfinite(pred)
                # one line per index
                if len(set(kr[ok].tolist())) != ok.sum():
                    continue
                cnt = int(ok.sum())
                err = float(np.sqrt(np.mean((pred[ok] - s[ok]) ** 2))) if cnt else 1e9
                if cnt > best[0] or (cnt == best[0] and err < best[1]):
                    best = (cnt, err, (kr, ok))
    if best[2] is None or best[0] < 3:
        return None, 1e9
    kr, ok = best[2]
    kk = [float(k - kr[ok].min()) if o else None for k, o in zip(kr, ok)]
    return kk, best[1]


# ---------------------------------------------------------------- camera model
R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # world (X along, Y across, Z up) -> camera axes


def rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry @ R0


def P_of(p, cx, cy):
    f, yaw, pitch, roll, Cx, Cy, Cz = p[:7]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Rm = rot(yaw, pitch, roll)
    t = -Rm @ np.array([Cx, Cy, Cz])
    return K @ np.hstack([Rm, t[:, None]])


def G_of(P):
    return P[:, [0, 1, 3]]          # field plane (Z=0) -> image


def proj(P, X, Y, Z=0.0):
    q = P @ np.array([X, Y, Z, 1.0])
    return q[:2] / q[2], q[2]


def line_img(G, X=None, Y=None):
    # field line X=c is (1,0,-c); its image line is G^-T l
    lf = np.array([1.0, 0, -X]) if X is not None else np.array([0, 1.0, -Y])
    li = np.linalg.inv(G).T @ lf
    return li / np.hypot(li[0], li[1])


def residuals(p, obs, cx, cy, fixed=None):
    q = np.array(p, float)
    if fixed is not None:
        q = np.concatenate([q[:5], fixed, q[5:]])
    P = P_of(q, cx, cy)
    G = G_of(P)
    r = []
    try:
        Gi = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return np.full(obs["n_res"], 1e3)
    for k, pts in obs["yard"]:
        li = line_img(G, X=STEP * k)
        r.extend((pts @ li[:2] + li[2]).tolist())
    for pts in obs["side"]:
        li = line_img(G, Y=0.0)
        r.extend((2.0 * (pts @ li[:2] + li[2])).tolist())
    if obs["goal"] is not None:
        gx = q[7]
        (u, v), z = proj(P, gx, GOAL_Y)
        r.extend([3.0 * (u - obs["goal"][0]), 3.0 * (v - obs["goal"][1])])
    for fx, fy, hpx in obs["heights"]:
        w = Gi @ np.array([fx, fy, 1.0]); X, Y = w[:2] / w[2]
        (u1, v1), z1 = proj(P, X, Y, 0.0)
        (u2, v2), z2 = proj(P, X, Y, PLAYER_H)
        hp = np.hypot(u2 - u1, v2 - v1)
        r.append(0.25 * (hp - hpx))
    # cameras sit behind the near sideline, above the field, in front of the lines they see
    pen = []
    pen.append(10.0 * max(0.0, q[5] + 3.0))      # Cy < -3
    pen.append(10.0 * max(0.0, 2.0 - q[6]))      # Cz > 2
    return np.array(r + pen)


def fit_camera(obs, cx, cy, xs_mid, fixed=None, init=None):
    best = None
    if init is not None:
        inits = [init]
    else:
        inits = []
        for Cy0 in (-12.0, -30.0, -55.0):
            for Cz0 in (6.0, 14.0, 25.0):
                for f0 in (900.0, 1700.0, 3200.0):
                    for dx in (-25.0, 0.0, 25.0):
                        Cx0 = xs_mid + dx
                        tx, ty = xs_mid, 22.0
                        yaw0 = np.arctan2(tx - Cx0, ty - Cy0)
                        dist = np.hypot(tx - Cx0, ty - Cy0)
                        pitch0 = np.arctan2(Cz0, dist)
                        inits.append([f0, yaw0, pitch0, 0.0, Cx0, Cy0, Cz0, obs.get("gx0", xs_mid)])
    for p0 in inits:
        p0 = np.array(p0, float)
        if fixed is not None:
            p0 = np.concatenate([p0[:5], p0[7:]])
        try:
            res = least_squares(residuals, p0, args=(obs, cx, cy, fixed), loss="soft_l1", f_scale=4.0, max_nfev=400)
        except Exception:
            continue
        if res.x[0] <= 200:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None
    q = best.x
    if fixed is not None:
        q = np.concatenate([q[:5], fixed, q[5:]])
    return q, best


def frame_obs(segs, side, goal, heights, W, H):
    """Observations for one frame: numbered yard lines, near sideline, goal base, player heights."""
    vp, fam = vp_family(segs)
    if vp is None or len(fam) < 3:
        return None, "no yard-line family"
    ref = goal if goal is not None and -W < goal[0] < 2 * W else np.array([W / 2, H / 2])
    lines, nrm = group_lines(segs, fam, vp, ref)
    if len(lines) < 3:
        return None, f"only {len(lines)} yard lines"
    kk, err1d = assign_indices([l["s"] for l in lines])
    if kk is None or err1d > 4:
        return None, "yard lines irregular"
    lines = [l for k, l in zip(kk, lines) if k is not None]
    kk = [k for k in kk if k is not None]
    yard = []
    for k, l in zip(kk, lines):
        pts = np.array([[s_[0], s_[1]] for s_ in l["segs"]] + [[s_[2], s_[3]] for s_ in l["segs"]], float)
        if len(pts) > 12:
            pts = pts[np.linspace(0, len(pts) - 1, 12).astype(int)]
        yard.append((float(k), pts))
    side_pts = []
    if side is not None:
        x1, y1, x2, y2 = side
        side_pts = [np.array([[x1 + (x2 - x1) * t, y1 + (y2 - y1) * t] for t in np.linspace(0, 1, 8)])]
    gx0 = STEP * float(np.mean(kk))
    if goal is not None:
        dists = [abs(line_dist(l, goal)) for l in lines]
        gx0 = STEP * float(kk[int(np.argmin(dists))])
    obs = dict(yard=yard, side=side_pts, goal=goal, heights=heights, gx0=gx0, xs_mid=STEP * float(np.mean(kk)),
               n_lines=len(lines), missing=int(max(kk) - len(kk) + 1))
    obs["n_res"] = sum(len(p_) for _, p_ in yard) + sum(len(p_) for p_ in side_pts) + (2 if goal is not None else 0) + len(heights) + 2
    return obs, "ok"


def fit_quality(q, obs, cx, cy):
    P = P_of(q, cx, cy); G = G_of(P); Gi = np.linalg.inv(G)
    dl = []
    for k, pts in obs["yard"]:
        li = line_img(G, X=STEP * k)
        dl.extend(np.abs(pts @ li[:2] + li[2]).tolist())
    ds = []
    for pts in obs["side"]:
        li = line_img(G, Y=0.0)
        ds.extend(np.abs(pts @ li[:2] + li[2]).tolist())
    hr = []
    for fx, fy, hpx in obs["heights"]:
        w = Gi @ np.array([fx, fy, 1.0]); X, Y = w[:2] / w[2]
        (u1, v1), _ = proj(P, X, Y); (u2, v2), _ = proj(P, X, Y, PLAYER_H)
        hr.append(np.hypot(u2 - u1, v2 - v1) / max(hpx, 1))
    return dl, ds, hr


KEY = 4      # a keyframe every 4 tracked frames (0.5 s at 8 fps)


def calibrate(md, fixed=None, cache=None, prev=None):
    meta = json.load(open(os.path.join(TR, md, "frames.json")))
    frames = meta["frames"]
    n = len(frames)
    W, H = meta["width"], meta["height"]
    cx, cy = W / 2, H / 2
    D = np.load(os.path.join(TR, md, "dets.npz"))["dets"]
    Hc = [np.eye(3)]
    for i in range(1, n):
        Hp = np.array(frames[i]["H_to_prev"])
        Hc.append(Hc[-1].copy() if frames[i]["inliers"] < 25 else Hc[-1] @ Hp)
    if cache is None:
        cm = CLIPM[md]
        fps = meta["fps"]
        cmd = ["ffmpeg", "-v", "error", "-i", os.path.join(a.clips, cm["clip"]), "-vf", f"fps={fps},scale={W}:{H}",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        off = int(round((cm["clip_start"] - frames[0]["t"]) * fps))
        per = {}
        j = 0
        while True:
            buf = proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            i = j + off
            if 0 <= i < n and i % KEY == 0:
                im = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                boxes = D[D[:, 0] == i][:, 3:7]
                per[i] = find_segments(im, boxes)
                if a.debug and i in (off + 8, off + 40):
                    dbg = im.copy()
                    for s_ in per[i][0]:
                        cv2.line(dbg, (int(s_[0]), int(s_[1])), (int(s_[2]), int(s_[3])), (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(a.debug, f"{md}_{i}.jpg"), dbg)
            j += 1
        proc.stdout.close(); proc.wait()
        cache = per
    per = cache
    gpts = [hpt(Hc[i], (f["goal"][0] + f["goal"][2]) / 2, f["goal"][3]) for i, f in enumerate(frames) if f.get("goal")]
    goal_c = np.median(np.array(gpts), 0) if len(gpts) >= 3 else None
    info = dict(moment=md, frames=n, keyframes=len(per), goal_frames=len(gpts))
    if goal_c is None:
        info["status"] = "goal not seen (across-field origin unknown)"
        return info, None, cache
    sols = {}
    fails = {}
    last = prev
    for i in sorted(per):
        segs, side = per[i]
        g = hpt(np.linalg.inv(Hc[i]), *goal_c)
        hs = []
        for r in D[D[:, 0] == i]:
            x1, y1, x2, y2 = r[3:7]
            if r[8] <= 0 or r[7] < 0.4 or y2 > 0.985 * H or y1 < 0.01 * H:
                continue
            hs.append(((x1 + x2) / 2, y2, float(y2 - y1)))
        obs, st = frame_obs(segs, side, g, hs, W, H)
        if obs is None:
            fails[st] = fails.get(st, 0) + 1
            continue
        init = None
        if last is not None:
            init = list(last[:7]) + [obs["gx0"]]
        out = fit_camera(obs, cx, cy, obs["xs_mid"], fixed=fixed, init=init)
        if out is not None and init is not None:
            dl, _, _ = fit_quality(out[0], obs, cx, cy)
            if np.sqrt(np.mean(np.square(dl))) > 6:
                out = fit_camera(obs, cx, cy, obs["xs_mid"], fixed=fixed)      # warm start failed: full search
        elif out is None:
            out = fit_camera(obs, cx, cy, obs["xs_mid"], fixed=fixed)
        if out is None:
            fails["fit failed"] = fails.get("fit failed", 0) + 1
            continue
        q = out[0]
        dl, ds, hr = fit_quality(q, obs, cx, cy)
        rms = float(np.sqrt(np.mean(np.square(dl))))
        if rms > 6:
            fails["poor fit"] = fails.get("poor fit", 0) + 1
            continue
        sols[i] = dict(q=q, rms=rms, side=float(np.sqrt(np.mean(np.square(ds)))) if ds else None,
                       hr=float(np.median(hr)) if hr else None, lines=obs["n_lines"], missing=obs["missing"])
        last = q
    info["fails"] = fails
    if not sols:
        info["status"] = "no keyframe calibrated"
        return info, None, cache
    keys = sorted(sols)
    # smooth camera parameters over keyframes (median of 3), goal position = median across the window
    Q = np.array([sols[k]["q"] for k in keys])
    Qs = Q.copy()
    for j in range(len(keys)):
        Qs[j] = np.median(Q[max(0, j - 1): j + 2], 0)
    # goal x is a per-keyframe free parameter tied to that keyframe's line numbering; express every keyframe
    # in goal-relative X so they agree
    Hk = {}
    for j, k in enumerate(keys):
        P = P_of(Qs[j], cx, cy)
        Gi = np.linalg.inv(G_of(P))
        T = np.array([[1.0, 0, -Qs[j][7]], [0, 1.0, 0], [0, 0, 1.0]])      # field X relative to goal line
        Hk[k] = T @ Gi
    Hs, src = [], []
    for i in range(n):
        k = min(keys, key=lambda kk_: abs(kk_ - i))
        if abs(k - i) > 4 * KEY:
            Hs.append(None); src.append(None); continue
        Hif = Hk[k] @ np.linalg.inv(Hc[k]) @ Hc[i]
        Hs.append((Hif / Hif[2, 2]).round(9).tolist()); src.append(k)
    rms = [sols[k]["rms"] for k in keys]
    sides = [sols[k]["side"] for k in keys if sols[k]["side"] is not None]
    hrs = [sols[k]["hr"] for k in keys if sols[k]["hr"] is not None]
    info.update(status="ok", keyframes_ok=len(keys), frames_with_field=int(sum(h is not None for h in Hs)),
                line_rms_px=round(float(np.median(rms)), 2), side_rms_px=round(float(np.median(sides)), 2) if sides else None,
                height_ratio_med=round(float(np.median(hrs)), 3) if hrs else None,
                lines_med=float(np.median([sols[k]["lines"] for k in keys])),
                cam=dict(f=round(float(np.median(Qs[:, 0])), 1), C=[round(float(v), 2) for v in np.median(Qs[:, 4:7], 0)]))
    return info, dict(H=Hs, key_of=src, goal_x=0.0, goal_y=GOAL_Y, cams=Qs.tolist(), keys=keys), cache


def line_dist(l, pt):
    S = np.array(l["segs"])
    ls = [seg_line(s) for s in S]
    return float(np.median([li[0] * pt[0] + li[1] * pt[1] + li[2] for li in ls]))


mds = sorted(d for d in os.listdir(TR) if d.startswith("m") and (not a.only or d in a.only.split(",")))
caches, infos, outs = {}, {}, {}
for md in mds:
    info, out, cache = calibrate(md)
    caches[md] = cache
    infos[md], outs[md] = info, out
    print(md, json.dumps({k: v for k, v in info.items() if k not in ("moment",)}), flush=True)

# pass 2: one physical camera all game -> median distance from sideline and height, refit pan/tilt/zoom
good = [o for m, o in outs.items() if o is not None]
if len(good) >= 5:
    Cy_m = float(np.median([c[5] for o in good for c in o["cams"]]))
    Cz_m = float(np.median([c[6] for o in good for c in o["cams"]]))
    mcy = [float(np.median([c[5] for c in o["cams"]])) for o in good]
    mcz = [float(np.median([c[6] for c in o["cams"]])) for o in good]
    spread = dict(Cy_iqr=float(np.subtract(*np.percentile(mcy, [75, 25]))), Cz_iqr=float(np.subtract(*np.percentile(mcz, [75, 25]))))
    print("pass 2: camera", round(Cy_m, 1), "yd behind sideline,", round(Cz_m, 1), "yd up; spread", spread, flush=True)
    for md in mds:
        info, out, _ = calibrate(md, fixed=np.array([Cy_m, Cz_m]), cache=caches[md])
        info["pass"] = 2
        infos[md], outs[md] = info, out
        print(md, "p2", json.dumps({k: v for k, v in info.items() if k in ("status", "line_rms_px", "side_rms_px", "height_ratio_med", "goal_x")}), flush=True)
    summary_cam = dict(Cy=Cy_m, Cz=Cz_m, **spread, n=len(good))
else:
    summary_cam = None

for md in mds:
    json.dump(dict(info=infos[md], **(outs[md] or {})), open(os.path.join(OUT, f"{md}.json"), "w"))
ok = [i for i in infos.values() if i.get("status") == "ok"]
summary = dict(moments=len(mds), calibrated=len(ok),
               line_rms_px_median=float(np.median([i["line_rms_px"] for i in ok])) if ok else None,
               side_rms_px_median=float(np.median([i["side_rms_px"] for i in ok if i.get("side_rms_px") is not None])) if any(i.get("side_rms_px") is not None for i in ok) else None,
               height_ratio_median=float(np.median([i["height_ratio_med"] for i in ok if i.get("height_ratio_med")])) if ok else None,
               camera=summary_cam,
               failures={m: i["status"] for m, i in infos.items() if i.get("status") != "ok"})
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
