"""v2 field calibration, fast and linear: one image->field homography per keyframe, in yards.

Shared turf fields carry the football grid: a painted line across the field every 5 yards, and a thick
near sideline. The lacrosse goal sits on the field's center line (across the field). From one frame:

  1. White painted lines are found (turf only, players masked out). The family of lines that meet at one
     vanishing point are the across-field yard lines; a 1-D projective fit numbers them (5 yards apart).
  2. Each numbered line meets the near sideline at a point whose field position is (5k, 0). The vanishing
     point is the field direction 'across' (a point at infinity). The goal base is (goal line, 26.67 yd).
     The goal's own X is read off the sideline: the line from the vanishing point through the goal hits the
     sideline at a spot the 1-D fit turns into a yard value. That fixes everything: a homography by DLT.
     No camera model, no search, no tuning: field dimensions are the only constants.
  3. Keyframes every 0.5 s; every other frame borrows the nearest good keyframe through the camera-motion
     chain from the v1 tracks (the camera only pans/tilts/zooms on its tripod).
  4. Quality: painted-line reprojection error (pixels) on the keyframe, and how far two keyframes of the
     same window disagree about where the same spot on the screen is (yards) -> an honest error estimate.

Fields without the football grid (or frames with no near sideline in view) get no field fit here; the set
reader then falls back to the v1 measurement for those frames, and reports which frames used which.
Field X = yards from the goal line (sign: toward the field interior positive is decided later by the
players); Y = yards from the near sideline (goal center at 26.67). Homographies map image -> field, the
roboflow/sports ViewTransformer convention.
"""
import argparse, json, os, subprocess
import cv2
import numpy as np

FIELD_W = 160 / 3.0
STEP = 5.0
GOAL_Y = FIELD_W / 2
KEY = 4                      # one keyframe per 4 tracked frames (0.5 s at 8 fps)

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("--clips", required=True, help="folder with mNN.mp4 + moments.json (clip_start per moment)")
ap.add_argument("--tracks", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--only", default="")
a = ap.parse_args()
TR = a.tracks or f"results/{a.game}-tracks-v1"
OUT = a.out or f"results/{a.game}-field-v2"
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




def hom(p):
    return np.array([p[0], p[1], 1.0])


def dlt(fpts, ipts):
    """Field -> image homography from homogeneous correspondences (points at infinity allowed)."""
    I = np.array(ipts, float)
    fin = np.abs(I[:, 2]) > 1e-9
    c = (I[fin, :2] / I[fin, 2:3]).mean(0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(I[fin, :2] / I[fin, 2:3] - c, axis=1)), 1e-6)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    F = np.array(fpts, float)
    finf = np.abs(F[:, 2]) > 1e-9
    cf = (F[finf, :2] / F[finf, 2:3]).mean(0)
    sf = np.sqrt(2) / max(np.mean(np.linalg.norm(F[finf, :2] / F[finf, 2:3] - cf, axis=1)), 1e-6)
    Tf = np.array([[sf, 0, -sf * cf[0]], [0, sf, -sf * cf[1]], [0, 0, 1.0]])
    A = []
    for X, x in zip(F, I):
        X = Tf @ X; x = T @ x
        X = X / np.linalg.norm(X); x = x / np.linalg.norm(x)
        u, v, w = x
        A.append(np.concatenate([np.zeros(3), -w * X, v * X]))
        A.append(np.concatenate([w * X, np.zeros(3), -u * X]))
    _, _, Vt = np.linalg.svd(np.array(A))
    Hn = Vt[-1].reshape(3, 3)
    return np.linalg.inv(T) @ Hn @ Tf


def number_lines(svals, max_gap=3, max_span=24):
    """Number the yard lines 0,1,2,.. (5 yards apart) from their screen positions s along a transversal.
    Screen position vs line number is a 1-D projective map with NO pole inside the visible range (the
    vanishing point is off to one side). RANSAC over triples with small integer gaps; lines that don't fit the
    equal-spacing pattern (numbers, crease, hash marks) get None."""
    import itertools
    s = np.array(svals, float)
    n = len(s)
    if n < 3:
        return None, 1e9
    lo, hi = s.min() - 50, s.max() + 50
    best = (0, 1e9, None)
    for i, j, l in itertools.combinations(range(n), 3):
        if not (s[i] < s[j] < s[l]):
            continue
        for d1 in range(1, max_gap + 1):
            for d2 in range(1, max_gap + 1):
                k3 = np.array([0.0, d1, d1 + d2])
                A = np.stack([k3, np.ones(3), -k3 * s[[i, j, l]]], 1)
                try:
                    p = np.linalg.solve(A, s[[i, j, l]])
                except np.linalg.LinAlgError:
                    continue
                den_lo, den_hi = p[0] - p[2] * lo, p[0] - p[2] * hi
                if den_lo * den_hi <= 0:          # pole inside the visible range
                    continue
                kh = (s - p[1]) / (p[0] - p[2] * s)
                kr = np.round(kh)
                pred = (p[0] * kr + p[1]) / (p[2] * kr + 1)
                spacing = np.abs((p[0] * (kr + 0.5) + p[1]) / (p[2] * (kr + 0.5) + 1) - (p[0] * (kr - 0.5) + p[1]) / (p[2] * (kr - 0.5) + 1))
                ok = (np.abs(pred - s) < np.maximum(3.0, 0.08 * spacing)) & np.isfinite(pred)
                if ok.sum() < 3 or len(set(kr[ok].tolist())) != ok.sum():
                    continue
                span = kr[ok].max() - kr[ok].min()
                if span > max_span or np.max(np.diff(np.sort(kr[ok]))) > max_gap:
                    continue
                cnt = int(ok.sum())
                err = float(np.sqrt(np.mean((pred[ok] - s[ok]) ** 2)))
                if cnt > best[0] or (cnt == best[0] and err < best[1]):
                    best = (cnt, err, (kr, ok))
    if best[2] is None:
        return None, 1e9
    kr, ok = best[2]
    return [float(k - kr[ok].min()) if o else None for k, o in zip(kr, ok)], best[1]


def fit_1d(k, t):
    """t = (p0 k + p1) / (p2 k + 1)."""
    A = np.stack([k, np.ones_like(k), -k * t], 1)
    p, *_ = np.linalg.lstsq(A, t, rcond=None)
    return p


def keyframe_h(segs, side, g):
    """-> (H image->field, info) or (None, reason)."""
    if side is None:
        return None, "near sideline not in view"
    vp, fam = vp_family(segs)
    if vp is None or len(fam) < 3:
        return None, "no yard-line family"
    lines, nrm = group_lines(segs, fam, vp, g)
    if len(lines) < 3:
        return None, "fewer than 3 yard lines"
    kk, err1d = number_lines([l["s"] for l in lines])
    if kk is None or err1d > 4:
        return None, "yard lines irregular"
    sl = seg_line(side)
    pts, ks, L = [], [], []
    for k, l in zip(kk, lines):
        if k is None:
            continue
        p_on = g + l["s"] * nrm
        li = np.cross(vp, hom(p_on))
        P = np.cross(li, sl)
        if abs(P[2]) < 1e-9:
            continue
        pts.append(P[:2] / P[2]); ks.append(k); L.append(l)
    if len(ks) < 3:
        return None, "fewer than 3 lines reach the sideline"
    pts = np.array(pts); ks = np.array(ks, float)
    d = np.array([side[2] - side[0], side[3] - side[1]]); d /= np.linalg.norm(d)
    t = pts @ d
    p = fit_1d(ks, t)
    pred = (p[0] * ks + p[1]) / (p[2] * ks + 1)
    r1 = float(np.sqrt(np.mean((pred - t) ** 2)))
    if r1 > 6:
        return None, "sideline crossings irregular"
    lg = np.cross(vp, hom(g))
    S = np.cross(lg, sl)
    if abs(S[2]) < 1e-9:
        return None, "goal line parallel to sideline"
    tg = (S[:2] / S[2]) @ d
    den = p[0] - p[2] * tg
    if abs(den) < 1e-9:
        return None, "goal off the numbering"
    kg = (tg - p[1]) / den
    if not (min(ks) - 8 <= kg <= max(ks) + 8):
        return None, "goal far outside the numbered lines"
    F = [[STEP * (k - kg), 0.0, 1.0] for k in ks] + [[0.0, GOAL_Y, 1.0], [0.0, 1.0, 0.0]]
    I = [hom(q) for q in pts] + [hom(g), vp]
    Hf2i = dlt(F, I)
    Hi2f = np.linalg.inv(Hf2i)
    Hi2f /= Hi2f[2, 2]
    # quality: painted segments vs reprojected field lines
    dl = []
    for k, l in zip(ks, L):
        X = STEP * (k - kg)
        li = Hi2f.T @ np.array([1.0, 0, -X])       # image line of field line X = const
        li = li / np.hypot(li[0], li[1])
        for s_ in l["segs"]:
            dl += [abs(li @ hom(s_[:2])), abs(li @ hom(s_[2:]))]
    rms = float(np.sqrt(np.mean(np.square(dl))))
    if rms > 6:
        return None, "poor line fit"
    return Hi2f, dict(rms=rms, lines=len(ks), r1d=r1, kg=float(kg))


def apply(Hm, x, y):
    q = Hm @ np.array([x, y, 1.0])
    return q[:2] / q[2]


def calibrate(md):
    meta = json.load(open(os.path.join(TR, md, "frames.json")))
    frames = meta["frames"]; n = len(frames)
    W, H = meta["width"], meta["height"]
    D = np.load(os.path.join(TR, md, "dets.npz"))["dets"]
    Hc = [np.eye(3)]
    for i in range(1, n):
        Hp = np.array(frames[i]["H_to_prev"])
        Hc.append(Hc[-1].copy() if frames[i]["inliers"] < 25 else Hc[-1] @ Hp)
    info = dict(moment=md, frames=n)
    gpts = [apply(Hc[i], (f["goal"][0] + f["goal"][2]) / 2, f["goal"][3]) for i, f in enumerate(frames) if f.get("goal")]
    if len(gpts) < 3:
        info.update(status="goal not seen", method="none", frames_with_field=0)
        return info, None
    goal_c = np.median(np.array(gpts), 0)
    cm = CLIPM[md]; fps = meta["fps"]
    cmd = ["ffmpeg", "-v", "error", "-i", os.path.join(a.clips, cm["clip"]), "-vf", f"fps={fps},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    off = int(round((cm["clip_start"] - frames[0]["t"]) * fps))
    keys, fails = {}, {}
    j = 0
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        i = j + off
        j += 1
        if not (0 <= i < n and i % KEY == 0):
            continue
        im = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        segs, side = find_segments(im, D[D[:, 0] == i][:, 3:7])
        g = apply(np.linalg.inv(Hc[i]), *goal_c)
        Hk, why = keyframe_h(segs, side, g)
        if Hk is None:
            fails[why] = fails.get(why, 0) + 1
        else:
            keys[i] = (Hk, why)
    proc.stdout.close(); proc.wait()
    info["keyframes_tried"] = sum(fails.values()) + len(keys)
    info["fails"] = fails
    if not keys:
        info.update(status="no keyframe calibrated", method="none", frames_with_field=0)
        return info, None
    K = sorted(keys)
    # every keyframe expressed on the canvas: canvas -> field
    Cf = {k: keys[k][0] @ np.linalg.inv(Hc[k]) for k in K}
    # disagreement between keyframes (yards): project a grid of canvas points near players through each pair
    foot = np.array([apply(Hc[int(r[0])], (r[3] + r[5]) / 2, r[6]) for r in D[::7] if r[8] > 0]) if len(D) else np.zeros((0, 2))
    dis = []
    for a1 in range(len(K)):
        for b1 in range(a1 + 1, min(a1 + 4, len(K))):
            if len(foot) == 0:
                break
            pa = np.array([apply(Cf[K[a1]], *p) for p in foot]); pb = np.array([apply(Cf[K[b1]], *p) for p in foot])
            ok = (np.abs(pa).max(1) < 120) & (np.abs(pb).max(1) < 120)
            if ok.any():
                dis.append(float(np.median(np.linalg.norm(pa[ok] - pb[ok], axis=1))))
    Hs, src = [], []
    for i in range(n):
        k = min(K, key=lambda kk_: abs(kk_ - i))
        if abs(k - i) > 6 * KEY:
            Hs.append(None); src.append(None); continue
        Hif = Cf[k] @ Hc[i]
        Hs.append((Hif / Hif[2, 2]).round(9).tolist()); src.append(k)
    info.update(status="ok", method="football-grid", keyframes_ok=len(K),
                frames_with_field=int(sum(h is not None for h in Hs)),
                line_rms_px=round(float(np.median([keys[k][1]["rms"] for k in K])), 2),
                keyframe_disagreement_yd=round(float(np.median(dis)), 2) if dis else None)
    return info, dict(H=Hs, key_of=src, goal_y=GOAL_Y)


mds = sorted(d for d in os.listdir(TR) if d.startswith("m") and (not a.only or d in a.only.split(",")))
for md in mds:
    info, out = calibrate(md)
    json.dump(dict(info=info, **(out or {})), open(os.path.join(OUT, f"{md}.json"), "w"))
    print(md, json.dumps({k: v for k, v in info.items() if k != "moment"}), flush=True)
