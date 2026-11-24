"""v1 pass 2 (light, runs anywhere): read the set over each whole possession, with spot memory.

Inputs: the v1 tracks (track_v1.py, 8 fps around each v0 settled moment) and the v0 detections/teams (only
to learn the two team colors for this game). Output: results/<game>-sets-v1/moments.json.

How it reads a possession:
  1. Shared canvas. The camera-motion homographies are chained so every frame of the window lands on the
     first frame's picture. A player's spot keeps its place after the camera pans away.
  2. Field position, as in v0 but on the canvas: the goal (orange frame, median over the window) and the
     painted cross-field lines give the goal line; distance from it is measured in the player's own height
     (about 2 yards). 'u' = yards above the goal line (negative = behind), 'lat' = across the field.
  3. Teams by jersey color (the v0 colors for this game, nearest match); refs (stripes, colorless) and the
     goalie (the defender nearest the goal) are left out. Offense = the team standing farther from the goal.
     If that flips and stays flipped for 2 s, the possession changed: all spots are reset.
  4. Six offensive SPOTS, not six people. Set reading never needs to know who is who:
       - a visible offensive player fills the spot he stands in (matched by track first, then by distance)
       - when he drops out of view his spot is HELD at his last position ('memory'), or, if he left
         through the edge of the picture, held with 'exit-edge'
       - if he left the picture heading up the field, past the top of the zone, or through a sideline edge
         while up top, his spot is RELEASED as open (on-the-fly sub through the box at midfield); the next
         offensive player to come in fills it
       - a held spot fades over ~8 s, and is dropped at once if the camera looks there and nobody is in it
       - always 6: spots not seen and not held count as 'unknown', and the set match lets them sit wherever
         fits best (with a lower confidence)
  5. Each frame's six spots are counted in four bands (top, middle, front of the net, behind) and matched
     to the set library (top-first). The calls are smoothed over 2 s; the possession gets the set at
     initiation plus any later change (e.g. 2-3-1 -> 1-4-1).
Nothing is tuned to one game, camera or uniform.
"""
import argparse, glob, json, os
from collections import Counter
import numpy as np

SETS = {"1-4-1": (1, 4, 0, 1), "2-2-2": (2, 0, 2, 2), "2-3-1": (2, 2, 1, 1), "1-3-2": (1, 2, 1, 2),
        "3-3": (3, 0, 3, 0), "3-1-2": (3, 0, 1, 2), "Open": (1, 2, 2, 1), "Open ": (2, 1, 1, 2)}
TOP_YD, MID_YD, LOW_YD, ZONE_YD = 10.0, 4.0, -1.5, 24.0
SUB_U = 14.0          # a player leaving the picture up the field from here on is treated as heading to the box
HOLD_TAU, HOLD_MAX = 8.0, 15.0
MATCH_YD = 4.0
YD_PER_HEIGHT = 2.0
SETTLED_SPEED = 3.0   # yards/s: median offensive player speed below this = the set is being run, not transition
SETTLED_SHARE = 0.2   # a possession needs this share of settled frames, otherwise it's 'Skip' (not a set yet)
SET_WIDTH_YD = 22.0   # typical wing-to-wing width of a set, used once per game to undo the camera's tilt


def flip(k):
    return "-".join(reversed(k.split("-"))) if k[:1].isdigit() else k


def band(u):
    return 0 if u >= TOP_YD else 1 if u >= MID_YD else 3 if u < LOW_YD else 2


# 2-D set templates (top-first names): (across, up) in yards, goal at (0, 0), restraining line at up = 20.
# Spots: top ~14, high-middle ~10, wings ~5 and wide, crease ~3 and central, X / behind ~-5.
T2 = {
    "2-3-1": [(-6, 14), (6, 14), (-11, 5), (11, 5), (0, 3), (0, -5)],
    "1-4-1": [(0, 14), (-11, 5), (11, 5), (-4, 3), (4, 3), (0, -5)],
    "2-2-2": [(-6, 14), (6, 14), (-3, 3), (3, 3), (-4, -5), (4, -5)],
    "1-3-2": [(0, 14), (-11, 5), (11, 5), (0, 3), (-4, -5), (4, -5)],
    "3-3":   [(-11, 13), (0, 15), (11, 13), (-10, 3), (0, 3), (10, 3)],
    "3-1-2": [(-11, 13), (0, 15), (11, 13), (0, 3), (-4, -5), (4, -5)],
    "Open":  [(0, 15), (-9, 10), (9, 10), (-9, 1), (9, 1), (0, -6)],
    "Open ": [(-6, 14), (6, 14), (-12, 4), (12, 4), (-4, -6), (4, -6)],
}
T2 = {k: np.array(v, float) for k, v in T2.items()}
CAP2 = 36.0          # a point more than 6 yards from any template spot costs the same as 6 yards


def match2d(pts, w, n_unknown):
    """Best set for known spots pts (k x 2, across/up in yards) with weights w; the n_unknown missing
    players can stand on any template spot left over. Each template may stretch across the field a little
    (0.8-1.25x) and slide sideways, since sets are run wider or tighter."""
    from scipy.optimize import linear_sum_assignment
    if len(pts) < 3:
        return "Other", 0.0
    w = np.asarray(w, float)
    res = []
    for k, T in T2.items():
        best = 1e9
        for sc in (0.8, 1.0, 1.25):
            for sh in (-2.0, 0.0, 2.0):
                Tm = T * [sc, 1.0] + [sh, 0.0]
                C = np.minimum(((pts[:, None, :] - Tm[None, :, :]) ** 2).sum(-1), CAP2)
                r_, c_ = linear_sum_assignment(C * w[:, None])
                best = min(best, float((C[r_, c_] * w[r_]).sum() / w[r_].sum()))
        res.append((best, k.strip()))
    res.sort()
    d1, k1 = res[0]
    d2 = next(d for d, k in res[1:] if k != k1)
    if d1 > 20.0:
        return "Other", 0.0
    conf = float(np.clip((d2 - d1) / 8.0, 0, 1) * np.exp(-d1 / 15.0) * (1 - 0.12 * n_unknown))
    return k1, round(max(conf, 0.0), 2)


def match_set(counts, n_unknown):
    """Best set for known band counts plus n_unknown spots that can go anywhere."""
    c = np.array(counts, float)
    best = []
    for k, t in SETS.items():
        t = np.array(t, float)
        over = np.clip(c - t, 0, None).sum()           # known players in a band the set doesn't have room for
        short = np.clip(t - c, 0, None).sum()           # empty places the unknowns could fill
        cost = over + max(0.0, short - n_unknown)
        best.append((cost, k.strip()))
    best.sort()
    (d1, k1) = best[0]
    d2 = next(d for d, k in best[1:] if k != k1)
    if d1 > 2.0:
        return "Other", 0.0
    conf = float(np.clip((d2 - d1) / 2.0, 0, 1) * np.exp(-d1 / 1.5) * (1 - 0.15 * n_unknown))
    return k1, round(max(conf, 0.0), 2)


ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("--matcher", choices=["bands", "2d"], default="bands",
                help="bands: count spots by distance from the goal only (default; across-field positions from one "
                     "broadcast camera are too unreliable so far); 2d: match across+up spots to the set diagrams")
ap.add_argument("--naming", choices=["top-first", "goal-first"], default=os.environ.get("SET_NAMING", "top-first"))
a = ap.parse_args()
R = "results"
v0 = json.load(open(f"{R}/{a.game}-sets/moments.json"))
team_colors = v0["team_colors"]

# ---- team colors for this game, from v0 (per 10-minute window, for changing light)
segs = sorted(glob.glob(f"{R}/{a.game}-dets/seg*"))
D0 = []
fr0 = []
for s in segs:
    meta = json.load(open(os.path.join(s, "frames.json")))
    d = np.load(os.path.join(s, "dets.npz"))["dets"]
    if len(d):
        D0.append(d)
D0 = np.concatenate(D0)
T0 = np.load(f"{R}/{a.game}-sets/teams.npz")
# teams.npz rows follow analyze.py's concatenation order, which matches seg order here; align by time+box
key0 = {(round(float(t), 2), *map(int, b)): int(tm) for t, b, tm in zip(T0["t"], T0["box"], T0["team"])}
F0 = D0[:, 11:27] / 255.0
lab0 = np.array([key0.get((round(float(r[1]), 2), *map(int, r[3:7])), -9) for r in D0])
turf0 = D0[:, 8] > 0
win_centers = {}
for w0 in np.arange(0, D0[:, 1].max() + 600, 600.0):
    sel = (D0[:, 1] >= w0) & (D0[:, 1] < w0 + 600)
    cs = []
    for tm in (0, 1):
        s2 = sel & (lab0 == tm)
        cs.append(F0[s2].mean(0) if s2.sum() > 30 else None)
    win_centers[w0] = cs
gl = [F0[lab0 == tm].mean(0) for tm in (0, 1)]
tsel = (lab0 == 0) | (lab0 == 1)
print("v0 team labels matched:", int(tsel.sum()), "of", len(lab0))
spread = np.percentile(np.linalg.norm(F0[tsel] - np.array(gl)[lab0[tsel]], axis=1), 95)
sat_low0 = F0[:, 8:10].sum(1)
STRIPE_T = float(np.percentile(D0[turf0, 9], 85))
print("team colors", team_colors, "spread", round(float(spread), 3), "stripe thr", round(STRIPE_T, 2))


def team_of(f, t):
    cs = win_centers.get(600.0 * np.floor(t / 600.0), gl)
    cs = [c if c is not None else g for c, g in zip(cs, gl)]
    d = [np.linalg.norm(f - c) for c in cs]
    j = int(np.argmin(d))
    return j if d[j] <= spread else -1


def line_x_at(seg, y):
    x1, y1, x2, y2 = seg
    if y2 == y1:
        return None
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def ruler(segs, y):
    """Cross-field lines at screen height y -> (a, b) with the gap between neighbouring lines g(x) = a + b*x
    in pixels, or None. Neighbouring lines are a fixed distance apart on the ground (5 yards on fields that
    have them), so g(x) is a perspective-correct ruler along the field at that height."""
    xs = sorted(x for x in (line_x_at(sg, y) for sg in segs) if x is not None and -200 < x < 1480)
    cl = []
    for x in xs:                       # Hough gives several pieces per painted line
        if cl and x - cl[-1][-1] < 10:
            cl[-1].append(x)
        else:
            cl.append([x])
    c = [float(np.mean(v)) for v in cl]
    if len(c) < 4:
        return None
    gaps = np.diff(c); mids = (np.array(c[1:]) + np.array(c[:-1])) / 2
    med = np.median(gaps)
    keep = (gaps > 0.6 * med) & (gaps < 1.6 * med)   # drop extra lines (crease, goal line on a shared field)
    if keep.sum() < 3:
        return None
    A = np.vstack([np.ones(keep.sum()), mids[keep]]).T
    sol, *_ = np.linalg.lstsq(A, gaps[keep], rcond=None)
    a_, b_ = sol
    resid = np.abs(A @ sol - gaps[keep]) / gaps[keep]
    if np.median(resid) > 0.2:
        return None
    return float(a_), float(b_)


def ruler_yards(rl, x0, x1, step_yd=5.0):
    a_, b_ = rl
    g0, g1 = a_ + b_ * x0, a_ + b_ * x1
    if g0 <= 2 or g1 <= 2:
        return None
    if abs(b_) < 1e-4:
        return step_yd * (x1 - x0) / a_
    return step_yd * np.log(g1 / g0) / b_


def apply_h(Hm, x, y):
    p = Hm @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def run_moment(mdir, LATS=None, LS=1.0):
    meta = json.load(open(os.path.join(mdir, "frames.json")))
    D = np.load(os.path.join(mdir, "dets.npz"))["dets"]
    v0m = v0["moments"][meta["moment"] - 1]
    names = {int(k_): v for k_, v in (meta.get("lax_names") or {}).items()}
    not_field = {k_ for k_, v in names.items() if v in ("Referee", "Goalie")}
    Wd, Hd = meta["width"], meta["height"]
    frames = meta["frames"]
    n = len(frames)
    lp = os.path.join(mdir, "lines.json")
    LINES = json.load(open(lp))["lines"] if os.path.exists(lp) else [[] for _ in frames]
    rec = dict(moment=meta["moment"], start=v0m["start"], end=v0m["end"], mid=v0m["mid"],
               v0_set=v0m["set"], v0_offense=v0m["offense"])
    if n == 0 or len(D) == 0:
        rec.update(set="Other", set_goal_first="Other", set_display="Other", confidence=0.0, note="no frames")
        return rec
    # 1. canvas: chain camera motion; a camera cut (few background matches) starts a new piece
    Hc = [np.eye(3)]
    cuts = 0
    for i in range(1, n):
        Hp = np.array(frames[i]["H_to_prev"])
        if frames[i]["inliers"] < 25:
            cuts += 1
            Hc.append(Hc[-1].copy())   # unknown motion: keep the last placement
        else:
            Hc.append(Hc[-1] @ Hp)
    # 2. goal on the canvas
    gpts = [apply_h(Hc[i], (f["goal"][0] + f["goal"][2]) / 2, f["goal"][3]) for i, f in enumerate(frames) if f.get("goal")]
    angs = [f["gline_angle"] for f in frames if f.get("gline_angle") is not None]
    if len(gpts) < 3:
        rec.update(set="Other", set_goal_first="Other", set_display="Other", confidence=0.0, note="goal not seen")
        return rec
    gx, gy = np.median(np.array(gpts), axis=0)
    ang = np.radians(np.median(angs) if angs else 90.0)
    cot = np.cos(ang) / np.sin(ang) if abs(np.sin(ang)) > 1e-3 else 0.0
    # goal on each frame's own picture (from the canvas), for the yard ruler
    Hinvs = [np.linalg.inv(H_) for H_ in Hc]
    goal_img = [apply_h(Hinvs[i], gx, gy) for i in range(n)]
    rul_cache = {}
    n_ruler = n_height = 0
    # per detection: canvas foot, canvas height, u / lat
    det = []
    for r in D:
        fi = int(r[0]); x1, y1, x2, y2 = r[3:7]
        if r[8] <= 0:
            continue
        fx, fy = apply_h(Hc[fi], (x1 + x2) / 2, y2)
        tx, ty = apply_h(Hc[fi], (x1 + x2) / 2, y1)
        h = max(np.hypot(fx - tx, fy - ty), 1.0)
        u = (fx - (gx + (fy - gy) * cot)) / h * YD_PER_HEIGHT
        lat = (fy - gy) / h * YD_PER_HEIGHT
        # perspective-correct yards from the painted lines, when this frame has a regular set of them
        px, py = (x1 + x2) / 2, y2
        key = (fi, int(py // 8))
        if key not in rul_cache:
            rul_cache[key] = ruler(LINES[fi], py) if fi < len(LINES) else None
        rl = rul_cache[key]
        if rl is not None:
            gix, giy = goal_img[fi]
            gxa = gix + (py - giy) * cot          # goal line at the player's screen height
            uy = ruler_yards(rl, gxa, px)
            if uy is not None and abs(uy) < 60:
                u = uy; n_ruler += 1
            else:
                n_height += 1
        else:
            n_height += 1
        f = r[11:27] / 255.0
        ref = (f[8:10].sum() > 0.7 and r[9] > STRIPE_T) 
        edge = None
        if x1 < 0.015 * Wd: edge = "left"
        elif x2 > 0.985 * Wd: edge = "right"
        elif y2 > 0.985 * Hd: edge = "bottom"
        elif y1 < 0.015 * Hd: edge = "top"
        det.append(dict(fi=fi, t=float(r[1]), tid=int(r[2]), u=u, lat=lat, cx=fx, cy=fy, team=-2 if ref else team_of(f, float(r[1])),
                        lax=int(r[10]), edge=edge))
    if not det:
        rec.update(set="Other", set_goal_first="Other", set_display="Other", confidence=0.0, note="no players")
        return rec
    side = np.sign(np.median([d["u"] for d in det])) or 1.0
    for d in det:
        d["u"] *= side
    # track -> team by vote; drop refs; per frame lists
    votes = {}
    for d in det:
        if d["tid"] >= 0:
            votes.setdefault(d["tid"], []).append(d["team"])
    tteam = {}
    for tid, v in votes.items():
        c = Counter(v)
        tm, k = c.most_common(1)[0]
        tteam[tid] = tm if k >= 0.5 * len(v) else -1
    for d in det:
        if d["tid"] >= 0:
            d["team"] = tteam[d["tid"]]
    # The lacrosse detector, by whole track: a goalie that stays in front of the cage is not a field player.
    # Its 'Referee' label is left out: on this broadcast film it tags many field players as refs (tested: using
    # it dropped real players and pushed more possessions to 'Skip'); refs are still found by the stripe check.
    lv = {}
    for d in det:
        if d["tid"] >= 0 and d["lax"] >= 0:
            lv.setdefault(d["tid"], []).append((names.get(d["lax"]), d["u"]))
    drop = set()
    for tid, v in lv.items():
        if len(v) < 4:
            continue
        c = Counter(nm for nm, _ in v)
        if c["Goalie"] >= 0.6 * len(v) and abs(np.median([u for _, u in v])) < 4:
            drop.add(tid)
    for d in det:
        if d["tid"] in drop:
            d["team"] = -2
    byf = [[] for _ in range(n)]
    for d in det:
        if d["team"] in (0, 1) and -8 <= d["u"] <= ZONE_YD + 6 and abs(d["lat"]) <= 32:
            byf[d["fi"]].append(d)
    # 3. offense per frame, smoothed; possession changes
    raw = []
    for fl in byf:
        mu = [np.mean([d["u"] for d in fl if d["team"] == tm]) if sum(d["team"] == tm for d in fl) >= 3 else np.nan for tm in (0, 1)]
        raw.append(np.nan if np.isnan(mu).any() else (0 if mu[0] > mu[1] else 1))
    raw = np.array(raw, float)
    k = int(3 * meta["fps"])
    off = np.full(n, -1)
    for i in range(n):
        w = raw[max(0, i - k // 2): i + k // 2 + 1]
        w = w[~np.isnan(w)]
        if len(w) >= k // 3 and abs(w.mean() - 0.5) >= 0.3:   # a clear majority over 3 s
            off[i] = int(round(w.mean()))
    if (off >= 0).sum() == 0:
        rec.update(set="Other", set_goal_first="Other", set_display="Other", confidence=0.0, note="offense unclear")
        return rec
    for i in range(n):            # fill gaps
        if off[i] < 0:
            off[i] = off[i - 1] if i and off[i - 1] >= 0 else int(np.bincount(off[off >= 0]).argmax())
    # runs shorter than 3 s are noise, not a change of possession
    runs = []
    for i in range(n):
        if runs and runs[-1][0] == off[i]:
            runs[-1][2] = i
        else:
            runs.append([off[i], i, i])
    major = int(Counter(off).most_common(1)[0][0])
    for r_ in runs:
        if (r_[2] - r_[1] + 1) < k and r_[0] != major:
            off[r_[1]:r_[2] + 1] = major
    poss_change = int((np.diff(off) != 0).sum())
    if LATS is not None:          # pass 1: across-field spread of offensive players, for this game's camera tilt
        for i in range(n):
            if v0m["start"] <= frames[i]["t"] <= v0m["end"]:
                LATS.extend(d["lat"] for d in byf[i] if d["team"] == off[i] and -8 <= d["u"] <= 20)
        return None
    # 4. spots
    spots = []    # dict(u, lat, cx, cy, state, src, tid, last_t)
    released = 0
    filled_open = 0
    timeline = []
    cur_off = off[0]
    corners = np.array([[0, 0], [Wd, 0], [Wd, Hd], [0, Hd]], float)
    prev_vis = {}
    for i in range(n):
        t = frames[i]["t"]
        if off[i] != cur_off:            # possession changed: reset all spots
            spots, cur_off = [], off[i]
            prev_vis = {}
        vis = [d for d in byf[i] if d["team"] == cur_off]
        # how fast the offense is moving (yards/s, from each track's last position): settled offenses walk
        sp = []
        for d in vis:
            pv = prev_vis.get(d["tid"]) if d["tid"] >= 0 else None
            if pv is not None and t - pv[2] > 0:
                sp.append(np.hypot(d["u"] - pv[0], (d["lat"] - pv[1]) * LS) / (t - pv[2]))
        for d in vis:
            if d["tid"] >= 0:
                prev_vis[d["tid"]] = (d["u"], d["lat"], t)
        speed = float(np.median(sp)) if len(sp) >= 2 else np.nan
        # the goalie (defense) is excluded by team; an offensive player can't be deep in the crease for long,
        # so nothing else is removed here
        used = set()
        # (a) same track continues in its spot
        for s in spots:
            if s["state"] == "seen" and s["tid"] >= 0:
                m = [d for d in vis if d["tid"] == s["tid"] and id(d) not in used]
                if m:
                    d = m[0]; used.add(id(d))
                    s.update(u=d["u"], lat=d["lat"], cx=d["cx"], cy=d["cy"], last_t=t, edge=d["edge"])
                    s["hit"] = i
        # spots whose player vanished this frame
        for s in spots:
            if s["state"] == "seen" and s.get("hit") != i:
                ex = s.get("edge")
                up_exit = (ex == ("right" if side > 0 else "left"))
                side_exit = ex in ("top", "bottom")
                if s["u"] >= SUB_U and (up_exit or side_exit):
                    s["state"], s["src"] = "open", "released"
                    released += 1
                else:
                    s["state"], s["src"] = "held", ("exit-edge" if ex else "memory")
        # (b) unmatched visible players: nearest held spot within reach, then an open spot, then a new spot
        for d in vis:
            if id(d) in used:
                continue
            held = [s for s in spots if s["state"] == "held"]
            if held:
                dist = [np.hypot(s["u"] - d["u"], s["lat"] - d["lat"]) for s in held]
                j = int(np.argmin(dist))
                if dist[j] <= MATCH_YD:
                    held[j].update(state="seen", src="seen", tid=d["tid"], u=d["u"], lat=d["lat"], cx=d["cx"], cy=d["cy"], last_t=t, edge=d["edge"], hit=i)
                    used.add(id(d)); continue
            opn = [s for s in spots if s["state"] == "open"]
            if opn:
                s = opn[0]
                s.update(state="seen", src="seen", tid=d["tid"], u=d["u"], lat=d["lat"], cx=d["cx"], cy=d["cy"], last_t=t, edge=d["edge"], hit=i)
                filled_open += 1
                used.add(id(d)); continue
            if len(spots) < 6:
                spots.append(dict(state="seen", src="seen", tid=d["tid"], u=d["u"], lat=d["lat"], cx=d["cx"], cy=d["cy"], last_t=t, edge=d["edge"], hit=i))
                used.add(id(d)); continue
            # a 7th offensive player: replace the stalest held spot if it is far older
            held = sorted([s for s in spots if s["state"] == "held"], key=lambda s: s["last_t"])
            if held and t - held[0]["last_t"] > 2.0:
                held[0].update(state="seen", src="seen", tid=d["tid"], u=d["u"], lat=d["lat"], cx=d["cx"], cy=d["cy"], last_t=t, edge=d["edge"], hit=i)
                used.add(id(d))
        # held spots: fade, and drop if the camera now shows that spot empty for 1 s
        Hinv = np.linalg.inv(Hc[i])
        for s in spots:
            if s["state"] != "held":
                continue
            if t - s["last_t"] > HOLD_MAX:
                s["state"], s["src"] = "open", "expired"; continue
            x, y = apply_h(Hinv, s["cx"], s["cy"])
            inview = 0.03 * Wd < x < 0.97 * Wd and 0.05 * Hd < y < 0.97 * Hd
            if inview:
                s["empty_since"] = s.get("empty_since", t)
                if t - s["empty_since"] >= 1.0:
                    s["state"], s["src"] = "open", "moved"
            else:
                s.pop("empty_since", None)
        seen = [s for s in spots if s["state"] == "seen"]
        held = [s for s in spots if s["state"] == "held"]
        n_unknown = max(0, 6 - len(seen) - len(held))
        counts = [0, 0, 0, 0]
        for s in seen + held:
            counts[band(s["u"])] += 1
        pts = np.array([[s["lat"] * LS, s["u"]] for s in seen + held]) if seen + held else np.zeros((0, 2))
        wts = [1.0] * len(seen) + [float(np.exp(-(t - s["last_t"]) / HOLD_TAU)) for s in held]
        st, cf = match2d(pts, wts, n_unknown) if a.matcher == "2d" else match_set(counts, n_unknown)
        in_zone = sum(-8 <= s["u"] <= 20 for s in seen + held)
        settled = bool(in_zone >= 4 and (np.isnan(speed) or speed < SETTLED_SPEED))
        hold_conf = np.mean([np.exp(-(t - s["last_t"]) / HOLD_TAU) for s in held]) if held else 1.0
        cf = round(cf * (0.7 + 0.3 * hold_conf), 2)
        timeline.append(dict(t=t, set=st, conf=cf, seen=len(seen), mem=sum(s["src"] == "memory" for s in held),
                             edge=sum(s["src"] == "exit-edge" for s in held), unknown=n_unknown, counts=counts,
                             settled=settled, speed=None if np.isnan(speed) else round(speed, 2)))
    # only the settled part (v0 start..end) counts for the call; the padding is just there to warm up memory
    TL = [x for x in timeline if v0m["start"] - 0.01 <= x["t"] <= v0m["end"] + 0.01] or timeline
    settled_share = float(np.mean([r["settled"] for r in TL]))
    if settled_share < SETTLED_SHARE:
        rec.update(offense=team_colors[int(Counter(off).most_common(1)[0][0])], set="Skip", set_goal_first="Skip",
                   set_display="Skip", confidence=round(1 - settled_share, 2), settled_share=round(settled_share, 2),
                   note="offense not settled (moving or fewer than 4 in the zone)", timeline=[],
                   players_seen=round(float(np.mean([r["seen"] for r in TL])), 1),
                   players_inferred=round(float(np.mean([6 - r["seen"] for r in TL])), 1))
        tot["skip"] += 1
        return rec
    TL = [r for r in TL if r["settled"]]
    # 5. smooth over 2 s and cut into segments
    k = max(1, int(2 * meta["fps"]))
    sm = []
    for i in range(len(TL)):
        w = TL[max(0, i - k // 2): i + k // 2 + 1]
        v = Counter()
        for x in w:
            v[x["set"]] += x["conf"] + 0.02
        sm.append(v.most_common(1)[0][0])
    segs_ = []
    for x, s in zip(TL, sm):
        if segs_ and segs_[-1]["set"] == s:
            segs_[-1]["to"] = x["t"]; segs_[-1]["rows"].append(x)
        else:
            segs_.append(dict(set=s, frm=x["t"], to=x["t"], rows=[x]))
    # merge segments shorter than 1.5 s into the neighbour before them
    merged = []
    for sgm in segs_:
        if merged and (sgm["to"] - sgm["frm"] < 1.5):
            merged[-1]["to"] = sgm["to"]; merged[-1]["rows"] += sgm["rows"]
        else:
            merged.append(sgm)
    tl_out = []
    for sgm in merged:
        rows = sgm["rows"]
        tl_out.append(dict(frm=round(sgm["frm"], 1), to=round(sgm["to"], 1), set=sgm["set"], set_goal_first=flip(sgm["set"]),
                           confidence=round(float(np.mean([r["conf"] for r in rows if r["set"] == sgm["set"]] or [0])), 2),
                           players_seen=round(float(np.mean([r["seen"] for r in rows])), 1),
                           players_inferred=round(float(np.mean([6 - r["seen"] for r in rows])), 1)))
    named = [s for s in tl_out if s["set"] != "Other"]
    init = next((s for s in tl_out if s["set"] != "Other" and s["to"] - s["frm"] >= 2.0), named[0] if named else tl_out[0])
    seen_med = float(np.median([r["seen"] for r in TL]))
    src = Counter()
    for r in TL:
        src["memory"] += r["mem"]; src["exit-edge"] += r["edge"]; src["unknown"] += r["unknown"]
    total_inf = sum(src.values())
    rec.update(offense=team_colors[int(Counter(off).most_common(1)[0][0])],
               set=init["set"], set_goal_first=flip(init["set"]),
               set_display=init["set"] if a.naming == "top-first" else flip(init["set"]),
               confidence=init["confidence"],
               set_changes=[s["set"] for s in tl_out if s["set"] != "Other"],
               players_seen=round(float(np.mean([r["seen"] for r in TL])), 1),
               players_inferred=round(float(np.mean([6 - r["seen"] for r in TL])), 1),
               inferred_from={k_: round(v / max(len(TL), 1), 2) for k_, v in src.items()},
               frames_all_six_seen=round(sum(r["seen"] >= 6 for r in TL) / len(TL), 2),
               spots_released_for_subs=released, open_spots_filled=filled_open,
               possession_changes=poss_change, camera_cuts=cuts, settled_share=round(settled_share, 2),
               yards_from_lines=round(n_ruler / max(n_ruler + n_height, 1), 2), timeline=tl_out)
    tot["frames"] += len(TL); tot["six_seen"] += sum(r["seen"] >= 6 for r in TL)
    tot["one_plus_inferred"] += sum(r["seen"] < 6 for r in TL)
    tot["inf_memory"] += src["memory"]; tot["inf_edge"] += src["exit-edge"]; tot["inf_unknown"] += src["unknown"]
    tot["ruler"] += n_ruler; tot["height"] += n_height
    tot["released"] += released; tot["filled"] += filled_open; tot["poss_changes"] += poss_change
    return rec

out_moments = []
tot = Counter()
mdirs = sorted(glob.glob(f"{R}/{a.game}-tracks-v1/m*"))
LATS = []
for mdir in mdirs:
    run_moment(mdir, LATS)
# One number per game: across-field distances come out squeezed by the camera's downward tilt (screen height
# for across-field depth). Offensive players in a set span about SET_WIDTH_YD wing to wing; the game's spread
# (10th-90th percentile) sets the stretch. Not tuned per moment or to any label.
spread_lat = float(np.percentile(LATS, 90) - np.percentile(LATS, 10)) if len(LATS) > 50 else SET_WIDTH_YD
LS = float(np.clip(SET_WIDTH_YD / max(spread_lat, 1e-3), 1.0, 6.0))
print("across-field stretch for this game:", round(LS, 2), "(raw spread", round(spread_lat, 1), "yd)")
tot = Counter()
for mdir in mdirs:
    out_moments.append(run_moment(mdir, None, LS))

os.makedirs(f"{R}/{a.game}-sets-v1", exist_ok=True)
summary = dict(across_field_stretch=round(LS, 2), skipped=tot["skip"], frames=tot["frames"], share_all_six_seen=round(tot["six_seen"] / max(tot["frames"], 1), 2),
               share_some_inferred=round(tot["one_plus_inferred"] / max(tot["frames"], 1), 2),
               inferred_player_frames=dict(memory=tot["inf_memory"], exit_edge=tot["inf_edge"], unknown=tot["inf_unknown"]),
               share_positions_from_lines=round(tot["ruler"] / max(tot["ruler"] + tot["height"], 1), 2),
               spots_released_for_subs=tot["released"], open_spots_filled=tot["filled"], possession_changes=tot["poss_changes"],
               agree_with_v0=round(np.mean([m["set"] == m["v0_set"] for m in out_moments if "v0_set" in m]), 2),
               sets=dict(Counter(m["set"] for m in out_moments)))
json.dump(dict(team_colors=team_colors, stored_naming="top-first", display_naming=a.naming,
               set_library=sorted({k.strip() for k in SETS}) + ["Other"], summary=summary, moments=out_moments),
          open(f"{R}/{a.game}-sets-v1/moments.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
for m in out_moments:
    print(m["moment"], m["start"], m.get("offense"), "v0", m["v0_set"], "-> v1", m["set"], m.get("confidence"),
          "seen", m.get("players_seen"), "rel", m.get("spots_released_for_subs"), m.get("set_changes"), m.get("note", ""))
