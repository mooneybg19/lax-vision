"""Pass 2 (light, runs anywhere): turn the saved detections into settled-offense moments and set calls.

Steps, all fit fresh for each game (nothing hard-coded to one field, camera or uniform):
  1. Team split: cluster the torso color signatures of everyone standing on the turf. The two biggest
     clusters are the teams; striped, colorless shirts are refs; the rest is 'other' (goalies in a
     different top, coaches, partial boxes).
  2. Field position: the goal is the landmark. The goal line runs through the goal parallel to the painted
     cross-field lines. Each player's distance from the goal line is measured in his own height on screen
     (a player is about 2 yards tall), so it works near and far from the camera.
  3. Settled offense: goal in view, at least 4 of each team within the offensive zone, held 4+ seconds.
  4. Offense = the team standing farther from the goal on average (the defense collapses toward the
     goal and owns the goalie).
  5. Set: count the offense in four bands - top, middle, front of the net, behind the goal - and match
     to the set library (top-first naming). Calls are smoothed over the moment.
     Calls are smoothed over the moment.
"""
import argparse, glob, json, os
import numpy as np
from sklearn.cluster import KMeans

# Set library, counted TOP-FIRST (Crown Point convention: from the restraining line toward/behind the goal).
# Bands: top (10+ yds above the goal line), middle (4-10 yds), front (in front of the net / crease /
# goal line extended, -1.5 to 4 yds), behind (more than 1.5 yds behind the goal line).
# Template = (top, middle, front, behind). Sources: crownpointlacrosse.com/oform, lacrosseunlimited.com.
SETS = {
    "1-4-1": (1, 4, 0, 1),
    "2-2-2": (2, 0, 2, 2),
    "2-3-1": (2, 2, 1, 1),
    "1-3-2": (1, 2, 1, 2),
    "3-3":   (3, 0, 3, 0),
    "3-1-2": (3, 0, 1, 2),   # umbrella
    "Open":  (1, 2, 2, 1),   # wheel / circle
    "Open ": (2, 1, 1, 2),   # rotated open (2 top, 2 behind); reported as "Open"
}
# Calls are always STORED top-first (Crown Point). Display direction is a switch: "top-first" or
# "goal-first" (Lacrosse Unlimited, where the same 1-3-2 is called 2-3-1). Set with --naming or the
# SET_NAMING environment variable. Named sets (Open, Other) never flip.
TOP_YD, MID_YD, LOW_YD, ZONE_YD = 10.0, 4.0, -1.5, 22.0


def set_name(k):
    return k.strip()


def flip(k):
    """Top-first name -> goal-first name (2-3-1 <-> 1-3-2, 3-1-2 <-> 2-1-3; symmetric and named sets unchanged)."""
    return "-".join(reversed(k.split("-"))) if k[:1].isdigit() else k
YD_PER_HEIGHT = 2.0

ap = argparse.ArgumentParser()
ap.add_argument("dets_dir")
ap.add_argument("out")
ap.add_argument("--naming", choices=["top-first", "goal-first"], default=os.environ.get("SET_NAMING", "top-first"))
a = ap.parse_args()
NAMING = a.naming

dets, frames = [], []
for seg in sorted(glob.glob(os.path.join(a.dets_dir, "seg*"))):
    meta = json.load(open(os.path.join(seg, "frames.json")))
    d = np.load(os.path.join(seg, "dets.npz"))["dets"]
    base = len(frames)
    for f in meta["frames"]:
        frames.append(f)
    if len(d):
        d = d.copy()
        d[:, 0] += base
        dets.append(d)
D = np.concatenate(dets) if dets else np.zeros((0, 27), np.float32)
order = np.argsort([f["t"] for f in frames])
remap = np.empty(len(frames), int)
remap[order] = np.arange(len(frames))
frames = [frames[i] for i in order]
D[:, 0] = remap[D[:, 0].astype(int)]
print(f"{len(frames)} frames, {len(D)} detections")

# ---- 1. team split (per game)
turf = D[:, 8] > 0
F = D[:, 11:27] / 255.0
sat_low = F[:, 8:10].sum(1)          # share of torso pixels with low saturation
stripes = D[:, 9]
is_ref = turf & (sat_low > 0.7) & (stripes > np.percentile(stripes[turf], 85))
cand = turf & ~is_ref
rng = np.random.default_rng(0)
idx = np.flatnonzero(cand)
fit_idx = rng.choice(idx, min(len(idx), 20000), replace=False)
# whole-game fit: 3 groups (two teams + everything else), the two biggest are the teams
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(F[fit_idx])
sizes = np.bincount(km.labels_, minlength=3)
teams = list(np.argsort(sizes)[::-1][:2])
centers = km.cluster_centers_[teams]
team = np.full(len(D), -1)            # 0 = team A, 1 = team B, 2 = ref, -1 = other/off field
# lighting changes over a game (sun, shadows, lights): refit the two team colors in 10-minute windows,
# starting from the whole-game colors so team A stays team A
tt = D[:, 1]
for w0 in np.arange(tt.min(), tt.max() + 1, 600.0):
    sel = cand & (tt >= w0) & (tt < w0 + 600)
    if sel.sum() < 50:
        continue
    kw = KMeans(n_clusters=2, init=centers, n_init=1, random_state=0).fit(F[sel])
    c = kw.cluster_centers_
    if np.linalg.norm(c[0] - centers[0]) + np.linalg.norm(c[1] - centers[1]) > \
       np.linalg.norm(c[0] - centers[1]) + np.linalg.norm(c[1] - centers[0]):
        c = c[::-1]
    d = np.stack([np.linalg.norm(F[sel] - c[0], axis=1), np.linalg.norm(F[sel] - c[1], axis=1)], 1)
    near = d.argmin(1)
    # far from both team colors (goalie in a different top, coach, partial box): leave as 'other'
    spread = np.percentile(d.min(1), 90)
    lab_w = np.where(d.min(1) <= spread, near, -1)
    team[np.flatnonzero(sel)] = lab_w
team[is_ref] = 2
sizes = np.array([(team == 0).sum(), (team == 1).sum(), (team == -1).sum() - (~turf).sum()])


def color_name(c):
    """Rough jersey color from the torso signature: light vs dark first, a hue only if it is strong."""
    hue = c[:8]; s_ = c[8:12]; v = c[12:16]
    names = ["red", "orange", "yellow", "green", "teal", "blue", "purple", "pink"]
    hname = names[int(np.argmax(hue))] if hue.sum() > 0.25 else None
    if v[2:].sum() > 0.55 and s_[:2].sum() > 0.6:
        return "white"
    if v[:2].sum() > 0.5:
        return f"dark {hname}" if hname else "dark (navy/black)"
    return hname or "gray"


team_colors = [color_name(centers[0]), color_name(centers[1])]
print("teams:", team_colors, "sizes", sizes.tolist(), "refs", int(is_ref.sum()))

# ---- 2-5. per frame
by_frame = {}
for i in range(len(D)):
    by_frame.setdefault(int(D[i, 0]), []).append(i)

per = []
for fi, fm in enumerate(frames):
    rec = dict(t=fm["t"], settled=False)
    g = fm.get("goal")
    rows = by_frame.get(fi, [])
    if g is None or not rows:
        per.append(rec); continue
    gx, gy = (g[0] + g[2]) / 2.0, float(g[3])
    ang = np.radians(fm["gline_angle"] if fm.get("gline_angle") is not None else 90.0)
    cot = np.cos(ang) / np.sin(ang) if abs(np.sin(ang)) > 1e-3 else 0.0
    # crowd filter: benches and sideline crowds stand shoulder to shoulder in a row; players don't
    rr = [i for i in rows if turf[i]]
    crowd = set()
    if len(rr) >= 6:
        fy = np.array([D[i, 6] for i in rr]); fx = np.array([(D[i, 3] + D[i, 5]) / 2 for i in rr])
        hh = np.array([D[i, 6] - D[i, 4] for i in rr])
        for j in range(len(rr)):
            near = (np.abs(fy - fy[j]) < 0.3 * hh[j]) & (np.abs(fx - fx[j]) < 2.5 * hh[j])
            if near.sum() >= 6:
                crowd.update(np.array(rr)[near].tolist())
    pts = []
    others = 0
    for i in rows:
        if i in crowd or not turf[i]:
            continue
        if team[i] not in (0, 1):
            if team[i] == -1:
                x1, y1, x2, y2 = D[i, 3:7]
                h = max(y2 - y1, 1.0)
                if abs((x1 + x2) / 2 - gx) / h * YD_PER_HEIGHT < ZONE_YD and abs(y2 - gy) / h * YD_PER_HEIGHT < 30:
                    others += 1
            continue
        x1, y1, x2, y2 = D[i, 3:7]
        h = max(y2 - y1, 1.0)
        fx, fy = (x1 + x2) / 2.0, y2
        xline = gx + (fy - gy) * cot
        u = (fx - xline) / h * YD_PER_HEIGHT
        lat = (fy - gy) / h * YD_PER_HEIGHT
        pts.append((int(team[i]), u, lat, i))
    if len(pts) < 8:
        per.append(rec); continue
    side = np.sign(np.median([p[1] for p in pts])) or 1.0   # midfield is where most players are
    pts = [(tm, u * side, lat, i) for tm, u, lat, i in pts]
    zone = [p for p in pts if -8 <= p[1] <= ZONE_YD and abs(p[2]) <= 30]
    nA = sum(p[0] == 0 for p in zone); nB = sum(p[0] == 1 for p in zone)
    # a real settled possession: both teams there, and almost nobody near the goal in a third color
    # (halftime youth groups, warmups, crowds on the field)
    if nA < 4 or nB < 4 or others > 3 or others > 0.3 * (nA + nB):
        per.append(rec); continue
    mu = {tm: float(np.mean([p[1] for p in zone if p[0] == tm])) for tm in (0, 1)}
    off = 0 if mu[0] > mu[1] else 1
    rec.update(settled=True, offense=off, off_margin=abs(mu[0] - mu[1]),
               zone=[[p[0], round(p[1], 1), round(p[2], 1), p[3]] for p in zone])
    per.append(rec)


def call_set(us):
    top = sum(u >= TOP_YD for u in us); mid = sum(MID_YD <= u < TOP_YD for u in us)
    behind = sum(u < LOW_YD for u in us); front = len(us) - top - mid - behind
    c = np.array([top, mid, front, behind], float)
    if len(us) < 5:
        return "Other", 0.0, c.tolist()
    if len(us) != 6:
        c = c * 6.0 / len(us)
    dist = sorted((np.abs(c - np.array(v)).sum(), k) for k, v in SETS.items())
    (d1, k1), (d2, _) = dist[0], dist[1]
    if d1 > 3.0:
        return "Other", 0.0, c.round(1).tolist()
    conf = float(np.clip((d2 - d1) / 2.0, 0, 1) * np.exp(-d1 / 2.0))
    return set_name(k1), round(conf, 2), c.round(1).tolist()


# ---- moments: runs of settled frames (gaps up to 1.5 s), at least 4 s long
moments, cur = [], []
for k, r in enumerate(per):
    if not r["settled"]:
        continue
    if cur and r["t"] - per[cur[-1]]["t"] > 3.1:   # allow short dropouts (a missed goal or player)
        if per[cur[-1]]["t"] - per[cur[0]]["t"] >= 4.0 and len(cur) >= 5:
            moments.append(cur)
        cur = []
    cur.append(k)
if cur and per[cur[-1]]["t"] - per[cur[0]]["t"] >= 4.0 and len(cur) >= 5:
    moments.append(cur)

out = []
for m in moments:
    offs = [per[k]["offense"] for k in m]
    off = int(np.bincount(offs, minlength=2).argmax())
    votes = {}
    calls = []
    for k in m:
        us = [p[1] for p in per[k]["zone"] if p[0] == off]
        s, c, counts = call_set(us)
        calls.append(dict(t=per[k]["t"], set=s, conf=c, counts=counts, n_off=len(us)))
        votes[s] = votes.get(s, 0) + c + 0.05
    best = max(votes, key=votes.get)
    share = votes[best] / sum(votes.values())
    mid_k = m[len(m) // 2]
    out.append(dict(start=per[m[0]]["t"], end=per[m[-1]]["t"], mid=per[mid_k]["t"],
                    offense=team_colors[off], offense_team=off, set=best, set_goal_first=flip(best),
                    set_display=best if NAMING == "top-first" else flip(best), confidence=round(share, 2),
                    offense_side_agreement=round(offs.count(off) / len(offs), 2), samples=calls))
os.makedirs(a.out, exist_ok=True)
json.dump(dict(team_colors=team_colors, stored_naming="top-first", display_naming=NAMING,
               set_library=sorted({set_name(k) for k in SETS}) + ["Other"],
               bands_yards=dict(top=f">= {TOP_YD}", middle=f"{MID_YD}-{TOP_YD}", front=f"{LOW_YD}-{MID_YD}", behind=f"< {LOW_YD}"),
               frames=len(frames), goal_seen=sum(f.get("goal") is not None for f in frames), moments=out),
          open(os.path.join(a.out, "moments.json"), "w"), indent=1)
# per-detection team labels, for drawing
np.savez_compressed(os.path.join(a.out, "teams.npz"), frame=D[:, 0].astype(np.int32), t=D[:, 1],
                    box=D[:, 3:7].astype(np.int16), team=team.astype(np.int8))
print(f"goal seen in {sum(f.get('goal') is not None for f in frames)}/{len(frames)} frames; "
      f"{sum(r['settled'] for r in per)} settled frames; {len(out)} settled-offense moments")
from collections import Counter
print(Counter(m["set"] for m in out))
for m in out[:40]:
    print(f"{m['start']:7.1f}-{m['end']:7.1f}  offense {m['offense']:<12} {m['set_display']:<6} conf {m['confidence']}")
