"""Engine pass 2 (light, runs anywhere): whole game -> field map, radar, events, player data, sets.

Reads the time-ordered chunks from engine pass 1 (results/<game>-engine-raw/chunk_NNN) and writes
results/<game>-engine/:
  field.json        men's lacrosse field (yards) for drawing, plus the coordinate convention
  radar/chunk_NNN.json  every frame: [player, x, y, team, role, seen, quality] (always on, flagged by quality)
  possessions.json  settled possessions: end, offense team, start/end, set call + timeline
  events.json       faceoffs (+ which team came out with it), goals, man-up/man-down, clears/rides
  sets.json         set call per settled possession (top-first; Skip when not settled)
  players.json      per player (stitched track): team, role, minutes seen, distance, sprints, top speed, heatmap
  quality.json      field-mapping coverage and error checks, timing
  summary.md        plain-language summary

State is carried across chunks in time order: track ids are stitched through the chunk overlaps, team colors
are learned once for the game (SigLIP -> UMAP -> KMeans, the roboflow/sports team-classifier recipe, with a
torso-color fallback), and each player's last field position is remembered for off-screen gaps.

Coordinates: x = yards along the field from the left end line (0..110), y = yards from the near sideline
(0 near .. 53.3 far, football width on shared turf). Goals: x = 15 (left), x = 95 (right), y = 26.67.
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import field_geom as fg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("--raw", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--fit-only", default="", help="compute and cache the field fit for these chunk numbers (e.g. 000,001) and stop")
ap.add_argument("--colors", default="", help="optional team names typed by the coach, e.g. white,blue")
a = ap.parse_args()
RAW = a.raw or f"results/{a.game}-engine-raw"
OUT = a.out or f"results/{a.game}-engine"
os.makedirs(os.path.join(OUT, "radar"), exist_ok=True)
T0 = time.time()

FIELD_L, FIELD_W = 110.0, 160 / 3.0
GOALS = {"L": 15.0, "R": 95.0}
GOAL_Y = FIELD_W / 2
MID = 55.0
RESTRAIN = 20.0           # restraining line: 20 yd from the goal line toward midfield

# ------------------------------------------------------------------ load chunks
chunks = []
for d in sorted(glob.glob(os.path.join(RAW, "chunk_*"))):
    meta = json.load(open(os.path.join(d, "frames.json")))
    z = np.load(os.path.join(d, "dets.npz"))
    dets = z["dets"]
    byf = {}
    if len(dets):
        o_ = np.argsort(dets[:, 0], kind="stable"); dets = dets[o_]
        fu, st_ = np.unique(dets[:, 0].astype(int), return_index=True)
        for k_, (f_, a_) in enumerate(zip(fu, st_)):
            b_ = st_[k_ + 1] if k_ + 1 < len(st_) else len(dets)
            byf[int(f_)] = dets[a_:b_]
    chunks.append(dict(dir=d, meta=meta, dets=dets, byf=byf, emb_ids=z["emb_ids"], emb=z["emb"].astype(np.float32)))
if not chunks:
    sys.exit("no chunks")
LAX = {int(k): v for k, v in chunks[0]["meta"]["lax_names"].items()}
CLS = {v.lower(): k for k, v in LAX.items()}
FPS = chunks[0]["meta"]["fps"]


EMPTY = np.zeros((0, 27), np.float32)


def _f(C, i):
    R = C["byf"].get(int(i), EMPTY)
    return R[R[:, 8] > 0]


def apply(Hm, x, y):
    q = Hm @ np.array([x, y, 1.0])
    return q[:2] / q[2]


# ------------------------------------------------------------------ 1. field mapping per chunk (cached: slow part)
def fit_chunk(C):
    fr = C["meta"]["frames"]; n = len(fr)
    keys = {int(k): v for k, v in C["meta"]["keys"].items()}
    # camera chains: a new 'shot' starts where frame-to-frame motion can't be measured (cut, replay, graphic)
    Hc, shot = [np.eye(3)], [0]
    for i in range(1, n):
        if fr[i]["inl"] < 25:
            Hc.append(np.eye(3)); shot.append(shot[-1] + 1)
        else:
            Hc.append(Hc[-1] @ np.array(fr[i]["H"])); shot.append(shot[-1])
    shot = np.array(shot)
    # goals on each shot's canvas (a pan can show both goals: cluster)
    goals_by_shot = {}
    for i, f in enumerate(fr):
        gb = f.get("goal")
        if gb and f["ph"] > 0 and not ((gb[2] - gb[0]) < 3.5 * f["ph"] and 0.5 * f["ph"] < (gb[3] - gb[1]) < 2.0 * f["ph"]):
            f["goal"] = None
        if f.get("goal"):
            gx = (f["goal"][0] + f["goal"][2]) / 2
            goals_by_shot.setdefault(shot[i], []).append((i, apply(Hc[i], gx, f["goal"][3]), f["goal"]))
    gcent = {}
    for s, lst in goals_by_shot.items():
        P = np.array([p for _, p, _ in lst])
        cl = []
        for p in P:
            for c in cl:
                if np.linalg.norm(c[0] / c[1] - p) < 400:
                    c[0] += p; c[1] += 1; break
            else:
                cl.append([p.copy(), 1])
        gcent[s] = [c[0] / c[1] for c in cl if c[1] >= 3]
    good, fails = {}, {}
    for i, kd in keys.items():
        cands = gcent.get(shot[i], [])
        if not cands:
            fails["no goal in this camera shot"] = fails.get("no goal in this camera shot", 0) + 1
            continue
        Hinv = np.linalg.inv(Hc[i])
        # the goal of the shot nearest to the picture center
        W, Hh = C["meta"]["width"], C["meta"]["height"]
        gi = min((apply(Hinv, *c) for c in cands), key=lambda q: abs(q[0] - W / 2))
        Hk, info = fg.keyframe_h(kd["segs"], kd["side"], gi)
        if Hk is not None:
            f = fr[i]
            if f.get("goal"):
                x1, _, x2, y2 = f["goal"]
                gwk = float(np.linalg.norm(apply(Hk, x1, y2) - apply(Hk, x2, y2)))
                if not 0.7 <= gwk <= 4.5:
                    Hk, info = None, "goal mouth maps to an impossible width"
        if Hk is not None:
            R = _f(C, i)
            if len(R):
                Q = np.array([apply(Hk, (r[3] + r[5]) / 2, r[6]) for r in R])
                inside = ((Q[:, 1] > -4) & (Q[:, 1] < 58) & (np.abs(Q[:, 0]) < 70)).mean()
                if inside < 0.8:
                    Hk, info = None, "players land off the field"
        if Hk is None:
            fails[info] = fails.get(info, 0) + 1
        else:
            good[i] = (Hk, info)
    # each frame borrows the nearest good keyframe in the same shot (within 4 s)
    Hf = [None] * n
    Qf = [0.0] * n
    K = sorted(good)
    Karr = np.array(K) if K else np.zeros(0, int)
    for i in range(n):
        if not len(Karr):
            break
        cand = Karr[(shot[Karr] == shot[i]) & (np.abs(Karr - i) <= 10 * FPS)]
        if not len(cand):
            continue
        k = int(cand[np.argmin(np.abs(cand - i))])
        Hm = good[k][0] @ np.linalg.inv(Hc[k]) @ Hc[i]
        Hf[i] = Hm / Hm[2, 2]
        Qf[i] = 1.0 if abs(k - i) <= 3 * FPS else 0.7
    # checks: goal mouth width (posts 6 ft = 2 yd apart) mapped through the frame's homography;
    # neighbouring keyframes (1 s apart, same shot) agreeing on where a player's feet are
    gw = []
    for i, f in enumerate(fr):
        if Hf[i] is not None and f.get("goal"):
            x1, _, x2, y2 = f["goal"]
            p1 = apply(Hf[i], x1, y2); p2 = apply(Hf[i], x2, y2)
            gw.append(float(np.linalg.norm(p1 - p2)))
    dis = []
    D = C["dets"]
    for k1, k2 in zip(K, K[1:]):
        if k2 - k1 > 2 * FPS or shot[k1] != shot[k2]:
            continue
        rows = _f(C, k2)
        if not len(rows):
            continue
        Ha = good[k1][0] @ np.linalg.inv(Hc[k1]) @ Hc[k2]
        for r in rows[:12]:
            fx, fy = (r[3] + r[5]) / 2, r[6]
            dis.append(float(np.linalg.norm(apply(Ha, fx, fy) - apply(good[k2][0], fx, fy))))
    goals_clean = [f.get("goal") for f in fr]
    return dict(qinfo=dict(chunk=os.path.basename(C["dir"]), frames=n, keyframes=len(keys), keyframes_ok=len(good),
                                  frames_mapped=int(sum(h is not None for h in Hf)), fails=fails,
                                  goal_width_yd_median=round(float(np.median(gw)), 2) if gw else None,
                                  neighbour_disagreement_yd_median=round(float(np.median(dis)), 2) if dis else None,
                                  seconds_compute=C["meta"].get("seconds_total")),
                Hf=[None if h is None else np.round(h, 9).tolist() for h in Hf], Qf=Qf, shot=shot.tolist(), goodK=[int(k) for k in K],
                goals=goals_clean)



FIT = os.path.join(OUT, "fieldfit")
os.makedirs(FIT, exist_ok=True)
if a.fit_only:
    for nm in a.fit_only.split(","):
        C = next(c for c in chunks if os.path.basename(c["dir"]).endswith(nm))
        json.dump(fit_chunk(C), open(os.path.join(FIT, os.path.basename(C["dir"]) + ".json"), "w"))
        print("fitted", nm, f"{time.time() - T0:.0f}s", flush=True)
    sys.exit(0)
quality = dict(chunks=[])
for C in chunks:
    fp = os.path.join(FIT, os.path.basename(C["dir"]) + ".json")
    R_ = json.load(open(fp)) if os.path.exists(fp) else fit_chunk(C)
    C["Hf"] = [None if h is None else np.array(h) for h in R_["Hf"]]
    C["Qf"], C["shot"], C["goodK"] = R_["Qf"], np.array(R_["shot"]), R_["goodK"]
    for f, gb in zip(C["meta"]["frames"], R_["goals"]):
        f["goal"] = gb
    quality["chunks"].append(R_["qinfo"])
print(f"field mapping done {time.time() - T0:.0f}s", flush=True)
# ------------------------------------------------------------------ 2. stitch tracks across chunk overlaps
parent = {}


def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[max(rx, ry)] = min(rx, ry)


for ci in range(1, len(chunks)):
    A, B = chunks[ci - 1], chunks[ci]
    ta = np.array([f["t"] for f in A["meta"]["frames"]]); tb = np.array([f["t"] for f in B["meta"]["frames"]])
    ov = tb[tb <= ta.max() + 1e-3]
    for t in ov[1:]:      # skip the very first frame of B: its tracker has no history yet
        ia = int(np.argmin(np.abs(ta - t))); ib = int(np.argmin(np.abs(tb - t)))
        if abs(ta[ia] - t) > 0.2:
            continue
        ra = A["byf"].get(ia, EMPTY); rb = B["byf"].get(ib, EMPTY)
        for r in rb:
            if r[2] < 0 or not len(ra):
                continue
            ix1 = np.maximum(r[3], ra[:, 3]); iy1 = np.maximum(r[4], ra[:, 4]); ix2 = np.minimum(r[5], ra[:, 5]); iy2 = np.minimum(r[6], ra[:, 6])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            uni = (r[5] - r[3]) * (r[6] - r[4]) + (ra[:, 5] - ra[:, 3]) * (ra[:, 6] - ra[:, 4]) - inter
            iou = inter / np.maximum(uni, 1); j = int(np.argmax(iou))
            if iou[j] > 0.6 and ra[j, 2] >= 0:
                union((ci - 1, int(ra[j, 2])), (ci, int(r[2])))

# ------------------------------------------------------------------ 2b. goal-anchored fallback (learned from fitted frames)
# Where no full field fit exists but the goal is in view, place players relative to the goal: offsets measured in
# each player's own height (about 2 yd), scaled by factors learned from the frames that do have a full fit. Low camera
# angles squash depth, which the learned depth factor absorbs.
def goal_feats(r, gb):
    h = max(r[6] - r[4], 1.0)
    gx, gy = (gb[0] + gb[2]) / 2, gb[3]
    return ((r[3] + r[5]) / 2 - gx) / h * 2.0, (r[6] - gy) / h * 2.0


fx_l, fy_l, X_l, Y_l = [], [], [], []
for C in chunks:
    fr = C["meta"]["frames"]
    for i in C["goodK"]:
        gb = fr[i].get("goal")
        if not gb:
            continue
        for r in _f(C, i):
            Xg, Yg = apply(C["Hf"][i], (r[3] + r[5]) / 2, r[6])
            if abs(Xg) < 40 and -5 < Yg < 58:
                f1, f2 = goal_feats(r, gb)
                fx_l.append(f1); fy_l.append(f2); X_l.append(Xg); Y_l.append(Yg - fg.GOAL_Y)
GA = None
if len(fx_l) >= 30:
    fx_a, fy_a, X_a, Y_a = map(np.array, (fx_l, fy_l, X_l, Y_l))
    ax = float(np.dot(fx_a, X_a) / np.dot(fx_a, fx_a)); ay = float(np.dot(fy_a, Y_a) / np.dot(fy_a, fy_a))
    rx = float(np.median(np.abs(ax * fx_a - X_a))); ry = float(np.median(np.abs(ay * fy_a - Y_a)))
    GA = (ax, ay)
    quality["goal_anchored_model"] = dict(along=round(ax, 2), depth=round(ay, 2), samples=len(fx_l),
                                          median_error_yd=[round(rx, 1), round(ry, 1)])
else:
    quality["goal_anchored_model"] = dict(samples=len(fx_l), note="too few fitted frames with a goal; default factors used")
    GA = (1.0, -3.0)

# ------------------------------------------------------------------ 3. one table of observations for the game
obs = []   # t, gid, x, y, mapped, role, stripes, turf, ph, box h, chunk, frame
gid_of = {}
feats = {}
for ci, C in enumerate(chunks):
    fr = C["meta"]["frames"]; t_hi = chunks[ci + 1]["meta"]["t_start"] if ci + 1 < len(chunks) else 1e9
    D = C["dets"]
    for r in D:
        fi = int(r[0]); t = fr[fi]["t"]
        if ci + 1 < len(chunks) and t >= t_hi:     # the overlap belongs to the next chunk
            continue
        if r[2] < 0:
            continue
        key = find((ci, int(r[2])))
        gid = gid_of.setdefault(key, len(gid_of))
        Hm = C["Hf"][fi]
        gb = fr[fi].get("goal")
        q = 0.0
        if Hm is not None and r[8] > 0:
            X, Y = apply(Hm, (r[3] + r[5]) / 2, r[6]); mapped = 1; q = C["Qf"][fi]
        elif gb and r[8] > 0:                      # goal-anchored estimate
            f1, f2 = goal_feats(r, gb)
            X, Y = GA[0] * f1, fg.GOAL_Y + GA[1] * f2; mapped = 1; q = 0.4
        else:
            X, Y, mapped = np.nan, np.nan, 0
        obs.append([t, gid, X, Y, mapped, r[10], r[9], r[8], fr[fi]["ph"], r[6] - r[4], ci, fi, q])
        feats.setdefault(gid, []).append(r[11:27])
O = np.array(obs, np.float64)
NG = len(gid_of)
print(f"{len(O)} observations, {NG} stitched players, {time.time() - T0:.0f}s", flush=True)

# ------------------------------------------------------------------ 4. which goal / field direction per frame
# X from keyframe_h = yards from the visible goal line along the sideline direction; interior side is where most
# mapped players stand. Interior to the image-right -> the left-end goal.
fkey = O[:, 10] * 100000 + O[:, 11]
absx = np.full(len(O), np.nan)
absy = O[:, 3].copy()
side_of_frame = {}
_fo = np.argsort(fkey, kind="stable")
uk, _fs = np.unique(fkey[_fo], return_index=True)
for j, k in enumerate(uk):
    idx = _fo[_fs[j]:(_fs[j + 1] if j + 1 < len(_fs) else len(_fo))]
    xs = O[idx, 2][O[idx, 4] > 0]
    if len(xs) < 3:
        continue
    s = 1.0 if np.median(xs) > 0 else -1.0
    end = "L" if s > 0 else "R"
    side_of_frame[k] = end
    absx[idx] = GOALS[end] + O[idx, 2]
ok = np.isfinite(absx) & (absx > -5) & (absx < FIELD_L + 5) & (absy > -5) & (absy < FIELD_W + 5)
absx[~ok] = np.nan; absy[~ok] = np.nan
quality["players_mapped_share"] = round(float(np.isfinite(absx).mean()), 3)
quality["mapped_inside_field_share"] = round(float(ok.sum() / max(1, (O[:, 4] > 0).sum())), 3)

# ------------------------------------------------------------------ 5. roles, refs, teams
role_names = {CLS.get("goalie"): "goalie", CLS.get("longpole"): "longpole", CLS.get("referee"): "ref", CLS.get("shortstick"): "shortstick"}
_go = np.argsort(O[:, 1], kind="stable")
_gu, _gs = np.unique(O[_go, 1].astype(int), return_index=True)
GIDX = {int(g_): _go[a_:(_gs[k_ + 1] if k_ + 1 < len(_gs) else len(_go))] for k_, (g_, a_) in enumerate(zip(_gu, _gs))}
P = {}
for g in range(NG):
    m = GIDX.get(g, np.zeros(0, int))
    rr = O[m]
    cls = rr[:, 5][rr[:, 5] >= 0].astype(int)
    votes = {role_names.get(c, "player"): int((cls == c).sum()) for c in np.unique(cls)} if len(cls) else {}
    P[g] = dict(n=int(len(m)), turf=float(rr[:, 7].mean()) if len(rr) else 0.0, stripes=float(np.median(rr[:, 6])), votes=votes,
                t0=float(rr[:, 0].min()) if len(rr) else 0.0, t1=float(rr[:, 0].max()) if len(rr) else 0.0)
strip_all = np.array([P[g]["stripes"] for g in range(NG)])
strip_cut = float(np.percentile(strip_all[strip_all > 0], 90)) if (strip_all > 0).any() else 1e9
for g in range(NG):
    v = P[g]["votes"]; tot = sum(v.values())
    role = "player"
    if tot >= 3 and v.get("goalie", 0) > 0.5 * tot:
        role = "goalie"
    elif (tot >= 3 and v.get("ref", 0) > 0.6 * tot) or (P[g]["stripes"] > strip_cut and v.get("ref", 0) > 0.3 * max(tot, 1)):
        role = "ref"
    elif tot >= 3 and v.get("longpole", 0) > 0.5 * tot:
        role = "longpole"
    elif tot >= 3:
        role = "shortstick"
    P[g]["role"] = role
    P[g]["field"] = P[g]["turf"] >= 0.5 and P[g]["n"] >= 6

# team split: SigLIP embeddings per stitched player -> UMAP(3) -> KMeans(2); torso color fallback
emb = {}
for ci, C in enumerate(chunks):
    for tid, e in zip(C["emb_ids"], C["emb"]):
        key = find((ci, int(tid)))
        if key in gid_of:
            emb.setdefault(gid_of[key], []).append(e)
cand = [g for g in range(NG) if P[g]["field"] and P[g]["role"] in ("player", "shortstick", "longpole")]
team_method = "torso color"
from sklearn.cluster import KMeans  # noqa: E402
X_e = [g for g in cand if g in emb]
labels = {}
if len(X_e) >= 40:
    try:
        import umap  # noqa: E402
        E = np.stack([np.mean(emb[g], 0) for g in X_e])
        R = umap.UMAP(n_components=3, random_state=0).fit_transform(E)
        lab = KMeans(2, n_init=10, random_state=0).fit_predict(R)
        labels = {g: int(l) for g, l in zip(X_e, lab)}
        team_method = "SigLIP + UMAP + KMeans"
    except Exception as e:
        print("umap/siglip split failed:", e)
torso = {g: np.mean(feats[g], 0) for g in range(NG) if g in feats}
if not labels:
    Xt = np.stack([torso[g] for g in cand])
    lab = KMeans(2, n_init=10, random_state=0).fit_predict(Xt)
    labels = {g: int(l) for g, l in zip(cand, lab)}
# remaining field players (no embedding): nearest team centroid in torso color
cent = [np.mean([torso[g] for g in labels if labels[g] == k], 0) for k in (0, 1)]
for g in cand:
    if g not in labels:
        labels[g] = int(np.argmin([np.linalg.norm(torso[g] - c) for c in cent]))


def color_name(f):
    hue = f[:8] / 255.0; sat = f[8:12] / 255.0; val = f[12:16] / 255.0
    s_mean = float(np.dot(sat, [32, 96, 160, 224]) / max(sat.sum(), 1e-6))
    v_mean = float(np.dot(val, [32, 96, 160, 224]) / max(val.sum(), 1e-6))
    if v_mean < 70:
        return "black"
    if s_mean < 70 and v_mean > 140:
        return "white"
    hb = int(np.argmax(hue)) * 22.5 + 11
    for lo, hi, nm in [(0, 10, "red"), (10, 22, "orange"), (22, 35, "gold"), (35, 85, "green"), (85, 130, "blue"),
                       (130, 160, "purple"), (160, 181, "red")]:
        if lo <= hb < hi:
            return nm
    return "dark"


tf_ = [np.mean([torso[g] for g in labels if labels[g] == k], 0) for k in (0, 1)]
team_color = [color_name(f) for f in tf_]
if team_color[0] == team_color[1]:
    vm = [float(np.dot(f[12:16] / 255.0, [32, 96, 160, 224]) / max((f[12:16] / 255.0).sum(), 1e-6)) for f in tf_]
    dk = int(np.argmin(vm))
    hb = int(np.argmax(tf_[dk][:8])) * 22.5 + 11
    team_color[dk] = "blue" if 85 <= hb < 130 else ("red" if hb < 10 or hb >= 160 else "dark")
    team_color[1 - dk] = "white" if team_color[1 - dk] in ("white", "dark") else team_color[1 - dk]
if team_color[0] == team_color[1]:
    team_color = [team_color[0] + " (A)", team_color[1] + " (B)"]
if a.colors:                       # names typed by the coach (Settings): the lighter jersey gets the light name
    nm = [c.strip().lower() for c in a.colors.split(",")]
    vm = [float(np.dot(f[12:16] / 255.0, [32, 96, 160, 224]) / max((f[12:16] / 255.0).sum(), 1e-6)) for f in tf_]
    light = int(np.argmax(vm))
    lname = next((c for c in nm if c in ("white", "silver", "gray", "grey", "yellow", "gold")), nm[1])
    team_color[light] = lname
    team_color[1 - light] = next(c for c in nm if c != lname)
for g in range(NG):
    P[g]["team"] = labels.get(g, -1)
team_of = np.array([P[int(g)]["team"] for g in O[:, 1]])
role_of = np.array([P[int(g)]["role"] for g in O[:, 1]])
quality["team_split"] = dict(method=team_method, colors=team_color, players=len(labels))

# ------------------------------------------------------------------ 6. per-frame state on a common clock
all_ft = []
for ci, C in enumerate(chunks):
    t_hi = chunks[ci + 1]["meta"]["t_start"] if ci + 1 < len(chunks) else 1e9
    all_ft += [f["t"] for f in C["meta"]["frames"] if f["t"] < t_hi]
times = np.unique(np.round(np.array(all_ft), 2))
frames = {}
order = np.argsort(O[:, 0], kind="stable")
for i in order:
    frames.setdefault(round(O[i, 0], 2), []).append(i)
last_pos = {}          # off-screen memory: gid -> (t, x, y)
state = []             # per time: t, end in play (L/R/None), counts
radar_rows = []
for t in times:
    idx = frames.get(round(t, 2), [])
    vis = [i for i in idx if np.isfinite(absx[i]) and role_of[i] != "ref" and P[int(O[i, 1])]["field"]]
    seen = set()
    rows = []
    for i in vis:
        g = int(O[i, 1]); seen.add(g)
        last_pos[g] = (t, absx[i], absy[i])
        rows.append([g, round(float(absx[i]), 1), round(float(absy[i]), 1), int(team_of[i]), P[g]["role"], 1, round(float(O[i, 12]), 2)])
    xs = [r[1] for r in rows]
    end = None
    if len(xs) >= 4:
        mx = float(np.median(xs))
        if mx < GOALS["L"] + RESTRAIN:
            end = "L"
        elif mx > GOALS["R"] - RESTRAIN:
            end = "R"
    # remembered players: last seen within 6 s inside the same zone and the play still there
    if end:
        for g, (tl, x, y) in list(last_pos.items()):
            if g in seen or t - tl > 6:
                continue
            if (end == "L" and x < GOALS["L"] + RESTRAIN + 3) or (end == "R" and x > GOALS["R"] - RESTRAIN - 3):
                rows.append([g, round(float(x), 1), round(float(y), 1), P[g]["team"], P[g]["role"], 0, 0.3])
    elif not rows:      # nothing mapped this frame (camera on the crowd, a cut, a replay): keep the last picture, flagged
        for g, (tl, x, y) in list(last_pos.items()):
            if t - tl <= 10:
                rows.append([g, round(float(x), 1), round(float(y), 1), P[g]["team"], P[g]["role"], 0, 0.1])
    fq = round(float(np.mean([r[6] for r in rows])), 2) if rows else 0.0
    state.append(dict(t=float(t), end=end, rows=rows, q=fq))
print(f"state built {time.time() - T0:.0f}s", flush=True)

# radar chunks (one file per engine chunk, time-ordered)
bounds = [C["meta"]["t_start"] for C in chunks[1:]] + [1e9]
for ci in range(len(chunks)):
    lo = chunks[ci]["meta"]["t_start"] if ci else -1
    sel = [s for s in state if lo <= s["t"] < bounds[ci]]
    json.dump(dict(fps=FPS, teams=team_color, columns=["player", "x", "y", "team", "role", "seen", "quality"],
                   quality_note="quality per row: 1.0 full field fit near a calibrated keyframe, 0.7 fit borrowed up to 10 s away, "
                                "0.4 goal-anchored estimate, 0.3 remembered off-screen spot, 0.1 last known picture (nothing mapped now)",
                   frames=[dict(t=round(s["t"], 2), q=s["q"], p=s["rows"]) for s in sel]),
              open(os.path.join(OUT, "radar", f"chunk_{ci:03d}.json"), "w"), separators=(",", ":"))

# ------------------------------------------------------------------ 7. possessions (settled play at one end)
segs = []
cur = None
for s in state:
    e = s["end"]
    if cur and (e != cur["end"]) and not (e is None and s["t"] - cur["t1"] < 4):
        segs.append(cur); cur = None
    if e and not cur:
        cur = dict(end=e, t0=s["t"], t1=s["t"], st=[])
    if cur and e:
        cur["t1"] = s["t"]; cur["st"].append(s)
if cur:
    segs.append(cur)
segs = [sg for sg in segs if sg["t1"] - sg["t0"] >= 3]
gx_ = lambda e: GOALS[e]
poss = []
for sg in segs:
    d = {0: [], 1: []}
    for s in sg["st"]:
        for g, x, y, tm, rl, seen, q_ in s["rows"]:
            if tm in (0, 1) and rl != "goalie" and seen:
                d[tm].append(abs(x - gx_(sg["end"])) + 0.0)
    if not d[0] or not d[1]:
        off = None
    else:
        off = 0 if np.median(d[0]) > np.median(d[1]) else 1      # defense sits between the offense and the goal
    poss.append(dict(end=sg["end"], start=round(sg["t0"], 1), end_t=round(sg["t1"], 1), offense=off, st=sg["st"],
                     field_quality=round(float(np.mean([s["q"] for s in sg["st"]])), 2)))

# ------------------------------------------------------------------ 8. faceoffs (image pattern, no field map needed)
fo_score = []
for ci, C in enumerate(chunks):
    D = C["dets"]; fr = C["meta"]["frames"]
    t_hi = bounds[ci]
    for fi, f in enumerate(fr):
        if f["t"] >= t_hi:
            continue
        R = _f(C, fi)
        if len(R) < 6 or f["ph"] <= 0:
            continue
        hts = R[:, 6] - R[:, 4]
        med = np.median(hts)
        low = R[hts < 0.72 * med]
        hit = 0
        for i in range(len(low)):
            for j in range(i + 1, len(low)):
                b1, b2 = low[i], low[j]
                gap = max(b1[3], b2[3]) - min(b1[5], b2[5])
                w = max(b1[5] - b1[3], b2[5] - b2[3])
                if gap < 0.5 * w and abs(b1[6] - b2[6]) < 0.35 * med:
                    t1 = P.get(gid_of.get(find((ci, int(b1[2]))), -1), {}).get("team", -1) if b1[2] >= 0 else -1
                    t2 = P.get(gid_of.get(find((ci, int(b2[2]))), -1), {}).get("team", -1) if b2[2] >= 0 else -1
                    if t1 != t2 or -1 in (t1, t2):
                        # a referee close by, standing
                        others = R[(hts >= 0.85 * med)]
                        cx = (b1[3] + b2[5]) / 2
                        near_ref = any(abs((o[3] + o[5]) / 2 - cx) < 3 * med and o[9] > strip_cut * 0.7 for o in others)
                        hit = max(hit, 2 if near_ref else 1)
        if hit:
            fo_score.append((f["t"], hit))
fos = []
for t, h in fo_score:
    if fos and t - fos[-1]["t1"] <= 3:
        fos[-1]["t1"] = t; fos[-1]["n"] += 1; fos[-1]["ref"] = max(fos[-1]["ref"], h)
    else:
        fos.append(dict(t0=t, t1=t, n=1, ref=h))
fos = [f for f in fos if f["n"] >= 2 and f["ref"] == 2]
merged = []
for f in fos:
    if merged and f["t0"] - merged[-1]["t1"] < 20:
        merged[-1]["t1"] = f["t1"]
    else:
        merged.append(f)
fos = merged

# breaks (between periods): long stretches with almost nobody on the field
activity = np.array([[s["t"], sum(1 for r in s["rows"] if r[5])] for s in state]) if state else np.zeros((0, 2))
all_t = np.array(sorted(set(np.round(O[:, 0], 2))))
cnt_t = np.array([len(frames.get(round(t, 2), [])) for t in all_t])


def was_break_before(t, span=60):
    m = (all_t > t - span - 30) & (all_t < t - 5)
    if not m.any():
        return True
    poss_before = [p for p in poss if t - 90 < p["end_t"] < t]
    return len(poss_before) == 0


events = []
goals = []
for f in fos:
    t = f["t0"]
    nxt = next((p for p in poss if p["start"] > f["t1"] and p["start"] - f["t1"] < 45 and p["offense"] is not None), None)
    prv = next((p for p in reversed(poss) if p["end_t"] < t and p["offense"] is not None), None)
    period_start = was_break_before(t)
    ev = dict(type="faceoff", t=round(t, 1), won_by=team_color[nxt["offense"]] if nxt else None,
              confidence="medium" if nxt else "low", after="period start" if period_start else "goal")
    events.append(ev)
    if not period_start:
        sc = team_color[prv["offense"]] if prv else None
        goals.append(dict(type="goal", t=round(t, 1), team=sc, how="faceoff followed", confidence="medium" if sc else "low"))
events += goals

# man-up / man-down: during a settled possession, the most defenders (not goalie) and attackers seen at once
for p in poss:
    if p["offense"] is None:
        continue
    o, dfn = p["offense"], 1 - p["offense"]
    mo = md = 0
    for s in p["st"]:
        co = sum(1 for r in s["rows"] if r[3] == o)
        cd = sum(1 for r in s["rows"] if r[3] == dfn and r[4] != "goalie")
        mo = max(mo, co); md = max(md, cd)
    p["max_offense_seen"], p["max_defense_seen"] = mo, md
    if mo >= 6 and md <= 5 and p["end_t"] - p["start"] >= 6:
        p["man_up"] = True
        events.append(dict(type="man-up", t=p["start"], t_end=p["end_t"], duration=round(p["end_t"] - p["start"], 1),
                           team=team_color[o], confidence="low",
                           note=f"saw {mo} attackers but never more than {md} defenders (goalie not counted)"))

# clears / rides between possessions
for p1, p2 in zip(poss, poss[1:]):
    if p1["offense"] is None or p2["offense"] is None:
        continue
    if any(p1["end_t"] <= f["t0"] <= p2["start"] for f in fos):
        continue
    if p2["end"] != p1["end"] and p2["offense"] != p1["offense"]:
        events.append(dict(type="clear", t=p1["end_t"], t_end=p2["start"], team=team_color[p2["offense"]], result="success",
                           ride_by=team_color[p1["offense"]], confidence="medium"))
    elif p2["end"] == p1["end"] and p2["offense"] == p1["offense"] and p2["start"] - p1["end_t"] < 30:
        events.append(dict(type="re-settle", t=p1["end_t"], t_end=p2["start"], team=team_color[p1["offense"]],
                           note="same offense settled again at the same end: a failed clear or an offensive rebound (no ball tracking yet)",
                           confidence="low"))
events.sort(key=lambda e: e["t"])

# ------------------------------------------------------------------ 9. sets per possession (template match + smoothing)
T = {   # top-first; points (dx toward midfield from the goal line, dy across), yards; goal at (0, 0)
    "2-3-1": [(15, -7), (15, 7), (7, -14), (3, 0), (7, 14), (-6, 0)],
    "1-4-1": [(16, 0), (7, -15), (4, -5), (4, 5), (7, 15), (-6, 0)],
    "1-3-2": [(16, 0), (7, -14), (3, 0), (7, 14), (-5, -6), (-5, 6)],
    "2-2-2": [(14, -7), (14, 7), (5, -9), (5, 9), (-5, -6), (-5, 6)],
    "3-3": [(14, -11), (15, 0), (14, 11), (3, -11), (3, 0), (3, 11)],
    "3-1-2": [(14, -11), (15, 0), (14, 11), (3, 0), (-5, -6), (-5, 6)],
    "Open": [(16, 0), (10, -13), (10, 13), (1, -14), (1, 14), (-7, 0)],
}
from scipy.optimize import linear_sum_assignment  # noqa: E402


def match_cost(pts, tpl):
    A = np.array(pts); B = np.array(tpl, float)
    Cm = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    r, c = linear_sum_assignment(Cm)
    return float(Cm[r, c].mean())


set_rows = []
for p in poss:
    p["set"], p["timeline"] = "Skip", []
    if p["offense"] is None:
        continue
    o = p["offense"]; gxx = gx_(p["end"])
    sgn = 1 if p["end"] == "L" else -1
    costs, used, prev = [], 0, None
    speeds = []
    for s in p["st"]:
        pts = [(sgn * (x - gxx), (y - GOAL_Y) * sgn) for g, x, y, tm, rl, sn, q_ in s["rows"] if tm == o]
        pts = [q for q in pts if -15 < q[0] < 30]
        if len(pts) < 5:
            continue
        pts = sorted(pts, key=lambda q: q[0])[:6]
        cs = {k: match_cost(pts, v) for k, v in T.items()}
        costs.append(cs); used += 1
        if prev is not None:
            speeds.append(np.median([np.hypot(a_[0] - b_[0], a_[1] - b_[1]) for a_, b_ in zip(sorted(pts), sorted(prev))]) * FPS)
        prev = pts
    settled = used >= 3 * FPS and (np.median(speeds) if speeds else 99) < 4.0
    if not settled:
        p["set"] = "Skip"
        continue
    avg = {k: float(np.mean([c[k] for c in costs])) for k in T}
    best = min(avg, key=avg.get)
    p["set"] = best if avg[best] < 6.5 else "Other"
    p["set_cost"] = round(avg[best], 2)
    # timeline: smoothed per-frame best over a 2 s window
    win = int(2 * FPS)
    lab_ = []
    for i in range(len(costs)):
        w = costs[max(0, i - win):i + win + 1]
        m = {k: np.mean([c[k] for c in w]) for k in T}
        lab_.append(min(m, key=m.get))
    st_t = [s["t"] for s in p["st"]]
    tl = []
    for i, l in enumerate(lab_):
        if tl and tl[-1]["set"] == l:
            tl[-1]["to"] = round(st_t[min(i, len(st_t) - 1)], 1)
        else:
            tl.append(dict(set=l, frm=round(st_t[min(i, len(st_t) - 1)], 1), to=round(st_t[min(i, len(st_t) - 1)], 1)))
    p["timeline"] = [x for x in tl if x["to"] - x["frm"] >= 1.5] or tl[:1]


def goal_first(sn):
    if sn and sn[0].isdigit():
        return "-".join(reversed(sn.split("-")))
    return sn


out_poss = []
for k, p in enumerate(poss):
    out_poss.append(dict(id=f"p{k + 1:03d}", start=p["start"], end=p["end_t"], end_of_field=p["end"],
                         offense=team_color[p["offense"]] if p["offense"] is not None else None,
                         set=p["set"], set_goal_first=goal_first(p["set"]), set_cost=p.get("set_cost"),
                         timeline=p["timeline"], man_up=bool(p.get("man_up")),
                         max_offense_seen=p.get("max_offense_seen"), max_defense_seen=p.get("max_defense_seen"),
                         field_quality=p["field_quality"],
                         confidence="high" if p["field_quality"] >= 0.8 else ("medium" if p["field_quality"] >= 0.5 else "low")))
json.dump(dict(possessions=out_poss), open(os.path.join(OUT, "possessions.json"), "w"), indent=0)
json.dump(dict(moments=out_poss), open(os.path.join(OUT, "sets.json"), "w"), indent=0)
json.dump(dict(teams=team_color, events=events), open(os.path.join(OUT, "events.json"), "w"), indent=0)

# ------------------------------------------------------------------ 10. per-player physical data
YD2M = 0.9144
players = []
heat_team = {0: np.zeros((6, 11)), 1: np.zeros((6, 11))}
attack_end = {}   # team -> list of (t, end attacked)
for p in poss:
    if p["offense"] is not None:
        attack_end.setdefault(p["offense"], []).append((p["start"], p["end"]))


def attacks_right(team, t):
    lst = attack_end.get(team, [])
    if not lst:
        return True
    near = min(lst, key=lambda z: abs(z[0] - t))
    return near[1] == "R"


for g in range(NG):
    if not P[g]["field"] or P[g]["role"] == "ref":
        continue
    m = GIDX.get(g, np.zeros(0, int)); m = m[np.isfinite(absx[m])]
    if len(m) < 6:
        continue
    t = O[m, 0]; x = absx[m]; y = absy[m]; qv = O[m, 12]
    o_ = np.argsort(t); t, x, y, qv = t[o_], x[o_], y[o_], qv[o_]
    good_share = float((qv >= 0.7).mean())
    # light smoothing (3-frame median) against box jitter
    if len(x) >= 3:
        x = np.array([np.median(x[max(0, i - 1):i + 2]) for i in range(len(x))])
        y = np.array([np.median(y[max(0, i - 1):i + 2]) for i in range(len(y))])
    dt = np.diff(t); dd = np.hypot(np.diff(x), np.diff(y))
    okm = (dt > 0) & (dt <= 1.5)
    sp = np.zeros_like(dd); sp[okm] = dd[okm] / dt[okm]
    okm &= sp < 11          # > 10 m/s is a mapping jump, not running
    dist = float(dd[okm].sum())
    top = float(np.percentile(sp[okm], 98)) if okm.sum() >= 5 else 0.0
    fast = (sp > 6.0) & okm
    sprints = int(np.sum(np.diff(np.concatenate([[0], fast.astype(int), [0]])) == 1))
    gm = qv >= 0.7
    hm = np.histogram2d(np.clip(y[gm], 0, FIELD_W - 0.01), np.clip(x[gm], 0, FIELD_L - 0.01), bins=[6, 11],
                        range=[[0, FIELD_W], [0, FIELD_L]])[0]
    hc = np.histogram2d(np.clip(y, 0, FIELD_W - 0.01), np.clip(x, 0, FIELD_L - 0.01), bins=[3, 3],
                        range=[[0, FIELD_W], [0, FIELD_L]])[0]
    tm = P[g]["team"]
    if tm in (0, 1):
        for xi, yi, ti in zip(x, y, t):
            xx = xi if attacks_right(tm, ti) else FIELD_L - xi
            yy = yi if attacks_right(tm, ti) else FIELD_W - yi
            heat_team[tm][min(5, max(0, int(yy / FIELD_W * 6))), min(10, max(0, int(xx / FIELD_L * 11)))] += 1
    players.append(dict(player=g, team=team_color[tm] if tm in (0, 1) else None, role=P[g]["role"],
                        first_seen=round(float(t[0]), 1), last_seen=round(float(t[-1]), 1),
                        seconds_mapped=round(float(len(t) / FPS), 1), distance_yd=round(dist, 1),
                        distance_m=round(dist * YD2M, 1), top_speed_mps=round(top * YD2M, 2), sprints=sprints,
                        field_quality=round(float(qv.mean()), 2), good_fit_share=round(good_share, 2),
                        physical_confidence="ok" if good_share >= 0.6 else "low (rough field fit; distances and speeds are estimates)",
                        heatmap_resolution="fine" if gm.sum() >= 30 else "coarse",
                        heatmap_fine=hm.astype(int).tolist(), heatmap_coarse=hc.astype(int).tolist()))
players.sort(key=lambda r: -r["seconds_mapped"])
json.dump(dict(note="Players are stitched tracks, not people yet: one athlete can appear as several tracks. heatmap_fine "
               "(good field fit only): rows = 6 bands from the near sideline, columns = 11 bands of 10 yd from the left end "
               "line. heatmap_coarse (every estimate): 3 lanes x 3 thirds. heatmap_resolution says which one to show. Team "
               "heatmaps are flipped so each team attacks to the right.",
               teams=team_color, team_heatmaps={team_color[k]: heat_team[k].astype(int).tolist() for k in (0, 1)},
               players=players), open(os.path.join(OUT, "players.json"), "w"), separators=(",", ":"))

# ------------------------------------------------------------------ field config
json.dump(dict(units="yards", length=FIELD_L, width=round(FIELD_W, 2),
               note="x from the left end line, y from the near (camera) sideline. Width is the football width of a shared "
                    "turf field (53.3 yd); a lacrosse-only field is 60 yd wide.",
               lines=[dict(name="end line L", x=0), dict(name="goal line L", x=15), dict(name="restraining line L", x=35),
                      dict(name="midfield", x=55), dict(name="restraining line R", x=75), dict(name="goal line R", x=95),
                      dict(name="end line R", x=110)],
               goals=[dict(x=15, y=round(GOAL_Y, 2), mouth_yd=2.0), dict(x=95, y=round(GOAL_Y, 2), mouth_yd=2.0)],
               crease_radius_yd=3.0, center_x=dict(x=55, y=round(GOAL_Y, 2)),
               wing_lines=[dict(x0=45, x1=65, y=round(GOAL_Y - 20, 2)), dict(x0=45, x1=65, y=round(GOAL_Y + 20, 2))],
               sidelines=[dict(y=0), dict(y=round(FIELD_W, 2))]), open(os.path.join(OUT, "field.json"), "w"), indent=1)

# ------------------------------------------------------------------ quality + summary
tot = sum(c["frames"] for c in quality["chunks"])
mapped = sum(c["frames_mapped"] for c in quality["chunks"])
gw = [c["goal_width_yd_median"] for c in quality["chunks"] if c["goal_width_yd_median"]]
nd = [c["neighbour_disagreement_yd_median"] for c in quality["chunks"] if c["neighbour_disagreement_yd_median"]]
fails = {}
for c in quality["chunks"]:
    for k, v in c["fails"].items():
        fails[k] = fails.get(k, 0) + v
quality.update(frames=tot, frames_mapped=mapped, frames_mapped_share=round(mapped / max(tot, 1), 3),
               goal_width_yd=round(float(np.median(gw)), 2) if gw else None,
               neighbour_disagreement_yd=round(float(np.median(nd)), 2) if nd else None, keyframe_fail_reasons=fails,
               faceoffs=sum(1 for e in events if e["type"] == "faceoff"), goals=len(goals),
               possessions=len(out_poss), seconds_analysis=round(time.time() - T0))
json.dump(quality, open(os.path.join(OUT, "quality.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in quality.items() if k != "chunks"}, indent=1))

# ------------------------------------------------------------------ plain-language summary
comp = sum((c.get("seconds_compute") or 0) for c in quality["chunks"])
game_min = (chunks[-1]["meta"]["t_end"] - chunks[0]["meta"]["t_start"]) / 60
ev_count = {}
for e in events:
    ev_count[e["type"]] = ev_count.get(e["type"], 0) + 1
set_count = {}
for p_ in out_poss:
    set_count[p_["set"]] = set_count.get(p_["set"], 0) + 1
lines = [f"# Engine run: {a.game}", "",
         f"- Film covered: {game_min:.0f} minutes in {len(chunks)} time-ordered chunks at {FPS:g} frames a second.",
         f"- Compute: {comp / 60:.0f} CPU-minutes for extraction ({comp / 60 / max(game_min, 1):.1f} per game minute) + {quality['seconds_analysis']} s analysis.",
         f"- Field mapping: {quality['frames_mapped_share'] * 100:.0f}% of frames have a field fit "
         f"(goal mouth maps to {quality['goal_width_yd']} yd, true 2 yd; neighbouring fits 1 s apart disagree by "
         f"{quality['neighbour_disagreement_yd']} yd). {quality['players_mapped_share'] * 100:.0f}% of player sightings are on the radar.",
         f"- Teams: {team_color[0]} vs {team_color[1]} ({team_method}).",
         f"- Players: {len(players)} stitched tracks with physical data (one athlete can be several tracks until jersey numbers are attached).",
         f"- Possessions: {len(out_poss)} settled possessions; sets: " + ", ".join(f"{k} {v}" for k, v in sorted(set_count.items(), key=lambda z: -z[1])) + ".",
         "- Events: " + ", ".join(f"{k} {v}" for k, v in ev_count.items()) + ".",
         "", "Confidence notes are in events.json; field-mapping failure reasons are in quality.json."]
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
