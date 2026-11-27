"""Load and validate field-landmark labels exported from the labeling tool (tools/labeler/index.html).

The landmark and line definitions below are an EXACT port of the labeler's JavaScript (laxPoints, fbPoints,
landmarks, fieldLines, yardLines, fitH); tests/test_labels_io.py checks the port against the JS run in node.

Label document (one per frame, see persist() in the labeler):
  frame, game, skip, img [w,h], pts {name:[u,v]} clicked px, world {name:[x,y]} yards, field "lacrosse"|"football",
  width (60 | 53.33), hash (17.78 | 20 | null), H (9, row-major, field yd -> image px, paint-snapped when quality is
  "paint"), H_clicks (9, the click-only fit), quality, paint_match, paint_background, click_off {name: px},
  click_off_yd, rms, loo, z_max, ...
World frame: x 0..110 left->right as seen from the camera, y = 0 at the near (camera) sideline, y = width far.
"""
import json
import numpy as np

FB_WIDTH = 53.33
HASH_HS = 17.78
HASH_COLLEGE = 20.0
QUALITIES = ("paint", "points", "paint_points_disagree")


# ------------------------------------------------------------------ landmark definitions (port of the labeler JS)
def lax_points(W):
    """laxPoints(W): {name: (x, y, label)} in insertion order (the order matters: it is the channel order)."""
    c = W / 2; lo = c - 20; hi = c + 20
    return {
        "corner_0_0": (0, 0, "Left end line at camera sideline"), "corner_0_60": (0, W, "Left end line at far sideline"),
        "corner_110_0": (110, 0, "Right end line at camera sideline"), "corner_110_60": (110, W, "Right end line at far sideline"),
        "restrain_L_near": (35, 0, "Left restraining line at camera sideline"), "restrain_L_far": (35, W, "Left restraining line at far sideline"),
        "restrain_R_near": (75, 0, "Right restraining line at camera sideline"), "restrain_R_far": (75, W, "Right restraining line at far sideline"),
        "center_near": (55, 0, "Midfield line at camera sideline"), "center_far": (55, W, "Midfield line at far sideline"), "faceoff_X": (55, c, "Faceoff X"),
        "boxL_near": (35, lo, "Left box corner on restraining line, camera side"), "boxL_far": (35, hi, "Left box corner on restraining line, far side"),
        "boxL_end_near": (0, lo, "Left box line at end line, camera side"), "boxL_end_far": (0, hi, "Left box line at end line, far side"),
        "boxR_near": (75, lo, "Right box corner on restraining line, camera side"), "boxR_far": (75, hi, "Right box corner on restraining line, far side"),
        "boxR_end_near": (110, lo, "Right box line at end line, camera side"), "boxR_end_far": (110, hi, "Right box line at end line, far side"),
        "wing_near_L": (45, lo, "Camera-side wing line, left end"), "wing_near_R": (65, lo, "Camera-side wing line, right end"),
        "wing_far_L": (45, hi, "Far wing line, left end"), "wing_far_R": (65, hi, "Far wing line, right end"),
        "creaseL_front": (18, c, "Left crease, front edge"), "creaseL_back": (12, c, "Left crease, back edge"),
        "creaseL_near": (15, c - 3, "Left crease, camera-side edge"), "creaseL_far": (15, c + 3, "Left crease, far edge"),
        "goalL_near": (15, c - 1, "Left goal, camera-side post"), "goalL_far": (15, c + 1, "Left goal, far post"),
        "creaseR_front": (92, c, "Right crease, front edge"), "creaseR_back": (98, c, "Right crease, back edge"),
        "creaseR_near": (95, c - 3, "Right crease, camera-side edge"), "creaseR_far": (95, c + 3, "Right crease, far edge"),
        "goalR_near": (95, c - 1, "Right goal, camera-side post"), "goalR_far": (95, c + 1, "Right goal, far post"),
    }


def fb_points(W, hashD):
    """fbPoints(W) with the labeler's global hashD passed explicitly. {name: (x, y, label)}."""
    o = {}

    def add(x, nm, key):
        if key not in ("50", "L30", "R30"):
            o[f"fb_{key}_near"] = (x, 0, f"{nm} at camera sideline")
            o[f"fb_{key}_far"] = (x, W, f"{nm} at far sideline")
        o[f"fb_{key}_hnear"] = (x, hashD, f"{nm} at camera-side hash marks")
        o[f"fb_{key}_hfar"] = (x, W - hashD, f"{nm} at far hash marks")

    for sd in ("L", "R"):
        for N in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45):
            side = "Left" if sd == "L" else "Right"
            add(5 + N if sd == "L" else 105 - N, f"{side} {'goal line' if N == 0 else str(N) + '-yard line'}", sd + str(N))
    add(55, "50-yard line", "50")
    return o


def landmarks(W, fb, hashD=HASH_HS):
    """landmarks(W, fb): lacrosse points, plus football points when fb."""
    d = dict(lax_points(W))
    if fb:
        d.update(fb_points(W, hashD))
    return d


def field_lines(W):
    c = W / 2; lo = c - 20; hi = c + 20
    return [[[0, 0], [110, 0]], [[0, W], [110, W]], [[0, 0], [0, W]], [[110, 0], [110, W]], [[55, 0], [55, W]],
            [[35, 0], [35, W]], [[75, 0], [75, W]],
            [[0, lo], [35, lo]], [[0, hi], [35, hi]], [[75, lo], [110, lo]], [[75, hi], [110, hi]], [[45, lo], [65, lo]],
            [[45, hi], [65, hi]], [[15, c - 1], [15, c + 1]], [[95, c - 1], [95, c + 1]]]


def yard_lines(W):
    return [[[x, 0], [x, W]] for x in range(5, 106, 5)]


def world_of(name, W, hashD, fb):
    """World (x, y) of one landmark for the given width / hash (used by the per-game solver)."""
    L = landmarks(W, fb, hashD)
    return float(L[name][0]), float(L[name][1])


# ------------------------------------------------------------------ homography (port of the labeler's fitH)
def fit_h(world, img):
    """Normalised DLT, world -> image, same normalisation, sign and scale convention as the labeler's fitH.
    (The labeler takes the smallest eigenvector of A^T A by Jacobi; here by SVD. Same solution.)"""
    world = np.asarray(world, float); img = np.asarray(img, float)

    def norm(P):
        c = P.mean(0); d = np.mean(np.hypot(*(P - c).T)); s = np.sqrt(2) / d if d > 0 else 1.0
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

    Tw, Ti = norm(world), norm(img)
    Wn = (Tw @ np.c_[world, np.ones(len(world))].T).T; In = (Ti @ np.c_[img, np.ones(len(img))].T).T
    A = []
    for (X, Y, _), (u, v, _) in zip(Wn, In):
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]); A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = np.linalg.inv(Ti) @ Vt[-1].reshape(3, 3) @ Tw
    if H[2, 0] * world[0, 0] + H[2, 1] * world[0, 1] + H[2, 2] < 0:
        H = -H
    return H / (abs(H[2, 2]) or 1.0)


def proj(H, P):
    """Apply H to (N,2) points. Returns (N,2) and the homogeneous w (N,)."""
    P = np.atleast_2d(np.asarray(P, float))
    Q = (np.asarray(H, float) @ np.c_[P, np.ones(len(P))].T).T
    return Q[:, :2] / Q[:, 2:3], Q[:, 2]


# ------------------------------------------------------------------ loading + validation
def load(path_or_list):
    """Load a JSON export: a list of documents, or {"labels": [...]}, or {frame_id: doc} (artifact db dump)."""
    data = path_or_list
    if isinstance(path_or_list, str):
        with open(path_or_list) as f:
            data = json.load(f)
    if isinstance(data, dict):
        data = data.get("labels", data.get("docs", list(data.values())))
    out = []
    for d in data:
        if isinstance(d, dict) and "data" in d and "frame" not in d:      # db_list style {id, data, version}
            d = d["data"]
        out.append(d)
    return out


def doc_geometry(doc):
    """(W, hashD, fb) of a label document, defaults as the labeler would have used."""
    fb = doc.get("field") == "football"
    W = float(doc.get("width") or (FB_WIDTH if fb else 60.0))
    hashD = float(doc.get("hash") or HASH_HS)
    return W, hashD, fb


def H_of(doc, key="H"):
    return np.asarray(doc[key], float).reshape(3, 3)


def validate(doc, tol_world=2e-3, tol_off_px=0.35):
    """Check one label. Returns (errors, warnings): lists of strings. errors => do not train on it."""
    E, Wn = [], []
    if doc.get("skip"):
        return ["skipped frame"], []
    for k in ("frame", "game", "img", "pts", "world", "field", "H"):
        if k not in doc:
            E.append(f"missing {k}")
    if E:
        return E, Wn
    if doc["field"] not in ("lacrosse", "football"):
        E.append(f"unknown field {doc['field']!r}")
    W, hashD, fb = doc_geometry(doc)
    if fb and abs(W - FB_WIDTH) > 1e-6:
        E.append(f"football field with width {W}")
    if not fb and W not in (60.0, FB_WIDTH):
        Wn.append(f"unusual width {W}")
    if fb and min(abs(hashD - HASH_HS), abs(hashD - HASH_COLLEGE)) > 1e-6:
        Wn.append(f"unusual hash {hashD}")
    if list(doc["img"]) != [1280, 720]:
        Wn.append(f"image size {doc['img']} (expected 1280x720)")
    L = landmarks(W, fb, hashD)
    names = list(doc["pts"].keys())
    if len(names) < 4:
        E.append(f"only {len(names)} clicks")
    for n in names:
        if n not in L:
            E.append(f"unknown landmark {n} for field={doc['field']}")
            continue
        if n not in doc["world"]:
            E.append(f"no world coords for {n}"); continue
        dw = np.hypot(doc["world"][n][0] - L[n][0], doc["world"][n][1] - L[n][1])
        if dw > tol_world:
            E.append(f"world coords of {n} {doc['world'][n]} != definition {L[n][:2]} (off {dw:.4f} yd)")
    if E:
        return E, Wn
    H = H_of(doc)
    if not np.all(np.isfinite(H)):
        return ["H not finite"], Wn
    s = np.linalg.svd(H, compute_uv=False)
    if not s[0] > 0 or s[-1] / s[0] < 1e-12:
        return ["H singular"], Wn
    wpts = np.array([L[n][:2] for n in names], float); ipts = np.array([doc["pts"][n] for n in names], float)
    q, w = proj(H, wpts)
    if np.any(w <= 1e-6):
        E.append("a clicked landmark projects behind the camera under H")
    off = np.hypot(*(q - ipts).T)
    co = doc.get("click_off") or {}
    if co:
        for n, o in zip(names, off):
            if n not in co:
                E.append(f"click_off missing {n}")
            elif abs(co[n] - o) > tol_off_px:
                E.append(f"click {n}: reprojection {o:.2f}px under H but stated click_off {co[n]}px")
        if doc.get("quality") == "paint" and max(co.values()) > 10.0 + 0.05:
            E.append(f"quality=paint but a click is {max(co.values())} px off (labeler limit 10)")
    else:
        # unsnapped (quality=points): H is the click fit, so its rms must match the stated rms
        rms = float(np.sqrt(np.mean(off ** 2)))
        if "rms" in doc and abs(rms - doc["rms"]) > tol_off_px:
            E.append(f"click rms under H {rms:.2f}px but stated rms {doc['rms']}")
    if "H_clicks" in doc and len(names) >= 4:
        Hc = H_of(doc, "H_clicks"); Hr = fit_h(wpts, ipts)
        d = np.max(np.hypot(*(proj(Hc, wpts)[0] - proj(Hr, wpts)[0]).T))
        if d > 0.5:
            E.append(f"H_clicks does not match a refit of the clicks ({d:.2f}px)")
    if doc.get("quality") not in QUALITIES:
        E.append(f"quality {doc.get('quality')!r}")
    if doc.get("quality") == "paint_points_disagree":
        Wn.append("paint_points_disagree: resolved by game_solve")
    if off.max() > 15:
        Wn.append(f"a click is {off.max():.1f}px from H")
    return E, Wn


def load_valid(path_or_list, verbose=True):
    """Load, validate, return (good_docs, report). Skipped frames are dropped silently (counted)."""
    docs = load(path_or_list)
    good, rep = [], {"total": len(docs), "skipped": 0, "rejected": {}, "warnings": {}}
    seen = set()
    for d in docs:
        if d.get("skip"):
            rep["skipped"] += 1; continue
        if d.get("frame") in seen:
            rep["rejected"][d["frame"]] = ["duplicate frame"]; continue
        seen.add(d.get("frame"))
        e, w = validate(d)
        if e:
            rep["rejected"][d.get("frame", "?")] = e
        else:
            good.append(d)
        if w:
            rep["warnings"][d.get("frame", "?")] = w
    rep["valid"] = len(good)
    if verbose:
        print(f"labels: {rep['total']} docs, {rep['skipped']} skipped, {len(good)} valid, {len(rep['rejected'])} rejected")
        for k, v in list(rep["rejected"].items())[:20]:
            print("  REJECT", k, "; ".join(v))
    return good, rep
