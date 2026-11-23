"""Pass 1 (heavy, runs on GitHub's computers): read one stretch of a game and save what's on the field.

For each sampled frame it stores:
  - every person box from YOLO11 (COCO 'person'). No cross-frame tracking yet: at 2 frames a second ByteTrack
    dropped too many new players, and reading a set only needs where everyone is at each moment
  - a small color signature of each person's torso (used later to split the two teams per game)
  - a 'stripes' score (refs wear black-and-white stripes)
  - whether the feet stand on turf (drops coaches, benches, fans)
  - the goal, found by its orange frame (the goal is the one fixed landmark every lacrosse field has)
  - the direction of the painted cross-field lines near the goal (gives the goal-line direction on screen)
Nothing here is tuned to one game: sizes are measured against the players' own height on screen.
"""
import argparse, json, os, subprocess
import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("out")
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--length", type=float, default=0)
ap.add_argument("--fps", type=float, default=2)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--model", default="yolo11s.pt")
a = ap.parse_args()

g = json.load(open(a.game))
W = a.width
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", g["video_url"]], capture_output=True, text=True).stdout.strip().split(",")
H = int(round(int(probe[1]) * W / int(probe[0]) / 2) * 2)
cmd = ["ffmpeg", "-v", "error", "-ss", str(a.start), "-i", g["video_url"]]
if a.length > 0:
    cmd += ["-t", str(a.length)]
cmd += ["-vf", f"fps={a.fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)

model = YOLO(a.model)
lax_model = YOLO("models/lax.pt") if os.path.exists("models/lax.pt") else None
LAX_NAMES = lax_model.names if lax_model else {}


def torso_feat(hsv, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    tx1, tx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    ty1, ty2 = int(y1 + 0.2 * h), int(y1 + 0.5 * h)
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        return np.zeros(16, np.uint8), 0.0
    p = hsv[ty1:ty2, tx1:tx2].reshape(-1, 3).astype(np.float32)
    # drop turf pixels from the torso patch
    turf = is_turf_px(p)
    q = p[~turf] if (~turf).sum() > 6 else p
    hh = np.histogram(q[:, 0], bins=8, range=(0, 180), weights=(q[:, 1] > 60).astype(np.float32))[0]
    sh = np.histogram(q[:, 1], bins=4, range=(0, 256))[0]
    vh = np.histogram(q[:, 2], bins=4, range=(0, 256))[0]
    f = np.concatenate([hh / max(len(q), 1), sh / len(q), vh / len(q)])
    stripes = float(np.std(np.diff(hsv[ty1:ty2, (tx1 + tx2) // 2, 2].astype(np.float32)))) if ty2 - ty1 > 4 else 0.0
    return np.clip(f * 255, 0, 255).astype(np.uint8), stripes


TURF = dict(h=60.0)


def learn_turf(hsv):
    """The field color for this frame: the most common hue in the lower two thirds of the picture
    among colored pixels. Follows sunset light, shadows, blue or brown fields."""
    p = hsv[hsv.shape[0] // 3:, ::4].reshape(-1, 3)
    p = p[(p[:, 1] > 35) & (p[:, 2] > 40)]
    if len(p) < 500:
        return
    hist = np.bincount(p[:, 0].astype(int), minlength=180).astype(float)
    hist = np.convolve(np.concatenate([hist[-5:], hist, hist[:5]]), np.ones(9) / 9, "same")[5:-5]
    TURF["h"] = float(np.argmax(hist))


def is_turf_px(p):
    dh = np.abs(p[:, 0].astype(float) - TURF["h"])
    dh = np.minimum(dh, 180 - dh)
    return (dh < 14) & (p[:, 1] > 30)


def on_turf(hsv, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    ys = slice(int(min(hsv.shape[0] - 1, y2)), int(min(hsv.shape[0], y2 + max(2, 0.15 * h))))
    xs = slice(int(max(0, x1 - 0.3 * (x2 - x1))), int(min(hsv.shape[1], x2 + 0.3 * (x2 - x1))))
    p = hsv[ys, xs].reshape(-1, 3)
    if len(p) == 0:
        return False
    green = is_turf_px(p)
    white = (p[:, 1] < 50) & (p[:, 2] > 170)
    return (green | white).mean() > 0.5


def find_goal(hsv, ph):
    """Orange goal: the two orange posts (about a player's height, thin) and/or the frame as one blob.
    Posts that stand side by side with their feet level are joined into one goal."""
    m = cv2.inRange(hsv, (0, 120, 100), (22, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    parts = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (0.45 * ph <= h <= 1.8 * ph and w <= 3.0 * ph):
            continue
        if area / float(w * h) > 0.6 and w > 0.3 * ph:  # solid orange patch (shirt, sign) is not a goal
            continue
        if not on_turf(hsv, (x, y, x + w, y + h)):
            continue
        parts.append([x, y, x + w, y + h])
    if not parts:
        return None
    parts.sort(key=lambda b: b[0])
    groups = []
    for b in parts:
        for gb in groups:
            if abs(gb[3] - b[3]) < 0.35 * ph and b[0] - gb[2] < 1.6 * ph:
                gb[0] = min(gb[0], b[0]); gb[1] = min(gb[1], b[1]); gb[2] = max(gb[2], b[2]); gb[3] = max(gb[3], b[3]); gb[4] += 1
                break
        else:
            groups.append(list(b) + [1])
    best = max(groups, key=lambda gb: (gb[4], gb[3] - gb[1]))
    return [int(v) for v in best[:4]]


def cross_line_angle(hsv, near, boxes):
    """Angle (degrees, image) of the painted lines running across the field near a point."""
    Hh, Ww = hsv.shape[:2]
    th = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    mk = ((th > 30) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    for x1, y1, x2, y2 in boxes:
        mk[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = 0
    L = cv2.HoughLinesP(mk, 1, np.pi / 360, 60, minLineLength=int(Hh * 0.12), maxLineGap=10)
    if L is None:
        return None
    segs = []
    for x1, y1, x2, y2 in L.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 25 <= ang <= 155 and y2 != y1:  # cross-field lines are steep on a sideline camera
            xc = x1 + (x2 - x1) * (near[1] - y1) / (y2 - y1)
            segs.append((abs(xc - near[0]), ang, np.hypot(x2 - x1, y2 - y1)))
    if not segs:
        return None
    segs.sort()
    near_s = segs[:6]
    an = np.radians([s[1] for s in near_s])
    w = [s[2] for s in near_s]
    return float(np.degrees(np.arctan2(np.average(np.sin(2 * an), weights=w), np.average(np.cos(2 * an), weights=w)) / 2) % 180)


os.makedirs(a.out, exist_ok=True)
rows, frames_meta = [], []
fi = 0
while True:
    buf = proc.stdout.read(W * H * 3)
    if len(buf) < W * H * 3:
        break
    im = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    t = a.start + fi / a.fps
    res = model.predict(im, classes=[0], conf=0.2, imgsz=W, verbose=False)[0]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    learn_turf(hsv)
    boxes = res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0, 4), int)
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
    ids = res.boxes.id.cpu().numpy().astype(int) if (res.boxes is not None and res.boxes.id is not None) else -np.ones(len(boxes), int)
    lax_cls = np.full(len(boxes), -1, int)
    if lax_model is not None and len(boxes):
        lr = lax_model(im, imgsz=960, conf=0.25, verbose=False)[0]
        lb = lr.boxes.xyxy.cpu().numpy()
        lc = lr.boxes.cls.cpu().numpy().astype(int)
        for k, b in enumerate(boxes):
            if len(lb) == 0:
                break
            ix1 = np.maximum(b[0], lb[:, 0]); iy1 = np.maximum(b[1], lb[:, 1])
            ix2 = np.minimum(b[2], lb[:, 2]); iy2 = np.minimum(b[3], lb[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            uni = (b[2] - b[0]) * (b[3] - b[1]) + (lb[:, 2] - lb[:, 0]) * (lb[:, 3] - lb[:, 1]) - inter
            iou = inter / np.maximum(uni, 1)
            j = int(np.argmax(iou))
            if iou[j] > 0.4:
                lax_cls[k] = lc[j]
    turf = np.array([on_turf(hsv, b) for b in boxes], bool)
    ph = float(np.median([b[3] - b[1] for b, tf in zip(boxes, turf) if tf])) if turf.any() else 0.0
    goal = find_goal(hsv, ph) if ph > 8 else None
    gang = None
    if goal is not None:
        gang = cross_line_angle(hsv, ((goal[0] + goal[2]) / 2, goal[3]), boxes[turf] if turf.any() else [])
    frames_meta.append(dict(t=round(t, 2), turf_hue=round(TURF['h'], 1), ph=round(ph, 1), goal=goal, gline_angle=None if gang is None else round(gang, 1)))
    for b, c, tid, tf, lc_ in zip(boxes, confs, ids, turf, lax_cls):
        f, s = torso_feat(hsv, b)
        rows.append([fi, t, tid, *b.tolist(), float(c), int(tf), float(s), int(lc_), *f.tolist()])
    fi += 1
    if fi % 200 == 0:
        print(f"{fi} frames, t={t:.0f}s, goal seen {sum(m['goal'] is not None for m in frames_meta)}", flush=True)
proc.wait()
arr = np.array(rows, np.float32) if rows else np.zeros((0, 28), np.float32)
np.savez_compressed(os.path.join(a.out, "dets.npz"), dets=arr.astype(np.float32),
                    cols=np.array(["frame", "t", "track", "x1", "y1", "x2", "y2", "conf", "turf", "stripes", "lax_cls"] +
                                  [f"f{i}" for i in range(16)]))
json.dump(dict(game=a.game, start=a.start, fps=a.fps, width=W, height=H, lax_names=LAX_NAMES, frames=frames_meta),
          open(os.path.join(a.out, "frames.json"), "w"))
print("done", fi, "frames", len(rows), "detections")
