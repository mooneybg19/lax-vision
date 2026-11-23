"""v1 pass 1 (heavy, runs on GitHub's computers): track players through each settled-offense window.

For every v0 moment (start-3s .. end+3s) it decodes the film at a higher rate (default 8 fps) and saves:
  - person boxes with a BoT-SORT track id (camera-motion compensated, lost tracks kept ~10 s so a player
    who is hidden for a moment keeps his id)
  - the torso color signature, stripes score and turf check per box (same as v0, for the team split)
  - the lacrosse detector's class per box (goalie / long pole / ref / short stick) when models/lax.pt exists
  - camera motion: a homography from each frame to the previous one, estimated from the turf and painted
    lines with players masked out. Chained, it puts every frame of the window on one shared canvas, so a
    player's spot is kept after the camera pans away from him.
  - the goal (orange frame) and the direction of the painted cross-field lines near it
Nothing is tuned to one game.
"""
import argparse, json, os, subprocess, sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from vision_common import torso_feat, learn_turf, on_turf, find_goal, cross_line_angle, TURF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("moments")
ap.add_argument("out")
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--parts", type=int, default=1)
ap.add_argument("--fps", type=float, default=8)
ap.add_argument("--pad", type=float, default=3)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--model", default="yolo11s.pt")
a = ap.parse_args()

g = json.load(open(a.game))
M = json.load(open(a.moments))["moments"]
W = a.width
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", g["video_url"]], capture_output=True, text=True).stdout.strip().split(",")
H = int(round(int(probe[1]) * W / int(probe[0]) / 2) * 2)

TRACKER = os.path.join(a.out, "botsort_lax.yaml")
os.makedirs(a.out, exist_ok=True)
open(TRACKER, "w").write("""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.3
track_buffer: 80
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.75
with_reid: True
model: auto
""")
lax_model = YOLO("models/lax.pt") if os.path.exists("models/lax.pt") else None

lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def cam_motion(prev_g, cur_g, boxes):
    """Homography mapping the current frame onto the previous one (players masked out)."""
    mask = np.full(prev_g.shape, 255, np.uint8)
    mask[: int(0.08 * mask.shape[0])] = 0            # sky / stands / score graphics at the very top
    mask[int(0.9 * mask.shape[0]):] = 0              # lower-third graphics
    for x1, y1, x2, y2 in boxes:
        mask[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6] = 0
    p0 = cv2.goodFeaturesToTrack(prev_g, 600, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 12:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None, **lk)
    ok = st.reshape(-1) == 1
    if ok.sum() < 12:
        return np.eye(3), 0
    Hm, inl = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, 2.0)
    if Hm is None:
        return np.eye(3), 0
    return Hm, int(inl.sum())


mine = [i for i in range(len(M)) if i % a.parts == a.part]
for mi in mine:
    m = M[mi]
    t0 = max(0.0, m["start"] - a.pad)
    L = m["end"] + a.pad - t0
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", g["video_url"], "-t", str(L),
           "-vf", f"fps={a.fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    model = YOLO(a.model)   # fresh tracker per window
    rows, fmeta, prev_g = [], [], None
    fi = 0
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        im = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        t = t0 + fi / a.fps
        try:
            res = model.track(im, persist=True, tracker=TRACKER, classes=[0], conf=0.2, imgsz=W, verbose=False)[0]
        except Exception as e:  # ReID not available in this ultralytics build: fall back to plain BoT-SORT
            if "reid" in str(e).lower() or "model" in str(e).lower():
                open(TRACKER, "a").write("")
                txt = open(TRACKER).read().replace("with_reid: True", "with_reid: False")
                open(TRACKER, "w").write(txt)
                model = YOLO(a.model)
                res = model.track(im, persist=True, tracker=TRACKER, classes=[0], conf=0.2, imgsz=W, verbose=False)[0]
            else:
                raise
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        learn_turf(hsv)
        bx = res.boxes
        boxes = bx.xyxy.cpu().numpy().astype(int) if bx is not None else np.zeros((0, 4), int)
        confs = bx.conf.cpu().numpy() if bx is not None else np.zeros(0)
        ids = bx.id.cpu().numpy().astype(int) if (bx is not None and bx.id is not None) else -np.ones(len(boxes), int)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        Hm, ninl = (np.eye(3), -1) if prev_g is None else cam_motion(prev_g, gray, boxes)
        prev_g = gray
        lax_cls = np.full(len(boxes), -1, int)
        if lax_model is not None and len(boxes):
            lr = lax_model(im, imgsz=960, conf=0.25, verbose=False)[0]
            lb = lr.boxes.xyxy.cpu().numpy(); lc = lr.boxes.cls.cpu().numpy().astype(int)
            for k, b in enumerate(boxes):
                if not len(lb):
                    break
                ix1 = np.maximum(b[0], lb[:, 0]); iy1 = np.maximum(b[1], lb[:, 1])
                ix2 = np.minimum(b[2], lb[:, 2]); iy2 = np.minimum(b[3], lb[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                uni = (b[2] - b[0]) * (b[3] - b[1]) + (lb[:, 2] - lb[:, 0]) * (lb[:, 3] - lb[:, 1]) - inter
                iou = inter / np.maximum(uni, 1); j = int(np.argmax(iou))
                if iou[j] > 0.4:
                    lax_cls[k] = lc[j]
        turf = np.array([on_turf(hsv, b) for b in boxes], bool)
        ph = float(np.median([b[3] - b[1] for b, tf in zip(boxes, turf) if tf])) if turf.any() else 0.0
        goal = find_goal(hsv, ph) if ph > 8 else None
        gang = cross_line_angle(hsv, ((goal[0] + goal[2]) / 2, goal[3]), boxes[turf] if turf.any() else []) if goal is not None else None
        fmeta.append(dict(t=round(t, 3), H_to_prev=np.round(Hm, 6).tolist(), inliers=ninl, ph=round(ph, 1), goal=goal,
                          gline_angle=None if gang is None else round(gang, 1)))
        for b, c, tid, tf, lc_ in zip(boxes, confs, ids, turf, lax_cls):
            f, s = torso_feat(hsv, b)
            rows.append([fi, t, tid, *b.tolist(), float(c), int(tf), float(s), int(lc_), *f.tolist()])
        fi += 1
    proc.wait()
    d = os.path.join(a.out, f"m{mi + 1:02d}")
    os.makedirs(d, exist_ok=True)
    arr = np.array(rows, np.float32) if rows else np.zeros((0, 27), np.float32)
    np.savez_compressed(os.path.join(d, "dets.npz"), dets=arr)
    json.dump(dict(moment=mi + 1, start=m["start"], end=m["end"], t0=t0, fps=a.fps, width=W, height=H,
                   lax_names=lax_model.names if lax_model else {}, frames=fmeta,
                   reid=("with_reid: True" in open(TRACKER).read())), open(os.path.join(d, "frames.json"), "w"))
    print(f"moment {mi + 1}: {fi} frames, {len(rows)} boxes, {len(set(int(r[2]) for r in rows))} track ids", flush=True)
