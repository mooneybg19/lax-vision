"""Engine pass 1 (heavy, GitHub Actions): one time-ordered chunk of the WHOLE game.

For each frame (default 3 fps, 1280 wide) it saves:
  - player boxes with a BoT-SORT track id (camera-motion compensated, ReID when available)
  - the lacrosse detector's role per box (Goalie / Longpole / Referee / Shortstick), torso color, stripes, turf
  - camera motion to the previous frame (homography, players masked out)
  - the goal (orange frame)
  - once a second: the painted line segments and near sideline (inputs for the field homography)
After the chunk: up to 3 crops per track go through SigLIP (the roboflow/sports team-classifier recipe:
SigLIP embeddings -> UMAP -> KMeans, done later over the whole game) and the mean embedding per track is saved.

Chunks overlap by a few seconds so the analysis can stitch track ids across chunk boundaries.
"""
import argparse, json, os, subprocess, sys, time
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from vision_common import torso_feat, learn_turf, on_turf, find_goal  # noqa: E402
from field_geom import find_segments  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("out")
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--parts", type=int, default=1)
ap.add_argument("--fps", type=float, default=3)
ap.add_argument("--overlap", type=float, default=4)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--siglip", default="google/siglip-base-patch16-224")
ap.add_argument("--limit", type=float, default=0, help="debug: only this many seconds")
a = ap.parse_args()

g = json.load(open(a.game))
D = float(g["duration_s"])
L = D / a.parts
t_start = max(0.0, a.part * L - (a.overlap if a.part else 0))
t_end = min(D, (a.part + 1) * L)
if a.limit:
    t_end = min(t_end, t_start + a.limit)
W = a.width
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", g["video_url"]], capture_output=True, text=True).stdout.strip().split(",")
H = int(round(int(probe[1]) * W / int(probe[0]) / 2) * 2)
os.makedirs(a.out, exist_ok=True)
TRACKER = os.path.join(a.out, "botsort.yaml")
open(TRACKER, "w").write("""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.3
track_buffer: 30
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
    mask = np.full(prev_g.shape, 255, np.uint8)
    mask[: int(0.08 * mask.shape[0])] = 0
    mask[int(0.9 * mask.shape[0]):] = 0
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


def track(model, im):
    global TRACKER
    try:
        return model, model.track(im, persist=True, tracker=TRACKER, classes=[0], conf=0.2, imgsz=W, verbose=False)[0]
    except Exception as e:
        if "reid" in str(e).lower() or "model" in str(e).lower():
            txt = open(TRACKER).read().replace("with_reid: True", "with_reid: False")
            open(TRACKER, "w").write(txt)
            model = YOLO(a.model)
            return model, model.track(im, persist=True, tracker=TRACKER, classes=[0], conf=0.2, imgsz=W, verbose=False)[0]
        raise


t_wall = time.time()
cmd = ["ffmpeg", "-v", "error", "-ss", str(t_start), "-i", g["video_url"], "-t", str(t_end - t_start),
       "-vf", f"fps={a.fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
model = YOLO(a.model)
rows, fmeta, keys, prev_g = [], [], {}, None
crops = {}          # track id -> list of (box height, frame index, crop)
fi = 0
kstep = max(1, int(round(a.fps)))
while True:
    buf = proc.stdout.read(W * H * 3)
    if len(buf) < W * H * 3:
        break
    im = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    t = t_start + fi / a.fps
    model, res = track(model, im)
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
    fmeta.append(dict(t=round(t, 3), H=np.round(Hm, 6).tolist(), inl=ninl, ph=round(ph, 1), goal=goal))
    if fi % kstep == 0:
        segs, side = find_segments(im, boxes[turf] if turf.any() else [])
        segs = sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))[:80]
        keys[fi] = dict(segs=[[round(v, 1) for v in s] for s in segs], side=side)
    for b, c, tid, tf, lc_ in zip(boxes, confs, ids, turf, lax_cls):
        f, s = torso_feat(hsv, b)
        rows.append([fi, t, tid, *b.tolist(), float(c), int(tf), float(s), int(lc_), *f.tolist()])
        if tid >= 0 and tf and (b[3] - b[1]) >= 24:
            x1, y1, x2, y2 = b
            cr = cv2.resize(im[max(0, y1):y2, max(0, x1):x2], (48, 96))
            lst = crops.setdefault(int(tid), [])
            if not lst or fi - lst[-1][1] >= 2 * a.fps:
                lst.append((int(y2 - y1), fi, cr))
                if len(lst) > 6:
                    lst.sort(key=lambda z: -z[0]); lst.pop()
                    lst.sort(key=lambda z: z[1])
    fi += 1
    if fi % 300 == 0:
        print(f"part {a.part}: {fi} frames, {time.time() - t_wall:.0f}s", flush=True)
proc.wait()
t_track = time.time() - t_wall

# ------------------------------------------------------------- SigLIP embeddings per track (team split later)
emb_ids, emb = [], []
try:
    import torch
    from transformers import AutoProcessor, SiglipVisionModel
    proc_s = AutoProcessor.from_pretrained(a.siglip)
    vis = SiglipVisionModel.from_pretrained(a.siglip).eval()
    batch_imgs, batch_ids = [], []
    keep = {k: v for k, v in crops.items() if len(v) >= 2}

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            inp = proc_s(images=[cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in batch_imgs], return_tensors="pt")
            out = vis(**inp).pooler_output.numpy()
        for tid, e in zip(batch_ids, out):
            emb_ids.append(tid); emb.append(e)
        batch_imgs.clear(); batch_ids.clear()

    for tid, lst in keep.items():
        for _, _, cr in sorted(lst, key=lambda z: -z[0])[:3]:
            batch_imgs.append(cr); batch_ids.append(tid)
            if len(batch_imgs) >= 32:
                flush()
    flush()
except Exception as e:  # team split falls back to torso colors
    print("siglip unavailable:", e, flush=True)
if emb:
    E = np.array(emb, np.float32); I = np.array(emb_ids)
    u = np.unique(I)
    M = np.stack([E[I == k].mean(0) for k in u]).astype(np.float16)
else:
    u = np.zeros(0, int); M = np.zeros((0, 768), np.float16)

arr = np.array(rows, np.float32) if rows else np.zeros((0, 27), np.float32)
np.savez_compressed(os.path.join(a.out, "dets.npz"), dets=arr, emb_ids=u, emb=M)
json.dump(dict(part=a.part, parts=a.parts, t_start=t_start, t_end=t_end, fps=a.fps, width=W, height=H,
               lax_names=lax_model.names if lax_model else {}, frames=fmeta, keys=keys,
               reid=("with_reid: True" in open(TRACKER).read()),
               seconds_tracking=round(t_track), seconds_total=round(time.time() - t_wall)),
          open(os.path.join(a.out, "frames.json"), "w"))
print(f"part {a.part}: {fi} frames, {len(rows)} boxes, {len(set(int(r[2]) for r in rows))} tracks, "
      f"{len(u)} embedded; {time.time() - t_wall:.0f}s", flush=True)
