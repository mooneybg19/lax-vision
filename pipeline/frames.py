"""Pass 3 (needs the video, runs on GitHub's computers): pull frames for the settled-offense moments.

Writes:
  <out>/label/tNNNNN.jpg + items.json  - clean frames, 1280 wide, for Brian to label (no markings)
  <out>/annotated/tNNNNN.jpg            - the same moments with the tool's reading drawn on
Label frames are spread evenly across the game so they aren't all from one stretch.
"""
import argparse, json, os, subprocess
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("sets_dir")
ap.add_argument("out")
ap.add_argument("--n-label", type=int, default=22)
ap.add_argument("--n-annot", type=int, default=12)
a = ap.parse_args()

g = json.load(open(a.game))
M = json.load(open(os.path.join(a.sets_dir, "moments.json")))
T = np.load(os.path.join(a.sets_dir, "teams.npz"))
moments = M["moments"]


def grab(t, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", g["video_url"], "-frames:v", "1",
                    "-vf", "scale=1280:-2", "-q:v", "4", "-y", dst], check=False)
    return os.path.exists(dst)


def pick(n):
    if len(moments) <= n:
        return list(range(len(moments)))
    return sorted({int(round(i)) for i in np.linspace(0, len(moments) - 1, n)})


os.makedirs(f"{a.out}/label", exist_ok=True)
os.makedirs(f"{a.out}/annotated", exist_ok=True)
items = []
for k in pick(a.n_label):
    t = moments[k]["mid"]
    sid = f"t{int(round(t)):05d}"
    dst = f"{a.out}/label/{sid}.jpg"
    if grab(t, dst):
        im = cv2.imread(dst)
        cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 82])
        mm, ss = divmod(int(round(t)), 60)
        items.append(dict(id=sid, seconds=int(round(t)), caption=f"{mm}:{ss:02d} into video"))
json.dump(items, open(f"{a.out}/label/items.json", "w"), indent=1)

COL = {0: (255, 255, 255), 1: (40, 40, 200), 2: (128, 128, 128)}
for k in pick(a.n_annot):
    m = moments[k]
    t = m["mid"]
    sid = f"t{int(round(t)):05d}"
    dst = f"{a.out}/annotated/{sid}.jpg"
    if not grab(t, dst):
        continue
    im = cv2.imread(dst)
    sel = np.abs(T["t"] - t) < 0.26
    for b, tm in zip(T["box"][sel], T["team"][sel]):
        if tm < 0:
            continue
        col = COL.get(int(tm), (0, 200, 200))
        thick = 3 if int(tm) == m["offense_team"] else 1
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, thick)
        if int(tm) == m["offense_team"]:
            cv2.putText(im, "O", (int(b[0]), int(b[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    s = min(m["samples"], key=lambda c: abs(c["t"] - t))
    txt = f"{sid}  offense: {m['offense']}  set: {m['set']} (conf {m['confidence']})  top/mid/front/behind={s['counts']}"
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(im, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 82])
print(len(items), "label frames")
