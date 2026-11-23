"""v1 field ruler pass (runs on GitHub's computers, no detector): for the same frames track_v1.py reads,
save the painted lines that run ACROSS the field (goal line, restraining line, and on shared fields the
5-yard lines). Lines across the field are parallel on the ground, so the spacing between neighbours on the
screen gives a perspective-correct ruler at every spot of the picture. analyze_v1.py uses it to measure how
far above or behind the goal line each player stands, in yards, instead of in player heights.
"""
import argparse, json, os, subprocess
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("tracks")       # results/<game>-tracks-v1 (for the exact frame times)
ap.add_argument("out")
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--parts", type=int, default=1)
a = ap.parse_args()
g = json.load(open(a.game))
mdirs = sorted(d for d in os.listdir(a.tracks) if d.startswith("m"))
mine = [d for i, d in enumerate(mdirs) if i % a.parts == a.part]


def cross_lines(im):
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    mk = ((th > 28) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    mk[: int(0.25 * H)] = 0   # stands, trees, fences
    L = cv2.HoughLinesP(mk, 1, np.pi / 360, 50, minLineLength=int(H * 0.08), maxLineGap=14)
    out = []
    if L is None:
        return out
    for x1, y1, x2, y2 in L.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 20 <= ang <= 160:          # across the field on a sideline camera: steep on screen
            out.append([int(x1), int(y1), int(x2), int(y2)])
    return out


for md in mine:
    meta = json.load(open(os.path.join(a.tracks, md, "frames.json")))
    W, H, fps, t0 = meta["width"], meta["height"], meta["fps"], meta["t0"]
    L = len(meta["frames"]) / fps + 0.5
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", g["video_url"], "-t", str(L),
           "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    res = []
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) < W * H * 3 or len(res) >= len(meta["frames"]):
            break
        res.append(cross_lines(np.frombuffer(buf, np.uint8).reshape(H, W, 3)))
    proc.wait()
    os.makedirs(os.path.join(a.out, md), exist_ok=True)
    json.dump(dict(moment=meta["moment"], lines=res), open(os.path.join(a.out, md, "lines.json"), "w"))
    print(md, len(res), "frames", round(float(np.mean([len(x) for x in res])) if res else 0, 1), "segments/frame", flush=True)
