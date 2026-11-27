"""Pull a spread of real frames from each game for landmark labeling and real-film evaluation.

Times are seeded-random across 4%..96% of the game so we get varied zoom, pan and field end.
"""
import glob, json, os, random, subprocess, sys

out = sys.argv[1] if len(sys.argv) > 1 else "results/label_frames"
per_game = int(sys.argv[2]) if len(sys.argv) > 2 else 45
only = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
for path in sorted(glob.glob("games/*.json")):
    name = os.path.basename(path)[:-5]
    if only and name != only:
        continue
    g = json.load(open(path))
    url = g.get("web_video_url") or g["video_url"]
    dur = g.get("duration_s", 3000)
    rng = random.Random(name)
    times = sorted(rng.sample(range(int(dur * .04), int(dur * .96)), per_game))
    for t in times:
        dst = os.path.join(out, f"{name}_t{t:05d}.jpg")
        if os.path.exists(dst):
            continue
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", url, "-frames:v", "1",
                            "-vf", "scale=1280:-2", "-q:v", "2", "-y", dst], capture_output=True, text=True)
        print(name, t, "ok" if os.path.exists(dst) else ("FAIL " + r.stderr[-200:]), flush=True)
