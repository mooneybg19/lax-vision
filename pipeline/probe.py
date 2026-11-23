"""Check each game's video link and pull a few sample frames, without downloading the whole file."""
import glob, json, os, subprocess, sys

out = sys.argv[1] if len(sys.argv) > 1 else "results/probe"
os.makedirs(out, exist_ok=True)
report = {}
for path in sorted(glob.glob("games/*.json")):
    name = os.path.basename(path)[:-5]
    g = json.load(open(path))
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_name,r_frame_rate",
                        "-of", "json", g["video_url"]], capture_output=True, text=True)
    info = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else {"error": p.stderr[-300:]}
    report[name] = info
    print(name, json.dumps(info)[:300], flush=True)
    dur = g.get("duration_s", 3000)
    for i in range(1, 9):
        t = int(dur * i / 9)
        dst = os.path.join(out, f"{name}_t{t:05d}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", g["video_url"], "-frames:v", "1",
                        "-vf", "scale=1280:-2", "-q:v", "3", "-y", dst], check=False)
json.dump(report, open(os.path.join(out, "probe.json"), "w"), indent=1)
