"""Cut one short H.264 clip per settled-offense moment (start-2s to end+2s) from the game video."""
import json, os, subprocess, sys

game = json.load(open(sys.argv[1]))
moments = json.load(open(sys.argv[2]))["moments"]
out = sys.argv[3]
os.makedirs(out, exist_ok=True)
url = game.get("web_video_url") or game["video_url"]
index, failed = [], []
for i, m in enumerate(moments, 1):
    mid = f"m{i:02d}"
    s = max(0.0, m["start"] - 2)
    dur = (m["end"] + 2) - s
    dst = os.path.join(out, mid + ".mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.2f}", "-i", url, "-t", f"{dur:.2f}", "-an",
           "-vf", "scale=1280:-2,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    ok = False
    for _ in range(2):
        subprocess.run(cmd, check=False)
        if os.path.exists(dst) and os.path.getsize(dst) > 10000:
            ok = True
            break
    print(("ok " if ok else "FAILED ") + mid, flush=True)
    if not ok:
        failed.append(mid)
        continue
    index.append({"id": mid, "clip": mid + ".mp4", "clip_start": round(s, 2), "start": m["start"], "end": m["end"],
                  "set": m.get("set"), "set_goal_first": m.get("set_goal_first"), "confidence": m.get("confidence"),
                  "offense": m.get("offense"), "offense_side_agreement": m.get("offense_side_agreement")})
json.dump(index, open(os.path.join(out, "moments.json"), "w"), indent=1)
json.dump(failed, open(os.path.join(out, "failed.json"), "w"))
print("failed:", failed)
