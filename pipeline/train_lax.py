"""Train a small YOLO11 lacrosse detector (Goalie, Longpole, Referee, Shortstick) on CPU with a time cap.

Data: the lacrosse-object-detection set from Roboflow100-VL (README says License: MIT), pulled from the
Hugging Face mirror gatilin/rf100-vl so no Roboflow key is needed. COCO json is converted to YOLO txt.
"""
import argparse, json, os, shutil, time
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=3.0)
ap.add_argument("--imgsz", type=int, default=960)
a = ap.parse_args()

root = snapshot_download("gatilin/rf100-vl", repo_type="dataset",
                         allow_patterns=["lacrosse-object-detection/*"], local_dir="work/rf100")
src = os.path.join(root, "lacrosse-object-detection")
print(sorted(os.listdir(src)))
dst = os.path.abspath("work/laxdata")
names = None
counts = {}
for split, ysplit in [("train", "train"), ("valid", "val"), ("test", "test")]:
    d = os.path.join(src, split)
    if not os.path.isdir(d):
        continue
    coco = json.load(open(os.path.join(d, "_annotations.coco.json")))
    cats = sorted([c for c in coco["categories"] if c.get("supercategory") != "none" or len(coco["categories"]) == 1],
                  key=lambda c: c["id"])
    if names is None:
        names = [c["name"] for c in cats]
    cid = {c["id"]: names.index(c["name"]) for c in cats if c["name"] in names}
    imgs = {im["id"]: im for im in coco["images"]}
    os.makedirs(f"{dst}/images/{ysplit}", exist_ok=True)
    os.makedirs(f"{dst}/labels/{ysplit}", exist_ok=True)
    lines = {i: [] for i in imgs}
    for an in coco["annotations"]:
        if an["category_id"] not in cid:
            continue
        im = imgs[an["image_id"]]
        x, y, w, h = an["bbox"]
        lines[an["image_id"]].append(f"{cid[an['category_id']]} {(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
                                     f"{w / im['width']:.6f} {h / im['height']:.6f}")
    for i, im in imgs.items():
        shutil.copy(os.path.join(d, im["file_name"]), f"{dst}/images/{ysplit}/{im['file_name']}")
        open(f"{dst}/labels/{ysplit}/{os.path.splitext(im['file_name'])[0]}.txt", "w").write("\n".join(lines[i]))
    counts[ysplit] = len(imgs)
print("classes", names, "images", counts, flush=True)
yaml = f"path: {dst}\ntrain: images/train\nval: images/val\nnames: {json.dumps(dict(enumerate(names)))}\n"
open("work/lax.yaml", "w").write(yaml)

from ultralytics import YOLO
m = YOLO("yolo11n.pt")
m.train(data="work/lax.yaml", imgsz=a.imgsz, epochs=300, time=a.hours, batch=8, device="cpu", workers=2,
        project=os.path.abspath("work/runs"), name="lax", exist_ok=True, patience=40, plots=False)
best = os.path.abspath("work/runs/lax/weights/best.pt")
os.makedirs("models", exist_ok=True)
shutil.copy(best, "models/lax.pt")
mb = YOLO(best)
r = mb.val(data="work/lax.yaml", imgsz=a.imgsz, device="cpu", split="val", plots=False)
mb.export(format="onnx", imgsz=a.imgsz)
shutil.copy(best.replace(".pt", ".onnx"), "models/lax.onnx")
per = {names[int(c)]: round(float(v), 3) for c, v in zip(r.box.ap_class_index, r.box.ap50)}
open("models/lax.yaml", "w").write(
    f"dataset: Roboflow100-VL lacrosse-object-detection (License: MIT), via HF mirror gatilin/rf100-vl\n"
    f"images: {json.dumps(counts)}\nimgsz: {a.imgsz}\nmap50: {r.box.map50:.3f}\nmap50_95: {r.box.map:.3f}\n"
    f"per_class_map50: {json.dumps(per)}\n")
print(open("models/lax.yaml").read())
