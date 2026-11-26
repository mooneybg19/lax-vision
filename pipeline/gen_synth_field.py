"""Synthetic training data for the lacrosse field-landmark detector (no hand labelling).

Renders the lacrosse field markings (lax_field) under randomised NORMAL elevated-sideline camera poses — the
range a phone/tripod/press-box camera actually sees — onto turf, with lighting, noise and player-like
distractors, and writes each frame plus a keypoint-heatmap target (one channel per named landmark). Finetune
the vendored HRNet (pnl/cls_hrnet) on this, then optionally mix in auto-labelled real frames (frames where the
engine's existing fit is confident, with the lacrosse landmarks projected in) to close the domain gap.

Run bounded in GitHub Actions; this file is CPU-only and deterministic per seed.
"""
import argparse, os, json
import numpy as np
import cv2
import lax_field as LF

KP = LF.KP_WORLD            # (N,2) world yd
NAMES = LF.KP_NAMES


def random_camera(rng, W, H):
    """A camera above/near a sideline looking across the field — the normal-film range."""
    side = rng.choice([-1, 1])                       # which sideline the camera is behind
    cam = np.array([rng.uniform(10, 100), (30 + side * rng.uniform(34, 70)), rng.uniform(4, 16)])
    aim = np.array([rng.uniform(20, 90), 30 + rng.uniform(-10, 10), 0.0])
    f = rng.uniform(800, 2200)
    fwd = aim - cam; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd])
    K = np.array([[f, 0, W / 2 + rng.uniform(-40, 40)], [0, f, H / 2 + rng.uniform(-30, 30)], [0, 0, 1.0]])
    return K, R, cam


def world_to_img_H(K, R, cam):
    """Ground-plane (z=0) world->image homography."""
    Rt = np.c_[R[:, 0], R[:, 1], (-R @ cam)]
    return K @ Rt


def turf(rng, W, H):
    base = np.array([rng.uniform(30, 70), rng.uniform(90, 150), rng.uniform(40, 90)])   # BGR green-ish
    img = np.ones((H, W, 3), np.float32) * base
    img += rng.normal(0, 6, (H, W, 3))
    # mowing stripes
    for _ in range(rng.integers(0, 3)):
        y0 = rng.integers(0, H); img[y0:y0 + rng.integers(20, 80)] *= rng.uniform(0.9, 1.1)
    g = rng.uniform(0.6, 1.3)
    img *= np.linspace(g, 2 - g, W)[None, :, None]   # lighting gradient
    return img


def draw_players(img, rng):
    H, W = img.shape[:2]
    for _ in range(rng.integers(6, 24)):
        x, y = rng.integers(0, W), rng.integers(int(H * 0.3), H)
        w, h = rng.integers(6, 22), rng.integers(20, 60)
        c = rng.integers(0, 255, 3).tolist()
        cv2.rectangle(img, (x, y - h), (x + w, y), c, -1)


def heatmaps(Himg, W, H, out_hw, sigma=2.5):
    oh, ow = out_hw
    P = (Himg @ np.c_[KP, np.ones(len(KP))].T).T
    vis = P[:, 2] > 0.1
    uv = np.full((len(KP), 2), -1.0)
    uv[vis] = P[vis, :2] / P[vis, 2:3]
    hm = np.zeros((len(KP), oh, ow), np.float32)
    ys, xs = np.mgrid[0:oh, 0:ow]
    for i, (u, v) in enumerate(uv):
        if not vis[i] or not (0 <= u < W and 0 <= v < H):
            continue
        cu, cv_ = u * ow / W, v * oh / H
        hm[i] = np.exp(-((xs - cu) ** 2 + (ys - cv_) ** 2) / (2 * sigma ** 2))
    return hm, uv, vis


def render(rng, W=960, H=540, out_hw=(135, 240)):
    K, R, cam = random_camera(rng, W, H)
    Himg = world_to_img_H(K, R, cam)
    img = turf(rng, W, H)
    LF.draw(img, Himg, color=(235, 235, 235), thick=rng.integers(2, 4))
    if rng.random() < 0.9:
        draw_players(img, rng)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.2))
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    hm, uv, vis = heatmaps(Himg, W, H, out_hw)
    return img, hm, uv, vis, Himg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="also write N annotated preview jpgs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    meta = {"names": NAMES, "n": a.n, "img_hw": [540, 960], "hm_hw": [135, 240]}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    for i in range(a.n):
        img, hm, uv, vis, _ = render(rng)
        cv2.imwrite(os.path.join(a.out, f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        np.savez_compressed(os.path.join(a.out, f"{i:06d}.npz"), hm=hm.astype(np.float16), uv=uv.astype(np.float32), vis=vis)
        if i < a.preview:
            p = img.copy()
            for (u, v), ok in zip(uv, vis):
                if ok and 0 <= u < 960 and 0 <= v < 540:
                    cv2.circle(p, (int(u), int(v)), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(a.out, f"preview_{i:06d}.jpg"), p)
        if i % 200 == 0:
            print("rendered", i, flush=True)
    print("done", a.n)
