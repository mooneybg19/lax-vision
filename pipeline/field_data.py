"""Training data: frames + corrected labels -> (image tensor, sparse targets), with augmentation, and the splits.

Augmentation (train only): horizontal flip p=0.5 (targets mirrored with the Left<->Right channel permutation,
near/far unchanged: field_targets.sparse_targets(flip=True)), colour jitter (brightness, contrast, saturation, small
hue), Gaussian blur, JPEG re-compression, sensor noise. No geometric warps other than the flip: the labels carry a
lens model, and a crop/scale would move the lens centre.
"""
import hashlib
import numpy as np
import cv2
import torch

import field_targets as FT

HOLDOUT_GAME = "jesuit-westview"
DEV_GAME = "scituate-hanover-2024"


def _u(frame, salt):
    """Stable pseudo-random number in [0,1) from a frame id (independent of list order / Python hash seed)."""
    return int(hashlib.sha1((salt + frame).encode()).hexdigest()[:8], 16) / 2 ** 32


def split(docs, holdout_game=HOLDOUT_GAME, dev_game=DEV_GAME, dev_test_frac=0.15, val_frac=0.10):
    """test = every frame of holdout_game + ~dev_test_frac of dev_game's frames; val = ~val_frac of the rest
    (for picking the best epoch only); train = the remainder. Deterministic per frame id."""
    tr, va, te = [], [], []
    for d in docs:
        f = d["frame"]
        if d["game"] == holdout_game or (d["game"] == dev_game and _u(f, "test") < dev_test_frac):
            te.append(d)
        elif _u(f, "val") < val_frac:
            va.append(d)
        else:
            tr.append(d)
    return tr, va, te


def load_image(frames_dir, frame):
    im = cv2.imread(f"{frames_dir}/{frame}.jpg")
    if im is None:
        raise FileNotFoundError(f"{frames_dir}/{frame}.jpg")
    if im.shape[1] != FT.IMG_W or im.shape[0] != FT.IMG_H:
        raise ValueError(f"{frame}: frame is {im.shape[1]}x{im.shape[0]}, labels are {FT.IMG_W}x{FT.IMG_H}")
    return im


def to_input(img_bgr):
    """BGR uint8 1280x720 -> float tensor (3, 540, 960) RGB in [0,1] (NBJW input)."""
    im = cv2.resize(img_bgr, (FT.IN_W, FT.IN_H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float() / 255.0


def augment_color(im, rng):
    f = im.astype(np.float32)
    if rng.random() < 0.8:
        f = f * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)                       # brightness
        m = f.mean(); f = (f - m) * rng.uniform(0.75, 1.25) + m                       # contrast
        f = np.clip(f, 0, 255)
        hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.7, 1.3)                                         # saturation
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180                       # hue (+-8 deg)
        f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.3:
        f = f + rng.normal(0, rng.uniform(1, 6), f.shape).astype(np.float32)
    im = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return im


class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, docs, frames_dir, train=True, seed=0, cache_targets=True):
        self.docs, self.frames_dir, self.train, self.seed = list(docs), frames_dir, train, seed
        self.cache = {} if cache_targets else None
        self.epoch = 0

    def __len__(self):
        return len(self.docs)

    def _targets(self, i, flip):
        key = (i, flip)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        t = FT.sparse_targets(self.docs[i], flip=flip)
        t = {"kp_xy": t["kp_xy"].astype(np.float32), "kp_mask": t["kp_mask"], "ln_xy": t["ln_xy"].astype(np.float32),
             "ln_mask": t["ln_mask"], "border": t["border"].astype(np.float16)}
        if self.cache is not None:
            self.cache[key] = t
        return t

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        d = self.docs[i]
        im = load_image(self.frames_dir, d["frame"])
        flip = bool(self.train and rng.random() < 0.5)
        if flip:
            im = cv2.flip(im, 1)
        if self.train:
            im = augment_color(im, rng)
        t = self._targets(i, flip)
        return {"image": to_input(im), "kp_xy": torch.from_numpy(t["kp_xy"]), "kp_mask": torch.from_numpy(t["kp_mask"]),
                "ln_xy": torch.from_numpy(t["ln_xy"]), "ln_mask": torch.from_numpy(t["ln_mask"]),
                "border": torch.from_numpy(t["border"]), "index": i, "flip": flip}
