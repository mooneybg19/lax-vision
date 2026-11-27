"""Glue: run the trained HRNet field detectors on a frame and calibrate (drop-in for field_geom.keyframe_h).

    H_eng, info = keyframe_h(frame_bgr, model_path, goal_in_view_x=15.0, near_sideline_is_y0=True)

Same interface as before: returns the image -> engine-field homography (X = yards from the in-view goal line toward
the field interior, Y = yards from the near sideline with the goal at Y = 26.67), or (None, reason) when the model is
missing or calibration fails. model_path is the kp checkpoint (kp_best.pt); a line checkpoint next to it
(line_best.pt) is used automatically for refinement when present.

The field type of a game (lacrosse 60 / 53.33 yd, or football lines with HS/college hashes) can be given with
field=...; by default every candidate is tried and the most consistent calibration wins (field="auto").
Images go in as BGR uint8 at any size; they are resized to 1280x720 first (the labels' frame size).
"""
import os
import numpy as np
import cv2
import torch

import labels_io as LI
import field_targets as FT
import field_model as FM
import field_calib as FC
from field_data import to_input

GOAL_Y_ENG = 160 / 3.0 / 2       # 26.67, the engine's goal Y (field_geom convention)
FIELD_CANDIDATES = {             # name: (width, hash, football lines, lacrosse lines painted)
    "lacrosse60": (60.0, LI.HASH_HS, False, True),
    "lacrosse53": (LI.FB_WIDTH, LI.HASH_HS, False, True),
    "football_hs": (LI.FB_WIDTH, LI.HASH_HS, True, True),
    "football_college": (LI.FB_WIDTH, LI.HASH_COLLEGE, True, True),
}
_CACHE = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class FieldDetector:
    def __init__(self, kp_path, line_path=None, device=None):
        self.device = device or _device()
        self.kp_net, self.kp_ck = FM.load_trained(kp_path, self.device)
        self.line_net = None
        if line_path and os.path.exists(line_path):
            self.line_net, _ = FM.load_trained(line_path, self.device)
        self.thresh = self.kp_ck.get("kp_thresh", 0.25)
        self.in_wh = tuple(self.kp_ck.get("in_wh", (FT.IN_W, FT.IN_H)))

    @torch.no_grad()
    def raw(self, frame_bgr):
        """Network outputs for one frame: kp logits (N_KP+1,h,w) and line probabilities (N_LINE+1,h,w) or None."""
        im = frame_bgr
        if im.shape[1] != FT.IMG_W or im.shape[0] != FT.IMG_H:
            im = cv2.resize(im, (FT.IMG_W, FT.IMG_H), interpolation=cv2.INTER_AREA)
        x = to_input(im, self.in_wh)[None].to(self.device)
        with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            kl = self.kp_net(x)[0].float()
            ln = torch.sigmoid(self.line_net(x)[0].float()) if self.line_net is not None else None
        return kl, ln

    def decode(self, kl, ln, fb, lax_painted=True, thresh=None):
        kpm, lnm = FT.masks(fb, lax_painted)
        p = FM.kp_probs(kl[None], torch.from_numpy(kpm)[None].to(kl.device))[0].cpu().numpy()
        kp = FT.decode_kp(p, kpm, thresh=self.thresh if thresh is None else thresh)
        lines, border = None, None
        if ln is not None:
            lnp = ln.cpu().numpy()
            lines = FT.decode_lines(lnp, lnm)
            border = lnp[FT.N_LINE]
        return kp, lines, border

    def calibrate(self, frame_bgr, field="auto", k1=0.0, lax_painted=True, use_lines=True):
        """-> (H world(yd)->image(1280x720 index px, undistorted if k1), info with info['field'])."""
        kl, ln = self.raw(frame_bgr)
        cands = FIELD_CANDIDATES if field == "auto" else {field: FIELD_CANDIDATES[field]} if isinstance(field, str) else {"given": field}
        best = (None, {"reason": "no candidate"})
        for name, (W, hashD, fb, lp) in cands.items():
            kp, lines, border = self.decode(kl, ln, fb, lax_painted and lp)
            H, info = FC.calibrate(kp, W, hashD, fb, lines=lines if use_lines else None,
                                   border=border if use_lines else None, k1=k1)
            info["field"] = name; info["width"] = W
            if H is None:
                if best[0] is None:
                    best = (None, info)
                continue
            score = (info["inliers"], -(info["reproj_px_median"] or 99))
            if best[0] is None or score > best[1]["_score"]:
                info["_score"] = score; best = (H, info)
        return best


def detector(model_path):
    if model_path not in _CACHE:
        line = os.path.join(os.path.dirname(model_path), "line_best.pt")
        _CACHE[model_path] = FieldDetector(model_path, line if os.path.exists(line) else None)
    return _CACHE[model_path]


def to_engine(H_w2i, W, goal_in_view_x, near_sideline_is_y0, image_scale=(1.0, 1.0)):
    """Absolute world(yd)->image(1280x720) H to the engine's image->field H (image at its own size: image_scale
    = (w/1280, h/720))."""
    S = np.diag([image_scale[0], image_scale[1], 1.0])
    H_i2w = np.linalg.inv(S @ H_w2i)
    a, bx = (1.0, -goal_in_view_x) if goal_in_view_x < 55 else (-1.0, goal_in_view_x)
    if near_sideline_is_y0:
        cy, dy = 1.0, GOAL_Y_ENG - W / 2
    else:
        cy, dy = -1.0, GOAL_Y_ENG + W / 2
    M = np.array([[a, 0, bx], [0, cy, dy], [0, 0, 1.0]])
    H = M @ H_i2w
    return H / H[2, 2]          # image -> field: (image points are always 'in front'; the sign is irrelevant here)


def keyframe_h(frame_bgr, model_path, goal_in_view_x=15.0, near_sideline_is_y0=True, field="auto", k1=0.0):
    """Signature parallels field_geom: returns (H_image->field_engine, info) or (None, reason)."""
    if not model_path or not os.path.exists(model_path):
        return None, "no trained field model"
    H, info = detector(model_path).calibrate(frame_bgr, field=field, k1=k1)
    if H is None:
        return None, info.get("reason", "calibration failed")
    sc = (frame_bgr.shape[1] / FT.IMG_W, frame_bgr.shape[0] / FT.IMG_H)
    info.pop("_score", None)
    return to_engine(H, info["width"], goal_in_view_x, near_sideline_is_y0, sc), info


def detect(frame_bgr, model_path, thresh=None, field="football_hs"):
    """{landmark: (u, v)} in frame pixels (compat with the old helper)."""
    d = detector(model_path)
    kl, ln = d.raw(frame_bgr)
    W, hashD, fb, lp = FIELD_CANDIDATES[field]
    kp, _, _ = d.decode(kl, ln, fb, lp, thresh)
    sx, sy = frame_bgr.shape[1] / FT.IMG_W, frame_bgr.shape[0] / FT.IMG_H
    return {n: ((u + 0.5) * sx - 0.5, (v + 0.5) * sy - 0.5) for n, (u, v, s) in kp.items()}
