"""Glue: run the trained lacrosse field-landmark detector on a frame and calibrate.

Drop-in replacement for field_geom.keyframe_h: returns the image->field (engine convention) homography and info,
from the learned landmarks fed through field_cal. If no model is present it raises, so the engine keeps using the
football-grid fit (field_geom) as the labelled fallback until a trained model exists.
"""
import os
import numpy as np
import cv2
import torch
import field_cal as FC
from train_field_kp import UNet

_CACHE = {}


def load(model_path):
    if model_path not in _CACHE:
        ck = torch.load(model_path, map_location="cpu")
        net = UNet(len(ck["names"]))
        net.load_state_dict(ck["model"]); net.eval()
        _CACHE[model_path] = (net, ck["names"])
    return _CACHE[model_path]


def detect(frame_bgr, model_path, thresh=0.35):
    """Return {landmark_name: (u,v)} in frame pixels for peaks above threshold."""
    net, names = load(model_path)
    H, W = frame_bgr.shape[:2]
    im = cv2.resize(frame_bgr.astype(np.float32) / 255.0, (224, 128)).transpose(2, 0, 1)
    with torch.no_grad():
        hm = net(torch.from_numpy(im[None]))[0].numpy()     # (C,272,480)
    out = {}
    for c, nm in enumerate(names):
        h = hm[c]
        j = int(h.argmax()); v, u = divmod(j, h.shape[1])
        if h[v, u] < thresh:
            continue
        out[nm] = (u * W / h.shape[1], v * H / h.shape[0])
    return out


def keyframe_h(frame_bgr, model_path, goal_in_view_x=15.0, near_sideline_is_y0=True):
    """Signature parallels field_geom: returns (H_image->field_engine, info) or (None, reason)."""
    if not model_path or not os.path.exists(model_path):
        return None, "no trained field model"
    kp = detect(frame_bgr, model_path)
    H, info = FC.calibrate(kp, (frame_bgr.shape[1], frame_bgr.shape[0]),
                           goal_in_view_x=goal_in_view_x, near_sideline_is_y0=near_sideline_is_y0)
    if H is None:
        return None, info.get("reason", "calibration failed")
    return H, info
