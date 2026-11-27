"""HRNetV2-w48 field detectors finetuned from PnLCalib's soccer weights (NBJW layout: two separate nets).

  kp net    pnl/cls_hrnet + SV_kp (57 soccer keypoints + background, softmax over channels). The final 1x1 conv is
            re-initialised to N_KP + 1 channels (field_targets.KP_NAMES + background); everything else is kept.
            The softmax is applied in the loss / at inference over the channels that EXIST for the frame's field
            type only (masked softmax): absent landmarks get no gradient at all, so they are not trained as negatives.
  line net  same backbone + SV_lines (23 soccer lines + all-lines channel, sigmoid). Final conv re-initialised to
            N_LINE + 1 channels; sigmoid per channel; masked MSE.
Input: RGB float in [0, 1], 960x540, no mean/std normalisation (as NBJW). Output: 480x270 logits.

Weights: https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp and .../SV_lines (CC BY 4.0, per the
PnLCalib README). Code (pnl/) is GPL-2.0-or-later.
"""
import os
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

import field_targets as FT
from pnl.cls_hrnet import get_cls_net

HERE = os.path.dirname(os.path.abspath(__file__))
SV_KP_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp"
SV_LINES_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines"
FROZEN_PREFIXES = ("conv1.", "bn1.", "conv2.", "bn2.", "layer1.", "transition1.", "stage2.")   # stages 1-2


def _cfg(num_joints):
    cfg = yaml.safe_load(open(os.path.join(HERE, "pnl", "hrnetv2_w48.yaml")))
    cfg = copy.deepcopy(cfg); cfg["MODEL"]["NUM_JOINTS"] = num_joints
    return cfg


def build(kind, pretrained=None, verbose=True):
    """kind 'kp' or 'line'. pretrained: path to SV_kp / SV_lines (soccer) or None (random init, for tests)."""
    soccer_out = 58 if kind == "kp" else 24
    n_out = FT.N_KP + 1 if kind == "kp" else FT.N_LINE + 1
    net = get_cls_net(_cfg(soccer_out))           # kaiming init; pretrained loaded below
    if pretrained:
        sd = torch.load(pretrained, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        if verbose:
            print(f"loaded soccer {kind} weights from {pretrained}: {len(sd)} tensors, all keys matched")
    head = net.head[0]
    old = head[3]
    new = nn.Conv2d(old.in_channels, n_out, kernel_size=old.kernel_size, padding=old.padding)
    nn.init.normal_(new.weight, std=0.001); nn.init.constant_(new.bias, 0.0)
    head[3] = new
    head[4] = nn.Identity()                       # was Softmax (kp) / Sigmoid (line): applied in loss/inference
    net.kind = kind
    return net


def load_trained(path, device="cpu"):
    """Load a checkpoint written by train_field.py: {'kind', 'model', 'kp_names', 'line_names', ...}."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build(ck["kind"], pretrained=None, verbose=False)
    net.load_state_dict(ck["model"])
    names = ck["kp_names"] if ck["kind"] == "kp" else ck["line_names"]
    expect = FT.KP_NAMES if ck["kind"] == "kp" else FT.LINE_NAMES
    if list(names) != list(expect):
        raise ValueError("checkpoint channel list differs from field_targets: retrain or pin the code version")
    return net.to(device).eval(), ck


def set_frozen(net, frozen):
    """Freeze/unfreeze stages 1-2 (stem, layer1, transition1, stage2). Frozen BN layers stay in eval mode."""
    for n, p in net.named_parameters():
        if n.startswith(FROZEN_PREFIXES):
            p.requires_grad = not frozen
    net._frozen = frozen


def train_mode(net):
    net.train()
    if getattr(net, "_frozen", False):
        for n, m in net.named_modules():
            if n.startswith(tuple(p.rstrip(".") for p in FROZEN_PREFIXES)) and isinstance(m, nn.BatchNorm2d):
                m.eval()


# ------------------------------------------------------------------ probabilities, losses
def kp_probs(logits, mask):
    """Masked softmax over channels. logits (B,C,h,w), mask (B,C) bool -> probs, exactly 0 on masked channels."""
    lg = logits.float().masked_fill(~mask[:, :, None, None], -1e4)
    p = torch.softmax(lg, dim=1)
    return p * mask[:, :, None, None]


def kp_loss(logits, target, mask, pos_weight=10.0):
    """MSE between masked-softmax probabilities and the Gaussian targets on the existing channels only, weighted
    (1 + pos_weight * target) so the few peak pixels are not drowned by the background (cf. NBJW's
    HeatmapWeightingMSELoss)."""
    p = kp_probs(logits, mask)
    w = (1 + pos_weight * target) * mask[:, :, None, None]
    return ((p - target) ** 2 * w).sum() / (mask.sum() * p.shape[-1] * p.shape[-2]).clamp(min=1)


def line_loss(logits, target, mask, pos_weight=10.0):
    p = torch.sigmoid(logits.float())
    w = (1 + pos_weight * target) * mask[:, :, None, None]
    return ((p - target) ** 2 * w).sum() / (mask.sum() * p.shape[-1] * p.shape[-2]).clamp(min=1)


# ------------------------------------------------------------------ targets on the GPU from point lists
def _gauss_grid(pts, h, w, sigma):
    """pts (B,C,2) output coords (NaN = none) -> (B,C,h,w) Gaussians, truncated to the same window as
    field_targets._gauss (floor(x)-r .. floor(x)+r+1, r = ceil(3 sigma)) so CPU and GPU targets are identical."""
    B, C, _ = pts.shape
    dev = pts.device
    xs = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    ys = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    ok = torch.isfinite(pts[..., 0])
    x = torch.where(ok, pts[..., 0], torch.full_like(pts[..., 0], -1e4)).view(B, C, 1, 1)
    y = torch.where(ok, pts[..., 1], torch.full_like(pts[..., 1], -1e4)).view(B, C, 1, 1)
    r = float(torch.ceil(torch.tensor(3 * sigma)))
    fx, fy = torch.floor(x), torch.floor(y)
    win = (xs >= fx - r) & (xs <= fx + r + 1) & (ys >= fy - r) & (ys <= fy + r + 1)
    g = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)) * win
    return g * ok.view(B, C, 1, 1)


def build_targets(batch, h=FT.OUT_H, w=FT.OUT_W, sigma=FT.SIGMA):
    """From the dataset's sparse targets (on any device) to dense heatmaps:
       kp (B, N_KP+1, h, w) with background = clip(1 - sum), ln (B, N_LINE+1, h, w)."""
    kp = _gauss_grid(batch["kp_xy"], h, w, sigma)
    bg = (1 - kp.sum(1, keepdim=True)).clamp(0, 1)
    kp = torch.cat([kp, bg], 1)
    e = batch["ln_xy"]                                   # (B, N_LINE, 2, 2)
    ln = torch.maximum(_gauss_grid(e[:, :, 0], h, w, sigma), _gauss_grid(e[:, :, 1], h, w, sigma))
    ln = torch.cat([ln, batch["border"].float().unsqueeze(1)], 1)
    return kp, ln
