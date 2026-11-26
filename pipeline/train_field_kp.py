"""Train the lacrosse field-landmark heatmap detector on synthetic data from gen_synth_field.

Default model is a small U-Net-style heatmap regressor (one channel per landmark) that is light enough to smoke-
train on a CPU Actions runner; for production accuracy swap in the vendored HRNet (pnl/cls_hrnet) and finetune from
PnLCalib's soccer weights on a GPU runner (same data, same targets). The detector's peaks become the named
landmarks fed to field_cal.calibrate.
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn as nn
import cv2


class DS(torch.utils.data.Dataset):
    def __init__(self, root):
        self.f = sorted(glob.glob(os.path.join(root, "[0-9]*.jpg")))
        self.root = root

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        im = cv2.imread(self.f[i]).astype(np.float32) / 255.0
        im = cv2.resize(im, (224, 128))             # divisible by 4 for the U-Net skips
        d = np.load(self.f[i][:-4] + ".npz")
        hm = d["hm"].astype(np.float32)             # (C, 135, 240)
        hm = np.stack([cv2.resize(h, (224, 128)) for h in hm])
        return torch.from_numpy(im.transpose(2, 0, 1)), torch.from_numpy(hm)


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.d1 = conv(3, 32); self.d2 = conv(32, 64); self.d3 = conv(64, 128)
        self.p = nn.MaxPool2d(2); self.u = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c2 = conv(128 + 64, 64); self.c1 = conv(64 + 32, 32)
        self.out = nn.Conv2d(32, nc, 1)

    def forward(self, x):
        a = self.d1(x); b = self.d2(self.p(a)); c = self.d3(self.p(b))
        y = self.c2(torch.cat([self.u(c), b], 1))
        y = self.c1(torch.cat([self.u(y), a], 1))
        return self.out(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    nc = len(meta["names"])
    ds = DS(a.data)
    if a.limit:
        ds.f = ds.f[:a.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
    net = UNet(nc).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.MSELoss()
    print(f"device {dev}  frames {len(ds)}  channels {nc}", flush=True)
    for ep in range(a.epochs):
        net.train(); tot = 0
        for im, hm in dl:
            im, hm = im.to(dev), hm.to(dev)
            opt.zero_grad(); p = net(im); loss = lossf(p, hm); loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep} loss {tot / max(1, len(dl)):.5f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"model": net.state_dict(), "names": meta["names"]}, os.path.join(a.out, "field_kp.pt"))
    print("saved", os.path.join(a.out, "field_kp.pt"))
