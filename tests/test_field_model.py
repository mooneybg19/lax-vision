import os, sys
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import field_targets as FT
import field_model as FM
from test_field_targets import doc_for

WEIGHTS = os.environ.get("PNL_WEIGHTS", "/home/claude/weights")


def test_gpu_targets_equal_cpu_render():
    for s, fb, flip in ((1, False, False), (2, True, True), (3, True, False)):
        d, _ = doc_for(60 + s, fb=fb)
        st = FT.sparse_targets(d, flip=flip)
        batch = {"kp_xy": torch.from_numpy(st["kp_xy"]).float()[None], "ln_xy": torch.from_numpy(st["ln_xy"]).float()[None],
                 "border": torch.from_numpy(st["border"])[None]}
        kp, ln = FM.build_targets(batch)
        r = FT.render(d, flip=flip)
        assert np.abs(kp[0].numpy() - r["kp"]).max() < 1e-5
        assert np.abs(ln[0].numpy() - r["ln"]).max() < 1e-5


def test_masked_softmax_gives_absent_channels_no_gradient():
    torch.manual_seed(0)
    C = FT.N_KP + 1
    logits = torch.randn(2, C, 12, 20, requires_grad=True)
    mask = torch.ones(2, C, dtype=torch.bool)
    mask[0, :FT.N_KP] = torch.from_numpy(~FT.KP_IS_FB)          # frame 0: lacrosse-only field
    target = torch.zeros(2, C, 12, 20); target[:, -1] = 1
    target[0, 3, 5, 5] = 1; target[0, -1, 5, 5] = 0
    loss = FM.kp_loss(logits, target, mask)
    loss.backward()
    g = logits.grad
    absent = torch.nonzero(~mask[0]).ravel()
    assert absent.numel() == int(FT.KP_IS_FB.sum())
    assert g[0, absent].abs().max().item() == 0.0                # not trained as negatives
    assert g[0, ~torch.from_numpy(np.r_[FT.KP_IS_FB, False])].abs().max().item() > 0
    p = FM.kp_probs(logits, mask)
    assert torch.allclose(p[0].sum(0), torch.ones(12, 20)) and p[0, absent].abs().max() == 0
    # line loss: masked channels get no gradient either
    lg = torch.randn(1, FT.N_LINE + 1, 6, 8, requires_grad=True)
    lm = torch.ones(1, FT.N_LINE + 1, dtype=torch.bool); lm[0, :FT.N_LINE] = torch.from_numpy(~FT.LINE_IS_FB)
    FM.line_loss(lg, torch.zeros_like(lg), lm).backward()
    assert lg.grad[0, torch.from_numpy(np.r_[FT.LINE_IS_FB, False])].abs().max() == 0


@pytest.mark.skipif(not os.path.exists(os.path.join(WEIGHTS, "SV_kp")), reason="soccer weights not downloaded")
def test_build_from_soccer_weights_keeps_encoder_and_resets_head():
    sd = torch.load(os.path.join(WEIGHTS, "SV_kp"), map_location="cpu", weights_only=False)
    net = FM.build("kp", os.path.join(WEIGHTS, "SV_kp"), verbose=False)
    st = net.state_dict()
    for k in ("conv1.weight", "stage4.2.fuse_layers.3.0.0.0.weight", "head.0.0.weight", "head.0.1.running_var"):
        assert torch.equal(st[k], sd[k]), k
    assert st["head.0.3.weight"].shape == (FT.N_KP + 1, 784, 1, 1)
    assert isinstance(net.head[0][4], torch.nn.Identity)
    netl = FM.build("line", os.path.join(WEIGHTS, "SV_lines"), verbose=False)
    sdl = torch.load(os.path.join(WEIGHTS, "SV_lines"), map_location="cpu", weights_only=False)
    assert torch.equal(netl.state_dict()["stage3.0.branches.0.0.conv1.weight"], sdl["stage3.0.branches.0.0.conv1.weight"])
    assert netl.state_dict()["head.0.3.weight"].shape[0] == FT.N_LINE + 1
    # freezing
    FM.set_frozen(net, True)
    fro = [n for n, p in net.named_parameters() if not p.requires_grad]
    assert fro and all(n.startswith(FM.FROZEN_PREFIXES) for n in fro)
    assert any(p.requires_grad for n, p in net.named_parameters() if n.startswith("stage3."))
    FM.train_mode(net)
    assert not net.layer1[0].bn1.training and net.stage3[0].branches[0][0].bn1.training
    FM.set_frozen(net, False)
    assert all(p.requires_grad for p in net.parameters())
    # output resolution = input / 2
    with torch.no_grad():
        net.eval()
        y = net(torch.rand(1, 3, 132, 240))        # (540x960 in training; any size with H/4 even works)
    assert tuple(y.shape) == (1, FT.N_KP + 1, 66, 120)
