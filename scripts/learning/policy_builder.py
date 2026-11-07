"""Variants of the official Diffusion Policy for the student experiments (2026-11-01).

build(image_size=96, down_dims=None, scratch_model=False) -> OfficialDiffusionPolicy whose .network is replaced by a
DiffusionUnetHybridImagePolicy with:
  - image_size: camera resolution (crop kept at 84/96 of it); every official weight loads, only the encoder's
    spatial-softmax position grids (pos_x/pos_y, computed from the feature-map size) are rebuilt;
  - down_dims: a wider/narrower action U-Net (e.g. [1024, 2048, 4096] ~ 1B parameters); its weights start from scratch;
  - scratch_model: start the action U-Net from scratch even at the official width (fair size comparisons).
The official checkpoint's size/hash checks still run first (OfficialDiffusionPolicy).
"""
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))


def arch_of(image_size=96, down_dims=None, scratch_model=False):
    return {"image_size": int(image_size), "down_dims": list(down_dims) if down_dims else None, "scratch_model": bool(scratch_model)}


def build(image_size=96, down_dims=None, scratch_model=False, device="cuda"):
    import sys
    sys.path.insert(0, str(ROOT))
    from tdyssey.official import OfficialDiffusionPolicy
    pol = OfficialDiffusionPolicy(ROOT/"third_party/diffusion_policy", ROOT/"checkpoints/official-dp/ema_policy.pt", device=device)
    if image_size == 96 and not down_dims and not scratch_model:
        return pol
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
    payload = torch.load(ROOT/"checkpoints/official-dp/ema_policy.pt", map_location="cpu", weights_only=True, mmap=True)
    config = dict(payload["policy_config"]); config.pop("_target_")
    sched = dict(config.pop("noise_scheduler")); sched.pop("_target_")
    config["shape_meta"] = {"obs": {"image": {"shape": [3, image_size, image_size], "type": "rgb"},
                                    "agent_pos": {"shape": [2], "type": "low_dim"}}, "action": {"shape": [2]}}
    config["crop_shape"] = [round(image_size*84/96)]*2
    if down_dims:
        config["down_dims"] = list(down_dims)
    net = DiffusionUnetHybridImagePolicy(noise_scheduler=DDPMScheduler(**sched), **config)
    old = payload["state_dict"]; new = net.state_dict()
    keep = {}
    for k, v in old.items():
        if k.startswith("model.") and (down_dims or scratch_model):
            continue                                                      # new action U-Net starts from scratch
        if k in new and new[k].shape == v.shape:
            keep[k] = v
    missing = [k for k in new if k not in keep]
    allowed = [k for k in missing if k.startswith("model.") or k.endswith("pos_x") or k.endswith("pos_y") or k.startswith("normalizer.")]
    assert len(allowed) == len(missing), f"unexpected missing weights: {[k for k in missing if k not in allowed][:5]}"
    net.load_state_dict(keep, strict=False)
    net.set_normalizer(pol.network.normalizer)                          # the official action/position scaling
    net.eval().requires_grad_(False).to(pol.device)
    pol.network = net
    pol.arch = arch_of(image_size, down_dims, scratch_model)
    return pol
