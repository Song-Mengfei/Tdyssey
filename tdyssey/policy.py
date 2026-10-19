"""Pinned, unranked GPC diffusion-policy inference; no imports/install on module load."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def policy_spec():
    value = json.loads((ROOT / "manifests/gpc_policy.json").read_text())
    config = value["policy"]
    validate_horizons(config)
    return value


def validate_horizons(config):
    for key in ("obs_horizon", "pred_horizon", "action_horizon", "num_diffusion_iters"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["obs_horizon"] - 1 + config["action_horizon"] > config["pred_horizon"]:
        raise ValueError("Executed action prefix exceeds prediction horizon")


def action_prefix(actions, config, remaining_steps):
    validate_horizons(config)
    if type(remaining_steps) is not int or remaining_steps < 1:
        raise ValueError("Remaining control-step budget must be positive")
    if len(actions) != config["pred_horizon"]:
        raise ValueError("Unexpected action prediction horizon")
    start = config["obs_horizon"] - 1
    return actions[start:start + min(config["action_horizon"], remaining_steps)]


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_model_source(gpc_root, spec):
    root = Path(gpc_root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["source_commit"]:
        raise ValueError("GPC source revision differs from the pinned policy")
    for name, expected in spec["source_sha256"].items():
        if hash_file(root / name) != expected:
            raise ValueError(f"Missing/modified policy source: {name}")
    path = root / "gpc_rank_evaluation/models.py"
    module_spec = importlib.util.spec_from_file_location("tdyssey_gpc_policy_models", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def load_ema_parameters(network, payload):
    """EMA holds every parameter in the upstream ModuleDict order, not names.

    Check all shapes before copying. Non-parameter buffers would require component
    state dictionaries, so refuse this compact loading path if any are present.
    """
    import torch

    if list(network.named_buffers()):
        raise ValueError("EMA-only loading requires a network without buffers")
    shadows = payload.get("shadow_params")
    parameters = list(network.parameters())
    if not isinstance(shadows, (list, tuple)) or len(shadows) != len(parameters):
        raise ValueError("EMA parameter count differs from the policy architecture")
    for parameter, shadow in zip(parameters, shadows):
        if (not isinstance(shadow, torch.Tensor) or shadow.shape != parameter.shape
                or shadow.dtype != parameter.dtype or not torch.isfinite(shadow).all()):
            raise ValueError("EMA tensor shape/dtype/values differ from the policy architecture")
    with torch.no_grad():
        for parameter, shadow in zip(parameters, shadows):
            parameter.copy_(shadow)


class GPCDiffusionPolicy:
    def __init__(self, gpc_root, checkpoint, device="cpu"):
        import torch
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.spec, self.device = policy_spec(), torch.device(device)
        asset, config = self.spec["checkpoint"], self.spec["policy"]
        checkpoint = Path(checkpoint)
        if checkpoint.stat().st_size != asset["bytes"] or hash_file(checkpoint) != asset["sha256"]:
            raise ValueError("Policy checkpoint differs from pinned size/hash")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use the installed CPU profile")
        module = load_model_source(gpc_root, self.spec)
        encoder = module.replace_bn_with_gn(module.get_resnet())
        # Preserve exact upstream insertion/parameter order for the EMA shadows.
        self.network = torch.nn.ModuleDict({
            "vision_encoder": encoder,
            "invariant": module.ConditionalUnet1D(input_dim=2, global_cond_dim=514 * config["obs_horizon"]),
        }).float()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        load_ema_parameters(self.network, payload)
        self.ema_step = payload.get("optimization_step")
        del payload
        self.network.to(self.device).eval().requires_grad_(False)
        self.scheduler = DDPMScheduler(num_train_timesteps=config["num_diffusion_iters"], **config["scheduler"])
        self.generator = None
        signature = [(name, list(parameter.shape)) for name, parameter in self.network.named_parameters()]
        self.parameter_signature = hashlib.sha256(json.dumps(signature).encode()).hexdigest()

    def reset_seed(self, seed):
        import torch

        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Policy seed must be uint32")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def predict(self, history):
        import numpy as np
        import torch

        config = self.spec["policy"]
        if len(history) != config["obs_horizon"] or self.generator is None:
            raise ValueError("Provide the full observation history and reset the policy RNG first")
        images = np.stack([obs["image"] for obs in history])
        positions = np.stack([obs["agent_pos"] for obs in history])
        if (images.shape != (config["obs_horizon"], *config["image_shape"])
                or positions.shape != (config["obs_horizon"], 2)
                or not np.isfinite(images).all() or not np.isfinite(positions).all()
                or images.min() < 0 or images.max() > 1):
            raise ValueError("Unexpected policy image/proprioception format")
        stats = config["normalization"]
        low = np.asarray(stats["agent_pos"]["min"], dtype=np.float32)
        high = np.asarray(stats["agent_pos"]["max"], dtype=np.float32)
        normalized = (positions - low) / (high - low) * 2 - 1
        with torch.inference_mode():
            features = self.network["vision_encoder"](torch.as_tensor(images, dtype=torch.float32, device=self.device))
            condition = torch.cat((features, torch.as_tensor(normalized, dtype=torch.float32, device=self.device)), dim=-1).unsqueeze(0).flatten(1)
            actions = torch.randn((1, config["pred_horizon"], 2), device=self.device, generator=self.generator)
            self.scheduler.set_timesteps(config["num_diffusion_iters"], device=self.device)
            for timestep in self.scheduler.timesteps:
                noise = self.network["invariant"](sample=actions, timestep=timestep, global_cond=condition)
                actions = self.scheduler.step(noise, timestep, actions, generator=self.generator).prev_sample
            result = actions[0].cpu().numpy()
        low = np.asarray(stats["action"]["min"], dtype=np.float32)
        high = np.asarray(stats["action"]["max"], dtype=np.float32)
        result = (result + 1) / 2 * (high - low) + low
        if not np.isfinite(result).all():
            raise ValueError("Policy produced non-finite actions")
        return result
