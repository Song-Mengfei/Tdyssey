"""Official RSS Diffusion Policy source and CNN checkpoint, separate from GPC."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

from .policy import hash_file

ROOT = Path(__file__).resolve().parents[1]


def official_spec():
    return json.loads((ROOT / "manifests/official_dp.json").read_text())


def load_official_source(source_root, spec=None):
    spec = spec or official_spec()
    root = Path(source_root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["source_commit"]:
        raise ValueError("Official Diffusion Policy source revision differs from the manifest")
    for name, expected in spec["source_sha256"].items():
        if hash_file(root / name) != expected:
            raise ValueError(f"Modified/missing official source: {name}")
    existing = sys.modules.get("diffusion_policy")
    if existing and {Path(p).resolve() for p in getattr(existing, "__path__", ())} != {root / "diffusion_policy"}:
        raise ValueError("Another diffusion_policy package is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("diffusion_policy.env.pusht.pusht_image_env").PushTImageEnv


class OfficialDiffusionPolicy:
    """Instantiate the unchanged official policy class; load its complete EMA state.

    The compact asset retains encoder, denoising network, normalizers, and buffers.
    No GPC model, preprocessing, or normalization statistics are involved.
    """

    def __init__(self, source_root, checkpoint, device="cuda"):
        import torch
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.spec, self.device = official_spec(), torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; no automatic CPU fallback is used")
        checkpoint = Path(checkpoint)
        asset = self.spec["compact"]
        if checkpoint.stat().st_size != asset["bytes"] or hash_file(checkpoint) != asset["sha256"]:
            raise ValueError("Official compact checkpoint differs from its pinned size/hash")
        load_official_source(source_root, self.spec)
        from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if (payload["source_checkpoint_sha256"] != self.spec["checkpoint"]["sha256"]
                or payload["policy_config"] != self.spec["policy_config"]):
            raise ValueError("Compact asset provenance/configuration differs from the official manifest")
        config = dict(payload["policy_config"])
        if config.pop("_target_") != "diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy":
            raise ValueError("Expected the official visual CNN policy")
        scheduler_config = dict(config.pop("noise_scheduler"))
        if scheduler_config.pop("_target_") != "diffusers.schedulers.scheduling_ddpm.DDPMScheduler":
            raise ValueError("Unexpected official noise scheduler")
        self.network = DiffusionUnetHybridImagePolicy(noise_scheduler=DDPMScheduler(**scheduler_config), **config)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.network.eval().requires_grad_(False).to(self.device)
        self.parameter_count = sum(p.numel() for p in self.network.parameters())
        self.generator = None

    def reset_seed(self, seed):
        import torch

        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Policy seed must be uint32")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        # This is the upstream conditional_sample() generator argument, passed
        # through its existing kwargs interface; no inference code is patched.
        self.network.kwargs["generator"] = self.generator
        self.network.reset()

    def predict_actions(self, history):
        import numpy as np
        import torch

        if self.generator is None or len(history) != self.spec["policy"]["obs_horizon"]:
            raise ValueError("Reset the RNG and provide the complete observation history")
        arrays = {name: np.stack([obs[name] for obs in history]) for name in ("image", "agent_pos")}
        if (arrays["image"].shape != (2, 3, 96, 96) or arrays["agent_pos"].shape != (2, 2)
                or any(not np.isfinite(a).all() for a in arrays.values())
                or arrays["image"].min() < 0 or arrays["image"].max() > 1):
            raise ValueError("Unexpected official observation format")
        observation = {name: torch.as_tensor(value, device=self.device).unsqueeze(0) for name, value in arrays.items()}
        with torch.inference_mode():
            result = self.network.predict_action(observation)
        actions = result["action"][0].cpu().numpy()
        if actions.shape != (8, 2) or not np.isfinite(actions).all():
            raise ValueError("Unexpected action chunk from the official policy")
        return actions

    def predict_actions_with_trace(self, history):
        """Return the usual actions plus every actual DDPM sample for inspection.

        Pixel-space intermediate samples are display coordinates, not predicted
        object positions or valid executable plans. No clipping is applied here.
        The unchanged upstream sampler still supplies all executed actions.
        """
        import torch
        from .policy_diagnostics import capture_scheduler_steps

        if not self.network.obs_as_global_cond:
            raise ValueError("Denoising capture currently supports the official global-conditioning policy")
        with capture_scheduler_steps(self.network.noise_scheduler) as record:
            actions = self.predict_actions(history)
        with torch.inference_mode():
            samples = torch.stack(record["samples"])[:, 0, :, :self.network.action_dim]
            pixels = self.network.normalizer["action"].unnormalize(samples)
        start = self.network.n_obs_steps - 1
        trace = {"normalized_samples": samples.cpu().numpy(),
                 "pixel_samples": pixels.cpu().numpy(),
                 "timesteps": record["timesteps"],
                 "executed_indices": list(range(start, start + self.network.n_action_steps))}
        return actions, trace


def physical_state(env):
    def body_state(body):
        return {"position": list(body.position), "angle": float(body.angle),
                "velocity": list(body.velocity), "angular_velocity": float(body.angular_velocity),
                "force": list(body.force), "torque": float(body.torque)}
    return {"object": body_state(env.block), "pusher": body_state(env.agent),
            "goal": [float(v) for v in env.goal_pose]}


def physical_metrics(env):
    # Derive geometry from actual official collision shapes. GPC's T coordinates,
    # mass/inertia conventions, and goals are different and must not leak here.
    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely

    goal = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
    actual = pymunk_to_shapely(env.block, env.block.shapes)
    shared = goal.intersection(actual).area
    return {"coverage": float(shared / goal.area), "iou": float(shared / (goal.area + actual.area - shared))}
