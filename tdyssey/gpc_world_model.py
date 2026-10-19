"""Verified pretrained GPC dynamics, with upstream inference kept unchanged."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .policy import hash_file

ROOT = Path(__file__).resolve().parents[1]


def world_model_spec():
    return json.loads((ROOT / "manifests/gpc_world_model.json").read_text())


def load_world_model_source(source_root, spec=None):
    """Load legacy absolute imports temporarily; restore other projects' modules.

    Only imports are isolated. No upstream source or model operation is patched.
    This loader is for sequential setup, not concurrent imports in several threads.
    """
    spec = spec or world_model_spec()
    root = Path(source_root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["source_commit"]:
        raise ValueError("GPC world-model source revision differs from the manifest")
    for name, expected in spec["source_sha256"].items():
        if hash_file(root / name) != expected:
            raise ValueError(f"Missing/modified world-model source: {name}")
    prefixes = ("diffusion", "data", "utils")
    def owned(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    saved = {name: module for name, module in sys.modules.items() if owned(name)}
    previous_path = list(sys.path)
    try:
        for name in saved:
            del sys.modules[name]
        sys.path.insert(0, str(root / "gpc_rank_evaluation"))
        package = importlib.import_module("diffusion")
        return SimpleNamespace(**{name: getattr(package, name) for name in (
            "Denoiser", "DenoiserConfig", "InnerModelConfig", "DiffusionSampler", "DiffusionSamplerConfig")})
    finally:
        for name in list(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = previous_path


def action_windows(previous_actions, future_actions):
    """Three executed controls plus each next proposed control: no frame offset."""
    previous, future = list(previous_actions), list(future_actions)
    if len(previous) != 3 or not 1 <= len(future) <= 4:
        raise ValueError("Provide three past actions and one to four future actions")
    result = []
    for action in future:
        result.append(previous + [action])
        previous = previous[1:] + [action]
    return result


class GPCWorldModel:
    def __init__(self, source_root, checkpoint, device="cuda"):
        import torch

        self.spec, self.device = world_model_spec(), torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Pretrained world-model inference requires CUDA; no CPU fallback")
        asset, checkpoint = self.spec["checkpoint"], Path(checkpoint)
        if checkpoint.stat().st_size != asset["bytes"] or hash_file(checkpoint) != asset["sha256"]:
            raise ValueError("World-model checkpoint differs from pinned size/hash")
        self.source = load_world_model_source(source_root, self.spec)
        config = dict(self.spec["model"])
        denoiser = {name: config.pop(name) for name in ("sigma_data", "sigma_offset_noise", "noise_previous_obs")}
        inner = self.source.InnerModelConfig(**config)
        self.network = self.source.Denoiser(self.source.DenoiserConfig(inner_model=inner, **denoiser)).float()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in payload.values()):
            raise ValueError("Non-tensor or non-finite world-model weights")
        self.network.load_state_dict(payload, strict=True)
        self.state_entries = len(payload)
        self.parameter_count = sum(parameter.numel() for parameter in self.network.parameters())
        del payload
        self.network.to(self.device).eval().requires_grad_(False)
        self.sampler = self.source.DiffusionSampler(
            self.network, self.source.DiffusionSamplerConfig(**self.spec["sampler"]))

    def predict(self, images, previous_actions, future_actions, seed=0):
        """Return raw upstream float RGB predictions; no reward/state is an input.

        Inputs are true RGB [0,1], as in upstream training, and absolute pixel
        targets. The first prediction uses three past actions and the proposed
        next one. Subsequent predictions feed back generated images only.
        """
        import numpy as np
        import torch

        images = np.asarray(images, dtype=np.float32)
        past = np.asarray(previous_actions, dtype=np.float32)
        future = np.asarray(future_actions, dtype=np.float32)
        if (images.shape != (4, 3, 96, 96) or past.shape != (3, 2)
                or future.ndim != 2 or future.shape[1:] != (2,) or not 1 <= len(future) <= 4
                or not all(np.isfinite(value).all() for value in (images, past, future))
                or images.min() < 0 or images.max() > 1
                or min(past.min(), future.min()) < 0 or max(past.max(), future.max()) > 511):
            raise ValueError("Expected four RGB frames [0,1] and finite pixel targets in [0,511]")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("World-model noise seed must be uint32")
        windows = action_windows(past, future)
        image_tensor = torch.as_tensor(images, device=self.device)[None]
        predictions = []
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index), torch.random.fork_rng(devices=[index]), torch.inference_mode():
            torch.cuda.manual_seed(seed)
            for window in windows:
                actions = torch.as_tensor(np.asarray(window), dtype=torch.float32, device=self.device)[None]
                actions = actions / 511 * 2 - 1
                prediction, _ = self.sampler.sample(image_tensor, actions)
                if not torch.isfinite(prediction).all():
                    raise ValueError("Pretrained world model produced non-finite images")
                predictions.append(prediction[0].cpu().numpy())
                image_tensor = torch.cat((image_tensor[:, 1:], prediction[:, None]), dim=1)
        return np.stack(predictions)
