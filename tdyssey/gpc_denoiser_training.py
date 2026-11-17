"""Fine-tune the GPC image denoiser on one finished official dataset.

Training calls the fetched denoiser loss directly. It does not run the upstream
training script, start wandb, read zarr data, or sample through DiffusionSampler
while computing gradients. The no-gradient sampler is used only to score
validation images.
"""

from pathlib import Path
import json
import math
import re
from types import SimpleNamespace

from .collection import integer
from .dataset import load_episode, read_manifest
from .gpc_world_model import hash_file, load_world_model_source, world_model_spec
from .motion_replay import (DEFAULT_MOVED_FRACTION, object_displacement, object_moved,
                            sample_replay_batch, split_motion_windows)
from .official_collection import ENVIRONMENT_FAMILY
from .region_weight import (DEFAULT_REGION_WEIGHT, foreground_weights, motion_mask,
                            window_case_region, window_useful)


def model_action(value):
    """Clip one official pixel target into the GPC range and map it to [-1, 1]."""
    value = float(value)
    if value < 0 or value > 511:
        value = min(511.0, max(0.0, value))
    return value / 511 * 2 - 1


def anchors(control_steps):
    """Last conditioning frame for every window that has four real history frames."""
    if type(control_steps) is not int or control_steps < 4:
        return range(0)
    return range(3, control_steps)


def window_slices(anchor):
    """Five images o[t-3:t+2] and four actions a[t-3:t+1], with t = anchor."""
    if type(anchor) is not int or anchor < 3:
        raise ValueError("Denoiser window anchor must start at frame 3")
    return slice(anchor - 3, anchor + 2), slice(anchor - 3, anchor + 1)


def phase_two_sigma(source_root):
    """Read the pinned phase-two noise distribution without executing its script."""
    text = (Path(source_root) / "world_model_train_phase_two" / "train.py").read_text()
    match = re.search(
        r"class SigmaDistributionConfig:\s*"
        r"loc\s*=\s*([^\s]+)\s*"
        r"scale\s*=\s*([^\s]+)\s*"
        r"sigma_min\s*=\s*([^\s]+)\s*"
        r"sigma_max\s*=\s*([^\s]+)",
        text)
    if match is None:
        raise ValueError("Pinned phase-two training file has no sigma distribution")
    parsed = SimpleNamespace(**{name: float(value) for name, value in zip(
        ("loc", "scale", "sigma_min", "sigma_max"), match.groups())})
    expected = {"loc": -1.2, "scale": 1.2, "sigma_min": 2e-3, "sigma_max": 20.0}
    if any(not math.isclose(getattr(parsed, name), value, rel_tol=0, abs_tol=0) for name, value in expected.items()):
        raise ValueError("Pinned phase-two sigma distribution differs from the recorded training settings")
    return parsed


def _image_name(entry):
    names = [item["name"] for item in entry["files"] if item["name"].endswith(".npz")]
    if len(names) != 1:
        raise ValueError("Episode is missing its image archive")
    return names[0]


def normalize_actions(actions):
    """Apply model_action at the training boundary. Stored targets stay unchanged."""
    import numpy as np

    values = np.asarray(actions, dtype=np.float64)
    return np.array([model_action(value) for value in values.reshape(-1)], dtype=np.float32).reshape(values.shape)


def load_official_windows(root, splits, opened):
    """Load train or validation windows. The caller chooses the splits, never model_test.

    A list of roots keeps each directory's frozen split. Windows are concatenated
    in root order; model_test files are never named or opened.
    """
    roots = list(root) if isinstance(root, (list, tuple)) else [root]
    if not roots:
        raise ValueError("Denoiser training needs at least one finished dataset")
    loaded, declared = [], []
    for one in roots:
        rows, shape = _load_root(one, splits, opened)
        loaded.extend(rows)
        declared.append(shape)
    shapes = {trajectory.transitions[0].before.shape for trajectory, _arrays, _record in loaded}
    if len(shapes) != 1:
        raise ValueError("Official denoiser windows refuse mixed shapes")
    if shapes != set(declared):
        raise ValueError("Collected shape differs from the official dataset")
    windows = []
    for trajectory, arrays, record in loaded:
        windows.extend(_windows(trajectory, arrays, record))
    return windows


def _load_root(root, splits, opened):
    root = Path(root)
    manifest = read_manifest(root)
    if manifest["provenance"].get("environment_family") != ENVIRONMENT_FAMILY:
        raise ValueError("Denoiser training expects a finished official dataset")
    if manifest["config"].get("mode") not in ("object-directed", "angle-coverage", "contact-balanced"):
        raise ValueError("Denoiser training expects object-directed, angle-coverage, or contact-balanced official play")
    by_id = {entry["trajectory_id"]: entry for entry in manifest["episodes"]}
    loaded = []
    for split in splits:
        if split == "model_test":
            raise ValueError("Denoiser training must not read model_test")
        for identifier in manifest["splits"][split]:
            entry = by_id[identifier]
            opened.append(_image_name(entry))
            trajectory, record, arrays = load_episode(root, entry)
            loaded.append((trajectory, arrays, record))
    if not loaded:
        raise ValueError("Selected splits contain no episodes")
    return loaded, manifest["config"]["shape"]


def _windows(trajectory, arrays, record=None):
    import numpy as np

    images = np.asarray(arrays["images"], dtype=np.float32) / 255
    actions = np.asarray([step.action for step in trajectory.transitions], dtype=np.float32)
    if images.shape[0] != len(actions) + 1 or not np.isfinite(images).all() or not np.isfinite(actions).all():
        raise ValueError("Official window frames or actions are incomplete")
    diagnostics = None if record is None else record["step_diagnostics"]
    if diagnostics is not None and len(diagnostics) != len(actions):
        raise ValueError("Episode diagnostics do not match control transitions")
    windows = []
    for anchor in anchors(len(actions)):
        image_slice, action_slice = window_slices(anchor)
        frame, action = images[image_slice], actions[action_slice]
        if frame.shape != (5, 3, 96, 96) or action.shape != (4, 2):
            raise ValueError("Official denoiser window has the wrong shape")
        predicted = trajectory.transitions[anchor]
        translation, rotation = object_displacement(predicted.before, predicted.after)
        moved = object_moved(predicted.before, predicted.after)
        n_contacts = 0 if diagnostics is None else integer(
            diagnostics[anchor]["n_contacts"], "contacts", 0, 100000)
        useful = window_useful(moved, n_contacts)
        windows.append({"episode_id": trajectory.trajectory_id, "anchor": anchor,
                        "image": frame, "action": action, "moved": moved,
                        "n_contacts": n_contacts, "useful": useful,
                        "region": window_case_region(
                            predicted.before, predicted.after, frame[4], frame[3],
                            moved=moved, n_contacts=n_contacts),
                        "translation": translation, "rotation": rotation})
    return windows


def decide(record, caps=None):
    """Choose stop, collect, or train from a finished round's logged errors.

    The record is the numbers already written for this round. Caps default to
    two collections and 20 updates. This function does not apply the round cap;
    the pipeline records the choice and may refuse to act on it.
    """
    limits = {"max_collections": 2, "max_updates": 20, "patience": 1}
    if caps:
        limits.update(caps)
    contact = _count(record["contact_steps"], "contact steps")
    collections = _count(record["collection_runs"], "collection runs")
    updates = _count(record["updates"], "updates")
    episodes = _count(record["episodes"], "episodes")
    steps = _count(record["steps"], "steps")
    seed = _count(record["seed"], "seed")
    adapted = _finite(record["adapted_mse"], "adapted mse")
    still = _finite(record["persistence_mse"], "persistence mse")
    previous = None if record.get("previous_mse") is None else _finite(record["previous_mse"], "previous mse")
    max_collections = _count(limits["max_collections"], "max collections")
    max_updates = _count(limits["max_updates"], "max updates")
    patience = _count(limits["patience"], "patience")
    if patience < 1:
        raise ValueError("patience must be a positive integer")
    can_collect = collections < max_collections
    reduced = previous is not None and adapted < previous
    streak = _count(record["non_improving_rounds"], "non-improving rounds") if "non_improving_rounds" in record else (0 if reduced else 1)
    if adapted <= still and contact > 0:
        return {"action": "stop", "reason": "validation image error is no worse than a still frame and the data includes contact"}
    if contact == 0 and can_collect:
        return _collect_choice(episodes, steps, seed, "logged episodes contain no contact, so more play is required before treating image error as dynamics")
    if contact > 0 and adapted > still and updates < max_updates and (reduced or streak < patience):
        reason = ("validation error is still above a still frame and the last round reduced it" if reduced
                  else "validation error did not fall this round, but the patience window is still open")
        return {"action": "train", "reason": reason}
    if not reduced and streak >= patience and can_collect and contact > 0 and updates < max_updates:
        return _collect_choice(episodes, steps, seed, "more training did not reduce validation error; collect a disjoint batch")
    if contact == 0:
        return {"action": "stop", "reason": "collection cap reached before any logged contact"}
    if updates >= max_updates:
        return {"action": "stop", "reason": "update cap reached while validation error is still above a still frame"}
    if not can_collect:
        return {"action": "stop", "reason": "collection cap reached while validation error is still above a still frame"}
    return {"action": "stop", "reason": "validation error is still above a still frame and another round is not available"}


def dataset_name(collection_index):
    """First batch keeps the short-check directory name. Later batches are numbered."""
    if type(collection_index) is not int or collection_index < 1:
        raise ValueError("Collection index must be a positive integer")
    if collection_index == 1:
        return "dataset"
    return f"dataset-{collection_index:03d}"


def limit_round(choice, rounds_completed, max_rounds):
    """Keep the logged choice, but do not act after the round cap."""
    if type(rounds_completed) is not int or type(max_rounds) is not int or min(rounds_completed, max_rounds) < 1:
        raise ValueError("Round counts must be positive integers")
    if rounds_completed >= max_rounds and choice["action"] != "stop":
        reason = f"round cap; logged choice was {choice['action']}: {choice['reason']}"
        return {**choice, "executed": "stop", "executed_reason": reason}
    return {**choice, "executed": choice["action"], "executed_reason": choice["reason"]}


def _collect_choice(episodes, steps, seed, reason):
    if steps < 100:
        steps = min(100, 1000 // episodes)
    if episodes < 7 or steps < 4 or episodes * steps > 1000:
        return {"action": "stop", "reason": "the next collection would exceed the 7-episode, 4-step, or 1000-control limit"}
    return {"action": "collect", "reason": reason, "steps": steps, "seed": seed + 1}


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite(value, name):
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return value


def weighted_denoiser_loss(network, image, action, weights, device):
    """One EDM step with pixel weights. Does not patch the pinned GPC module."""
    import torch

    batch, steps, channels, height, width = image.size()
    history = network.cfg.inner_model.num_steps_conditioning
    if steps != history + 1:
        raise ValueError("Weighted denoiser loss expects four conditioning frames and one next frame")
    if network.sample_sigma_training is None:
        raise ValueError("Denoiser training sigma sampler is not set")
    previous = image[:, :history].reshape(batch, history * channels, height, width)
    previous_action = action[:, :history]
    next_obs = image[:, history]
    if network.cfg.noise_previous_obs:
        sigma_cond = network.sample_sigma_training(batch, device)
        previous = network.apply_noise(previous, sigma_cond, network.cfg.sigma_offset_noise)
    else:
        sigma_cond = None
    sigma = network.sample_sigma_training(batch, device)
    noisy = network.apply_noise(next_obs, sigma, network.cfg.sigma_offset_noise)
    conditioners = network.compute_conditioners(sigma, sigma_cond)
    predicted = network.compute_model_output(noisy, previous, previous_action, conditioners)
    target = (next_obs - conditioners.c_skip * noisy) / conditioners.c_out
    error = (predicted - target) ** 2
    weight = weights[:, None] if weights.ndim == 3 else weights
    if weight.shape[0] != batch or weight.shape[-2:] != (height, width):
        raise ValueError("Region weights must match the image batch")
    expanded = weight.expand_as(error)
    loss = (error * expanded).sum() / expanded.sum().clamp_min(1e-8)
    if not torch.isfinite(loss):
        raise ValueError("Denoiser training loss is not finite")
    return loss


SCORE_FIELDS = (
    "mse", "persistence_mse", "object_roi_mse", "object_roi_persistence",
    "motion_mse", "motion_persistence", "translation", "rotation",
    "object_centroid_error_pixels", "predicted_centroid_shift_pixels",
    "actual_centroid_shift_pixels",
)


def score_frames(predicted, target, still, region=None, translation=None, rotation=None,
                 useful=None):
    """Full-image, region, and motion errors. Appearance centroid is a color proxy."""
    import numpy as np

    from .appearance import prediction_metrics

    predicted = np.clip(np.asarray(predicted, dtype=np.float64), 0, 1)
    target = np.asarray(target, dtype=np.float64)
    still = np.asarray(still, dtype=np.float64)
    if predicted.shape != (3, 96, 96) or target.shape != predicted.shape or still.shape != predicted.shape:
        raise ValueError("Frame scores expect 3x96x96 official images")
    error = (predicted - target) ** 2
    persist = (still - target) ** 2
    motion = motion_mask(target, still)
    mask = None if region is None else np.asarray(region, dtype=bool)
    row = {"mse": float(error.mean()), "persistence_mse": float(persist.mean()),
           "object_roi_mse": _masked_mean(error, mask),
           "object_roi_persistence": _masked_mean(persist, mask),
           "motion_mse": _masked_mean(error, motion),
           "motion_persistence": _masked_mean(persist, motion),
           "translation": None if translation is None else float(translation),
           "rotation": None if rotation is None else float(rotation),
           "object_centroid_error_pixels": None,
           "predicted_centroid_shift_pixels": None,
           "actual_centroid_shift_pixels": None,
           "useful": None if useful is None else bool(useful)}
    if mask is not None and mask.any():
        appearance = prediction_metrics(predicted, target, still, mask)
        row["object_centroid_error_pixels"] = appearance["object_centroid_error_pixels"]
        row["predicted_centroid_shift_pixels"] = appearance["predicted_centroid_shift_pixels"]
        row["actual_centroid_shift_pixels"] = appearance["actual_centroid_shift_pixels"]
    return row


def _masked_mean(error, mask):
    import numpy as np

    if mask is None:
        return None
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != error.shape[-2:] or not selected.any():
        return None
    return float(error[:, selected].mean())


class GPCDenoiserTrainer:
    """A few CUDA denoising updates from the pinned GPC checkpoint."""

    def __init__(self, source_root, checkpoint, updates=4, batch_size=4, seed=0, seconds=120,
                 moved_fraction=DEFAULT_MOVED_FRACTION, region_weight=DEFAULT_REGION_WEIGHT):
        from .collection import integer

        self.source_root = Path(source_root)
        self.checkpoint = Path(checkpoint)
        self.updates = integer(updates, "updates", 1, 20)
        self.batch_size = integer(batch_size, "batch size", 1, 8)
        self.seed = integer(seed, "seed", 0, 2**32 - 1)
        self.seconds = integer(seconds, "seconds", 1, 900)
        self.moved_fraction = moved_fraction
        self.region_weight = integer(region_weight, "region weight", 1, 64)

    def run(self, dataset, output, validation=None, training=None):
        import time
        import numpy as np
        import torch

        start = time.monotonic()
        deadline = start + self.seconds
        opened = []
        scored_training = training if training is not None else load_official_windows(
            dataset, ("train",), opened)
        scored_validation = validation if validation is not None else load_official_windows(
            dataset, ("validation",), opened)
        if not scored_training or not scored_validation:
            raise ValueError("Denoiser training needs both training and validation windows")
        if torch.cuda.is_available() is False:
            raise ValueError("GPC denoiser training requires CUDA; no CPU fallback")
        torch.set_num_threads(4)
        device = torch.device("cuda")
        spec = world_model_spec()
        source = load_world_model_source(self.source_root, spec)
        config = dict(spec["model"])
        denoiser_config = {name: config.pop(name) for name in ("sigma_data", "sigma_offset_noise", "noise_previous_obs")}
        network = source.Denoiser(source.DenoiserConfig(
            inner_model=source.InnerModelConfig(**config), **denoiser_config)).float()
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in payload.values()):
            raise ValueError("Non-tensor or non-finite world-model weights")
        network.load_state_dict(payload, strict=True)
        del payload
        sidecar = self.checkpoint.with_name("denoiser.json")
        resumed = self.checkpoint.name == "denoiser.pt" and sidecar.exists()
        saved = json.loads(sidecar.read_text()) if resumed else None
        network.to(device)
        network.setup_training(phase_two_sigma(self.source_root))
        sampler = source.DiffusionSampler(network, source.DiffusionSamplerConfig(**spec["sampler"]))
        reference = next(network.parameters()).detach().cpu().clone()
        network.eval()
        before = _score(sampler, scored_validation, device, self.seed)
        if time.monotonic() >= deadline:
            raise TimeoutError("Denoiser training wall-time limit exceeded")
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
        generator = torch.Generator().manual_seed(self.seed)
        moved, still = split_motion_windows(scored_training)
        losses, replay_batches = [], []
        network.train()
        for _step in range(self.updates):
            if time.monotonic() >= deadline:
                raise TimeoutError("Denoiser training wall-time limit exceeded")
            batch, mix = sample_replay_batch(moved, still, self.batch_size, self.moved_fraction, generator)
            replay_batches.append(mix)
            image = torch.as_tensor(np.stack([item["image"] for item in batch]), device=device)
            action = torch.as_tensor(np.stack([normalize_actions(item["action"]) for item in batch]), device=device)
            weight = torch.as_tensor(np.stack([
                foreground_weights(item["region"]) if item.get("region") is not None
                else np.ones((96, 96), dtype=np.float32) for item in batch]), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_denoiser_loss(network, image, action, weight, device)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        network.eval()
        adapted = _score(sampler, scored_validation, device, self.seed)
        changed = not torch.equal(reference, next(network.parameters()).detach().cpu())
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        prior_updates = 0 if saved is None else saved["updates"]
        init_hash = hash_file(self.checkpoint) if saved is None else saved["init_sha256"]
        torch.save({name: value.detach().cpu() for name, value in network.state_dict().items()}, output / "denoiser.pt")
        (output / "denoiser.json").write_text(json.dumps(
            {"updates": prior_updates + self.updates, "init_sha256": init_hash}) + "\n")
        scored = {"windows": len(scored_validation), "before": before, "adapted": adapted}
        if not resumed:
            scored["frozen"] = before
        return {"updates": self.updates, "losses": losses, "weights_changed": changed,
                "opened_images": opened, "model_test_used": False, "resumed": resumed,
                "validation": scored, "elapsed_seconds": time.monotonic() - start,
                "replay": {"moved_windows": len(moved), "still_windows": len(still),
                           "moved_fraction": self.moved_fraction, "region_weight": self.region_weight,
                           "loss": "case_split_foreground", "batches": replay_batches}}


def _score(sampler, windows, device, seed):
    """Compare clipped sampler RGB with the true next frame and a still frame."""
    import numpy as np
    import torch

    rows, raw_min, raw_max = [], [], []
    for index, window in enumerate(windows):
        image = torch.as_tensor(window["image"][:4], device=device)[None]
        action = torch.as_tensor(normalize_actions(window["action"]), device=device)[None]
        gpu = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(gpu), torch.random.fork_rng(devices=[gpu]), torch.inference_mode():
            torch.cuda.manual_seed(seed + index)
            prediction, _steps = sampler.sample(image, action)
        raw = prediction[0].detach().cpu().numpy()
        useful = window.get("useful")
        if useful is None:
            useful = window_useful(window.get("moved", False), window.get("n_contacts", 0))
        rows.append(score_frames(np.clip(raw, 0, 1), window["image"][4], window["image"][3],
                                 window.get("region"), window.get("translation"),
                                 window.get("rotation"), useful=useful))
        raw_min.append(float(raw.min()))
        raw_max.append(float(raw.max()))
    return summarize_score_rows(rows, raw_min, raw_max)


def summarize_score_rows(rows, raw_min=None, raw_max=None):
    """Average frame scores overall and on useful versus miss windows."""
    summary = {name: _mean_optional(row.get(name) for row in rows) for name in SCORE_FIELDS}
    useful_rows = [row for row in rows if row.get("useful")]
    miss_rows = [row for row in rows if row.get("useful") is False]
    summary["useful_windows"] = len(useful_rows)
    summary["miss_windows"] = len(miss_rows)
    summary["useful"] = {name: _mean_optional(row.get(name) for row in useful_rows) for name in SCORE_FIELDS}
    summary["miss"] = {name: _mean_optional(row.get(name) for row in miss_rows) for name in SCORE_FIELDS}
    if raw_min is not None:
        import numpy as np
        summary["raw_min"] = float(np.min(raw_min))
        summary["raw_max"] = float(np.max(raw_max))
    return summary


def _mean_optional(values):
    present = [float(value) for value in values if value is not None]
    return None if not present else float(sum(present) / len(present))


def load_denoiser_sampler(source_root, checkpoint):
    """Load a saved denoiser for one-step image prediction. The pinned hash is not required."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("World-model prediction requires CUDA; no CPU fallback")
    spec = world_model_spec()
    source = load_world_model_source(source_root, spec)
    config = dict(spec["model"])
    denoiser_config = {name: config.pop(name) for name in ("sigma_data", "sigma_offset_noise", "noise_previous_obs")}
    network = source.Denoiser(source.DenoiserConfig(
        inner_model=source.InnerModelConfig(**config), **denoiser_config)).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in payload.values()):
        raise ValueError("Non-tensor or non-finite world-model weights")
    network.load_state_dict(payload, strict=True)
    device = torch.device("cuda")
    network.to(device).eval()
    sampler = source.DiffusionSampler(network, source.DiffusionSamplerConfig(**spec["sampler"]))
    return sampler, device


def predict_next_frame(sampler, images, actions, device, seed=0):
    """Predict one next RGB frame from four images and four pixel targets."""
    import numpy as np
    import torch

    image = torch.as_tensor(np.asarray(images, dtype=np.float32), device=device)[None]
    action = torch.as_tensor(normalize_actions(actions), device=device)[None]
    if image.shape != (1, 4, 3, 96, 96) or action.shape != (1, 4, 2):
        raise ValueError("Prediction expects four 96x96 frames and four actions")
    gpu = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(gpu), torch.random.fork_rng(devices=[gpu]), torch.inference_mode():
        torch.cuda.manual_seed(seed)
        prediction, _steps = sampler.sample(image, action)
    return np.clip(prediction[0].detach().cpu().numpy(), 0, 1)
