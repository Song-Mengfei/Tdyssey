"""Bounded pretrained T prediction check in official and native GPC scenes."""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def color_mask(image, color):
    import numpy as np
    return np.linalg.norm(image - np.asarray(color)[:, None, None], axis=0) < .15


def centroid(mask):
    import numpy as np
    y, x = np.nonzero(mask)
    return None if len(x) < 12 else np.array([x.mean() + .5, y.mean() + .5]) * 512 / 96


def measure(predicted, actual, latest, roi, color):
    """Color centroids are image diagnostics, not an independent pose decoder."""
    import numpy as np
    result = {}
    actual_mask = color_mask(actual, color)
    target_center = centroid(actual_mask)
    for name, image in (("model", predicted), ("persistence", latest)):
        error = (image - actual) ** 2
        mask = color_mask(image, color)
        center = centroid(mask)
        union = np.logical_or(mask, actual_mask).sum()
        result[name] = {
            "image_mse": float(error.mean()),
            "object_roi_mse": float(error[:, roi].mean()),
            "color_centroid_error_pixels": None if center is None or target_center is None
                else float(np.linalg.norm(center - target_center)),
            "color_mask_pixels": int(mask.sum()),
            "color_mask_iou": float((mask & actual_mask).sum() / union) if union else None,
        }
    return result


def run(args):
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from examples.gpc_shape_env import load_upstream
    from tdyssey.envs.gpc import make_shape_env_class
    from tdyssey.envs.official import make_official_shape_env_class, official_shape
    from tdyssey.geometry import Pose, get_shape
    from tdyssey.gpc_world_model import GPCWorldModel
    from tdyssey.official import load_official_source
    from tdyssey.policy import hash_file
    from tdyssey.state import BodyState
    from tdyssey.world_model_data import object_masks

    started = time.monotonic()
    torch.set_num_threads(2)
    model = GPCWorldModel(args.gpc_root, args.checkpoint)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report = {
        "schema_version": 1, "kind": "pretrained_gpc_t_prediction_diagnostic",
        "status": "running", "asset_spec": model.spec,
        "parameter_count": model.parameter_count, "state_entries": model.state_entries,
        "device": str(model.device), "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "future": args.future, "seed": args.seed, "weights_updated": False,
        "model_test_used": False, "normalization_fitted": False,
        "protocol": "two hand-constructed, pre-contact T scenes; four branches per family; no policy learning",
        "candidate_prior": "push/away/side directions and scene placement use disclosed simulator geometry",
        "noise_protocol": "same seed per scene across candidates; fork/restore global CUDA RNG",
        "image_protocol": "training-native RGB [0,1]; raw generated history; clip only for observation metrics/preview",
        "metric_limits": "color centroid/IoU are appearance-sensitive proxies, not verified pose/reward models",
        "restore_limits": "full physical/controller state, fresh solver; contact impulses are not serialized",
        "counts": {"control_steps": 0, "control_physics_substeps": 0, "reset_physics_substeps": 0,
                   "state_restores": 0, "model_forecast_steps": 0, "denoising_steps": 0},
        "source_sha256": {p: hash_file(ROOT / p) for p in (
            "tdyssey/gpc_world_model.py", "scripts/probe_gpc_world_model.py", "manifests/gpc_world_model.json")},
        "scenes": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    old_cwd, env, previews = Path.cwd(), None, []
    counts = report["counts"]

    def deadline():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Pretrained prediction probe exceeded its time budget")

    try:
        families = ("official", "gpc") if args.family == "both" else (args.family,)
        for family in families:
            deadline()
            if family == "official":
                cls = make_official_shape_env_class(load_official_source(args.official_root))
                env = cls(shape="T", legacy=True)
                shape = official_shape("T")
            else:
                module, _ = load_upstream(args.gpc_root)
                os.chdir(args.gpc_root / "gpc_rank_evaluation")
                env = make_shape_env_class(module.PushTImageEnv)(shape="T", domain_filename="push_t", resize_scale=96)
                shape = get_shape("T")
            env.seed(args.seed)
            env.reset()
            counts["reset_physics_substeps"] += 1
            template = env.capture_state()
            for scene_index, (name, pusher, direction) in enumerate((
                    ("bottom", (256, 400), (0, -1)), ("left", (216, 320), (1, 0)))):
                deadline()
                object_pose = Pose((256, 240 if family == "official" else 285), 0)
                goal = Pose((256, 256), math.pi / 4)
                if family == "gpc":
                    goal = Pose(goal.transform((0, 45)), goal.angle)
                initial = replace(template, object_body=BodyState(object_pose),
                                  pusher_body=BodyState(Pose(pusher, 0)), goal=goal,
                                  control_step=0, pusher_target=None)
                observation = env.restore_state(initial)
                counts["state_restores"] += 1
                if family == "gpc":
                    observation = observation[0]
                images, past, warmup_contacts = [observation["image"]], [], []
                for _ in range(3):
                    result = env.step(np.asarray(pusher, dtype=np.float64))
                    observation, info = result[0], result[-1]
                    images.append(observation["image"])
                    past.append(pusher)
                    warmup_contacts.append(int(info["n_contacts"]))
                    counts["control_steps"] += 1
                    counts["control_physics_substeps"] += 10
                if any(warmup_contacts):
                    raise ValueError("Diagnostic history must be free of prior contact")
                anchor = env.capture_state()
                images = np.stack(images)
                initial_roi = object_masks(shape, [(*anchor.object_body.pose.position, anchor.object_body.pose.angle)])[0, 0]
                colors, color_counts = np.unique((images[-1][:, initial_roi].T * 255).round().astype(np.uint8), axis=0, return_counts=True)
                color = colors[color_counts.argmax()].astype(float) / 255
                if centroid(color_mask(images[-1], color)) is None:
                    raise ValueError("Cannot identify the current object's appearance")
                row = {"family": family, "name": name, "initial_state": asdict(anchor),
                       "warmup_contacts": warmup_contacts, "reference_rgb": color.tolist(),
                       "physics": {"mass": env.block.mass, "moment": env.block.moment,
                                   "cog": list(env.block.center_of_gravity)}, "candidates": []}
                report["scenes"].append(row)
                base = np.asarray(pusher)
                direction = np.asarray(direction)
                delta = np.arange(1, args.future + 1)[:, None] * 12
                candidates = {"hold": np.repeat(base[None], args.future, axis=0),
                              "push": base + delta * direction, "away": base - delta * direction,
                              "side": base + delta * np.array([-direction[1], direction[0]])}
                for candidate, actions in candidates.items():
                    deadline()
                    restored = env.restore_state(anchor)
                    counts["state_restores"] += 1
                    restored = restored[0] if family == "gpc" else restored
                    if not np.array_equal(restored["image"], images[-1]):
                        raise ValueError("Restored anchor image differs from the supplied history")
                    noise_seed = args.seed + scene_index
                    raw = model.predict(images, past, actions, seed=noise_seed)
                    torch.cuda.synchronize()
                    counts["model_forecast_steps"] += len(actions)
                    counts["denoising_steps"] += len(actions) * 3
                    predicted = np.clip(raw, 0, 1)
                    entry = {"name": candidate, "actions": actions.tolist(), "noise_seed": noise_seed,
                             "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                             "clipped_fraction": float(((raw < 0) | (raw > 1)).mean()), "steps": []}
                    row["candidates"].append(entry)
                    actual_images = []
                    for horizon, action in enumerate(actions, 1):
                        deadline()
                        result = env.step(action)
                        actual, reward, done, info = result[0], result[1], result[2], result[-1]
                        counts["control_steps"] += 1
                        counts["control_physics_substeps"] += 10
                        state = env.capture_state()
                        before, after = anchor.object_body.pose, state.object_body.pose
                        roi = initial_roi | object_masks(shape, [(*after.position, after.angle)])[0, 0]
                        metrics = measure(predicted[horizon - 1], actual["image"], images[-1], roi, color)
                        entry["steps"].append({"horizon": horizon, "n_contacts": int(info["n_contacts"]),
                            "reward": float(reward), "success": bool(done), "state": asdict(state),
                            "actual_object_origin_displacement_pixels": float(math.dist(before.position, after.position)),
                            "metrics": metrics})
                        actual_images.append(actual["image"])
                        if done:
                            break
                    # Each preview row shows initial / actual / prediction / persistence.
                    previews.append((f"{family} {name} {candidate} h{len(actual_images)}",
                                     [images[-1], actual_images[-1], predicted[len(actual_images)-1], images[-1]]))
                    entry["predicted_final_sha256"] = __import__("hashlib").sha256(predicted[-1].tobytes()).hexdigest()
                if not any(s["n_contacts"] for c in row["candidates"] for s in c["steps"]):
                    raise ValueError("Scene produced no future contact; diagnostic is incomplete")
            env.close()
            env = None
        canvas = Image.new("RGB", (4*192, len(previews)*218), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (label, frames) in enumerate(previews):
            draw.text((4, row_index*218+4), label, fill="black")
            for column, frame in enumerate(frames):
                array = (frame.transpose(1, 2, 0)*255).round().astype(np.uint8)
                canvas.paste(Image.fromarray(array).resize((192,192)), (column*192, row_index*218+22))
        canvas.save(args.output / "predictions.png")
        report["preview_columns"] = ["initial", "physical future", "model future", "persistence"]
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["retained_checkpoint_bytes"] = args.checkpoint.stat().st_size
        content = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
        if len(content) + sum(p.stat().st_size for p in args.output.iterdir()) > 4*1024**2:
            raise ValueError("Prediction diagnostic exceeds its 4 MiB output budget")
        (args.output / "report.json").write_bytes(content)
    print(json.dumps({"status": report["status"], "counts": counts,
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--official-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gpc-world-model/denoiser.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("official", "gpc", "both"), default="both")
    parser.add_argument("--future", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=300000)
    args = parser.parse_args()
    if not 1 <= args.future <= 4 or not 1 <= args.seconds <= 300 or not 0 <= args.seed < 2**32 - 1:
        parser.error("Use 1–4 future controls, 1–300 seconds and a uint32 seed below its maximum")
    for key in ("gpc_root", "official_root", "checkpoint", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    run(args)


if __name__ == "__main__":
    main()
