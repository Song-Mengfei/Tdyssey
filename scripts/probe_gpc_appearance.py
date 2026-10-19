"""Separate pretrained T appearance effects from physics with shared branches."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(rows):
    """Group each horizon separately; report missing estimates, never as zero error."""
    result = {}
    for row in rows:
        for step in row["steps"]:
            for group in ("all", "contact" if step["n_contacts"] else "noncontact"):
                key = f"{row['family']}/{row['view']}/h{step['horizon']}/{group}"
                result.setdefault(key, []).append(step)
    summary = {}
    for key, steps in result.items():
        summary[key] = {"samples": len(steps), "metrics": {}}
        for method in ("model", "persistence"):
            metrics = {}
            for metric in steps[0][method]:
                values = [s[method][metric] for s in steps if s[method][metric] is not None]
                metrics[metric] = {"mean": statistics.mean(values) if values else None,
                                   "valid": len(values), "missing": len(steps)-len(values)}
            summary[key]["metrics"][method] = metrics
    return summary


def run(args):
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from examples.gpc_shape_env import load_upstream
    from tdyssey.appearance import prediction_metrics, source_palette_image, DISPLAY_OBJECT_RGB, DISPLAY_TARGET_RGB
    from tdyssey.envs.gpc import make_shape_env_class
    from tdyssey.envs.official import make_official_shape_env_class, official_shape
    from tdyssey.geometry import Pose, get_shape
    from tdyssey.gpc_world_model import GPCWorldModel
    from tdyssey.official import load_official_source
    from tdyssey.policy import hash_file
    from tdyssey.state import BodyState
    from tdyssey.world_model_data import object_masks

    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    from tdyssey.collection import integer
    if (protocol["protocol"] != "gpc-t-appearance-v1" or protocol["future"] != 4
            or protocol["history_controls"] != 3 or protocol["families"] != ["official", "gpc"]
            or protocol["official_views"] != ["original", "source_palette"]
            or protocol["native_views"] != ["source_palette"]
            or protocol["candidates"] != ["hold", "push", "away", "side"]):
        raise ValueError("Unknown appearance protocol or changed comparison structure")
    integer(len(protocol["scenes"]), "scene count", 1, 6)
    integer(len(protocol["noise_seeds"]), "noise repeats", 1, 3)
    integer(protocol["max_seconds"], "wall-time cap", 1, 300)
    integer(protocol["max_output_bytes"], "output byte cap", 1024, 8*1024**2)
    integer(protocol["target_increment_pixels"], "target increment", 1, 12)
    for seed in protocol["noise_seeds"]:
        integer(seed, "noise seed", 0, 2**32-7)
    if not math.isfinite(protocol["image_color_tolerance"]) or not 0 < protocol["image_color_tolerance"] <= .2:
        raise ValueError("Color tolerance must be finite and at most .2")
    torch.set_num_threads(2)
    model = GPCWorldModel(args.gpc_root, args.checkpoint)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    counts = {"control_steps": 0, "control_physics_substeps": 0, "reset_physics_substeps": 0,
              "state_restores": 0, "auxiliary_renders": 0, "model_forecast_steps": 0, "denoising_steps": 0}
    report = {"schema_version": 1, "kind": "frozen_pretrained_t_palette_ablation", "status": "running",
        "protocol": protocol, "protocol_sha256": hash_file(args.protocol),
        "checkpoint_sha256": model.spec["checkpoint"]["sha256"],
        "source_sha256": {p: hash_file(ROOT / p) for p in (
            "scripts/probe_gpc_appearance.py", "tdyssey/appearance.py", "tdyssey/gpc_world_model.py")},
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "counts": counts, "scenes": [], "predictions": [],
        "model_test_used": False, "weights_updated": False, "dataset_loaded": False,
        "palette_access": "auxiliary simulator renderer; privileged rendering control, not an image-only adapter",
        "metric_limits": "known-color image masks are proxies; RGB MSE is not comparable across palettes as a pure motion measure",
        "object_rgb_palette": DISPLAY_OBJECT_RGB, "target_rgb_palette": DISPLAY_TARGET_RGB,
        "inference_noise": "same seed per scene/candidate/view, with random-state restoration; no best-seed selection",
        "restore_limits": "fresh cold solver; full physical/controller state, no cached contact impulses",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    old_cwd, env = Path.cwd(), None
    previews, paired = [], {}

    def deadline():
        if time.monotonic()-started > protocol["max_seconds"]:
            raise TimeoutError("Appearance experiment time budget exceeded")

    def views(observation, family):
        before, cache = env.capture_state(), env.render_cache.copy()
        auxiliary = source_palette_image(env)
        counts["auxiliary_renders"] += 1
        if env.capture_state() != before or not np.array_equal(env.render_cache, cache):
            raise ValueError("Appearance rendering changed physical state or the policy render cache")
        again = env._get_obs()["image"]
        if not np.array_equal(again, observation["image"]):
            raise ValueError("Appearance rendering changed the original policy observation")
        output = {"source_palette": auxiliary}
        if family == "official":
            output["original"] = observation["image"].copy()
        elif not np.array_equal(auxiliary, observation["image"]):
            raise ValueError("Source palette control differs from the native GPC observation")
        return output

    try:
        for family in protocol["families"]:
            deadline()
            if family == "official":
                env = make_official_shape_env_class(load_official_source(args.official_root))(shape="T", legacy=True)
                shape = official_shape("T")
            else:
                module, _ = load_upstream(args.gpc_root)
                os.chdir(args.gpc_root / "gpc_rank_evaluation")
                env = make_shape_env_class(module.PushTImageEnv)(shape="T", domain_filename="push_t", resize_scale=96)
                shape = get_shape("T")
            env.seed(protocol["noise_seeds"][0])
            env.reset()
            counts["reset_physics_substeps"] += 1
            template = env.capture_state()
            for scene_index, scene in enumerate(protocol["scenes"]):
                deadline()
                angle = scene["angle"]
                center = Pose(tuple(protocol["object_physical_center"]), angle)
                pusher = np.asarray(center.transform(scene["pusher_offset"]))
                direction = np.asarray(Pose((0, 0), angle).transform(scene["push_direction"]))
                object_origin = center.transform((0, -45)) if family == "official" else center.position
                goal = Pose(tuple(protocol["goal_official_origin"]), protocol["goal_angle"])
                if family == "gpc":
                    goal = Pose(goal.transform((0,45)), goal.angle)
                initial = replace(template, object_body=BodyState(Pose(object_origin, angle)),
                                  pusher_body=BodyState(Pose(tuple(pusher), 0)), goal=goal,
                                  control_step=0, pusher_target=None)
                restored = env.restore_state(initial)
                counts["state_restores"] += 1
                observation = restored[0] if family == "gpc" else restored
                histories = {view: [image] for view, image in views(observation, family).items()}
                past, warm_contacts = [], []
                for _ in range(protocol["history_controls"]):
                    outcome = env.step(pusher)
                    observation, info = outcome[0], outcome[-1]
                    counts["control_steps"] += 1
                    counts["control_physics_substeps"] += env.sim_hz//env.control_hz
                    warm_contacts.append(int(info["n_contacts"]))
                    past.append(pusher.copy())
                    for view, image in views(observation, family).items():
                        histories[view].append(image)
                if any(warm_contacts):
                    raise ValueError(f"{family}/{scene['name']} has contact in its warmup")
                histories = {view: np.stack(images) for view, images in histories.items()}
                anchor = env.capture_state()
                anchor_pose = anchor.object_body.pose
                initial_roi = object_masks(shape, [(*anchor_pose.position, anchor_pose.angle)])[0,0]
                scene_row = {"family": family, "name": scene["name"], "initial_state": asdict(anchor),
                    "previously_examined": scene["previously_examined"], "warmup_contacts": warm_contacts,
                    "physics": {"mass": env.block.mass, "moment": env.block.moment, "cog": list(env.block.center_of_gravity)},
                    "candidates": []}
                report["scenes"].append(scene_row)
                delta = np.arange(1, protocol["future"]+1)[:,None]*protocol["target_increment_pixels"]
                candidates = {"hold": np.repeat(pusher[None], protocol["future"], axis=0),
                    "push": pusher+delta*direction, "away": pusher-delta*direction,
                    "side": pusher+delta*np.array([-direction[1], direction[0]])}
                for name in protocol["candidates"]:
                    deadline()
                    actions = candidates[name]
                    restored = env.restore_state(anchor)
                    counts["state_restores"] += 1
                    restored = restored[0] if family == "gpc" else restored
                    restored_views = views(restored, family)
                    if any(not np.array_equal(restored_views[v], images[-1]) for v, images in histories.items()):
                        raise ValueError("Cold-restored image differs from the supplied history")
                    actual = {view: [] for view in histories}
                    physical, rois = [], []
                    for action in actions:
                        outcome = env.step(action)
                        observation, info = outcome[0], outcome[-1]
                        counts["control_steps"] += 1
                        counts["control_physics_substeps"] += env.sim_hz//env.control_hz
                        state = env.capture_state()
                        pose = state.object_body.pose
                        rois.append(initial_roi | object_masks(shape, [(*pose.position, pose.angle)])[0,0])
                        physical.append({"state": asdict(state), "n_contacts": int(info["n_contacts"]),
                            "reward": float(outcome[1]), "success": bool(outcome[2]),
                            "object_angle_change_degrees": math.degrees(math.atan2(math.sin(pose.angle-angle), math.cos(pose.angle-angle)))})
                        for view, image in views(observation, family).items():
                            actual[view].append(image)
                        if outcome[2]:
                            break
                    scene_row["candidates"].append({"name": name, "actions": actions.tolist(), "physical": physical})
                    archive = {f"history_{view}": (images*255).round().astype(np.uint8) for view, images in histories.items()}
                    archive.update({f"actual_{view}": (np.stack(images)*255).round().astype(np.uint8) for view, images in actual.items()})
                    archive["actions"] = actions
                    archive["past_actions"] = np.asarray(past)
                    np.savez_compressed(args.output/f"{family}-{scene['name']}-{name}.npz", **archive)
                    for view, images in histories.items():
                        for repeat, base_seed in enumerate(protocol["noise_seeds"]):
                            deadline()
                            noise_seed = base_seed + scene_index
                            raw = model.predict(images, past, actions, noise_seed)
                            counts["model_forecast_steps"] += len(actions)
                            counts["denoising_steps"] += len(actions)*3
                            predicted = np.clip(raw, 0, 1)
                            record = {"family": family, "scene": scene["name"], "candidate": name,
                                "view": view, "seed": noise_seed, "steps": [],
                                "prediction_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                                "raw_range": [float(raw.min()), float(raw.max())],
                                "clipped_fraction": float(((raw < 0)|(raw > 1)).mean())}
                            report["predictions"].append(record)
                            for t, image in enumerate(actual[view]):
                                labels = {key: value for key,value in physical[t].items() if key != "state"}
                                record["steps"].append({"horizon": t+1, **labels,
                                    "model": prediction_metrics(predicted[t], image, images[-1], rois[t], protocol["image_color_tolerance"]),
                                    "persistence": prediction_metrics(images[-1], image, images[-1], rois[t], protocol["image_color_tolerance"])})
                            if family == "official" and name in ("hold", "push") and repeat == 0:
                                paired[(scene["name"], name, view)] = (images[-1], actual[view][-1], predicted[len(physical)-1])
                if not any(step["n_contacts"] for c in scene_row["candidates"] for step in c["physical"]):
                    raise ValueError(f"No contact occurred in {family}/{scene['name']}")
            env.close()
            env = None
        torch.cuda.synchronize()
        report["summary"] = summarize(report["predictions"])
        for scene in protocol["scenes"]:
            for candidate in ("hold", "push"):
                normal = paired[(scene["name"], candidate, "original")]
                palette = paired[(scene["name"], candidate, "source_palette")]
                previews.append((f"{scene['name']} {candidate}; fixed first noise seed", [*normal, *palette]))
        canvas = Image.new("RGB", (6*160, len(previews)*184), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (label, frames) in enumerate(previews):
            draw.text((4,row_index*184+2), label, fill="black")
            for col, frame in enumerate(frames):
                image = Image.fromarray((frame.transpose(1,2,0)*255).round().astype(np.uint8)).resize((160,160))
                canvas.paste(image,(col*160,row_index*184+20))
        canvas.save(args.output/"appearance-comparison.png")
        report["preview_columns"] = ["original current", "original physical future", "original-input prediction",
                                      "palette current", "palette physical future", "palette-input prediction"]
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)
        report["elapsed_seconds"] = time.monotonic()-started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["files"] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": hash_file(p)} for p in sorted(args.output.iterdir())]
        content = json.dumps(report, indent=2, allow_nan=False).encode()+b"\n"
        if len(content) + sum(p["bytes"] for p in report["files"]) > protocol["max_output_bytes"]:
            raise ValueError("Appearance experiment exceeded its output byte budget")
        (args.output/"report.json").write_bytes(content)
    print(json.dumps({"status": report["status"], "counts": counts, "elapsed_seconds": report["elapsed_seconds"]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/gpc_appearance_probe.json")
    parser.add_argument("--gpc-root", type=Path, default=ROOT/"third_party/gpc")
    parser.add_argument("--official-root", type=Path, default=ROOT/"third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/gpc-world-model/denoiser.pth")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in vars(args):
        setattr(args, key, getattr(args,key).resolve())
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    run(args)


if __name__ == "__main__":
    main()
