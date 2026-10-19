"""Compare frozen official T-trained CNN policy on matched capital-letter scenes."""

import argparse
from collections import deque
import contextlib
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.collection import derived_seed
from tdyssey.letters import ALL_SHAPES
from tdyssey.envs.official import PROTOCOL, make_official_shape_env_class, matched_reset, official_shape
from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics
from tdyssey.policy import hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/official-dp/ema_policy.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1, help="Number of common scenes (1–3)")
    parser.add_argument("--shapes", nargs="+", choices=ALL_SHAPES, default=None,
                        help="Include T and at least one other shape; default T L")
    parser.add_argument("--alphabet", action="store_true", help="All 26 capitals plus T_narrow")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=360)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.alphabet and args.shapes:
        parser.error("Choose --alphabet or --shapes")
    args.shapes = list(ALL_SHAPES) if args.alphabet else (args.shapes or ["T", "L"])
    if len(set(args.shapes)) != len(args.shapes) or "T" not in args.shapes or len(args.shapes) < 2:
        parser.error("Include T and at least one other shape, without duplicates")
    args.shapes = ["T", *[name for name in args.shapes if name != "T"]]
    if not (1 <= args.pairs <= 3 and 1 <= args.steps <= 300 and 1 <= args.seconds <= 7200
            and 100000 <= args.seed <= 2**32 - 61 and 0 <= args.policy_seed < 2**32
            and 1 <= args.threads <= 16 and 256 <= args.window_size <= 1536
            and 0 <= args.hold_seconds <= 60 and (not args.hold_seconds or args.display)):
        parser.error("Use 1–3 pairs, 1–300 controls per shape, valid seeds, and bounded display/time options")
    output, source, checkpoint = args.output.resolve(), args.source_root.resolve(), args.checkpoint.resolve()
    if output.exists():
        parser.error("Output exists; choose a new directory")
    if args.display and os.environ.get("SDL_VIDEODRIVER") == "dummy":
        parser.error("Remove SDL_VIDEODRIVER=dummy to open a desktop window")
    if not args.display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for key, value in {"SDL_AUDIODRIVER": "dummy", "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    start = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        policy = OfficialDiffusionPolicy(source, checkpoint)
        cls = make_official_shape_env_class(load_official_source(source, policy.spec))
    paths = ["tdyssey/envs/official.py", "tdyssey/official.py", "tdyssey/geometry.py", "tdyssey/state.py",
             "tdyssey/collection.py", "tdyssey/letters.py", "scripts/evaluate_shapes.py", "manifests/official_dp.json", "pyproject.toml", "uv.lock"]
    report = {"schema_version": 1, "kind": "official_cnn_matched_letter_probe", "status": "running",
              "config": {k: v for k, v in vars(args).items() if k not in ("output", "source_root", "checkpoint")},
              "protocol": {"id": PROTOCOL if args.shapes == ["T", "L"] else "official-capital-sweep-com-v1", "mass": 1, "L_height": 120, "L_width": 90, "L_stroke": 30,
                           "initial_matching": "full physical state at equal mass centers, orientation, pusher, and target mass center",
                           "reset_selection": "common rotated geometry/clearance checks; reject for all selected shapes",
                           "T_construction": "unchanged official mass/inertia/CoG and legacy reset",
                           "L_construction": "area-centroid coordinates and uniform-density inertia; same total mass",
                           "other_letters": "versioned 5x7 edge-connected blocks, 100x120; mass 1; uniform-density inertia",
                           "T_narrow": "108-pixel top bar vs original 120; original mass 1, inertia 3000, physical-center convention",
                           "policy": "frozen official T-trained CNN; no world model, target-shape data, or training",
                           "source_flags": "unchanged upstream done; separate runner stop reason"},
              "asset_spec": policy.spec, "inference_device": str(policy.device), "gpu": torch.cuda.get_device_name(),
              "provenance": {"project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                             "project_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
                             "source_sha256": {name: hash_file(ROOT / name) for name in paths},
                             "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()}},
              "counts": dict.fromkeys(("control_steps", "control_physics_substeps", "reset_physics_substeps", "planning_calls", "denoising_steps"), 0),
              "reset_records": [], "episodes": []}
    output.mkdir(parents=True, exist_ok=False)
    environments, viewer, video = {}, None, None
    try:
        environments = {name: cls(shape=name, legacy=True) for name in args.shapes}
        report["body_parameters"] = {}
        if args.display:
            from tdyssey.viewer import PushTViewer
            viewer = PushTViewer(window_size=args.window_size, hold_seconds=args.hold_seconds)
        candidate = args.seed
        for index in range(args.pairs):
            if time.monotonic() >= start + args.seconds or (viewer and viewer.closed):
                break
            observations, attempts = matched_reset(environments, candidate, report["counts"])
            report["reset_records"].append(attempts)
            scene_seed, candidate = attempts[-1]["seed"], attempts[-1]["seed"] + 1
            action_seed = derived_seed(args.policy_seed, index, "official-cnn")
            for name, env in environments.items():
                if time.monotonic() >= start + args.seconds or (viewer and viewer.closed):
                    break
                report["body_parameters"][name] = {"geometry_hash": official_shape(name).geometry_hash,
                                                    "mass": env.block.mass, "moment": env.block.moment,
                                                    "center_of_gravity": list(env.block.center_of_gravity),
                                                    "symmetry_order": official_shape(name).symmetry_order,
                                                    "rectangles": [asdict(r) for r in official_shape(name).components]}
                policy.reset_seed(action_seed)
                history = deque([observations[name]] * 2, maxlen=2)
                episode = {"shape": name, "pair_index": index, "scene_seed": scene_seed, "policy_seed": action_seed,
                           "initial_state": asdict(env.capture_state()), "steps": [], "planning_seconds": [],
                           "stop_reason": "control_budget", "success": False}
                report["episodes"].append(episode)
                if args.video:
                    import cv2
                    video = cv2.VideoWriter(str(output / f"{name}-{scene_seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 512))
                    if not video.isOpened():
                        raise RuntimeError("Installed OpenCV cannot encode video")

                def draw():
                    image = env.render(mode="rgb_array")
                    if video is not None:
                        video.write(cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (512, 512)))
                    if viewer:
                        viewer.draw(image, f"Official T-trained policy | {name} | seed {scene_seed} | {len(episode['steps'])}/{args.steps} | CUDA", 10)

                draw()
                while len(episode["steps"]) < args.steps:
                    if time.monotonic() >= start + args.seconds or (viewer and not viewer.poll()):
                        episode["stop_reason"] = "viewer_closed" if viewer and viewer.closed else "wall_time_budget"
                        break
                    torch.cuda.synchronize()
                    planned = time.monotonic()
                    actions = policy.predict_actions(list(history))
                    torch.cuda.synchronize()
                    episode["planning_seconds"].append(time.monotonic() - planned)
                    report["counts"]["planning_calls"] += 1
                    report["counts"]["denoising_steps"] += 100
                    for action in actions[:args.steps - len(episode["steps"])]:
                        if time.monotonic() >= start + args.seconds or (viewer and not viewer.poll()):
                            episode["stop_reason"] = "viewer_closed" if viewer and viewer.closed else "wall_time_budget"
                            break
                        observation, reward, done, info = env.step(action)
                        history.append(observation)
                        report["counts"]["control_steps"] += 1
                        report["counts"]["control_physics_substeps"] += env.sim_hz // env.control_hz
                        episode["steps"].append({"action": action.tolist(), "reward": float(reward), "upstream_done": bool(done),
                                                 "n_contacts": int(info["n_contacts"]), "state": asdict(env.capture_state()),
                                                 "metrics": physical_metrics(env)})
                        draw()
                        if done:
                            episode["success"], episode["stop_reason"] = True, "upstream_success"
                            break
                    if episode["stop_reason"] != "control_budget":
                        break
                if video is not None:
                    video.release()
                    video = None
                rows = episode["steps"]
                episode.update({"control_steps": len(rows), "max_iou": max((r["metrics"]["iou"] for r in rows), default=0),
                                "final_iou": rows[-1]["metrics"]["iou"] if rows else None,
                                "max_reward": max((r["reward"] for r in rows), default=0),
                                "contact_steps": sum(r["n_contacts"] > 0 for r in rows)})
                print(json.dumps({k: episode[k] for k in ("shape", "scene_seed", "control_steps", "max_iou", "success", "stop_reason")}), file=sys.stderr, flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "complete" if (len(report["episodes"]) == len(args.shapes) * args.pairs and all(e["stop_reason"] in ("control_budget", "upstream_success") for e in report["episodes"])) else "stopped_early"
        report["evaluation_elapsed_seconds"] = time.monotonic() - start
        if viewer:
            viewer.finish()
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if video is not None:
            video.release()
        if viewer:
            viewer.close()
        for env in environments.values():
            env.close()
        report["elapsed_seconds"] = time.monotonic() - start
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "evaluation_elapsed_seconds": report["evaluation_elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
