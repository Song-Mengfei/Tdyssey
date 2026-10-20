"""Bounded original Diffusion Policy CNN evaluation in its unchanged PushT environment."""

import argparse
from collections import deque
import contextlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.collection import derived_seed
from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics, physical_state
from tdyssey.policy import hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/official-dp/ema_policy.pt")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100000, help="First official test-scene seed")
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=360)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="Stream small previews, one per episode")
    args = parser.parse_args()
    if not (1 <= args.episodes <= 3 and 1 <= args.steps <= 300 and 1 <= args.seconds <= 900
            and 100000 <= args.seed < 2**32 - 3 and 0 <= args.policy_seed < 2**32
            and 1 <= args.threads <= 16 and 256 <= args.window_size <= 1536
            and 0 <= args.hold_seconds <= 60 and (not args.hold_seconds or args.display)):
        parser.error("Use bounded runs: 1–3 episodes, 1–300 controls, 1–900 seconds, valid seeds/display options")
    output, source, checkpoint = args.output.resolve(), args.source_root.resolve(), args.checkpoint.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.display and os.environ.get("SDL_VIDEODRIVER") == "dummy":
        parser.error("Remove SDL_VIDEODRIVER=dummy for a desktop window")
    if not args.display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for name, value in {"SDL_AUDIODRIVER": "dummy", "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(name, value)
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    start = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        policy = OfficialDiffusionPolicy(source, checkpoint, device="cuda")
        cls = load_official_source(source, policy.spec)
    paths = ["tdyssey/official.py", "scripts/evaluate_official.py", "tdyssey/viewer.py", "manifests/official_dp.json", "pyproject.toml", "uv.lock"]
    report = {"schema_version": 1, "kind": "official_rss_diffusion_policy_cnn_t_baseline", "status": "running",
              "config": {key: value for key, value in vars(args).items() if key not in ("source_root", "checkpoint", "output")},
              "asset_spec": policy.spec, "inference_device": str(policy.device), "gpu": torch.cuda.get_device_name(),
              "cuda_runtime": torch.version.cuda, "parameter_count": policy.parameter_count,
              "provenance": {"project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                             "project_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
                             "source_sha256": {name: hash_file(ROOT / name) for name in paths},
                             "python": platform.python_version(),
                             "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()}},
              "counts": dict.fromkeys(("control_steps", "control_physics_substeps", "reset_physics_substeps", "planning_calls", "denoising_steps"), 0),
              "episodes": [], "protocol": {"shape": "T", "legacy_test": True, "geometry_rejection": False,
                                            "world_model": False, "training": False,
                                            "metric": "mean episode maximum clipped upstream reward; not success fraction",
                                            "rng": "local CUDA generator seeded per episode; not exact original evaluation RNG stream"}}
    output.mkdir(parents=True, exist_ok=False)
    env, viewer, video = None, None, None
    try:
        env = cls(legacy=policy.spec["evaluation"]["legacy_test"], render_size=96)
        if args.display:
            from tdyssey.viewer import PushTViewer
            viewer = PushTViewer(window_size=args.window_size, hold_seconds=args.hold_seconds)
        for index in range(args.episodes):
            if time.monotonic() >= start + args.seconds or (viewer and viewer.closed):
                break
            scene_seed = args.seed + index
            action_seed = derived_seed(args.policy_seed, index, "official-cnn")
            env.seed(scene_seed)
            observation = env.reset()
            report["counts"]["reset_physics_substeps"] += 1
            history = deque([observation] * 2, maxlen=2)
            policy.reset_seed(action_seed)
            episode = {"scene_seed": scene_seed, "policy_seed": action_seed, "initial_state": physical_state(env),
                       "steps": [], "planning_seconds": [], "stop_reason": "control_budget", "success": False}
            report["episodes"].append(episode)
            if args.video:
                import cv2
                video = cv2.VideoWriter(str(output / f"T-{scene_seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 512))
                if not video.isOpened():
                    raise RuntimeError("Installed OpenCV cannot encode video")

            def draw():
                image = env.render(mode="rgb_array")
                if video is not None:
                    video.write(cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (512, 512)))
                if viewer:
                    viewer.draw(image, f"Official Diffusion Policy | T | seed {scene_seed} | {len(episode['steps'])}/{args.steps} | CUDA", 10)

            draw()
            while len(episode["steps"]) < args.steps:
                if time.monotonic() >= start + args.seconds:
                    episode["stop_reason"] = "wall_time_budget"
                    break
                if viewer and not viewer.poll():
                    episode["stop_reason"] = "viewer_closed"
                    break
                torch.cuda.synchronize()
                before = time.monotonic()
                actions = policy.predict_actions(list(history))
                torch.cuda.synchronize()
                episode["planning_seconds"].append(time.monotonic() - before)
                report["counts"]["planning_calls"] += 1
                report["counts"]["denoising_steps"] += 100
                for action in actions[:args.steps - len(episode["steps"])]:
                    if time.monotonic() >= start + args.seconds or (viewer and not viewer.poll()):
                        episode["stop_reason"] = "viewer_closed" if viewer and viewer.closed else "wall_time_budget"
                        break
                    observation, reward, done, info = env.step(action)
                    history.append(observation)
                    report["counts"]["control_steps"] += 1
                    report["counts"]["control_physics_substeps"] += 10
                    episode["steps"].append({"action": action.tolist(), "reward": float(reward), "done": bool(done),
                                             "contacts": int(info["n_contacts"]), "state": physical_state(env),
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
            episode.update({"control_steps": len(rows), "max_reward": max((r["reward"] for r in rows), default=0),
                            "max_iou": max((r["metrics"]["iou"] for r in rows), default=0),
                            "final_iou": rows[-1]["metrics"]["iou"] if rows else None,
                            "contact_steps": sum(r["contacts"] > 0 for r in rows)})
            print(json.dumps({key: episode[key] for key in ("scene_seed", "control_steps", "max_reward", "max_iou", "success", "stop_reason")}), file=sys.stderr, flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "complete" if (len(report["episodes"]) == args.episodes and all(e["stop_reason"] in ("control_budget", "upstream_success") for e in report["episodes"])) else "stopped_early"
        report["mean_max_upstream_reward"] = float(np.mean([e["max_reward"] for e in report["episodes"]])) if report["episodes"] else None
        report["success_fraction"] = float(np.mean([e["success"] for e in report["episodes"]])) if report["episodes"] else None
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
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.monotonic() - start
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "counts", "mean_max_upstream_reward", "success_fraction", "evaluation_elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
