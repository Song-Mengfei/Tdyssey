"""Evaluate a saved PPO/forward-policy delta, with optional desktop/video output."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT/"third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/official-dp/ema_policy.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")+("T_narrow",), default="L")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument("--hold-seconds", type=int, default=20)
    parser.add_argument("--reference-report", type=Path, help="Verify against saved final evaluation")
    parser.add_argument("--video", type=Path, help="Optional new MP4 path, independent of desktop preview")
    parser.add_argument("--max-video-mib", type=int, default=32)
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.steps <= 500 or not 1 <= args.seconds <= 900
            or not 0 <= args.seed < 2**32-20 or not 256 <= args.window_size <= 1536
            or not 0 <= args.hold_seconds <= 60):
        parser.error("Use a new output path and bounded seed/control/time/display options")
    if (not 1 <= args.max_video_mib <= 128 or args.video and
            (args.video.exists() or args.video.suffix != ".mp4" or args.video == args.output)):
        parser.error("Use a new .mp4 video path and 1-128 MiB video limit")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if not args.display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import torch
    from tdyssey.collection import derived_seed
    from tdyssey.dppo import TailPPO
    from tdyssey.forward_policy import ForwardPolicy
    from tdyssey.forward_protocol import ForwardConfig
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics
    from tdyssey.policy import hash_file
    from tdyssey.rl_protocol import RLConfig
    from tdyssey.viewer import presentation_frame

    torch.set_num_threads(4)
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    if payload["kind"] == "official_tail_ppo":
        config, agent_class, label = RLConfig(**payload["config"]), TailPPO, "tail-PPO"
    elif payload["kind"] == "official_forward_rl":
        config, agent_class = ForwardConfig(**payload["config"]), ForwardPolicy
        label = config.objective
    else:
        parser.error("Unknown trained policy checkpoint type")
    start = time.monotonic()
    report = {"kind": "trained_tail_ppo_physical_preview" if agent_class is TailPPO else "trained_forward_rl_physical_preview",
              "method": label, "status": "running", "training": False,
              "shape": args.shape, "requested_seed": args.seed, "steps": [],
              "adapter_sha256": hash_file(args.adapter), "display": args.display,
              "counts": {"reset_physics_substeps": 0, "controls": 0, "plans": 0}}
    reference = None
    if args.reference_report:
        old = json.loads(args.reference_report.read_text())
        if old["checkpoint"]["sha256"] != report["adapter_sha256"]:
            raise ValueError("Reference report belongs to another checkpoint")
        reference = next(e for e in old["final"] if e["shape"] == args.shape and e["requested_seed"] == args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    envs, viewer, video = {}, None, None
    video_frames = 0

    def record_video(env):
        nonlocal video, video_frames
        import cv2
        frame = presentation_frame(env)
        if video is None:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"),
                                    env.control_hz, (frame.shape[1], frame.shape[0]))
            if not video.isOpened():
                raise ValueError("Cannot open MP4 writer")
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_frames += 1
        if args.video.stat().st_size > args.max_video_mib*1024**2:
            raise ValueError("Video output budget exhausted")
    try:
        official = OfficialDiffusionPolicy(args.source_root, args.checkpoint)
        agent = agent_class(official, config)
        agent.load(payload)
        cls = make_official_shape_env_class(load_official_source(args.source_root))
        shapes = getattr(config, "eval_shapes", ("T", "L"))
        if args.shape not in shapes:
            shapes = ("T", args.shape)
        envs = {shape: cls(shape=shape, legacy=True) for shape in shapes}
        observations, attempts = matched_reset(envs, args.seed, report["counts"])
        report["reset_attempts"] = attempts
        env = envs[args.shape]
        history = deque([observations[args.shape]]*2, maxlen=2)
        if args.video:
            record_video(env)
        agent.generator.manual_seed(derived_seed(config.seed, "rl-eval", args.seed))
        if args.display:
            from tdyssey.viewer import PushTViewer
            viewer = PushTViewer(window_size=args.window_size, hold_seconds=args.hold_seconds)
        done = False
        while len(report["steps"]) < args.steps and not done:
            if time.monotonic()-start > args.seconds:
                raise TimeoutError("Preview wall-time budget exhausted")
            if viewer and not viewer.poll():
                break
            actions, _ = agent.sample(agent.encode(history))
            report["counts"]["plans"] += 1
            for action in actions[:min(8, args.steps-len(report["steps"]))]:
                observation, reward, done, _ = env.step(action)
                history.append(observation)
                row = {"action": action.tolist(), "raw_reward": float(reward), "done": bool(done),
                       **physical_metrics(env)}
                if reference:
                    expected = reference["steps"][len(report["steps"])]
                    np.testing.assert_array_equal(action, np.asarray(expected["action"], dtype=np.float32))
                    for key in ("raw_reward", "coverage", "iou"):
                        np.testing.assert_allclose(row[key], expected[key], rtol=0, atol=1e-9)
                    if row["done"] != expected["done"]:
                        raise AssertionError("Saved termination differs")
                report["steps"].append(row)
                report["counts"]["controls"] += 1
                if args.video:
                    record_video(env)
                if viewer:
                    caption = (f"Trained {label} | {args.shape} | {len(report['steps'])}/{args.steps} | "
                               f"coverage {row['coverage']:.1%} | CUDA | preview, no training")
                    if not viewer.draw(presentation_frame(env), caption, env.control_hz):
                        break
                if done:
                    break
            if viewer and viewer.closed:
                break
        complete = done or len(report["steps"]) == args.steps
        report.update(status="complete" if complete else "closed_early", success=bool(done),
                      reference_prefix_verified=bool(reference),
                      final_coverage=report["steps"][-1]["coverage"] if report["steps"] else None)
        if viewer:
            viewer.finish()
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if video:
            video.release()
            report["video"] = {"frames": video_frames, "bytes": args.video.stat().st_size,
                               "sha256": hash_file(args.video)}
            if report["video"]["bytes"] > args.max_video_mib*1024**2:
                report["status"], report["error"] = "failed", "Video output budget exceeded"
                args.video.unlink()
                report["video"]["retained"] = False
        if viewer:
            viewer.close()
        for env in envs.values():
            env.close()
        report["elapsed_seconds"] = time.monotonic()-start
        report["counts"]["control_physics_substeps"] = report["counts"]["controls"]*10
        report["counts"]["denoising_steps"] = report["counts"]["plans"]*100
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "success", "counts", "reference_prefix_verified")}))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
