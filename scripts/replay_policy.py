"""Export a recorded baseline to a small video; verify replay without loading weights."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.policy import hash_file
from tdyssey.state import Snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .mp4 file; no overwrites")
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--shape", choices=("T", "I", "L"), default="T")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    output, report_path, gpc_root = args.output.resolve(), args.report.resolve(), args.gpc_root.resolve()
    sidecar = output.with_suffix(".json")
    if output.suffix != ".mp4" or output.exists() or sidecar.exists():
        parser.error("Choose a new .mp4 path with no existing video/report")
    report = json.loads(report_path.read_text())
    if report["kind"] != "unranked_pretrained_policy_baseline":
        parser.error("Expected a baseline report")
    matches = [e for e in report["episodes"] if e["shape"] == args.shape and e["episode_index"] == args.episode]
    if len(matches) != 1 or not 1 <= len(matches[0]["steps"]) <= 300:
        parser.error("Choose one recorded episode containing 1–300 actions")
    episode = matches[0]
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "MPLBACKEND": "Agg",
                       "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    import cv2
    import numpy as np
    from examples.gpc_shape_env import load_upstream

    module, revision = load_upstream(gpc_root)
    if revision != report["provenance"]["upstream_commit"]:
        parser.error("Recorded upstream revision differs from the checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    old_cwd, env, writer = Path.cwd(), None, None
    frames = 0
    try:
        os.chdir(gpc_root / "gpc_rank_evaluation")
        env = make_shape_env_class(module.PushTImageEnv)(shape=args.shape, domain_filename="push_t", resize_scale=96)
        env.seed(100000)
        env.reset()
        env.restore_state(Snapshot.from_dict(episode["initial_state"]))
        image = env.render(mode="rgb_array")
        height, width = image.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), env.control_hz, (width, height))
        if not writer.isOpened():
            raise RuntimeError("The installed OpenCV build cannot encode mp4v")

        def write_frame(step):
            nonlocal frames
            image = cv2.cvtColor(env.render(mode="rgb_array"), cv2.COLOR_RGB2BGR)
            cv2.putText(image, f"T-trained policy | {args.shape} | step {step}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            writer.write(image)
            frames += 1

        write_frame(0)
        for index, row in enumerate(episode["steps"], 1):
            _, reward, terminated, truncated, _ = env.step(np.asarray(row["action"], dtype=np.float64))
            expected, actual = Snapshot.from_dict(row["state"]), env.capture_state()
            for name in ("object_body", "pusher_body"):
                before, after = getattr(expected, name), getattr(actual, name)
                np.testing.assert_allclose([*before.pose.position, before.pose.angle, *before.velocity, before.angular_velocity],
                                           [*after.pose.position, after.pose.angle, *after.velocity, after.angular_velocity],
                                           atol=1e-7, rtol=0)
            if abs(reward - row["reward"]) > 1e-9 or bool(terminated) != row["terminated"] or bool(truncated) != row["truncated"]:
                raise ValueError("Reward/terminal flags diverged during replay")
            write_frame(index)
    finally:
        if writer is not None:
            writer.release()
        if env is not None:
            env.close()
        os.chdir(old_cwd)
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != frames:
        raise ValueError("Encoded video frame count failed read-back")
    result = {"status": "verified", "shape": args.shape, "episode_index": args.episode,
              "video_sha256": hash_file(output), "source_report_sha256": hash_file(report_path),
              "frames": frames, "fps": report["rates"]["control_hz"], "bytes": output.stat().st_size,
              "diagnostic_replay_control_steps": len(episode["steps"]), "reset_physics_substeps": 1,
              "new_policy_predictions": 0, "pose_velocity_tolerance": 1e-7}
    sidecar.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
