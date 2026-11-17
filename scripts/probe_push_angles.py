"""Compare one-step predictions at 16 object-relative directions and three distances.

Each branch starts from the same official T scene. Two sampler seeds show whether
a missing shape is stable. The real next frame is one simulator control.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.angle_probe import summarize_cases, world_side, worse_object_iou
from tdyssey.appearance import DISPLAY_OBJECT_RGB, mask_iou, palette_mask
from tdyssey.gpc_denoiser_training import load_denoiser_sampler, predict_next_frame
from tdyssey.motion_replay import object_displacement
from tdyssey.world_model_click import conditioning_window


DISTANCES = (20.0, 60.0, 150.0)


def target_for(pusher, angle, direction, distance):
    import math
    theta = angle + direction * (math.pi / 8)
    raw = (pusher[0] + distance * math.cos(theta), pusher[1] + distance * math.sin(theta))
    return tuple(min(511.0, max(0.0, value)) for value in raw)


def frame_of(observation):
    import numpy as np
    return np.asarray(observation["image"], dtype=np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--seed", type=int, default=100003)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists")
    import numpy as np
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source

    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        import os
        os.environ.setdefault(key, value)
    sampler, device = load_denoiser_sampler(args.gpc_root, args.checkpoint)
    env = make_official_shape_env_class(load_official_source(args.source_root))(shape="T", legacy=True)
    env.seed(args.seed)
    initial = env.reset()
    anchor = env.capture_state()
    hold = tuple(float(value) for value in initial["agent_pos"])
    current = frame_of(initial)
    args.output.mkdir()
    rows = []
    try:
        import matplotlib.pyplot as plt
        for direction in range(16):
            panels = []
            for distance in DISTANCES:
                target = target_for(hold, anchor.object_body.pose.angle, direction, distance)
                env.restore_state(anchor)
                actual_obs, _reward, _done, info = env.step(np.asarray(target, dtype=np.float64))
                after = env.capture_state()
                actual = frame_of(actual_obs)
                translation, _rotation = object_displacement(anchor, after)
                env.restore_state(anchor)
                images, actions = conditioning_window([current], [], target, hold)
                predictions = [predict_next_frame(sampler, np.stack(images), np.stack(actions), device, seed=noise)
                               for noise in (0, 1)]
                true_mask = palette_mask(actual, DISPLAY_OBJECT_RGB)
                still = mask_iou(true_mask, palette_mask(current, DISPLAY_OBJECT_RGB))
                scores = [float(mask_iou(true_mask, palette_mask(predicted, DISPLAY_OBJECT_RGB)))
                          for predicted in predictions]
                worse = worse_object_iou(scores)
                rows.append({
                    "direction": direction, "distance": distance, "target": target,
                    "object_iou": scores, "worse_object_iou": worse,
                    "still_object_iou": 0.0 if still is None else float(still),
                    "n_contacts": int(info["n_contacts"]),
                    "world_side": world_side(target, anchor.object_body.pose.position),
                    "object_translation": translation,
                    "stable_missing": all(score < 0.3 for score in scores),
                })
                panels.append((actual, *predictions))
            figure, axes = plt.subplots(3, 3, figsize=(6, 6))
            for row, (actual, first, second) in enumerate(panels):
                for column, image in enumerate((actual, first, second)):
                    axes[row, column].imshow(np.clip(image, 0, 1).transpose(1, 2, 0))
                    axes[row, column].axis("off")
            figure.savefig(args.output / f"direction-{direction:02d}.png", dpi=80)
            plt.close(figure)
            print(json.dumps(rows[-3:]), flush=True)
    finally:
        env.close()
    missing = [row for row in rows if row["stable_missing"]]
    slices = summarize_cases(rows)
    report = {"checkpoint": str(args.checkpoint), "scene_seed": args.seed, "cases": rows,
              "stable_missing": missing, "stable_missing_count": len(missing),
              "privileged_labels": True, "slices": slices}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stable_missing_count": len(missing), "cases": len(rows),
                      "slices": slices}, indent=2))


if __name__ == "__main__":
    main()
