"""Replay saved official decisions and export actual denoising and near-miss views.

No training, new assets or altered action sampling. GPU inference is bounded to
three recorded episodes through the chunk after each episode's best-IoU scene.
"""

import argparse
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics
from tdyssey.policy import hash_file
from tdyssey.policy_diagnostics import near_miss_facts
from tdyssey.state import Snapshot


def check_state(actual, expected):
    """Check numeric body/controller fields; do not silently accept drift."""
    import numpy as np

    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Replayed state keys differ")
        for key, value in expected.items():
            check_state(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Replayed state lengths differ")
        for a, b in zip(actual, expected):
            check_state(a, b)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    elif actual != expected:
        raise ValueError("Replayed state identity differs")


def observation_frame(observation):
    import numpy as np

    return np.rint(observation["image"].transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/official-dp/ema_policy.pt")
    parser.add_argument("--report", type=Path, required=True, help="Existing evaluate_shapes.py report")
    parser.add_argument("--shapes", nargs="+", default=["T", "D", "T_narrow"])
    parser.add_argument("--output", type=Path, required=True, help="New ignored artifact directory")
    parser.add_argument("--seconds", type=int, default=360)
    parser.add_argument("--video", action="store_true", help="Also export one compact denoising MP4 per shape")
    args = parser.parse_args()
    if not 1 <= len(args.shapes) <= 3 or len(set(args.shapes)) != len(args.shapes):
        parser.error("Choose one to three distinct shapes")
    if not 1 <= args.seconds <= 900 or args.output.exists():
        parser.error("Use a new output directory and a 1-900 second compute budget")
    original = json.loads(args.report.read_text())
    if original.get("kind") != "official_cnn_matched_letter_probe" or original.get("status") != "complete":
        parser.error("Use a completed official matched-letter report")
    episodes = []
    for shape in args.shapes:
        matches = [e for e in original["episodes"] if e["shape"] == shape]
        if len(matches) != 1 or not 1 <= len(matches[0]["steps"]) <= 300:
            parser.error("Select a report with one bounded episode per requested shape")
        episodes.append(matches[0])
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    import numpy as np
    import torch
    from tdyssey.policy_visualization import write_policy_views

    torch.set_num_threads(4)
    start = time.monotonic()
    args.output.mkdir(parents=True)
    report = {"schema_version": 1, "kind": "official_policy_decision_replay", "status": "running",
              "source_report": {"name": args.report.name, "sha256": hash_file(args.report)},
              "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: hash_file(ROOT / name) for name in
                                ("tdyssey/official.py", "tdyssey/policy_diagnostics.py",
                                 "tdyssey/policy_visualization.py", "scripts/inspect_official_policy.py",
                                 "tdyssey/policy_trace_template.html")},
              "protocol": {"training": False, "world_model": False, "modified_sampler": False,
                           "selection": "earliest max-IoU plan, previous plan and next plan, posthoc",
                           "policy_input": "two RGB images and pusher positions; no reward or done input",
                           "denoising_display": "raw sampler states, inverse action normalization, no coordinate clipping",
                           "hold_branch": "privileged posthoc intervention: keep pusher target at its peak-time position",
                           "hold_restoration": "replay from initial fresh space; never restore mid-contact state"},
              "counts": {"policy_replay_controls": 0, "policy_calls": 0, "denoising_steps": 0,
                         "hold_prefix_replay_controls": 0, "hold_intervention_controls": 0,
                         "reset_physics_substeps": 0, "initial_state_restores": 0},
              "episodes": []}
    views = []

    def deadline():
        if time.monotonic() - start > args.seconds:
            raise TimeoutError("Diagnostic compute budget exhausted")

    def new_environment(episode):
        deadline()
        env = cls(shape=episode["shape"], legacy=True)
        env.seed(episode["scene_seed"])
        env.reset()
        report["counts"]["reset_physics_substeps"] += 1
        observation = env.restore_state(Snapshot.from_dict(episode["initial_state"]))
        report["counts"]["initial_state_restores"] += 1
        check_state(asdict(env.capture_state()), episode["initial_state"])
        return env, observation

    try:
        policy = OfficialDiffusionPolicy(args.source_root, args.checkpoint)
        report["asset_spec"] = policy.spec
        report["runtime"] = {"device": str(policy.device), "torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
        cls = make_official_shape_env_class(load_official_source(args.source_root))
        for episode in episodes:
            facts = near_miss_facts(episode)
            peak_start = facts["plan_start_control"]
            selected = {p for p in (peak_start - 8, peak_start, peak_start + 8) if 0 <= p < len(episode["steps"])}
            limit = min(len(episode["steps"]), peak_start + 16)
            case = {"shape": episode["shape"], "scene_seed": episode["scene_seed"], "facts": facts,
                    "verified_controls": 0, "verified_plans": 0, "traces": [], "hold_comparison": None}
            report["episodes"].append(case)
            env, observation = new_environment(episode)
            try:
                history = deque([observation] * 2, maxlen=2)
                policy.reset_seed(episode["policy_seed"])
                for control in range(0, limit, 8):
                    deadline()
                    capture = control in selected
                    if capture:
                        actions, trace = policy.predict_actions_with_trace(list(history))
                        trace["conditioning_images"] = np.stack([o["image"] for o in history])
                        trace["conditioning_pusher_positions"] = np.stack([o["agent_pos"] for o in history])
                        frames = [observation_frame(history[-1])]
                        positions = [history[-1]["agent_pos"].tolist()]
                    else:
                        actions = policy.predict_actions(list(history))
                    size = min(8, limit - control)
                    expected = np.asarray([r["action"] for r in episode["steps"][control:control + size]], dtype=np.float32)
                    np.testing.assert_array_equal(actions[:size], expected)
                    report["counts"]["policy_calls"] += 1
                    report["counts"]["denoising_steps"] += policy.network.num_inference_steps
                    case["verified_plans"] += 1
                    for offset, action in enumerate(actions[:size]):
                        deadline()
                        observation, reward, done, info = env.step(action)
                        history.append(observation)
                        row = episode["steps"][control + offset]
                        check_state(asdict(env.capture_state()), row["state"])
                        np.testing.assert_allclose(reward, row["reward"], rtol=0, atol=1e-9)
                        check_state(physical_metrics(env), row["metrics"])
                        if bool(done) != row["upstream_done"] or info["n_contacts"] != row["n_contacts"]:
                            raise ValueError("Replayed contact or termination differs")
                        report["counts"]["policy_replay_controls"] += 1
                        case["verified_controls"] += 1
                        if capture:
                            frames.append(observation_frame(observation))
                            positions.append(observation["agent_pos"].tolist())
                    if capture:
                        stem = f"{episode['shape']}-plan-{control:03d}"
                        arrays = {key: value for key, value in trace.items() if isinstance(value, np.ndarray)}
                        arrays.update(executed_frames=np.stack(frames), pusher_positions=np.asarray(positions))
                        np.savez_compressed(args.output / (stem + ".npz"), **arrays)
                        trace_record = {"file": stem + ".npz", "sha256": hash_file(args.output / (stem + ".npz")),
                                        "start_control": control, "executed_controls": size,
                                        "timesteps": trace["timesteps"], "executed_indices": trace["executed_indices"]}
                        case["traces"].append(trace_record)
                        views.append({"shape": episode["shape"], "start_control": control, "facts": facts,
                                      "episode": episode, "trace": trace, "frames": frames, "positions": positions})
                    print(f"{episode['shape']}: verified controls {control + 1}-{control + size}" +
                          ("; captured all denoising steps" if capture else ""), flush=True)
            finally:
                env.close()

            # Isolate continuing the old action targets from residual physical
            # motion. Rebuild contacts by replaying the full original prefix.
            remaining = min(8, len(episode["steps"]) - facts["best_control"])
            if remaining and not facts["best_upstream_done"]:
                env, observation = new_environment(episode)
                try:
                    for row in episode["steps"][:facts["best_control"]]:
                        deadline()
                        observation, _, _, _ = env.step(np.asarray(row["action"], dtype=np.float32))
                        report["counts"]["hold_prefix_replay_controls"] += 1
                        check_state(asdict(env.capture_state()), row["state"])
                    target = observation["agent_pos"].copy()
                    held = []
                    for _ in range(remaining):
                        deadline()
                        _, reward, done, _ = env.step(target)
                        report["counts"]["hold_intervention_controls"] += 1
                        held.append({"metrics": physical_metrics(env), "reward": float(reward), "done": bool(done)})
                        if done:
                            break
                    endpoint = facts["best_control"] + len(held)
                    case["hold_comparison"] = {"start_control": facts["best_control"], "controls": len(held),
                                               "hold_target": target.tolist(), "hold_results": held,
                                               "original_endpoint_metrics": episode["steps"][endpoint - 1]["metrics"],
                                               "interpretation": "posthoc physical intervention, not a deployable stopping policy"}
                finally:
                    env.close()
        report["status"] = "complete"
        report["counts"]["control_physics_substeps"] = 10 * sum(report["counts"][key] for key in
            ("policy_replay_controls", "hold_prefix_replay_controls", "hold_intervention_controls"))
        report["compute_elapsed_seconds"] = time.monotonic() - start
        write_policy_views(args.output, views, report, video=args.video)
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
