"""Run the pinned T-trained policy on matched T/I/L scenes; never installs/downloads."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.baseline import matched_resets, rollout, validate_budget
from tdyssey.collection import derived_seed
from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.policy import GPCDiffusionPolicy, hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gpc-policy/ema_nets.pth")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--shapes", nargs="+", choices=("T", "I", "L"), default=["T", "I", "L"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--seconds", type=int, default=240, help="Run-wide wall-time limit; an active denoising call finishes")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        validate_budget(args.shapes, args.episodes, args.steps, args.seed, args.seconds)
        if not 1 <= args.threads <= 16:
            raise ValueError("Use 1–16 CPU threads")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new directory")
    except ValueError as exc:
        parser.error(str(exc))
    gpc_root, checkpoint, output = args.gpc_root.resolve(), args.checkpoint.resolve(), args.output.resolve()
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "MPLBACKEND": "Agg",
                       "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    import torch
    from examples.gpc_shape_env import load_upstream
    from scripts.collect_play import provenance

    torch.set_num_threads(args.threads)
    started = time.monotonic()
    policy = GPCDiffusionPolicy(gpc_root, checkpoint, args.device)
    module, revision = load_upstream(gpc_root)
    report = {"schema_version": 1, "status": "running", "kind": "unranked_pretrained_policy_baseline",
              "provenance": provenance(module, revision), "asset_spec": policy.spec,
              "parameter_signature": policy.parameter_signature, "ema_step": policy.ema_step,
              "config": {key: value for key, value in vars(args).items() if key not in ("gpc_root", "checkpoint", "output")},
              "counts": dict.fromkeys(("reset_attempts", "reset_physics_substeps", "control_steps",
                                       "control_physics_substeps", "planning_calls", "denoising_steps"), 0),
              "matched_resets": [], "episodes": []}
    # Replace collector-specific provenance statements with the evaluation protocol.
    report["provenance"].update({"goal_protocol": "Unchanged upstream randomized goal, shared across shapes",
                                 "reset_protocol": "T-state reference; all shapes must pass geometry checks; fresh spaces",
                                 "action_prior": "One sampled pretrained diffusion action chunk; no object-state action input"})
    paths = ["tdyssey/policy.py", "tdyssey/baseline.py", "scripts/evaluate_policy.py", "manifests/gpc_policy.json"]
    report["provenance"]["source_sha256"].update({name: hash_file(ROOT / name) for name in paths})
    output.mkdir(parents=True, exist_ok=False)
    old_cwd, environments = Path.cwd(), {}
    try:
        os.chdir(gpc_root / "gpc_rank_evaluation")
        cls = make_shape_env_class(module.PushTImageEnv)
        for name in dict.fromkeys(["T", *args.shapes]):
            environments[name] = cls(shape=name, domain_filename="push_t", resize_scale=96)
        reference = environments["T"]
        report["rates"] = {"simulation_hz": reference.sim_hz, "control_hz": reference.control_hz}
        for index in range(args.episodes):
            if time.monotonic() - started >= args.seconds:
                break
            observations, resets = matched_resets(environments, args.seed + 20 * index, report["counts"])
            report["matched_resets"].append(resets)
            for name in args.shapes:
                result = rollout(environments[name], policy, observations[name], args.steps,
                                 derived_seed(args.seed, index, "policy"), started + args.seconds, report["counts"])
                result["episode_index"] = index
                report["episodes"].append(result)
                print(json.dumps({k: result[k] for k in ("shape", "control_steps", "contact_steps", "max_iou", "stop_reason")}), file=sys.stderr, flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "complete" if (len(report["episodes"]) == args.episodes * len(args.shapes)
                                           and all(e["stop_reason"] != "wall_time_budget" for e in report["episodes"])) else "budget_exhausted"
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for env in environments.values():
            env.close()
        os.chdir(old_cwd)
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "elapsed_seconds": report["elapsed_seconds"], "output": str(output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
