"""Collect a bounded, non-expert GPC episode dataset; no installs or asset downloads."""

import argparse
from dataclasses import asdict
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

from tdyssey.collection import CollectionConfig, collect_episode
from tdyssey.dataset import DatasetWriter, digest, load_dataset
from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.trajectories import split_trajectories


def provenance(module, revision):
    paths = ["tdyssey/collection.py", "tdyssey/dataset.py", "tdyssey/geometry.py", "tdyssey/state.py",
             "tdyssey/trajectories.py", "tdyssey/evaluation.py", "tdyssey/envs/gpc.py",
             "scripts/collect_play.py", "examples/gpc_shape_env.py", "pyproject.toml", "uv.lock"]
    return {
        "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "project_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
        "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in paths},
        "upstream_commit": revision, "upstream_source_sha256": digest(Path(module.__file__).read_bytes()),
        "upstream_domain_sha256": digest((Path(module.__file__).parent / "domains_yaml/push_t.yml").read_bytes()),
        "python": platform.python_version(), "platform": f"{sys.platform}-{platform.machine()}",
        "torch_cuda_runtime": module.torch.version.cuda,
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "privileged_labels": ["object state", "velocities", "contact count", "geometric reward and scores"],
        "action_prior": "Object position used only in object-directed mode; neither mode optimizes reward.",
        "goal_protocol": {"position_pixels": [256, 256], "angle_radians": 0.7853981633974483},
        "contact_semantics": "GPC ceil(mean contact-point count over the control step); not unique pairs or impulses",
        "reset_protocol": "Reject unsafe geometry, fix goal, restore fresh space; count GPC reset substep separately",
        "image_semantics": "GPC rendered goal overlay retained; each CHW frame stored once, no history windows",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory; existing directories are refused")
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--shape", choices=("T", "I", "L"), default="L")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100, help="Maximum actions per episode, at most 250")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--mode", choices=("random-waypoints", "object-directed"), default="random-waypoints")
    parser.add_argument("--max-mib", type=int, default=16, help="Published-file byte limit, at most 64 MiB")
    parser.add_argument("--validate-only", action="store_true", help="Verify an existing dataset without simulating")
    args = parser.parse_args()
    if args.validate_only:
        manifest, episodes = load_dataset(args.output)
        print(json.dumps({"status": "verified", "episodes": len(episodes), "counts": manifest["counts"]}, indent=2))
        return
    try:
        config = CollectionConfig(shape=args.shape, episodes=args.episodes, steps=args.steps,
                                  seed=args.seed, mode=args.mode, max_bytes=args.max_mib * 1024 * 1024)
    except ValueError as exc:
        parser.error(str(exc))
    output, gpc_root = args.output.resolve(), args.gpc_root.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    from examples.gpc_shape_env import load_upstream

    module, revision = load_upstream(gpc_root)
    writer = DatasetWriter(output, config, provenance(module, revision))
    start, previous_cwd, env = time.monotonic(), Path.cwd(), None
    try:
        os.chdir(gpc_root / "gpc_rank_evaluation")
        env = make_shape_env_class(module.PushTImageEnv)(shape=config.shape, domain_filename="push_t", resize_scale=96)
        writer.manifest["rates"] = {"simulation_hz": env.sim_hz, "control_hz": env.control_hz}
        trajectories = []
        for index in range(config.episodes):
            episode = collect_episode(env, config, index, writer.manifest["counts"])
            writer.add(episode)
            trajectories.append(episode.trajectory)
        splits = {name: [record.trajectory_id for record in records]
                  for name, records in split_trajectories(trajectories, config.seed).items()}
        writer.manifest["elapsed_seconds"] = time.monotonic() - start
        writer.finish(splits)
        # Read every persisted record back and verify hashes/timing, not just file creation.
        manifest, verified = load_dataset(output)
        print(json.dumps({"status": "complete", "kind": "non_expert_play_not_policy_evaluation",
                          "episodes": len(verified), "counts": manifest["counts"],
                          "contact_steps": sum(row["contact_steps"] for row in manifest["episodes"]),
                          "output_bytes": sum(p.stat().st_size for p in output.iterdir()),
                          "output": str(output), "config": asdict(config)}, indent=2))
    except BaseException as exc:
        if writer.closed:
            # Post-write verification failure must not leave the run labelled complete.
            writer.closed = False
        writer.manifest["elapsed_seconds"] = time.monotonic() - start
        writer.finish(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
