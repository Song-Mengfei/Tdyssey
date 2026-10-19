"""Later-machine smoke check using the existing GPC simulator, not a new engine.

Requires an authorized environment and an existing pinned GPC source checkout.
No checkpoint/data download, policy training, or environment installation occurs.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.evaluation import goal_metrics
from tdyssey.geometry import get_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpc-root", type=Path, required=True, help="Existing pinned GPC checkout")
    parser.add_argument("--shape", choices=("T", "I", "L"), default="T")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 100 or args.seed < 0:
        parser.error("Use 1–100 smoke steps and a nonnegative seed")
    source_root = args.gpc_root.resolve()
    source_file = source_root / "gpc_rank_evaluation/pusht_env.py"
    domain_file = source_file.parent / "domains_yaml/push_t.yml"
    manifest = json.loads((ROOT / "manifests/upstream.json").read_text())
    expected = next(repo["commit"] for repo in manifest["repositories"] if repo["name"] == "gpc")
    try:
        actual = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"Expected GPC revision {expected}; found {actual}")
        for path in (source_file, domain_file):
            relative = path.relative_to(source_root).as_posix()
            published = subprocess.check_output(["git", "-C", str(source_root), "show", f"HEAD:{relative}"])
            if path.read_bytes() != published:
                raise ValueError(f"Source baseline has an unrecorded local patch: {relative}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))

    # Upstream resolves its domain YAML relative to process cwd. Keep this change
    # in the standalone command, not in the adapter/library or parallel workers.
    previous_cwd, env = Path.cwd(), None
    try:
        os.chdir(source_file.parent)
        spec = importlib.util.spec_from_file_location("tdyssey_gpc_upstream", source_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        environment_class = make_shape_env_class(module.PushTImageEnv)
        env = environment_class(shape=args.shape, domain_filename="push_t", resize_scale=96)
        env.seed(args.seed)
        observation, _ = env.reset()
        state = env.capture_state()
        shape = get_shape(args.shape, env.block_scale)
        rewards = []
        for _ in range(args.steps):
            # Holding the current pusher position tests the interface. This is
            # explicitly not a diffusion policy or a task-solving demonstration.
            observation, reward, terminated, truncated, info = env.step(observation["agent_pos"].copy())
            rewards.append(float(reward))
            state = env.capture_state()
            if terminated or truncated:
                break
        image = observation["image"]
        if image.shape != (3, 96, 96) or float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError("Unexpected upstream image format/range")
        print(json.dumps({
            "kind": "simulator_interface_smoke_not_policy_evaluation",
            "upstream_commit": actual,
            "upstream_source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "shape": args.shape, "geometry_hash": shape.geometry_hash,
            "control_steps": state.control_step, "image_shape": list(image.shape),
            "scores": asdict(goal_metrics(shape, state.object_body.pose, state.goal)),
            "last_upstream_reward": rewards[-1], "last_contact_count": int(info["n_contacts"]),
        }, indent=2, allow_nan=False))
    except ImportError as exc:
        raise SystemExit(f"Missing simulator dependency: {exc}. Set up the agreed profile on an authorized machine.") from exc
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
