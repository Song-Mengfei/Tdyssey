"""Collect bounded official letter-shape play; no checkpoint, download or training."""

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

from tdyssey.dataset import DatasetWriter, digest
from tdyssey.envs.official import PROTOCOL, make_official_shape_env_class, official_shape
from tdyssey.official import load_official_source, official_spec
from tdyssey.official_collection import (DEVELOPMENT_SEEDS, ENVIRONMENT_FAMILY, VERSION,
    OfficialCollectionConfig, collect_official_episode, freeze_splits, validate_official_dataset)


def provenance(config):
    paths = ["tdyssey/official_collection.py", "tdyssey/collection.py", "tdyssey/envs/official.py", "tdyssey/official.py",
             "tdyssey/geometry.py", "tdyssey/letters.py", "tdyssey/state.py", "tdyssey/dataset.py", "tdyssey/trajectories.py",
             "tdyssey/evaluation.py", "scripts/collect_official_play.py", "manifests/official_dp.json", "pyproject.toml", "uv.lock"]
    spec = official_spec()
    return {"environment_family": ENVIRONMENT_FAMILY, "collector_version": VERSION,
            "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "project_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
            "source_sha256": {name: digest((ROOT/name).read_bytes()) for name in paths},
            "upstream_commit": spec["source_commit"], "upstream_source_sha256": spec["source_sha256"],
            "geometry_hash": official_shape(config.shape).geometry_hash, "physics_protocol": PROTOCOL,
            "python": platform.python_version(), "platform": f"{sys.platform}-{platform.machine()}",
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
            "privileged_labels": ["full physical state", "contact count", "geometric reward/IoU/coverage"],
            "action_prior": "object-directed uses privileged object position with 75% waypoint probability; no reward optimization",
            "observation_protocol": "official RGB/agent_pos; unchanged goal overlay; CHW uint8 frames stored once",
            "termination": "unwrapped original done is success; collector horizon is truncated, not terminal",
            "reset_protocol": "official fixed goal and centroid mapping, reject unsafe scenes; fresh restore; one substep per simulated reset",
            "excluded_evaluation_reset_seeds": list(config.excluded_reset_seeds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT/'third_party/diffusion_policy')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape', choices=tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')+('T_narrow',), default='L')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=200000)
    parser.add_argument('--mode', choices=('random-waypoints','object-directed'), default='object-directed')
    parser.add_argument('--max-mib', type=int, default=32)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--exclude-seed', action='append', type=int, default=[])
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        manifest, episodes = validate_official_dataset(args.output)
        print(json.dumps({'status':'verified','episodes':len(episodes),'counts':manifest['counts'],
                          'splits':{name:len(ids) for name,ids in manifest['splits'].items()}},indent=2))
        return
    try:
        config = OfficialCollectionConfig(shape=args.shape, episodes=args.episodes, steps=args.steps,
            seed=args.seed, mode=args.mode, max_bytes=args.max_mib*1024*1024, seconds=args.seconds,
            excluded_reset_seeds=tuple(sorted(set(DEVELOPMENT_SEEDS)|set(args.exclude_seed))))
        if config.episodes < 7:
            raise ValueError('Use at least seven episodes for nonempty whole-episode splits')
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    for key,value in {'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','MPLBACKEND':'Agg','PYGAME_HIDE_SUPPORT_PROMPT':'1'}.items():
        os.environ.setdefault(key,value)
    start = time.monotonic()
    cls = make_official_shape_env_class(load_official_source(args.source_root))
    writer = DatasetWriter(args.output, config, provenance(config))
    env = None
    try:
        env = cls(shape=config.shape, legacy=True)
        writer.manifest['rates'] = {'simulation_hz':env.sim_hz,'control_hz':env.control_hz}
        trajectories = []
        for index in range(config.episodes):
            episode = collect_official_episode(env, config, index, writer.manifest['counts'], start+config.seconds)
            writer.add(episode)
            trajectories.append(episode.trajectory)
        writer.manifest['elapsed_seconds'] = time.monotonic()-start
        writer.finish(freeze_splits(trajectories, config.seed))
        manifest, loaded = validate_official_dataset(args.output)
        print(json.dumps({'status':'complete','kind':'official_non_expert_play_not_training',
              'episodes':len(loaded),'counts':manifest['counts'],
              'contact_steps':sum(e['contact_steps'] for e in manifest['episodes']),
              'splits':{name:len(ids) for name,ids in manifest['splits'].items()},
              'output_bytes':sum(p.stat().st_size for p in args.output.iterdir()),'config':asdict(config)},indent=2))
    except BaseException as exc:
        writer.closed = False
        writer.manifest['elapsed_seconds'] = time.monotonic()-start
        writer.finish(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
