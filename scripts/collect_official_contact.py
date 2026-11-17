"""Collect many official T batches with contact-balanced privileged play.

Each batch stays inside the 1000-control dataset cap. This does not train a
policy or a world model. The geometric prior is disclosed in the manifest.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_official_play import provenance
from tdyssey.collection import integer
from tdyssey.official_contact_sampler import (ACTION_PRIOR, DEFAULT_BATCHES, DEFAULT_SEED,
                                              collect_contact_batches, preserve_init_checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Root directory for finished batch-* folders")
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy",
                        help="Official PushT source (default: third_party/diffusion_policy)")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES,
                        help="Finished batches to keep (default: 40)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Collection seed (default: 240000)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Collector processes (default: 2)")
    parser.add_argument("--seconds", type=int, default=300,
                        help="Wall-time budget per batch in seconds (default: 300)")
    parser.add_argument("--episodes", type=int, default=8,
                        help="Episodes per batch (default: 8)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Controls per episode (default: 100)")
    parser.add_argument("--preserve-init", action="store_true",
                        help="Copy the current best denoiser to artifacts/official-wm-T-init")
    parser.add_argument("--resume", action="store_true",
                        help="Skip finished batch directories and continue until --batches")
    args = parser.parse_args(argv)
    try:
        integer(args.batches, "batches", 1, 20000)
        integer(args.episodes, "episodes", 7, 20)
        integer(args.steps, "steps", 4, 250)
        if args.preserve_init:
            preserved = preserve_init_checkpoint(root=ROOT)
            print(json.dumps({"init_checkpoint": str(preserved["checkpoint"]),
                              "copied": preserved["copied"]}, indent=2), flush=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    def note(config):
        payload = provenance(config)
        payload["action_prior"] = ACTION_PRIOR
        payload["source_sha256"] = {**payload.get("source_sha256", {}),
                                    "tdyssey/official_contact_sampler.py": _file_digest("tdyssey/official_contact_sampler.py"),
                                    "tdyssey/motion_replay.py": _file_digest("tdyssey/motion_replay.py"),
                                    "scripts/collect_official_contact.py": _file_digest("scripts/collect_official_contact.py")}
        return payload

    report = collect_contact_batches(
        args.output, args.source_root, note, batches=args.batches, seed=args.seed,
        workers=args.workers, seconds=args.seconds, episodes=args.episodes, steps=args.steps,
        resume=args.resume)
    print(json.dumps({"status": report["status"], "controls": report["controls"],
                      "contact_steps": report["contact_steps"],
                      "batches": len(report["batches"])}, indent=2))
    return report


def _file_digest(relative):
    from tdyssey.dataset import digest

    return digest((ROOT / relative).read_bytes())


if __name__ == "__main__":
    main()
