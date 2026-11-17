"""Continue one GPC denoiser on finished contact-balanced official T batches.

Each chunk samples train windows from every finished batch and scores a frozen
validation subset. Ctrl+C keeps the last finished weights. The same command
with --resume continues. This does not train a policy.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.collection import integer
from tdyssey.official_contact_sampler import preserve_init_checkpoint
from tdyssey.official_contact_trainer import (DEFAULT_POOL_WINDOWS, DEFAULT_SEED, DEFAULT_UPDATES,
                                              DEFAULT_VAL_WINDOWS, train_contact_batches)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True,
                        help="Root of finished official contact-balanced batch-* directories")
    parser.add_argument("--output", type=Path, required=True,
                        help="Train directory for denoiser.pt, denoiser.json and report.json")
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc",
                        help="Pinned GPC source (default: third_party/gpc)")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gpc-world-model/denoiser.pth",
                        help="Weights copied on a new run; ignored once output/denoiser.pt exists")
    parser.add_argument("--init-from-best", action="store_true",
                        help="Start from artifacts/official-wm-T-init if this is a new run")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a previous output directory, or start if it does not exist")
    parser.add_argument("--pool-windows", type=int, default=DEFAULT_POOL_WINDOWS,
                        help="Train windows sampled from every batch each chunk (default: 64)")
    parser.add_argument("--validation-windows", type=int, default=DEFAULT_VAL_WINDOWS,
                        help="Frozen validation windows (default: 64, scored as 32 useful / 32 miss)")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES,
                        help="AdamW steps per chunk (default: 8, range 1-20)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Windows per optimizer step (default: 4, range 1-8)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Sampling and frozen-validation seed (default: 250000)")
    parser.add_argument("--train-seconds", type=int, default=900,
                        help="Per-chunk wall-time limit in seconds (default: 900)")
    parser.add_argument("--budget-seconds", type=int, default=0,
                        help="Stop after this many seconds; 0 means no cap")
    parser.add_argument("--max-updates", type=int, default=0,
                        help="Stop after this many total updates; 0 means no cap")
    args = parser.parse_args(argv)
    try:
        integer(args.pool_windows, "pool windows", 1, 512)
        integer(args.validation_windows, "validation windows", 1, 512)
        integer(args.updates, "updates", 1, 20)
        integer(args.batch_size, "batch size", 1, 8)
        integer(args.seed, "seed", 0, 2**32 - 1)
        integer(args.train_seconds, "train seconds", 1, 900)
        integer(args.budget_seconds, "budget seconds", 0, 86400)
        integer(args.max_updates, "max updates", 0, 1_000_000)
        checkpoint = args.checkpoint
        if args.init_from_best and not (args.resume and (args.output / "denoiser.pt").is_file()):
            checkpoint = preserve_init_checkpoint(root=ROOT)["checkpoint"]
            print(json.dumps({"init_checkpoint": str(checkpoint)}), flush=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    report = train_contact_batches(
        args.datasets, args.output, checkpoint, args.gpc_root, resume=args.resume,
        pool_windows=args.pool_windows, validation_windows=args.validation_windows,
        updates=args.updates, batch_size=args.batch_size, seed=args.seed,
        train_seconds=args.train_seconds, budget_seconds=args.budget_seconds,
        max_updates=args.max_updates)
    last = report["rounds"][-1] if report["rounds"] else {}
    print(json.dumps({"status": report["status"], "updates": report["updates"],
                      "elapsed_seconds": report["elapsed_seconds"],
                      "validation_mse": last.get("validation_mse"),
                      "persistence_mse": last.get("persistence_mse"),
                      "object_roi_mse": last.get("object_roi_mse"),
                      "motion_mse": last.get("motion_mse"),
                      "object_centroid_error_pixels": last.get("object_centroid_error_pixels"),
                      "useful_windows": last.get("useful_windows"),
                      "miss_windows": last.get("miss_windows"),
                      "useful_motion_mse": last.get("useful_motion_mse"),
                      "useful_motion_persistence": last.get("useful_motion_persistence"),
                      "useful_centroid_error_pixels": last.get("useful_centroid_error_pixels"),
                      "useful_predicted_centroid_shift_pixels": last.get(
                          "useful_predicted_centroid_shift_pixels"),
                      "useful_actual_centroid_shift_pixels": last.get(
                          "useful_actual_centroid_shift_pixels"),
                      "miss_object_roi_mse": last.get("miss_object_roi_mse"),
                      "miss_object_roi_persistence": last.get("miss_object_roi_persistence"),
                      "miss_predicted_centroid_shift_pixels": last.get(
                          "miss_predicted_centroid_shift_pixels"),
                      "train_windows": report.get("train_windows"),
                      "validation_windows": len(report["validation_keys"])}, indent=2))
    if report["status"] == "interrupted":
        sys.exit(130)
    if report["status"] == "failed":
        sys.exit(1)
    return report


if __name__ == "__main__":
    main()
