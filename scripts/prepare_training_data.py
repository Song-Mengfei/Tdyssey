"""Freeze dataset membership and verify small CPU batches; no model training."""

import argparse
from itertools import islice
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.training_data import SPLITS, WindowDataset, WindowSpec, write_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--datasets", nargs="+", type=Path, help="Verified collector run directories")
    source.add_argument("--catalog", type=Path, help="Use an existing frozen catalog")
    parser.add_argument("--output", type=Path, help="New catalog file when using --datasets")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--future", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        spec = WindowSpec(args.history, args.future)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.batch_size <= 32 or not 1 <= args.max_batches <= 10 or not 0 <= args.seed < 2**32:
        parser.error("Use batch size 1–32, at most ten verification batches, and a uint32 seed")
    if args.datasets and args.output is None:
        parser.error("--datasets requires --output for the new catalog")
    if args.catalog and args.output is not None:
        parser.error("--output is only used when creating a catalog")
    catalog_path = args.catalog or args.output
    if args.datasets:
        write_catalog(args.datasets, catalog_path)
    datasets = {split: WindowDataset(catalog_path, split, spec) for split in SPLITS}
    selected = datasets[args.split]

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=args.split == "train" and len(selected) > 0,
                        num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    shapes, batches, examples = {}, 0, 0
    for batch in islice(loader, args.max_batches):
        for group in ("observations", "targets"):
            for value in batch[group].values():
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("Non-finite or unexpected training observation dtype")
        if not torch.isfinite(batch["actions"]).all():
            raise ValueError("Non-finite action targets")
        shapes = {"history_images": list(batch["observations"]["image"].shape),
                  "future_images": list(batch["targets"]["image"].shape),
                  "actions": list(batch["actions"].shape)}
        batches += 1
        examples += len(batch["episode_id"])
    print(json.dumps({"kind": "data_path_validation_not_model_training", "verified_batches": batches,
                      "verified_examples": examples, "last_batch_shapes": shapes,
                      "splits": {split: dataset.summary() for split, dataset in datasets.items()},
                      "validation_has_windows": len(datasets["validation"]) > 0,
                      "notes": ["Coordinates/actions retain pixel units; images are float32 in [0,1].",
                                "Empty splits stay empty; no reassignment or padding.",
                                "Reward/contact/overlap labels are privileged simulator supervision."]}, indent=2))


if __name__ == "__main__":
    main()
