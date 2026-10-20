"""Plan or fetch just the verified phase-two GPC world-model checkpoint."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_policy_assets import fetch_checkpoint
from tdyssey.gpc_world_model import world_model_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/gpc-world-model")
    parser.add_argument("--max-mib", type=int, default=40)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_mib <= 64:
        parser.error("Use a download budget of 1–64 MiB")
    asset = world_model_spec()["checkpoint"]
    destination = args.output / asset["filename"]
    status = fetch_checkpoint(asset, destination, args.max_mib * 1024**2) if args.download else "plan_only"
    print(json.dumps({"status": status, **asset, "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
