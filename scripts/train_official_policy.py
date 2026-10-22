"""Train a bounded direct-simulator visual tail-PPO pilot on CUDA."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.rl_protocol import RLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT/"third_party/diffusion_policy")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/official-dp/ema_policy.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = RLConfig(**json.loads(args.config.read_text()))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "PYGAME_HIDE_SUPPORT_PROMPT": "1", "MPLBACKEND": "Agg"}.items():
        os.environ.setdefault(key, value)
    from tdyssey.policy_training import train
    report = train(args.source_root, args.checkpoint, args.output, config)
    print(json.dumps({k: report[k] for k in ("status", "counts", "elapsed_seconds")}))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
