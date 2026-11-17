"""Blend a distilled student with the frozen base: theta = (1-s)*base + s*student for the action network and,
if the student trained it, the image encoder (WiSE-FT-style weight interpolation; s = 0 is exactly the base).
Writes a checkpoint in distill.py's format that base_rollouts.py --student can evaluate.
  .venv/bin/python -B interpolate.py --student RUN/actor-epoch60.pt --s 0.5 --out RUN/blend-0.5.pt
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import base_rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--s", type=float, required=True, help="Weight of the student (0 = base, 1 = student)")
    parser.add_argument("--s-encoder", type=float, default=None, help="Separate student share for the image encoder (default: --s)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.student, map_location="cpu", weights_only=True)
    from dataclasses import replace                           # untouched base weights, same trainable scope as the student
    config = replace(base_rollouts.ForwardConfig(**json.loads((ROOT/"configs/nft_nearmiss_batched.json").read_text())),
                     trainable=payload["trainable"])
    agent = base_rollouts.ForwardPolicy(base_rollouts.OfficialDiffusionPolicy(
        ROOT/"third_party/diffusion_policy", ROOT/"checkpoints/official-dp/ema_policy.pt"), config)
    base_actor = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
    base_enc = {k: v.detach().cpu() for k, v in agent.network.obs_encoder.state_dict().items()}
    def blend(base, student, s):
        return {k: ((1-s)*base[k].float() + s*student[k].float()).to(base[k].dtype) if base[k].is_floating_point()
                else student[k] for k in base}
    s_enc = args.s if args.s_encoder is None else args.s_encoder
    out = {"actor": blend(base_actor, payload["actor"], args.s), "trainable": payload["trainable"], "epoch": payload.get("epoch"),
           "blend": {"student": str(args.student), "s": args.s, "s_encoder": s_enc}}
    if "encoder" in payload:
        out["encoder"] = blend(base_enc, payload["encoder"], s_enc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"wrote": str(args.out), "s": args.s, "encoder": "encoder" in payload}))


if __name__ == "__main__":
    main()
