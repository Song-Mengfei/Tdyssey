"""Minimal loader (numpy only): turn the package into (image_t, action_t, image_t+1) training examples.

    python scripts/load_example.py            # run from the package folder; prints a summary

    from load_example import load_pairs
    train, val = load_pairs(".", history=1, contact_only=False, val_fraction=0.1)
    train["image"]       (N, history, 96, 96, 3) uint8   frames t-history+1 .. t
    train["action"]      (N, 2) float32                  pusher target (simulator pixels, 0-512) applied at step t
    train["next_image"]  (N, 96, 96, 3) uint8            frame t+1
    train["state"], train["next_state"]  (N, 9)          exact poses, for evaluation (see README)
    train["contact"]     (N,) bool                       pusher touched the letter during step t

The validation split holds out whole scenes (scene seeds), so a starting layout never appears in both splits.
Everything is loaded into memory: about 55 KB per example with history 1, so roughly 35-40 GB for all data. Pass
`parts` (e.g. ["part0", "planner"]) to load a subset, or index the frames arrays directly in your own data loader.
"""
from pathlib import Path
import sys

import numpy as np


def load_pairs(root=".", parts=None, history=1, contact_only=False, val_fraction=0.1, split_seed=0):
    root = Path(root)
    files = sorted((root/"data").glob("*/*.npz"))
    if parts:
        files = [f for f in files if any(p in f"{f.parent.name}/{f.name}" for p in parts)]   # e.g. "part0", "planner/"
    rng = np.random.default_rng(split_seed)
    all_seeds = sorted({(f.parent.name, int(s)) for f in files for s in np.load(f)["seed"]})
    val_seeds = {all_seeds[k] for k in rng.permutation(len(all_seeds))[:int(val_fraction*len(all_seeds))]}
    out = {"train": {k: [] for k in ("image", "action", "next_image", "state", "next_state", "contact", "shape")},
           "val": {k: [] for k in ("image", "action", "next_image", "state", "next_state", "contact", "shape")}}
    for f in files:
        with np.load(f) as zf:                                # read each array once (np.load is lazy per key)
            z = {k: zf[k] for k in zf.files}
        frames = np.load(root/"frames"/f.parent.name/f"{f.stem}_frames.npz")["frames"]
        for i in range(len(z["seed"])):
            n = int(z["mask"][i].sum())                       # valid states 0..n-1, actions 0..n-2
            split = "val" if (f.parent.name, int(z["seed"][i])) in val_seeds else "train"
            for t in range(history-1, n-1):
                contact = bool(z["contacts"][i, t+1] > 0)
                if contact_only and not contact:
                    continue
                d = out[split]
                d["image"].append(frames[i, t-history+1:t+1]); d["action"].append(z["action"][i, t])
                d["next_image"].append(frames[i, t+1]); d["state"].append(z["state"][i, t])
                d["next_state"].append(z["state"][i, t+1]); d["contact"].append(contact); d["shape"].append(str(z["shape"][i]))
    return tuple({k: np.stack(v) if v else np.zeros(0) for k, v in out[s].items()} for s in ("train", "val"))


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    train, val = load_pairs(root, parts=["part0-noise0"])
    for name, d in (("train", train), ("val", val)):
        print(f"{name}: {len(d['action'])} examples from part0-noise0; image {d['image'].shape}, next_image "
              f"{d['next_image'].shape}; contact {d['contact'].mean():.0%}; letters {sorted(set(d['shape']))}")
