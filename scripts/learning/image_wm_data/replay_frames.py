"""Regenerate the exact camera frames of every recorded episode by replaying it in the official simulator.

Run from the root of the Tdyssey repository with its environment (needs tdyssey/ and third_party/diffusion_policy):

    python /path/to/replay_frames.py DATA.npz OUT_frames.npz [--size 96] [--workers 16]

Why replay instead of drawing each recorded pose: the official camera image also draws small contact markers that
only exist while the physics is stepping, so only a replay reproduces contact frames exactly. Replay is
deterministic: every step is checked against the recorded pose (tolerance 1e-3 px / rad) and the script stops with
an error if any step differs.

Output: frames (episodes, 301, size, size, 3) uint8 RGB; frames[i, t] is the camera image of state[i, t]. Steps
after the end of an episode (mask False) are zeros.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd()
SHAPES = ("T", "T_narrow", "Y", "D", "L")


def _setup(planner):
    sys.path.insert(0, str(ROOT))
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import load_official_source, physical_state
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party"/"diffusion_policy"))
    return cls, matched_reset, physical_state


def replay(job):
    path, episodes, size, planner = job
    cls, matched_reset, physical_state = _setup(planner)
    z = np.load(path)
    out = np.zeros((len(episodes), z["state"].shape[1], size, size, 3), np.uint8)
    u8 = lambda im: (np.moveaxis(im, 0, -1)*255).round().astype(np.uint8)
    for j, i in enumerate(episodes):
        shape, seed, n = str(z["shape"][i]), int(z["seed"][i]), int(z["mask"][i].sum())
        if planner:   # world-model planner episodes: the shape is reset together with T (L for T itself)
            partner = "L" if shape == "T" else shape
            envs = {"T": cls(shape="T", legacy=True, render_size=size), partner: cls(shape=partner, legacy=True, render_size=size)}
        else:         # base-policy episodes: all five letters reset together (matched scenes)
            envs = {s: cls(shape=s, legacy=True, render_size=size) for s in SHAPES}
        matched_reset(envs, seed, defaultdict(int))
        env = envs[shape]
        for t in range(n):
            st = physical_state(env); o, p = st["object"], st["pusher"]
            err = np.abs(np.array([*o["position"], o["angle"], *p["position"]]) - z["state"][i, t, [0, 1, 2, 6, 7]]).max()
            if err > 1e-3:
                raise RuntimeError(f"{path} episode {i} step {t}: replay differs from the record by {err}")
            out[j, t] = u8(env._get_obs()["image"])
            if t < n-1:
                env.step(z["action"][i, t])
        for e in envs.values():
            e.close()
    return episodes, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    z = np.load(a.data)
    planner = "source" in z.files and "planner" in str(z["source"])
    n_ep = len(z["seed"])
    chunks = [list(range(k, min(k+8, n_ep))) for k in range(0, n_ep, 8)]
    frames = np.zeros((n_ep, z["state"].shape[1], a.size, a.size, 3), np.uint8)
    with ProcessPoolExecutor(a.workers) as pool:
        for eps, out in pool.map(replay, [(str(a.data), c, a.size, planner) for c in chunks]):
            frames[eps] = out
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, frames=frames)
    print(f"{a.data.name}: {n_ep} episodes, {int(z['mask'].sum())} frames verified against the record -> {a.out}")


if __name__ == "__main__":
    main()
