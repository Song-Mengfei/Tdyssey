"""Camera frames with the D066 goal overlay for the training data that can be re-rendered (CPU only).

  demos  Replay every teacher demonstration file DIR/*.npz in the simulator exactly as replay_frames.py does (every
         step checked against the recorded pose, stops on any difference), rendering with --goal-overlay R, into
         DIR/frames-ovR/<stem>_frames.npz (frames<size>-ovR/ when --size is not 96; bc_train.frames_subdir). Files whose
         output exists are skipped; the episode chunks of all files share one process pool.
  human  Render the official human T demonstrations from their recorded states (agent x, y, block x, y, angle; the
         dataset's legacy pose convention) with the project's renderer and the overlay, into
         ZARR's folder/frames-ovR/<zarr stem>_frames.npz (uint8, one frame per dataset step). The dataset's own images
         come from an older renderer that is not reproduced exactly (block fill and edges differ on about 0.5% of the
         pixels; no contact markers), so these frames replace them only when training with the overlay.

Correction records keep the frames they were collected with and, before D066, no block pose, so they cannot be
re-rendered; bc_train refuses to mix them with overlay frames unless --mixed-records is given (docs/GOAL_OVERLAY.md).

  python -B scripts/learning/overlay_frames.py demos --goal-overlay 9 --workers 32 $W/runs/teacher-plan-demos*
  python -B scripts/learning/overlay_frames.py human --goal-overlay 9 --workers 32 $W/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent


def subdir(size, overlay):
    sys.path.insert(0, str(HERE))
    from bc_train import frames_subdir                                   # one naming rule for the builder and loader
    return frames_subdir(SimpleNamespace(image_size=size, goal_overlay=overlay))


def _replay(job):
    os.chdir(ROOT)                                                       # replay_frames finds tdyssey/ from the cwd
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(HERE/"image_wm_data"))
    import replay_frames
    return replay_frames.replay(job)


def demos(a):
    sub = subdir(a.size, a.goal_overlay)
    files = sorted(f for d in a.dirs for f in Path(d).glob("*.npz") if not (f.parent/sub/f"{f.stem}_frames.npz").exists())
    print(f"{len(files)} demonstration files to replay into */{sub}/", flush=True)
    pending, done_frames = deque(), 0

    def finish(f, n_ep, length, futures):
        frames = np.zeros((n_ep, length, a.size, a.size, 3), np.uint8)
        for fut in futures:
            eps, out = fut.result()
            frames[eps] = out
        out_dir = f.parent/sub; out_dir.mkdir(exist_ok=True)
        tmp = out_dir/f"{f.stem}_frames.tmp.npz"
        np.savez_compressed(tmp, frames=frames)
        tmp.rename(out_dir/f"{f.stem}_frames.npz")                      # never leave a half-written frames file
        return int(sum(frames.any(axis=(2, 3, 4)).sum(1)))

    with ProcessPoolExecutor(a.workers) as pool:
        for f in files:
            with np.load(f) as z:
                n_ep, length = len(z["seed"]), z["state"].shape[1]
                planner = "source" in z.files and "planner" in str(z["source"])
            chunks = [list(range(k, min(k + 8, n_ep))) for k in range(0, n_ep, 8)]
            pending.append((f, n_ep, length, [pool.submit(_replay, (str(f), c, a.size, planner, a.goal_overlay)) for c in chunks]))
            while len(pending) > max(2, a.workers // 6):                 # bounded memory: a few files in flight
                done_frames += finish(*pending.popleft()); print(f"[demos] {done_frames} frames written", flush=True)
        while pending:
            done_frames += finish(*pending.popleft()); print(f"[demos] {done_frames} frames written", flush=True)
    print(f"done: {len(files)} files", flush=True)


def _render_human(job):
    zarr_path, start, stop, size, overlay = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(ROOT))
    import zarr
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party"/"diffusion_policy"))
    env = cls(shape="T", legacy=True, render_size=size, **({"goal_overlay": overlay} if overlay else {}))
    env.seed(0); env.reset()
    state = zarr.open(str(zarr_path), "r")["data"]["state"][start:stop]
    out = np.zeros((stop - start, size, size, 3), np.uint8)
    for k, s in enumerate(np.asarray(state, np.float64)):
        env.agent.position, env.agent.velocity = (s[0], s[1]), (0.0, 0.0)
        env.block.position = (s[2], s[3])                                # legacy order (the dataset's convention):
        env.block.angle = s[4]                                           # position first, then angle
        env.block.velocity, env.block.angular_velocity = (0.0, 0.0), 0.0
        env.space.reindex_shapes_for_body(env.block); env.space.reindex_shapes_for_body(env.agent)
        out[k] = (np.moveaxis(env._get_obs()["image"], 0, -1) * 255).round().astype(np.uint8)
    env.close()
    return start, out


def human(a):
    import zarr
    n = len(zarr.open(str(a.zarr), "r")["data"]["state"])
    out_dir = a.zarr.parent/subdir(a.size, a.goal_overlay); out_dir.mkdir(exist_ok=True)
    target = out_dir/f"{a.zarr.stem}_frames.npz"
    if target.exists():
        print(f"{target} exists; nothing to do"); return
    frames = np.zeros((n, a.size, a.size, 3), np.uint8)
    jobs = [(a.zarr, s, min(s + 1000, n), a.size, a.goal_overlay) for s in range(0, n, 1000)]
    with ProcessPoolExecutor(a.workers) as pool:
        for start, out in pool.map(_render_human, jobs):
            frames[start:start + len(out)] = out
    tmp = out_dir/f"{a.zarr.stem}_frames.tmp.npz"
    np.savez_compressed(tmp, frames=frames); tmp.rename(target)
    print(f"{n} human T frames -> {target}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("demos"); d.add_argument("dirs", type=Path, nargs="+")
    h = sub.add_parser("human"); h.add_argument("zarr", type=Path)
    for q in (d, h):
        q.add_argument("--goal-overlay", type=int, required=True, help="Overlay radius in simulator px (D066: 9)")
        q.add_argument("--size", type=int, default=96)
        q.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.goal_overlay == 0:
        raise SystemExit("--goal-overlay must not be 0 (the official frames already exist; < 0 = slivers only)")
    demos(a) if a.mode == "demos" else human(a)


if __name__ == "__main__":
    main()
