"""Standard Diffusion Policy training on demonstrations (the official recipe), for "can the student match the teacher".

The teacher's episodes are treated like human demonstrations: every step is a training sample (official windowing:
2 observation frames, 16-step action window starting one step back, padding 1 before / 7 after), the official
loss (DiffusionUnetHybridImagePolicy.compute_loss, which includes the official random-crop augmentation in train
mode), AdamW (lr 1e-4, betas 0.95/0.999, wd 1e-6) with 500 warm-up steps and cosine decay, and the official EMA
schedule (power 0.75, max 0.9999). Starts from the official checkpoint and keeps its normalizer. Data:
  --demos DIR...   teacher trajectories (mpc_teacher.py --save-traj *.npz) with replayed frames in DIR/frames/
  --human-t ZARR   the official human T demonstrations (pusht_cchi_v7_replay.zarr), so T is learned from the
                   data the original policy was trained on (not new demonstrations)
Letters are sampled in equal shares (T split evenly between human and teacher episodes).
Saves EMA checkpoints readable by base_rollouts.py eval --student.

  .venv/bin/python -B bc_train.py --demos $W/runs/teacher-noisy-demos --human-t $W/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr \
      --steps 15000 --save-every 5000 --out $W/runs/bc/v1
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from tdyssey.official import OfficialDiffusionPolicy  # noqa: E402

LETTERS = ("T", "T_narrow", "Y", "D", "L")


def load_episodes(args, poses=False):
    return list(iter_episodes(args, poses))


def iter_episodes(args, poses=False):
    """Yield (frames uint8 (m,S,S,3), agent_pos (m,2), actions (m,2), group, coverage, plan) one episode at a time.
    poses=True appends the block pose of every frame, (m, 3) x, y, angle (privileged labels for flow_train's pose head)."""
    sub = "frames" if args.image_size == 96 else f"frames{args.image_size}"
    rng = np.random.default_rng(1234)
    for f in sorted(f for d in args.demos for f in Path(d).glob("*.npz")):
        ff = f.parent/sub/f"{f.stem}_frames.npz"
        if not ff.exists():                                              # e.g. a file that arrived after its frames pass
            print(f"skipped {f.name}: no {sub}/ file", flush=True); continue
        z = {k: v for k, v in np.load(f).items()}
        fr = np.load(ff)["frames"]
        for i in range(len(z["seed"])):
            n = int(z["mask"][i].sum()); m = n - 1                         # m (frame, action) pairs
            if m < 2 or rng.random() >= args.data_fraction:               # --data-fraction: fixed random subset
                continue
            plan = z["plan"][i, :m] if (args.label == "plan" and "plan" in z) else None
            item = (fr[i, :m], z["state"][i, :m, 6:8], z["action"][i, :m], f"{z['shape'][i]}-teacher", z["coverage"][i, :m], plan)
            yield item + (z["state"][i, :m, 0:3],) if poses else item
        del fr
    if args.human_t:
        import zarr
        g = zarr.open(str(args.human_t), "r")
        img, st, act = g["data"]["img"][:], g["data"]["state"][:], g["data"]["action"][:]
        ends = g["meta"]["episode_ends"][:]
        for s, e in zip(np.concatenate([[0], ends[:-1]]), ends):
            item = (resize_frames(img[s:e].round().clip(0, 255).astype(np.uint8), args.image_size), st[s:e, :2], act[s:e],
                    "T-human", None, None)                                  # 96 px; enlarged for 192 px students
            yield item + (st[s:e, 2:5],) if poses else item                 # zarr state: agent x, y, block x, y, angle


def resize_frames(img, size):
    """Enlarge/shrink uint8 frames (..., H, W, 3) to size x size (bilinear), e.g. 96 px records for a 192 px student."""
    import cv2
    if img.shape[-2] == size:
        return img
    flat = img.reshape(-1, *img.shape[-3:])
    out = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in flat])
    return out.reshape(*img.shape[:-3], size, size, 3)


def load_records(dirs, unnormalize, size=96):
    return list(iter_records(dirs, unnormalize, size))


def iter_records(dirs, unnormalize, size=96):
    """DAgger records (mpc_dagger.py): 2 frames + the teacher's 16-step label chunk (normalized) at visited states.
    Yields (frames (2,S,S,3) uint8, agent_pos (2,2), actions (16,2) pixels, group) one record at a time."""
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            with np.load(f) as z:
                if "candidate_x" not in z.files or len(z["choice"]) == 0:
                    continue
                n = len(z["choice"])
                lab = z["candidate_x"][np.arange(n), z["choice"]].astype(np.float32)          # (n, 16, 2) normalized
                acts = unnormalize(lab)
                img = resize_frames(np.moveaxis(z["image"], 2, -1), size)                     # (n, 2, S, S, 3)
                for r in range(n):
                    yield (img[r], z["agent_pos"][r].astype(np.float32), acts[r], f"{z['shape']}-dagger")


# ------------------------------------------------------------------------------------- parallel file loading
# The serial generators above decompress every file on one core (~35 min for the 26-letter set). With
# TDYSSEY_LOAD_WORKERS=N the same files are read by N worker processes; the main process consumes them in the original
# order and does everything else exactly as the serial code (episode selection with the same random stream, the same
# unnormalize call per file), so the yielded items, and the built arrays, are identical.

def _load_demo_file(f, sub):
    ff = f.parent/sub/f"{f.stem}_frames.npz"
    if not ff.exists():
        return None
    z = {k: v for k, v in np.load(f).items()}
    z["__frames"] = np.load(ff)["frames"]
    return z


def _load_record_file(f, size):
    with np.load(f) as z:
        if "candidate_x" not in z.files or len(z["choice"]) == 0:
            return None
        n = len(z["choice"])
        lab = z["candidate_x"][np.arange(n), z["choice"]].astype(np.float32)
        img = resize_frames(np.moveaxis(z["image"], 2, -1), size)
        return lab, img, z["agent_pos"].astype(np.float32), str(z["shape"])


def _ordered(fn, items, workers, *extra):
    """Results of fn(item, *extra) in item order, at most 2*workers in flight (bounded host memory)."""
    import collections, concurrent.futures, multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        q, it = collections.deque(), iter(items)
        for item in it:
            q.append((item, ex.submit(fn, item, *extra)))
            if len(q) >= 2 * workers:
                break
        while q:
            item, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(fn, nxt, *extra)))
            yield item, fut.result()


def iter_episodes_parallel(args, workers, poses=False):
    """iter_episodes with the demo files read by worker processes (same items, same order)."""
    sub = "frames" if args.image_size == 96 else f"frames{args.image_size}"
    rng = np.random.default_rng(1234)
    files = sorted(f for d in args.demos for f in Path(d).glob("*.npz"))
    for f, z in _ordered(_load_demo_file, files, workers, sub):
        if z is None:
            print(f"skipped {f.name}: no {sub}/ file", flush=True); continue
        fr = z.pop("__frames")
        for i in range(len(z["seed"])):
            n = int(z["mask"][i].sum()); m = n - 1
            if m < 2 or rng.random() >= args.data_fraction:
                continue
            plan = z["plan"][i, :m] if (args.label == "plan" and "plan" in z) else None
            item = (fr[i, :m], z["state"][i, :m, 6:8], z["action"][i, :m], f"{z['shape'][i]}-teacher", z["coverage"][i, :m], plan)
            yield item + (z["state"][i, :m, 0:3],) if poses else item
        del fr
    if args.human_t:
        import zarr
        g = zarr.open(str(args.human_t), "r")
        img, st, act = g["data"]["img"][:], g["data"]["state"][:], g["data"]["action"][:]
        ends = g["meta"]["episode_ends"][:]
        for s, e in zip(np.concatenate([[0], ends[:-1]]), ends):
            item = (resize_frames(img[s:e].round().clip(0, 255).astype(np.uint8), args.image_size), st[s:e, :2], act[s:e],
                    "T-human", None, None)
            yield item + (st[s:e, 2:5],) if poses else item


def iter_records_parallel(dirs, unnormalize, size, workers):
    """iter_records with the record files read by worker processes (same items, same order)."""
    files = [f for d in dirs for f in sorted(Path(d).glob("*.npz"))]
    for f, r in _ordered(_load_record_file, files, workers, size):
        if r is None:
            continue
        lab, img, apos, shape = r
        acts = unnormalize(lab)
        for k in range(len(lab)):
            yield (img[k], apos[k], acts[k], f"{shape}-dagger")


class _NoFrames:
    """Stands in for frames during build_index's bookkeeping pass: has a length, ignores writes."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __setitem__(self, key, value):
        pass


def _demo_meta(f, sub, plan_label):
    if not (f.parent/sub/f"{f.stem}_frames.npz").exists():
        return None
    with np.load(f) as z:
        keep = ["seed", "mask", "state", "action", "coverage", "shape"] + (["plan"] if plan_label and "plan" in z.files else [])
        return {k: z[k] for k in keep}


def _record_meta(f):
    with np.load(f) as z:
        if "candidate_x" not in z.files or len(z["choice"]) == 0:
            return None
        n = len(z["choice"])
        return z["candidate_x"][np.arange(n), z["choice"]].astype(np.float32), z["agent_pos"].astype(np.float32), str(z["shape"])


def _demo_write(job, sub, path):
    f, spans = job                                                       # spans: (episode, length, offset)
    fr = np.load(f.parent/sub/f"{f.stem}_frames.npz")["frames"]
    mm = np.load(path, mmap_mode="r+")
    for i, m, off in spans:
        mm[off:off+m] = fr[i, :m]
    mm.flush()
    return len(spans)


def _record_write(job, size, path):
    f, off = job
    with np.load(f) as z:
        img = resize_frames(np.moveaxis(z["image"], 2, -1), size)
    mm = np.load(path, mmap_mode="r+")
    mm[off:off + 2*len(img)] = img.reshape(-1, *img.shape[2:])
    mm.flush()
    return len(img)


def build_streamed_parallel(args, unnormalize, out, workers, poses=None):
    """build_index(iter_episodes(args), iter_records(...), out=out) for a file-backed out, with the files read and the
    frames written by worker processes. The bookkeeping is build_index itself (run on frame stand-ins), with the same
    episode selection and the same unnormalize call per record file, so every returned array is identical.
    poses: a list, filled as build_index(..., poses=poses) fills it (block pose labels; NaN for records)."""
    sub = "frames" if args.image_size == 96 else f"frames{args.image_size}"
    rng = np.random.default_rng(1234)
    files = sorted(f for d in args.demos for f in Path(d).glob("*.npz"))
    eps, demo_jobs, off = [], [], 0
    for f, z in _ordered(_demo_meta, files, workers, sub, args.label == "plan"):
        if z is None:
            print(f"skipped {f.name}: no {sub}/ file", flush=True); continue
        spans = []
        for i in range(len(z["seed"])):
            n = int(z["mask"][i].sum()); m = n - 1
            if m < 2 or rng.random() >= args.data_fraction:
                continue
            plan = z["plan"][i, :m] if "plan" in z else None
            eps.append((_NoFrames(m), z["state"][i, :m, 6:8], z["action"][i, :m], f"{z['shape'][i]}-teacher", z["coverage"][i, :m], plan)
                       + ((z["state"][i, :m, 0:3],) if poses is not None else ()))
            spans.append((i, m, off)); off += m
        demo_jobs.append((f, spans))
    human = []
    if args.human_t:
        import zarr
        g = zarr.open(str(args.human_t), "r")
        img, st, act = g["data"]["img"][:], g["data"]["state"][:], g["data"]["action"][:]
        ends = g["meta"]["episode_ends"][:]
        for s, e in zip(np.concatenate([[0], ends[:-1]]), ends):
            fr = resize_frames(img[s:e].round().clip(0, 255).astype(np.uint8), args.image_size)
            human.append((off, fr)); off += len(fr)
            eps.append((_NoFrames(len(fr)), st[s:e, :2], act[s:e], "T-human", None, None) + ((st[s:e, 2:5],) if poses is not None else ()))
    rfiles = [f for d in args.records for f in sorted(Path(d).glob("*.npz"))]
    recs, rec_jobs = [], []
    for f, r in _ordered(_record_meta, rfiles, workers):
        if r is None:
            continue
        lab, apos, shape = r
        acts = unnormalize(lab)
        rec_jobs.append((f, off))
        for k in range(len(lab)):
            recs.append((_NoFrames(2), apos[k], acts[k], f"{shape}-dagger")); off += 2
    _, pos, acts_all, obs_idx, act_idx, group, near = build_index(eps, recs, args.near_coverage, out=_NoFrames(off), poses=poses)
    assert off <= len(out), f"{off} frames for an array of {len(out)}"
    path = out.filename
    for o, fr in human:
        out[o:o+len(fr)] = fr
    out.flush()
    for _ in _ordered(_demo_write, demo_jobs, workers, sub, path):
        pass
    for _ in _ordered(_record_write, rec_jobs, workers, args.image_size, path):
        pass
    return out, pos, acts_all, obs_idx, act_idx, group, near


def count_frames(args):
    """Frames the streaming build will write, without loading any frames (same choices as iter_episodes)."""
    sub = "frames" if args.image_size == 96 else f"frames{args.image_size}"
    rng = np.random.default_rng(1234); total = 0
    for f in sorted(f for d in args.demos for f in Path(d).glob("*.npz")):
        if not (f.parent/sub/f"{f.stem}_frames.npz").exists():
            continue
        with np.load(f) as z:
            mask = z["mask"]
        for i in range(len(mask)):
            m = int(mask[i].sum()) - 1
            if m < 2 or rng.random() >= args.data_fraction:
                continue
            total += m
    if args.human_t:
        import zarr
        total += int(zarr.open(str(args.human_t), "r")["meta"]["episode_ends"][-1])
    for d in args.records:
        for f in sorted(Path(d).glob("*.npz")):
            with np.load(f) as z:
                if "candidate_x" in z.files and len(z["choice"]) > 0:
                    total += 2*len(z["choice"])
    return total


_NO_POSE = np.full((1, 3), np.nan, np.float32)                            # a record sample: no block pose


def build_index(eps, recs=(), near_cov=0.8, out=None, poses=None):
    """poses: optional list; receives per-sample block poses (x, y, angle) of each sample's latest frame, from episode
    tuples that carry them (iter_episodes(..., poses=True)), NaN otherwise; np.concatenate(poses) aligns with the
    samples. Without it nothing changes."""
    pos, acts, obs_idx, act_idx, group, near = [], [], [], [], [], []
    if out is None:                                                       # in-memory lists: one host copy of every frame
        total = sum(len(e[0]) for e in eps) + 2*len(recs)
        shape = eps[0][0].shape[1:] if eps else recs[0][0].shape[1:]
        out = np.empty((total, *shape), np.uint8)
    frames = out
    off = aoff = 0
    released = [0]

    def release():                                                        # file-backed frames: write out and drop the
        if isinstance(frames, np.memmap) and (off - released[0]) * frames[0].nbytes >= 4 * 2**30:   # pages every 4 GB, so
            import mmap                                                    # host memory stays small (250 GB jobs)
            frames.flush()
            frames._mmap.madvise(mmap.MADV_DONTNEED)
            fd = os.open(frames.filename, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            released[0] = off
    for n_ep, item in enumerate(eps):
        fr, ap, ac, g, cov, plan = item[:6]
        if isinstance(eps, list):
            eps[n_ep] = None                                                # release the source as we go
        m = len(fr)
        frames[off:off+m] = fr; del fr; pos.append(ap)
        if plan is None:
            acts.append(ac)
        for t in range(0, max(1, m-7)):                                     # official padding: 1 before, 7 after
            obs_idx.append([off + max(t-1, 0), off + t])
            if plan is None:                                                # executed pushes a_(t-1) .. a_(t+14)
                act_idx.append([aoff + min(max(k, 0), m-1) for k in range(t-1, t+15)])
            else:                                                           # the teacher's intended plan at t
                acts.append(np.concatenate([ac[max(t-1, 0)][None], plan[t][:15]]))
                act_idx.append(list(range(aoff, aoff+16))); aoff += 16
            group.append(g); near.append(bool(cov is not None and cov[t] >= near_cov))
        if poses is not None:                                               # sample t's latest frame is frame t
            k = max(1, m-7)
            poses.append(np.array(item[6][:k], np.float32) if len(item) > 6 else np.full((k, 3), np.nan, np.float32))
        off += m
        if plan is None:
            aoff += m
        release()
    for n_rec, item in enumerate(recs):                                     # records: obs = their own 2 frames
        fr2, ap2, ac16, g = item
        if isinstance(recs, list):
            recs[n_rec] = None
        frames[off:off+2] = fr2; pos.append(ap2); acts.append(ac16)
        obs_idx.append([off, off+1]); act_idx.append(list(range(aoff, aoff+16))); group.append(g); near.append(False)
        if poses is not None:
            poses.append(_NO_POSE)
        off += 2; aoff += 16
        release()
    return (frames, np.concatenate(pos).astype(np.float32), np.concatenate(acts).astype(np.float32),
            np.array(obs_idx), np.array(act_idx), np.array(group), np.array(near))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", type=Path, nargs="+", required=True, help="One or more demonstration folders")
    p.add_argument("--human-t", type=Path, default=None)
    p.add_argument("--records", type=Path, nargs="*", default=[], help="DAgger record folders (mpc_dagger.py .../data)")
    p.add_argument("--init", type=Path, default=None, help="Start from a bc_train/distill checkpoint instead of the original")
    p.add_argument("--near-goal-weight", type=float, default=1.0, help="Oversample teacher steps near the goal by this factor")
    p.add_argument("--near-coverage", type=float, default=0.8, help="Coverage at which a teacher step counts as near the goal")
    p.add_argument("--label", choices=("executed", "plan"), default="executed",
                   help="Teacher episodes: copy the pushes actually executed, or the teacher's intended 16-step plan")
    p.add_argument("--image-size", type=int, default=96, help="Camera resolution (frames from frames<size>/ when not 96)")
    p.add_argument("--down-dims", type=int, nargs=3, default=None, help="Action U-Net widths (official 512 1024 2048)")
    p.add_argument("--scratch-model", action="store_true", help="Start the action U-Net from scratch")
    p.add_argument("--data-fraction", type=float, default=1.0, help="Use a fixed random fraction of teacher episodes")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); dev = "cuda"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import policy_builder
    arch = policy_builder.arch_of(args.image_size, args.down_dims, args.scratch_model)
    policy = policy_builder.build(args.image_size, args.down_dims, args.scratch_model)
    net = policy.network
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=True)
        net.model.load_state_dict(ck["actor"])
        enc = net.obs_encoder.state_dict()                                  # a 96 px student into a 192 px network: only the
        keep = {k: v for k, v in ck["encoder"].items() if enc[k].shape == v.shape}   # spatial-softmax grids differ (rebuilt)
        assert all(k.endswith(("pos_x", "pos_y")) for k in set(enc) - set(keep)), "unexpected encoder mismatch"
        net.obs_encoder.load_state_dict(keep, strict=False)
    unnorm = lambda x: net.normalizer["action"].unnormalize(torch.tensor(x, device=dev)).cpu().numpy()
    mm = os.environ.get("TDYSSEY_FRAMES_MMAP")                           # a local-disk directory: stream every source
    if mm:                                                                # into one file-backed array (small host memory)
        path = Path(mm)/f"frames-{os.getpid()}.npy"; path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(count_frames(args), args.image_size, args.image_size, 3))
        import atexit; atexit.register(lambda: path.unlink(missing_ok=True))
        print(f"frames streamed to {path} ({out.nbytes/2**30:.0f} GB)", flush=True)
        eps = iter_episodes(args); recs = iter_records(args.records, unnorm, args.image_size)
        frames, pos, acts, obs_idx, act_idx, group, near = build_index(eps, recs, args.near_coverage, out=out)
        assert len(pos) == len(frames), f"streamed {len(pos)} frames into an array of {len(frames)}"
    else:
        eps = load_episodes(args)
        recs = load_records(args.records, unnorm, args.image_size)
        frames, pos, acts, obs_idx, act_idx, group, near = build_index(eps, recs, args.near_coverage)
    groups = sorted(set(group))
    letters = sorted({g.rsplit("-", 1)[0] for g in groups})                 # each letter an equal share,
    share = {g: (1/len(letters)) / sum(1 for h in groups if h.rsplit("-", 1)[0] == g.rsplit("-", 1)[0]) for g in groups}
    # within each group, steps whose coverage is already >= --near-coverage get --near-goal-weight x the weight
    raw = np.where(near, args.near_goal_weight, 1.0)
    gsum = {g: raw[group == g].sum() for g in groups}
    w = np.array([share[g]/gsum[g] for g in group]) * raw
    info = {"samples": int(len(group)), "frames": int(len(frames)), "per_group": {g: int(np.sum(group == g)) for g in groups},
            "near_goal_samples": int(near.sum()),
            "share": share, "args": {k: str(v) for k, v in vars(args).items()}}
    print(json.dumps(info), flush=True)
    (args.out/"info.json").write_text(json.dumps(info, indent=1))
    on_cpu = True                    # frames in pinned host memory (96 px: ~31 GB; GPU memory is shared with other jobs)
    del eps, recs
    frames_mm = None
    if isinstance(frames, np.memmap):
        frames.flush(); frames_mm = frames; frames = torch.from_numpy(frames)
    elif frames.nbytes > 40e9:                                                  # 192 px: no pinned copy (512 GB job limit)
        frames = torch.from_numpy(frames)
    else:
        frames = torch.from_numpy(frames).pin_memory()
    pos, acts = torch.tensor(pos, device=dev), torch.tensor(acts, device=dev)
    obs_idx, act_idx = torch.tensor(obs_idx, device=dev), torch.tensor(act_idx, device=dev)
    weights = torch.tensor(w, device=dev, dtype=torch.float64)

    net.train()
    net.model.requires_grad_(True); net.obs_encoder.requires_grad_(True)  # normalizer stays fixed, as in the official recipe
    trained = list(net.model.parameters()) + list(net.obs_encoder.parameters())
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(trained, lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_at = lambda s: args.lr * (s/500 if s < 500 else 0.5*(1 + math.cos(math.pi*(s-500)/max(1, args.steps-500))))
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    start = time.time(); running = 0.0; raw_losses = []
    for step in range(1, args.steps+1):
        b = torch.multinomial(weights, args.batch, replacement=True, generator=gen)
        oi = obs_idx[b]                                                       # (B, 2)
        img = (frames[oi.cpu()].to(dev, non_blocking=True) if on_cpu else frames[oi]).permute(0, 1, 4, 2, 3).float()/255
        batch = {"obs": {"image": img, "agent_pos": pos[oi]}, "action": acts[act_idx[b]]}
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        loss = net.compute_loss(batch)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():                                                 # official EMA schedule
            decay = min(0.9999, max(0.0, 1 - (1 + step)**-0.75))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(decay).add_(pn.detach(), alpha=1-decay)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        if frames_mm is not None and step % 500 == 0:                        # file-backed frames: drop pages read so far
            import mmap
            frames_mm._mmap.madvise(mmap.MADV_DONTNEED)
        raw_losses.append(loss.item())                                         # every step, saved with each checkpoint
        running = 0.98*running + 0.02*loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            print(f"[bc {time.time()-start:5.0f}s] step {step} loss {running:.4f} lr {lr_at(step):.2e}", flush=True)
            np.save(args.out/"loss_per_step.npy", np.array(raw_losses, dtype=np.float32))   # live view (W&B sync)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"trainable": "action_network", "actor": ema.model.state_dict(), "encoder": ema.obs_encoder.state_dict(),
                        "step": step, "arch": arch, "label": args.label,
                        "recipe": "official DP recipe on teacher demos (+ human T)"}, args.out/f"ema-step{step}.pt")
            np.save(args.out/"loss_per_step.npy", np.array(raw_losses, dtype=np.float32))
            print(f"saved {args.out}/ema-step{step}.pt", flush=True)


if __name__ == "__main__":
    main()
