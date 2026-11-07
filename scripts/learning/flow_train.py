"""Train the D059 flow-matching DiT camera policy (flow_policy.py: DINOv2 ViT-B/14 + DiT head, ~505M parameters).

Data: exactly bc_train.py's pipeline (its functions are imported, the sample weighting is copied from its main()):
teacher episodes (--demos, intended-plan labels with --label plan, the default here), the human T demonstrations
(--human-t), correction records (--records); every letter an equal share (T split between its sources); frames streamed
to one disk-backed array when TDYSSEY_FRAMES_MMAP is set (never all in memory: the 26-letter set is ~5.8M frames).
Each sample: 2 camera frames + 2 pusher positions -> the 16-step label chunk (pixels; normalized inside the model with the
official checkpoint's normalizer). Augmentation: the official random 84 px crop of each frame (inside the model).

Optimisation: AdamW (betas 0.9/0.95), head lr --lr (1e-4), encoder lr --lr x --encoder-lr-scale (1e-5), weight decay
--weight-decay on head matrices and --encoder-weight-decay on encoder matrices (none on biases, norms, embeddings),
--warmup linear warm-up then cosine decay to 0 at --steps, bf16 autocast, global gradient clipping (--clip), EMA weights
with the official schedule (power 0.75, max 0.9999), saved for evaluation. The loss is compiled (torch.compile) unless
--no-compile; --grad-checkpoint recomputes encoder activations (batch 128: ~18 GB GPU instead of ~40 GB, ~25% slower).
Batches are assembled on a background thread.
Outputs in --out: info.json, train.log (one line per --log-every steps), loss_per_step.npy, ema-step<N>.pt
(base_rollouts.py eval --student reads them), resume.pt (latest full training state; --resume continues from it).

  REPO=/tmp/Tdyssey/lastmile /tmp/Tdyssey/repo/.venv/bin/python -B scripts/learning/flow_train.py \\
      --demos $R/teacher-plan-demos --human-t $HUMAN --records $R/mpc-data/dagger26m-r1/data --steps 200 --out $R/flow-smoke/train
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bc_train  # noqa: E402  (data pipeline)
import flow_policy as fp  # noqa: E402


def build_data(args, normalizer, dev):
    """bc_train.main()'s data assembly and equal-share weights (copied; the loaders are bc_train's own functions)."""
    unnorm = lambda x: normalizer["action"].unnormalize(torch.tensor(x, device=dev)).cpu().numpy()
    mm = os.environ.get("TDYSSEY_FRAMES_MMAP")
    if mm:
        path = Path(mm)/f"frames-{os.getpid()}.npy"; path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                        shape=(bc_train.count_frames(args), args.image_size, args.image_size, 3))
        import atexit; atexit.register(lambda: path.unlink(missing_ok=True))
        print(f"frames streamed to {path} ({out.nbytes/2**30:.0f} GB)", flush=True)
        eps = bc_train.iter_episodes(args); recs = bc_train.iter_records(args.records, unnorm, args.image_size)
        frames, pos, acts, obs_idx, act_idx, group, near = bc_train.build_index(eps, recs, args.near_coverage, out=out)
        assert len(pos) == len(frames), f"streamed {len(pos)} frames into an array of {len(frames)}"
        frames.flush()
    else:
        eps = bc_train.load_episodes(args)
        recs = bc_train.load_records(args.records, unnorm, args.image_size)
        frames, pos, acts, obs_idx, act_idx, group, near = bc_train.build_index(eps, recs, args.near_coverage)
    del eps, recs
    groups = sorted(set(group))
    letters = sorted({g.rsplit("-", 1)[0] for g in groups})
    share = {g: (1/len(letters)) / sum(1 for h in groups if h.rsplit("-", 1)[0] == g.rsplit("-", 1)[0]) for g in groups}
    raw = np.where(near, args.near_goal_weight, 1.0)
    gsum = {g: raw[group == g].sum() for g in groups}
    w = np.array([share[g]/gsum[g] for g in group]) * raw
    info = {"samples": int(len(group)), "frames": int(len(frames)), "per_group": {g: int(np.sum(group == g)) for g in groups},
            "near_goal_samples": int(near.sum()), "share": share}
    return frames, pos, acts, obs_idx, act_idx, w, info


class Batches:
    """Weighted sampling with replacement + frame gathering + pinning on a background thread (keeps the GPU fed)."""

    def __init__(self, frames, pos, acts, obs_idx, act_idx, weights, batch, seed, depth=6):
        self.frames = torch.from_numpy(frames) if isinstance(frames, np.ndarray) else frames
        self.pos, self.acts = torch.from_numpy(pos), torch.from_numpy(acts)
        self.obs_idx, self.act_idx = torch.from_numpy(obs_idx), torch.from_numpy(act_idx)
        self.cdf = np.cumsum(weights / weights.sum())
        self.batch, self.rng = batch, np.random.default_rng(seed)
        self.q, self.stop, self.error = queue.Queue(depth), False, None
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def _run(self):
        try:
            while not self.stop:
                b = torch.from_numpy(np.minimum(np.searchsorted(self.cdf, self.rng.random(self.batch), side="right"),
                                                len(self.cdf) - 1))
                oi = self.obs_idx[b]                                                          # (B, 2)
                img = self.frames[oi.reshape(-1)].reshape(self.batch, oi.shape[1], *self.frames.shape[1:]).pin_memory()
                self.q.put((img, self.pos[oi].pin_memory(), self.acts[self.act_idx[b]].pin_memory()))
        except BaseException as exc:                                                         # surfaced by get()
            self.error = exc; self.q.put(None)

    def get(self):
        item = self.q.get()
        if item is None:
            raise RuntimeError("batch thread failed") from self.error
        return item

    def close(self):
        self.stop = True
        while self.thread.is_alive():
            try:
                self.q.get_nowait()
            except queue.Empty:
                time.sleep(0.01)


def param_groups(net, args):
    groups = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        enc = name.startswith("vit.")
        decay = p.ndim >= 2 and not any(s in name for s in ("token", "position_embeddings", "_pos", "_emb", "mod_table"))
        key = ("encoder" if enc else "head") + ("" if decay else "-no-decay")
        g = groups.setdefault(key, {"params": [], "name": key, "base_lr": args.lr * (args.encoder_lr_scale if enc else 1.0),
                                    "weight_decay": (args.encoder_weight_decay if enc else args.weight_decay) if decay else 0.0})
        g["params"].append(p)
    return list(groups.values())


def host_memory_gb():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM", "RssAnon", "RssFile")):
                k, v = line.split(":"); out[k] = round(int(v.split()[0]) / 2**20, 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", type=Path, nargs="+", required=True, help="One or more demonstration folders")
    p.add_argument("--human-t", type=Path, default=None)
    p.add_argument("--records", type=Path, nargs="*", default=[], help="Correction record folders (mpc_dagger.py .../data)")
    p.add_argument("--label", choices=("executed", "plan"), default="plan", help="Teacher episodes: executed pushes or intended plan")
    p.add_argument("--near-goal-weight", type=float, default=1.0)
    p.add_argument("--near-coverage", type=float, default=0.8)
    p.add_argument("--data-fraction", type=float, default=1.0, help="Use a fixed random fraction of teacher episodes")
    p.add_argument("--init", type=Path, default=None, help="Start from a flow_train checkpoint (its EMA weights and arch)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--adaln", choices=("single", "full"), default="single", help="Shared (PixArt adaLN-single) or per-block time modulation")
    p.add_argument("--pool", type=int, default=8, help="Patch tokens per frame are pooled to pool x pool")
    p.add_argument("--no-cls", action="store_true", help="Drop the CLS token of each frame")
    p.add_argument("--no-qk-norm", action="store_true")
    p.add_argument("--inference-steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4, help="Head learning rate")
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--encoder-weight-decay", type=float, default=1e-6)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true", help="Recompute encoder activations in backward (less GPU memory)")
    p.add_argument("--no-compile", action="store_true", help="Skip torch.compile of the loss (compiled: ~2x faster, ~2 min start-up)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--resume", action="store_true", help="Continue from --out/resume.pt if it exists")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.save_every % args.log_every:
        p.error("--save-every must be a multiple of --log-every (the saved loss history is complete at every checkpoint)")
    args.image_size = 96                                           # bc_train loaders: 96 px frames
    resume = args.out/"resume.pt" if args.resume and (args.out/"resume.pt").exists() else None
    args.out.mkdir(parents=True, exist_ok=resume is not None)
    torch.manual_seed(args.seed); dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    t_start = time.time()

    normalizer = fp.official_normalizer()
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=True)
        net = fp.build_network(init); arch = net.arch
    else:
        arch = fp.default_arch(width=args.width, depth=args.depth, heads=args.heads, mlp_ratio=args.mlp_ratio, adaln=args.adaln,
                               pool=args.pool, use_cls=not args.no_cls, qk_norm=not args.no_qk_norm,
                               inference_steps=args.inference_steps)
        net = fp.FlowPolicy(arch, normalizer, pretrained=True)
    net.vit.grad_checkpoint = args.grad_checkpoint
    counts = net.parameter_counts()
    print(f"flow policy parameters: encoder {counts['encoder']:,}  head {counts['head']:,}  total {counts['total']:,}", flush=True)
    net = net.to(dev).train()

    frames, pos, acts, obs_idx, act_idx, w, info = build_data(args, net.normalizer, dev)
    info.update({"parameters": counts, "arch": arch, "args": {k: str(v) for k, v in vars(args).items()},
                 "data_seconds": round(time.time() - t_start, 1)})
    print(json.dumps({k: info[k] for k in ("samples", "frames", "per_group", "parameters", "data_seconds")}), flush=True)
    if not resume:
        (args.out/"info.json").write_text(json.dumps(info, indent=1))

    groups = param_groups(net, args)
    trainable = [q for g in groups for q in g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=tuple(args.betas), fused=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    named_ema = dict(ema.named_parameters())
    net_params = [q for n, q in net.named_parameters() if q.requires_grad]                 # EMA pairs, same order
    ema_params = [named_ema[n] for n, q in net.named_parameters() if q.requires_grad]
    lr_mult = lambda s: s/args.warmup if s < args.warmup else 0.5*(1 + math.cos(math.pi*(s-args.warmup)/max(1, args.steps-args.warmup)))
    step0, raw_losses = 0, []
    if resume:
        st = torch.load(resume, map_location="cpu", weights_only=True)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        net.to(dev); ema.to(dev)                       # the official normalizer's loader rebuilds its tables on the CPU
        step0, raw_losses = int(st["step"]), st["losses"].tolist()
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
        del st
        print(f"resumed from {resume} at step {step0}", flush=True)
    batches = Batches(frames, pos, acts, obs_idx, act_idx, w, args.batch, seed=args.seed*1000003 + step0)
    frames_mm = frames if isinstance(frames, np.memmap) else None
    log = open(args.out/"train.log", "a")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    def save(step):
        payload = {"arch": arch, "ema": fp.head_state(ema), "normalizer": net.normalizer.state_dict(), "trainable": "flow_dit",
                   "step": step, "label": args.label, "parameters": counts,
                   "recipe": "flow-matching DiT on DINOv2 (flow_train.py, D059); bc_train data pipeline"}
        torch.save(payload, args.out/f"ema-step{step}.pt")
        tmp = args.out/"resume.pt.tmp"
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "losses": torch.tensor(raw_losses, dtype=torch.float32), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state()}, tmp)
        tmp.replace(args.out/"resume.pt")
        np.save(args.out/"loss_per_step.npy", np.array(raw_losses, dtype=np.float32))
        say(f"saved {args.out}/ema-step{step}.pt")

    loss_fn = net.compute_loss if args.no_compile else torch.compile(net.compute_loss)
    torch.cuda.reset_peak_memory_stats()
    loss_buf = torch.zeros(args.log_every, device=dev); gn_buf = torch.zeros(args.log_every, device=dev)
    t_log, n_buf = time.time(), 0
    try:
        for step in range(step0 + 1, args.steps + 1):
            img, apos, act = batches.get()
            img = img.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)   # (B, 2, 3, 96, 96) in [0, 1]
            apos, act = apos.to(dev, non_blocking=True), act.to(dev, non_blocking=True)
            m = lr_mult(step)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * m
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(img, apos, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
            with torch.no_grad():                                                  # official EMA schedule
                decay = min(0.9999, max(0.0, 1 - (1 + step)**-0.75))
                torch._foreach_lerp_(ema_params, net_params, 1 - decay)
            loss_buf[n_buf] = loss.detach(); gn_buf[n_buf] = gn.detach(); n_buf += 1
            if n_buf == args.log_every or step == args.steps:
                ls = loss_buf[:n_buf].cpu().numpy(); g = gn_buf[:n_buf].cpu().numpy(); raw_losses.extend(ls.tolist())
                if not np.isfinite(ls).all():
                    raise FloatingPointError(f"non-finite loss near step {step}")
                dt = (time.time() - t_log) / n_buf
                say(f"[flow {time.time()-t_start:6.0f}s] step {step} loss {ls.mean():.4f} grad-norm {g.mean():.3f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {dt:.3f}s/step {args.batch/dt:.0f} samples/s "
                    f"gpu-peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB host {host_memory_gb()}")
                np.save(args.out/"loss_per_step.npy", np.array(raw_losses, dtype=np.float32))
                t_log, n_buf = time.time(), 0
            if frames_mm is not None and step % 500 == 0:                            # drop file pages read so far
                import mmap
                frames_mm._mmap.madvise(mmap.MADV_DONTNEED)
            if step % args.save_every == 0 or step == args.steps:
                save(step); t_log = time.time()                                   # step times exclude saving
    finally:
        batches.close()
        log.close()


if __name__ == "__main__":
    main()
