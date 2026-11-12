"""Checks of the prefix (pi0-style block-causal) action head in flow_policy.py, and that the default head is unchanged.

  a  mask: the prefix head written as ONE masked attention over [condition | action] per block (the pi0 formulation,
     `joint_reference` below) gives bit-identical condition states at every layer when the action tokens or the flow
     time change, and different velocities when the condition changes; the same reference without the mask does
     leak (negative control); the fast two-call implementation equals the masked reference.
  b  cache: integrate() (condition stream once per decision, key/value buffers with action slots) equals uncached
     integration that reruns the masked joint sequence at every Euler step (fp32), and, under bf16 autocast, equals the
     fast path recomputed at every step.
  c  cross head unchanged: the default arch built by this file and by the pre-change flow_policy.py (read from git,
     --orig-rev) has the same parameter names and bit-identical initial weights, loss and gradients for the same seed;
     with --checkpoint (a flow_train.py checkpoint of the cross head, on CUDA) both files' agents return bit-identical
     plans for the same observations and noise.
Runs on the CPU (fp32) unless --device cuda; prints one line per check and exits non-zero if any fails.

  REPO=$PWD .venv/bin/python -B scripts/learning/flow_head_check.py --device cuda \\
      --checkpoint runs/flow26b16/v1/ema-step160000.pt --out runs/joint-smoke/check.json
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flow_policy as fp  # noqa: E402


def randomize_zero_init(net, seed):
    """Fill every all-zero head parameter (adaLN-Zero gates, zero-initialised outputs, offsets) with small noise, so
    that every path of the head is active (at initialisation the action blocks are the identity)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if not name.startswith(("vit.", "normalizer.")) and not p.any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def joint_reference(net, cond, x, t, mask=True):
    """The prefix head as one attention over the sequence [condition | action] per block, with the block-causal mask
    (condition rows may not attend to action columns). Returns velocity and the condition stream's per-layer outputs
    (keys and values, normalised, and the refined tokens)."""
    a = net.arch
    B, N, H, w = cond.shape[0], cond.shape[1], x.shape[1], a["width"]
    h = net.act_in(x) + net.act_pos
    te = net.t_mlp(fp.timestep_embedding(t * a["time_scale"], a["time_freq_dim"]))
    shared = net.t_mod(te).view(B, 6, w) if a["adaln"] == "single" else None
    allowed = torch.ones(N + H, N + H, dtype=torch.bool, device=cond.device)
    if mask:
        allowed[:N, N:] = False
    c, states = cond, []
    for cb, ab in zip(net.cond_blocks, net.blocks):
        mod = shared + ab.mod_table if shared is not None else ab.ada(te).view(B, 6, w)
        s1, c1, g1, s2, c2, g2 = mod.unbind(1)
        hc = cb.norm1(c)
        if cb.last:                                      # no condition queries in the last block (outputs unused)
            kc, vc = cb.kv(hc).chunk(2, -1)
            qc = torch.zeros_like(kc)
        else:
            qc, kc, vc = cb.qkv(hc).chunk(3, -1)
            qc = cb.q_norm(cb.heads_view(qc))
        kc, vc = cb.k_norm(cb.heads_view(kc)), cb.heads_view(vc)
        if cb.last:
            qc = cb.heads_view(qc)
        qa, ka, va = ab.qkv(fp._modulate(ab.norm1(h), s1, c1)).chunk(3, -1)
        q = torch.cat([qc, ab.q_norm(ab.heads_view(qa))], 2)
        k = torch.cat([kc, ab.k_norm(ab.heads_view(ka))], 2)
        v = torch.cat([vc, ab.heads_view(va)], 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed).transpose(1, 2)    # (B, N + H, heads, dh)
        oc, oa = o[:, :N].reshape(B, N, -1), o[:, N:].reshape(B, H, -1)
        states.append((kc, vc, c))
        if not cb.last:
            c = c + cb.proj(oc)
            c = c + cb.mlp(cb.norm2(c))
        h = h + g1[:, None] * ab.proj(oa)
        h = h + g2[:, None] * ab.mlp(fp._modulate(ab.norm2(h), s2, c2))
    shift, scale = net.final_mod(te).chunk(2, -1)
    return net.out(fp._modulate(net.final_norm(h), shift, scale)), states


def max_diff(a, b):
    return float((a.float() - b.float()).abs().max())


def states_diff(s1, s2):
    return max(max_diff(x, y) for p, q in zip(s1, s2) for x, y in zip(p, q))


def load_orig_module(rev):
    """The flow_policy.py of git revision rev as a module (compiled from git show; nothing written)."""
    src = subprocess.run(["git", "show", f"{rev}:scripts/learning/flow_policy.py"], cwd=HERE, check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType("flow_policy_orig")
    mod.__file__ = str(HERE / "flow_policy.py")
    exec(compile(src, f"{rev}:flow_policy.py", "exec"), mod.__dict__)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--small", action="store_true", help="Small head (width 256, depth 4) for a quick run")
    p.add_argument("--orig-rev", default="0fc202c", help="Git revision of flow_policy.py before the prefix head")
    p.add_argument("--checkpoint", type=Path, default=None, help="Cross-head flow_train.py checkpoint for check c2 (CUDA)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    size = dict(width=256, depth=4, heads=4) if args.small else {}
    norm = fp.official_normalizer()
    results, ok = {}, True

    def report(name, passed, **vals):
        nonlocal ok
        ok &= bool(passed)
        results[name] = {"pass": bool(passed), **vals}
        print(f"{'PASS' if passed else 'FAIL'} {name}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                                                for k, v in vals.items()), flush=True)

    # ------------------------------------------------------------------------------------------ a, b: prefix head
    for pool in (8, 16):
        torch.manual_seed(0)
        cw = 128 if args.small else 512
        net = fp.FlowPolicy(fp.default_arch(pool=pool, **size, **fp.prefix_arch(cw)), norm, pretrained=False)
        randomize_zero_init(net, 1)
        net = net.to(dev).eval()
        g = torch.Generator(device=dev).manual_seed(2)
        B, H = 3, 16
        img = torch.rand(B, 2, 3, 96, 96, device=dev, generator=g)
        pos = torch.rand(B, 2, 2, device=dev, generator=g) * 512
        with torch.no_grad():
            cond = net.encode(img, pos)
            N = cond.shape[1]
            x = torch.randn(B, H, 2, device=dev, generator=g)
            t = torch.rand(B, device=dev, generator=g)
            x2, t2 = torch.randn(B, H, 2, device=dev, generator=g), torch.rand(B, device=dev, generator=g)
            cond2 = cond + 0.1 * torch.randn(cond.shape, device=dev, generator=g)
            v_ref, st = joint_reference(net, cond, x, t)
            _, st_x = joint_reference(net, cond, x2, t)
            _, st_t = joint_reference(net, cond, x, t2)
            v_c, _ = joint_reference(net, cond2, x, t)
            _, st_open = joint_reference(net, cond, x2, t, mask=False)
            _, st_open0 = joint_reference(net, cond, x, t, mask=False)
            tag = f"pool{pool} (N={N})"
            report(f"a1 {tag}: masked joint attention, condition states vs action tokens / time changed",
                   states_diff(st, st_x) == 0 and states_diff(st, st_t) == 0,
                   max_diff_actions=states_diff(st, st_x), max_diff_time=states_diff(st, st_t))
            report(f"a2 {tag}: condition changed -> velocity changes", max_diff(v_ref, v_c) > 1e-3,
                   max_diff=max_diff(v_ref, v_c))
            report(f"a3 {tag}: negative control, no mask -> condition states change with the action tokens",
                   states_diff(st_open0, st_open) > 1e-4, max_diff=states_diff(st_open0, st_open))
            kv = net.kv_cache(cond)
            v_fast = net.velocity(x, t, kv)
            kv_diff = max(max(max_diff(a_, b_[0]), max_diff(c_, b_[1])) for (a_, c_), b_ in zip(kv, st))
            report(f"a4 {tag}: fast path (two attention calls) = masked joint reference (fp32)",
                   max_diff(v_fast, v_ref) < 1e-4 and kv_diff < 1e-4,
                   velocity_max_diff=max_diff(v_fast, v_ref), condition_kv_max_diff=kv_diff,
                   velocity_scale=float(v_ref.abs().max()))
            x0 = torch.randn(B, H, 2, device=dev, generator=g)
            cached = net.integrate(cond, x0.clone())
            xr = x0.clone()
            steps = net.arch["inference_steps"]
            for k in range(steps):
                xr = xr + joint_reference(net, cond, xr, torch.full((B,), k / steps, device=dev))[0] / steps
            report(f"b1 {tag}: cached integrate = uncached masked joint sequence every step (fp32)",
                   max_diff(cached, xr) < 1e-4, max_diff=max_diff(cached, xr), plan_scale=float(xr.abs().max()))
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cached_bf = net.integrate(cond, x0.clone())
                    xf = x0.clone().float()
                    for k in range(steps):
                        tk = torch.full((B,), k / steps, device=dev)
                        xf = xf + net.velocity(xf, tk, net.kv_cache(cond)).float() / steps
                    xr_bf = x0.clone().float()
                    for k in range(steps):
                        tk = torch.full((B,), k / steps, device=dev)
                        xr_bf = xr_bf + joint_reference(net, cond, xr_bf, tk)[0].float() / steps
                report(f"b2 {tag}: bf16 cached integrate = fast path recomputed every step", max_diff(cached_bf, xf) == 0,
                       max_diff=max_diff(cached_bf, xf))
                report(f"b3 {tag}: bf16 cached integrate vs masked joint reference every step (bf16 noise)",
                       max_diff(cached_bf, xr_bf) < 0.1, max_diff=max_diff(cached_bf, xr_bf),
                       fp32_vs_bf16=max_diff(cached_bf, cached))
        del net

    # ------------------------------------------------------------------------------------------ c: cross unchanged
    orig = load_orig_module(args.orig_rev)
    arch = fp.default_arch(**size)
    torch.manual_seed(0); new_net = fp.FlowPolicy(arch, norm, pretrained=False)
    torch.manual_seed(0); old_net = orig.FlowPolicy(arch, norm, pretrained=False)
    sn, so = new_net.state_dict(), old_net.state_dict()
    same = list(sn) == list(so) and all(torch.equal(sn[k], so[k]) for k in sn)
    report("c1 cross head: parameter names and initial weights = pre-change code", same, parameters=len(sn))
    randomize_zero_init(new_net, 3); randomize_zero_init(old_net, 3)
    losses, grads = [], []
    # CUDA backward kernels are not deterministic: the new code runs twice, and its gradients may differ from the old
    # code's by no more than they differ between its own two runs (CPU: everything bit-identical)
    for net in (new_net, old_net) + ((new_net,) if dev.type == "cuda" else ()):
        net.to(dev).train()
        net.zero_grad(set_to_none=True)
        g = torch.Generator(device=dev).manual_seed(4)
        img = torch.rand(2, 2, 3, 96, 96, device=dev, generator=g)
        pos = torch.rand(2, 2, 2, device=dev, generator=g) * 512
        act = torch.rand(2, 16, 2, device=dev, generator=g) * 512
        torch.manual_seed(5)                                       # random crops use the global generator
        loss = net.compute_loss(img, pos, act, generator=torch.Generator(device=dev).manual_seed(6))
        loss.backward()
        losses.append(loss.detach())
        grads.append({n: q.grad.clone() for n, q in net.named_parameters() if q.grad is not None})
        net.cpu()
    gdiff = lambda a_, b_: max(max_diff(a_[k], b_[k]) for k in a_) if list(a_) == list(b_) else float("inf")
    old_diff = gdiff(grads[0], grads[1])
    rerun = gdiff(grads[0], grads[2]) if dev.type == "cuda" else 0.0
    report("c2 cross head: training loss bit-identical to the pre-change code, gradients within run-to-run noise",
           torch.equal(losses[0], losses[1]) and old_diff <= 2 * rerun, loss_new=float(losses[0]),
           loss_old=float(losses[1]), grad_max_diff_vs_old=old_diff, grad_max_diff_same_code_rerun=rerun,
           grads_compared=len(grads[0]))
    del new_net, old_net
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        rng = np.random.default_rng(7)
        hist = [[{"image": rng.random((3, 96, 96), dtype=np.float32), "agent_pos": rng.uniform(50, 460, 2).astype(np.float32)}
                 for _ in range(2)] for _ in range(4)]
        outs = []
        for mod in (fp, orig):
            agent = mod.FlowAgent(mod.build_network(payload), dev)
            gens = [torch.Generator(device=dev).manual_seed(100 + i) for i in range(len(hist))]
            actions, plan = agent.sample_batch(agent.encode_batch(hist), gens)
            outs.append((actions, plan.cpu()))
            del agent
            torch.cuda.empty_cache()
        report(f"c3 cross checkpoint {args.checkpoint.name}: plans from this code = pre-change code (bf16, bit-identical)",
               np.array_equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]),
               max_diff_px=float(np.abs(outs[0][0] - outs[1][0]).max()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
