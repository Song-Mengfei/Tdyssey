"""Flow-matching DiT camera policy (D059): DINOv2 ViT-B/14 image encoder + transformer action head, about 500M parameters.

Observation: the last 2 camera frames (96 px RGB in [0, 1]) and the 2 pusher positions; output: 16 future xy targets
(normalized with the official checkpoint's LinearNormalizer, so labels and outputs share the official scale), of which
rows 1-8 are executed, exactly as the official Diffusion Policy (n_obs_steps 2 -> plan row 1 is the current control).

Encoder: a minimal DINOv2 ViT-B/14 (patch 14, width 768, 12 layers, 12 heads, LayerScale, pre-norm, final layernorm;
facebook/dinov2-base weights in the transformers key layout, loaded strictly). Each frame is cropped to 84 px (random
crop in training as in the official recipe, centre crop at evaluation), resized to 224 px (bilinear), ImageNet-normalized
and encoded; the 16 x 16 patch tokens are average-pooled to 8 x 8 = 64 tokens per frame (+ the CLS token), projected to
the head width, and given learned frame and 2-D position embeddings. The pretrained position embeddings (37 x 37 grid)
are interpolated to the input grid as in the original DINOv2 code (bicubic, scale factor (n + 0.1) / 37, no antialias).
Proprioception: the 2 normalized pusher positions -> MLP -> 1 token. Condition = [frame tokens, pusher token].

Action head: DiT-style, width 1024, depth 24, 16 heads, MLP ratio 4. Each block: self-attention over the 16 action
tokens, cross-attention to the condition tokens, MLP; all three with adaptive layer norm driven by the flow-time
embedding and zero-initialised gates (adaLN-Zero: every block starts as the identity). Modulation "single" (default)
shares one time -> 9 x width projection across blocks plus a learned per-block offset table (PixArt-alpha adaLN-single);
"full" gives every block its own projection (original DiT; +9.4M parameters per block, ~630M head at depth 24).
Queries and keys are RMS-normalized per head (QK-norm) for stable bf16 training.

Flow matching (convention used everywhere here): t = 0 is noise, t = 1 is data,
    x_t = t * x1 + (1 - t) * eps,  target velocity v = x1 - eps,
with x1 the normalized 16 x 2 plan label. Training time: t = 0.999 * (1 - u), u ~ Beta(1.5, 1) (pi0, arXiv 2410.24164:
p(tau) = Beta((s - tau) / s; 1.5, 1), s = 0.999, which puts more weight on the noisy end; openpi writes the same
distribution with time reversed, t_openpi = 1 - t). Sampled by inverse CDF, u = U ** (2/3), U ~ Uniform(0, 1).
Inference: x_0 ~ N(0, I) drawn row by row from the caller's torch.Generator (same generator semantics as the official
agent), then K = 10 Euler steps x <- x + v(x, k/K) / K, k = 0..K-1. The sampler is deterministic given the starting
noise (useful for noise steering).

Optional block-pose head (arch "pose_head": True; flow_train.py --pose-weight > 0): a small head reads the latest frame's
full 16 x 16 patch-token grid (after the encoder's final layer norm, before pooling, whatever "pool" is) and predicts the
block's pose, (x - 256) / 256, (y - 256) / 256, cos(angle), sin(angle). Its loss (smooth-L1 on the positions, squared
error on cos/sin, the better of angle and angle + pi for shapes with 2-fold symmetry; only samples with a label) is
added to the flow loss with weight "pose_weight" during training only. The labels are privileged simulator state; the
head is never run at inference (encode / sample are unchanged), so the policy stays camera-only.

FlowAgent provides the interface the evaluation and collection tools use (base_rollouts.build returns it for checkpoints
whose arch type is "flow_dit"): device, render_size, generator, network.normalizer, encode / encode_batch (histories of
2 observation dicts -> condition tensor (B, N, width)), sample / sample_batch (condition, generators) -> (actions
(B, 8, 2) pixels as numpy, samples (B, 16, 2) normalized).
"""
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DINO_REPO = "facebook/dinov2-base"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_arch(**overrides):
    arch = {"type": "flow_dit", "version": 1, "image_size": 96, "crop": 84, "vit_input": 224, "obs_steps": 2,
            "horizon": 16, "exec_start": 1, "n_exec": 8, "encoder": DINO_REPO, "vit_dim": 768, "vit_depth": 12,
            "vit_heads": 12, "patch": 14, "pool": 8, "use_cls": True, "width": 1024, "depth": 24, "heads": 16,
            "mlp_ratio": 4.0, "adaln": "single", "qk_norm": True, "time_freq_dim": 256, "time_scale": 1000.0,
            "flow_time": "pi0 beta(1.5,1), s=0.999; t=0 noise, t=1 data", "flow_s": 0.999, "beta_a": 1.5,
            "inference_steps": 10}
    arch.update(overrides)
    return arch


def pose_arch(weight, beta=0.01, proj=128, hidden=512):
    """Arch keys of the training-only block-pose head (flow_train.py --pose-weight); absent = no head."""
    return {"pose_head": True, "pose_weight": float(weight), "pose_beta": float(beta), "pose_proj": proj, "pose_hidden": hidden,
            "pose_input": "latest frame, all patch tokens after the encoder's final layer norm, before pooling",
            "pose_label": "block x, y, angle of the latest frame (privileged simulator state; training only, unused at inference)"}


# ----------------------------------------------------------------------------------------------- DINOv2 ViT-B/14
class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.lambda1


class _DinoSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)


class _DinoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.attention = _DinoSelfAttention(dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = (lin(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)
                   for lin in (self.attention.query, self.attention.key, self.attention.value))
        o = F.scaled_dot_product_attention(q, k, v)
        return self.output.dense(o.transpose(1, 2).reshape(B, N, C))


class _DinoMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))                                   # exact (erf) GELU, as "gelu" in HF


class _DinoLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attention, self.mlp = _DinoAttention(dim, heads), _DinoMLP(dim, dim * mlp_ratio)
        self.layer_scale1, self.layer_scale2 = _LayerScale(dim), _LayerScale(dim)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DinoV2(nn.Module):
    """DINOv2 backbone with the transformers Dinov2Model parameter names (no pooler); returns normalized tokens."""

    def __init__(self, dim=768, depth=12, heads=12, patch=14, grid=37):
        super().__init__()
        self.patch, self.grad_checkpoint = patch, False
        self.embeddings = nn.Module()
        self.embeddings.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.embeddings.mask_token = nn.Parameter(torch.zeros(1, dim))        # unused (masked pretraining); kept for keys
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = nn.Conv2d(3, dim, patch, patch)
        self.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, 1 + grid * grid, dim))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(_DinoLayer(dim, heads) for _ in range(depth))
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def position_embedding(self, h, w):
        pe = self.embeddings.position_embeddings
        g = int(math.sqrt(pe.shape[1] - 1))
        if (h, w) == (g, g):
            return pe
        grid = pe[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        grid = F.interpolate(grid, scale_factor=((h + 0.1) / g, (w + 0.1) / g), mode="bicubic", align_corners=False)
        assert grid.shape[-2:] == (h, w), grid.shape
        return torch.cat([pe[:, :1], grid.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(pe.dtype)], 1)

    def forward(self, x):
        p = self.embeddings.patch_embeddings.projection(x)                   # (N, dim, h, w)
        h, w = p.shape[-2:]
        tok = torch.cat([self.embeddings.cls_token.expand(len(x), -1, -1).to(p.dtype), p.flatten(2).transpose(1, 2)], 1)
        tok = tok + self.position_embedding(h, w)
        for layer in self.encoder.layer:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                tok = torch.utils.checkpoint.checkpoint(layer, tok, use_reentrant=False)
            else:
                tok = layer(tok)
        return self.layernorm(tok), (h, w)


def load_dinov2(model, repo=DINO_REPO):
    """Strictly load public DINOv2 weights (facebook/dinov2-base or -large; HF_HOME cache; weights only, no transformers)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    state = load_file(hf_hub_download(repo, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"DINOv2 key mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model


# ----------------------------------------------------------------------------------------------- block-pose head
class PoseHead(nn.Module):
    """Training-only auxiliary head: patch tokens (B, N, dim) -> (B, 4) = normalized x, y, cos, sin of the block pose.
    Per-token projection, then the flattened grid (keeps where each token is) -> MLP."""

    def __init__(self, dim, tokens, proj=128, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, proj)
        self.fc = nn.Linear(tokens * proj, hidden)
        self.out = nn.Linear(hidden, 4)
        for lin in (self.proj, self.fc):
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, tokens):
        h = F.gelu(self.proj(self.norm(tokens))).flatten(1)
        return self.out(F.gelu(self.fc(h)))


def pose_targets(pose):
    """Pose labels (B, 3) simulator px / radians -> (B, 4) head targets."""
    return torch.cat([(pose[:, :2] - 256.0) / 256.0, torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])], 1)


def pose_errors(pred, pose, sym2):
    """Per-sample position error (simulator px, Euclidean), |dx|, |dy| and angle error (degrees, modulo the shape's
    symmetry) of head outputs pred (B, 4) against labels pose (B, 3); sym2 (B,) bool: 2-fold symmetric shape."""
    d = pred[:, :2].float() * 256.0 + 256.0 - pose[:, :2]
    order = sym2.float() + 1
    delta = (torch.atan2(pred[:, 3].float(), pred[:, 2].float()) - pose[:, 2]) * order
    return d.norm(dim=-1), d.abs(), torch.rad2deg(torch.atan2(torch.sin(delta), torch.cos(delta)).abs() / order)


# ----------------------------------------------------------------------------------------------- DiT action head
class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight.to(x.dtype)


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, adaln="single", qk_norm=True):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.cq, self.ckv, self.cproj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        norm = (lambda: _RMSNorm(self.dh)) if qk_norm else (lambda: nn.Identity())
        self.q_norm, self.k_norm, self.cq_norm, self.ck_norm = norm(), norm(), norm(), norm()
        if adaln == "single":                                   # per-block offset to the shared time modulation
            self.mod_table = nn.Parameter(torch.zeros(9, dim))
        else:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def heads_view(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.dh).transpose(1, 2)

    def cross_kv(self, cond):
        k, v = self.ckv(cond).chunk(2, -1)
        return self.ck_norm(self.heads_view(k)), self.heads_view(v)

    def forward(self, x, mod, kv):
        B, N, C = x.shape
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = mod.unbind(1)
        q, k, v = self.qkv(_modulate(self.norm1(x), s1, c1)).chunk(3, -1)
        a = F.scaled_dot_product_attention(self.q_norm(self.heads_view(q)), self.k_norm(self.heads_view(k)), self.heads_view(v))
        x = x + g1[:, None] * self.proj(a.transpose(1, 2).reshape(B, N, C))
        q = self.cq_norm(self.heads_view(self.cq(_modulate(self.norm2(x), s2, c2))))
        a = F.scaled_dot_product_attention(q, *kv)
        x = x + g2[:, None] * self.cproj(a.transpose(1, 2).reshape(B, N, C))
        return x + g3[:, None] * self.mlp(_modulate(self.norm3(x), s3, c3))


class FlowPolicy(nn.Module):
    def __init__(self, arch, normalizer, pretrained=True):
        super().__init__()
        self.arch = a = dict(arch)
        if a["type"] != "flow_dit":
            raise ValueError(f"not a flow_dit arch: {a['type']}")
        self.vit = DinoV2(a["vit_dim"], a["vit_depth"], a["vit_heads"], a["patch"])
        if pretrained:
            load_dinov2(self.vit, a["encoder"])
        w, T, H = a["width"], a["obs_steps"], a["horizon"]
        self.tokens_per_frame = a["pool"] ** 2 + int(a["use_cls"])
        self.obs_proj = nn.Linear(a["vit_dim"], w)
        self.obs_pos = nn.Parameter(torch.zeros(1, 1, self.tokens_per_frame, w))
        self.frame_emb = nn.Parameter(torch.zeros(1, T, 1, w))
        self.proprio = nn.Sequential(nn.Linear(2 * T, w), nn.SiLU(), nn.Linear(w, w))
        self.proprio_emb = nn.Parameter(torch.zeros(1, 1, w))
        self.cond_norm = nn.LayerNorm(w, eps=1e-6)
        self.act_in = nn.Linear(2, w)
        self.act_pos = nn.Parameter(torch.zeros(1, H, w))
        self.t_mlp = nn.Sequential(nn.Linear(a["time_freq_dim"], w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList(DiTBlock(w, a["heads"], a["mlp_ratio"], a["adaln"], a["qk_norm"]) for _ in range(a["depth"]))
        if a["adaln"] == "single":
            self.t_mod = nn.Sequential(nn.SiLU(), nn.Linear(w, 9 * w))
        self.final_norm = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        self.out = nn.Linear(w, 2)
        self.normalizer = normalizer
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self._init_head()
        self.pose_head = None                     # created after the other weights: their initialisation is unchanged
        if a.get("pose_head"):
            g = a["vit_input"] // a["patch"]
            self.pose_head = PoseHead(a["vit_dim"], g * g, a["pose_proj"], a["pose_hidden"])

    def _init_head(self):
        for name, m in self.named_modules():
            if name.startswith("vit") or name.startswith("normalizer"):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for p in (self.obs_pos, self.frame_emb, self.proprio_emb, self.act_pos):
            nn.init.normal_(p, std=0.02)
        for lin in (self.t_mlp[0], self.t_mlp[2]):
            nn.init.normal_(lin.weight, std=0.02)
        zero = [self.final_mod[1], self.out] + ([self.t_mod[1]] if self.arch["adaln"] == "single" else [b.ada[1] for b in self.blocks])
        for lin in zero:                                                         # adaLN-Zero: identity blocks at start
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def parameter_counts(self):
        enc = sum(p.numel() for p in self.vit.parameters())
        head = sum(p.numel() for n, p in self.named_parameters() if not n.startswith(("vit.", "normalizer.", "pose_head.")))
        if self.pose_head is None:
            return {"encoder": enc, "head": head, "total": enc + head}
        pose = sum(p.numel() for p in self.pose_head.parameters())            # training only; not used by the policy
        return {"encoder": enc, "head": head, "pose_head": pose, "total": enc + head + pose}

    # -------------------------------------------------------------------------------------------- observation
    def _crop(self, x):
        c, S = self.arch["crop"], x.shape[-1]
        if self.training:                                     # official recipe: independent random crop per image
            n = len(x)
            oy = torch.randint(0, S - c + 1, (n,), device=x.device)
            ox = torch.randint(0, S - c + 1, (n,), device=x.device)
            r = torch.arange(c, device=x.device)
            rows, cols = (oy[:, None] + r)[:, None, :, None], (ox[:, None] + r)[:, None, None, :]
            return x[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
        o = (S - c) // 2
        return x[..., o:o + c, o:o + c]

    def encode(self, image, agent_pos, patch_grid=False):
        """image (B, T, 3, S, S) float in [0, 1]; agent_pos (B, T, 2) pixels -> condition tokens (B, N, width).
        patch_grid=True (training with the pose head) also returns the latest frame's patch tokens (B, h*w, vit_dim)."""
        B, T = image.shape[:2]
        x = self._crop(image.reshape(B * T, *image.shape[2:]).float())
        x = F.interpolate(x, size=(self.arch["vit_input"],) * 2, mode="bilinear", align_corners=False)
        tok, (h, w) = self.vit((x - self.img_mean) / self.img_std)
        grid = tok[:, 1:].transpose(1, 2).reshape(B * T, -1, h, w)
        pooled = F.adaptive_avg_pool2d(grid, self.arch["pool"]).flatten(2).transpose(1, 2)        # (BT, pool^2, 768)
        feats = torch.cat([tok[:, :1], pooled], 1) if self.arch["use_cls"] else pooled
        feats = self.obs_proj(feats).view(B, T, self.tokens_per_frame, -1) + self.obs_pos + self.frame_emb
        pos = self.normalizer["agent_pos"].normalize(agent_pos.float()).reshape(B, -1)
        cond = torch.cat([feats.flatten(1, 2), (self.proprio(pos) + self.proprio_emb[:, 0])[:, None]], 1)
        if patch_grid:
            return self.cond_norm(cond), tok.view(B, T, *tok.shape[1:])[:, -1, 1:]
        return self.cond_norm(cond)

    # -------------------------------------------------------------------------------------------- flow head
    def kv_cache(self, cond):
        return [blk.cross_kv(cond) for blk in self.blocks]

    def velocity(self, x, t, kv):
        """x (B, H, 2) normalized plan at flow time t (B,) in [0, 1] -> velocity (B, H, 2)."""
        B, w = len(x), self.arch["width"]
        h = self.act_in(x) + self.act_pos
        te = self.t_mlp(timestep_embedding(t * self.arch["time_scale"], self.arch["time_freq_dim"]))
        shared = self.t_mod(te).view(B, 9, w) if self.arch["adaln"] == "single" else None
        for blk, cache in zip(self.blocks, kv):
            mod = shared + blk.mod_table if shared is not None else blk.ada(te).view(B, 9, w)
            h = blk(h, mod, cache)
        shift, scale = self.final_mod(te).chunk(2, -1)
        return self.out(_modulate(self.final_norm(h), shift, scale))

    def sample_flow_time(self, n, device, generator=None):
        u = torch.rand(n, device=device, generator=generator) ** (1.0 / self.arch["beta_a"])   # Beta(a, 1) by inverse CDF
        return self.arch["flow_s"] * (1 - u)

    def compute_loss(self, image, agent_pos, action, generator=None, pose=None, pose_sym2=None):
        """Flow-matching loss for pixel-space plan labels action (B, H, 2).
        With pose (B, 3) block-pose labels (NaN rows: no label) and pose_sym2 (B,) bool (2-fold symmetric shape):
        returns (flow + pose_weight * pose loss, flow loss, pose loss, head outputs (B, 4)); the last three detached."""
        x1 = self.normalizer["action"].normalize(action.float())
        noise = torch.randn(x1.shape, device=x1.device, generator=generator)
        t = self.sample_flow_time(len(x1), x1.device, generator)
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * noise
        if pose is None:
            v = self.velocity(xt, t, self.kv_cache(self.encode(image, agent_pos)))
            return F.mse_loss(v.float(), x1 - noise)
        cond, grid = self.encode(image, agent_pos, patch_grid=True)
        flow = F.mse_loss(self.velocity(xt, t, self.kv_cache(cond)).float(), x1 - noise)
        pred = self.pose_head(grid).float()
        pose_loss = self.pose_loss(pred, pose.float(), pose_sym2)
        return flow + self.arch["pose_weight"] * pose_loss, flow.detach(), pose_loss.detach(), pred.detach()

    def pose_loss(self, pred, pose, sym2):
        """Mean over labelled samples of smooth-L1 (beta pose_beta) on the normalized position + squared error on
        (cos, sin), for 2-fold symmetric shapes the smaller of the errors for angle and angle + pi."""
        lab = torch.isfinite(pose).all(-1)
        tgt = pose_targets(torch.where(lab[:, None], pose, torch.zeros_like(pose)))
        pos = F.smooth_l1_loss(pred[:, :2], tgt[:, :2], beta=self.arch["pose_beta"], reduction="none").mean(-1)
        ang = (pred[:, 2:] - tgt[:, 2:]).pow(2).mean(-1)
        if sym2 is not None:
            ang = torch.where(sym2, torch.minimum(ang, (pred[:, 2:] + tgt[:, 2:]).pow(2).mean(-1)), ang)
        return ((pos + ang) * lab).sum() / lab.sum().clamp(min=1)

    @torch.no_grad()
    def integrate(self, cond, x, steps=None):
        """Euler integration from noise x (t = 0) to a plan (t = 1)."""
        steps = steps or self.arch["inference_steps"]
        kv = self.kv_cache(cond)
        x = x.float()
        for k in range(steps):
            t = torch.full((len(x),), k / steps, device=x.device)
            x = x + self.velocity(x, t, kv).float() / steps
        return x


# ----------------------------------------------------------------------------------------------- normalizer
def _linear_normalizer(state):
    from tdyssey.official import load_official_source
    load_official_source(ROOT / "third_party/diffusion_policy")
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    norm = LinearNormalizer()
    norm.load_state_dict(state)
    return norm.requires_grad_(False)


def official_normalizer(checkpoint=None):
    """The official checkpoint's LinearNormalizer (action, agent_pos, image), after its pinned size/hash check."""
    from tdyssey.official import official_spec
    from tdyssey.policy import hash_file
    checkpoint = Path(checkpoint or ROOT / "checkpoints/official-dp/ema_policy.pt")
    asset = official_spec()["compact"]
    if checkpoint.stat().st_size != asset["bytes"] or hash_file(checkpoint) != asset["sha256"]:
        raise ValueError("Official compact checkpoint differs from its pinned size/hash")
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    return _linear_normalizer({k[len("normalizer."):]: v.clone() for k, v in sd.items() if k.startswith("normalizer.")})


def head_state(net):
    """State dict without the (fixed) normalizer, which checkpoints store separately."""
    return {k: v for k, v in net.state_dict().items() if not k.startswith("normalizer.")}


def build_network(payload, pretrained=False):
    arch = payload["arch"]
    net = FlowPolicy(arch, _linear_normalizer(payload["normalizer"]), pretrained=pretrained)
    # strict, normalizer included: the official normalizer's loader rebuilds its tables from whatever keys it is given
    net.load_state_dict({**payload["ema"], **{"normalizer." + k: v for k, v in payload["normalizer"].items()}}, strict=True)
    return net


# ----------------------------------------------------------------------------------------------- agent
class FlowAgent:
    """Same calling interface as tdyssey.forward_policy.ForwardPolicy for evaluation / data collection."""

    def __init__(self, network, device="cuda", seed=0, encode_chunk=256, sample_chunk=128, amp=True):
        self.device = torch.device(device)
        self.network = network.to(self.device).eval().requires_grad_(False)
        self.arch = network.arch
        self.render_size = self.arch["image_size"]
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.encode_chunk, self.sample_chunk, self.amp = encode_chunk, sample_chunk, amp

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda")

    @torch.no_grad()
    def encode_batch(self, histories):
        histories = [list(h) for h in histories]
        image = torch.as_tensor(np.stack([np.stack([o["image"] for o in h]) for h in histories]), device=self.device)
        pos = torch.as_tensor(np.stack([np.stack([o["agent_pos"] for o in h]) for h in histories]), device=self.device)
        if image.shape[1:] != (self.arch["obs_steps"], 3, self.render_size, self.render_size):
            raise ValueError(f"unexpected observation shape {tuple(image.shape)}")
        out = []
        with self._autocast():
            for s in range(0, len(image), self.encode_chunk):
                out.append(self.network.encode(image[s:s + self.encode_chunk].float(), pos[s:s + self.encode_chunk].float()).float())
        return torch.cat(out)

    def encode(self, history):
        return self.encode_batch([history])

    @torch.no_grad()
    def sample_batch(self, condition, generators, mode="actor"):
        """Row i starts from noise drawn with generators[i] (shape (1, 16, 2), as the official agent draws it)."""
        if mode != "actor":
            raise ValueError("FlowAgent has only the actor view")
        generators = list(generators)
        if len(generators) != condition.shape[0]:
            raise ValueError("Use one generator per conditioning row")
        H = self.arch["horizon"]
        x = torch.cat([torch.randn((1, H, 2), device=self.device, generator=g) for g in generators])
        out = []
        with self._autocast():
            for s in range(0, len(x), self.sample_chunk):
                out.append(self.network.integrate(condition[s:s + self.sample_chunk], x[s:s + self.sample_chunk]))
        x = torch.cat(out)
        a0 = self.arch["exec_start"]
        actions = self.network.normalizer["action"].unnormalize(x)[:, a0:a0 + self.arch["n_exec"]].cpu().numpy()
        if not np.isfinite(actions).all() or not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite sampled plan")
        return actions, x.detach()

    def sample(self, condition, mode="actor"):
        actions, x = self.sample_batch(condition, [self.generator], mode)
        return actions[0], x

    def child_generators(self, count):
        seeds = torch.randint(0, 2**62, (count,), device=self.device, generator=self.generator).tolist()
        return [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]


def load_agent(payload, device="cuda"):
    """FlowAgent from a flow_train.py checkpoint payload (or its path)."""
    if not isinstance(payload, dict):
        payload = torch.load(payload, map_location="cpu", weights_only=True)
    return FlowAgent(build_network(payload), device)
