"""Clean/weak saved-RGB conditioning; no model construction or training entry.

Actions arrive already normalized once. The target residual objective and target
noise draw order are native; only the history-noise policy changes. New target
RNG state is independent of the branch and weak-history streams, so it is not
claimed to reproduce an unaligned old combined history/target RNG stream.
"""
from dataclasses import dataclass
import hashlib
import json
import math

POLICY = {
    "kind": "native-gpc-clean-weak-history.v1",
    "branch_unit": "one Bernoulli per microbatch, shared by its samples",
    "clean_probability": .5,
    "clean_sigma_cond": None,
    "clean_history_offset": 0.,
    "clean_history_epsilon": 0.,
    "weak_sigma_unit": "independent log-uniform per sample of a weak microbatch",
    "weak_sigma_min": .002,
    "weak_sigma_max": .05,
    "weak_history_offset": 0.,
    "target_sigma_loc": -1.2,
    "target_sigma_scale": 1.2,
    "target_sigma_min": .002,
    "target_sigma_max": 20.,
    "target_offset_std": .1,
    "target_draw_order": ["sigma normal", "offset normal", "epsilon normal"],
    "actions": "already float32[-1,1], pass through without normalization",
}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha(value):
    """Explicit optional audit only; no weight/model operation is performed."""
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


class ConditioningRNGs:
    """Three explicit generators; serializable CPU state tensors, no globals."""
    def __init__(self, branch, target, weak, *, seeds):
        import torch
        if not all(isinstance(g, torch.Generator) for g in (branch, target, weak)):
            raise TypeError("Three explicit torch.Generator instances required")
        if str(branch.device) != "cpu" or target.device != weak.device:
            raise ValueError("Branch RNG must be CPU; target/weak must share the input device")
        if len({id(branch), id(target), id(weak)}) != 3:
            raise ValueError("Branch, target and weak generators must be independent objects")
        self.branch, self.target, self.weak = branch, target, weak
        self.seeds = dict(seeds)

    @property
    def device(self):
        return self.target.device

    def hashes(self):
        return {name: tensor_sha(getattr(self, name).get_state()) for name in ("branch", "target", "weak")}

    def state_dict(self):
        return dict(kind="native-gpc-clean-weak-rngs.v1", policy_sha256=POLICY_SHA,
            seeds=dict(self.seeds), target_device=str(self.device), branch_device="cpu",
            states={name: getattr(self, name).get_state().cpu().clone() for name in ("branch", "target", "weak")},
            state_sha256=self.hashes())

    def load_state_dict(self, state):
        import torch
        if (set(state) != {"kind", "policy_sha256", "seeds", "target_device", "branch_device", "states", "state_sha256"}
                or state["kind"] != "native-gpc-clean-weak-rngs.v1" or state["policy_sha256"] != POLICY_SHA
                or state["seeds"] != self.seeds or state["target_device"] != str(self.device)
                or state["branch_device"] != "cpu" or set(state["states"]) != {"branch", "target", "weak"}
                or set(state["state_sha256"]) != {"branch", "target", "weak"}):
            raise ValueError("Conditioning RNG policy/seeds/device/state identity differs")
        # Validate all states on temporary explicit generators before mutating any
        # of the live streams; failed restore must not partially advance them.
        temporary = make_conditioning_rngs(**self.seeds, device=self.device)
        for name in ("branch", "target", "weak"):
            value = state["states"][name]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8
                    or value.ndim != 1 or tensor_sha(value) != state["state_sha256"][name]):
                raise ValueError("Conditioning RNG tensor/checksum differs: "+name)
            getattr(temporary, name).set_state(value.clone())
        for name in ("branch", "target", "weak"):
            getattr(self, name).set_state(getattr(temporary, name).get_state())


def make_conditioning_rngs(*, branch_seed, target_seed, weak_seed, device="cpu"):
    import torch
    seeds = dict(branch_seed=branch_seed, target_seed=target_seed, weak_seed=weak_seed)
    if any(type(s) is not int or not 0 <= s < 2**63 for s in seeds.values()) or len(set(seeds.values())) != 3:
        raise ValueError("Three distinct unsigned63-bit seeds are required")
    return ConditioningRNGs(torch.Generator(device="cpu").manual_seed(branch_seed),
        torch.Generator(device=device).manual_seed(target_seed), torch.Generator(device=device).manual_seed(weak_seed), seeds=seeds)


def _batch(batch, device):
    import torch
    if not isinstance(batch, dict) or not all(k in batch for k in ("image", "action")):
        raise ValueError("Explicit normalized image/action batch required")
    image, action = batch["image"], batch["action"]
    if (not isinstance(image, torch.Tensor) or not isinstance(action, torch.Tensor)
            or image.dtype != torch.float32 or action.dtype != torch.float32
            or image.ndim != 5 or tuple(image.shape[1:]) != (5, 3, 96, 96) or len(image) < 1
            or tuple(action.shape) != (len(image), 4, 2) or image.device != device or action.device != device
            or image.requires_grad or action.requires_grad or not bool(torch.isfinite(image).all())
            or not bool(torch.isfinite(action).all()) or bool((image < 0).any()) or bool((image > 1).any())
            or bool((action < -1).any()) or bool((action > 1).any())):
        raise ValueError("Expected constant finite float32 image[B,5,3,96,96]0..1 and action[B,4,2]-1..1 on explicit RNG device")
    return image, action


@dataclass(frozen=True)
class ConditioningNoisePlan:
    branch: str
    branch_uniform: float
    sigma_cond: object
    history_offset: object
    history_epsilon: object
    sigma: object
    target_offset: object
    target_epsilon: object
    rng_before: dict
    rng_after: dict
    policy_sha256: str = POLICY_SHA

    def tensor_hashes(self):
        return {name: None if getattr(self, name) is None else tensor_sha(getattr(self, name))
            for name in ("sigma_cond", "history_offset", "history_epsilon", "sigma", "target_offset", "target_epsilon")}


def make_conditioning_plan(batch, rngs):
    import torch
    if not isinstance(rngs, ConditioningRNGs):
        raise TypeError("Explicit ConditioningRNGs required")
    image, _ = _batch(batch, rngs.device); b = len(image); before = rngs.hashes()
    uniform = float(torch.rand((), generator=rngs.branch, device="cpu", dtype=torch.float32))
    branch = "clean" if uniform < .5 else "weak"
    # Original target *substream* order, with exactly the original distribution.
    # History branch/shape never consumes this independent target generator.
    def target_normal(shape):
        return torch.randn(shape, device=image.device, dtype=torch.float32, generator=rngs.target)
    sigma = (target_normal((b,))*1.2-1.2).exp().clamp(.002, 20.)
    target_offset = .1*target_normal((b, 3, 1, 1))
    target_epsilon = target_normal((b, 3, 96, 96))
    history_offset = torch.zeros((b, 12, 1, 1), device=image.device, dtype=torch.float32)
    if branch == "clean":
        sigma_cond = None
        history_epsilon = torch.zeros((b, 12, 96, 96), device=image.device, dtype=torch.float32)
    else:
        u = torch.rand((b,), device=image.device, dtype=torch.float32, generator=rngs.weak)
        sigma_cond = (math.log(.002)+u*(math.log(.05)-math.log(.002))).exp().clamp(.002, .05)
        history_epsilon = torch.randn((b, 12, 96, 96), device=image.device, dtype=torch.float32, generator=rngs.weak)
    return ConditioningNoisePlan(branch, uniform, sigma_cond, history_offset, history_epsilon,
        sigma, target_offset, target_epsilon, before, rngs.hashes())


def _validate_plan(plan, image):
    import torch
    if (not isinstance(plan, ConditioningNoisePlan) or plan.policy_sha256 != POLICY_SHA
            or plan.branch not in ("clean", "weak") or not math.isfinite(plan.branch_uniform)
            or not 0 <= plan.branch_uniform < 1
            or (plan.branch_uniform < .5) != (plan.branch == "clean")):
        raise ValueError("Explicit clean/weak plan identity/branch differs")
    b = len(image)
    for name, shape in (("history_offset", (b, 12, 1, 1)), ("history_epsilon", (b, 12, 96, 96)),
            ("sigma", (b,)), ("target_offset", (b, 3, 1, 1)), ("target_epsilon", (b, 3, 96, 96))):
        value = getattr(plan, name)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != image.device
                or value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError("Conditioning noise tensor shape/device/dtype differs: "+name)
    if bool((plan.sigma < .002).any()) or bool((plan.sigma > 20).any()) or bool(plan.history_offset.any()):
        raise ValueError("Native target sigma bounds or zero history offset differ")
    if plan.branch == "clean":
        if plan.sigma_cond is not None or bool(plan.history_epsilon.any()):
            raise ValueError("Clean means actual None condition and exactly zero history noise")
    else:
        v = plan.sigma_cond
        if (not isinstance(v, torch.Tensor) or v.shape != (b,) or v.dtype != torch.float32
                or v.device != image.device or v.requires_grad or not bool(torch.isfinite(v).all())
                or bool((v < .002).any()) or bool((v > .05).any())):
            raise ValueError("Weak condition must be its real per-sample sigma in.002...05")


def conditioning_residual_loss(denoiser, batch, *, rngs=None, noise_plan=None):
    """Native FP32 residual MSE with clean/weak history; no optimizer/sampler.

    Clean passes sigma_cond=None to the actual native compute_conditioners.
    Target epsilon/offset never enter that history-condition argument.
    """
    import torch
    import torch.nn.functional as F
    if (rngs is None) == (noise_plan is None):
        raise ValueError("Provide exactly explicit rngs or explicit noise_plan")
    cfg = denoiser.cfg
    if (cfg.sigma_data != .5 or cfg.sigma_offset_noise != .1 or cfg.noise_previous_obs is not True
            or cfg.inner_model.num_steps_conditioning != 4 or getattr(denoiser, "is_upsampler", False)):
        raise ValueError("Original native model/objective conditioning required")
    device = next(denoiser.parameters()).device
    image, action = _batch(batch, device)
    if rngs is not None and rngs.device != device:
        raise ValueError("Noise generators and denoiser must use the same device")
    plan = noise_plan if noise_plan is not None else make_conditioning_plan(batch, rngs)
    _validate_plan(plan, image)
    b = len(image); history = image[:, :4].reshape(b, 12, 96, 96); clean_target = image[:, 4]
    noisy_history = history if plan.branch == "clean" else history+plan.history_epsilon*plan.sigma_cond[:, None, None, None]
    noisy_target = clean_target+plan.target_offset+plan.target_epsilon*plan.sigma[:, None, None, None]
    cs = denoiser.compute_conditioners(plan.sigma, plan.sigma_cond)
    output = denoiser.compute_model_output(noisy_target, noisy_history, action, cs)
    if output.shape != clean_target.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("Nonfinite/wrong-shape native residual output")
    target = (clean_target-cs.c_skip*noisy_target)/cs.c_out
    loss = F.mse_loss(output.float(), target.float())
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise ValueError("Native residual loss must remain finite FP32")
    return loss, dict(loss_denoising=float(loss.detach()), batch_size=b, target_frames=1,
        history_branch=plan.branch, branch_uniform=plan.branch_uniform,
        sigma_cond_is_none=plan.sigma_cond is None,
        history_sigma_min=None if plan.sigma_cond is None else float(plan.sigma_cond.min()),
        history_sigma_max=None if plan.sigma_cond is None else float(plan.sigma_cond.max()),
        history_offset_std=0., model_output_calls=1, inference_sampler_calls=0,
        actions_renormalized=False, policy_sha256=POLICY_SHA,
        rng_before=plan.rng_before, rng_after=plan.rng_after)
