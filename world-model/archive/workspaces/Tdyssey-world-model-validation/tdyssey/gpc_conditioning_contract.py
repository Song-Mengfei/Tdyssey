"""Two explicit conditioning arms around the unchanged pinned native sampler.

Import is standard-library-only. Real Torch is imported only by an explicitly
executed factory call; engineering tests inject an ordinary arithmetic facade.
This module grants no execution authority and owns no model, RNG or call budget.
"""
from __future__ import annotations

import hashlib
import json
import math

SAMPLER_SPEC = dict(num_steps_denoising=3, sigma_min=.002, sigma_max=5,
                    rho=7, order=1, s_churn=0, s_cond=0)
_FIXED_SAMPLER_ITEMS = tuple(SAMPLER_SPEC.items())
CONTRACT_SPEC = dict(version="gpc_clean_history_conditioning.v1",
    arms={"A": {"sigma_cond": None}, "B": {"sigma_cond": .002}},
    sampler=SAMPLER_SPEC, history_noise=False,
    implementation="original native denoise unbound on a private observing facade",
    changed_conditioner_only="c_noise_cond", model_or_sampler_copy=False)
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT_SPEC, sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
_CONDITIONERS = ("c_in", "c_out", "c_skip", "c_noise", "c_noise_cond")


def _values(value):
    """Small actual conditioner values; no recomputation of their formula."""
    values = value.detach().reshape(-1).cpu().tolist()
    if not isinstance(values, list):
        values = [values]
    if not 1 <= len(values) <= 64:
        raise ValueError("conditioner observation exceeds bounded batch")
    values = [float(item) for item in values]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("nonfinite native conditioner")
    return values


class _ConditioningFacade:
    """Dynamic native self-dispatch; upstream objects/methods are never patched."""
    def __init__(self, delegate, native_denoise, arm, torch_module, observer):
        self._delegate = delegate
        self._native_denoise = native_denoise
        self.arm = arm
        self._torch = torch_module
        self._observer = observer
        self.records = []
        self._denoise_calls = 0

    @property
    def cfg(self):
        return self._delegate.cfg

    @property
    def device(self):
        return self._delegate.device

    def denoise(self, noisy_next_obs, sigma, sigma_cond, obs, act):
        if sigma_cond is not None:
            raise ValueError("native s_cond must remain zero/None in both arms")
        self._denoise_calls += 1
        effective = (None if self.arm == "A" else self._torch.full(
            (noisy_next_obs.shape[0],), fill_value=.002, device=noisy_next_obs.device))
        # Original method sequence and quantization; native self dispatch observes
        # exactly the conditioners used by the model and wrap, once per call.
        return self._native_denoise(self, noisy_next_obs, sigma, effective, obs, act)

    def compute_conditioners(self, sigma, sigma_cond):
        conditioners = self._delegate.compute_conditioners(sigma, sigma_cond)
        record = dict(arm=self.arm, denoise_call=self._denoise_calls,
            sigma=_values(sigma),
            sigma_cond=None if sigma_cond is None else _values(sigma_cond),
            conditioners={name: _values(getattr(conditioners, name)) for name in _CONDITIONERS})
        self.records.append(record)
        if self._observer is not None:
            self._observer(json.loads(json.dumps(record, allow_nan=False)))
        return conditioners

    def compute_model_output(self, noisy_next_obs, obs, act, conditioners):
        return self._delegate.compute_model_output(noisy_next_obs, obs, act, conditioners)

    def wrap_model_output(self, noisy_next_obs, model_output, conditioners):
        return self._delegate.wrap_model_output(noisy_next_obs, model_output, conditioners)

    def apply_noise(self, *args, **kwargs):
        raise RuntimeError("clean-history contract forbids historical noise injection")


def make_conditioning_sampler(denoiser, source, arm, *, execute=False,
                              torch_module=None, observer=None):
    """Return the actual pinned DiffusionSampler, with its s_cond unchanged.

    source is independently verified by the caller. No model construction/load or
    sampler call occurs here. An injected fake torch_module permits marker tests.
    Real execution still requires separate authority, asset checks and matched
    sampler RNG capsules; this factory grants none of these.

    The returned sampler's ``denoiser.records`` observes each native denoise call.
    A caller reusing the sampler can retain the new slice after each forecast;
    a three-Euler forecast emits three records. Both arms keep history clean and
    do not reproduce the offset/history noise used in training.
    """
    if arm not in ("A", "B"):
        raise ValueError("only explicit conditioning arms A/B are defined")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable")
    if torch_module is None:
        if execute is not True:
            raise PermissionError("real Torch requires separately authorized explicit execution")
        import torch as torch_module
    elif execute is not True and getattr(torch_module, "__name__", None) == "torch":
        raise PermissionError("real Torch injection requires explicit execution")
    facade = _ConditioningFacade(denoiser, source.Denoiser.denoise, arm, torch_module, observer)
    sampler = source.DiffusionSampler(facade, source.DiffusionSamplerConfig(**dict(_FIXED_SAMPLER_ITEMS)))
    if any(getattr(sampler.cfg, name) != value for name, value in _FIXED_SAMPLER_ITEMS):
        raise ValueError("native sampler changed the fixed three-Euler contract")
    return sampler
