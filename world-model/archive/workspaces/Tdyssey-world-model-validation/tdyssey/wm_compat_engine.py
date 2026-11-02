"""Paired saved-context prediction mechanics, injectable without a real model.

Only the optional TorchBackend constructor imports/loads GPC. Array/RNG tests
exercise the same engine with a tiny fake sampler and a controllable RNG.
No simulator, policy or optimizer is constructed anywhere in this module.
"""

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .wm_compat_transform import transform_for_arm
from .wm_compat_appearance import appearance_metrics


FRAME_BYTES = 3 * 96 * 96 * 4
RNG_MAX_BYTES = 1024
FORECAST_RECORD_BYTES = 3072
REAL_RECORD_BYTES = 2304


def array_hash(value):
    array = np.asarray(value)
    return hashlib.sha256(array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()).hexdigest()


def rng_hash(state):
    if not isinstance(state, bytes) or not 0 < len(state) <= RNG_MAX_BYTES:
        raise ValueError("RNG state must be nonempty bytes within its declared storage bound")
    return hashlib.sha256(state).hexdigest()


def _real_images(value, count):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32
            or value.shape != (count, 3, 96, 96) or not np.isfinite(value).all()
            or value.min() < 0 or value.max() > 1):
        raise ValueError("Real RGB requires exact float32 shape and finite [0,1] values; no clipping")
    return value


@dataclass(frozen=True)
class SavedBranch:
    anchor: str
    candidate: str
    shape: str
    history: np.ndarray
    past_actions: np.ndarray
    future_actions: np.ndarray
    actual: np.ndarray
    goal: object = None
    truth: object = None
    latest: object = None
    contact_by_h4: object = None

    def __post_init__(self):
        if self.shape not in ("T", "L") or self.candidate not in ("hold", "toward", "side", "away"):
            raise ValueError("Unsupported fixed branch")
        if self.anchor not in [f"T{i}" for i in range(12,16)]+[f"L{i:02d}" for i in range(12)] or self.anchor[0]!=self.shape:
            raise ValueError("Anchor and shape must match the fixed plan")
        _real_images(self.history, 4)
        _real_images(self.actual, 4)
        for name, count in (("past_actions", 3), ("future_actions", 4)):
            actions = np.asarray(getattr(self, name))
            if actions.shape != (count, 2) or not np.isfinite(actions).all() or actions.min() < 0 or actions.max() > 511:
                raise ValueError("Actions must be finite absolute workspace targets in [0,511]")


def action_window(branch, horizon):
    if type(horizon) is not int or not 1 <= horizon <= 4:
        raise ValueError("Horizon must be 1..4")
    values = np.concatenate((branch.past_actions, branch.future_actions[:horizon]), axis=0)[-4:]
    return values.astype(np.float32) / np.float32(511) * np.float32(2) - np.float32(1)


def real_context(branch, horizon):
    """TF conditioning ends immediately before its target; H1 is not sampled twice."""
    if horizon not in (1, 2, 4):
        raise ValueError("Only H1 reuse and TF H2/H4 are declared")
    frames = np.concatenate((branch.history, branch.actual[:horizon-1]), axis=0)[-4:]
    return frames.copy()


def arm_context(frames, arm):
    transformed = [transform_for_arm(frame, arm)[0] for frame in frames]
    return np.stack(transformed)


def compact_measurement(value):
    """Keep independent records, bounded fit diagnostics and explicit uncertainty."""
    fields = {
        "state": ("valid", "pose", "candidate_pose", "reason", "ambiguity_assessed", "ambiguity"),
        "shape": ("valid", "reason", "area_ratio", "component_mass_fraction", "full_soft_iou", "roi_soft_iou",
                  "squared_residual", "outside_roi_template_mass", "palette", "alternative_fits",
                  "fitted_outline", "fitted_blur_sigma"),
        "pusher": ("valid", "position", "reason", "soft_iou", "fit_iou", "area_ratio", "component_mass_fraction"),
        "coverage": ("valid", "estimate", "interval", "uncertainty_kind", "uncertainty_valid"),
        "termination": ("status", "reason", "threshold", "development_assumption"),
    }
    output = {}
    for record, names in fields.items():
        source = value.get(record, {})
        if not isinstance(source, dict):
            raise ValueError("Invalid independent measurement record: " + record)
        output[record] = {key: source[key] for key in names if key in source}
    return output


class ChunkSink:
    """One immutable NPY/JSON chunk at a time; no growing archive rewrites."""
    def __init__(self, budget):
        self.budget = budget

    def raw(self, key, frame):
        buffer = io.BytesIO()
        np.save(buffer, frame, allow_pickle=False)
        payload = buffer.getvalue()
        if len(payload) > FRAME_BYTES + 256:
            raise ValueError("Unexpected NPY serialization bound")
        name = "predictions/" + key.replace("/", "-") + ".npy"
        self.budget.write_chunk(name, payload, raw_bytes=FRAME_BYTES, max_bytes=FRAME_BYTES+256)
        return name

    def state(self, key, state):
        rng_hash(state)
        name = "rng/" + key.replace("/", "-") + ".bin"
        self.budget.write_chunk(name, state, max_bytes=RNG_MAX_BYTES)

    def record(self, name, record, *, maximum=FORECAST_RECORD_BYTES, final=False):
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
        if len(payload) > maximum:
            raise ValueError("Metadata exceeds predeclared serialization bound: " + name)
        self.budget.write_chunk(name, payload, max_bytes=maximum, final=final)


def run_paired_branch(branch, seed, *, sampler, rng, budget, sink=None, measure=None, progress=None):
    """Generate AR A/B then TF H2/H4 with identical per-horizon RNG states.

    sampler.sample receives only readonly RGB and normalized action arrays.
    measure receives only RGB, known shape and fixed goal, never future truth.
    This API has no simulator callback. The caller owns M/T/L authorization.
    """
    from .wm_compat_budget import call_key
    if type(seed) is not int or seed not in (5478, 5479, 5480):
        raise ValueError("Only the three fixed seeds are declared")
    original_rng = rng.capture()
    rng_hash(original_rng)
    entries, exits, rows = {}, {}, []

    def invoke(arm, mode, horizon, context):
        key = call_key(branch.anchor, branch.candidate, seed, arm, mode, horizon)
        completion_started = False
        if arm == "A" and mode == "AR":
            entry = rng.capture()
            entries[horizon] = entry
            if sink is not None:
                sink.state(key, entry)
        else:
            rng.restore(entries[horizon])
            entry = rng.capture()
            if entry != entries[horizon]:
                raise ValueError("Sampler entry RNG replay differs")
        entry_id = rng_hash(entry)
        context = context.copy()
        actions = action_window(branch, horizon)
        context.setflags(write=False)
        actions.setflags(write=False)
        budget.reserve_call(key, entry_id)
        try:
            sample_started=time.monotonic()
            output = sampler.sample(context, actions)
            sampler_seconds=time.monotonic()-sample_started
            if (not isinstance(output, np.ndarray) or output.dtype != np.float32
                    or output.shape != (3, 96, 96) or not np.isfinite(output).all()):
                raise ValueError("Sampler output must be finite raw float32 RGB")
            output = output.copy()  # No range clipping or recoloring of generated frames.
            exit_state = rng.capture()
            exit_id = rng_hash(exit_state)
            if arm == "A" and mode == "AR":
                exits[horizon] = exit_state
            elif exit_state != exits[horizon]:
                raise ValueError("Paired sampler RNG consumption differs")
            raw_path = sink.raw(key, output) if sink is not None else None
            measurement = compact_measurement(measure(output, branch.shape, branch.goal)) if measure else None
            row = dict(key=key, anchor=branch.anchor, candidate=branch.candidate, shape=branch.shape,
                seed=seed, arm=arm, mode=mode, horizon=horizon, status="completed",
                source_history_sha256=array_hash(branch.history),
                source_context_sha256=array_hash(real_context(branch,horizon)) if mode=="TF" else array_hash(branch.history),
                conditioning_source="saved_real_window" if mode=="TF" else "raw_AR_feedback",
                rng_entry=entry_id, rng_exit=exit_id, input_sha256=array_hash(context),
                action_sha256=array_hash(actions), raw_sha256=array_hash(output), raw_path=raw_path,
                raw_min=float(output.min()), raw_max=float(output.max()),
                sampler_seconds=sampler_seconds,
                clipped_fraction=float(((output < 0) | (output > 1)).mean()),
                measurement=measurement, truth=branch.truth[horizon-1] if branch.truth else None,
                contact_by_h4=branch.contact_by_h4,
                appearance=appearance_metrics(output,branch,arm,horizon))
            if sink is not None:
                sink.record("records/"+key.replace("/", "-")+".json", row)
            completion_started = True
            budget.complete_call(key, exit_id, array_hash(output), FRAME_BYTES)
            rows.append(row)
            if progress:
                progress(row)
            return output
        except BaseException as exc:
            reason = type(exc).__name__+": "+str(exc)
            # A completion/IO failure may already have durably closed the call.
            # Never issue a second failure transition that overwrites its reason.
            state = budget.snapshot() if hasattr(budget, "snapshot") else None
            pending = (next((r for r in state["calls"] if r["key"] == key), {}).get("status") == "pending"
                       if state is not None else not completion_started)
            try:
                if pending and (state is None or state["status"] == "active"):
                    budget.fail_call(key, reason)
                else:
                    budget.stop(reason)
            except BaseException:
                pass  # Preserve the original error and the first durable stop.
            raise

    try:
        rng.seed(seed)
        for arm in ("A", "B"):
            context = arm_context(branch.history, arm)
            for horizon in range(1, 5):
                output = invoke(arm, "AR", horizon, context)
                context = np.concatenate((context[1:], output[None]), axis=0)
        for arm in ("A", "B"):
            for horizon in (2, 4):
                context = arm_context(real_context(branch, horizon), arm)
                invoke(arm, "TF", horizon, context)
    except BaseException as exc:
        try:
            budget.stop(type(exc).__name__+": "+str(exc))
        except BaseException:
            pass
        raise
    finally:
        primary_error=sys.exc_info()[1]
        try:
            rng.restore(original_rng)
        except BaseException as exc:
            try:
                budget.stop("external_rng_restore_failed: "+str(exc))
            except BaseException:
                pass
            if primary_error is None:
                raise
            primary_error.add_note("Secondary external RNG restore failure: "+type(exc).__name__+": "+str(exc))
    return dict(rows=rows, h1_reuse={arm:call_key(branch.anchor, branch.candidate, seed, arm, "AR", 1)
                                  for arm in ("A", "B")})


class TorchBackend:
    """Lazy real backend for a separately authorized guarded invocation only."""
    def __init__(self, source_root, checkpoint):
        import torch
        from .gpc_world_model import GPCWorldModel
        self.torch = torch
        # Network construction must not alter the caller's CPU generator.
        with torch.random.fork_rng(devices=[]):
            self.model = GPCWorldModel(source_root, checkpoint)
        self.device = self.model.device
        self.original = self.capture()
        self.weight_identity = self._fingerprint()

    def _fingerprint(self):
        digest = hashlib.sha256()
        for name,value in sorted(self.model.network.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def capture(self):
        return self.torch.cuda.get_rng_state(self.device).cpu().numpy().tobytes()

    def restore(self, state):
        rng_hash(state)
        tensor = self.torch.from_numpy(np.frombuffer(state, dtype=np.uint8).copy())
        self.torch.cuda.set_rng_state(tensor, self.device)

    def seed(self, seed):
        with self.torch.cuda.device(self.device):
            self.torch.cuda.manual_seed(seed)

    def sample(self, context, normalized_actions):
        torch = self.torch
        with torch.inference_mode(), torch.cuda.device(self.device):
            images = torch.tensor(context, dtype=torch.float32, device=self.device)[None]
            actions = torch.tensor(normalized_actions, dtype=torch.float32, device=self.device)[None]
            prediction, _ = self.model.sampler.sample(images, actions)
            torch.cuda.synchronize(self.device)
            self.stats()  # Peak counters catch a transient breach between heartbeats.
            return prediction[0].cpu().numpy()

    def stats(self):
        torch = self.torch
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        if peak_allocated > 512*1024**2 or peak_reserved > 1024**3:
            raise MemoryError("CUDA allocator peak exceeded the fixed compatibility ceiling")
        return dict(model_instances=1, model_loads=1, optimizer_updates=0,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                    cuda_peak_allocated_bytes=peak_allocated,cuda_peak_reserved_bytes=peak_reserved)

    def close(self):
        try:
            if self._fingerprint()!=self.weight_identity:
                raise RuntimeError("Frozen world-model weights changed")
        finally:
            primary_error=sys.exc_info()[1]
            try:
                self.restore(self.original)
            except BaseException as exc:
                if primary_error is None:
                    raise
                primary_error.add_note("Secondary backend RNG restore failure: "+type(exc).__name__+": "+str(exc))
