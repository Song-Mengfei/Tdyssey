"""Explicit stage3C metadata migration around the unchanged native train bytecode.

No model framework is imported here. Checkpoint loading/state transfer happens
only when the externally timed training run invokes the private restore adapter.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import types

PARENT_SHA256 = "e94f9349e1affb8372aacc4613bd57ade496704a50b63d257f77cb4c5ad3626a"
PHASE = "stage3c-visible-microfit-continuation"
PRIOR_SECONDS = 444.6250000000073
START_UPDATE = 129
START_DRAWS = 2064
TARGET_UPDATE = 250
LOCAL_ATTEMPTS = 121
CONFIG_CHANGE_KEYS = frozenset((
    "kind", "output", "stage_max_updates", "stage_max_model_work_seconds",
    "no_new_update_after_seconds", "prior_stage", "prior_charged_model_seconds",
    "prior_charged_updates", "prior_ledger_sha256", "prior_model_work_file",
    "parent_checkpoint", "parent_checkpoint_sha256", "continuation_start_updates",
    "continuation_target_updates"))
SOURCE_ADDITIONS = frozenset((
    "tdyssey/native_rgb_stage3c_continuation.py",
    "scripts/native_rgb_stage3c_run.py",
    "configs/native_rgb_supervised_stage3c.json"))
STATE_KEYS = ("model", "optimizer", "scaler", "scheduler", "rng", "order")


def _sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_digest(value):
    """Type/shape/dtype-aware CPU state digest; no imports or RNG draws."""
    digest = hashlib.sha256()

    def emit(item):
        if hasattr(item, "detach") and hasattr(item, "cpu"):
            item = item.detach().cpu().contiguous().numpy()
        if hasattr(item, "dtype") and hasattr(item, "shape") and hasattr(item, "tobytes"):
            digest.update(b"array\0" + str(item.dtype).encode() + b"\0")
            digest.update(json.dumps(list(item.shape)).encode() + b"\0")
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                emit(key)
                emit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0" + str(len(item)).encode() + b"\0")
            for child in item:
                emit(child)
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + str(len(item)).encode() + b"\0" + item)
        elif isinstance(item, float):
            digest.update(b"float\0" + struct.pack("!d", item))
        elif item is None or isinstance(item, (str, bool, int)):
            digest.update(type(item).__name__.encode() + b"\0" + repr(item).encode() + b"\0")
        else:
            raise TypeError("Unsupported protected state type: " + type(item).__name__)

    emit(value)
    return digest.hexdigest()


def validate_closed_prior_ledger(value, expected):
    """Validate the new authorization starts after the exact closed 3+3B cost."""
    if (value != expected or value.get("active_stage") is not None
            or value.get("charged_updates") != START_UPDATE
            or value.get("completed_updates") != START_UPDATE
            or value.get("charged_seconds") != PRIOR_SECONDS
            or value.get("max_updates") != 3000 or value.get("max_seconds") != 5400.
            or len(value.get("stages", [])) != 2
            or value["stages"][-1]["phase"] != "stage3b-visible-microfit"
            or value["stages"][-1]["reserved_updates"] != START_UPDATE
            or value["stages"][-1]["completed_updates"] != START_UPDATE
            or value["stages"][-1]["status"] != "bounded_stop"):
        raise ValueError("Exact closed stage3+3B ledger/cost is required; no reset or refund")
    return dict(max_attempts=LOCAL_ATTEMPTS, max_seconds=600., update_cutoff_seconds=420.)


def _current_source(core, parent_identity, additions):
    original = core.source_identity()
    if original != parent_identity["source"]:
        raise ValueError("Original core/source semantics changed since parent checkpoint")
    if set(additions) != SOURCE_ADDITIONS:
        raise ValueError("Exactly the three declared stage3C source additions are required")
    result = deepcopy(original)
    root = Path(core.ROOT).resolve()
    for name, expected in additions.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in result["files"] or _sha256_file(path) != expected:
            raise ValueError("Continuation source hash/path differs: " + name)
        result["files"][name] = expected
    return result


def _validate_new_config(config, parent_config, changes, output, resume):
    if set(changes) != CONFIG_CHANGE_KEYS:
        raise ValueError("Only the exact fourteen continuation metadata changes are allowed")
    expected = deepcopy(parent_config)
    expected.update(changes)
    if config != expected:
        raise ValueError("Continuation config differs from the explicit literal metadata delta")
    fixed = dict(kind="native-gpc-stage3c-exact-continuation.v1", output=str(Path(output).resolve()),
        stage_max_updates=LOCAL_ATTEMPTS, stage_max_model_work_seconds=600.,
        no_new_update_after_seconds=420., prior_stage="3B", prior_charged_model_seconds=PRIOR_SECONDS,
        prior_charged_updates=START_UPDATE, parent_checkpoint=str(Path(resume).resolve()),
        parent_checkpoint_sha256=PARENT_SHA256, continuation_start_updates=START_UPDATE,
        continuation_target_updates=TARGET_UPDATE)
    if any(config[key] != value for key, value in fixed.items()):
        raise ValueError("Continuation phase/target/local caps/parent binding differs")
    return expected


def _validate_parent_payload(payload, spec):
    required = {"model", "optimizer", "scaler", "scheduler", "rng", "order", "progress", "config", "identity", "ledger"}
    if not required.issubset(payload):
        raise ValueError("Incomplete parent training checkpoint")
    if (payload["config"] != spec["config"] or payload["identity"] != spec["identity"]
            or payload["progress"] != spec["progress"]):
        raise ValueError("Parent payload config/identity/progress differs from frozen expected metadata")
    progress = payload["progress"]
    scheduler = payload["scheduler"]
    optimizer = payload["optimizer"]
    lr = 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * START_UPDATE / 3000))
    if (progress["updates"] != START_UPDATE or progress["samples"] != START_DRAWS
            or progress.get("attempted_draws") != START_DRAWS or payload["order"]["cursor"] != START_DRAWS
            or len(progress["initial_parameter_sha256"]) != 296
            or len(optimizer["state"]) != 296 or len(optimizer["param_groups"]) != 1
            or scheduler["T_max"] != 3000 or scheduler["last_epoch"] != START_UPDATE
            or scheduler["eta_min"] != 1e-5):
        raise ValueError("Parent update/order/optimizer/scheduler boundary is not complete129")
    group = optimizer["param_groups"][0]
    if (not math.isclose(group["lr"], lr, rel_tol=1e-12) or group["initial_lr"] != 1e-4
            or group["weight_decay"] != .01 or tuple(group["betas"]) != (.9, .999)
            or group["eps"] != 1e-8 or scheduler["base_lrs"] != [1e-4]
            or not math.isclose(scheduler["_last_lr"][0], lr, rel_tol=1e-12)
            or any(float(state["step"]) != START_UPDATE for state in optimizer["state"].values())):
        raise ValueError("Parent optimizer state/learning-rate schedule differs")


def train_continuation(config, output, data_root, source_root, init_checkpoint, *, parent_spec,
        allowed_config_changes, source_additions, review_path, resume, work_budget,
        execute=False, phase=PHASE, updates=TARGET_UPDATE, _core_module=None, **train_kwargs):
    """Run the original train bytecode with three private, audited global bindings.

    The underscored core injection exists only for CPU fixtures. Normal callers
    use the frozen native_rgb_training module; no shared globals are patched.
    """
    if execute is not True or phase != PHASE or updates != TARGET_UPDATE:
        raise ValueError("Only the released stage3C total250 continuation may execute")
    if train_kwargs.get("before_updates_callback") is not None:
        raise ValueError("Stage3C must not rerun a frozen or update129 native baseline")
    if _core_module is None:
        from . import native_rgb_training as core
    else:
        core = _core_module
    output, resume, review_path = Path(output).resolve(), Path(resume).resolve(), Path(review_path).resolve()
    parent = deepcopy(parent_spec)
    changes = deepcopy(allowed_config_changes)
    additions = deepcopy(source_additions)
    if (parent["checkpoint_sha256"] != PARENT_SHA256 or parent["last"]["sha256"] != PARENT_SHA256
            or Path(parent["last"]["path"]).resolve() != resume or _sha256_file(resume) != PARENT_SHA256
            or resume.stat().st_size != parent["last"]["bytes"] or output == resume.parent
            or (output / "last.pt").exists() or review_path.exists()):
        raise ValueError("Pinned parent/new output/checkpoint review binding differs")
    _validate_new_config(config, parent["config"], changes, output, resume)
    _current_source(core, parent["identity"], additions)
    prior_work_file = Path(config["prior_model_work_file"]).resolve()
    if (prior_work_file != resume.parent.parent / "model-work.json"
            or _sha256_file(prior_work_file) != parent["model_work_sha256"]):
        raise ValueError("Frozen parent model-work identity differs")
    prior_work = json.loads(prior_work_file.read_text(encoding="utf-8"))
    prior = prior_work["budget"]["ledger"]
    prior_copy = output.parent / "prior-ledger.json"
    if _sha256_file(prior_copy) != config["prior_ledger_sha256"]:
        raise ValueError("Frozen closed prior ledger hash differs")
    validate_closed_prior_ledger(json.loads(prior_copy.read_text(encoding="utf-8")), prior)
    current = work_budget.ledger.snapshot()
    active = current["active_stage"]
    if (active is None or active["token"] != work_budget.stage.token or active["phase"] != PHASE
            or active["seconds_before"] != PRIOR_SECONDS or active["reserved_updates"] != 0
            or active["completed_updates"] != 0 or current["charged_updates"] != START_UPDATE
            or current["completed_updates"] != START_UPDATE or current["stages"] != prior["stages"]
            or current["historical_cancelled_experiment"] != prior["historical_cancelled_experiment"]
            or work_budget.max_attempts != LOCAL_ATTEMPTS or work_budget.max_seconds != 600.
            or work_budget.latest_update_seconds != 420. or work_budget.initial_tail_reserve_seconds < 180.
            or work_budget.ledger.max_updates != 3000 or work_budget.ledger.max_seconds != 5400.):
        raise ValueError("Continuation active ledger ownership/local121/global3000 budget differs")
    called = False
    restore_verified = False

    def save_training_checkpoint(path, model, optimizer, scaler, scheduler, generator, order,
            progress, config, identity, *, ledger_state=None):
        # The original train exception path may otherwise save its fresh t0
        # objects after a rejected/unfinished restore. Never label those usable.
        if (not restore_verified or type(progress.get("updates")) is not int
                or not START_UPDATE <= progress["updates"] <= TARGET_UPDATE
                or progress.get("samples", 0) < START_DRAWS or progress.get("phase") != PHASE
                or Path(path).resolve() != output / "last.pt"):
            raise RuntimeError("Unverified continuation state may not replace a complete checkpoint")
        return core.save_training_checkpoint(path, model, optimizer, scaler, scheduler, generator,
            order, progress, config, identity, ledger_state=ledger_state)

    def source_identity():
        return _current_source(core, parent["identity"], additions)

    def restore_checkpoint(path, model, optimizer, scaler, scheduler, generator, order, new_config, identity):
        nonlocal called, restore_verified
        if called or Path(path).resolve() != resume:
            raise ValueError("Exactly one pinned parent restore is permitted")
        called = True
        _validate_new_config(new_config, parent["config"], changes, output, resume)
        expected_identity = deepcopy(parent["identity"])
        expected_identity.update(phase=PHASE, source=source_identity(), model_work_budget=work_budget.identity())
        if identity != expected_identity:
            raise ValueError("Continuation identity differs outside phase/budget/declared real source")
        work_budget.require_operation(30., "complete parent checkpoint restore and fingerprint")
        if _sha256_file(resume) != PARENT_SHA256:
            raise ValueError("Parent checkpoint changed before restore")
        payload = core.load_checkpoint_payload(resume)
        _validate_parent_payload(payload, parent)
        old_ledger = payload["ledger"]
        if (old_ledger["charged_updates"] != START_UPDATE or old_ledger["completed_updates"] != START_UPDATE
                or old_ledger["max_updates"] != 3000 or old_ledger["max_seconds"] != 5400.
                or old_ledger["stages"] != prior["stages"][:-1]
                or old_ledger["active_stage"]["token"] != prior["stages"][-1]["token"]
                or old_ledger["charged_seconds"] > PRIOR_SECONDS):
            raise ValueError("Parent checkpoint ledger is not the matching pre-save3B snapshot")
        before = {key: state_digest(payload[key]) for key in STATE_KEYS}
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        scheduler.load_state_dict(payload["scheduler"])
        order.load_state_dict(payload["order"])
        core.restore_rng(payload["rng"], generator)
        actual = core._cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), order=order.state_dict(),
            rng=core.capture_rng(generator)))
        after = {key: state_digest(actual[key]) for key in STATE_KEYS}
        del actual
        if before != after:
            raise ValueError("Protected model/optimizer/scaler/scheduler/RNG/order state changed on restore")
        parameter_hashes = core.tensor_hashes(model)
        previous_fitted = json.loads((resume.parent.parent / "predictions/fitted-train18.json").read_text(encoding="utf-8"))
        if parameter_hashes != previous_fitted["parameter_sha256"]:
            raise ValueError("Restored parameter hashes differ from the actual update129 fitted model")
        progress = deepcopy(payload["progress"])
        previous_stop = progress.pop("stop_reason", None)
        previous_phase = progress["phase"]
        progress["phase"] = PHASE
        progress["stage3c_resume_parameter_sha256"] = parameter_hashes
        progress["continuation"] = dict(parent_checkpoint=str(resume), parent_checkpoint_sha256=PARENT_SHA256,
            start_updates=START_UPDATE, start_samples=START_DRAWS, target_updates=TARGET_UPDATE,
            previous_phase=previous_phase, prior_stop_reason=previous_stop,
            inherited_first_gradient_norms_scope="stage3B; unchanged cumulative evidence, not remeasured stage3C",
            inherited_initial_parameter_hashes_scope="original published initialization; unchanged",
            state_sha256=before)
        restore_verified = True
        work_budget.before_checkpoint(output / "last.pt")
        boundary = save_training_checkpoint(output / "last.pt", model, optimizer, scaler, scheduler,
            generator, order, progress, new_config, identity, ledger_state=work_budget.ledger.snapshot())
        if state_digest(core.capture_rng(generator)) != after["rng"]:
            raise ValueError("Boundary save changed resumed RNG")
        peak = dict(restore_boundary_peak_cuda_allocated=None, restore_boundary_peak_cuda_reserved=None,
                    restore_boundary_peak_note="CUDA counters unavailable; no extra import or probe")
        framework = sys.modules.get("torch")
        try:
            if framework is not None and framework.cuda.is_initialized():
                peak = dict(restore_boundary_peak_cuda_allocated=int(framework.cuda.max_memory_allocated()),
                    restore_boundary_peak_cuda_reserved=int(framework.cuda.max_memory_reserved()),
                    restore_boundary_peak_note="Existing CUDA allocation counters before core post-restore peak reset; inside model clock")
        except (AttributeError, RuntimeError) as error:
            peak["restore_boundary_peak_note"] = type(error).__name__ + ": " + str(error)
        core.atomic_json(review_path, dict(kind="stage3c-complete-state-restore-review.v1", passed=True,
            parent_checkpoint=str(resume), parent_checkpoint_sha256=PARENT_SHA256,
            parent_config=parent["config"], new_config=new_config, allowed_config_changes=changes,
            parent_identity=parent["identity"], new_identity=identity, protected_before_sha256=before,
            protected_after_sha256=after, all_protected_state_equal=True, updates=START_UPDATE,
            samples=START_DRAWS, order_cursor=order.state_dict()["cursor"], optimizer_state_entries=296,
            scheduler_T_max=3000, scheduler_last_epoch=START_UPDATE,
            stage3c_resume_parameter_sha256=parameter_hashes, boundary_checkpoint=boundary,
            prior_stop_reason=previous_stop, cumulative_initial_and_gradient_fields_preserved=True,
            progress_metadata_changes=["phase", "stop_reason removed", "continuation added", "stage3c_resume_parameter_sha256 added"],
            model_work_elapsed=work_budget.elapsed, no_native_baseline_called=True,
            live_ledger_not_restored_from_checkpoint=True,
            private_global_overrides=["restore_checkpoint", "source_identity", "save_training_checkpoint"],
            train_bytecode_unchanged=True, unverified_or_fresh_checkpoint_save_rejected=True, **peak))
        return progress

    namespace = core.train.__globals__.copy()
    namespace.update(restore_checkpoint=restore_checkpoint, source_identity=source_identity,
                     save_training_checkpoint=save_training_checkpoint)
    cloned = types.FunctionType(core.train.__code__, namespace, core.train.__name__,
        core.train.__defaults__, core.train.__closure__)
    cloned.__kwdefaults__ = deepcopy(core.train.__kwdefaults__)
    cloned.__annotations__ = dict(core.train.__annotations__)
    result = cloned(config, output, data_root, source_root, init_checkpoint, resume=resume,
        execute=True, phase=phase, updates=updates, work_budget=work_budget, **train_kwargs)
    if not called:
        raise RuntimeError("Continuation training returned without restoring the pinned parent")
    return result
