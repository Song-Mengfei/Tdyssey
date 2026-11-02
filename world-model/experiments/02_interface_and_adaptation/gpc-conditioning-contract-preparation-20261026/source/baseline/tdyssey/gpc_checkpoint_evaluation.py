"""Fixed checkpoint500 evaluation only; imports never load a tensor runtime."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from . import gpc_adaptation as adaptation
from .gpc_adaptation_runtime import AdaptationBudget, AdaptationRuntimeError, _exclusive, _json


CANDIDATE = "gpc-checkpoint500-evaluation-01"
AUTHORIZATION = "20261026-user-autonomous-gpc-checkpoint500-evaluation-01"
OUTPUT = "artifacts/gpc-checkpoint500-evaluation-20261026"
SOURCES = {
    "frozen": dict(path="artifacts/frozen-wm-assets/denoiser.pth", bytes=35043916,
        sha256="4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892"),
    "adapted": dict(path="artifacts/gpc-supervised-adaptation-pilot03-20261026/adapted-latest.pt", bytes=105134565,
        sha256="0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3", logical_update=500),
}
SEEDS = (5478, 5479, 5480)
MODES = (("AR", (1, 2, 3, 4)), ("TF", (2, 4)))
ROLE_FORECASTS = 111*3*6
PROGRESS_MAX = 2+2+32+3996+2+1
JOURNAL_BUDGET = 16*1024**2
METADATA_SLOT = 65536
MAX_FAILURE_REASON = 2000


def configuration_template():
    """Pure JSON template; only the immutable original config is opened."""
    raw = adaptation.CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != adaptation.BASE_CONFIG_SHA256:
        raise ValueError("Original scientific configuration changed")
    value = json.loads(raw)
    for key in ("train", "optimizer", "objective", "trainable_prefixes"):
        value.pop(key)
    value.update(kind="tdyssey.gpc_checkpoint_evaluation.v1", candidate_id=CANDIDATE,
                 output_dir=OUTPUT, sources=deepcopy(SOURCES))
    value["limits"].update(model_loads=2, checkpoint_bytes=0, max_updates=0, max_train_examples=0)
    return value


def validate_config(config):
    if not isinstance(config, dict) or config != configuration_template():
        raise ValueError("Configuration differs from the fixed evaluation-only protocol")
    return deepcopy(config)


def _forecast_plan(case_keys):
    return tuple((anchor, candidate, seed, mode, horizon) for anchor, candidate in case_keys
                 for seed in SEEDS for mode, horizons in MODES for horizon in horizons)


class EvaluationBudget(AdaptationBudget):
    """Fresh narrow account; reuse the journal, clocks and first-error writer."""
    CANDIDATE_ID = CANDIDATE
    AUTHORIZATION_ID = AUTHORIZATION
    ACCOUNT_KIND = "gpc_checkpoint_evaluation_account.v1"
    CASE_COUNT = 111
    LOSS_PER_ROLE = 16
    FORECASTS_PER_ROLE = ROLE_FORECASTS
    FORECAST_COUNT = 3996
    EULER_COUNT = 11988
    TOTAL_SECONDS = 3600
    FORECAST_SEEDS = SEEDS
    FORECAST_MODES = MODES

    @classmethod
    def _plan(cls, case_keys):
        return tuple((*key, seed, mode, horizon) for key in case_keys
                     for seed in cls.FORECAST_SEEDS
                     for mode, horizons in cls.FORECAST_MODES for horizon in horizons)

    def _append(self, kind, data):
        upper = len(_json(data))+512
        if upper > METADATA_SLOT or self._journal_bytes+upper+2*METADATA_SLOT > JOURNAL_BUDGET:
            self._broken = True
            raise AdaptationRuntimeError("Evaluation journal metadata admission exceeds the fixed 16MiB reserve")
        return super()._append(kind, data)

    def fail(self, reason):
        # Full original exceptions/tracebacks remain the heartbeat's first cause.
        return super().fail(str(reason)[:MAX_FAILURE_REASON])

    @classmethod
    def create(cls, ledger_dir, binding, *, case_keys, deadline_seconds, clock=time.time, monotonic=time.monotonic):
        if (not isinstance(binding, dict) or binding.get("candidate_id") != cls.CANDIDATE_ID
                or binding.get("authorization_id") != cls.AUTHORIZATION_ID):
            raise ValueError("Fixed evaluation execution binding required")
        if not 0 < deadline_seconds <= cls.TOTAL_SECONDS:
            raise ValueError("Evaluation uses the original unextended total deadline")
        case_keys = tuple(tuple(key) for key in case_keys)
        if (len(case_keys) != cls.CASE_COUNT or len(set(case_keys)) != cls.CASE_COUNT
                or any(len(key) != 2 or any(not isinstance(v, str) or not 0 < len(v) <= 64 for v in key) for key in case_keys)):
            raise ValueError(f"Exactly {cls.CASE_COUNT} distinct predeclared cases required")
        wall, mono = clock(), monotonic()
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (wall, mono)):
            raise ValueError("Finite evaluation clocks required")
        wall_deadline = min(wall+deadline_seconds, binding.get("deadline_unix", wall+deadline_seconds))
        mono_deadline = min(mono+deadline_seconds, binding.get("deadline_monotonic", mono+deadline_seconds))
        if (not math.isfinite(wall_deadline) or not math.isfinite(mono_deadline)
                or wall_deadline <= wall or mono_deadline <= mono):
            raise ValueError("Original evaluation deadline already expired")
        path = Path(ledger_dir).absolute()
        attempt = dict(kind=cls.ACCOUNT_KIND, budget_id=uuid.uuid4().hex,
            binding=deepcopy(binding), binding_sha256=hashlib.sha256(adaptation.encoded(binding)).hexdigest(),
            ledger_dir=str(path), source_logical_steps=500, source_weights=deepcopy(SOURCES),
            forecast_order_sha256=hashlib.sha256(adaptation.encoded(cls._plan(case_keys))).hexdigest(),
            limits=dict(max_updates=0, max_train_examples=0, max_eval_batches=2*cls.LOSS_PER_ROLE,
                max_eval_examples=2*cls.LOSS_PER_ROLE, max_batch_size=1,
                max_forecasts=cls.FORECAST_COUNT, max_euler_steps=cls.EULER_COUNT, max_model_loads=2,
                deadline_seconds=deadline_seconds), started_unix=wall, deadline_unix=wall_deadline,
            started_monotonic=mono, deadline_monotonic=mono_deadline)
        # The failure record contains both reason and state.first_reason. JSON
        # needs at most 12 ASCII bytes per Unicode character; fixed state and
        # the longest allowed pending branch metadata fit the other 4096 bytes.
        if len(adaptation.encoded(attempt))+2*MAX_FAILURE_REASON*12+4096 > METADATA_SLOT:
            raise ValueError("Evaluation account identity leaves no bounded first-failure headroom")
        path.mkdir(parents=True, exist_ok=False); _exclusive(path/"attempt.json", attempt)
        self = cls._initialize(path, attempt, clock, monotonic)
        self._training_schedule_identity = None
        self._forecast_order = cls._plan(case_keys)
        self._state.update(source_logical_steps=500, active_models=0, released_model_roles=[],
            per_role={role: dict(reserved_loss=0, completed_loss=0, reserved_forecasts=0, completed_forecasts=0)
                      for role in ("frozen", "adapted")}, evaluation_complete=False)
        self._append("created", {})
        return self

    @classmethod
    def resume(cls, *args, **kwargs):
        raise AdaptationRuntimeError("Evaluation-only accounts cannot resume old or new work")

    def _deny(self, operation):
        error = AdaptationRuntimeError("Evaluation-only account forbids "+operation)
        self.fail(str(error)); raise error

    def reserve_update(self, *args, **kwargs): return self._deny("reserve_update")
    def complete_update(self, *args, **kwargs): return self._deny("complete_update")
    def select_checkpoint(self, *args, **kwargs): return self._deny("checkpoint selection")
    def reserve_optimizer(self, *args, **kwargs): return self._deny("optimizer")
    def restore_rng(self, *args, **kwargs): return self._deny("training RNG restore")
    def restore(self, *args, **kwargs): return self._deny("checkpoint/account restore")

    def _transition(self, kind, data, sequence):
        state = self._state
        if kind in ("created", "failed"):
            return super()._transition(kind, data, sequence)
        if kind == "model_released":
            cleanup = data.get("cleanup") is True
            if (data.get("role") != state["active_model_role"] or state["active_models"] != 1
                    or (cleanup and state["status"] != "stopped")
                    or (not cleanup and (state["status"] != "active" or state["pending"] is not None))):
                raise AdaptationRuntimeError("Model release does not match the one active role")
            if not cleanup:
                role = state["per_role"][data["role"]]
                if (role["completed_loss"] != self.LOSS_PER_ROLE
                        or role["completed_forecasts"] != self.FORECASTS_PER_ROLE):
                    raise AdaptationRuntimeError("Cannot release an incomplete role as successful")
            state["active_models"] = 0; state["active_model_role"] = None
            state["released_model_roles"].append(data["role"])
            return
        if state["status"] != "active":
            raise AdaptationRuntimeError("Terminal evaluation cannot acquire operations")
        if kind == "reserve_model_load":
            roles = ("frozen", "adapted"); index = state["spent_reserved_model_loads"]
            if (state["pending"] is not None or state["active_models"] or index >= 2
                    or data.get("role") != roles[index]
                    or (index == 1 and state["released_model_roles"] != ["frozen"])):
                raise AdaptationRuntimeError("Exactly two serial frozen/release/adapted loads required")
            role = data["role"]
            state["spent_reserved_model_loads"] += 1; state["model_load_roles"].append(role)
            state["active_models"] = 1; state["active_model_role"] = role
            state["pending"] = dict(operation_id=sequence, kind="model_load", role=role)
            return
        if kind == "complete_model_load":
            pending = state["pending"]
            if not pending or pending["kind"] != "model_load" or data.get("operation_id") != pending["operation_id"]:
                raise AdaptationRuntimeError("Model-load completion differs from reservation")
            state["completed_model_loads"] += 1; state["pending"] = None
            return
        role = state["active_model_role"]
        if kind in ("reserve_eval", "reserve_forecast"):
            if state["pending"] is not None or state["active_models"] != 1 or role not in state["per_role"]:
                raise AdaptationRuntimeError("Evaluation requires its one loaded role and no pending operation")
            current = state["per_role"][role]; metadata = data.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("role") != role:
                raise AdaptationRuntimeError("Evaluation metadata role differs")
            if kind == "reserve_eval":
                if (type(data.get("batch_size")) is not int or data["batch_size"] != 1
                        or set(metadata) != {"role", "example"}
                        or type(metadata["example"]) is not int or metadata["example"] != current["reserved_loss"]
                        or current["reserved_loss"] >= self.LOSS_PER_ROLE or current["reserved_forecasts"]):
                    raise AdaptationRuntimeError("Loss examples differ from the fixed per-role allowance")
                current["reserved_loss"] += 1; state["spent_reserved_eval_batches"] += 1; state["spent_eval_examples"] += 1
                state["pending"] = dict(operation_id=sequence, kind="eval", batch_size=1, role=role, metadata=deepcopy(metadata))
            else:
                key = (*metadata.get("case", ()), metadata.get("mode"), metadata.get("horizon"))
                ordinal = current["reserved_forecasts"]
                if (type(data.get("actual_count")) is not int or data["actual_count"] != 1
                        or type(data.get("euler_per_image")) is not int or data["euler_per_image"] != 3
                        or set(metadata) != {"role", "case", "mode", "horizon"}
                        or type(metadata["horizon"]) is not int or current["completed_loss"] != self.LOSS_PER_ROLE
                        or ordinal >= self.FORECASTS_PER_ROLE or key != self._forecast_order[ordinal]):
                    raise AdaptationRuntimeError("Forecast differs from the full fixed branch/seed/AR/TF order")
                current["reserved_forecasts"] += 1
                state["spent_reserved_forecasts"] += 1; state["spent_reserved_euler_steps"] += 3
                state["pending"] = dict(operation_id=sequence, kind="forecast", actual_count=1, euler_steps=3,
                                        role=role, metadata=deepcopy(metadata))
            return
        if kind in ("complete_eval", "complete_forecast"):
            pending = state["pending"]
            wanted = "eval" if kind == "complete_eval" else "forecast"
            if not pending or pending["kind"] != wanted or data.get("operation_id") != pending["operation_id"]:
                raise AdaptationRuntimeError("Evaluation completion differs from reservation")
            current = state["per_role"][pending["role"]]
            super()._transition(kind, data, sequence)
            current["completed_loss" if wanted == "eval" else "completed_forecasts"] += 1
            return
        if kind == "evaluation_finished":
            if (state["pending"] is not None or state["active_models"] or state["released_model_roles"] != ["frozen", "adapted"]
                    or state["completed_eval_batches"] != 2*self.LOSS_PER_ROLE
                    or state["completed_forecasts"] != self.FORECAST_COUNT
                    or state["completed_euler_steps"] != self.EULER_COUNT or state["completed_model_loads"] != 2):
                raise AdaptationRuntimeError("Cannot complete a partial evaluation")
            state["evaluation_complete"] = True
            return
        raise AdaptationRuntimeError("Operation forbidden in evaluation-only account: "+str(kind))

    def release_model(self, role, *, cleanup=False):
        if cleanup:
            with self._lock:
                self._append("model_released", dict(role=role, cleanup=True))
                return self.snapshot()
        self._operation("model_released", dict(role=role, cleanup=False))
        return self.snapshot()

    def finish(self):
        self._operation("evaluation_finished", {})
        return self.snapshot()


def _validated_cases(cases):
    cases = list(cases)
    keys = [(row["anchor"], row["candidate"]) for row in cases]
    if (len(cases) != 111 or len(set(keys)) != 111
            or sum(row.get("role") == "regression" for row in cases) != 63
            or sum(row.get("role") == "independent" for row in cases) != 48
            or any(row.get("shape") not in ("T", "L") or len(row.get("truth", [])) != 4 for row in cases)):
        raise ValueError("Full 63 regression + 48 independent physical branches required")
    return cases, keys


def _default_measure(raw, shape, goal):
    from .wm_compat_engine import compact_measurement
    from .wm_image_measurement_v2 import decode
    return compact_measurement(decode(raw, shape, goal))


def run_evaluation(prepared, binding, *, factory, close_model, writer, make_sampler=None,
                   progress=lambda *args: None, memory_check=lambda: None, case_loader=None,
                   dataset_factory=None, state_loader=None, loss_evaluator=None, rollout=None,
                   measure=None, reporter=None, clock=time.time, monotonic=time.monotonic):
    """Fresh evaluation; injected marker hooks never invoke real tensor code.

    factory(role, state_dict=...) constructs exactly one role using model-only
    state. The default loss, rollout, sampler and report preserve old semantics.
    """
    from .gpc_evaluation_assets import check_prepared_assets, load_model_only_state
    if prepared.get("ready") is not True:
        raise ValueError("Readiness and reviewed source assets are required")
    cfg = validate_config(prepared["config"])
    assets = prepared["assets"]; check_prepared_assets(assets)
    if (assets.get("sources") != cfg["sources"] or assets.get("source_logical_steps") != 500
            or assets.get("new_account_updates") != 0):
        raise ValueError("Source checkpoint provenance must remain separate from zero new updates")
    case_loader = case_loader or adaptation.load_cases
    if dataset_factory is None:
        from .gpc_adaptation_data import GPCAdaptationDataset
        dataset_factory = lambda data: GPCAdaptationDataset(data["path"], "independent_validation", require_ready=True)
    state_loader = state_loader or load_model_only_state
    loss_evaluator = loss_evaluator or adaptation.evaluate_loss
    rollout = rollout or adaptation.rollout_case
    make_sampler = make_sampler or (lambda model: adaptation.NativeSampler(model, memory_check))
    if reporter is None:
        from .gpc_adaptation_reporting import summarize
        reporter = summarize
    measure = measure or _default_measure
    cases, case_keys = _validated_cases(case_loader(prepared))
    dataset = dataset_factory(prepared["data"])
    seconds = min(cfg["limits"]["wall_seconds"], binding["deadline_unix"]-clock(),
                  binding["deadline_monotonic"]-monotonic())
    budget = EvaluationBudget.create(writer.directory/"ledger", binding, case_keys=case_keys,
        deadline_seconds=seconds, clock=clock, monotonic=monotonic)
    model = sampler = state_dict = None
    close_attempted = False
    loss_rows, matched, forecast_rows = {}, {}, []
    callbacks = 0

    def check_progress(phase, step=0):
        nonlocal callbacks
        callbacks += 1
        if callbacks > PROGRESS_MAX:
            raise AdaptationRuntimeError("Fixed progress publication bound exceeded")
        budget.check(); memory_check(); progress(phase, step, budget.snapshot())

    try:
        for role in ("frozen", "adapted"):
            close_attempted = False
            operation = budget.reserve_model_load(role)
            check_progress("model_loading")
            state_dict = state_loader(role, assets)
            model = factory(role, state_dict=state_dict); state_dict = None
            model.requires_grad_(False).eval()
            memory_check(); budget.complete_model_load(operation)
            check_progress("evaluation")
            loss_rows[role] = loss_evaluator(model, dataset, cfg=cfg, budget=budget, role=role, progress=check_progress)
            sampler = make_sampler(model)
            for case in cases:
                for seed in SEEDS:
                    rows = rollout(case, seed, role=role, sampler=sampler, rng=sampler, budget=budget,
                        matched=matched, sink=writer, measure=measure, progress=check_progress)
                    forecast_rows.extend(rows)
            sampler = model = None
            close_attempted = True; close_model()
            budget.release_model(role)
            check_progress("evaluation")
        if ([row["noise"] for row in loss_rows["frozen"]] != [row["noise"] for row in loss_rows["adapted"]]
                or [row["index"] for row in loss_rows["frozen"]] != [row["index"] for row in loss_rows["adapted"]]):
            raise ValueError("Both roles require exactly matched independent validation noise/examples")
        expected = {(role, *key) for role in ("frozen", "adapted") for key in _forecast_plan(case_keys)}
        observed = [(row.get("role"), row.get("anchor"), row.get("candidate"), row.get("seed"), row.get("mode"), row.get("horizon"))
                    for row in forecast_rows]
        if len(observed) != 3996 or set(observed) != expected or any(row.get("status") != "completed" for row in forecast_rows):
            raise ValueError("Every planned forecast row, including invalid measurements, must remain in the report")
        state = budget.finish()
        writer.json("screens.json", reporter(forecast_rows, SEEDS, planned_cases=cases))
        report = dict(kind="gpc_checkpoint_evaluation_result.v1", status="complete",
            source_logical_steps=500, new_optimizer_updates=0, new_training_examples=0,
            model_only_loads=2, optimizer_state_restored=False, training_rng_restored=False,
            old_forecasts_reused=0, loss_rows=loss_rows, forecast_records=len(forecast_rows), budget=state,
            progress_publications=callbacks+1, max_progress_publications=PROGRESS_MAX,
            physical_operations=0, policy_operations=0, new_collection=0,
            acceptance="Completion is not acceptance; retain original per-stratum evidence and support gates")
        writer.json("completion.json", report)
        check_progress("complete")
        return report
    except BaseException as error:
        budget.fail("evaluation_only_failed: "+type(error).__name__+": "+str(error))
        raise
    finally:
        import sys
        primary = sys.exc_info()[1]
        state_dict = sampler = model = None
        active_role = budget.snapshot()["active_model_role"]
        if active_role is not None and not close_attempted:
            try:
                close_attempted = True; close_model()
                budget.release_model(active_role, cleanup=True)
            except BaseException as cleanup:
                if primary is not None:
                    primary.add_note("Evaluation model cleanup failed: "+str(cleanup))
                else:
                    raise
