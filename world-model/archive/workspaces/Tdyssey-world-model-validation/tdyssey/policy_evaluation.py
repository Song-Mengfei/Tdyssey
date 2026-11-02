"""Independent, budgeted development evaluation; no optimizer or training loop.

Production callers load one method in its own process. Identity markers prevent
accidental method mislabelling; they are not cryptographic weight proofs. The
CLI must verify the actual base/delta files before binding those markers.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
import math
import random
import re
import time

from .collection import reset_issues
from .envs.official import PROTOCOL, T_REFERENCE_COG, matching_shape_state, official_shape
from .official import physical_metrics, physical_state


ORIGINAL_T_PROTOCOL = "official-t-original-v1"
MATCHED_TL_PROTOCOL = PROTOCOL
METHODS = ("base_original", "base_training", "delta_training")
SCHEMA_VERSION = 1
np = torch = None  # Tensor dependencies load only for actual/fake evaluation.


def _runtime_dependencies():
    global np, torch
    if np is None or torch is None:
        import numpy as runtime_numpy
        import torch as runtime_torch
        np, torch = runtime_numpy, runtime_torch


def _seed(value, label):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{label} must be uint32")
    return value


@dataclass(frozen=True)
class EvaluationCase:
    shape: str
    scene_seed: int
    policy_seed: int

    def __post_init__(self):
        if self.shape not in ("T", "L"):
            raise ValueError("Development evaluation supports T/L cases")
        _seed(self.scene_seed, "scene_seed")
        _seed(self.policy_seed, "policy_seed")


def _identity(value):
    if not isinstance(value, dict) or set(value) != {"base_sha256", "delta_sha256"}:
        raise ValueError("Evaluation identity needs base_sha256 and delta_sha256")
    for key, digest in value.items():
        if digest is None and key == "delta_sha256":
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid evaluation {key}")
    return dict(value)


def bind_evaluation_identity(actor, *, base_sha256, delta_sha256=None):
    """Bind verified loader provenance, without inspecting/copying any weights."""
    actor.evaluation_checkpoint_identity = _identity({"base_sha256": base_sha256,
                                                     "delta_sha256": delta_sha256})
    return actor


def _check_method(actor, method, delta_applied, checkpoint_identity):
    if method not in METHODS or type(delta_applied) is not bool:
        raise ValueError("Choose an explicit evaluation method and delta_applied flag")
    identity = _identity(checkpoint_identity)
    if getattr(actor, "evaluation_checkpoint_identity", None) != identity:
        raise ValueError("Actor loader identity marker differs from evaluation provenance")
    expected_delta = method == "delta_training"
    if delta_applied != expected_delta or (identity["delta_sha256"] is not None) != expected_delta:
        raise ValueError("Base/delta provenance does not match the named evaluation method")
    return identity, "baseline" if method == "base_original" else "training"


@contextmanager
def _isolated_state(official, actor):
    """Restore global RNGs, existing local generators, module modes and scheduler.

    The evaluator never receives a training environment, cursor or episode
    buffer. DenoisingPolicy is stateless between complete plans except for the
    scheduler attributes restored here. Production evaluation uses a process of
    its own rather than sharing a stateful controller with ongoing training.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in actor.modules()]
    network = actor.network
    scheduler = getattr(network, "noise_scheduler", None)
    scheduler_state = {}
    for name in ("timesteps", "num_inference_steps", "custom_timesteps"):
        if scheduler is not None and hasattr(scheduler, name):
            value = getattr(scheduler, name)
            scheduler_state[name] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
    candidates = [getattr(official, "generator", None), getattr(actor, "generator", None),
                  getattr(network, "kwargs", {}).get("generator")]
    generators = {id(value): value for value in candidates if isinstance(value, torch.Generator)}
    generator_states = [(generator, generator.get_state().clone()) for generator in generators.values()]
    try:
        actor.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        for name, value in scheduler_state.items():
            setattr(scheduler, name, value)
        for generator, state in generator_states:
            generator.set_state(state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _counts():
    return dict.fromkeys(("reset_attempts", "reset_physics_substeps", "planning_attempts",
                          "planning_calls", "denoising_steps", "control_attempts", "control_steps",
                          "control_physics_substeps"), 0)


def _metrics(env, metrics_fn):
    measured = metrics_fn(env)
    result = {key: float(measured[key]) for key in ("coverage", "iou")}
    if any(not math.isfinite(value) or not -1e-9 <= value <= 1 + 1e-9 for value in result.values()):
        raise ValueError("Evaluation coverage/IoU must be finite proportions")
    return result


def _reserve(budget, kind, phase, shape):
    budget.check_time()
    return budget.reserve(kind, phase=phase, shape=shape, amount=1)


def _reset(env, seed, budget, phase, shape, case):
    env.seed(seed)
    reservation = _reserve(budget, "reset", phase, shape)
    case["counts"]["reset_attempts"] += 1
    before = time.perf_counter()
    case["interrupted_operation"] = "reset"
    try:
        observation = env.reset()
    finally:
        case["latency"]["reset_seconds"] += time.perf_counter() - before
    case["interrupted_operation"] = None
    case["counts"]["reset_physics_substeps"] += 1
    budget.mark_completed(reservation)
    return observation


def _initial_observation(environments, case_spec, protocol, budget, phase, result):
    if protocol == ORIGINAL_T_PROTOCOL:
        return _reset(environments["T"], case_spec.scene_seed, budget, phase, "T", result)
    for shape, env in environments.items():
        _reset(env, case_spec.scene_seed, budget, phase, shape, result)
    reference = environments["T"].capture_state()
    if (reference.control_step != 0 or tuple(environments["T"].block.center_of_gravity) != T_REFERENCE_COG):
        raise ValueError("Matched evaluation requires an original T initial reset, never a contact snapshot")
    template = environments["L"].capture_state()
    if template.control_step != 0:
        raise ValueError("Matched L template must be an initial reset")
    states = {"T": reference, "L": matching_shape_state(reference, template)}
    issues = {shape: reset_issues(official_shape(shape), state) for shape, state in states.items()}
    result["initial_scene_issues"] = issues
    if any(issues.values()):
        raise ValueError(f"Fixed matched scene rejected without retry: {issues}")
    # Only the target fresh environment is restored, before any control. The
    # other reset exists solely to validate the same common T/L initial scene.
    return environments[case_spec.shape].restore_state(states[case_spec.shape])


def _evaluate_case(official, actor, environments, spec, result, *, protocol, mode, budget,
                   phase, max_controls, metrics_fn):
    observation = _initial_observation(environments, spec, protocol, budget, phase, result)
    env = environments[spec.shape]
    if hasattr(env, "capture_state"):
        result["initial_state"] = asdict(env.capture_state())
    else:
        result["initial_state"] = physical_state(env)
    initial = _metrics(env, metrics_fn)
    result.update(initial_metrics=initial, final_metrics=dict(initial), peak_metrics=dict(initial))
    history = [observation, observation]
    generator = torch.Generator(device=actor.network.device).manual_seed(spec.policy_seed)
    while result["counts"]["control_steps"] < max_controls:
        budget.check_time()
        if budget.remaining("control", phase=phase) < 1:
            raise RuntimeError("Development control budget exhausted before the episode completed")
        reservation = _reserve(budget, "plan", phase, spec.shape)
        result["counts"]["planning_attempts"] += 1
        _synchronize(actor.network.device)
        start = time.perf_counter()
        result["interrupted_operation"] = "plan"
        try:
            sample = actor.sample(official.observation_tensors(history), generator=generator, mode=mode,
                                  actor_version=f"evaluation:{result['method']}")
            _synchronize(actor.network.device)
        finally:
            result["latency"]["planning_seconds"].append(time.perf_counter() - start)
        result["interrupted_operation"] = None
        actions = sample.actions.detach().cpu().numpy()
        if actions.shape != (1, 8, 2) or not np.isfinite(actions).all():
            raise ValueError("Evaluation requires a complete finite official eight-action prefix")
        result["counts"]["planning_calls"] += 1
        result["counts"]["denoising_steps"] += 100
        budget.mark_completed(reservation)
        for action in actions[0]:
            if result["counts"]["control_steps"] >= max_controls:
                break
            reservation = _reserve(budget, "control", phase, spec.shape)
            result["counts"]["control_attempts"] += 1
            start = time.perf_counter()
            result["interrupted_operation"] = "control"
            try:
                transition = env.step(action.copy())
            finally:
                result["latency"]["control_seconds"] += time.perf_counter() - start
            result["interrupted_operation"] = None
            result["counts"]["control_steps"] += 1
            sim_hz, control_hz = env.sim_hz, env.control_hz
            if (type(sim_hz) is not int or type(control_hz) is not int or control_hz < 1
                    or sim_hz < control_hz or sim_hz % control_hz):
                raise ValueError("Environment must expose integral physical/control frequencies")
            result["counts"]["control_physics_substeps"] += sim_hz // control_hz
            budget.mark_completed(reservation)
            if not isinstance(transition, tuple) or len(transition) != 4:
                raise ValueError("Expected the original official four-item step result")
            observation, reward, done, info = transition
            if not isinstance(info, dict) or not math.isfinite(float(reward)):
                raise ValueError("Official evaluation needs finite reward and an info mapping")
            metrics = _metrics(env, metrics_fn)
            result["final_metrics"] = metrics
            result["peak_metrics"] = {key: max(result["peak_metrics"][key], value) for key, value in metrics.items()}
            success = metrics["coverage"] > .95
            result["steps"].append({"action": action.tolist(), "upstream_reward": float(reward),
                                    "upstream_done": bool(done), **metrics})
            history = [history[-1], observation]
            if success:
                result.update(status="completed", success=True, success_step=result["counts"]["control_steps"],
                              stop_reason="success")
                return
            if info.get("TimeLimit.truncated", False):
                result.update(status="completed", success=False, stop_reason="episode_timeout",
                              failure_reason="success_not_reached_before_timeout")
                return
            if done:
                raise ValueError("Upstream terminated below strict coverage > 0.95")
    result.update(status="completed", success=False, stop_reason="control_cap",
                  failure_reason="success_not_reached_within_control_cap")


def evaluate_cases(official, actor, env_factory, cases, *, method, protocol, budget,
                   max_controls=300, metrics_fn=physical_metrics, delta_applied=False,
                   checkpoint_identity=None, phase="development", on_case_start=None, on_case=None):
    """Evaluate fixed cases on fresh environments and local per-case generators.

    ``env_factory(shape)`` must create an independent, unreset environment. No
    training environment is accepted as an argument. Matched cases reset both
    T/L, check the common initial scene once, and never substitute another seed.
    A case abort stops the group; later cases remain explicitly unstarted.
    Incomplete groups have success_fraction=None, with completed-only telemetry
    separately labelled. The root CLI owns file storage and separate processes.
    """
    identity, mode = _check_method(actor, method, delta_applied, checkpoint_identity)
    cases = tuple(cases)
    if (not cases or any(not isinstance(case, EvaluationCase) for case in cases)
            or len(set(cases)) != len(cases)):
        raise ValueError("Provide unique, explicitly ordered EvaluationCase records")
    if protocol not in (ORIGINAL_T_PROTOCOL, MATCHED_TL_PROTOCOL):
        raise ValueError("Original T and common matched T/L protocols must be named separately")
    if protocol == ORIGINAL_T_PROTOCOL and any(case.shape != "T" for case in cases):
        raise ValueError("Original unrestricted T protocol accepts T only")
    if phase != "development" or type(max_controls) is not int or not 1 <= max_controls <= 300:
        raise ValueError("Evaluation uses its development budget and a per-case cap in [1,300]")
    _runtime_dependencies()
    from .policy_training import state_fingerprint
    report = {"schema_version": SCHEMA_VERSION, "kind": "tdyssey.policy_development_evaluation",
              "method": method, "sampler_mode": mode, "protocol": protocol,
              "checkpoint_identity": identity, "delta_applied": delta_applied,
              "max_controls": max_controls, "phase": phase,
              "training": False, "optimizer_steps": 0, "legacy_test": True,
              "success_criterion": "coverage > 0.95", "cases": [], "counts": _counts(),
              "success_observation": "after_physical_control",
              "matched_initial_state_only": protocol == MATCHED_TL_PROTOCOL,
              "mid_contact_restore_validated": False}
    created, stopped = [], False
    with _isolated_state(official, actor):
        before = state_fingerprint(actor.network)
        for spec in cases:
            result = {**asdict(spec), "method": method, "status": "not_started", "success": None,
                      "success_step": None, "failure_reason": None, "abort_reason": None,
                      "interrupted_operation": None,
                      "stop_reason": "prior_case_aborted" if stopped else None,
                      "initial_state": None, "initial_metrics": None, "final_metrics": None, "peak_metrics": None,
                      "counts": _counts(), "steps": [],
                      "latency": {"reset_seconds": 0., "planning_seconds": [], "control_seconds": 0.}}
            report["cases"].append(result)
            if stopped:
                if on_case is not None:
                    on_case(deepcopy(result))
                continue
            if on_case_start is not None:
                on_case_start({**asdict(spec), "method": method, "status": "started"})
            start, environments = time.perf_counter(), {}
            try:
                budget.check_time()
                for shape in (("T",) if protocol == ORIGINAL_T_PROTOCOL else ("T", "L")):
                    env = env_factory(shape)
                    if any(env is previous for previous in created):
                        raise ValueError("Environment factory reused an instance; evaluation requires fresh episodes")
                    created.append(env)
                    environments[shape] = env
                    if getattr(env, "project_shape", shape) != shape:
                        raise ValueError("Environment shape differs from the requested case")
                    if getattr(env, "legacy", None) is not True:
                        raise ValueError("Evaluation requires the locked official legacy=True reset convention")
                _evaluate_case(official, actor, environments, spec, result, protocol=protocol, mode=mode,
                               budget=budget, phase=phase, max_controls=max_controls, metrics_fn=metrics_fn)
            except Exception as exc:
                result.update(status="aborted", success=None, abort_reason=f"{type(exc).__name__}: {exc}",
                              stop_reason="aborted")
                stopped = True
            finally:
                for env in reversed(tuple(environments.values())):
                    try:
                        env.close()
                    except Exception as exc:
                        result.update(status="aborted", success=None, stop_reason="cleanup_error",
                                      abort_reason=f"Environment close failed: {type(exc).__name__}: {exc}")
                        stopped = True
                result["latency"]["total_seconds"] = time.perf_counter() - start
                for name, value in result["counts"].items():
                    report["counts"][name] += value
            if on_case is not None:
                on_case(deepcopy(result))
        after = state_fingerprint(actor.network)
        if before != after:
            raise ValueError("Evaluation changed actor parameters or normalizer/buffers")
        report.update(state_fingerprint_before=before, state_fingerprint_after=after)
    completed = [case for case in report["cases"] if case["status"] == "completed"]
    success_count = sum(case["success"] for case in completed)
    report.update(status="completed" if len(completed) == len(cases) else "incomplete",
                  requested_case_count=len(cases), completed_case_count=len(completed),
                  aborted_case_count=sum(case["status"] == "aborted" for case in report["cases"]),
                  unstarted_case_count=sum(case["status"] == "not_started" for case in report["cases"]),
                  success_count=success_count,
                  success_fraction=success_count / len(cases) if len(completed) == len(cases) else None,
                  completed_case_success_fraction=success_count / len(completed) if completed else None)
    report["per_scene"] = _group_summary(report["cases"], ("shape", "scene_seed"))
    report["per_shape"] = _group_summary(report["cases"], ("shape",))
    return report


def _group_summary(cases, key_names):
    grouped = {}
    for case in cases:
        grouped.setdefault(tuple(case[key] for key in key_names), []).append(case)
    rows = []
    for key, selected in grouped.items():
        completed = [case for case in selected if case["status"] == "completed"]
        successes = sum(case["success"] for case in completed)
        row = {name: value for name, value in zip(key_names, key)}
        row.update(requested_case_count=len(selected), completed_case_count=len(completed),
                   success_count=successes,
                   success_fraction=successes / len(selected) if len(completed) == len(selected) else None,
                   completed_case_success_fraction=successes / len(completed) if completed else None)
        row["completed_case_mean_metrics"] = {
            f"{scope}_{metric}": math.fsum(case[scope][metric] for case in completed) / len(completed)
            if completed else None
            for scope in ("final_metrics", "peak_metrics") for metric in ("coverage", "iou")}
        rows.append(row)
    return rows


def paired_comparison(reports):
    """Compare complete same-protocol groups offline, preserving case order.

    A comparison_label (or group_id) identifies a checkpoint stage, while method
    identifies its base/sampler/delta treatment. Multiple delta stages may share
    a method, but relabelling the same method/checkpoint is not another group.
    """
    reports = tuple(reports)
    labels = [report.get("comparison_label", report.get("group_id", report.get("method"))) for report in reports]
    if (len(reports) < 2 or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Provide at least two distinct evaluation comparison labels")
    reference = reports[0]
    if (reference.get("kind") != "tdyssey.policy_development_evaluation"
            or reference.get("protocol") not in (ORIGINAL_T_PROTOCOL, MATCHED_TL_PROTOCOL)):
        raise ValueError("Unknown paired evaluation schema/protocol")
    keys = [(case["shape"], case["scene_seed"], case["policy_seed"]) for case in reference["cases"]]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Paired reports need unique nonempty cases")
    treatment_identities = set()
    for report in reports:
        identity = _identity(report.get("checkpoint_identity"))
        expected_delta = report.get("method") == "delta_training"
        if (report.get("method") not in METHODS or report.get("delta_applied") != expected_delta
                or (identity["delta_sha256"] is not None) != expected_delta
                or report.get("sampler_mode") != ("baseline" if report["method"] == "base_original" else "training")):
            raise ValueError("Paired report method/sampler/provenance mismatch")
        treatment_identity = (report["method"], identity["base_sha256"], identity["delta_sha256"])
        if treatment_identity in treatment_identities:
            raise ValueError("The same method/checkpoint identity cannot be duplicated under another label")
        treatment_identities.add(treatment_identity)
        if (report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != reference.get("kind")
                or report.get("protocol") != reference.get("protocol")
                or report.get("max_controls") != reference.get("max_controls")
                or report.get("phase") != "development"
                or report.get("success_observation") != "after_physical_control"
                or identity["base_sha256"] != reference["checkpoint_identity"]["base_sha256"]):
            raise ValueError("Paired reports need the same schema, protocol, cap and base checkpoint")
        if (report.get("status") != "completed" or report.get("completed_case_count") != len(keys)
                or any(case.get("status") != "completed" or type(case.get("success")) is not bool
                       for case in report["cases"])):
            raise ValueError("Paired comparison refuses incomplete/aborted cases")
        if [(case["shape"], case["scene_seed"], case["policy_seed"]) for case in report["cases"]] != keys:
            raise ValueError("Paired case identities/order differ; missing cases cannot be substituted")
    comparisons = []
    for (left_label, left), (right_label, right) in combinations(zip(labels, reports), 2):
        differences = []
        for previous, current in zip(left["cases"], right["cases"]):
            row = {key: previous[key] for key in ("shape", "scene_seed", "policy_seed")}
            row["success_difference"] = int(current["success"]) - int(previous["success"])
            for scope in ("final_metrics", "peak_metrics"):
                for metric in ("coverage", "iou"):
                    row[f"{scope}_{metric}_difference"] = current[scope][metric] - previous[scope][metric]
            row["success_step_difference"] = (current["success_step"] - previous["success_step"]
                                               if current["success"] and previous["success"] else None)
            row["control_steps_difference"] = current["counts"]["control_steps"] - previous["counts"]["control_steps"]
            row["planning_seconds_difference"] = (math.fsum(current["latency"]["planning_seconds"])
                                                    - math.fsum(previous["latency"]["planning_seconds"]))
            row["total_seconds_difference"] = current["latency"]["total_seconds"] - previous["latency"]["total_seconds"]
            differences.append(row)
        mean_keys = ("success_difference", "final_metrics_coverage_difference", "final_metrics_iou_difference",
                     "peak_metrics_coverage_difference", "peak_metrics_iou_difference", "control_steps_difference")
        mean_keys += ("planning_seconds_difference", "total_seconds_difference")
        comparisons.append({"reference_method": left["method"], "candidate_method": right["method"],
                            "reference_label": left_label, "candidate_label": right_label,
                            "reference_checkpoint_identity": deepcopy(left["checkpoint_identity"]),
                            "candidate_checkpoint_identity": deepcopy(right["checkpoint_identity"]),
                            "cases": differences,
                            "mean_differences": {key: math.fsum(row[key] for row in differences) / len(keys)
                                                 for key in mean_keys}})
    return {"schema_version": SCHEMA_VERSION, "kind": "tdyssey.paired_policy_development_evaluation",
            "protocol": reference["protocol"], "max_controls": reference["max_controls"],
            "case_count": len(keys), "base_sha256": reference["checkpoint_identity"]["base_sha256"],
            "groups": [{"comparison_label": label, "method": report["method"],
                        "checkpoint_identity": deepcopy(report["checkpoint_identity"])}
                       for label, report in zip(labels, reports)],
            "comparisons": comparisons}
