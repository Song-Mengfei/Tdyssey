"""Independent image measurements and conservative predicted stopping; stdlib.

A development absolute-error interval is an empirical engineering gate, not a
probability, confidence interval, or guaranteed bound for generated images. Fit
residuals are never converted to probabilities. Producers remain responsible for
declaring whether their uncertainty interval applies to the image being measured.
No pusher, future simulator truth, decoder, or model is read by this module.
"""

from copy import deepcopy
import math


SUCCESS_THRESHOLD = .95
DEVELOPMENT_BOUND = "development_absolute_error_bound"
INTERVAL_KINDS = frozenset((DEVELOPMENT_BOUND, "validated_absolute_error_bound", "deterministic_absolute_error_bound"))
STATUSES = frozenset(("success", "non_success", "unknown"))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def wrap_radians(angle):
    """Wrap a finite radian angle to [-pi, pi], preserving rotation direction."""
    if not _finite(angle):
        raise ValueError("Angle must be a finite radian number")
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_error_degrees(predicted, reference):
    """Unsigned wrapped error; inputs are radians, output is degrees."""
    if not _finite(predicted) or not _finite(reference):
        raise ValueError("Both angles must be finite radian numbers")
    return abs(math.degrees(wrap_radians(predicted-reference)))


def _interval(value, estimate):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_finite(x) for x in value):
        return None
    lower, upper = map(float, value)
    if not 0 <= lower <= estimate <= upper <= 1:
        return None
    return [lower, upper]


def coverage_measurement(estimate, *, half_width=None, interval=None, uncertainty_kind="none", valid=True, reason=None):
    """Keep point-estimate validity separate from interval/termination eligibility.

    A valid point estimate does not require a visible pusher or usable error
    interval. An absent, malformed, or unrecognized interval keeps termination
    unknown. Half-width intervals clip only the physical coverage bounds [0,1].
    ``development_absolute_error_bound`` is a disclosed development assumption,
    with no probabilistic interpretation or guarantee of generated-image domain.
    """
    if type(valid) is not bool:
        raise ValueError("Coverage validity must be an explicit bool")
    if half_width is not None and interval is not None:
        raise ValueError("Declare half_width or interval, not both")
    if not isinstance(uncertainty_kind, str):
        raise ValueError("Uncertainty kind must be a string")
    point_valid = bool(valid and _finite(estimate) and 0 <= estimate <= 1)
    if not point_valid:
        return {"valid": False, "estimate": None, "interval": None, "uncertainty_kind": uncertainty_kind,
                "uncertainty_valid": False, "reason": reason or "invalid_or_missing_coverage_estimate",
                "uncertainty_reason": "invalid_point_estimate", "is_probability": False}
    estimate = float(estimate)
    uncertainty_reason = "uncertainty_interval_not_declared"
    if half_width is not None:
        if _finite(half_width) and 0 <= half_width <= 1:
            interval = [max(0., estimate-half_width), min(1., estimate+half_width)]
        else:
            uncertainty_reason = "invalid_absolute_error_half_width"
    accepted = _interval(interval, estimate)
    if interval is not None and accepted is None:
        uncertainty_reason = "invalid_or_inconsistent_uncertainty_interval"
    recognized = uncertainty_kind in INTERVAL_KINDS
    if accepted is not None:
        uncertainty_reason = "declared_absolute_error_interval" if recognized else "unrecognized_uncertainty_kind"
    return {"valid": True, "estimate": estimate, "interval": accepted, "uncertainty_kind": uncertainty_kind,
            "uncertainty_valid": bool(accepted is not None and recognized), "reason": reason or "ok",
            "uncertainty_reason": uncertainty_reason, "is_probability": False,
            "development_bound_is_confidence_interval": False}


def termination_from_coverage(coverage):
    """Strict three-state success under a declared, structurally valid interval.

    lower > .95 is success; upper <= .95 is non-success; every overlap, missing
    point, missing bound, or unsupported uncertainty declaration is unknown.
    Development bounds produce conditional engineering decisions, not success
    probabilities and not claims of calibrated confidence on generated images.
    """
    result = {"status": "unknown", "reason": "invalid_or_missing_coverage", "threshold": SUCCESS_THRESHOLD,
              "is_probability": False, "decision_basis": "declared_absolute_error_interval"}
    if not isinstance(coverage, dict) or coverage.get("valid") is not True:
        return result
    estimate = coverage.get("estimate")
    if not _finite(estimate) or not 0 <= estimate <= 1:
        return result
    interval = _interval(coverage.get("interval"), estimate)
    kind = coverage.get("uncertainty_kind")
    result["uncertainty_kind"] = kind
    if interval is None:
        result["reason"] = "missing_or_invalid_uncertainty_interval"
        return result
    if not isinstance(kind, str) or kind not in INTERVAL_KINDS or coverage.get("uncertainty_valid") is False:
        result["reason"] = "uncertainty_not_usable_for_termination"
        return result
    lower, upper = interval
    if lower > SUCCESS_THRESHOLD:
        result.update(status="success", reason="interval_lower_strictly_above_success_threshold")
    elif upper <= SUCCESS_THRESHOLD:
        result.update(status="non_success", reason="interval_upper_at_or_below_success_threshold")
    else:
        result["reason"] = "interval_overlaps_strict_success_threshold"
    result["interval"] = interval
    result["development_assumption"] = kind == DEVELOPMENT_BOUND
    return result


def measurement_frame(*, state, shape, pusher, coverage):
    """Assemble independently valid measurements without a pusher-to-coverage gate."""
    for name, value in (("state", state), ("shape", shape), ("pusher", pusher), ("coverage", coverage)):
        if not isinstance(value, dict) or type(value.get("valid")) is not bool:
            raise ValueError(name + " requires an explicit independent validity flag")
    result = {"state": deepcopy(state), "shape": deepcopy(shape), "pusher": deepcopy(pusher),
              "coverage": deepcopy(coverage)}
    result["termination"] = termination_from_coverage(coverage)
    return result


def _frame_termination(frame):
    derived = termination_from_coverage(frame.get("coverage"))
    declared = frame.get("termination")
    if declared is None:
        return derived
    if not isinstance(declared, dict) or declared.get("status") not in STATUSES:
        return {**derived, "status": "unknown", "reason": "invalid_declared_termination"}
    if declared["status"] == "unknown":
        # A producer can abstain for stronger domain checks than interval syntax.
        return {**derived, "status": "unknown", "reason": declared.get("reason", "producer_abstention")}
    if declared["status"] != derived["status"]:
        return {**derived, "status": "unknown", "reason": "declared_termination_not_supported_by_coverage_interval"}
    return derived


def sequence_endpoint(frames):
    """Separate raw endpoint information from stopping-consistent score eligibility.

    Only predicted intervals decide stopping. An unknown frame before a known
    success or the endpoint makes its prior stopping time ambiguous, so the
    sequence score abstains. Every raw frame remains usable independently. A
    clear success preceded only by clear non-success makes the model absorbing:
    later malformed frames cannot change that already-stopped score/state.
    No physical truth, pusher visibility, or missing-value imputation is used.
    """
    if not isinstance(frames, (list, tuple)) or not frames or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("Sequence requires one or more predicted measurement frames")
    raw = deepcopy(frames[-1])
    raw_coverage = raw.get("coverage") if isinstance(raw.get("coverage"), dict) else {}
    raw_valid = (raw_coverage.get("valid") is True and _finite(raw_coverage.get("estimate"))
                 and 0 <= raw_coverage["estimate"] <= 1)
    unknown, decisions, first_success, stop_index = [], [], None, None
    stopped_state = None
    ambiguous_before_stop = False
    for index, frame in enumerate(frames):
        decision = _frame_termination(frame)
        decisions.append({"index": index, "horizon": index+1, **decision})
        if decision["status"] == "unknown":
            unknown.append({"index": index, "horizon": index+1, "reason": decision["reason"],
                            "after_certain_absorption": stop_index is not None})
            if stop_index is None:
                ambiguous_before_stop = True
        elif decision["status"] == "success":
            if first_success is None:
                first_success = index
            if stop_index is None and not ambiguous_before_stop:
                stop_index, stopped_state = index, deepcopy(frame)
    eligible = stop_index is not None or not ambiguous_before_stop
    score_frame = stopped_state if stopped_state is not None else frames[-1]
    score_coverage = score_frame.get("coverage") if isinstance(score_frame.get("coverage"), dict) else {}
    eligible = bool(eligible and score_coverage.get("valid") is True and _finite(score_coverage.get("estimate")))
    return {"raw_endpoint": raw, "raw_endpoint_coverage_valid": bool(raw_valid),
            "raw_endpoint_coverage": float(raw_coverage["estimate"]) if raw_valid else None,
            "stopped_score": float(score_coverage["estimate"]) if eligible else None,
            "stopped_score_valid": eligible, "stopped_score_validity": "eligible" if eligible else "abstain",
            "stopped_score_interval": deepcopy(score_coverage.get("interval")) if eligible else None,
            "stopped_score_reason": ("absorbed_at_predicted_success" if stop_index is not None else
                "all_predicted_frames_certain_non_success" if eligible else "unknown_termination_before_score_endpoint"),
            "unknown_history": unknown, "unknown_before_stop_or_endpoint": ambiguous_before_stop,
            "per_frame_termination": decisions, "first_interval_success_index": first_success,
            "stop_index": stop_index, "stop_horizon": None if stop_index is None else stop_index+1,
            "absorbing_model_state": stopped_state, "absorbing": stopped_state is not None,
            "uncertainty_is_probability": False, "future_truth_used": False}
