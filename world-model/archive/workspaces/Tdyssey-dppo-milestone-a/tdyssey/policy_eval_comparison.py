"""Stdlib-only paired summary for the authorized 20-case C comparison.

Reports are scalar evidence, not a cryptographic check of model contents. Two
policy seeds from the same scene are correlated; this module reports descriptive
paired differences and makes no independent-sample significance claim.
"""

from copy import deepcopy
import math
from statistics import median


BASE_SHA256 = "37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0"
SCENES = tuple(range(100000, 100005))
POLICY_SEEDS = (5478, 5479)
SHAPES = ("T", "L")
CASE_KEYS = tuple((shape, scene, seed) for shape in SHAPES for scene in SCENES for seed in POLICY_SEEDS)
METRICS = ("final_coverage", "final_iou", "peak_coverage", "peak_iou")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, minimum=None, maximum=None):
    _require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    _require((minimum is None or value >= minimum) and (maximum is None or value <= maximum),
             name + " is outside its permitted range")
    return value


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate(report, method):
    _require(isinstance(report, dict), "Expected an evaluation report")
    expected = {"schema_version": 1, "kind": "tdyssey.policy_development_evaluation",
                "method": method, "sampler_mode": "training", "protocol": "official-t-l-com-v1",
                "max_controls": 300, "phase": "development", "training": False,
                "optimizer_steps": 0, "success_criterion": "coverage > 0.95",
                "success_observation": "after_physical_control", "matched_initial_state_only": True,
                "mid_contact_restore_validated": False}
    for key, value in expected.items():
        _require(type(report.get(key)) is type(value) and report[key] == value, "Evaluation " + key + " differs")
    config = report.get("sampler_config")
    _require(isinstance(config, dict) and config.get("version") == "tdyssey-ddpm-gaussian-v1",
             "Explicit training sampler provenance is required")
    _require(type(config.get("std_floor")) in (int, float) and config["std_floor"] == .01,
             "Both methods require the fixed positive std_floor=0.01")
    identity = report.get("checkpoint_identity")
    _require(isinstance(identity, dict) and set(identity) == {"base_sha256", "delta_sha256"}
             and identity["base_sha256"] == BASE_SHA256, "Pinned base checkpoint identity differs")
    delta = method == "delta_training"
    _require(report.get("delta_applied") is delta
             and (_digest(identity["delta_sha256"]) if delta else identity["delta_sha256"] is None),
             "Base/delta checkpoint identity differs from method")
    before, after = report.get("state_fingerprint_before"), report.get("state_fingerprint_after")
    _require(_digest(before) and before == after, "Evaluation weight/state preservation evidence differs")
    cases = report.get("cases")
    _require(isinstance(cases, list) and all(isinstance(case, dict) for case in cases), "Missing evaluation cases")
    keys = [(case.get("shape"), case.get("scene_seed"), case.get("policy_seed")) for case in cases]
    _require(keys == list(CASE_KEYS), "Fixed ordered cases differ; missing/reordered cases cannot be substituted")
    completed, successes = 0, 0
    statuses = {"completed": 0, "aborted": 0, "not_started": 0}
    for case in cases:
        _require(case.get("method") == method and case.get("status") in statuses, "Invalid case method/status")
        statuses[case["status"]] += 1
        counts = case.get("counts")
        controls = counts.get("control_steps") if isinstance(counts, dict) else None
        _require(type(controls) is int and 0 <= controls <= 300, "Invalid case control count")
        for scope in ("final_metrics", "peak_metrics"):
            metrics = case.get(scope)
            if metrics is not None:
                _require(isinstance(metrics, dict), "Invalid case metrics")
                for metric in ("coverage", "iou"):
                    _number(metrics.get(metric), scope + " " + metric, minimum=0, maximum=1)
        if case["status"] == "completed":
            completed += 1
            _require(type(case.get("success")) is bool and controls > 0
                     and case.get("final_metrics") is not None and case.get("peak_metrics") is not None,
                     "Completed cases require real controls and final/peak measurements")
            for metric in ("coverage", "iou"):
                _require(case["peak_metrics"][metric] + 1e-12 >= case["final_metrics"][metric], "Peak is below final metric")
            success = case["success"]
            _require(success == (case["final_metrics"]["coverage"] > .95), "Strict physical coverage success differs")
            if success:
                _require(type(case.get("success_step")) is int and case["success_step"] == controls,
                         "First success step must be the actual final executed control")
                successes += 1
            else:
                _require(case.get("success_step") is None, "Failure success steps are censored, not imputed")
        else:
            _require(case.get("success") is None and case.get("success_step") is None,
                     "Aborted/unstarted outcomes must remain unknown")
            if case["status"] == "not_started":
                _require(controls == 0 and case.get("final_metrics") is None and case.get("peak_metrics") is None,
                         "Unstarted cases cannot contain executed outcomes")
    _require(report.get("requested_case_count") == 20 and report.get("completed_case_count") == completed
             and report.get("aborted_case_count") == statuses["aborted"]
             and report.get("unstarted_case_count") == statuses["not_started"]
             and report.get("success_count") == successes, "Evaluation aggregate counts disagree with cases")
    complete = completed == 20
    _require(report.get("status") == ("completed" if complete else "incomplete"), "Group completion status differs")
    if complete:
        _require(report.get("success_fraction") == successes / 20, "Full denominator success fraction differs")
    else:
        _require(report.get("success_fraction") is None, "Incomplete groups cannot claim a success fraction")
    label = report.get("comparison_label", report.get("group_id", report.get("method")))
    _require(type(label) is str and bool(label.strip()), "An explicit nonempty method label is required")
    return complete, label


def _summary(values):
    return {"mean": math.fsum(values) / len(values), "median": median(values)} if values else {"mean": None, "median": None}


def _wins(differences, tolerance, *, lower_better=False):
    signed = [-value if lower_better else value for value in differences]
    return {"win": sum(value > tolerance for value in signed),
            "tie": sum(abs(value) <= tolerance for value in signed),
            "loss": sum(value < -tolerance for value in signed)}


def _aggregate(rows, complete, tolerance):
    count = len(rows)
    result = {"n": count, "comparison_available": complete}
    for owner in ("baseline", "delta"):
        observed = [row[owner] for row in rows if row[owner]["status"] == "completed"]
        successes = sum(case["success"] for case in observed)
        result[owner] = {"completed_count": len(observed), "observed_success_count": successes,
                         "success_count": successes if complete else None, "denominator": count,
                         "success_fraction": successes / count if complete else None}
    result["success_difference_pp"] = ((result["delta"]["success_count"] - result["baseline"]["success_count"])
                                        * 100 / count if complete else None)
    result["paired_success"] = (_wins([row["success_difference"] for row in rows], 0) if complete else None)
    result["metrics"] = {}
    for metric in METRICS:
        left = [row["baseline"][metric] for row in rows] if complete else []
        right = [row["delta"][metric] for row in rows] if complete else []
        differences = [row["differences"][metric] for row in rows] if complete else []
        result["metrics"][metric] = {"baseline": _summary(left), "delta": _summary(right),
            "paired_difference": _summary(differences),
            "paired_outcomes": _wins(differences, tolerance) if complete else None}
    steps = [row["success_step_difference"] for row in rows if row["success_step_difference"] is not None]
    result["first_success_steps"] = {"both_success_count": len(steps) if complete else None,
        "censored_count": count - len(steps) if complete else None,
        "paired_difference": _summary(steps),
        "paired_outcomes": _wins(steps, tolerance, lower_better=True) if complete else None,
        "difference_definition": "delta minus baseline; negative means delta reached success earlier"}
    return result


def _case_values(case):
    values = {key: deepcopy(case.get(key)) for key in ("status", "success", "success_step", "failure_reason", "abort_reason", "stop_reason")}
    values["control_steps"] = case["counts"]["control_steps"]
    for metric in METRICS:
        scope, name = metric.split("_", 1)
        source = case.get(scope + "_metrics")
        values[metric] = source[name] if source is not None else None
    return values


def compare_evaluation_reports(baseline_report, delta_report, *, tolerance=1e-9):
    """Compare base_training then delta_training on the exact authorized matrix.

    Missing identities/order/provenance are errors. An intact matrix containing
    aborted or unstarted cases is honestly incomplete: all comparison aggregates
    are withheld, including shapes whose observed subset happens to be complete.
    """
    _number(tolerance, "tolerance", minimum=0, maximum=1e-3)
    base_complete, base_label = _validate(baseline_report, "base_training")
    delta_complete, delta_label = _validate(delta_report, "delta_training")
    _require(base_label != delta_label, "Comparison labels must be unique")
    _require(baseline_report["sampler_config"] == delta_report["sampler_config"], "Training sampler configurations differ")
    complete = base_complete and delta_complete
    cases = []
    for key, left, right in zip(CASE_KEYS, baseline_report["cases"], delta_report["cases"]):
        row = dict(zip(("shape", "scene_seed", "policy_seed"), key))
        row.update(baseline=_case_values(left), delta=_case_values(right))
        row["success_difference"] = int(right["success"]) - int(left["success"]) if complete else None
        row["differences"] = {metric: row["delta"][metric] - row["baseline"][metric] if complete else None for metric in METRICS}
        both = complete and left["success"] and right["success"]
        row["success_step_difference"] = right["success_step"] - left["success_step"] if both else None
        row["success_step_comparison"] = "both_success" if both else ("censored_either_not_success" if complete else "group_incomplete")
        cases.append(row)
    return {"schema_version": 1, "kind": "tdyssey.C.paired_training_sampler_evaluation",
        "status": "completed" if complete else "incomplete", "training_steps": 0,
        "protocol": baseline_report["protocol"], "sampler_config": deepcopy(baseline_report["sampler_config"]),
        "baseline_label": base_label, "delta_label": delta_label,
        "baseline_checkpoint_identity": deepcopy(baseline_report["checkpoint_identity"]),
        "delta_checkpoint_identity": deepcopy(delta_report["checkpoint_identity"]),
        "overall": _aggregate(cases, complete, tolerance),
        "per_shape": {shape: _aggregate([row for row in cases if row["shape"] == shape], complete, tolerance) for shape in SHAPES},
        "cases": cases,
        "scenes": [{"scene_seed": scene, "policy_seeds": list(POLICY_SEEDS),
                    "per_shape": {shape: _aggregate([row for row in cases if row["shape"] == shape and row["scene_seed"] == scene], complete, tolerance) for shape in SHAPES}}
                   for scene in SCENES],
        "limits": {"case_count_per_method": 20, "case_count_per_shape": 10, "scene_count": 5,
            "policy_seeds_per_scene": 2, "max_controls_per_case": 300, "metric_tolerance": tolerance,
            "difference_direction": "delta minus baseline", "metric_win_direction": "higher is better",
            "incomplete_policy": "No imputation; all comparison aggregates withheld until both groups complete",
            "success_steps": "Conditional on both methods succeeding; failures and aborts are censored, never assigned step 300",
            "statistical_unit": "Five scene blocks; two policy seeds within a scene are correlated, not independent scenes",
            "inference": "Descriptive fixed-scene development comparison only; no independence-based confidence interval or untouched-test claim"}}
