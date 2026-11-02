"""Fixed-denominator compatibility reports from saved numeric records only.

This module never reads images, fits measurements, samples, or imports a model.
Truth is used only for error labels and physical reference outcomes. It never
changes measurement validity, predicted stopping, or candidate selection.
Development intervals remain assumptions, not validated confidence intervals.
"""

from collections import Counter
from copy import deepcopy
import math

from .wm_measurement_contract import angle_error_degrees, sequence_endpoint


ARMS = ("A", "B")
ANCHORS = tuple("L%02d" % i for i in range(12)) + tuple("T%02d" % i for i in range(12, 16))
CANDIDATES = ("hold", "toward", "side", "away")
SEEDS = (5478, 5479, 5480)
HORIZONS = (1, 2, 3, 4)
PRIMARY_HORIZONS = (1, 2, 4)
TIE_TOLERANCE = 1e-6
PAIR_SEPARATION = .05


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantile(values, fraction=.9):
    """The linear percentile convention, without a numeric-library dependency."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower] + (values[upper]-values[lower]) * (position-lower)


def _stats(values):
    return {"count": len(values), "mean": _mean(values), "p90": _quantile(values),
            "max": max(values) if values else None}


def _point(value):
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(x) for x in value):
        return [float(x) for x in value]
    return None


def _pose(value):
    if not isinstance(value, dict):
        return None
    position = _point(value.get("position"))
    return {"position": position, "angle": float(value["angle"])} if position is not None and _finite(value.get("angle")) else None


def _coverage(value):
    return float(value) if _finite(value) and 0 <= value <= 1 else None


def _distance(first, second):
    return math.hypot(first[0]-second[0], first[1]-second[1])


def _measurement(row):
    measurement = row.get("measurement")
    measurement = measurement if isinstance(measurement, dict) else {}
    parts = {name: measurement.get(name) if isinstance(measurement.get(name), dict) else {}
             for name in ("state", "shape", "pusher", "coverage")}
    pose = _pose(parts["state"].get("pose"))
    pusher = _point(parts["pusher"].get("position"))
    estimate = _coverage(parts["coverage"].get("estimate"))
    completed = row.get("status") == "completed"
    flags = {"object": bool(completed and parts["state"].get("valid") is True and pose is not None),
             "shape": bool(completed and parts["shape"].get("valid") is True),
             "pusher": bool(completed and parts["pusher"].get("valid") is True and pusher is not None)}
    flags["object_shape"] = flags["object"] and flags["shape"]
    flags["coverage"] = bool(flags["object_shape"] and parts["coverage"].get("valid") is True and estimate is not None)
    # Preserve the producer's diagnostics; only the separate scoring copy is
    # invalidated when a local prerequisite or completion status is absent.
    scoring = deepcopy(measurement)
    if not flags["coverage"]:
        scoring["coverage"] = {**deepcopy(parts["coverage"]), "valid": False,
                               "reason": "ineligible_object_shape_coverage_or_status"}
    decision = sequence_endpoint([scoring])["per_frame_termination"][0]
    return {"measurement": measurement, "scoring": scoring, "flags": flags,
            "pose": pose, "pusher": pusher, "estimate": estimate, "termination": decision}


def _errors(row, view):
    truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
    pose, pusher, coverage = _pose(truth.get("pose")), _point(truth.get("pusher")), _coverage(truth.get("coverage"))
    return {"position": _distance(view["pose"]["position"], pose["position"]) if view["flags"]["object"] and pose else None,
            "angle": angle_error_degrees(view["pose"]["angle"], pose["angle"]) if view["flags"]["object"] and pose else None,
            "pusher": _distance(view["pusher"], pusher) if view["flags"]["pusher"] and pusher else None,
            "coverage": abs(view["estimate"]-coverage) if view["flags"]["coverage"] and coverage is not None else None,
            "truth_coverage": coverage,
            "truth_complete": pose is not None and pusher is not None and coverage is not None}


def _confusion(records, interval):
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn", "unknown", "truth_unknown")}
    for record in records:
        actual = record["truth_coverage"]
        if actual is None:
            counts["truth_unknown"] += 1
            continue
        status = record["termination"] if interval else (
            "success" if record["coverage_valid"] and record["estimate"] > .95 else
            "non_success" if record["coverage_valid"] else "unknown")
        if status == "unknown":
            counts["unknown"] += 1
        else:
            counts[("t" if (status == "success") == (actual > .95) else "f") + ("p" if status == "success" else "n")] += 1
    positives = counts["tp"]+counts["fp"]
    return {**counts, "total": len(records), "predicted_positives": positives,
            "precision": counts["tp"]/positives if positives else None,
            "unknown_is_success": False, "interval_claim": "conditional_development_assumption" if interval else "point_diagnostic_only"}


def _real_plan():
    result = {}
    for shape in ("L", "T"):
        for i in range(12):
            result["cal-%s-%02d" % (shape, i)] = {"shape": shape, "source": "calibration", "known_missing": False}
    for anchor in ANCHORS:
        for candidate in CANDIDATES:
            for i in range(4):
                result["%s-%s-history-%d" % (anchor, candidate, i)] = {
                    "shape": anchor[0], "source": "history", "known_missing": anchor == "L01" and candidate == "hold"}
            for horizon in HORIZONS:
                result["%s-%s-future-H%d" % (anchor, candidate, horizon)] = {
                    "shape": anchor[0], "source": "physical_future", "known_missing": anchor == "L01" and candidate == "hold"}
    return result


def measurement_screen(rows):
    """M: exact 536 planned/528 available source identities per arm.

    Known L01/hold's eight missing RGB are retained but do not fail M. Every
    other source must be completed and locally valid. P90 gates concern object
    position/angle/coverage; pusher errors are reported, with validity required.
    Forecast rows are explicitly excluded and cannot replace a real source.
    """
    plan, indexed, unexpected = _real_plan(), {}, []
    rows = list(rows)
    for row in rows:
        if row.get("mode") != "REAL":
            continue
        arm, source_key = row.get("arm"), row.get("source_key")
        if arm not in ARMS or source_key not in plan:
            unexpected.append({"key": row.get("key"), "arm": arm, "source_key": source_key})
            continue
        index = (arm, source_key)
        if index in indexed:
            raise ValueError("Duplicate REAL arm/source_key: %r" % (index,))
        indexed[index] = row
    records, cells = [], []
    for arm in ARMS:
        for source_key, planned in plan.items():
            row = indexed.get((arm, source_key))
            row = row if row is not None else {"status": "missing", "measurement": {}, "truth": {}}
            view = _measurement(row)
            errors = _errors(row, view)
            identity_ok = row.get("shape") == planned["shape"] and row.get("source") == planned["source"]
            records.append({"key": row.get("key"), "source_key": source_key, "arm": arm, **planned,
                            "present": (arm, source_key) in indexed, "status": row.get("status"),
                            "identity_ok": identity_ok, "flags": view["flags"], "errors": errors,
                            "coverage_valid": view["flags"]["coverage"], "estimate": view["estimate"],
                            "truth_coverage": errors["truth_coverage"], "termination": view["termination"]["status"],
                            "reasons": {name: view["measurement"].get(name, {}).get("reason")
                                        if isinstance(view["measurement"].get(name), dict) else None
                                        for name in ("state", "shape", "pusher", "coverage")}})
        for shape in ("L", "T"):
            selected = [r for r in records if r["arm"] == arm and r["shape"] == shape]
            available = [r for r in selected if not r["known_missing"]]
            error_stats = {name: _stats([r["errors"][name] for r in available if r["errors"][name] is not None])
                           for name in ("position", "angle", "coverage", "pusher")}
            validity = {name: sum(r["flags"][name] for r in available) for name in ("object", "shape", "object_shape", "pusher", "coverage")}
            interval, nominal = _confusion(available, True), _confusion(available, False)
            checks = {"exact_available_source_set": all(r["present"] and r["identity_ok"] and r["status"] == "completed" for r in available),
                      "known_missing_sources_not_replaced": all(r["status"] != "completed" for r in selected if r["known_missing"]),
                      "all_local_measurements_valid": all(count == len(available) for count in validity.values()),
                      "all_truth_labels_available": all(r["errors"]["truth_complete"] for r in available),
                      "position_p90_le_2": error_stats["position"]["count"] == len(available) and error_stats["position"]["p90"] <= 2,
                      "angle_p90_le_2_degrees": error_stats["angle"]["count"] == len(available) and error_stats["angle"]["p90"] <= 2,
                      "coverage_error_p90_le_003": error_stats["coverage"]["count"] == len(available) and error_stats["coverage"]["p90"] <= .03+1e-12,
                      "interval_false_positives_zero": interval["fp"] == 0}
            cells.append({"arm": arm, "shape": shape, "planned": len(selected), "available": len(available),
                          "known_missing": len(selected)-len(available),
                          "status_counts": dict(Counter(r["status"] for r in selected)), "valid_counts": validity,
                          "errors": error_stats, "interval_confusion": interval, "nominal_confusion": nominal,
                          "checks": checks, "passed": all(checks.values()),
                          "losses": [{"source_key": r["source_key"], "status": r["status"], "flags": r["flags"], "reasons": r["reasons"]}
                                     for r in available if not all(r["flags"].values()) or not r["identity_ok"]]})
    paired_positives = []
    by_id = {(r["arm"], r["source_key"]): r for r in records}
    for source_key in plan:
        a, b = by_id[("A", source_key)], by_id[("B", source_key)]
        if b["termination"] == "success" and a["termination"] != "success":
            paired_positives.append({"source_key": source_key, "A_status": a["termination"], "B_status": b["termination"],
                                     "truth_coverage": b["truth_coverage"]})
    return {"stage": "M", "passed": not unexpected and all(c["passed"] for c in cells),
            "planned_per_arm": 536, "available_per_arm": 528, "known_missing_per_arm": 8,
            "planned_total": 1072, "available_total": 1056, "known_missing_total": 16,
            "ignored_forecast_rows": sum(r.get("mode") != "REAL" for r in rows),
            "unexpected_sources": unexpected, "by_arm_shape": cells, "rows": records,
            "B_only_interval_positives": paired_positives,
            "interval_precision_validated": False, "intervals_are_development_assumptions": True}


def _ar_index(rows):
    indexed = {}
    for row in rows:
        if row.get("mode") != "AR":
            raise ValueError("AR-only API rejects TF/REAL or undeclared modes")
        key = (row.get("arm"), row.get("anchor"), row.get("candidate"), row.get("seed"), row.get("horizon"))
        arm, anchor, candidate, seed, horizon = key
        if (arm not in ARMS or anchor not in ANCHORS or candidate not in CANDIDATES
                or type(seed) is not int or seed not in SEEDS or type(horizon) is not int or horizon not in HORIZONS
                or row.get("shape") != anchor[0] or row.get("status") not in ("completed", "missing", "not_attempted")):
            raise ValueError("Unexpected fixed AR plan identity or status: %r" % (key,))
        if key in indexed:
            raise ValueError("Duplicate AR arm/anchor/candidate/seed/horizon: %r" % (key,))
        indexed[key] = row
    return indexed


def _planned_row(indexed, arm, anchor, candidate, seed, horizon):
    key = (arm, anchor, candidate, seed, horizon)
    if key in indexed:
        return indexed[key]
    return {"key": "%s/%s/%s/%d/AR/H%d" % key,
            "arm": arm, "anchor": anchor, "candidate": candidate, "seed": seed, "horizon": horizon,
            "shape": anchor[0], "mode": "AR", "status": "missing", "reason": "row_absent_from_input",
            "measurement": {}, "truth": {}}


def t_stage_gate(rows):
    """B-AR T H1/H2/H4 screen, including all 48 rows in every cell.

    Contact-stratum regression gates joint object/full-shape validity only;
    pusher regression is disclosed, with its separate >=46/48 cell gate.
    TF records are counted as excluded and can never rescue the AR screen.
    """
    rows = list(rows)
    indexed = _ar_index([r for r in rows if r.get("mode") == "AR"])
    cells, strata, invalid_labels = [], [], 0
    branch_contacts = {}
    for row in indexed.values():
        if row["anchor"][0] == "T":
            label = row.get("contact_by_h4")
            if type(label) is bool:
                branch_contacts.setdefault((row["anchor"], row["candidate"]), set()).add(label)
    conflicting_contact_branches = sorted("%s/%s" % key for key, labels in branch_contacts.items() if len(labels) > 1)
    for horizon in PRIMARY_HORIZONS:
        paired = []
        for anchor in ANCHORS:
            if anchor[0] != "T":
                continue
            for candidate in CANDIDATES:
                for seed in SEEDS:
                    a = _planned_row(indexed, "A", anchor, candidate, seed, horizon)
                    b = _planned_row(indexed, "B", anchor, candidate, seed, horizon)
                    labels_ok = (type(a.get("contact_by_h4")) is bool and type(b.get("contact_by_h4")) is bool
                                 and a["contact_by_h4"] == b["contact_by_h4"]
                                 and len(branch_contacts.get((anchor, candidate), ())) == 1)
                    invalid_labels += not labels_ok
                    paired.append((a, b, _measurement(a), _measurement(b), labels_ok))
        for arm, offset in (("A", 0), ("B", 1)):
            arm_rows = [pair[offset] for pair in paired]
            views = [pair[offset+2] for pair in paired]
            cells.append({"arm": arm, "horizon": horizon, "planned": 48, "available": 48,
                          "completed": sum(r["status"] == "completed" for r in arm_rows),
                          "object_shape_valid": sum(v["flags"]["object_shape"] for v in views),
                          "pusher_valid": sum(v["flags"]["pusher"] for v in views),
                          "shape_required": 44, "pusher_required": 46,
                          "passed": all(r["status"] == "completed" for r in arm_rows)
                                    and sum(v["flags"]["object_shape"] for v in views) >= 44
                                    and sum(v["flags"]["pusher"] for v in views) >= 46})
        for contact in (False, True):
            selected = [pair for pair in paired if pair[4] and pair[0]["contact_by_h4"] is contact]
            n = len(selected)
            counts = {arm: {flag: sum(pair[offset]["flags"][flag] for pair in selected)
                            for flag in ("object_shape", "pusher")}
                      for arm, offset in (("A", 2), ("B", 3))}
            regression = {flag: (counts["B"][flag]-counts["A"][flag])/n if n else None
                          for flag in ("object_shape", "pusher")}
            strata.append({"horizon": horizon, "contact_by_h4": contact, "rows": n,
                           "physical_branches": len({(p[0]["anchor"], p[0]["candidate"]) for p in selected}),
                           "anchors": len({p[0]["anchor"] for p in selected}), "valid_counts": counts,
                           "B_minus_A_validity": regression,
                           "passed": n == 0 or regression["object_shape"] >= -.05-1e-12})
    b_cells = [c for c in cells if c["arm"] == "B"]
    a_complete = all(c["completed"] == 48 for c in cells if c["arm"] == "A")
    return {"stage": "T", "passed": all(c["passed"] for c in b_cells) and a_complete and not invalid_labels and all(s["passed"] for s in strata),
            "AR_only": True, "ignored_TF_rows": sum(r.get("mode") == "TF" for r in rows),
            "by_arm_horizon": cells, "contact_strata": strata, "invalid_or_unpaired_contact_labels": invalid_labels,
            "conflicting_contact_branches": conflicting_contact_branches,
            "pusher_regression_is_gate": False, "regression_gate": "joint_object_full_shape_validity",
            "interpretation": "fixed_development_engineering_screen_not_general_dynamics_validation"}


def ar_sequence_scores(rows):
    """Return all 1536 planned AR prefixes; reject TF rather than scoring it.

    Endpoint measurements remain independent of earlier unknown frames. The
    stopping score uses the pure interval contract and model-only absorption.
    The original numeric diagnostics and explicit eligibility flags both survive.
    """
    indexed, output = _ar_index(list(rows)), []
    for arm in ARMS:
        for anchor in ANCHORS:
            for candidate in CANDIDATES:
                for seed in SEEDS:
                    frames = []
                    for horizon in HORIZONS:
                        row = _planned_row(indexed, arm, anchor, candidate, seed, horizon)
                        view = _measurement(row)
                        frames.append(view["scoring"])
                        score = sequence_endpoint(frames)
                        actual = _coverage(row.get("truth", {}).get("coverage")) if isinstance(row.get("truth"), dict) else None
                        output.append({"key": row.get("key"), "arm": arm, "anchor": anchor, "candidate": candidate,
                                       "seed": seed, "shape": anchor[0], "mode": "AR", "horizon": horizon,
                                       "status": row["status"], "input_present": (arm, anchor, candidate, seed, horizon) in indexed,
                                       "known_missing_physical_branch": anchor == "L01" and candidate == "hold",
                                       "measurement": deepcopy(view["measurement"]), "raw_endpoint": deepcopy(view["measurement"]),
                                       "raw_endpoint_valid": view["flags"]["coverage"], "raw_endpoint_coverage": view["estimate"],
                                       "raw_endpoint_termination": view["termination"],
                                       "sequence_score_eligible": score["stopped_score_valid"], "sequence_score": score["stopped_score"],
                                       "sequence_score_reason": score["stopped_score_reason"], "sequence_score_interval": score["stopped_score_interval"],
                                       "stop_horizon": score["stop_horizon"], "absorbing": score["absorbing"],
                                       "absorbing_model_state": score["absorbing_model_state"], "unknown_history": score["unknown_history"],
                                       "actual_coverage": actual, "future_truth_used_for_prediction_or_score": False})
    cells = []
    for arm in ARMS:
        for shape in ("L", "T"):
            for horizon in HORIZONS:
                selected = [r for r in output if r["arm"] == arm and r["shape"] == shape and r["horizon"] == horizon]
                cells.append({"arm": arm, "shape": shape, "horizon": horizon, "planned": len(selected),
                              "available_physical": sum(not r["known_missing_physical_branch"] for r in selected),
                              "status_counts": dict(Counter(r["status"] for r in selected)),
                              "raw_endpoint_eligible": sum(r["raw_endpoint_valid"] for r in selected),
                              "stopping_eligible": sum(r["sequence_score_eligible"] for r in selected)})
    return {"rows": output, "summary": {"planned_rows": 1536, "planned_streams": 384, "physical_branches_planned": 64,
                                         "physical_branches_available": 63, "input_rows": len(indexed), "by_arm_shape_horizon": cells},
            "TF_rejected": True, "raw_endpoint_is_independent_diagnostic": True,
            "intervals_are_development_assumptions": True}


def _pair_counts(candidate_scores, actual):
    counts = {name: 0 for name in ("eligible_pairs", "scored_pairs", "abstained_pairs", "correct_pairs", "incorrect_scored_pairs", "unknown_actual_pairs", "inseparable_actual_pairs")}
    for i, first in enumerate(CANDIDATES):
        for second in CANDIDATES[i+1:]:
            if actual[first] is None or actual[second] is None:
                counts["unknown_actual_pairs"] += 1
                continue
            difference = actual[first]-actual[second]
            if abs(difference)+1e-12 < PAIR_SEPARATION:
                counts["inseparable_actual_pairs"] += 1
                continue
            counts["eligible_pairs"] += 1
            a, b = candidate_scores[first], candidate_scores[second]
            if not a["eligible"] or not b["eligible"]:
                counts["abstained_pairs"] += 1
                continue
            counts["scored_pairs"] += 1
            predicted = a["score"]-b["score"]
            correct = abs(predicted) > TIE_TOLERANCE and predicted*difference > 0
            counts["correct_pairs" if correct else "incorrect_scored_pairs"] += 1
    return {**counts, "observed_pair_accuracy": counts["correct_pairs"]/counts["scored_pairs"] if counts["scored_pairs"] else None,
            "conservative_correct_over_all_eligible": counts["correct_pairs"]/counts["eligible_pairs"] if counts["eligible_pairs"] else None}


def _pair_summary(records):
    names = ("eligible_pairs", "scored_pairs", "abstained_pairs", "correct_pairs", "incorrect_scored_pairs", "unknown_actual_pairs", "inseparable_actual_pairs")
    totals = {name: sum(r["pairs"][name] for r in records) for name in names}
    return {**totals, "pooled_observed_correct_over_scored": totals["correct_pairs"]/totals["scored_pairs"] if totals["scored_pairs"] else None,
            "pooled_conservative_correct_over_all_eligible": totals["correct_pairs"]/totals["eligible_pairs"] if totals["eligible_pairs"] else None,
            "anchor_equal_observed_pair_accuracy": _mean([r["pairs"]["observed_pair_accuracy"] for r in records if r["pairs"]["observed_pair_accuracy"] is not None]),
            "anchor_equal_conservative_pair_accuracy": _mean([r["pairs"]["conservative_correct_over_all_eligible"] for r in records if r["pairs"]["conservative_correct_over_all_eligible"] is not None])}


def paired_rankings(ar_rows):
    """A/B comparisons and hold/uniform on the identical complete AR subset.

    Three fixed noises are averaged inside each candidate. Anchors receive
    equal weight. Independent physical references use their own disclosed
    denominator, and are never subtracted from selected-subset model means.
    Missing/blocked records and every abstained pair remain in the output.
    """
    scores = ar_sequence_scores(ar_rows)
    indexed = {(r["arm"], r["anchor"], r["candidate"], r["seed"], r["horizon"]): r for r in scores["rows"]}
    anchor_rows, physical_rows, common_rows = [], [], []
    for horizon in HORIZONS:
        for anchor in ANCHORS:
            actual, conflicts = {}, []
            for candidate in CANDIDATES:
                values = [indexed[(arm, anchor, candidate, seed, horizon)]["actual_coverage"] for arm in ARMS for seed in SEEDS]
                known = [v for v in values if v is not None]
                if known and max(known)-min(known) > 1e-12:
                    conflicts.append(candidate)
                # A physical label is reusable across noises/arms, independent
                # of forecast status. Conflicting labels are never averaged.
                actual[candidate] = known[0] if known and candidate not in conflicts else None
            physical_complete = all(v is not None for v in actual.values())
            best = max(actual.values()) if physical_complete else None
            hold, uniform = actual["hold"], _mean(list(actual.values())) if physical_complete else None
            physical_rows.append({"anchor": anchor, "shape": anchor[0], "horizon": horizon,
                                  "actual_coverage": actual, "truth_conflicts": conflicts, "complete": physical_complete,
                                  "hold_reward": hold, "uniform_expected_reward": uniform,
                                  "hold_regret": best-hold if physical_complete else None,
                                  "uniform_expected_regret": best-uniform if physical_complete else None})
            for mode, value_key, flag_key in (("raw_endpoint", "raw_endpoint_coverage", "raw_endpoint_valid"),
                                               ("stopping", "sequence_score", "sequence_score_eligible")):
                arm_records = {}
                for arm in ARMS:
                    candidates = {}
                    for candidate in CANDIDATES:
                        noise_rows = [indexed[(arm, anchor, candidate, seed, horizon)] for seed in SEEDS]
                        eligible = all(r[flag_key] is True and _finite(r[value_key]) for r in noise_rows)
                        values = [r[value_key] for r in noise_rows]
                        score = _mean(values) if eligible else None
                        candidates[candidate] = {"eligible": eligible, "score": score,
                                                 "noise_std": math.sqrt(_mean([(v-score)**2 for v in values])) if eligible else None,
                                                 "noises": [{"seed": r["seed"], "status": r["status"], "eligible": r[flag_key], "value": r[value_key]}
                                                            for r in noise_rows]}
                    eligible = all(c["eligible"] for c in candidates.values())
                    chosen = None
                    if eligible:
                        maximum = max(c["score"] for c in candidates.values())
                        chosen = next(c for c in CANDIDATES if maximum-candidates[c]["score"] <= TIE_TOLERANCE)
                    record = {"arm": arm, "anchor": anchor, "shape": anchor[0], "horizon": horizon, "scoring_mode": mode,
                              "candidates": candidates, "selection_eligible": eligible, "selected_candidate": chosen,
                              "physical_complete": physical_complete, "selected_actual_reward": actual[chosen] if chosen else None,
                              "regret": best-actual[chosen] if chosen and physical_complete else None,
                              "pairs": _pair_counts(candidates, actual)}
                    anchor_rows.append(record)
                    arm_records[arm] = record
                common = physical_complete and all(r["selection_eligible"] for r in arm_records.values())
                common_rows.append({"anchor": anchor, "shape": anchor[0], "horizon": horizon, "scoring_mode": mode,
                                    "included": common,
                                    "omission_reasons": ([] if physical_complete else ["incomplete_or_conflicting_physical_labels"])
                                                        + [arm+"_candidate_noise_abstention" for arm in ARMS if not arm_records[arm]["selection_eligible"]],
                                    "A_regret": arm_records["A"]["regret"] if common else None,
                                    "B_regret": arm_records["B"]["regret"] if common else None,
                                    "A_reward": arm_records["A"]["selected_actual_reward"] if common else None,
                                    "B_reward": arm_records["B"]["selected_actual_reward"] if common else None,
                                    "hold_reward": hold if common else None, "uniform_expected_reward": uniform if common else None,
                                    "hold_regret": best-hold if common else None,
                                    "uniform_expected_regret": best-uniform if common else None})
    paired, baselines = [], []
    for horizon in HORIZONS:
        for shape in ("L", "T", "all"):
            matching = lambda r: r["horizon"] == horizon and (shape == "all" or r["shape"] == shape)
            physical = [r for r in physical_rows if matching(r)]
            complete = [r for r in physical if r["complete"]]
            baselines.append({"shape": shape, "horizon": horizon, "planned_anchors": len(physical),
                              "complete_physical_anchors": len(complete), "omitted_anchors": [r["anchor"] for r in physical if not r["complete"]],
                              **{key: _mean([r[key] for r in complete]) for key in ("hold_reward", "uniform_expected_reward", "hold_regret", "uniform_expected_regret")},
                              "weighting": "equal_physical_anchors_independent_of_model_eligibility"})
            for mode in ("raw_endpoint", "stopping"):
                selected = [r for r in common_rows if matching(r) and r["scoring_mode"] == mode]
                included = [r for r in selected if r["included"]]
                paired.append({"shape": shape, "horizon": horizon, "scoring_mode": mode,
                               "planned_anchors": len(selected), "paired_complete_anchors": len(included),
                               "included_anchors": [r["anchor"] for r in included],
                               "omitted_anchors": [{"anchor": r["anchor"], "reasons": r["omission_reasons"]} for r in selected if not r["included"]],
                               **{key: _mean([r[key] for r in included]) for key in ("A_regret", "B_regret", "A_reward", "B_reward", "hold_reward", "uniform_expected_reward", "hold_regret", "uniform_expected_regret")},
                               "all_planned_anchor_pair_metrics": {arm: _pair_summary([r for r in anchor_rows if matching(r) and r["scoring_mode"] == mode and r["arm"] == arm]) for arm in ARMS},
                               "weighting": "noises_inside_candidate_then_equal_common_anchors",
                               "raw_is_diagnostic": mode == "raw_endpoint"})
    return {"anchor_rows": anchor_rows, "paired_complete_subset": paired, "paired_anchor_rows": common_rows,
            "independent_physical_baselines": baselines, "physical_anchor_rows": physical_rows,
            "denominators": scores["summary"], "TF_rejected": True,
            "pair_separation": PAIR_SEPARATION, "tie_tolerance": TIE_TOLERANCE,
            "uniform_reference": "exact_mean_of_four_saved_physical_outcomes_no_sampling",
            "subset_model_minus_full_baseline_comparison": "prohibited"}
