"""Development-only image-interface analysis of immutable saved diagnostics.

No environment, network, checkpoint, policy or torch module is imported. Truth
is used after decoding as an error label, never to crop, initialize, fit, repair
or select an answer. Every planned row remains present, including missing input.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


CANDIDATES = ("hold", "toward", "side", "away")
NOISE_SEEDS = (5478, 5479, 5480)
FAMILIES = ("noncontact",)*3 + ("translation",)*3 + ("rotation_cw", "rotation_ccw", "rotation_ccw") + ("near_goal",)*3 + ("noncontact", "translation", "rotation", "near_goal")
METRICS = ("position_error_pixels", "angle_error_degrees", "pusher_error_pixels", "coverage_absolute_error")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def average(values):
    selected = [float(v) for v in values if finite(v)]
    return statistics.fmean(selected) if selected else None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _step(rows, horizon):
    matches = [r for r in rows if r.get("horizon") == horizon]
    if len(matches) > 1:
        raise ValueError("Duplicate saved horizon")
    return matches[0] if matches else None


def _state_label(state, shape):
    if not state:
        return None
    from .envs.official import official_shape
    from .evaluation import goal_metrics
    from .geometry import Pose
    pose, goal = state["object_body"]["pose"], state["goal"]
    return dict(pose=pose, pusher=state["pusher_body"]["pose"]["position"],
                coverage=goal_metrics(official_shape(shape), Pose(**pose), Pose(**goal)).coverage)


def build_inventory(report, run_dir):
    """Plan all 1304 rows; only RGB paths/indices, never simulator construction."""
    run_dir = Path(run_dir).resolve()
    lookup = {}
    for branch in report.get("branches", []):
        key = branch["anchor"], branch["candidate"]
        if key in lookup:
            raise ValueError("Duplicate physical branch")
        lookup[key] = branch
    calibrations = {r["id"]: r for r in report.get("calibrations", [])}
    items = []
    for shape in ("L", "T"):
        for index in range(12):
            identity = f"cal-{shape}-{index:02d}"
            row = calibrations.get(identity, {})
            items.append(dict(id=identity, source="calibration", shape=shape, anchor=identity,
                candidate=None, noise_seed=None, horizon=None, family="render_calibration",
                path=str(run_dir / (identity+".npz")), key="rgb", index=None,
                goal=row.get("state", {}).get("goal"), truth=row.get("truth"), latest=None,
                stored_v1=row.get("prediction"), branch_status="calibration",
                contact_by_H4=None, rotation_gt_5=None, physical_absorbing=False))
    for index in range(16):
        shape = "L" if index < 12 else "T"
        anchor = f"{shape}{index:02d}"
        for candidate in CANDIDATES:
            branch = lookup.get((anchor, candidate), {})
            initial = branch.get("initial_state", {})
            goal = initial.get("goal")
            physical = str(run_dir / f"{anchor}-{candidate}-physical.npz")
            latest = branch.get("latest")
            actual = branch.get("actual", [])
            h4 = _step(actual, 4)
            contact = any(r.get("contacts", 0)>0 for r in actual) if h4 else None
            common = dict(shape=shape, anchor=anchor, candidate=candidate,
                family=branch.get("family", FAMILIES[index]), goal=goal, latest=latest,
                branch_status=branch.get("status", "missing_branch"), contact_by_H4=contact)
            history = branch.get("history", [])
            for k in range(4):
                state = initial if k == 0 else history[k-1].get("state") if len(history)>=k else None
                items.append(dict(common, id=f"{anchor}-{candidate}-history-{k}", source="history",
                    noise_seed=None, horizon=k-3, path=physical, key="history", index=k,
                    truth=_state_label(state, shape), stored_v1=None, rotation_gt_5=None,
                    physical_absorbing=False))
            predictions = {}
            for prediction in branch.get("predictions", []):
                if prediction["seed"] in predictions:
                    raise ValueError("Duplicate forecast seed")
                predictions[prediction["seed"]] = prediction
            for horizon in range(1, 5):
                step = _step(actual, horizon) or {}
                truth = step.get("truth")
                rotation = (math.degrees(wrapped(truth["pose"]["angle"]-latest["pose"]["angle"]))
                            if truth and latest else None)
                values = dict(common, horizon=horizon, truth=truth,
                    rotation_gt_5=abs(rotation)>5 if rotation is not None else None,
                    true_rotation_degrees=rotation, physical_absorbing=step.get("absorbing", False),
                    actual_contacts=step.get("contacts"), actual_success=step.get("success"))
                items.append(dict(values, id=f"{anchor}-{candidate}-future-H{horizon}", source="physical_future",
                    noise_seed=None, path=physical, key="actual", index=horizon-1, stored_v1=None))
                for seed in NOISE_SEEDS:
                    prediction = predictions.get(seed, {})
                    predicted = _step(prediction.get("steps", []), horizon) or {}
                    items.append(dict(values, id=f"{anchor}-{candidate}-noise-{seed}-H{horizon}", source="forecast",
                        noise_seed=seed, path=str(run_dir / f"{anchor}-{candidate}-noise-{seed}.npz"),
                        key="raw", index=horizon-1, stored_v1=predicted.get("decoded"),
                        stored_v1_score=predicted.get("score_coverage"),
                        stored_v1_predicted_absorbing=predicted.get("predicted_absorbing")))
    assert len(items) == 1304
    return items


@lru_cache(maxsize=6)
def _archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_rgb(item):
    path = Path(item["path"])
    if not path.is_file():
        return None
    arrays = _archive(str(path))
    if item["key"] not in arrays:
        return None
    values = arrays[item["key"]]
    if item["index"] is not None:
        if len(values) <= item["index"]:
            return None
        values = values[item["index"]]
    if values.shape not in ((3, 96, 96), (96, 96, 3)):
        raise ValueError("Saved RGB has wrong shape")
    return values


def _pose(value):
    if not isinstance(value, dict):
        return None
    pose = value.get("pose", value)
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    angle = pose.get("angle")
    if not isinstance(position, (list, tuple)) or len(position)!=2 or not all(finite(v) for v in position) or not finite(angle):
        return None
    return dict(position=list(position), angle=float(angle))


def measurement_view(value, version):
    """A flat reporting view; the untouched full decoder record is also saved."""
    value = value or {}
    if version == "v1":
        valid = value.get("valid") is True
        pose = _pose(value.get("pose")) if valid else None
        coverage = value.get("coverage") if valid else None
        pusher = value.get("pusher") if valid else None
        return dict(object_valid=pose is not None, shape_valid=pose is not None, pose=pose,
            pusher_valid=pusher is not None, pusher=pusher, coverage_valid=finite(coverage),
            coverage=coverage if finite(coverage) else None, coverage_interval=None,
            termination="success" if finite(coverage) and coverage>.95 else "non_success" if finite(coverage) else "unknown",
            reason=value.get("reason", "missing_input"), ambiguity=None, ambiguity_assessed=False)
    state = value.get("state", {})
    shape = value.get("shape", {})
    pusher = value.get("pusher", {})
    coverage = value.get("coverage", {})
    termination = value.get("termination", {})
    pose = _pose(state)
    object_valid = state.get("valid", state.get("measurable", pose is not None)) is True if isinstance(state, dict) else False
    shape_valid = shape.get("valid", shape.get("measurable", object_valid)) is True if isinstance(shape, dict) else object_valid
    center = pusher.get("position", pusher.get("center")) if isinstance(pusher, dict) else pusher
    pusher_valid = (pusher.get("valid", pusher.get("measurable", center is not None)) is True) if isinstance(pusher, dict) else center is not None
    estimate = coverage.get("estimate", coverage.get("value")) if isinstance(coverage, dict) else coverage
    interval = coverage.get("interval") if isinstance(coverage, dict) else None
    coverage_valid = (coverage.get("valid", finite(estimate)) is True) if isinstance(coverage, dict) else finite(estimate)
    decision = termination.get("decision", termination.get("status", "unknown")) if isinstance(termination, dict) else termination
    ambiguity_assessed = isinstance(state, dict) and state.get("ambiguity_assessed") is True
    return dict(object_valid=object_valid and pose is not None, shape_valid=shape_valid, pose=pose,
        pusher_valid=pusher_valid and center is not None, pusher=center,
        coverage_valid=coverage_valid and finite(estimate), coverage=estimate if finite(estimate) else None,
        coverage_interval=interval, termination=decision or "unknown",
        reason=value.get("reason", state.get("reason", "unspecified") if isinstance(state, dict) else "unspecified") if value else "missing_input",
        ambiguity=state.get("ambiguity", state.get("ambiguous")) if ambiguity_assessed else None,
        ambiguity_assessed=ambiguity_assessed)


def error_labels(item, view):
    truth, latest = item.get("truth"), item.get("latest")
    result = {key: None for key in METRICS}
    result.update(coverage_bias=None, truth_coverage=None, actual_success=None, nominal_predicted_success=None,
                  near_095=None, actual_rotation_degrees=item.get("true_rotation_degrees"),
                  predicted_rotation_degrees=None, rotation_direction_correct=None)
    if not truth:
        return result
    if view["object_valid"] and view["pose"]:
        pose = view["pose"]
        result["position_error_pixels"] = math.dist(pose["position"], truth["pose"]["position"])
        result["angle_error_degrees"] = abs(math.degrees(wrapped(pose["angle"]-truth["pose"]["angle"])))
        if latest:
            rotation = math.degrees(wrapped(pose["angle"]-latest["pose"]["angle"]))
            result["predicted_rotation_degrees"] = rotation
            actual_rotation = item.get("true_rotation_degrees")
            result["rotation_direction_correct"] = float(rotation*actual_rotation>0) if actual_rotation is not None else None
    if view["pusher_valid"]:
        result["pusher_error_pixels"] = math.dist(view["pusher"], truth["pusher"])
    if view["coverage_valid"]:
        result["coverage_bias"] = view["coverage"]-truth["coverage"]
        result["coverage_absolute_error"] = abs(result["coverage_bias"])
        result["nominal_predicted_success"] = view["coverage"]>.95
    result["actual_success"] = truth["coverage"]>.95
    result["truth_coverage"] = truth["coverage"]
    result["near_095"] = abs(truth["coverage"]-.95) <= .05+1e-12
    return result


def measure_inventory(items, *, decoder_v1, decoder_v2, progress=None):
    caches = {"v1": {}, "v2": {}}
    rows, calls = [], Counter()
    for index, item in enumerate(items):
        image = load_rgb(item)
        if image is not None:
            image.setflags(write=False)
        available = image is not None and item.get("goal") is not None
        digest = (hashlib.sha256(str(image.dtype).encode()+str(image.shape).encode()+image.tobytes()).hexdigest()
                  if image is not None else None)
        cache_key = (digest, item["shape"], json.dumps(item.get("goal"), sort_keys=True))
        record = dict(item, input_available=available, rgb_sha256=digest)
        for version, decoder in (("v1", decoder_v1), ("v2", decoder_v2)):
            if version == "v1" and item.get("stored_v1") is not None:
                measured = item["stored_v1"]
                origin = "stored_report"
            elif not available:
                measured = None
                origin = "missing_input"
            elif cache_key in caches[version]:
                measured = caches[version][cache_key]
                origin = "saved_rgb_hash_cache"
            else:
                # Deliberately only RGB, disclosed geometry ID and fixed goal.
                measured = decoder(image, item["shape"], item["goal"])
                caches[version][cache_key] = measured
                calls[version] += 1
                origin = "decoded_saved_rgb"
            view = measurement_view(measured, version)
            record[version] = dict(measurement=measured, origin=origin, view=view,
                                   errors=error_labels(item, view))
        rows.append(record)
        if progress and (index+1)%50 == 0:
            progress(index+1, len(items), dict(calls))
    return rows, dict(calls)


def flat_images(rows):
    output = []
    for row in rows:
        base = {k: row.get(k) for k in ("id", "source", "shape", "anchor", "candidate", "noise_seed", "horizon", "family", "branch_status", "contact_by_H4", "rotation_gt_5", "physical_absorbing", "input_available", "rgb_sha256", "path", "key", "index")}
        for version in ("v1", "v2"):
            value = row[version]
            view = value["view"]
            flat = dict(base, version=version, measurement_origin=value["origin"],
                        **{k: v for k, v in view.items() if k not in ("pose", "pusher")}, **value["errors"])
            pose = view.get("pose") or {}
            position = pose.get("position") or [None, None]
            pusher = view.get("pusher") or [None, None]
            flat.update(estimated_x=position[0], estimated_y=position[1], estimated_angle=pose.get("angle"),
                        estimated_pusher_x=pusher[0], estimated_pusher_y=pusher[1],
                        truth_coverage=(row.get("truth") or {}).get("coverage"))
            output.append(flat)
    return output


def weighted_measurements(rows, metric):
    """Observed subset errors, noise -> branch/frame -> equal anchor means."""
    branches = defaultdict(list)
    anchors = defaultdict(list)
    for row in rows:
        branches[row["anchor"], row.get("candidate"), row.get("horizon")].append(row.get(metric))
    valid_branches = full_branches = 0
    for (anchor, _, _), values in branches.items():
        observed = [float(v) for v in values if finite(v)]
        if observed:
            anchors[anchor].append(average(observed))
            valid_branches += 1
        full_branches += bool(values and len(observed)==len(values))
    valid = sum(finite(r.get(metric)) for r in rows)
    return dict(anchor_equal_observed_mean=average([average(v) for v in anchors.values()]),
        measured_image_rows=valid, expected_image_rows=len(rows),
        measurable_fraction=valid/len(rows) if rows else None,
        expected_branch_frames=len(branches), measured_branch_frames=valid_branches,
        fully_measured_branch_frames=full_branches,
        expected_anchors=len({r["anchor"] for r in rows}), measured_anchors=len(anchors),
        aggregate_is_development_only=True, passes_validation_gate=False)


def confusion(rows, *, interval=False):
    counts = Counter()
    for row in rows:
        actual = row.get("actual_success")
        if actual is None:
            counts["truth_unknown"] += 1
            continue
        predicted = row.get("nominal_predicted_success")
        if interval:
            predicted = True if row.get("termination")=="success" else False if row.get("termination")=="non_success" else None
        if predicted is None:
            counts["unknown"] += 1
        else:
            counts["TP" if actual and predicted else "TN" if not actual and not predicted else "FP" if predicted else "FN"] += 1
    result = {key: counts[key] for key in ("TP", "TN", "FP", "FN", "unknown", "truth_unknown")}
    predicted_positive = result["TP"]+result["FP"]
    actual_positive = result["TP"]+result["FN"]
    result.update(total=len(rows), precision=result["TP"]/predicted_positive if predicted_positive else None,
        recall_among_classified=result["TP"]/actual_positive if actual_positive else None,
        classified_fraction=(result["TP"]+result["TN"]+result["FP"]+result["FN"])/len(rows) if rows else None,
        precision_undefined_reason=None if predicted_positive else "no predicted-positive states",
        method="declared_interval" if interval else "nominal_point_estimate",
        development_only=True, success_precision_validated=False)
    return result


def stratified_tables(flat):
    errors, classifications = [], []
    for source in ("calibration", "history", "physical_future", "forecast"):
        horizons = (None,) if source=="calibration" else ("history",) if source=="history" else (1, 2, 3, 4)
        for shape in ("L", "T"):
            for version in ("v1", "v2"):
                for horizon in horizons:
                    base = [r for r in flat if r["source"]==source and r["shape"]==shape and r["version"]==version
                            and (source in ("calibration", "history") or r["horizon"]==horizon)]
                    strata = {"all": base, "near_095": [r for r in base if r.get("near_095") is True]}
                    if source in ("physical_future", "forecast"):
                        dynamics = [r for r in base if not r.get("physical_absorbing")]
                        strata.update(contact=[r for r in dynamics if r.get("contact_by_H4") is True],
                            noncontact=[r for r in dynamics if r.get("contact_by_H4") is False],
                            rotation_gt_5=[r for r in dynamics if r.get("rotation_gt_5") is True])
                    for stratum, members in strata.items():
                        common = dict(source=source, shape=shape, version=version, horizon=horizon, stratum=stratum,
                            unknown_contact_labels=sum(r.get("contact_by_H4") is None for r in base),
                            excluded_physical_absorbing_rows=sum(bool(r.get("physical_absorbing")) for r in base) if stratum in ("contact", "noncontact", "rotation_gt_5") else 0)
                        for metric in (*METRICS, "rotation_direction_correct"):
                            errors.append(dict(common, metric=metric, **weighted_measurements(members, metric)))
                        # V1 declares only point-based stopping, never an interval.
                        for interval in ((False,) if version=="v1" else (False, True)):
                            classifications.append(dict(common, **confusion(members, interval=interval)))
    return errors, classifications


def sequence_table(rows):
    """Keep endpoint measurement independent of stopping-prefix eligibility."""
    from .wm_measurement_contract import sequence_endpoint
    grouped = defaultdict(dict)
    for row in rows:
        if row["source"]=="forecast":
            grouped[row["anchor"], row["candidate"], row["noise_seed"]][row["horizon"]] = row
    results = []
    for (anchor, candidate, seed), frames in grouped.items():
        for version in ("v1", "v2"):
            nominal_stop = None
            nominal_invalid = False
            for horizon in range(1, 5):
                row = frames[horizon]
                view = row[version]["view"]
                if version=="v1":
                    if nominal_stop is not None:
                        score = nominal_stop
                    elif nominal_invalid or not view["coverage_valid"]:
                        nominal_invalid, score = True, None
                    else:
                        score = view["coverage"]
                        if score>.95:
                            nominal_stop = score
                    # The original report is the v1 sequence-scoring authority.
                    if row.get("stored_v1") is not None and "stored_v1_score" in row:
                        score = row["stored_v1_score"]
                    sequence = dict(stopped_score=score, stopped_score_valid=finite(score),
                        stopped_score_reason="original_v1_nominal_point_stopping", absorbing=nominal_stop is not None,
                        stop_horizon=None, unknown_history=[] if not nominal_invalid else [{"reason":"invalid_v1_prefix"}])
                else:
                    prefix = [frames[h][version]["measurement"] or {} for h in range(1, horizon+1)]
                    sequence = sequence_endpoint(prefix)
                truth = row.get("truth") or {}
                score = sequence["stopped_score"] if sequence.get("stopped_score_valid") else None
                results.append(dict(anchor=anchor, candidate=candidate, noise_seed=seed, version=version,
                    shape=row["shape"], family=row["family"], horizon=horizon,
                    raw_endpoint_coverage=view["coverage"], raw_endpoint_valid=view["coverage_valid"],
                    raw_endpoint_termination=view["termination"], sequence_score=score,
                    sequence_score_eligible=finite(score), sequence_reason=sequence.get("stopped_score_reason"),
                    predicted_absorbing=sequence.get("absorbing"), predicted_stop_horizon=sequence.get("stop_horizon"),
                    unknown_history_count=len(sequence.get("unknown_history", [])),
                    physical_absorbing=row.get("physical_absorbing"), actual_coverage=truth.get("coverage"),
                    score_error=abs(score-truth["coverage"]) if finite(score) and finite(truth.get("coverage")) else None,
                    raw_endpoint_error=abs(view["coverage"]-truth["coverage"]) if view["coverage_valid"] and truth else None))
    return results


def baseline_table(rows):
    """Baselines are evaluated regardless of model or decoder measurability."""
    output = []
    for row in rows:
        if row["source"]!="physical_future":
            continue
        truth, latest = row.get("truth"), row.get("latest")
        record = {k: row.get(k) for k in ("anchor", "shape", "family", "candidate", "horizon", "contact_by_H4", "physical_absorbing")}
        record.update(actual_coverage=truth.get("coverage") if truth else None,
            privileged_observed_pose_persistence_coverage=latest.get("coverage") if latest else None,
            privileged_observed_pose_persistence_error=abs(latest["coverage"]-truth["coverage"]) if latest and truth else None,
            privileged_observed_pose_position_error=math.dist(latest["pose"]["position"], truth["pose"]["position"]) if latest and truth else None,
            privileged_observed_pose_angle_error=abs(math.degrees(wrapped(latest["pose"]["angle"]-truth["pose"]["angle"]))) if latest and truth else None,
            denominator_independent_of_model_validity=True)
        output.append(record)
    return output


def ranking_table(sequences):
    grouped = defaultdict(list)
    for row in sequences:
        grouped[row["version"], row["anchor"], row["horizon"]].append(row)
    results = []
    for (version, anchor, horizon), members in grouped.items():
        for mode, key in (("raw_endpoint", "raw_endpoint_coverage"), ("stopping_consistent", "sequence_score")):
            validity_key = "raw_endpoint_valid" if mode=="raw_endpoint" else "sequence_score_eligible"
            actual, predicted, eligible_noises, noise_std, diagnostic_values, diagnostic_flags = [], [], [], [], [], []
            for candidate in CANDIDATES:
                branch = [r for r in members if r["candidate"]==candidate]
                by_seed = {}
                for row in branch:
                    if row["noise_seed"] in by_seed:
                        raise ValueError("Duplicate candidate noise in ranking input")
                    by_seed[row["noise_seed"]] = row
                if set(by_seed)-set(NOISE_SEEDS):
                    raise ValueError("Unexpected candidate noise in ranking input")
                values = [by_seed.get(seed, {}).get(key) for seed in NOISE_SEEDS]
                flags = [by_seed.get(seed, {}).get(validity_key) is True for seed in NOISE_SEEDS]
                # Diagnostic point values can remain finite after abstention.
                # A score requires both explicit eligibility and finite value
                # for every one of the three predeclared noise seeds.
                valid_values = [value for value, flag in zip(values,flags) if flag and finite(value)]
                count = len(valid_values)
                actual.append(branch[0]["actual_coverage"] if branch else None)
                predicted.append(average(valid_values) if count==3 else None)
                eligible_noises.append(count)
                noise_std.append(statistics.pstdev(valid_values) if count==3 else None)
                diagnostic_values.append(values)
                diagnostic_flags.append(flags)
            prediction_complete = all(finite(v) for v in predicted)
            selected = None
            if prediction_complete:
                maximum = max(predicted)
                selected = next(i for i, value in enumerate(predicted) if maximum-value <= 1e-6)
            actual_complete = all(finite(v) for v in actual)
            eligible_pairs = correct_pairs = known_pairs = scored_pairs = 0
            for i in range(4):
                for j in range(i+1, 4):
                    if finite(actual[i]) and finite(actual[j]):
                        known_pairs += 1
                        delta = actual[i]-actual[j]
                        if abs(delta)+1e-12 >= .05:
                            eligible_pairs += 1
                            if finite(predicted[i]) and finite(predicted[j]):
                                scored_pairs += 1
                                pdiff = predicted[i]-predicted[j]
                                correct_pairs += int(abs(pdiff)>1e-6 and pdiff*delta>0)
            best = max(actual) if actual_complete else None
            regret = best-actual[selected] if best is not None and selected is not None else None
            result = dict(version=version, shape=members[0]["shape"], anchor=anchor, horizon=horizon, scoring_mode=mode,
                actual_complete=actual_complete, prediction_complete=prediction_complete,
                eligible_pairs=eligible_pairs, scored_pairs=scored_pairs, abstained_pairs=eligible_pairs-scored_pairs,
                correct_pairs=correct_pairs, incorrect_scored_pairs=scored_pairs-correct_pairs,
                unknown_actual_pairs=6-known_pairs,
                pair_accuracy=correct_pairs/scored_pairs if scored_pairs else None,
                observed_pair_accuracy=correct_pairs/scored_pairs if scored_pairs else None,
                observed_pair_accuracy_definition="correct / scored eligible actual pairs; unscored pairs are abstentions, not observed model mistakes",
                conservative_pair_accuracy=correct_pairs/eligible_pairs if eligible_pairs else None,
                conservative_pair_accuracy_definition="correct / all eligible actual pairs; abstentions receive zero credit only in this conservative engineering bound",
                selected_candidate=CANDIDATES[selected] if selected is not None else None,
                selection_abstention=selected is None, selected_regret=regret,
                regret_bound_lower=regret if regret is not None else 0.0,
                regret_bound_upper=regret if regret is not None else 1.0,
                hold_regret=best-actual[0] if best is not None else None,
                uniform_candidate_expectation_regret=best-average(actual) if best is not None else None,
                constant_reward_selects="hold", physical_absorption_present=any(r.get("physical_absorbing") for r in members),
                interpretation="raw endpoint is a separate diagnostic, not stopping-consistent task reward" if mode=="raw_endpoint" else "prediction-only interval stopping, no future truth repair")
            for i, name in enumerate(CANDIDATES):
                result.update({"actual_"+name:actual[i], "predicted_"+name:predicted[i],
                    "eligible_noises_"+name:eligible_noises[i], "noise_std_"+name:noise_std[i],
                    "diagnostic_values_"+name:diagnostic_values[i], "eligibility_flags_"+name:diagnostic_flags[i]})
            results.append(result)
    return results


def rank_summary(rows):
    output = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["version"], row["shape"], row["horizon"], row["scoring_mode"]].append(row)
    for (version, shape, horizon, mode), members in grouped.items():
        output.append(dict(version=version, shape=shape, horizon=horizon, scoring_mode=mode,
            expected_anchors=len(members), actual_complete_anchors=sum(r["actual_complete"] for r in members),
            prediction_complete_anchors=sum(r["prediction_complete"] for r in members),
            selection_abstentions=sum(r["selection_abstention"] for r in members),
            eligible_pairs=sum(r["eligible_pairs"] for r in members), unknown_actual_pairs=sum(r["unknown_actual_pairs"] for r in members),
            scored_pairs=sum(r["scored_pairs"] for r in members), abstained_pairs=sum(r["abstained_pairs"] for r in members),
            incorrect_scored_pairs=sum(r["incorrect_scored_pairs"] for r in members),
            anchors_with_scored_pairs=sum(r["scored_pairs"]>0 for r in members),
            anchor_equal_pair_accuracy=average([r["observed_pair_accuracy"] for r in members]),
            anchor_equal_observed_pair_accuracy=average([r["observed_pair_accuracy"] for r in members]),
            anchor_equal_conservative_pair_accuracy=average([r["conservative_pair_accuracy"] for r in members]),
            observed_pair_accuracy_definition="equal anchor mean of correct/scored; anchors with no scored pair are unavailable and counted explicitly",
            conservative_pair_accuracy_definition="equal anchor mean of correct/all eligible; abstentions receive no engineering-gate credit, not an observed mistake classification",
            observed_selected_regret=average([r["selected_regret"] for r in members]),
            conservative_regret_lower=average([r["regret_bound_lower"] for r in members]),
            conservative_regret_upper=average([r["regret_bound_upper"] for r in members]),
            hold_regret_independent=average([r["hold_regret"] for r in members]),
            uniform_expectation_regret_independent=average([r["uniform_candidate_expectation_regret"] for r in members]),
            development_only=True, validates_new_model_or_success=False))
    return output


def case_evidence(rows):
    """Posthoc explanation labels, never inputs to fitting or score selection."""
    groups = defaultdict(list)
    for row in rows:
        v1, v2 = row["v1"], row["v2"]
        old, new = v1["view"], v2["view"]
        errors = v2["errors"]
        if not row["input_available"]:
            continue
        categories = []
        if old["coverage_valid"] is False and new["coverage_valid"] is True:
            categories.append("interface_measurement_recovered")
        if new["object_valid"] and not new["pusher_valid"]:
            categories.append("object_measurement_independent_of_pusher")
        if errors.get("near_095") and new["termination"]=="unknown":
            categories.append("near_threshold_declared_interval_ambiguity" if new["coverage_valid"] and new.get("coverage_interval")
                              else "near_threshold_missing_measurement")
        if row["source"]!="forecast" and finite(v1["errors"].get("coverage_absolute_error")) and finite(errors.get("coverage_absolute_error")):
            if errors["coverage_absolute_error"]+1e-6 < v1["errors"]["coverage_absolute_error"]:
                categories.append("real_image_interface_bias_reduced")
        if row["source"]=="forecast":
            if not new["shape_valid"]:
                categories.append("model_shape_or_image_ambiguity")
            if new["object_valid"] and (finite(errors["position_error_pixels"]) and errors["position_error_pixels"]>5
                                       or finite(errors["angle_error_degrees"]) and errors["angle_error_degrees"]>5):
                categories.append("measurable_model_state_error")
            if row.get("rotation_gt_5") and finite(errors.get("predicted_rotation_degrees")) and abs(errors["predicted_rotation_degrees"])<2:
                categories.append("model_missed_true_rotation")
        elif not new["object_valid"]:
            categories.append("interface_abstention_on_real_rgb")
        for category in categories:
            groups[category].append(dict(id=row["id"], source=row["source"], shape=row["shape"],
                anchor=row["anchor"], candidate=row["candidate"], noise_seed=row.get("noise_seed"), horizon=row.get("horizon"),
                path=row["path"], key=row["key"], index=row["index"], rgb_sha256=row["rgb_sha256"],
                v1_view=old, v2_view=new, error_labels=errors,
                interpretation="posthoc label only; interval overlap is not proof of intrinsic RGB information loss, and no category establishes an unseen-data validation pass"))
    return {name: dict(total_matching_rows=len(values), examples=values[:6],
                       selection="first six in fixed inventory order; no decoder tuning or scenario filtering")
            for name, values in groups.items()}


def analyse(run_dir, *, decoder_v1, decoder_v2, progress=None):
    """Read/measure only saved RGB; caller owns output files and decoder versions."""
    started = time.monotonic()
    cpu_started = time.process_time()
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / "report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    inventory = build_inventory(report, run_dir)
    input_files = {str(report_path): dict(sha256=hashlib.sha256(report_bytes).hexdigest(), bytes=len(report_bytes))}
    for path in sorted({item["path"] for item in inventory}):
        file = Path(path)
        if file.is_file():
            input_files[path] = dict(sha256=sha256(file), bytes=file.stat().st_size)
    measured, calls = measure_inventory(inventory, decoder_v1=decoder_v1, decoder_v2=decoder_v2, progress=progress)
    flat = flat_images(measured)
    strata, classifications = stratified_tables(flat)
    sequences = sequence_table(measured)
    ranks = ranking_table(sequences)
    summary = dict(kind="tdyssey.wm_interface_v2_development.v1", development_only=True,
        data_seen_before_decoder_revision=True, independent_validation=False,
        expected_image_rows=len(inventory), available_image_rows=sum(r["input_available"] for r in measured),
        available_by_source=dict(Counter(r["source"] for r in measured if r["input_available"])),
        expected_by_source=dict(Counter(r["source"] for r in measured)), decoder_calls=calls,
        repeated_rgb_cache_preserves_all_row_denominators=True,
        expected_forecast_rows=768, expected_physical_branches=64, expected_noise_seeds=list(NOISE_SEEDS),
        noise_is_independent_physics=False, additional_physics_controls=0, additional_model_inference=0,
        policy_optimizer_updates=0, world_model_optimizer_updates=0,
        operation_counts_scope="additional operations by this saved-image diagnostic only",
        environment_creations=0, resets=0, restores=0, true_renders=0, controls=0,
        model_inference=0, policy_sampling=0, optimizer_updates=0,
        statistics="measurable subset errors; noises nested inside branch/frame, then anchors weighted equally; all missing denominators retained",
        caveats=["All reused data are development data already inspected before v2; no validation claim.",
                 "Development absolute-error intervals are assumptions, not probabilities or calibrated guarantees on generated images.",
                 "Raw image measurements are separate from sequence score eligibility; unknown earlier stopping is never repaired from future truth.",
                 "Baselines have independent physical denominators; model abstention does not hide them.",
                 "Contact is a posthoc real solver-contact stratum, not a decoded contact prediction."],
        rank_summary=rank_summary(ranks), wall_seconds=time.monotonic()-started,
        cpu_seconds=time.process_time()-cpu_started, input_files=input_files)
    for path, identity in input_files.items():
        if sha256(path)!=identity["sha256"]:
            raise RuntimeError("Saved source asset changed during offline analysis: "+path)
    _archive.cache_clear()
    return dict(summary=summary, images=measured, flat_images=flat, strata=strata,
                classifications=classifications, sequences=sequences, rankings=ranks,
                baselines=baseline_table(measured), cases=case_evidence(measured))
