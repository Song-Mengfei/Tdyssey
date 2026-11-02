"""Anchor-weighted frozen diagnostic summaries and saved-array scientific plots.

Only saved RGB and reported measurements are read. No simulator, model, decoder,
future-truth repair, or scene selection is performed here.
"""

from collections import defaultdict
import math
from pathlib import Path
import statistics

from .frozen_wm_protocol import CANDIDATES, NOISE_SEEDS, anchors, rank_scores, wrapped_degrees


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _mean(values):
    values = [float(v) for v in values if _number(v)]
    return statistics.fmean(values) if values else None


def _step(rows, horizon):
    return next((row for row in rows if row.get("horizon") == horizon), None)


def _nested(values):
    valid = [float(value) for value in values if _number(value)]
    return {"mean": _mean(valid) if len(valid) == len(NOISE_SEEDS) else None,
            "observed_mean": _mean(valid), "valid_noises": len(valid),
            "expected_noises": len(NOISE_SEEDS),
            "noise_std": statistics.pstdev(valid) if len(valid) == len(NOISE_SEEDS) else None,
            "noise_min": min(valid) if valid else None, "noise_max": max(valid) if valid else None}


def _weighted(rows, key, *, conservative_missing=None):
    """Noise means -> branch means -> equal anchor means; retain all denominators."""
    grouped, complete = defaultdict(list), defaultdict(list)
    for row in rows:
        value = row.get(key)
        complete[row["anchor"]].append(_number(value))
        if _number(value):
            grouped[row["anchor"]].append(float(value))
        elif conservative_missing is not None:
            grouped[row["anchor"]].append(float(conservative_missing))
    available = sum(_number(row.get(key)) for row in rows)
    return {"anchor_equal_mean": _mean([_mean(values) for values in grouped.values()]),
            "expected_anchors": len(complete), "anchors_with_measurements": sum(any(v) for v in complete.values()),
            "fully_measured_anchors": sum(all(v) for v in complete.values()),
            "expected_branches": len(rows), "valid_branches": available, "missing_branches": len(rows)-available,
            "denominator": "all expected branches retained; means nest noises inside branches and weight anchors equally",
            "missing_policy": "fixed bound imputation" if conservative_missing is not None else "observed means only; missing blocks a passing gate"}


def _screen(calibrated, enough, passed, missing=0, reason=None):
    if not calibrated:
        return {"status": "unavailable", "reason": "image measurement calibration failed or missing"}
    if not enough:
        return {"status": "coverage insufficient", "reason": reason or "predeclared physical sample minima not met"}
    if missing:
        return {"status": "fail", "reason": "invalid predictions retained in denominator", "missing": missing}
    return {"status": "pass" if passed else "fail"}


def _candidate(row, spec, candidate, horizon, calibrated):
    actual = _step(row.get("actual", []), horizon) if row else None
    truth = actual.get("truth") if actual else None
    valid_truth = isinstance(truth, dict) and _number(truth.get("coverage"))
    latest = row.get("latest") if row else None
    physical_absorbing = bool(actual and actual.get("absorbing"))
    actual_h4 = _step(row.get("actual", []), 4) if row else None
    contact_known = actual_h4 is not None
    contact = bool(row and any(step.get("contacts", 0) > 0 for step in row.get("actual", []))) if contact_known else None
    values = defaultdict(list)
    noise_rows = []
    for seed in NOISE_SEEDS:
        prediction = next((p for p in row.get("predictions", []) if p.get("seed") == seed), None) if row else None
        step = _step(prediction.get("steps", []), horizon) if prediction else None
        decoded = step.get("decoded", {}) if step else {}
        score = step.get("score_coverage", decoded.get("coverage") if decoded.get("valid") else None) if step else None
        if not calibrated or not _number(score):
            score = None
        values["score"].append(score)
        noise_rows.append({"seed": seed, "score_coverage": score,
                           "predicted_absorbing": bool(step and step.get("predicted_absorbing")),
                           "decoder_valid": bool(decoded.get("valid")), "decoder_reason": decoded.get("reason", "missing_prediction")})
        motion = step.get("motion") if step else None
        for key in ("position_error", "angle_error", "pusher_error"):
            values[key].append(motion.get(key) if calibrated and motion and motion.get("valid") and not physical_absorbing else None)
        values["persistence_position_error"].append(math.dist(latest["pose"]["position"], truth["pose"]["position"])
            if valid_truth and latest and not physical_absorbing else None)
        values["persistence_angle_error"].append(abs(wrapped_degrees(latest["pose"]["angle"]-truth["pose"]["angle"]))
            if valid_truth and latest and not physical_absorbing else None)
        values["rotation_direction"].append(float(bool(motion.get("rotation_direction_correct")))
            if calibrated and motion and motion.get("valid") and not physical_absorbing else None)
        pose = decoded.get("pose")
        drift = math.dist(pose["position"], latest["pose"]["position"]) if (
            calibrated and decoded.get("valid") and pose and latest and not physical_absorbing) else None
        values["position_drift"].append(drift)
        for method in ("model", "persistence"):
            rgb = step.get("rgb", {}).get(method, {}) if step else {}
            for metric in ("image_mse", "object_mask_iou", "centroid_error_pixels"):
                values[method + "_" + metric].append(rgb.get(metric))
    nested = {key: _nested(value) for key, value in values.items()}
    record = {"anchor": spec["id"], "shape": spec["shape"], "family": spec["family"], "candidate": candidate,
        "status": row.get("status", "missing_branch") if row else "missing_branch", "horizon": horizon,
        "actual_available": valid_truth, "actual_coverage": truth["coverage"] if valid_truth else None,
        "contact": contact, "physical_absorbing": physical_absorbing, "noise": noise_rows,
        "noise_statistics": nested, **{key: item["mean"] for key, item in nested.items()}}
    record["actual_rotation"] = (wrapped_degrees(truth["pose"]["angle"]-latest["pose"]["angle"])
                                  if valid_truth and latest else None)
    record["coverage_error"] = _mean([abs(value-truth["coverage"]) for value in values["score"]]) if valid_truth and record["score"] is not None else None
    record["mean_score_coverage_error"] = abs(record["score"]-truth["coverage"]) if valid_truth and record["score"] is not None else None
    record["persistence_coverage_error"] = abs(latest["coverage"]-truth["coverage"]) if valid_truth and latest else None
    record["coverage_bias"] = record["score"]-truth["coverage"] if valid_truth and record["score"] is not None else None
    record["near_goal"] = bool(valid_truth and abs(truth["coverage"]-.95) <= .05 + 1e-12)
    return record


def _ranking(records, calibrated, shape, horizon):
    grouped = defaultdict(dict)
    for row in records:
        grouped[row["anchor"]][row["candidate"]] = row
    ranks = []
    for anchor, candidates in grouped.items():
        rows = [candidates[name] for name in CANDIDATES]
        actual = [row["actual_coverage"] for row in rows]
        predicted = [row["score"] if calibrated else None for row in rows]
        if all(_number(value) for value in actual):
            result = rank_scores(actual, predicted)
            result["unknown_actual_pairs"] = 0
        else:
            eligible = sum(_number(actual[i]) and _number(actual[j]) and abs(actual[i]-actual[j]) >= .05-1e-12
                           for i in range(4) for j in range(i+1, 4))
            known_pairs = sum(_number(actual[i]) and _number(actual[j]) for i in range(4) for j in range(i+1, 4))
            result = dict(eligible_pairs=eligible, correct_pairs=0, selected=None, selection_abstention=True,
                regret=None, regret_conservative=1.0, hold_regret=None, uniform_regret=None, actual_range=None,
                optimism=None, unknown_actual_pairs=6-known_pairs)
        result.update(anchor=anchor, actual=actual, predicted=predicted,
                      selected_candidate=None if result["selected"] is None else CANDIDATES[result["selected"]])
        ranks.append(result)
    eligible = sum(row["eligible_pairs"] for row in ranks)
    correct = sum(row["correct_pairs"] for row in ranks)
    distinguishable = sum(row["eligible_pairs"] > 0 for row in ranks)
    anchor_accuracy = _mean([row["correct_pairs"]/row["eligible_pairs"] for row in ranks if row["eligible_pairs"]])
    regret = _mean([row["regret_conservative"] for row in ranks])
    hold, uniform = (_mean([row[key] for row in ranks]) for key in ("hold_regret", "uniform_regret"))
    minimum_anchors, minimum_pairs = (6, 12) if shape == "L" else (2, 4)
    missing_truth = sum(row["unknown_actual_pairs"] for row in ranks)
    enough = distinguishable >= minimum_anchors and eligible >= minimum_pairs and not missing_truth
    passed = (anchor_accuracy is not None and anchor_accuracy >= .75 and regret is not None and regret < .03
              and hold is not None and uniform is not None and regret < hold and regret < uniform)
    return {"anchors": ranks, "anchor_count": len(ranks), "distinguishable_anchors": distinguishable,
        "eligible_pairs": eligible, "correct_pairs": correct, "invalid_pairs_count_as_incorrect": True,
        "unknown_actual_pairs": missing_truth, "anchor_equal_pair_accuracy": anchor_accuracy if calibrated else None,
        "pooled_pair_accuracy_diagnostic": correct/eligible if eligible and calibrated else None,
        "mean_selected_endpoint_regret_conservative": regret if calibrated else None,
        "conservative_abstention_bound_upper": regret,
        "mean_observed_selected_regret": _mean([r["regret"] for r in ranks]) if calibrated else None,
        "selection_abstentions": sum(r["selection_abstention"] for r in ranks),
        "selection_abstention_rate": sum(r["selection_abstention"] for r in ranks)/len(ranks) if ranks else None,
        "abstention_regret_bound": [0, 1], "missing_regret_gate_value": 1,
        "hold_mean_regret": hold, "uniform_candidate_expectation_mean_regret": uniform,
        "constant_reward_selects": "hold", "selected_optimism": _mean([r["optimism"] for r in ranks]),
        "gate": _screen(calibrated, enough, passed) if horizon == 4 else {
            "status": "unavailable", "reason": "engineering ranking screen applies to H4; H1 is diagnostic"}}


def summarize(report):
    """Return JSON-safe H1/H4 scores with fixed-plan, anchor-level denominators."""
    lookup = {(r.get("anchor"), r.get("candidate")): r for r in report.get("branches", [])}
    output = {"kind": "tdyssey.frozen_wm_summary.v1.2", "aggregation_unit": "anchor",
        "noise_is_independent_physics": False, "expected_noise_seeds": list(NOISE_SEEDS),
        "contact_head": report.get("contact_head", {"status": "unavailable", "reason": "GPC has no contact head"}),
        "contact_proxy_is_solver_contact": False, "training_authorized": False,
        "contact_stratum_definition": "any real solver contact by H4, including in H1 summaries; not instantaneous H1 contact",
        "persistence_state_baseline": "Privileged last true pose held fixed as an offline reference; available independently of model decoding",
        "metric_notes": ["H4 solver-contact strata are fixed from shared physical branches.",
            "Image motion after real success absorption is excluded because action sequences differ; counts are explicit.",
            "Coverage scores use model-only predicted stopping/absorption; no future truth repairs a score.",
            "Observed means include explicit missing counts; missing predictions block passing measurement screens."], "by_shape": {}}
    for shape in ("L", "T"):
        calibrated = report.get("calibration_gate", {}).get(shape, {}).get("passed") is True
        planned = [spec for spec in anchors() if spec["shape"] == shape]
        group = {"planned_anchors": len(planned), "planned_branches": len(planned)*4,
                 "calibration_passed": calibrated, "horizons": {}}
        contacts = [float(step.get("contacts", 0) > 0) for row in report.get("branches", []) if row.get("shape") == shape
                    for step in row.get("actual", []) if not step.get("absorbing")]
        marginal = _mean(contacts)
        group["actual_contact_constant_baseline"] = {
            "executed_future_controls": len(contacts), "contact_controls": sum(contacts), "marginal_probability": marginal,
            "brier": _mean([(value-marginal)**2 for value in contacts]) if marginal is not None else None,
            "meaning": "descriptive empirical marginal on executed future controls; no prospective GPC contact prediction"}
        for horizon in (1, 2, 3, 4):
            records = [_candidate(lookup.get((spec["id"], candidate)), spec, candidate, horizon, calibrated)
                       for spec in planned for candidate in CANDIDATES]
            metrics = {"expected_branches": len(records), "actual_available_branches": sum(r["actual_available"] for r in records),
                "missing_actual_branches": sum(not r["actual_available"] for r in records),
                "physical_absorbing_branches": sum(r["physical_absorbing"] for r in records),
                "predicted_absorbing_noise_rows": sum(n["predicted_absorbing"] for r in records for n in r["noise"]),
                "prediction_expected_noises": len(records)*len(NOISE_SEEDS),
                "prediction_valid_score_noises": sum(r["noise_statistics"]["score"]["valid_noises"] for r in records),
                "coverage": {"observed": _weighted(records, "coverage_error"),
                    "mean_score_error": _weighted(records, "mean_score_coverage_error"),
                    "per_noise_absolute_error": _weighted(records, "coverage_error"),
                    "persistence_error": _weighted(records, "persistence_coverage_error"),
                    "conservative_missing_error_one": _weighted(records, "coverage_error", conservative_missing=1),
                    "optimism": _weighted(records, "coverage_bias"),
                    "near_goal": _weighted([r for r in records if r["near_goal"]], "coverage_error"),
                    "noise_std": _weighted([{**r, "score_noise_std": r["noise_statistics"]["score"]["noise_std"]}
                                             for r in records], "score_noise_std")},
                "rgb": {key: _weighted(records, key) for key in (
                    "model_image_mse", "persistence_image_mse", "model_object_mask_iou", "persistence_object_mask_iou",
                    "model_centroid_error_pixels", "persistence_centroid_error_pixels")},
                "ranking": _ranking(records, calibrated, shape, horizon)}
            dynamics = [r for r in records if r["actual_available"] and not r["physical_absorbing"]]
            strata = {}
            for name, members in (("contact", [r for r in dynamics if r["contact"] is True]),
                                  ("noncontact", [r for r in dynamics if r["contact"] is False]),
                                  ("rotation_gt_5deg", [r for r in dynamics if _number(r["actual_rotation"]) and abs(r["actual_rotation"]) > 5])):
                strata[name] = {"anchors": len({r["anchor"] for r in members}), "branches": len(members),
                    **{key: _weighted(members, key) for key in ("position_error", "angle_error", "pusher_error", "position_drift",
                        "persistence_position_error", "persistence_angle_error", "rotation_direction",
                        "model_image_mse", "persistence_image_mse", "model_object_mask_iou", "persistence_object_mask_iou",
                        "model_centroid_error_pixels", "persistence_centroid_error_pixels")}}
            metrics["motion"] = strata
            if horizon == 4:
                contact, noncontact, rotation = (strata[k] for k in ("contact", "noncontact", "rotation_gt_5deg"))
                contact_min = (6, 8) if shape == "L" else (2, 2)
                rotation_min = 3 if shape == "L" else 2
                def value(row, key):
                    return row[key]["anchor_equal_mean"]
                coverage_mean = metrics["coverage"]["observed"]["anchor_equal_mean"]
                metrics["gates"] = {
                    "noncontact_drift": _screen(calibrated, noncontact["branches"] > 0,
                        value(noncontact, "position_drift") is not None and value(noncontact, "position_drift") <= 2,
                        noncontact["position_drift"]["missing_branches"] + metrics["missing_actual_branches"]),
                    "contact_motion": _screen(calibrated, contact["anchors"] >= contact_min[0] and contact["branches"] >= contact_min[1],
                        value(contact, "position_error") is not None
                        and value(contact, "position_error") < value(contact, "persistence_position_error"),
                        contact["position_error"]["missing_branches"] + metrics["missing_actual_branches"]),
                    "rotation": _screen(calibrated, rotation["anchors"] >= rotation_min and rotation["branches"] >= rotation_min,
                        value(rotation, "rotation_direction") is not None
                        and value(rotation, "rotation_direction") >= .8 and value(rotation, "angle_error") <= 5
                        and value(rotation, "angle_error") < value(rotation, "persistence_angle_error"),
                        rotation["angle_error"]["missing_branches"] + metrics["missing_actual_branches"]),
                    "coverage": _screen(calibrated, metrics["actual_available_branches"] == len(records),
                        coverage_mean is not None and coverage_mean <= .05,
                        metrics["coverage"]["observed"]["missing_branches"]),
                    "ranking": metrics["ranking"]["gate"],
                    "contact_probability": {"status": "unavailable", "reason": "GPC has no direct contact-probability head; geometry proximity is a proxy"}}
                for name, stratum in (("noncontact_drift", noncontact), ("contact_motion", contact), ("rotation", rotation)):
                    key = "position_drift" if name == "noncontact_drift" else "position_error" if name == "contact_motion" else "angle_error"
                    metrics["gates"][name].update(missing_model_in_stratum=stratum[key]["missing_branches"],
                                                  unassigned_physical_branches=metrics["missing_actual_branches"])
            # All four horizons keep RGB/proxy curves; H1/H4 also keep branch/noise tables.
            if horizon in (1, 4):
                metrics["branch_records"] = records
            group["horizons"][f"H{horizon}"] = metrics
        output["by_shape"][shape] = group
    return output


def make_previews(report, output_dir):
    """Render saved RGB grids, calibration labels and mean RGB error curves."""
    import numpy as np
    from PIL import Image, ImageDraw
    directory = Path(output_dir)
    previews = directory / "previews"
    previews.mkdir(exist_ok=True)
    written = []
    def frame(array):
        array = np.asarray(array)
        if array.shape == (3, 96, 96):
            array = np.moveaxis(array, 0, -1)
        if array.shape != (96, 96, 3):
            raise ValueError("Saved RGB array shape differs from 96x96x3")
        if array.dtype != np.uint8:
            array = np.nan_to_num(np.clip(array, 0, 1), nan=0, posinf=1, neginf=0)*255
        return Image.fromarray(array.astype(np.uint8), "RGB")
    def save(image, name):
        path = previews / name
        with path.open("xb") as stream:
            image.save(stream, format="PNG")
        written.append(str(path.relative_to(directory)))
    for spec in anchors():
        anchor = spec["id"]
        canvas = Image.new("RGB", (800, 970), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{anchor} {spec['shape']} / {spec['family']} | display clips RGB to [0,1]", fill="black")
        for column, label in enumerate(("Latest real history", "Physical H4 (absorbing if stopped)", "GPC raw H4 / seed 5478", "Absolute RGB error [0,1]")):
            draw.text((column*200+5, 35), label, fill="black")
        for index, candidate in enumerate(CANDIDATES):
            y = 58 + index*226
            draw.text((5, y), candidate, fill="black")
            physical_path = directory / f"{anchor}-{candidate}-physical.npz"
            predicted_path = directory / f"{anchor}-{candidate}-noise-{NOISE_SEEDS[0]}.npz"
            images = [None]*4
            if physical_path.is_file():
                with np.load(physical_path, allow_pickle=False) as data:
                    latest = data["history"][-1].copy() if len(data["history"]) else None
                    actual = data["actual"][-1].copy() if len(data["actual"]) else None
                images[:2] = [frame(latest) if latest is not None else None, frame(actual) if actual is not None else None]
                if predicted_path.is_file():
                    with np.load(predicted_path, allow_pickle=False) as data:
                        predicted = data["raw"][-1].copy() if len(data["raw"]) else None
                    images[2:] = [frame(predicted) if predicted is not None else None,
                                  frame(np.abs(np.clip(predicted, 0, 1)-actual)) if predicted is not None and actual is not None else None]
            for column, image in enumerate(images):
                x = column*200
                canvas.paste((image if image is not None else Image.new("RGB", (96, 96), "#dddddd")).resize((192, 192)), (x+4, y+20))
                if image is None:
                    draw.text((x+10, y+100), "unavailable", fill="black")
        save(canvas, anchor + ".png")
    for shape in ("L", "T"):
        rows = [r for r in report.get("calibrations", []) if r.get("shape_name", r.get("shape")) == shape]
        canvas = Image.new("RGB", (1000, 930), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{shape}: independent render-only calibration; labels include privileged truth for error review", fill="black")
        for index, row in enumerate(rows[:12]):
            x, y = (index % 4)*250, 32 + (index//4)*296
            path = directory / (row["id"] + ".npz")
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    image = frame(data["rgb"])
                canvas.paste(image.resize((192, 192)), (x+29, y+20))
            prediction, truth = row["prediction"], row["truth"]
            draw.text((x+5, y), row["id"] + " " + ("valid" if prediction.get("valid") else "INVALID"), fill="black")
            draw.text((x+5, y+215), "reason: " + str(prediction.get("reason")), fill="black")
            def label(value):
                pose = value.get("pose")
                return "unavailable" if not pose else f"({pose['position'][0]:.1f},{pose['position'][1]:.1f}) {math.degrees(pose['angle']):.1f}deg"
            draw.text((x+5, y+233), "truth " + label(truth), fill="black")
            draw.text((x+5, y+251), "pred  " + label(prediction), fill="black")
            draw.text((x+5, y+269), "coverage true=" + f"{truth['coverage']:.3f}" + " pred=" + (
                f"{prediction['coverage']:.3f}" if _number(prediction.get("coverage")) else "NA"), fill="black")
        save(canvas, "calibration-" + shape + ".png")
    summary = report.get("summary") or summarize(report)
    canvas = Image.new("RGB", (1000, 520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Saved RGB MSE by control horizon; equal anchors, nested candidate/noise means; display clipping only", fill="black")
    for index, shape in enumerate(("L", "T")):
        left, top, width, height = 65 + 500*index, 65, 390, 330
        curves = {method: [summary["by_shape"][shape]["horizons"][f"H{h}"]["rgb"][method+"_image_mse"]["anchor_equal_mean"]
                           for h in (1, 2, 3, 4)] for method in ("model", "persistence")}
        maximum = max([v for values in curves.values() for v in values if v is not None] + [.001])*1.1
        draw.line((left, top, left, top+height, left+width, top+height), fill="black", width=2)
        draw.text((left, top-25), shape + " RGB error", fill="black")
        draw.text((left-55, top), f"{maximum:.4f}", fill="black")
        draw.text((left-30, top+height-5), "0", fill="black")
        for method, color in (("model", "#ba312d"), ("persistence", "#2867b2")):
            points = [(left+i*width/3, top+height-value/maximum*height) for i, value in enumerate(curves[method]) if value is not None]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
            draw.text((left, top+height+48+(0 if method == "model" else 20)), method + ": " + ", ".join("NA" if v is None else f"{v:.5f}" for v in curves[method]), fill=color)
        for i in range(4):
            draw.text((left+i*width/3-7, top+height+10), f"H{i+1}", fill="black")
    save(canvas, "rgb-error-curves.png")
    return written
