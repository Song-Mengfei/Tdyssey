"""Export saved frozen-model diagnostics; no decoder, simulator or model calls.

The full fixed 16 x 4 x 3 x 4 matrix is exported even after aborted branches.
Stored summary statistics are preferred; absent summaries use only the existing
pure report summarizer. Output must be separate from all immutable inputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("hold", "toward", "side", "away")
SEEDS = (5478, 5479, 5480)
FAMILIES = ("noncontact",)*3 + ("translation",)*3 + ("rotation_cw", "rotation_ccw", "rotation_ccw") + ("near_goal",)*3 + ("noncontact", "translation", "rotation", "near_goal")
MOTION_KEYS = ("position_error", "angle_error", "pusher_error", "coverage_error", "coverage_bias",
               "actual_rotation", "predicted_rotation", "rotation_direction_correct",
               "persistence_position_error", "persistence_angle_error", "persistence_coverage_error")
RGB_KEYS = ("image_mse", "object_mask_iou", "centroid_error_pixels", "object_pixels")
GATE_CRITERIA = {
    "noncontact_drift": "H4 noncontact position drift <=2 px",
    "contact_motion": "H4 contact motion error < persistence; L >=6 anchors/8 branches, T >=2/2",
    "rotation": "True rotation >5 deg: direction >=80%, angle MAE <=5 deg and < zero-rotation baseline; L >=3 anchors/branches, T >=2",
    "coverage": "H4 endpoint coverage MAE <=0.05; complete physical denominator",
    "ranking": "Eligible true difference >=0.05; anchor pair accuracy >=75%; mean regret <0.03 and <hold/uniform; L >=6 distinguishable anchors/12 pairs, T >=2/4",
    "contact_probability": "Direct contact head Brier < constant baseline; recall reported; GPC has no such head",
}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    finite = [float(v) for v in values if number(v)]
    return statistics.fmean(finite) if finite else None


def read_json(path, sources):
    path = Path(path).resolve()
    raw = path.read_bytes()
    sources[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        label = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, label + "."))
        elif isinstance(item, (list, tuple)):
            result[label] = json.dumps(item, separators=(",", ":"), allow_nan=False)
        else:
            result[label] = item
    return result


def write_csv(path, rows, fields=()):
    columns = list(fields)
    seen = set(columns)
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def step_at(steps, horizon):
    matches = [r for r in steps if r.get("horizon") == horizon]
    if len(matches) > 1:
        raise ValueError(f"Duplicate saved horizon {horizon}")
    return matches[0] if matches else None


def pose_fields(prefix, record):
    pose = record.get("pose") or {}
    position = pose.get("position") or [None, None]
    pusher = record.get("pusher") or [None, None]
    return {prefix+"x": position[0], prefix+"y": position[1],
            prefix+"angle_radians": pose.get("angle"), prefix+"pusher_x": pusher[0],
            prefix+"pusher_y": pusher[1]}


def planned_rows(report):
    """Export denominators from the frozen plan, not only successful entries."""
    lookup = {}
    for row in report.get("branches", []):
        key = row.get("anchor"), row.get("candidate")
        if key in lookup:
            raise ValueError(f"Duplicate physical branch: {key}")
        lookup[key] = row
    expected = {(f"{'L' if i < 12 else 'T'}{i:02d}", candidate)
                for i in range(16) for candidate in CANDIDATES}
    if set(lookup) - expected:
        raise ValueError("Report includes branches outside the frozen 64-branch plan")
    result = []
    for index in range(16):
        shape = "L" if index < 12 else "T"
        anchor = f"{shape}{index:02d}"
        calibrated = report.get("calibration_gate", {}).get(shape, {}).get("passed") is True
        for candidate in CANDIDATES:
            branch = lookup.get((anchor, candidate), {})
            h4 = step_at(branch.get("actual", []), 4)
            branch_contact = (any(s.get("contacts", 0) > 0 for s in branch.get("actual", []))
                              if h4 is not None else None)
            latest = branch.get("latest", {})
            predictions = {}
            for prediction in branch.get("predictions", []):
                if prediction.get("seed") in predictions:
                    raise ValueError(f"Duplicate prediction seed: {anchor}/{candidate}")
                predictions[prediction.get("seed")] = prediction
            for seed in SEEDS:
                prediction = predictions.get(seed, {})
                for horizon in range(1, 5):
                    actual = step_at(branch.get("actual", []), horizon)
                    forecast = step_at(prediction.get("steps", []), horizon)
                    truth = (actual or {}).get("truth", {})
                    decoded = (forecast or {}).get("decoded", {})
                    motion = (forecast or {}).get("motion") or {}
                    action = (actual or {}).get("action") or [None, None]
                    targets = branch.get("actions", [])
                    target = targets[horizon-1] if len(targets) >= horizon else [None, None]
                    score = (forecast or {}).get("score_coverage", decoded.get("coverage") if decoded.get("valid") else None)
                    score = score if calibrated and number(score) else None
                    missing = ("missing_branch" if not branch else
                               "missing_noise" if not prediction else
                               "missing_horizon" if forecast is None else
                               "failed_calibration" if not calibrated else
                               decoded.get("reason", "invalid_measurement") if score is None else "")
                    row = dict(anchor=anchor, shape=shape, family=branch.get("family", FAMILIES[index]),
                        candidate=candidate, noise_seed=seed, horizon=horizon,
                        branch_status=branch.get("status", "missing_branch"), branch_error=branch.get("error"),
                        branch_seed=branch.get("seed"), history_digest=branch.get("history_digest"),
                        actual_available=actual is not None, prediction_available=forecast is not None,
                        calibration_passed=calibrated, score_available=score is not None,
                        missing_reason=missing, measurement_valid=bool(decoded.get("valid")),
                        decoder_reason=decoded.get("reason"), measurement_gate=decoded.get("measurement_gate"),
                        score_coverage=score, decoded_coverage=decoded.get("coverage"),
                        predicted_absorbing=(forecast or {}).get("predicted_absorbing"),
                        actual_coverage=truth.get("coverage"), actual_success=(actual or {}).get("success"),
                        physical_absorbing=(actual or {}).get("absorbing"),
                        actual_contacts=(actual or {}).get("contacts"), branch_contact_by_H4=branch_contact,
                        actual_upstream_clipped_reward=(actual or {}).get("upstream_clipped_reward"),
                        action_x=action[0], action_y=action[1], proposed_target_x=target[0], proposed_target_y=target[1],
                        latest_coverage=latest.get("coverage"), decoded_contact_proxy=decoded.get("contact_proxy"),
                        fit_iou=decoded.get("fit_iou"), decoded_clipped_fraction=decoded.get("clipped_fraction"),
                        raw_min=prediction.get("raw_min"), raw_max=prediction.get("raw_max"),
                        raw_clipped_fraction=prediction.get("clipped_fraction"),
                        forecast_latency_seconds=prediction.get("latency_seconds"),
                        motion_valid=motion.get("valid"), **pose_fields("actual_", truth),
                        **pose_fields("decoded_", decoded), **pose_fields("latest_", latest))
                    for key in MOTION_KEYS:
                        row["motion_"+key] = motion.get(key)
                    for method in ("model", "persistence"):
                        rgb = (forecast or {}).get("rgb", {}).get(method, {})
                        for key in RGB_KEYS:
                            row[method+"_"+key] = rgb.get(key)
                    row["score_coverage_error"] = abs(score-truth["coverage"]) if score is not None and number(truth.get("coverage")) else None
                    row["score_coverage_bias"] = score-truth["coverage"] if score is not None and number(truth.get("coverage")) else None
                    row["near_goal_truth"] = abs(truth["coverage"]-.95) <= .05+1e-12 if number(truth.get("coverage")) else None
                    result.append(row)
    if len(result) != 768:
        raise AssertionError("Full frozen prediction matrix must contain 768 rows")
    return result


def branch_table(raw):
    grouped = defaultdict(list)
    for row in raw:
        grouped[row["anchor"], row["candidate"], row["horizon"]].append(row)
    result = []
    for (anchor, candidate, horizon), rows in grouped.items():
        first = rows[0]
        record = {key: first[key] for key in ("shape", "family", "branch_status", "branch_error",
            "actual_available", "actual_coverage", "physical_absorbing", "branch_contact_by_H4", "near_goal_truth")}
        record.update(anchor=anchor, candidate=candidate, horizon=horizon, expected_noises=3,
                      available_noise_predictions=sum(r["prediction_available"] for r in rows),
                      valid_score_noises=sum(r["score_available"] for r in rows))
        for metric in ("score_coverage", "score_coverage_error", "score_coverage_bias",
                       "motion_position_error", "motion_angle_error", "motion_pusher_error",
                       "model_image_mse", "persistence_image_mse", "model_object_mask_iou", "persistence_object_mask_iou"):
            values = [r[metric] for r in rows]
            count = sum(number(v) for v in values)
            record[metric+"_mean"] = mean(values) if count == 3 else None
            record[metric+"_observed_mean"] = mean(values)
            record[metric+"_valid_noises"] = count
            record[metric+"_noise_std"] = statistics.pstdev(values) if count == 3 else None
        result.append(record)
    return result


def hierarchy_tables(summary):
    motion, coverage, ranking, ranking_summary, gates = [], [], [], [], []
    for shape, group in summary.get("by_shape", {}).items():
        for horizon, result in group.get("horizons", {}).items():
            for stratum, values in result.get("motion", {}).items():
                for metric, value in values.items():
                    if isinstance(value, dict):
                        motion.append(dict(shape=shape, horizon=horizon, stratum=stratum, metric=metric,
                                           stratum_anchors=values.get("anchors"), stratum_branches=values.get("branches"), **flatten(value)))
            for metric, value in result.get("coverage", {}).items():
                coverage.append(dict(shape=shape, horizon=horizon, metric=metric, **flatten(value)))
            rank = result.get("ranking", {})
            ranking_summary.append(dict(shape=shape, horizon=horizon,
                                        **flatten({k: v for k, v in rank.items() if k != "anchors"})))
            for row in rank.get("anchors", []):
                item = dict(shape=shape, horizon=horizon, **{k: v for k, v in row.items() if k not in ("actual", "predicted")})
                for key in ("actual", "predicted"):
                    values = row.get(key, [])
                    for i, name in enumerate(CANDIDATES):
                        item[key+"_"+name] = values[i] if len(values) > i else None
                ranking.append(item)
            for name, gate in result.get("gates", {}).items():
                gates.append(dict(shape=shape, stage="engineering", horizon=horizon, gate=name,
                                  criterion=GATE_CRITERIA.get(name), **flatten(gate)))
    return dict(contact_rotation=motion, coverage=coverage, ranking=ranking,
                ranking_summary=ranking_summary, gates=gates)


def calibration_tables(report):
    rows, gates = [], []
    for row in report.get("calibrations", []):
        truth, prediction = row["truth"], row["prediction"]
        tc, pc = truth.get("coverage"), prediction.get("coverage")
        rows.append(dict(id=row["id"], shape=row.get("shape_name", row.get("shape")), seed=row.get("seed"),
                         truth_coverage=tc, predicted_coverage=pc, valid=prediction.get("valid"), reason=prediction.get("reason"),
                         truth_success=tc>.95 if number(tc) else None, predicted_success=pc>.95 if number(pc) else None,
                         coverage_bias=pc-tc if number(tc) and number(pc) else None,
                         near_095=abs(tc-.95) <= .05+1e-12 if number(tc) else None,
                         **pose_fields("truth_", truth), **pose_fields("predicted_", prediction)))
    for shape, gate in report.get("calibration_gate", {}).items():
        gates.append(dict(shape=shape, stage="calibration", horizon="render-only", gate="image_measurement",
                          status="pass" if gate.get("passed") else "fail",
                          criterion="Measurable >=95%; object/pusher position P90 <=5 px; angle P90 <=5 deg; coverage MAE <=0.03; success precision not established",
                          **flatten(gate)))
    return rows, gates


def resource_tables(run_dir, report, ledger_dir, sources):
    monitors = sorted(run_dir.rglob("monitor-report.json"))
    resources = []
    for path in monitors:
        monitor = read_json(path, sources)
        peaks = monitor.get("process_observed_peaks", {})
        resources.append(dict(phase=path.parent.relative_to(run_dir).as_posix(), source=str(path.resolve()),
            status=monitor.get("status"), actual_worker_pid=monitor.get("actual_worker_pid"),
            guard_reason=monitor.get("guard_reason"), worker_exit_code=monitor.get("worker_exit_code"),
            elapsed_seconds=monitor.get("elapsed_seconds"), samples=monitor.get("samples"),
            sampled_minimum_available_bytes=monitor.get("sampled_minimum_available_bytes"),
            sampled_minimum_commit_headroom_bytes=monitor.get("sampled_minimum_commit_headroom_bytes"),
            sampled_minimum_gpu_free_bytes=monitor.get("sampled_minimum_gpu_free_bytes"),
            **peaks))
    ledger = read_json(ledger_dir / "ledger.json", sources) if ledger_dir else report.get("budget", {})
    costs = []
    for name, limit in ledger.get("limits", {}).items():
        costs.append(dict(resource=name, limit=limit, reserved=ledger.get("reserved", {}).get(name),
                          completed=ledger.get("completed", {}).get(name), aborted=ledger.get("aborted", {}).get(name)))
    costs.extend([dict(resource="worker_report_wall_seconds", completed=report.get("wall_seconds")),
                  dict(resource="ledger_elapsed_seconds", completed=ledger.get("elapsed_seconds"), limit=ledger.get("wall_limit_seconds")),
                  dict(resource="guarded_phase_seconds_sum", completed=sum(r["elapsed_seconds"] for r in resources if number(r.get("elapsed_seconds")))),
                  dict(resource="policy_optimizer_updates", completed=report.get("policy_optimizer_updates")),
                  dict(resource="world_model_optimizer_updates", completed=report.get("world_model_optimizer_updates"))])
    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        costs.append(dict(resource=key, completed=report.get("model", {}).get(key)))
    return resources, costs


def display(value):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(display(v) for v in row) + " |" for row in rows)])


def summary_markdown(report, raw, tables, resources, costs):
    lines = ["# Frozen world-model diagnostic tables", "", f"Saved run status: `{report.get('status', 'unknown')}`.", "",
        f"The fixed export contains 768 anchor/candidate/noise/horizon rows; {sum(r['prediction_available'] for r in raw)} have saved forecast steps and {sum(r['score_available'] for r in raw)} have calibrated scores. Missing rows remain in the CSV.", "",
        "Noise repeats are nested inside physical branches; summaries weight anchors equally. Contact strata mean any real solver contact by H4. Image motion after real success absorption is excluded by the saved summary. Missing values are unavailable, never zero.", "",
        "## Calibration", "", markdown_table(["Shape", "Valid / total", "Position P90 px", "Pusher P90 px", "Angle P90 deg", "Coverage MAE", "Gate"],
            [[shape, f"{g.get('valid')} / {g.get('total')}", g.get("position_p90_pixels"), g.get("pusher_position_p90_pixels"),
              g.get("angle_p90_degrees"), g.get("coverage_mae"), "pass" if g.get("passed") else "fail"]
             for shape, g in report.get("calibration_gate", {}).items()]), "",
        "Calibration of continuous errors does not establish precise strict-success classification at coverage >0.95. Inspect calibration.csv for crossings and near-threshold errors.", "",
        "## H4 engineering gates", "", markdown_table(["Shape", "Gate", "Status", "Reason"],
            [[r["shape"], r["gate"], r.get("status"), r.get("reason", "")] for r in tables["gates"] if r.get("stage") == "engineering"]), "",
        "## Coverage error", "", markdown_table(["Shape", "Horizon", "Stratum", "Anchor mean MAE", "Valid / expected branches"],
            [[r["shape"], r["horizon"], r["metric"], r.get("anchor_equal_mean"), f"{r.get('valid_branches')} / {r.get('expected_branches')}"]
             for r in tables["coverage"] if r["horizon"] in ("H1", "H4") and r["metric"] in ("observed", "near_goal", "persistence_error")]), "",
        "## H4 contact and rotation", "", markdown_table(["Shape", "Stratum", "Metric", "Anchor mean", "Valid / expected branches"],
            [[r["shape"], r["stratum"], r["metric"], r.get("anchor_equal_mean"), f"{r.get('valid_branches')} / {r.get('expected_branches')}"]
             for r in tables["contact_rotation"] if r["horizon"] == "H4" and r["metric"] in
             ("position_error", "persistence_position_error", "angle_error", "persistence_angle_error", "rotation_direction", "position_drift")]), "",
        "## H4 ranking", "", markdown_table(["Shape", "Eligible pairs", "Anchor pair accuracy", "Conservative regret", "Hold regret", "Uniform regret", "Abstentions"],
            [[r["shape"], r.get("eligible_pairs"), r.get("anchor_equal_pair_accuracy"), r.get("mean_selected_endpoint_regret_conservative"),
              r.get("hold_mean_regret"), r.get("uniform_candidate_expectation_mean_regret"), r.get("selection_abstentions")]
             for r in tables["ranking_summary"] if r["horizon"] == "H4"]), "",
        "## Windows monitor phases", "", markdown_table(["Phase", "Worker PID", "Status", "Seconds", "Min RAM GiB", "Min commit GiB", "Min GPU GiB", "Peak working set GiB"],
            [[r["phase"], r.get("actual_worker_pid"), r.get("status"), r.get("elapsed_seconds"),
              *(r.get(key)/1024**3 if number(r.get(key)) else None for key in
                ("sampled_minimum_available_bytes", "sampled_minimum_commit_headroom_bytes", "sampled_minimum_gpu_free_bytes", "peak_working_set_bytes"))]
             for r in resources]), "",
        "Monitor minima/peaks are sampled observations. Initial and recovery phases remain separate; their elapsed sum is not a new physics or inference budget.", "",
        "## Costs", "", markdown_table(["Resource", "Reserved", "Completed", "Aborted", "Limit"],
            [[r["resource"], r.get("reserved"), r.get("completed"), r.get("aborted"), r.get("limit")] for r in costs]), "",
        "Exports read saved reports only. No rendering, decoder invocation, simulation, forecasting, optimizer update or scene selection occurs.", ""]
    return "\n".join(lines)


def export_results(run_dir, output_dir, ledger_dir=None, recompute_summary=False):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    ledger_dir = Path(ledger_dir).resolve() if ledger_dir else None
    for source in (run_dir, ledger_dir):
        if source and (output_dir == source or output_dir.is_relative_to(source) or source.is_relative_to(output_dir)):
            raise ValueError("Export output must be separate from run and ledger directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing export")
    sources = {}
    report = read_json(run_dir / "report.json", sources)
    summary = report.get("summary")
    if not summary or recompute_summary:
        sys.path.insert(0, str(ROOT))
        from tdyssey.frozen_wm_summary import summarize
        summary = summarize(report)
        summary_source = "recomputed from saved report by pure frozen_wm_summary.summarize"
    else:
        summary_source = "stored report.summary"
    if recompute_summary:
        report["summary"] = summary
    raw = planned_rows(report)
    tables = hierarchy_tables(summary)
    calibration, calibration_gates = calibration_tables(report)
    tables["gates"] = calibration_gates + tables["gates"]
    resources, costs = resource_tables(run_dir, report, ledger_dir, sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    if recompute_summary:
        with (output_dir / "reviewed-summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    datasets = {"raw_results": raw, "branch_summary": branch_table(raw), "calibration": calibration,
                "resources": resources, "costs": costs, **tables}
    for name, rows in datasets.items():
        write_csv(output_dir / (name + ".csv"), rows)
    with (output_dir / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(summary_markdown(report, raw, tables, resources, costs))
    manifest = dict(kind="tdyssey.frozen_wm_offline_export.v1", created_at=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run_dir), ledger_dir=str(ledger_dir) if ledger_dir else None,
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary_implementation_sha256=hashlib.sha256((ROOT / "tdyssey/frozen_wm_summary.py").read_bytes()).hexdigest(),
        run_status=report.get("status"), summary_source=summary_source, input_sources=sources,
        expected_prediction_rows=768, prediction_rows_with_saved_steps=sum(r["prediction_available"] for r in raw),
        calibrated_score_rows=sum(r["score_available"] for r in raw),
        row_counts={name: len(rows) for name, rows in datasets.items()},
        no_runtime_calls=True, additional_physics_controls=0, additional_forecast_images=0,
        outputs={p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(output_dir.iterdir()) if p.is_file()})
    # Input artifacts must remain byte-identical throughout export.
    for path, identity in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != identity["sha256"]:
            raise RuntimeError("Input artifact changed during offline export: " + path)
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path)
    parser.add_argument("--recompute-summary", action="store_true", help="Versioned offline recalculation; original reports remain unchanged")
    args = parser.parse_args()
    manifest = export_results(args.run_dir, args.output_dir, args.ledger_dir, args.recompute_summary)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "row_counts": manifest["row_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
