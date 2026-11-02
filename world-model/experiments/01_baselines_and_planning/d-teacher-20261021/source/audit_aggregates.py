"""Independently audit final matrix aggregates from raw reports, without physics.

Run after both matrices and all independent media audits finish. This helper
does not import the summarizer or simulator and never modifies source reports.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


BASE = Path(__file__).resolve().parent
COUNT_KEYS = ("restores", "replay_steps", "candidate_steps")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm(mapping):
    return {key.replace("\\", "/"): value for key, value in mapping.items()}


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-9, rel_tol=0)


def check(condition, description, issues):
    if not condition:
        issues.append(description)


def audit_matrix(version, inventory, audit_files):
    directory = BASE / f"matrix-v{version}"
    protocol, summary = load(directory / "protocol.json"), load(directory / "summary.json")
    issues, rows = [], []
    expected = {(scene, seed) for scene in protocol["requested_scene_seeds"] for seed in protocol["search_seeds"]}
    pairs = [(item["scene_request"], item["search_seed"]) for item in protocol["all_runs"]]
    check(len(pairs) == len(set(pairs)) == 8 and set(pairs) == expected, "Declared pairs omitted/duplicated", issues)
    check(protocol["status"] == "complete", "Protocol is not final", issues)
    check(not summary["provisional"], "Saved summary is still provisional", issues)
    sources = norm(protocol["source_sha256"])
    for name, digest in sources.items():
        path = directory / "source" / name
        check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Frozen copy mismatch: {name}", issues)
    totals = {key: 0 for key in ("actual_controls", "restores", "replay_controls", "forecast_controls", "planning_seconds", "contact_controls", "contact_count_sum", "reset_physics_substeps")}
    audit_costs = {"controls": 0, "reset_physics_substeps": 0, "seconds": 0}
    for item in protocol["all_runs"]:
        key = (item["scene_request"], item["search_seed"])
        prefix = f"{key[0]}/{key[1]}: "
        run_issues = []
        check(item["status"] in ("process_complete", "process_failed"), "Process still queued/running", run_issues)
        path = Path(item["output"]) / "report.json"
        report = load(path)
        steps, counts = report["steps"], report["counts"]
        check(report["status"] in ("complete", "stopped_early", "failed"), "Report is not final", run_issues)
        check(all(report.get(name) is False for name in ("training", "checkpoint_loaded", "torch_loaded")), "Unexpected training/checkpoint/Torch flag", run_issues)
        check(report["initial_state"] == inventory[key[0]]["initial_state"], "Initial snapshot differs from fixed inventory", run_issues)
        check(report["search_start_state"] == report["initial_state"], "Unexpected resume state", run_issues)
        check(counts.get("resume_replay_controls", 0) == 0, "Unexpected resume replay cost/history", run_issues)
        check(report["scene_seed"] == inventory[key[0]]["accepted_seed"], "Accepted scene differs", run_issues)
        source_union = {**norm(report["provenance"]["source_sha256"]), **norm(report.get("teacher_implementation", {}).get("source_sha256", {}))}
        check(all(sources.get(name) == digest for name, digest in source_union.items()), "Recorded source IDs differ from frozen protocol", run_issues)
        check(set(sources) - set(source_union) == ({"tdyssey/official.py"} if version == 1 else set()), "Unexpected missing source IDs", run_issues)
        config_map = {"steps": "steps", "seconds": "seconds_per_run", "horizon": "horizon", "population": "population", "generations": "generations", "elite_count": "elite_count", "max_evaluations": "max_evaluations_per_plan", "execute_steps": "execute_steps", "settle_steps": "settle_steps"}
        check(all(report["config"][name] == protocol[field] for name, field in config_map.items()), "Evaluator limits differ from protocol", run_issues)
        check(report["config"]["seed"] == key[0] and report["config"]["search_seed"] == key[1], "Scene/search seed differs", run_issues)
        if version == 2:
            check(report["teacher_implementation"]["version"] == 2, "Wrapper version differs", run_issues)
            check(all(decision.get("v2_config") == protocol["v2_config"] for decision in report["decisions"]), "Actual v2 config differs", run_issues)
        native = [index for index, step in enumerate(steps) if step["upstream_done"]]
        check(not native or native == [len(steps)-1], "Controls after native success", run_issues)
        check(bool(native) == report["success"], "Success flag differs from actual terminal rows", run_issues)
        check(all(step["upstream_done"] == (step["metrics"]["coverage"] > .95) for step in steps), "Native coverage rule differs", run_issues)
        check(counts["control_steps"] == len(steps) <= protocol["steps"], "Actual count/budget differs", run_issues)
        check([step["state"]["control_step"] for step in steps] == list(range(1, len(steps)+1)), "Noncontiguous actual control indices", run_issues)
        check(all(step["prediction_state_error"] <= 1e-7 for step in steps), "Selected prediction differs from actual saved state", run_issues)
        for metric in ("coverage", "iou"):
            peak = max([report["initial_metrics"][metric]] + [step["metrics"][metric] for step in steps])
            check(close(peak, report[f"max_{metric}"]), f"Peak {metric} differs from actual metrics", run_issues)
            check(close(report["final_metrics"][metric], steps[-1]["metrics"][metric]), f"Final {metric} differs from actual metrics", run_issues)
        attempts = report["planning_attempts"]
        for name in COUNT_KEYS:
            check(sum(attempt["counts"][name] for attempt in attempts) == report["search_counts"][name], f"Planning attempts lose {name}", run_issues)
        for decision in report["decisions"]:
            check(all(sum(candidate["counts"][name] for candidate in decision["candidates"]) == decision["counts"][name] for name in COUNT_KEYS), "Candidate counts lose completed/rejected/timeout work", run_issues)
        if report["status"] == "failed" or item.get("exit_code", 0):
            outcome = "protocol_error"
        elif report["success"]:
            outcome = "native_success"
        elif report["stop_reason"] == "wall_time_budget":
            outcome = "timeout"
        elif report["status"] == "complete" and report["stop_reason"] == "control_budget":
            outcome = "control_failure"
            check(len(steps) == protocol["steps"], "Control-budget failure stopped too early", run_issues)
        else:
            outcome = "protocol_error"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matched = [audit for audit_path, audit in audit_files if Path(audit.get("source_report", "")).resolve() == path.resolve()]
        check(bool(matched), "No independent replay/video audit", run_issues)
        for audit in matched:
            check(audit["status"] == "passed" and audit.get("source_report_sha256") == digest, "Replay audit failed/stale", run_issues)
            video = audit.get("video", {})
            check(video.get("verified") and Path(video.get("path", "")).is_file(), "Missing verified video", run_issues)
            audit_costs["controls"] += audit["counts"]["source_controls"] + audit["counts"].get("prefix_controls", 0)
            audit_costs["reset_physics_substeps"] += audit["counts"]["reset_physics_substeps"]
            audit_costs["seconds"] += audit["elapsed_seconds"]
        final = steps[-1]["state"]
        pose, goal = final["object_body"]["pose"], final["goal"]
        angle = goal["angle"]-pose["angle"]
        row = {"scene_request": key[0], "search_seed": key[1], "outcome": outcome,
               "actual_controls": len(steps), "restores": report["search_counts"]["restores"],
               "replay_controls": report["search_counts"]["replay_steps"], "forecast_controls": report["search_counts"]["candidate_steps"],
               "planning_seconds": sum(attempt["seconds"] for attempt in attempts),
               "contact_controls": sum(step["n_contacts"] > 0 for step in steps),
               "contact_count_sum": sum(step["n_contacts"] for step in steps),
               "reset_physics_substeps": counts["reset_physics_substeps"],
               "max_coverage": report["max_coverage"], "final_coverage": report["final_metrics"]["coverage"],
               "max_iou": report["max_iou"], "final_iou": report["final_metrics"]["iou"],
               "final_position_error_px": math.dist(pose["position"], goal["position"]),
               "final_angle_error_deg": math.degrees(abs(math.atan2(math.sin(angle), math.cos(angle)))),
               "completed_composite_candidates": sum(candidate.get("status") == "completed" and candidate.get("recover_translation", False) for decision in report["decisions"] for candidate in decision["candidates"]),
               "selected_composite_decisions": sum(decision["selected"].get("recover_translation", False) for decision in report["decisions"]),
               "completed_native_success_candidates": sum(candidate.get("status") == "completed" and candidate.get("native_success", False) for decision in report["decisions"] for candidate in decision["candidates"]),
               "selected_native_success_decisions": sum(decision["selected"].get("native_success", False) for decision in report["decisions"]),
               "selected_family_counts": dict(Counter(decision["selected"].get("family", decision["selected"]["kind"]) for decision in report["decisions"])),
               "initial_state": report["initial_state"], "issues": run_issues}
        saved = next((value for value in summary["rows"] if (value["scene_request"], value["search_seed"]) == key), {})
        for name in ("outcome", *totals, "max_coverage", "final_coverage", "max_iou", "final_iou", "final_position_error_px", "final_angle_error_deg"):
            actual, expected_value = row[name], saved.get(name)
            check(close(actual, expected_value) if isinstance(actual, (int, float)) and isinstance(expected_value, (int, float)) else actual == expected_value, f"Summary differs: {name}", run_issues)
        for name in totals:
            totals[name] += row[name]
        issues.extend(prefix+issue for issue in run_issues)
        rows.append(row)
    for name, total in totals.items():
        check(close(total, summary["totals"][name]), f"Aggregate summary differs: {name}", issues)
    totals["all_simulated_controls"] = totals["actual_controls"]+totals["replay_controls"]+totals["forecast_controls"]
    outcome_counts = Counter(row["outcome"] for row in rows)
    return {"version": version, "protocol_status": protocol["status"], "elapsed_concurrent_wall_seconds": protocol.get("elapsed_seconds"),
            "outcomes": {name: outcome_counts[name] for name in ("queued", "running", "control_failure", "timeout", "native_success", "protocol_error")}, "totals": totals,
            "independent_replay_costs_separate": audit_costs, "rows": rows, "issues": issues,
            "mean_final_coverage": mean(row["final_coverage"] for row in rows),
            "mean_final_iou": mean(row["final_iou"] for row in rows),
            "mean_final_position_error_px": mean(row["final_position_error_px"] for row in rows),
            "median_final_angle_error_deg": median(row["final_angle_error_deg"] for row in rows)}


def main():
    inventory = {row["requested_seed"]: row for row in load(BASE / "scene_inventory.json")["scenes"]}
    audits = [(path, load(path)) for version in (1, 2) for path in (BASE / f"videos-v{version}").rglob("audit.json")]
    matrices = [audit_matrix(version, inventory, audits) for version in (1, 2)]
    paired = []
    for first in matrices[0]["rows"]:
        second = next(row for row in matrices[1]["rows"] if (row["scene_request"], row["search_seed"]) == (first["scene_request"], first["search_seed"]))
        check(first["initial_state"] == second["initial_state"], f"V1/V2 initial mismatch: {first['scene_request']}/{first['search_seed']}", matrices[1]["issues"])
        row = {"scene_request": first["scene_request"], "search_seed": first["search_seed"], "v1_outcome": first["outcome"], "v2_outcome": second["outcome"]}
        for metric in ("final_coverage", "max_coverage", "final_iou", "max_iou", "final_position_error_px", "final_angle_error_deg", "actual_controls", "planning_seconds", "replay_controls", "forecast_controls"):
            row[f"v1_{metric}"], row[f"v2_{metric}"] = first[metric], second[metric]
            row[f"delta_{metric}"] = second[metric]-first[metric]
        paired.append(row)
    issues = [f"V{matrix['version']}: {issue}" for matrix in matrices for issue in matrix["issues"]]
    gallery = load(BASE / "gallery-validation.json")
    check(gallery["status"] == "passed" and gallery["cards"] == gallery["unique_cards"] == gallery["embedded_verified_videos"] == 16, "Gallery does not contain all 16 verified videos", issues)
    result = {"kind": "independent_raw_report_aggregate_audit", "status": "passed" if not issues else "failed",
              "training": False, "simulation_or_search_started": False,
              "scope": "all 16 predeclared V1/V2 runs including failures/timeouts; previous prototypes and capture/render costs excluded",
              "matrices": matrices, "paired_differences_v2_minus_v1": paired, "issues": issues}
    (BASE / "matrix-comparison-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (BASE / "matrix-comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    print(json.dumps({"status": result["status"], "matrices": [{key: value for key, value in matrix.items() if key != "rows"} for matrix in matrices], "issues": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
