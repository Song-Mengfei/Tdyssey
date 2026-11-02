"""Snapshot all predeclared teacher runs using saved evidence only (stdlib).

Usage: python -B artifacts/d-teacher-20261021/summarize_matrix.py MATRIX_DIR
No simulator, Torch, training, replay, or search is launched. Re-run after audits
or processes finish. Outputs summary.json/.csv/.md inside MATRIX_DIR by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


COST_KEYS = ("restores", "replay_steps", "candidate_steps")
FINAL = {"native_success", "control_failure", "timeout", "protocol_error"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_counts(items):
    items = list(items)
    return {key: sum(item.get(key, 0) for item in items) for key in COST_KEYS}


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, abs_tol=1e-9, rel_tol=0)


def path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def source_map(mapping):
    """Source metadata uses repository-relative paths on either host OS."""
    return {name.replace("\\", "/").removeprefix("./"): value for name, value in mapping.items()}


def link(path, parent, label):
    if not path:
        return "-"
    return f"[{label}]({os.path.relpath(path, parent).replace(os.sep, '/')})"


def find_audits(root):
    found = {}
    for path in sorted(root.rglob("audit.json")):
        try:
            audit = read_json(path)
        except (OSError, ValueError):
            continue
        source = audit.get("source_report")
        if source:
            found.setdefault(path_key(source), []).append((path, audit))
    return found


def source_copy_audit(matrix, protocol):
    issues = []
    for name, expected in source_map(protocol.get("source_sha256", {})).items():
        path = matrix / "source" / name
        if not path.is_file():
            issues.append(f"Missing frozen source copy: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            issues.append(f"Frozen source copy differs: {name}")
    return {"status": "passed" if not issues else "failed", "issues": issues,
            "files": len(protocol.get("source_sha256", {})),
            "scope": "frozen source copy vs protocol; current working tree is not substituted"}


def summarize_run(item, protocol, scenes, audits, source_copy):
    output = Path(item["output"])
    path = output / "report.json"
    row = {"scene_request": item["scene_request"], "search_seed": item["search_seed"],
           "scene_role": "development" if item["scene_request"] == protocol.get("known_development_scene") else "predeclared_additional",
           "outcome": "queued", "process_status": item.get("status", "missing"),
           "report_status": None, "stop_reason": None, "success": False,
           "actual_controls": 0, "task_control": 0, "max_coverage": None, "final_coverage": None,
           "max_iou": None, "final_iou": None, "contact_controls": 0, "contact_count_sum": 0,
           "final_position_error_px": None, "final_angle_error_deg": None,
           "restores": 0, "replay_controls": 0, "forecast_controls": 0,
           "control_physics_substeps": 0, "reset_physics_substeps": 0, "resume_replay_controls": 0,
           "planning_seconds": 0, "evaluation_seconds": None, "process_seconds": item.get("process_seconds"),
           "deadline_attempt_count": 0, "deadline_attempt_seconds": 0,
           "deadline_attempt_counts": add_counts([]), "discarded_partial_candidate_counts": add_counts([]),
           "unreturned_timeout_attempt_counts": add_counts([]),
           "source_parity": "pending", "initial_parity": "pending", "config_parity": "pending",
           "v2_config_parity": "not_applicable", "source_fields_pending": [],
           "replay_audit_status": "pending", "audit_paths": [], "video_paths": [],
           "independent_audit_controls": 0, "independent_audit_reset_physics_substeps": 0,
           "independent_audit_physics_substeps": 0, "independent_audit_seconds": 0,
           "independent_audit_routes": [],
           "report_path": str(path.resolve()), "issues": [], "notes": [],
           "costs_complete": False, "provisional": True}
    active = item.get("status") in ("running", "queued")
    if item.get("status") == "running":
        row["outcome"] = "running"
    try:
        report = read_json(path)
        full_report_read = True
    except (OSError, ValueError) as exc:
        if active:
            if path.exists():
                row["notes"].append(f"Active report temporarily unreadable: {type(exc).__name__}")
            return row
        row["outcome"] = "protocol_error"
        row["issues"].append(f"Final report missing/unreadable: {type(exc).__name__}")
        # Preserve available final process evidence even if the full trace is lost.
        report = item.get("result", {})
        full_report_read = False
    row.update(report_status=report.get("status"), stop_reason=report.get("stop_reason"),
               success=bool(report.get("success")), accepted_scene_seed=report.get("scene_seed"),
               evaluation_seconds=report.get("evaluation_seconds"))
    steps = report.get("steps", [])
    counts = report.get("counts", {})
    row["actual_controls"] = counts.get("control_steps", len(steps))
    row["task_control"] = steps[-1].get("state", {}).get("control_step", row["actual_controls"]) if steps else report.get("search_start_state", {}).get("control_step", 0)
    final_state = steps[-1].get("state") if steps else report.get("search_start_state", report.get("initial_state"))
    if final_state:
        # Official D uses its body origin as mass center and has symmetry one.
        # Use the last actually executed state; candidate predictions are unused.
        pose, goal = final_state["object_body"]["pose"], final_state["goal"]
        row["final_position_error_px"] = math.dist(pose["position"], goal["position"])
        delta = goal["angle"] - pose["angle"]
        row["final_angle_error_deg"] = math.degrees(abs(math.atan2(math.sin(delta), math.cos(delta))))
    for key in ("control_physics_substeps", "reset_physics_substeps", "resume_replay_controls"):
        row[key] = counts.get(key, 0)
    metrics = [report.get("initial_metrics", {})] + [step.get("metrics", {}) for step in steps]
    final_metrics = report.get("final_metrics", metrics[-1])
    for key in ("coverage", "iou"):
        observed = [item[key] for item in metrics if key in item]
        peak = max(observed) if observed else report.get(f"max_{key}")
        row[f"max_{key}"] = peak
        row[f"final_{key}"] = final_metrics.get(key)
        if steps and report.get(f"max_{key}") is not None and not close(peak, report[f"max_{key}"]):
            row["issues"].append(f"Stored max_{key} differs from actual per-control metrics")
    row["contact_controls"] = sum(step.get("n_contacts", 0) > 0 for step in steps)
    row["contact_count_sum"] = sum(step.get("n_contacts", 0) for step in steps)
    # This is pusher direction at contact, not an inferred exact surface contact.
    contact_sectors = set()
    for step in steps:
        if step.get("n_contacts", 0) <= 0:
            continue
        state = step.get("state", {})
        obj, pusher = state.get("object_body", {}).get("pose", {}), state.get("pusher_body", {}).get("pose", {})
        if "position" in obj and "position" in pusher:
            angle = math.atan2(pusher["position"][1] - obj["position"][1], pusher["position"][0] - obj["position"][0]) - obj["angle"]
            contact_sectors.add(int(((angle + math.pi) % (2 * math.pi)) / (math.pi / 4)))
    row["object_frame_pusher_contact_sector_proxy"] = sorted(contact_sectors)
    attempts = report.get("planning_attempts", [])
    attempt_counts = add_counts(attempt.get("counts", {}) for attempt in attempts)
    search_counts = report.get("search_counts", attempt_counts)
    row.update(restores=search_counts.get("restores", 0), replay_controls=search_counts.get("replay_steps", 0),
               forecast_controls=search_counts.get("candidate_steps", 0),
               planning_seconds=sum(attempt.get("seconds", 0) for attempt in attempts))
    deadline = [attempt for attempt in attempts if attempt.get("timeout")]
    row["deadline_attempt_count"] = len(deadline)
    row["deadline_attempt_seconds"] = sum(attempt.get("seconds", 0) for attempt in deadline)
    row["deadline_attempt_counts"] = add_counts(attempt.get("counts", {}) for attempt in deadline)
    row["unreturned_timeout_attempt_counts"] = add_counts(attempt.get("counts", {}) for attempt in deadline if not attempt.get("completed"))
    row["discarded_partial_candidate_counts"] = add_counts(candidate.get("counts", {})
        for decision in report.get("decisions", []) for candidate in decision.get("candidates", []) if candidate.get("status") == "timed_out")
    final_report = report.get("status") in ("complete", "stopped_early", "failed")
    row["costs_complete"] = final_report and full_report_read and "search_counts" in report
    if "search_counts" in report and attempts and search_counts != attempt_counts:
        row["issues"].append("Final search_counts differ from all planning_attempts counts")
    if "steps" in report and row["actual_controls"] != len(steps):
        row["issues"].append("Reported actual controls differ from saved step rows")
    if row["task_control"] > protocol["steps"]:
        row["issues"].append("Actual episode exceeded declared control budget")
    expected_sources = source_map(protocol.get("source_sha256", {}))
    base_sources = source_map(report.get("provenance", {}).get("source_sha256", {}))
    implementation = report.get("teacher_implementation", {})
    implementation_sources = source_map(implementation.get("source_sha256", {}))
    conflicts = [name for name in set(base_sources) & set(implementation_sources)
                 if base_sources[name] != implementation_sources[name]]
    source_hashes = {**base_sources, **implementation_sources}
    mismatches = [name for name, value in source_hashes.items() if expected_sources.get(name) != value]
    # The v2 wrapper adds its two experiment-local source IDs only after the
    # unchanged evaluator finishes. A final evaluator save can precede that
    # append while the process is still running, so defer missing-ID errors.
    required_implementation_sources = {name for name in expected_sources if name.startswith("artifacts/")}
    missing_implementation = sorted(required_implementation_sources - set(source_hashes))
    row["source_fields_pending"] = missing_implementation if active else []
    row["source_parity"] = ("failed" if mismatches or conflicts or source_copy["status"] == "failed"
        or (missing_implementation and not active) else "pending" if missing_implementation else
        "passed" if source_hashes else "unknown")
    row["source_fields_not_in_report"] = sorted(set(expected_sources) - set(source_hashes))
    if row["source_parity"] == "failed":
        row["issues"].append(f"Frozen source provenance mismatch: values={mismatches}, conflicts={conflicts}, missing final implementation IDs={missing_implementation if not active else []}")
    scene = scenes.get(item["scene_request"])
    if scene and "initial_state" in report:
        row["initial_parity"] = "passed" if report["initial_state"] == scene["initial_state"] and report.get("scene_seed") == scene["accepted_seed"] else "failed"
        if row["initial_parity"] == "failed":
            row["issues"].append("Initial full snapshot/accepted seed differs from predeclared inventory")
    expected_config = {"seed": item["scene_request"], "search_seed": item["search_seed"], "steps": protocol["steps"],
        "seconds": protocol["seconds_per_run"], "horizon": protocol["horizon"], "population": protocol["population"],
        "generations": protocol["generations"], "elite_count": protocol["elite_count"],
        "max_evaluations": protocol["max_evaluations_per_plan"], "execute_steps": protocol["execute_steps"],
        "settle_steps": protocol["settle_steps"], "resume_step": 0, "resume_report": None}
    config_bad = [name for name, expected in expected_config.items() if report.get("config", {}).get(name) != expected]
    row["config_parity"] = "failed" if config_bad else "passed"
    if config_bad:
        row["issues"].append(f"Declared evaluator configuration mismatch: {config_bad}")
    if "v2_config" in protocol:
        decisions = report.get("decisions", [])
        actual_v2_configs = [decision.get("v2_config") for decision in decisions]
        if "v2_config" in implementation:
            actual_v2_configs.append(implementation["v2_config"])
        if actual_v2_configs:
            row["v2_config_parity"] = "passed" if all(value == protocol["v2_config"] for value in actual_v2_configs) else "failed"
        else:
            row["v2_config_parity"] = "pending" if active else "failed"
        if row["v2_config_parity"] == "failed":
            row["config_parity"] = "failed"
            row["issues"].append("Actual v2 configuration missing or differs from declared v2_config")
        elif row["v2_config_parity"] == "pending" and row["config_parity"] == "passed":
            row["config_parity"] = "pending"
    if any(report.get(name) is True for name in ("training", "checkpoint_loaded", "torch_loaded")):
        row["issues"].append("Training/checkpoint/Torch unexpectedly active")
    if implementation.get("training") is True:
        row["issues"].append("Teacher implementation unexpectedly reports training")
    native_rows = [index for index, step in enumerate(steps) if step.get("upstream_done", step.get("native_done", False))]
    if any(bool(step.get("upstream_done", step.get("native_done", False))) != (step.get("metrics", {}).get("coverage", 0) > .95) for step in steps):
        row["issues"].append("Native terminal flag differs from >95% actual coverage")
    if native_rows and native_rows != [len(steps) - 1]:
        row["issues"].append("Controls continued after native success")
    if final_report and bool(native_rows) != row["success"]:
        row["issues"].append("Success flag differs from saved native terminal rows")
    if report.get("status") == "failed" or item.get("exit_code", 0) != 0:
        row["issues"].append(f"Run/process error: {report.get('error', item.get('exit_code'))}")
    if row["success"]:
        row["outcome"] = "native_success"
    elif report.get("status") == "stopped_early" and row["stop_reason"] == "wall_time_budget":
        row["outcome"] = "timeout"
    elif report.get("status") == "complete" and row["stop_reason"] == "control_budget":
        row["outcome"] = "control_failure"
        if row["task_control"] != protocol["steps"]:
            row["issues"].append("Completed control-budget run ended before declared budget")
    elif not final_report:
        row["outcome"] = "running"
    else:
        row["issues"].append(f"Unexpected final stop: {report.get('status')}/{row['stop_reason']}")
    matched_audits = audits.get(path_key(path), [])
    for audit_path, audit in matched_audits:
        row["audit_paths"].append(str(audit_path.resolve()))
        audit_counts = audit.get("counts", {})
        row["independent_audit_controls"] += audit_counts.get("source_controls", 0) + audit_counts.get("prefix_controls", 0)
        row["independent_audit_reset_physics_substeps"] += audit_counts.get("reset_physics_substeps", 0)
        row["independent_audit_physics_substeps"] += audit_counts.get("replayed_control_physics_substeps", 0)
        row["independent_audit_seconds"] += audit.get("elapsed_seconds", 0)
        if audit.get("route"):
            row["independent_audit_routes"].append(audit["route"])
        if full_report_read and audit.get("source_report_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != audit["source_report_sha256"]:
            row["issues"].append(f"Independent audit refers to an older report snapshot: {audit_path}")
        if audit.get("video", {}).get("path"):
            row["video_paths"].append(audit["video"]["path"])
        if audit.get("status") in ("failed", "mismatch"):
            row["issues"].append(f"Independent replay audit {audit.get('status')}: {audit_path}")
    if matched_audits:
        row["replay_audit_status"] = "passed" if all(audit.get("status") == "passed" for _, audit in matched_audits) else "pending" if all(audit.get("status") in ("passed", "running") for _, audit in matched_audits) else "failed"
    for video in sorted(output.glob("*.mp4")):
        row["video_paths"].append(str(video.resolve()))
    if row["issues"]:
        row["outcome"] = "protocol_error"
    row["provisional"] = not final_report or active
    return row


def write_outputs(matrix, prefix, inventory_path, audit_root):
    protocol_path = matrix / "protocol.json"
    protocol = read_json(protocol_path)
    inventory = read_json(inventory_path) if inventory_path.is_file() else {}
    scenes = {scene["requested_seed"]: scene for scene in inventory.get("scenes", [])}
    copy_audit = source_copy_audit(matrix, protocol)
    audits = find_audits(audit_root)
    declared = protocol.get("all_runs", [])
    runs = {(item["scene_request"], item["search_seed"]): item for item in declared}
    matrix_issues = []
    if len(runs) != len(declared):
        matrix_issues.append("Duplicate predeclared run rows")
    pairs = [(scene, seed) for scene in protocol["requested_scene_seeds"] for seed in protocol["search_seeds"]]
    rows = []
    for scene, seed in pairs:
        item = runs.get((scene, seed))
        if item is None:
            item = {"scene_request": scene, "search_seed": seed, "status": "missing",
                    "output": str(matrix / f"scene-{scene}-search-{seed}")}
        rows.append(summarize_run(item, protocol, scenes, audits, copy_audit))
    if set(runs) != set(pairs):
        matrix_issues.append("Protocol rows differ from declared scene x search-seed Cartesian product")
    if protocol.get("training") is not False or protocol.get("checkpoint_loaded") is not False:
        matrix_issues.append("Protocol lacks explicit no-training/no-checkpoint flags")
    provisional = protocol.get("status") != "complete" or any(row["provisional"] for row in rows)
    totals = {name: sum(row.get(name) or 0 for row in rows) for name in (
        "actual_controls", "control_physics_substeps", "reset_physics_substeps", "resume_replay_controls", "contact_controls", "contact_count_sum",
        "restores", "replay_controls", "forecast_controls", "planning_seconds", "evaluation_seconds", "process_seconds", "deadline_attempt_count", "deadline_attempt_seconds")}
    for name in ("deadline_attempt_counts", "discarded_partial_candidate_counts", "unreturned_timeout_attempt_counts"):
        totals[name] = add_counts(row[name] for row in rows)
    audit_totals = {name: sum(row[name] for row in rows) for name in (
        "independent_audit_controls", "independent_audit_reset_physics_substeps",
        "independent_audit_physics_substeps", "independent_audit_seconds")}
    totals["all_simulated_controls"] = totals["actual_controls"] + totals["resume_replay_controls"] + totals["replay_controls"] + totals["forecast_controls"]
    totals["costs_complete"] = all(row["costs_complete"] for row in rows)
    counter = Counter(row["outcome"] for row in rows)
    outcomes = {name: counter[name] for name in ("queued", "running", "control_failure", "timeout", "native_success", "protocol_error")}
    summary = {"schema_version": 1, "kind": "all_predeclared_D_teacher_matrix_summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "matrix_version": protocol.get("version"),
        "protocol_path": str(protocol_path.resolve()), "protocol_status": protocol.get("status"),
        "inventory_path": str(inventory_path.resolve()), "provisional": provisional,
        "scope": protocol.get("scope"), "known_development_scene": protocol.get("known_development_scene"),
        "training": False, "checkpoint_loaded": False, "expected_runs": len(pairs), "outcome_counts": outcomes,
        "protocol_issues": matrix_issues, "frozen_source_copy_audit": copy_audit, "rows": rows, "totals": totals,
        "independent_audit_totals_separate": audit_totals,
        "cost_notes": ["All declared rows and all persisted work included, including failures/timeouts.",
            "Running or queued rows are lower bounds; an in-progress planning call is not yet saved.",
            "Deadline-attempt work includes completed candidates in that call; partial-candidate work is a subset, never added twice.",
            "Contact counts are actual logged contact-bearing control steps / n_contacts sums, not unique physical contact points.",
            "Pusher contact sectors are an object-frame direction proxy, not exact surface contact locations.",
            "Aggregate evaluation/process seconds sum runs, not elapsed concurrent wall time.",
            "Inventory captures, preview renders, earlier prototype runs and independent replay costs are separate from matrix execution totals."]}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = list(rows[0])
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    percent = lambda value: "-" if value is None else f"{100 * value:.2f}%"
    lines = [f"# D teacher matrix v{protocol.get('version', '?')} — {'provisional' if provisional else 'final'}", "",
        f"All {len(pairs)} predeclared runs; native success requires actual coverage >95% and immediate stop. Scope: {protocol.get('scope')}.", "",
        "| Scene | Search | Outcome | Controls | Peak coverage | Final coverage | Peak IoU | Final IoU | Position px | Angle deg | Contact controls | Replay | Forecast | Planning s | Parity S/I/C | Audit | Video |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for row in rows:
        audit_link = link(row["audit_paths"][-1] if row["audit_paths"] else None, prefix.parent, row["replay_audit_status"])
        video_link = link(row["video_paths"][-1] if row["video_paths"] else None, prefix.parent, "video")
        parity = "/".join(row[key] for key in ("source_parity", "initial_parity", "config_parity"))
        position = "-" if row["final_position_error_px"] is None else f"{row['final_position_error_px']:.2f}"
        angle = "-" if row["final_angle_error_deg"] is None else f"{row['final_angle_error_deg']:.2f}"
        lines.append(f"| {row['scene_request']} | {row['search_seed']} | {row['outcome']} | {row['actual_controls']} | {percent(row['max_coverage'])} | {percent(row['final_coverage'])} | {percent(row['max_iou'])} | {percent(row['final_iou'])} | {position} | {angle} | {row['contact_controls']} | {row['replay_controls']} | {row['forecast_controls']} | {row['planning_seconds']:.3f} | {parity} | {audit_link} | {video_link} |")
    lines += ["", f"Outcomes: `{json.dumps(outcomes, sort_keys=True)}`.", "",
        f"Total persisted costs: {totals['actual_controls']:,} actual controls, {totals['restores']:,} branch restores, {totals['replay_controls']:,} replay controls, {totals['forecast_controls']:,} forecast controls; {totals['planning_seconds']:.3f} summed planning seconds. " + ("Final costs." if totals["costs_complete"] else "Provisional lower bounds."), "",
        f"Deadline attempts: {totals['deadline_attempt_count']}; their full work: `{json.dumps(totals['deadline_attempt_counts'])}`; discarded partial candidates (subset): `{json.dumps(totals['discarded_partial_candidate_counts'])}`.", "",
        f"Frozen source copy: {copy_audit['status']}. S/I/C = source provenance / exact initial snapshot / declared evaluator configuration. Independent replay audit is separate from these static checks.", "",
        "Position/angle errors use the last actually executed D snapshot (body origin = mass center; symmetry one). Angle is the absolute wrapped goal-minus-object angle, in degrees. Running rows describe the latest saved actual state.", "",
        f"Independent replay costs so far (separate): {audit_totals['independent_audit_controls']:,} controls, {audit_totals['independent_audit_reset_physics_substeps']} reset substeps, {audit_totals['independent_audit_seconds']:.3f} summed seconds.", "",
        "Costs include failed and timed-out runs. Per-run evaluation/process time sums are not concurrent elapsed wall time. Preview captures and independent replay costs remain separate. Running reports omit work in the currently executing planning call.", ""]
    for row in rows:
        if row["issues"]:
            lines.append(f"- {row['scene_request']}/{row['search_seed']}: {'; '.join(row['issues'])}")
    for issue in matrix_issues + copy_audit["issues"]:
        lines.append(f"- Protocol: {issue}")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(prefix.with_suffix('.json')), "provisional": provisional,
                      "outcomes": outcomes, "protocol_issues": matrix_issues, "totals": totals}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    matrix = args.matrix.resolve()
    write_outputs(matrix, (args.output_prefix or matrix / "summary").resolve(),
                  (args.inventory or matrix.parent / "scene_inventory.json").resolve(),
                  (args.audit_root or matrix.parent).resolve())


if __name__ == "__main__":
    main()
