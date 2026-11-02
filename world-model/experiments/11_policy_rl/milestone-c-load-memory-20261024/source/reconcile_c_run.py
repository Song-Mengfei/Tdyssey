"""Offline C closeout checks. Run only after the guarded training process exits.

Uses stdlib only; never imports the project runtime, torch, or a budget owner.
Optional --checkpoint-metadata accepts a JSON export made separately from small
delta files: {"checkpoints": [{"path": ..., "sha256": ..., "metadata": ...,
"base_identity": ..., "optimizer_steps": {"actor": int, "critic": int}}]}.
Tensor-valued metadata may be represented by primitive digest/shape summaries.
This helper cannot itself verify metadata inside a PyTorch archive.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHYSICAL = ("control", "plan", "reset")
GLOBAL = ("actor_step", "critic_step", "batch")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def argument(command, name):
    if command.count(name) != 1:
        raise ValueError("Expected exactly one declared " + name)
    return command[command.index(name) + 1]


def runtime_source(name):
    name = name.replace("\\", "/")
    return ((name.startswith("tdyssey/") and name != "tdyssey/policy_memory.py")
            or name in ("scripts/run_c_policy.py", "scripts/probe_c_policy_load.py",
                        "configs/policy_c.example.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-dir", type=Path, default=HERE / "train")
    parser.add_argument("--output", type=Path, default=HERE / "train-reconciliation.json")
    parser.add_argument("--checkpoint-metadata", type=Path)
    args = parser.parse_args()
    monitor_dir = args.monitor_dir.resolve()
    # Never mistake a live run or active ledger for a completed observation.
    if not (monitor_dir / "monitor-report.json").is_file():
        parser.error("Final monitor report is absent; wait for the guarded run to exit")
    declaration = read(monitor_dir / "declaration.json")
    command = declaration["command"]
    run_dir = Path(argument(command, "--run-dir")).resolve()
    registry = Path(argument(command, "--registry")).resolve()
    experiment = argument(command, "--experiment-id")
    config_path = Path(argument(command, "--config")).resolve()
    ledger_path = registry / experiment / "ledger.json"
    if (ledger_path.parent / "owner.lock").exists():
        parser.error("Experiment owner lock remains; do not inspect as a completed run or alter its authority")
    if args.output.exists():
        parser.error("Use a new --output; existing reconciliation evidence is immutable")

    issues, warnings = [], []
    def check(condition, message):
        if not condition:
            issues.append(message)
    def jsonl(path):
        rows = []
        if not path.exists():
            return rows
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except ValueError:
                issues.append(f"Malformed JSONL: {path.name}:{index}")
        return rows

    config, ledger, monitor = read(config_path), read(ledger_path), read(monitor_dir / "monitor-report.json")
    config_hash = canonical(config)
    check(ledger["experiment_id"] == experiment, "Ledger experiment identity differs")
    check(ledger["config"] == config and ledger["config_hash"] == config_hash, "Ledger/config canonical identity differs")
    check(sha(config_path) == declaration["config_file_sha256"], "Configuration bytes changed after launch")
    index = read(registry / "index.json")
    check(index["experiments"][experiment]["config_hash"] == config_hash, "Registry config binding differs")
    binding = read(run_dir / "c-experiment.json")
    check(binding["experiment_id"] == experiment and binding["config_hash"] == config_hash,
          "Output directory binding differs")
    source_checks = []
    for relative, expected in declaration["source_sha256"].items():
        path = ROOT / Path(relative.replace("\\", "/"))
        actual = sha(path) if path.is_file() else None
        source_checks.append({"path": relative, "launch_sha256": expected,
                              "current_sha256": actual, "runtime_source": runtime_source(relative)})
        if actual != expected:
            (issues if runtime_source(relative) else warnings).append("Source changed since launch: " + relative)

    reserved, completed = ledger["reserved"], ledger["completed"]
    pending_totals = {phase: dict.fromkeys(GLOBAL if phase == "global" else PHYSICAL, 0)
                      for phase in ("train", "development", "diagnostic", "global")}
    pending_storage = []
    for identifier, event in ledger["pending"].items():
        check(str(event["id"]) == identifier and not event["completed"], "Malformed pending event identity")
        if event["kind"] == "storage":
            pending_storage.append(event)
        else:
            bucket = "global" if event["kind"] in GLOBAL else event["phase"]
            pending_totals[bucket][event["kind"]] += event["amount"]
    for phase, values in reserved.items():
        for kind, value in values.items():
            check(type(value) is int and 0 <= completed[phase][kind] <= value <= config["budgets"][phase][kind],
                  f"Counter bound/ordering violation: {phase}/{kind}")
            check(value - completed[phase][kind] == pending_totals[phase][kind],
                  f"Pending reservations differ from counter delta: {phase}/{kind}")
    check(reserved["train"]["control"] <= 2048 and reserved["train"]["plan"] <= 384,
          "Physical hard cap exceeded")
    check(reserved["global"]["batch"] <= 24 and reserved["global"]["actor_step"] <= 72
          and reserved["global"]["critic_step"] <= 72, "Global hard cap exceeded")
    check(not ledger["evaluation_groups"], "An evaluation group was consumed during the training-only run")
    for phase in ("development", "diagnostic"):
        check(not any(reserved[phase].values()) and not any(completed[phase].values()),
              "Unexpected non-training operations: " + phase)
    for mode, counts in (("reserved", reserved), ("completed", completed)):
        for kind in PHYSICAL:
            shape_total = sum(value[mode][kind] for value in ledger["by_shape"].get("train", {}).values())
            check(shape_total == counts["train"][kind], f"T/L totals differ from ledger: {mode}/{kind}")
    if ledger["pending"]:
        warnings.append("Incomplete durable reservations remain; no refund or automatic retry is authorized")

    rows = jsonl(run_dir / "training.jsonl")
    batches = [row for row in rows if row.get("event") == "batch"]
    passes = [row for row in rows if row.get("event") == "ppo_pass"]
    gates = [row for row in rows if row.get("event") == "first_batch_gate"]
    check([row["batch"] for row in batches] == list(range(1, len(batches) + 1)), "Batch log order/sequence differs")
    pass_counts = Counter(row["batch"] for row in passes)
    summaries = []
    for row in batches:
        update = row["update"]
        count = update.get("completed_passes", 0)
        check(pass_counts[row["batch"]] == count, "PPO pass logs differ from update report")
        check(0 <= count <= row["budget_allowed_passes"] <= row["requested_passes"] <= 3,
              "PPO pass limits differ")
        if row["batch"] == 1:
            check(row["requested_passes"] == 1, "First batch requested more than one pass")
        entry = update.get("entry", {})
        if not update.get("interrupted"):
            check(update.get("frozen_unchanged") is True, "Frozen preservation evidence missing")
            check(entry.get("transition_count") == row["collection"]["plans"] * 100, "Entry transition count differs")
            check(type(entry.get("max_logprob_error")) in (int, float)
                  and math.isfinite(entry["max_logprob_error"]) and entry["max_logprob_error"] <= 1e-4,
                  "Entry behavior probability check failed")
        after = [item.get("after", {}) for item in update.get("passes", [])]
        summaries.append({"batch": row["batch"], "controls": row["collection"]["controls"],
            "plans": row["collection"]["plans"], "completed_passes": count,
            "entry": entry, "after": after, "stop_reason": update.get("stop_reason"),
            "critic_mse": [item.get("critic_mse") for item in update.get("passes", [])],
            "critic_explained_variance": [item.get("critic_explained_variance") for item in update.get("passes", [])]})
    reports = sorted(run_dir.glob("train-report-*.json"))
    report, result = None, None
    if reports:
        report = read(reports[-1]); result = report["result"]
        check(report["config_hash"] == config_hash and report["experiment_id"] == experiment,
              "Training report identity differs")
        for name, kind in (("batches", "batch"), ("actor_steps", "actor_step"), ("critic_steps", "critic_step")):
            check(result[name] == completed["global"][kind], "Reported updater total differs: " + name)
        check(result["controls"] == completed["train"]["control"], "Reported control total differs")
        check(result["counts"] == {"reserved": reserved, "completed": completed}, "Final operation snapshots differ")
        check(len(passes) == result["actor_steps"] == result["critic_steps"], "Completed pass/optimizer totals differ")
        check(len(batches) == result["batches"], "Completed batch log count differs")
        check(sum(row["collection"]["controls"] for row in batches) == result["controls"], "Batch controls do not sum to final total")
        collector = result["collector"]
        for shape in ("T", "L"):
            stats = collector["by_shape"][shape]
            authority = ledger["by_shape"].get("train", {}).get(shape, {}).get("completed", {})
            for stat, kind in (("controls", "control"), ("sampled_plans", "plan"), ("reset_completed", "reset")):
                check(stats[stat] == authority.get(kind, 0), f"Collector/ledger differ: {shape}/{stat}")
            check(stats["successes"] + stats["timeouts"] <= stats["episodes_started"], "Episode termination counts exceed starts")
            check(stats["episodes_started"] == stats["reset_completed"], "Episode starts differ from completed resets")
            diagnostics = result["diagnostics_by_shape"][shape]
            check(diagnostics["controls"] == stats["controls"], "Per-shape diagnostic controls differ")
        check(result.get("first_batch_gate", {}).get("status") == "passed", "First in-budget batch gate did not pass")
    else:
        issues.append("No final training report: reconcile ledger/failure evidence only")
    if not gates or gates[0].get("status") != "passed":
        warnings.append("No passing first-batch gate in the append-only log")

    expected_hashes = {}
    if result and result.get("last_checkpoint"):
        saved = result["last_checkpoint"]
        expected_hashes[str(Path(saved["path"]).resolve())] = saved
    marker = ledger.get("recovery_boundary")
    if marker:
        expected_hashes[str(Path(marker["path"]).resolve())] = marker
    exported = {}
    if args.checkpoint_metadata:
        exported = {str(Path(item["path"]).resolve()): item
                    for item in read(args.checkpoint_metadata)["checkpoints"]}
    checkpoints = []
    for item in ledger["checkpoint_files"]:
        path = Path(item["path"])
        if not path.is_file():
            continue  # Recorded immutable files may have been safely retired.
        actual = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}
        check(actual["bytes"] == item["bytes"], "Checkpoint registered byte count differs: " + path.name)
        expected = expected_hashes.get(actual["path"])
        if expected:
            check(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"],
                  "Checkpoint report/boundary digest differs: " + path.name)
        actual["report_or_boundary_hash_verified"] = expected is not None
        external = exported.get(actual["path"])
        actual["metadata_verified"] = False
        if external:
            metadata = external["metadata"]
            check(external["sha256"] == actual["sha256"], "Metadata export identifies different checkpoint bytes")
            check(metadata["kind"] == "tdyssey-C" and metadata["experiment_id"] == experiment
                  and metadata["config_hash"] == config_hash and metadata["config"] == config,
                  "Checkpoint metadata provenance differs")
            check(external["base_identity"] == config["base"], "Checkpoint base identity differs")
            for owner in ("actor", "critic"):
                check(external["optimizer_steps"][owner] == metadata["updater"][owner + "_steps"],
                      "Exported optimizer state steps differ from updater metadata")
            actual.update(metadata_verified=True, metadata_sha256=canonical(metadata),
                          resumable_at_save=metadata["resumable"], terminal=metadata["terminal"])
        checkpoints.append(actual)
    check(len(checkpoints) <= config["budgets"]["max_checkpoints"], "Retained checkpoint cap exceeded")
    if any(not item["metadata_verified"] for item in checkpoints):
        warnings.append("Checkpoint internal metadata remains unverified until a separate small-delta weights_only JSON export is supplied")
    current_boundary = bool(marker and Path(marker["path"]).is_file()
        and marker["last_train_operation_id"] == ledger["last_train_operation_id"]
        and all(marker[mode][phase] == ledger[mode][phase]
                for mode in ("reserved", "completed") for phase in ("train", "global")))
    check(not current_boundary or not ledger["pending"], "A current boundary cannot have pending operations")
    if result and result["collector"]["active_shapes"]:
        check(not result.get("last_checkpoint", {}).get("resumable", False), "Active-episode tail is incorrectly marked resumable")

    # The final report itself is written after its budget snapshot. Compare
    # operation counters above; file/storage/event IDs legitimately advance.
    files = {}
    for directory in [ledger_path.parent, *(Path(path) for path in ledger["output_dirs"])]:
        for path in directory.rglob("*"):
            if path.is_file():
                files[str(path.resolve())] = path.stat().st_size
    files[str(registry / "index.json")] = (registry / "index.json").stat().st_size
    current_bytes = sum(files.values())
    pending_bytes = sum(max(0, event["amount"] - max(0, files.get(event.get("path"), 0)
                            - event.get("existing_bytes", 0))) for event in pending_storage)
    check(current_bytes + pending_bytes <= config["budgets"]["storage_bytes"], "Aggregate registered storage cap exceeded")
    preserved = read(ROOT / "artifacts/milestone-c-preflight-20261023/reset-run/report.json")["preserved_evidence_sha256"]
    preserved_b = {path: sha(path) == expected for path, expected in preserved.items()}
    check(all(preserved_b.values()), "Milestone B evidence changed")
    check(monitor.get("pid") != monitor.get("launcher_pid") and monitor.get("worker_exit_code") is not None,
          "Final monitor lacks the actual worker identity/exit result")
    outcome = {"schema_version": 1, "kind": "tdyssey.C.offline_reconciliation",
        "at_utc": datetime.now(timezone.utc).isoformat(), "integrity_checks_passed": not issues,
        "issues": issues, "warnings": warnings, "experiment_id": experiment, "config_hash": config_hash,
        "launch_declaration_sha256": sha(monitor_dir / "declaration.json"), "source_checks": source_checks,
        "ledger_path": str(ledger_path), "ledger_sha256": sha(ledger_path),
        "reserved": reserved, "completed": completed, "pending": ledger["pending"],
        "by_shape": ledger["by_shape"], "elapsed_wall_seconds": ledger["elapsed_wall_seconds"],
        "recent_events_are_complete_history": False,
        "evaluation_groups": ledger["evaluation_groups"], "monitor": monitor,
        "training_report": str(reports[-1]) if reports else None,
        "training_report_sha256": sha(reports[-1]) if reports else None,
        "stop_reason": result.get("stop_reason") if result else None,
        "target_2048_completed": bool(result and result["controls"] == 2048 and result["stop_reason"] == "control_target"),
        "first_batch_gate": result.get("first_batch_gate") if result else None,
        "collector": result.get("collector") if result else None,
        "diagnostics_by_shape": result.get("diagnostics_by_shape") if result else None,
        "batch_summaries": summaries, "completed_pass_reports": len(passes),
        "retained_checkpoints": checkpoints, "recovery_boundary_is_current": current_boundary,
        "resume_eligibility": "requires_current_exact_boundary_and_verified_checkpoint_metadata; never restore physical contact caches",
        "storage": {"current_bytes": current_bytes, "pending_bytes": pending_bytes,
                    "limit_bytes": config["budgets"]["storage_bytes"]},
        "preserved_B_evidence_unchanged": preserved_b,
        "base_digest_rehashed": False,
        "base_identity": config["base"],
        "limitations": ["No model or torch import, checkpoint reload, physical operation or ledger owner initialization occurred.",
                        "Recorded training diagnostics establish implementation/runtime behavior, not independent evaluation or policy improvement.",
                        "Absent internal metadata export is an explicit verification gap; this script never opens checkpoint tensors."]}
    payload = (json.dumps(outcome, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output.resolve()), "integrity_checks_passed": not issues,
                      "issues": issues, "warnings": warnings,
                      "target_2048_completed": outcome["target_2048_completed"]}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
