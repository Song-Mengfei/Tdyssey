"""Offline evidence reconciliation; invoke only after the paired worker exits.

Stdlib only. No torch/model/checkpoint loading, simulator calls, ledger owner or
runtime polling. Source, report and immutable artifact files are read directly;
the pure comparison helper is imported only for descriptive recomputation.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("base_training", "delta_training")
KINDS = ("control", "plan", "reset")
LIMITS = {"groups": 2, "control": 12000, "plan": 1520, "reset": 80,
          "wall_time_seconds": 7200, "storage_bytes": 536870912}
CONFIG_HASH = "0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3"
DELTA_SHA = "d3128c7fcfb1a23842b4863d96ae88b7f124e93d751b371b23696eb24d6703dd"
CASES = [{"shape": shape, "scene_seed": scene, "policy_seed": seed}
         for shape in ("T", "L") for scene in range(100000, 100005) for seed in (5478, 5479)]
_HASH_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    path = Path(path).resolve(); stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def argument(command, name):
    if command.count(name) != 1:
        raise ValueError("Expected exactly one declared " + name)
    return command[command.index(name) + 1]


def key(case):
    return tuple(case[name] for name in ("shape", "scene_seed", "policy_seed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-dir", type=Path, default=HERE / "monitor")
    parser.add_argument("--output", type=Path, default=HERE / "paired-reconciliation.json")
    args = parser.parse_args()
    meta = args.monitor_dir.resolve()
    if not (meta / "monitor-report.json").is_file():
        parser.error("Final monitor report is absent; do not invoke while evaluation runs")
    declaration = read(meta / "declaration.json"); command = declaration["command"]
    run = Path(argument(command, "--run-dir")).resolve()
    registry = Path(argument(command, "--registry")).resolve()
    account = argument(command, "--account-id")
    ledger_path = registry / account / "ledger.json"
    if (ledger_path.parent / "owner.lock").exists():
        parser.error("Evaluation owner lock remains; no completed-run reconciliation is permitted")
    if args.output.exists():
        parser.error("Use a new --output; existing acceptance evidence is immutable")
    issues, warnings = [], []
    def check(condition, message):
        if not condition:
            issues.append(message)
    ledger = read(ledger_path); monitor = read(meta / "monitor-report.json")
    result = read(run / "evaluation-pair-report.json")
    config_path = Path(argument(command, "--config")).resolve(); config = read(config_path)
    config_hash = hashlib.sha256(canonical(config)).hexdigest()
    binding = {"training_experiment_id": "c-tl-2048-20261024", "config_hash": CONFIG_HASH,
        "base_identity": config["base"], "delta_identity": {"bytes": 16003916, "sha256": DELTA_SHA},
        "case_digest": hashlib.sha256(canonical(CASES)).hexdigest(),
        "methods": list(METHODS), "max_controls": 300}
    check(config_hash == CONFIG_HASH and result["config"] == config, "Training configuration differs")
    check(sha(config_path) == declaration["config_file_sha256"], "Configuration bytes changed after launch")
    check(ledger["kind"] == "tdyssey.paired_evaluation_ledger" and ledger["evaluation_id"] == account,
          "Evaluation authority identity differs")
    check(ledger["binding"] == binding and ledger["binding_hash"] == hashlib.sha256(canonical(binding)).hexdigest(),
          "Evaluation artifact/case/method binding differs")
    check(ledger["limits"] == LIMITS, "Evaluation fixed limits differ")
    check(result["status"] == "completed" and result["training_steps"] == 0,
          "Paired result incomplete or declares training")
    check(result["evaluation_id"] == account and result["training_experiment_id"] == binding["training_experiment_id"]
          and result["config_hash"] == CONFIG_HASH, "Paired result provenance differs")
    check(not ledger["pending"], "Pending evaluation operation/write remains; no automatic retry is permitted")
    check(ledger["reserved"] == ledger["completed"] and set(ledger["reserved"]) == {"development"},
          "Evaluation reservations/completions differ or contain training accounts")
    for kind in KINDS:
        check(type(ledger["reserved"]["development"][kind]) is int
              and 0 <= ledger["reserved"]["development"][kind] <= LIMITS[kind], "Operation cap exceeded: " + kind)
    check(ledger["elapsed_wall_seconds"] <= 7200, "Evaluation wall account exceeded its limit")

    source_checks = []
    for relative, expected in declaration["source_sha256"].items():
        path = ROOT / Path(relative.replace("\\", "/")); current = sha(path) if path.is_file() else None
        source_checks.append({"path": relative, "launch_sha256": expected, "current_sha256": current})
        check(current == expected, "Source changed after evaluation launch: " + relative)
    index = read(registry / "index.json")
    check(index["evaluations"][account]["binding_hash"] == ledger["binding_hash"], "Registry evaluation binding differs")
    check(read(run / "evaluation-account.json")["binding_hash"] == ledger["binding_hash"], "Output binding differs")

    declared = read(run / "preserved-declared.json"); preserved = read(run / "preserved-before.json")
    check(declared == preserved and result["preserved_before_after_equal"] is True,
          "Original B/C preservation declarations differ")
    preservation = {"files_checked": 0, "trees_checked": 0, "changed_files": [], "changed_trees": []}
    for name, identity in declared["files"].items():
        path = Path(name); preservation["files_checked"] += 1
        if not path.is_file() or path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
            preservation["changed_files"].append(name)
    for name, expected in declared["trees"].items():
        actual = sorted(str(path.resolve()) for path in Path(name).rglob("*") if path.is_file())
        preservation["trees_checked"] += 1
        if actual != expected:
            preservation["changed_trees"].append(name)
    check(not preservation["changed_files"] and not preservation["changed_trees"],
          "Original training/base/B inventory files or directory trees changed")
    original_training = read(Path(argument(command, "--training-ledger")))
    check(original_training["completed"]["global"] == {"actor_step": 52, "critic_step": 52, "batch": 18}
          and original_training["reserved"] == original_training["completed"]
          and original_training["completed"]["train"]["control"] == 2048
          and not original_training["pending"] and not original_training["evaluation_groups"],
          "Original C account acquired new training/evaluation operations")
    base_path = Path(argument(command, "--base-checkpoint")); delta_path = Path(argument(command, "--checkpoint"))
    for path, identity in ((base_path, binding["base_identity"]), (delta_path, binding["delta_identity"])):
        check(path.stat().st_size == identity["bytes"] and sha(path) == identity["sha256"],
              "Pinned artifact bytes/digest differs: " + path.name)

    groups = [read(run / (method + ".json")) for method in METHODS]
    check(groups == result["groups"], "Standalone method reports differ from paired result")
    claims = sorted(ledger["evaluation_groups"].values(), key=lambda row: row["claim_id"])
    check(len(claims) == 2 and [row["method"] for row in claims] == list(METHODS)
          and all(row["completed"] for row in claims), "Two fixed group claims did not complete in order")
    expected_case_keys = [key(case) for case in CASES]
    aggregate = Counter(); physical_by_shape = {shape: Counter() for shape in ("T", "L")}
    group_summaries = []
    expected_events = []
    for method, report in zip(METHODS, groups):
        claim = ledger["evaluation_groups"][method]
        check(sha(run / (method + ".json")) == claim["report_sha256"], "Group marker report digest differs: " + method)
        check(report["method"] == method and report["status"] == "completed"
              and report["requested_case_count"] == report["completed_case_count"] == 20,
              "Method case matrix did not complete: " + method)
        check([key(case) for case in report["cases"]] == expected_case_keys, "Case identity/order differs: " + method)
        check(report["optimizer_steps"] == report["new_training_steps"] == 0 and report["training"] is False,
              "Method report contains training/optimizer operations")
        identity = {"base_sha256": binding["base_identity"]["sha256"],
                    "delta_sha256": DELTA_SHA if method == "delta_training" else None}
        check(report["checkpoint_identity"] == identity and report["sampler_mode"] == "training"
              and report["config_hash"] == CONFIG_HASH and report["max_controls"] == 300,
              "Method sampler/config/artifact identity differs")
        check(report["state_fingerprint_before"] == report["state_fingerprint_after"], "Weights changed during evaluation")
        check(report["frozen_fingerprint_before"] == report["frozen_fingerprint_after"], "Frozen subset changed during evaluation")
        check(report["isolation_verified"] is True and report["isolation_before"] == report["isolation_after"],
              "Global RNG, modes or scheduler were not restored")
        sums = Counter()
        for case in report["cases"]:
            check(case["status"] == "completed" and type(case["success"]) is bool, "Aborted/unstarted case in complete group")
            counts = case["counts"]; controls = counts["control_steps"]; plans = counts["planning_calls"]
            check(1 <= controls <= 300 and controls == counts["control_attempts"] == len(case["steps"]),
                  "Case physical control counters differ")
            check(plans == counts["planning_attempts"] == math.ceil(controls / 8)
                  and counts["denoising_steps"] == 100 * plans, "Case plan/denoising counts differ")
            check(counts["reset_attempts"] == counts["reset_physics_substeps"] == 2
                  and counts["control_physics_substeps"] == controls * 10, "Case reset/physics substeps differ")
            success_steps = [i for i, step in enumerate(case["steps"], 1) if step["coverage"] > .95]
            check(case["success"] == bool(success_steps)
                  and case["success_step"] == (success_steps[0] if success_steps else None),
                  "Strict first physical success accounting differs")
            if success_steps:
                check(success_steps[0] == controls, "Controls continued after success")
            for metric in ("coverage", "iou"):
                check(case["final_metrics"][metric] == case["steps"][-1][metric], "Final metric differs from last physical step")
                peak = max(case["initial_metrics"][metric], *(step[metric] for step in case["steps"]))
                check(case["peak_metrics"][metric] == peak, "Peak metric differs from physical observations")
            sums.update(counts)
            physical_by_shape[case["shape"]].update(control=controls, plan=plans)
            physical_by_shape["T"].update(reset=1); physical_by_shape["L"].update(reset=1)
            start = {**dict(zip(("shape", "scene_seed", "policy_seed"), key(case))),
                     "method": method, "status": "started", "event": "started"}
            expected_events.extend((start, {"event": "finished", **case}))
        check(dict(sums) == report["counts"], "Case counters do not sum to group counters")
        operation_totals = {"control": sums["control_steps"], "plan": sums["planning_calls"], "reset": sums["reset_attempts"]}
        check(claim["reserved"] == claim["completed_counts"] == operation_totals, "Group ledger/report operation totals differ")
        check(operation_totals["control"] <= 6000 and operation_totals["plan"] <= 760
              and operation_totals["reset"] == 40, "Per-method fixed caps differ")
        aggregate.update(operation_totals)
        group_summaries.append({"method": method, "counts": report["counts"], "success_count": report["success_count"],
            "success_fraction": report["success_fraction"], "per_shape": report["per_shape"], "resources": report["resources"]})
    check(dict(aggregate) == ledger["completed"]["development"], "Report totals differ from evaluation ledger")
    for shape, counts in physical_by_shape.items():
        check(ledger["by_shape"][shape]["reserved"] == ledger["by_shape"][shape]["completed"] == dict(counts),
              "Per-shape evaluation ledger/report totals differ: " + shape)
    events = [json.loads(line) for line in (run / "case-events.jsonl").read_text().splitlines()]
    check(events == expected_events and Counter(row["event"] for row in events) == {"started": 40, "finished": 40},
          "Durable case journal does not match 40 ordered starts and finishes")
    releases = result["model_releases"]
    # CUDA/cuDNN workspaces can remain allocated after the actual network dies.
    # The pinned FP32 official network has this verified parameter count.
    full_parameter_bytes = 262709056 * 4
    check([row["method"] for row in releases] == list(METHODS)
          and all(release["network_released"]
                  and 0 <= release["resources"]["allocated_bytes"] < full_parameter_bytes
                  and group["resources"]["allocated_bytes"] - release["resources"]["allocated_bytes"] >= full_parameter_bytes
                  for group, release in zip(groups, releases)),
          "Sequential model release was not established")
    for method, release in zip(METHODS, releases):
        check(read(run / (method + "-released.json")) == release, "Standalone model release record differs")
    check(groups[0]["frozen_fingerprint_before"] == groups[1]["frozen_fingerprint_before"]
          and groups[0]["state_fingerprint_before"] != groups[1]["state_fingerprint_before"],
          "Base/delta frozen state differs or learned full fingerprints are identical")

    sys.path.insert(0, str(ROOT))
    from tdyssey.policy_eval_comparison import compare_evaluation_reports
    comparison = compare_evaluation_reports(*groups)
    check(comparison == read(run / "paired-comparison.json") == result["comparison"]["report"]
          and result["comparison"]["available"] is True, "Stored comparison differs from raw-case recomputation")
    check(not any(name in sys.modules for name in ("torch", "numpy", "pymunk")), "Offline helper imported a runtime library")
    files = {}
    for directory in [ledger_path.parent, *(Path(name) for name in ledger["output_dirs"])]:
        for path in directory.rglob("*"):
            if path.is_file():
                files[str(path.resolve())] = path.stat().st_size
    files[str(registry / "index.json")] = (registry / "index.json").stat().st_size
    check(sum(files.values()) <= LIMITS["storage_bytes"], "Aggregate evaluation storage exceeded 512 MiB")
    check(monitor["status"] == "exited" and monitor["returncode"] == monitor["worker_exit_code"] == 0
          and monitor["guard_reason"] is None and monitor["monitor_error"] is None,
          "Guarded worker did not exit cleanly")
    worker = read(meta / "worker.json"); claimed = read(meta / "worker-claimed.json")
    check(monitor["pid"] == worker["pid"] == claimed["identity"]["pid"]
          and monitor["launcher_pid"] == claimed["launcher_pid"]
          and monitor["pid"] != monitor["launcher_pid"], "Actual worker handshake/monitor PID differs")

    output = {"schema_version": 1, "kind": "tdyssey.C.paired_evaluation_reconciliation",
        "at_utc": datetime.now(timezone.utc).isoformat(), "integrity_checks_passed": not issues,
        "issues": issues, "warnings": warnings, "evaluation_id": account, "binding": binding,
        "ledger_sha256": sha(ledger_path), "report_sha256": sha(run / "evaluation-pair-report.json"),
        "launch_declaration_sha256": sha(meta / "declaration.json"), "source_checks": source_checks,
        "preservation": preservation, "original_C_global_steps": original_training["completed"]["global"],
        "evaluation_reserved": ledger["reserved"], "evaluation_completed": ledger["completed"],
        "evaluation_pending": ledger["pending"], "by_shape": ledger["by_shape"],
        "case_journal_events": dict(Counter(row["event"] for row in events)),
        "groups": group_summaries, "model_releases": releases,
        "release_verification": {"full_fp32_parameter_bytes": full_parameter_bytes,
            "criterion": "Network weak reference is dead and freed CUDA allocations cover all full-model parameters; residual workspaces are permitted.",
            "residual_allocated_bytes": [row["resources"]["allocated_bytes"] for row in releases],
            "freed_allocated_bytes": [group["resources"]["allocated_bytes"] - release["resources"]["allocated_bytes"]
                                     for group, release in zip(groups, releases)],
            "supersedes_initial_zero_allocator_assumption": True},
        "frozen_common": groups[0]["frozen_fingerprint_before"] == groups[1]["frozen_fingerprint_before"],
        "full_fingerprints_differ": groups[0]["state_fingerprint_before"] != groups[1]["state_fingerprint_before"],
        "comparison_recomputed_exactly": comparison == result["comparison"]["report"],
        "comparison_overall": comparison["overall"], "comparison_per_shape": comparison["per_shape"],
        "storage_bytes": sum(files.values()), "storage_limit_bytes": LIMITS["storage_bytes"],
        "monitor": monitor, "new_training_operations": 0,
        "limitations": ["Fixed development scene blocks with correlated policy seeds; no untouched-test claim.",
                        "An observed comparison does not establish causal learning or broad generalization.",
                        "All checks were offline, with no tensor/model load, physical operation or ledger mutation."]}
    payload = (json.dumps(output, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output.resolve()), "integrity_checks_passed": not issues,
                      "issues": issues, "warnings": warnings}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
