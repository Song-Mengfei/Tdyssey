"""Run only after root's explicit cold notification; never wait/poll or import Torch.

Default checks closed raw/state/accounting records. --with-score additionally
requires the cold original score and 129/final comparison outputs.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import pickletools
import zipfile
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PARENT = BASE.parent / "stage3b"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(row, path):
    path = Path(path)
    return path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def equal_arrays(first, second):
    if np.issubdtype(first.dtype, np.inexact):
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def storage_bytes(roots):
    seen = set()
    size = 0
    files = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*") if root.is_dir() else (root,):
            if path.is_symlink():
                raise ValueError("Unexpected link in budgeted roots")
            if path.is_file():
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    size += stat.st_size
                    files += 1
    return dict(bytes=size, files=files, atomic_reserve_bytes=256 * 1024**2, cap_bytes=3 * 1024**3)


def check_score(checks, final_update):
    score = read(BASE / "score/score-report.json")
    prior_score = read(PARENT / "score/score-report.json")
    comparison = read(BASE / "comparison129-final/comparison.json")
    windows = read(BASE / "frozen-windows.json")
    independent = read(BASE / "review-primary-recompute.json")
    summaries = {}
    checks["score_source_and_protocol_unchanged"] = (
        score["source_sha256"] == prior_score["source_sha256"] == sha(ROOT / "scripts/native_rgb_stage3_score.py")
        and score["protocol_sha256"] == prior_score["protocol_sha256"] == sha(ROOT / "configs/native_rgb_stage3_score.json"))
    checks["score_and_comparison_no_new_model_or_physics"] = (
        score["model_calls"] == score["optimizer_updates"] == score["physics_controls"] == 0
        and comparison["inference_calls_for_old_baseline_or129"] == 0
        and not comparison["models_or_Torch_loaded"] and comparison["physics_controls"] == 0
        and not comparison["new_scoring_or_thresholds"])
    checks["comparison_old_new_score_binding"] = (
        comparison["old_score_sha256"] == sha(PARENT / "score/score-report.json")
        and comparison["new_score_sha256"] == sha(BASE / "score/score-report.json")
        and comparison["parent_updates"] == 129 and comparison["final_updates"] == final_update
        and all(row["raw_array_bytes_identical"] for row in comparison["input_array_identity_checks"]))
    for cohort, key, n in (("train", "train", 18), ("development", "dev", 36)):
        current, previous = score[cohort], prior_score[cohort]
        npz_name = "train18.npz" if cohort == "train" else "dev36.npz"
        checks[cohort + "_score_raw_file_hash_binding"] = (
            current["raw_npz_sha256"] == sha(BASE / "predictions" / npz_name)
            and previous["raw_npz_sha256"] == sha(PARENT / "predictions" / npz_name))
        ids = [row["window_id"] for row in windows[key]]
        checks[cohort + "_all_fixed_rows_and_methods"] = (
            current["denominator"] == n and len(current["rows"]) == n
            and [row["window_id"] for row in current["rows"]] == ids
            and [row["window_id"] for row in previous["rows"]] == ids)
        for method in ("hold", "frozen"):
            checks[cohort + "_" + method + "_exact_old_metrics"] = all(
                a["methods"][method] == b["methods"][method]
                for a, b in zip(current["rows"], previous["rows"]))
        fields = {"hold": [row["methods"]["hold"] for row in current["rows"]],
                  "frozen": [row["methods"]["frozen"] for row in current["rows"]],
                  "update129": [row["methods"]["fitted"] for row in previous["rows"]],
                  "final": [row["methods"]["fitted"] for row in current["rows"]]}
        independently_recomputed = independent["cohorts"][cohort]["rows"]
        checks[cohort + "_216same_primary_pixel_formula_rows_match"] = (
            [row["window_id"] for row in independently_recomputed] == ids
            and all(math.isclose(row["methods"][method]["raw_object_error"], recorded["raw_object_error"], abs_tol=1e-12)
                    and row["methods"][method]["hard_generation_failure"] == recorded["hard_generation_failure"]
                    for method, measured in fields.items()
                    for row, recorded in zip(independently_recomputed, measured)))
        for method, rows in fields.items():
            summary = comparison["summary"][cohort]["methods"][method]
            checks[cohort + "_" + method + "_same_formula_summary"] = (
                summary["denominator"] == n
                and math.isclose(float(np.mean([row["raw_object_error"] for row in rows])),
                                 summary["means_all_rows"]["raw_object_error"], abs_tol=1e-12)
                and sum(row["hard_generation_failure"] for row in rows) == summary["hard_generation_failures"])
        final_error = [row["raw_object_error"] for row in fields["final"]]
        hold_error = [row["raw_object_error"] for row in fields["hold"]]
        wins = sum(a < b - 1e-12 for a, b in zip(final_error, hold_error))
        checks[cohort + "_wins_and_full_denominator"] = wins == current["fitted_vs_hold"]["strict_wins"]
        shape = current["methods"]["fitted"]["legacy_deformation"]
        checks[cohort + "_shape_unknown_not_passed"] = shape == dict(
            assessable=sum(row["legacy_shape_assessable"] for row in fields["final"]),
            failed=sum(row["legacy_shape_assessable"] and bool(row["legacy_shape_deformation_failure"])
                       for row in fields["final"]), total=n,
            unknown=sum(not row["legacy_shape_assessable"] for row in fields["final"]))
        summaries[cohort] = dict(fitted_vs_hold=current["fitted_vs_hold"],
            final_vs_update129=comparison["summary"][cohort]["final_vs_update129"],
            final_shape=shape, H1_motion=comparison["summary"][cohort]["H1_motion"])
        current_gallery = read(BASE / "score" / ("gallery-train" if cohort == "train" else "gallery-development") / "pages.json")
        checks[cohort + "_all_original_score_gallery_order"] = (
            len(current_gallery["pages"]) == n // 6
            and [wid for page in current_gallery["pages"] for wid in page["window_ids"]] == ids
            and [i for page in current_gallery["pages"] for i in page["original_order_indices"]] == list(range(n)))
    target = score["train"]["dynamic_fit_target"]
    expected = (target["relative_reduction"] >= .2 and target["strict_wins"] >= 12
                and target["method_hard_failures"] == 0 and target["protocol_valid"]
                and target["truth_valid"] and target["hold_valid"])
    checks["unchanged20percent12wins_target"] = (
        target["required_relative_reduction"] == .2 and target["required_strict_wins"] == 12
        and target["passed"] == expected)
    checks["DEV31_4_1_preserved"] = {
        key: value["methods"]["fitted"]["denominator"]
        for key, value in score["development"]["groups"]["H1_motion"].items()
    } == {"moving": 31, "near_static": 4, "intermediate": 1}
    gallery = read(BASE / "comparison129-final/gallery-pages.json")
    checks["all18_comparison_gallery_fixed_order"] = (
        gallery["all18_shown"] and gallery["raw_arrays_unchanged"] and len(gallery["pages"]) == 3
        and [wid for page in gallery["pages"] for wid in page["window_ids"]]
        == [row["window_id"] for row in windows["train"]]
        and [i for page in gallery["pages"] for i in page["frozen_order_indices"]] == list(range(18)))
    for folder in (BASE / "score", BASE / "comparison129-final"):
        checks[folder.name + "_inventory_hashes"] = all(
            matches(row, folder / row["file"]) for row in read(folder / "files.json"))
    with (BASE / "comparison129-final/all54-method-comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    checks["four_method_CSV216_rows54windows"] = len(rows) == 216 and len({(r["cohort"], r["window_id"]) for r in rows}) == 54
    method_sets = {}
    for row in rows:
        method_sets.setdefault((row["cohort"], row["window_id"]), set()).add(row["method"])
    checks["all54_CSVwindows_keep_exact_four_methods"] = all(
        methods == {"hold", "frozen", "update129", "final"} for methods in method_sets.values())
    return dict(summaries=summaries, registered_target=target,
                score_sha256=sha(BASE / "score/score-report.json"),
                comparison_sha256=sha(BASE / "comparison129-final/comparison.json"))


def main(with_score=False):
    work = read(BASE / "model-work.json")
    ledger = read(BASE.parent / "native-rgb-cost-ledger.json")
    if not work["budget"]["closed"] or ledger["active_stage"] is not None:
        raise RuntimeError("Cold notification and fully closed model ledger are required; this helper never polls")
    result = work["training_result"]
    if result is None:
        raise RuntimeError("Failed run requires manual cold error/checkpoint review; never infer missing state")
    progress = result["progress"]
    final = progress["updates"]
    identity = read(BASE / "source-identity.json")
    protected = read(BASE / "review-protected-inputs.json")
    parent = read(BASE / "parent-expected.json")
    restore = read(BASE / "restore-review.json")
    plan = read(BASE / "continuation-plan.json")
    prior = read(BASE / "prior-ledger.json")
    windows = read(BASE / "frozen-windows.json")
    batches = read(BASE / "paired-batches.json")
    checks = {}
    checks["96sources_and_snapshots_unchanged"] = len(identity["files"]) == 96 and all(
        matches(row, row["source"]) and matches(row, BASE / row["snapshot"]) for row in identity["files"])
    checks["742old_protected_files_unchanged"] = len(protected["files"]) == 742 and all(
        matches(row, row["source"]) for row in protected["files"])
    local = ledger["stages"][-1]
    completed = final - 129
    checks["single3C_closed_stage_prior_history_preserved"] = (
        len(ledger["stages"]) == 3 and ledger["stages"][:2] == prior["stages"]
        and local["phase"] == "stage3c-visible-microfit-continuation"
        and local["seconds_before"] == prior["charged_seconds"] == 444.6250000000073
        and ledger == work["budget"]["ledger"]
        and ledger["historical_cancelled_experiment"] == prior["historical_cancelled_experiment"])
    checks["local121_600_and_cumulative250_global3000_5400"] = (
        129 <= final <= 250 and 0 <= completed <= local["reserved_updates"] <= 121
        and local["completed_updates"] == completed
        and ledger["charged_updates"] == 129 + local["reserved_updates"]
        and ledger["completed_updates"] == final
        and work["budget"]["elapsed_model_work_seconds"] <= 600
        and ledger["charged_seconds"] == 444.6250000000073 + work["budget"]["elapsed_model_work_seconds"]
        and ledger["max_updates"] == 3000 and ledger["max_seconds"] == 5400)
    checks["real6state_resume_equal_and129_boundary"] = (
        restore["passed"] and restore["all_protected_state_equal"]
        and restore["protected_before_sha256"] == restore["protected_after_sha256"]
        and set(restore["protected_before_sha256"]) == {"model", "optimizer", "scaler", "scheduler", "rng", "order"}
        and restore["updates"] == 129 and restore["samples"] == restore["order_cursor"] == 2064
        and restore["optimizer_state_entries"] == 296 and restore["scheduler_last_epoch"] == 129
        and restore["boundary_checkpoint"]["updates"] == 129
        and restore["unverified_or_fresh_checkpoint_save_rejected"])
    checks["only_declared_identity_metadata_migrated"] = (
        restore["parent_config"] == parent["config"] and restore["parent_identity"] == parent["identity"]
        and restore["new_config"] == result["config"] and restore["new_identity"] == result["identity"]
        and restore["allowed_config_changes"] == plan["config_changes"]
        and restore["private_global_overrides"] == ["restore_checkpoint", "source_identity", "save_training_checkpoint"]
        and restore["train_bytecode_unchanged"] and restore["live_ledger_not_restored_from_checkpoint"])
    old_meta = read(PARENT / "predictions/fitted-train18.json")
    resumed = restore["stage3c_resume_parameter_sha256"]
    checks["restored129_parameter_hashes_match_actual_parent"] = resumed == old_meta["parameter_sha256"] == progress["stage3c_resume_parameter_sha256"]
    checks["old_gradient_init_evidence_not_relabelled_new"] = (
        progress["first_gradient_norms"] == parent["progress"]["first_gradient_norms"]
        and progress["initial_parameter_sha256"] == parent["progress"]["initial_parameter_sha256"]
        and "not remeasured stage3C" in progress["continuation"]["inherited_first_gradient_norms_scope"])
    history = [json.loads(line) for line in (BASE / "run/history.jsonl").read_text().splitlines()]
    draws = [json.loads(line) for line in (BASE / "run/sampling-history.jsonl").read_text().splitlines()]
    cycle = [row["dataset_index"] for row in sorted(windows["train"], key=lambda row: row["letter"])]
    expected = [cycle[i % 18] for i in range(2064, final * 16)]
    flattened = [index for row in history for index in row["selected_indices"]]
    checks["130_to_final_exact_order_draws_attempts_and_finite_history"] = (
        len(history) == completed and flattened == expected and progress["samples"] == final * 16
        and all(row["update"] == 130 + i and row["samples"] == row["update"] * 16
                and math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"])
                for i, row in enumerate(history))
        and all(row["attempt"] == 130 + i for i, row in enumerate(draws))
        and all(a["selected_indices"] == d["selected_indices"] for a, d in zip(history, draws)))
    exposure = Counter({int(key): value for key, value in parent["progress"]["selected_window_exposures"].items()})
    exposure.update(index for row in draws for index in row["selected_indices"])
    checks["cumulative_exposure_and_attempted_draws_preserved"] = (
        progress["selected_window_exposures"] == {str(key): value for key, value in exposure.items()}
        and progress["attempted_draws"] == 2064 + sum(len(row["selected_indices"]) for row in draws))
    checks["learning_rate_not_reset_Tmax3000"] = all(
        math.isclose(row["learning_rate"], 1e-5 + (1e-4 - 1e-5) * (1 + math.cos(math.pi * row["update"] / 3000)) / 2,
                     abs_tol=1e-12) for row in history)
    raw_summary = {}
    for name, key, n in (("train18", "train", 18), ("dev36", "dev", 36)):
        native = arrays(BASE / "predictions" / ("fitted-" + name + ".npz"))
        meta = read(BASE / "predictions" / ("fitted-" + name + ".json"))
        combo = arrays(BASE / "predictions" / (name + ".npz"))
        previous = arrays(PARENT / "predictions" / (name + ".npz"))
        ids = [row["window_id"] for row in windows[key]]
        statuses = ["nonfinite_native_RGB" if not np.isfinite(v).all() else
                    "out_of_range_native_RGB" if v.min() < 0 or v.max() > 1 else "ok" for v in native["prediction"]]
        expected_ranges = []
        for index, value in enumerate(native["prediction"]):
            finite = np.isfinite(value)
            expected_ranges.append(dict(index=index, window_id=ids[index], status=statuses[index],
                finite_values=int(finite.sum()), min=float(value[finite].min()) if finite.any() else None,
                max=float(value[finite].max()) if finite.any() else None,
                negative_values=int((value < 0).sum()), above_one_values=int((value > 1).sum())))
        checks[name + "_final_raw54_status_and_fixed_batches"] = (
            native["prediction"].shape == (n, 3, 96, 96) and native["prediction"].dtype == np.float32
            and [str(v) for v in native["window_ids"]] == ids
            and statuses == native["status"].tolist() and meta["statuses"] == dict(Counter(statuses))
            and meta["range_rows"] == expected_ranges and meta["unmodified_native_RGB"] and not meta["postprocessing"]
            and meta["updates"] == final and meta["failure"] is None
            and meta["prediction_sha256"] == sha(BASE / "predictions" / ("fitted-" + name + ".npz"))
            and len(meta["batch_records"]) == len(batches["cohorts"][name])
            and all(all(a[k] == v for k, v in planned.items()) for a, planned in zip(meta["batch_records"], batches["cohorts"][name])))
        checks[name + "_true_hold_original_baseline_exact_reference"] = all(
            equal_arrays(combo[field], previous[field])
            for field in ("window_ids", "last", "true", "hold", "last_full_mask", "true_full_mask", "frozen", "frozen_status"))
        checks[name + "_combined_final_raw_exact"] = equal_arrays(combo["fitted"], native["prediction"]) and equal_arrays(combo["fitted_status"], native["status"])
        current = meta["parameter_sha256"]
        checks[name + "_final_parameter_digest"] = meta["parameter_digest"] == hashlib.sha256(
            json.dumps(current, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        raw_summary[name] = dict(windows=n, statuses=dict(Counter(statuses)), min=float(np.nanmin(native["prediction"])),
            max=float(np.nanmax(native["prediction"])), changed_parameter_tensors_since129=sum(current[k] != v for k, v in resumed.items()),
            parameter_digest=meta["parameter_digest"])
    checks["final_model_shared_across_train_and_dev"] = raw_summary["train18"]["parameter_digest"] == raw_summary["dev36"]["parameter_digest"]
    checks["only_one_final54_no_rebaseline_or129_inference"] = (
        not list((BASE / "predictions").glob("frozen-*.npz"))
        and len(list((BASE / "predictions").glob("*.npz"))) == 4
        and restore["no_native_baseline_called"]
        and progress["after_updates_summary"]["native_sampler_calls"] == 14
        and progress["after_updates_summary"]["native_prediction_frames"] == 54
        and progress["after_updates_summary"]["all_requires_grad"]
        and progress["after_updates_summary"]["model_parameters"] == 8729827)
    checkpoint = BASE / "run/last.pt"
    with zipfile.ZipFile(checkpoint) as archive:
        crc = archive.testzip() is None
        data = archive.read(next(name for name in archive.namelist() if name.endswith("/data.pkl")))
        key_names = {arg for op, arg, _ in pickletools.genops(data) if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE")}
    required = {"model", "optimizer", "scaler", "scheduler", "rng", "order", "progress", "config", "identity", "ledger"}
    root_check = work["checkpoint_review"]
    checks["complete_final_archive_and_inbudget_root_deserialization"] = (
        crc and required <= key_names and sha(checkpoint) == result["last"]["sha256"] == root_check["checkpoint_sha256"]
        and root_check["passed"] and root_check["updates"] == final
        and root_check["optimizer_parameter_state_entries"] == 296 and root_check["order_cursor"] == final * 16
        and root_check["scheduler_T_max"] == 3000 and root_check["model_work_elapsed"] <= work["budget"]["elapsed_model_work_seconds"]
        and len(list((BASE / "run").glob("*.pt"))) == 1 and not list((BASE / "run").glob("*.tmp")))
    checks["no_new_physics_expanded_eval_server_or_packaging"] = (
        work["physical_controls"] == 0 and work["no_H4"] and work["no_full319"] and work["no_unseen"]
        and all(not result["config"][key] for key in ("allow_H4", "allow_full_validation319", "allow_unseen", "allow_server", "allow_packaging")))
    storage = storage_bytes(read(BASE / "MODEL_WORK_STARTED.json")["initial_storage"]["roots"])
    checks["old_new_source_cache_plus_atomic_reserve_below3GiB"] = storage["bytes"] + storage["atomic_reserve_bytes"] < storage["cap_bytes"]
    checks["peak_includes_restore_boundary_and_training_native"] = (
        work["peak_cuda_allocated"] == max(result["peak_cuda_allocated"], restore["restore_boundary_peak_cuda_allocated"])
        and work["peak_cuda_reserved"] == max(result["peak_cuda_reserved"], restore["restore_boundary_peak_cuda_reserved"]))
    score = check_score(checks, final) if with_score else None
    checks = {key: bool(value) for key, value in checks.items()}
    report = dict(kind="stage3c-independent-cold-postrun-review.v1", checked_at_utc=datetime.now(timezone.utc).isoformat(),
        integrity_checks_passed=all(checks.values()), finalized=with_score, pending=[] if with_score else ["cold original score and129/final comparison"],
        checks=checks, parent_updates=129, final_updates=final, local_attempts=local["reserved_updates"], local_completed=completed,
        local_model_work_seconds=work["budget"]["elapsed_model_work_seconds"], aggregate_attempts=ledger["charged_updates"],
        aggregate_model_work_seconds=ledger["charged_seconds"], raw_summary=raw_summary, score=score,
        fixed_noise_train_loss_at129=parent["progress"]["fixed_train_loss_after"],
        fixed_noise_train_loss_final=progress["fixed_train_loss_after"], storage=storage,
        peak_cuda_allocated=work["peak_cuda_allocated"], peak_cuda_reserved=work["peak_cuda_reserved"],
        source_identity_sha256=sha(BASE / "source-identity.json"), restore_review_sha256=sha(BASE / "restore-review.json"),
        inherited_gradient_evidence_is_3B=True, actual_3C_full_gradient_presence_guard_unchanged=True,
        checkpoint_archive_CRC_passed=crc, reviewer_Torch_load_calls=0, reviewer_model_loads=0,
        reviewer_GPU_calls=0, reviewer_optimizer_updates=0, reviewer_physics_steps=0, reviewer_training_iterator_calls=0,
        qualification="Root actual six-state restore and GPU work is cross-checked, never repeated. Loss/change evidence is separate from raw motion/shape acceptance. No later execution is authorized.")
    (BASE / "review-postrun-integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (BASE / "review-postrun-integrity.md").write_text(
        "# Independent stage 3C cold audit\n\nIntegrity checks pass: " + str(report["integrity_checks_passed"]).lower()
        + ". Finalized: " + str(with_score).lower() + ".\n\nExact129 continuation reached update" + str(final)
        + ", with" + str(local["reserved_updates"]) + "new attempts and" + str(completed) + "new completed updates.\n\n"
        + "The96 frozen sources and742 protected files are checked. Actual six-state restore, parent hashes, current truthful metadata, sampling/RNG continuity evidence, final raw54 and single last archive are cross-checked without Torch/model loading.\n\n"
        + "New work seconds: " + str(work["budget"]["elapsed_model_work_seconds"]) + "; aggregate: " + str(ledger["charged_seconds"])
        + ". Original3/3B fees are preserved. Inherited gradient norms remain3B evidence.\n\n"
        + "Original score/four-method summaries are in theJSON when finalized. Unknown shapes are never counted as passed; loss decrease cannot replace RGB motion acceptance.\n\n"
        + "Reviewer:0Torch/model loads,0GPU,0optimizer,0physics,0training-iterator calls. Stop for parent review.\n", encoding="utf-8")
    print(json.dumps(dict(passed=report["integrity_checks_passed"], finalized=with_score, checks=checks,
                         review_sha256=sha(BASE / "review-postrun-integrity.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-score", action="store_true")
    main(parser.parse_args().with_score)
