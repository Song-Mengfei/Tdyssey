"""Independent cold-file stage3 audit. No Torch/model/env imports or restoration."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import pickletools
import zipfile
import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_matches(row, path):
    path = Path(path)
    return path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]


def array_file(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def footprint(roots):
    seen = set()
    size = files = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            if path.is_symlink():
                raise ValueError("Unexpected link in audited roots")
            if path.is_file():
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    size += stat.st_size
                    files += 1
    return dict(bytes=size, files=files, cap_bytes=3 * 1024**3,
                below_cap=size < 3 * 1024**3, existing_temporary_files_counted=True)


def main():
    windows = read(BASE / "frozen-windows.json")
    plans = read(BASE / "paired-batches.json")
    identity = read(BASE / "source-identity.json")
    protected = read(BASE / "review-protected-inputs.json")
    model_work = read(BASE / "model-work.json")
    ledger = read(BASE.parent / "native-rgb-cost-ledger.json")
    error = read(BASE / "run/error.json")
    score = read(BASE / "score/score-report.json")
    checks = {}
    frozen_bad = [str(path) for row in identity["files"]
                  for path in (Path(row["source"]), BASE / row["snapshot"])
                  if not file_matches(row, path)]
    protected_bad = [row["source"] for row in protected["files"] if not file_matches(row, row["source"])]
    checks["all56_frozen_files_and_snapshots_unchanged"] = len(identity["files"]) == 56 and not frozen_bad
    checks["all381_protected_inputs_unchanged"] = len(protected["files"]) == 381 and not protected_bad
    checks["single_closed_zero_attempt_stage"] = (ledger["active_stage"] is None and len(ledger["stages"]) == 1
        and ledger["charged_updates"] == ledger["completed_updates"] == 0
        and ledger["stages"][0]["reserved_updates"] == ledger["stages"][0]["completed_updates"] == 0)
    elapsed = ledger["charged_seconds"]
    checks["actual_model_clock_within600_and_aggregate5400"] = (0 < elapsed <= 600
        and ledger["max_updates"] == 3000 and ledger["max_seconds"] == 5400
        and elapsed == model_work["budget"]["elapsed_model_work_seconds"]
        and ledger == model_work["budget"]["ledger"] and model_work["budget"]["closed"])
    checks["historical_native_cost_separate"] = ledger["historical_cancelled_experiment"] == dict(
        historical_native_attempts=1899, historical_native_model_seconds=5352.516, included_in_new_round_budget=False)
    progress = error["progress"]
    checks["no_training_draws_losses_gradients_or_validation"] = (progress["updates"] == progress["samples"] == 0
        and progress["validation"] == [] and "first_gradient_norms" not in progress
        and "fixed_train_loss_before" not in progress and not (BASE / "run/history.jsonl").exists()
        and not (BASE / "run/sampling-history.jsonl").exists())
    checks["baseline_failure_before_updates"] = model_work["failure"] == "FloatingPointError: Invalid native RGB retained without repair" and progress["previous_complete_checkpoint_preserved"]

    raw_path = BASE / "predictions/frozen-train18.npz"
    raw = array_file(raw_path)
    raw_meta = read(raw_path.with_suffix(".json"))
    raw_rows = []
    for index, (wid, value, status) in enumerate(zip(raw["window_ids"], raw["prediction"], raw["status"])):
        finite = np.isfinite(value)
        raw_rows.append(dict(index=index, window_id=str(wid), status=str(status), finite_values=int(finite.sum()),
            below_zero=int((value < 0).sum()), above_one=int((value > 1).sum()),
            min=float(value[finite].min()) if finite.any() else None,
            max=float(value[finite].max()) if finite.any() else None))
    checks["raw18_order_status_and_first4_generated"] = (raw["prediction"].shape == (18, 3, 96, 96)
        and [str(x) for x in raw["window_ids"]] == [x["window_id"] for x in windows["train"]]
        and raw["status"].tolist() == ["out_of_range_native_RGB", "ok", "ok", "ok"] + ["not_attempted"] * 14
        and np.isfinite(raw["prediction"][:4]).all() and np.isnan(raw["prediction"][4:]).all())
    checks["six_real_negative_RGB_values_retained"] = (raw_rows[0]["below_zero"] == 6
        and raw_rows[0]["min"] == -0.0196077823638916 and raw_rows[0]["max"] == 1.0
        and all(x["below_zero"] == x["above_one"] == 0 for x in raw_rows[1:]))
    checks["first_native_batch_frozen_pairing_and_original_parameter_digest"] = (len(raw_meta["batch_records"]) == 1
        and all(raw_meta["batch_records"][0][key] == value for key, value in plans["cohorts"]["train18"][0].items())
        and raw_meta["updates"] == 0 and raw_meta["prediction_sha256"] == sha(raw_path)
        and raw_meta["parameter_sha256"] == progress["initial_parameter_sha256"]
        and raw_meta["parameter_digest"] == hashlib.sha256(json.dumps(raw_meta["parameter_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    checks["no_other_native_prediction_files"] = not any((BASE / "predictions" / name).exists() for name in (
        "frozen-dev36.npz", "fitted-train18.npz", "fitted-dev36.npz"))
    comparisons = {}
    gallery_checks = []
    for name, key, report_key, n in (("train18", "train", "train", 18), ("dev36", "dev", "development", 36)):
        actual = array_file(BASE / "predictions" / f"{name}.npz")
        truth = array_file(BASE / "truth" / f"{name}.npz")
        ids = [x["window_id"] for x in windows[key]]
        common_ok = ([str(x) for x in actual["window_ids"]] == ids
            and all(np.array_equal(actual[field], truth[field], equal_nan=True) for field in (
                "last", "true", "hold", "last_full_mask", "true_full_mask"))
            and np.isnan(actual["fitted"]).all()
            and actual["fitted_status"].tolist() == ["not_measured_budget_or_failure"] * n)
        if key == "train":
            common_ok = common_ok and np.array_equal(actual["frozen"], raw["prediction"], equal_nan=True) and np.array_equal(actual["frozen_status"], raw["status"])
        else:
            common_ok = common_ok and np.isnan(actual["frozen"]).all() and actual["frozen_status"].tolist() == ["not_measured_budget_or_failure"] * n
        checks[name + "_comparison_keeps_missing_denominator_and_exact_raw"] = common_ok
        cohort = score[report_key]
        checks[name + "_score_all_original_rows_and_failure_penalties"] = (cohort["denominator"] == n
            and [x["window_id"] for x in cohort["rows"]] == ids
            and all(x["methods"]["fitted"]["denominator_retained"] and x["methods"]["fitted"]["hard_generation_failure"] for x in cohort["rows"])
            and cohort["methods"]["fitted"]["hard_generation_failures"] == n
            and cohort["methods"]["fitted"]["legacy_deformation"] == dict(assessable=0, failed=0, total=n, unknown=n))
        gallery_dir = BASE / "score" / ("gallery-train" if key == "train" else "gallery-development")
        gallery = read(gallery_dir / "pages.json")
        gallery_checks.append([x for page in gallery["pages"] for x in page["window_ids"]] == ids
            and [x for page in gallery["pages"] for x in page["original_order_indices"]] == list(range(n))
            and len(gallery["pages"]) == n // 6 and gallery["raw_arrays_unchanged"])
        comparisons[name] = dict(windows=n, fitted_measured=0, frozen_measured=4 if key == "train" else 0,
            fitted_unknown_shape=n, fitted_failure_penalty_mean=cohort["methods"]["fitted"]["means_all_rows"]["raw_object_error"],
            frozen_hard_failures=cohort["methods"]["frozen"]["hard_generation_failures"],
            source_npz_sha256=sha(BASE / "predictions" / f"{name}.npz"))
    checks["all9_gallery_pages_exact_frozen_order"] = all(gallery_checks)
    target = score["train"]["dynamic_fit_target"]
    checks["missing_fitted_cannot_pass_dynamic_fit_target"] = not target["passed"] and target["denominator"] == 18 and target["method_hard_failures"] == 18 and target["strict_wins"] == 0
    checks["frozen_score_empty_fourth_output_retained"] = score["train"]["rows"][3]["methods"]["frozen"]["failure_reasons"] == ["empty_object", "empty_pusher"] and score["train"]["methods"]["frozen"]["hard_generation_failures"] == 16
    checks["score_source_protocol_frozen_and_CPU_only"] = (score["protocol_sha256"] == sha(REPO / "configs/native_rgb_stage3_score.json")
        and score["source_sha256"] == sha(REPO / "scripts/native_rgb_stage3_score.py")
        and score["model_calls"] == score["optimizer_updates"] == score["physics_controls"] == 0)
    files = read(BASE / "score/files.json")
    checks["all_score_artifact_file_hashes_match"] = all(file_matches(row, BASE / "score" / row["file"]) for row in files)

    checkpoint = BASE / "run/last.pt"
    with zipfile.ZipFile(checkpoint) as archive:
        crc_passed = archive.testzip() is None
        pickles = [x for x in archive.namelist() if x.endswith("/data.pkl")]
        values = {arg for op, arg, _ in pickletools.genops(archive.read(pickles[0])) if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE")}
        required = {"model", "optimizer", "scaler", "scheduler", "rng", "order", "progress", "config", "identity", "ledger"}
        checkpoint_archive = dict(bytes=checkpoint.stat().st_size, sha256=sha(checkpoint), zip_CRC_passed=crc_passed,
            required_key_strings_present=sorted(required & values), required_key_count=len(required & values),
            torch_load_calls=0, model_restoration_performed=False,
            qualification="Archive CRC and pickle opcode-string inspection only. This does not establish a tested semantic restore or actual updated optimizer state.")
    root_archive = read(BASE / "checkpoint-archive-audit.json")
    checks["t0_last_archive_intact_single_and_root_hash_matches"] = (crc_passed and required <= values
        and root_archive["sha256"] == checkpoint_archive["sha256"] and root_archive["bytes"] == checkpoint_archive["bytes"]
        and len(list((BASE / "run").glob("*.pt"))) == 1 and not (BASE / "run/best.pt").exists()
        and not list((BASE / "run").glob("*.tmp")))
    roots = read(BASE / "MODEL_WORK_STARTED.json")["initial_storage"]["roots"]
    storage = footprint(roots)
    checks["combined_old_new_source_cache_storage_with_reserve_below3GiB"] = storage["bytes"] + 256 * 1024**2 < storage["cap_bytes"]
    state = read(BASE.parent / "stage-state.json")
    checks["physical_control_budgets_unchanged_and_no_extra_calls"] = (model_work["physical_controls"] == 0
        and state["budget"]["formal_total_planned_supervised_controls"] == 39680
        and state["budget"]["old_native_controls"] == 34560 and model_work["no_H4"] and model_work["no_full319"] and model_work["no_unseen"])
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(kind="native-gpc-stage3-independent-postrun-review.v1", checked_at_utc=datetime.now(timezone.utc).isoformat(),
        integrity_and_protocol_audit_passed=all(checks.values()), model_experiment_completed=False,
        outcome="pretraining_inference_range_abort; no microfit or generalization result exists", checks=checks,
        raw_rows=raw_rows, comparisons=comparisons, checkpoint_archive=checkpoint_archive, storage=storage,
        charged_model_work_seconds=elapsed, optimizer_attempts=0, completed_optimizer_updates=0,
        aggregate_remaining_attempts=3000, aggregate_remaining_seconds=5400-elapsed,
        historical_native_attempts=1899, historical_native_seconds=5352.516,
        protected_files_checked=len(protected["files"]), frozen_source_files_checked=len(identity["files"]),
        frozen_mismatches=frozen_bad, protected_mismatches=protected_bad,
        source_identity_sha256=sha(BASE / "source-identity.json"), score_report_sha256=sha(BASE / "score/score-report.json"),
        train_dynamic_target=target, frozen_first_four=[dict(window_id=row["window_id"], **{k:row["methods"]["frozen"][k] for k in (
            "prediction_status", "failure_reasons", "raw_mask_iou", "hard_generation_failure")}) for row in score["train"]["rows"][:4]],
        qualifications=["The first TRAIN native batch produced4 rows. One has6 real negative RGB values;14 TRAIN baseline rows and allDEV/fitted rows were not generated.",
            "Runtime ok for3 rows means finite/in-range only. The fourth row (E/train-2a-021:28) has empty gray-object and blue-pusher masks under the unchanged frozen palette score; it is not an entirely blank RGB image.",
            "Visual review of gallery train-01 shows red objects/fragments in the three in-range frozen B/C/E outputs, unlike the gray truths. No recoloring or revised threshold is applied; empty_object refers specifically to the frozen gray-object mask.",
            "Fitted penalty means are missing-data bookkeeping, not observed trained-model performance. No loss, gradient reachability, parameter updates, fitted motion, or generalization was tested.",
            "The complete last archive was written at0updates before baseline. CRC/key checks do not substitute for a full semantic restore test.",
            "No threshold changes, clipping, raw-array repairs, extra model calls, retraining, full319/H4/unseen evaluation, physics, server, or packaging were performed by reviewer.",
            "Await parent decision before changing any range policy or rerunning. The unused budget is not permission to continue."],
        reviewer_model_loads=0, reviewer_torch_load_calls=0, reviewer_model_inferences=0,
        reviewer_optimizer_updates=0, reviewer_env_steps=0, reviewer_training_iterator_calls=0)
    (BASE / "review-postrun.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = ("# Independent stage3 post-run review\n\nIntegrity and protocol audit: " + str(result["integrity_and_protocol_audit_passed"]).lower()
        + ". Experiment outcome: pretraining inference-range abort, before any loss, gradient, training draw, optimizer attempt, or update.\n\n"
        + "The unique closed model-work stage charged " + str(elapsed) + " seconds,0attempts/0updates. New aggregate remains3000attempts/5400seconds; historical1899attempts/5352.516seconds is separate. Remaining new budget is " + str(5400-elapsed) + " seconds, without authorization to rerun.\n\n"
        + "The first frozen TRAIN batch generated4of18 rows. A/train-2b-018:11 retained6 negative RGB values, minimum -0.0196077823638916, maximum1. This is not rounding epsilon. The other3 rows are finite/in-range; unchanged offline score additionally marks E/train-2a-021:28 empty-object and empty-pusher. Here empty-object means no gray-object mask under the frozen palette, not a blank RGB image: the first gallery visibly contains red objects/fragments in frozen B/C/E, unlike gray truths. No recoloring or threshold change is applied. All other14 TRAIN baseline rows, all36 DEV baseline rows, and all18+36 fitted rows are missing and retained.\n\n"
        + "All18/36 denominator rows, fixed orders,54raw comparison references,3TRAIN+6DEV gallery pages, statuses, and score hashes match. Fitted18/18 and36/36 are missing-data failure penalties with0shape-assessable outputs; their means are not measured trained-model quality. TRAIN motion target is unproven and cannot pass. This does not establish microfit failure, generalization failure, or rejection of the GPC route.\n\n"
        + "All56 frozen source/input/protocol files and snapshots and381 protected C/index/joint/released-init/old-weight files remain unchanged. Native raw arrays are not clipped or repaired.\n\n"
        + "The sole t0complete last.pt is35,096,361bytes, SHA256 " + checkpoint_archive["sha256"] + ". Independent ZIP CRC and pickle-string checks find all10required state-key names. No Torch loading/model restoration was performed; this is an archive-structure check, not a semantic restore or updated-optimizer test.\n\n"
        + "Combined old/new/native source/cache files total " + str(storage["bytes"]) + " bytes; plus256MiB atomic reserve remains below3GiB. Physical controls remain39680supervised/74240historical-inclusive, with0new steps.\n\n"
        + "Reviewer:0model loads/inferences,0Torch loads,0optimizer updates,0env steps,0training-iterator calls. Stop and return the evidence to the parent; no altered condition or second run is authorized.\n")
    (BASE / "review-postrun.md").write_text(text, encoding="utf-8")
    print(json.dumps(dict(passed=result["integrity_and_protocol_audit_passed"], checks=checks,
        review_json_sha256=sha(BASE / "review-postrun.json"), review_md_sha256=sha(BASE / "review-postrun.md"))))


if __name__ == "__main__":
    main()
