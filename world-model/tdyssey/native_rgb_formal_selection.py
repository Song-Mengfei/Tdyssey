"""Frozen DEV36 checkpoint selection; standard library only, no model work.

Failures remain in both horizon denominators. Denoising loss, shape unknowns,
and predictions never redefine the frozen truth-based moving groups.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

POLICY = "formal-dev36-hardfail-then-balanced-moving-gray-error-v1"
CATALOG_KIND = "native-rgb-formal-selection-catalog.v1"
FROZEN_INDEX_SHA256 = "4f00b9372708d1229b0d22e6b957e4be2f6fe7b5c23cbcf180eed1c610766dbf"
SEEN_LETTERS = "ABCEFHIJLMOPSTUVXY"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _validate_windows(windows):
    if len(windows) != 36:
        raise ValueError("Selection requires exactly the frozen36 development windows")
    ids = []
    for row in windows:
        if (not isinstance(row["trajectory_id"], str) or not row["trajectory_id"]
                or type(row["anchor"]) is not int or not 3 <= row["anchor"] <= 60
                or row["window_id"] != f"{row['trajectory_id']}:{row['anchor']}"
                or row["letter"] not in SEEN_LETTERS
                or row["h1_motion_class"] not in ("moving", "near_static", "intermediate")
                or type(row["h4_endpoint_moving"]) is not bool):
            raise ValueError("Invalid frozen development identity/labels")
        ids.append(row["window_id"])
    if len(set(ids)) != 36 or Counter(row["letter"] for row in windows) != Counter({letter: 2 for letter in SEEN_LETTERS}):
        raise ValueError("Frozen development IDs must be distinct and contain two per seen letter")
    if not any(row["h1_motion_class"] == "moving" for row in windows):
        raise ValueError("H1 moving group is empty; stop before model work")
    if not any(row["h4_endpoint_moving"] for row in windows):
        raise ValueError("H4 endpoint-moving group is empty; stop before model work")


def build_selection_catalog(development_windows, *, source_sha256):
    """Build from frozen truth metadata; use load_selection_catalog in production."""
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256)):
        raise ValueError("A SHA256 identity for the frozen index file is required")
    windows = []
    for row in development_windows:
        if row.get("split") != "seen_val":
            raise ValueError("Selection cannot include TRAIN or held-out windows")
        windows.append(dict(window_id=row["window_id"], trajectory_id=row["trajectory_id"],
            anchor=row["anchor"], letter=row["letter"],
            development_role=row.get("development_role"),
            h1_motion_class=row["h1"]["motion_class"],
            h4_endpoint_moving=row["h4"]["endpoint_moving"]))
    _validate_windows(windows)
    payload = dict(kind=CATALOG_KIND, source_sha256=source_sha256, windows=windows)
    payload["catalog_sha256"] = _digest(payload)
    payload["summary"] = dict(windows=36, independent_trajectories=len({r["trajectory_id"] for r in windows}),
        h1_motion=dict(Counter(r["h1_motion_class"] for r in windows)),
        h4_endpoint_moving=sum(r["h4_endpoint_moving"] for r in windows),
        development_roles=dict(Counter(r["development_role"] for r in windows)))
    return payload


def load_selection_catalog(path, *, expected_sha256=FROZEN_INDEX_SHA256):
    """Read only the original frozen index JSON, never image/state/model data."""
    path = Path(path)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected_sha256 != FROZEN_INDEX_SHA256 or actual != expected_sha256:
        raise ValueError("The original frozen stage2C development index file changed")
    catalog = build_selection_catalog(json.loads(data)["development"], source_sha256=actual)
    if (catalog["summary"]["independent_trajectories"] != 32
            or catalog["summary"]["h1_motion"] != {"moving": 31, "near_static": 4, "intermediate": 1}
            or catalog["summary"]["h4_endpoint_moving"] != 32
            or catalog["summary"]["development_roles"] != {"natural": 18, "extra_moving": 18}):
        raise ValueError("Frozen36 selection denominators differ from the audited truth catalog")
    return catalog


def _catalog_windows(catalog):
    payload = {key: catalog[key] for key in ("kind", "source_sha256", "windows")}
    if payload["kind"] != CATALOG_KIND or _digest(payload) != catalog["catalog_sha256"]:
        raise ValueError("Frozen selection catalog was modified")
    _validate_windows(payload["windows"])
    return payload["windows"]


def assess_candidate(candidate, catalog, *, formal_experiment_id):
    """Return an auditable eligible/rank record; no missing row is dropped."""
    if not isinstance(formal_experiment_id, str) or not formal_experiment_id.strip():
        raise ValueError("An explicit formal experiment identity is required")
    windows = _catalog_windows(catalog)
    ids = [row["window_id"] for row in windows]
    update = candidate.get("update")
    role = candidate.get("candidate_role")
    reasons = []
    if candidate.get("formal_experiment_id") != formal_experiment_id:
        reasons.append("different_formal_experiment; microfit candidates are excluded")
    if (type(update) is not int or not ((role == "scheduled" and update in (250, 500))
            or (role == "early_stop" and 1 <= update < 500))):
        reasons.append("not_an_authorized_formal250_500_or_early_stop_candidate")
    if candidate.get("evaluation_complete") is not True:
        reasons.append("evaluation_incomplete")
    metrics = {}
    for horizon, moving in (("h1", {r["window_id"] for r in windows if r["h1_motion_class"] == "moving"}),
                            ("h4", {r["window_id"] for r in windows if r["h4_endpoint_moving"]})):
        records = candidate.get(horizon)
        if not isinstance(records, list) or len(records) != 36:
            reasons.append(horizon + "_requires_all36_records")
            continue
        if any(not isinstance(row, dict) for row in records) or [row.get("window_id") for row in records] != ids:
            reasons.append(horizon + "_record_IDs_or_order_differ")
            continue
        if any(type(row.get("hard_generation_failure")) is not bool
               or type(row.get("raw_object_error")) not in (int, float)
               or not math.isfinite(row["raw_object_error"]) or not 0 <= row["raw_object_error"] <= 1
               for row in records):
            reasons.append(horizon + "_invalid_failure_flag_or_finite_gray_error")
            continue
        errors = [float(row["raw_object_error"]) for row in records if row["window_id"] in moving]
        metrics[horizon] = dict(denominator=36, moving_denominator=len(errors),
            hard_generation_failures=sum(row["hard_generation_failure"] for row in records),
            moving_mean_gray_error=math.fsum(errors) / len(errors))
    eligible = not reasons and set(metrics) == {"h1", "h4"}
    failures = sum(row["hard_generation_failures"] for row in metrics.values()) if eligible else None
    balanced = .5 * metrics["h1"]["moving_mean_gray_error"] + .5 * metrics["h4"]["moving_mean_gray_error"] if eligible else None
    return dict(kind="native-rgb-formal-selection-candidate.v1", policy=POLICY,
        formal_experiment_id=formal_experiment_id, candidate_role=role, update=update,
        catalog_sha256=catalog["catalog_sha256"], eligible=eligible, ineligible_reasons=reasons,
        metrics=metrics, total_hard_generation_failures=failures, balanced_moving_gray_error=balanced,
        rank=[failures, balanced, update] if eligible else None,
        criterion="all72 hard failures; then0.5H1-moving+0.5H4-endpoint-moving gray1-IoU; then earlier formal update",
        qualification="Finite-candidate development selection only; not model, shape or generalization acceptance. Denoising loss is diagnostic only.")


def select_best(candidates, catalog, *, formal_experiment_id):
    """Assess original records again; never trust a cached/edited caller rank."""
    candidates = list(candidates)
    updates = [row.get("update") for row in candidates]
    if len(set(updates)) != len(updates):
        raise ValueError("Each formal update may have only one candidate evaluation")
    assessments = [assess_candidate(row, catalog, formal_experiment_id=formal_experiment_id) for row in candidates]
    eligible = [row for row in assessments if row["eligible"]]
    best = min(eligible, key=lambda row: tuple(row["rank"])) if eligible else None
    return dict(kind="native-rgb-formal-selection-result.v1", policy=POLICY,
        formal_experiment_id=formal_experiment_id, catalog_sha256=catalog["catalog_sha256"],
        best=deepcopy(best), eligible_candidates=len(eligible), candidates=assessments)
