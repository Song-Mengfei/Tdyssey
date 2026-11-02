"""Posthoc numeric diagnostics over forecast records, without image/model access.

Rows, saved physical labels and latest-state references are the only inputs.
AR and teacher-forced diagnostics remain separate. This module never computes
rankings or stopping scores, and never changes a predicted measurement.
"""

from collections import defaultdict

from .wm_compat_reporting import (
    ANCHORS, ARMS, CANDIDATES, SEEDS, PRIMARY_HORIZONS,
    _coverage, _distance, _errors, _finite, _point, _pose, _stats,
)
from .wm_measurement_contract import angle_error_degrees


METRICS = ("position", "angle", "pusher", "coverage")
STRATA = ("all", "contact", "no_contact", "rotation_gt_5", "near_095")
MAX_SERIALIZED_BYTES = 192*1024
COUNT_COLUMNS = ("planned_per_arm", "available_per_arm", "available_branches", "available_anchors",
                 "A_completed", "B_completed", "A_missing", "B_missing", "A_not_attempted", "B_not_attempted",
                 "paired_completed", "paired_unknown", "A_completed_branches", "B_completed_branches",
                 "A_completed_anchors", "B_completed_anchors")
METRIC_COLUMNS = (
    "metric", "common_rows_per_arm", "common_fraction_of_available", "common_branches", "common_anchors",
    "A_row_p90", "A_row_max", "B_row_p90", "B_row_max",
    "A_equal_anchor_mean", "B_equal_anchor_mean", "B_minus_A_equal_anchor_mean",
    "B_minus_A_anchor_p90", "B_minus_A_anchor_max",
    "persistence_common_rows_per_arm", "persistence_common_branches", "persistence_common_anchors",
    "persistence_row_p90", "persistence_row_max", "persistence_equal_anchor_mean",
    "A_minus_persistence_equal_anchor_mean", "B_minus_persistence_equal_anchor_mean",
)
APPEARANCE_COLUMNS = ("arm", "metric", "rows", "branches", "anchors", "row_p90", "row_max", "equal_anchor_mean")
REASONS = frozenset(("invalid_rgb", "unsupported_rgb_dtype", "object_missing", "scene_clipped", "object_area_mismatch",
                     "fragmented_object", "rigid_shape_mismatch", "pusher_missing", "ambiguous_multiple_pushers",
                     "pusher_size_or_shape_mismatch"))


def _get(record, key, default=None):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _truth(value):
    if not isinstance(value, dict):
        return None
    return dict(pose=_pose(value.get("pose")), pusher=_point(value.get("pusher")), coverage=_coverage(value.get("coverage")))


def _local_view(row):
    # reporting._measurement additionally derives predicted stopping. These
    # posthoc diagnostics, including TF, require only independent local flags.
    value = row.get("measurement") if isinstance(row.get("measurement"), dict) else {}
    parts = {name: value.get(name) if isinstance(value.get(name), dict) else {}
             for name in ("state", "shape", "pusher", "coverage")}
    pose, pusher = _pose(parts["state"].get("pose")), _point(parts["pusher"].get("position"))
    estimate, completed = _coverage(parts["coverage"].get("estimate")), row["status"] == "completed"
    flags = dict(object=bool(completed and parts["state"].get("valid") is True and pose is not None),
                 shape=bool(completed and parts["shape"].get("valid") is True),
                 pusher=bool(completed and parts["pusher"].get("valid") is True and pusher is not None))
    flags["object_shape"] = flags["object"] and flags["shape"]
    flags["coverage"] = bool(flags["object_shape"] and parts["coverage"].get("valid") is True and estimate is not None)
    return dict(flags=flags, pose=pose, pusher=pusher, estimate=estimate)


def _same_truth(first, second):
    for key in ("pose", "pusher", "coverage"):
        if first[key] is None or second[key] is None:
            continue
        if key == "pose":
            if (_distance(first[key]["position"], second[key]["position"]) > 1e-9
                    or angle_error_degrees(first[key]["angle"], second[key]["angle"]) > 1e-9):
                return False
        elif key == "pusher":
            if _distance(first[key], second[key]) > 1e-9:
                return False
        elif abs(first[key]-second[key]) > 1e-12:
            return False
    return True


def _hierarchy(entries):
    """(anchor,candidate,value): noise mean, branch mean, equal-anchor mean."""
    by_branch = defaultdict(list)
    for anchor, candidate, value in entries:
        by_branch[(anchor, candidate)].append(value)
    by_anchor = defaultdict(list)
    for (anchor, _), values in by_branch.items():
        by_anchor[anchor].append(sum(values)/len(values))
    anchor_values = [sum(values)/len(values) for values in by_anchor.values()]
    stats = _stats(anchor_values)
    return dict(rows=len(entries), branches=len(by_branch), anchors=len(by_anchor),
                mean=stats["mean"], p90=stats["p90"], max=stats["max"])


def _persistence_errors(latest, truth):
    latest, truth = _truth(latest), _truth(truth)
    result = dict.fromkeys(METRICS)
    if latest is None or truth is None:
        return result
    if latest["pose"] is not None and truth["pose"] is not None:
        result["position"] = _distance(latest["pose"]["position"], truth["pose"]["position"])
        result["angle"] = angle_error_degrees(latest["pose"]["angle"], truth["pose"]["angle"])
    if latest["pusher"] is not None and truth["pusher"] is not None:
        result["pusher"] = _distance(latest["pusher"], truth["pusher"])
    if latest["coverage"] is not None and truth["coverage"] is not None:
        result["coverage"] = abs(latest["coverage"]-truth["coverage"])
    return result


def _inventory(rows, branches):
    indexed, labels, branch_lookup = {}, defaultdict(list), {}
    for branch in branches:
        key = (_get(branch, "anchor"), _get(branch, "candidate"))
        if key[0] not in ANCHORS or key[1] not in CANDIDATES or key in branch_lookup:
            raise ValueError("Unexpected or duplicate physical branch identity")
        branch_lookup[key] = branch
    for row in rows:
        mode, horizon = row.get("mode"), row.get("horizon")
        key = (mode, row.get("arm"), row.get("anchor"), row.get("candidate"), row.get("seed"), horizon)
        if (mode not in ("AR", "TF") or row.get("arm") not in ARMS or row.get("anchor") not in ANCHORS
                or row.get("candidate") not in CANDIDATES or type(row.get("seed")) is not int or row["seed"] not in SEEDS
                or type(horizon) is not int or horizon not in ((1, 2, 3, 4) if mode == "AR" else (2, 4))
                or row.get("shape") != row["anchor"][0] or row.get("status") not in ("completed", "missing", "not_attempted")
                or key in indexed):
            raise ValueError("Unexpected or duplicate fixed forecast row identity")
        indexed[key] = row
        labels[(row["anchor"], row["candidate"], horizon)].append(row)
    physical = {}
    for anchor in ANCHORS:
        for candidate in CANDIDATES:
            branch = branch_lookup.get((anchor, candidate))
            latest = _get(branch, "latest")
            saved_truth = _get(branch, "truth")
            for horizon in PRIMARY_HORIZONS:
                available = not (anchor == "L01" and candidate == "hold")
                group = labels.get((anchor, candidate, horizon), [])
                truths = [_truth(r.get("truth")) for r in group]
                if isinstance(saved_truth, (list, tuple)) and len(saved_truth) >= horizon:
                    truths.insert(0, _truth(saved_truth[horizon-1]))
                truths = [value for value in truths if value is not None]
                truth = dict.fromkeys(("pose", "pusher", "coverage"))
                for value in truths:
                    if not _same_truth(truth, value):
                        raise ValueError("Conflicting physical truth labels across arms/noises/modes")
                    truth.update({key: item for key, item in value.items() if item is not None})
                contacts = {r["contact_by_h4"] for r in group if type(r.get("contact_by_h4")) is bool}
                if type(_get(branch, "contact_by_h4")) is bool:
                    contacts.add(_get(branch, "contact_by_h4"))
                if len(contacts) > 1:
                    raise ValueError("Conflicting physical contact labels across arms/noises/modes")
                contact = next(iter(contacts)) if contacts else None
                actual_pose, latest_pose = truth["pose"], _pose((latest or {}).get("pose")) if isinstance(latest, dict) else None
                rotation = angle_error_degrees(actual_pose["angle"], latest_pose["angle"]) if actual_pose and latest_pose else None
                physical[(anchor, candidate, horizon)] = dict(available=available, truth=truth, latest=latest,
                    contact=contact, no_contact=None if contact is None else not contact,
                    rotation_gt_5=None if rotation is None else rotation > 5,
                    near_095=None if truth["coverage"] is None else .92 <= truth["coverage"] <= .98,
                    persistence=_persistence_errors(latest, truth))
    return indexed, physical


def _metric_row(metric, pairs):
    common = [pair for pair in pairs if pair["errors"]["A"][metric] is not None and pair["errors"]["B"][metric] is not None]
    entry = lambda pair, value: (pair["anchor"], pair["candidate"], value)
    values = {arm: [p["errors"][arm][metric] for p in common] for arm in ARMS}
    hierarchy = {arm: _hierarchy([entry(p, p["errors"][arm][metric]) for p in common]) for arm in ARMS}
    difference = _hierarchy([entry(p, p["errors"]["B"][metric]-p["errors"]["A"][metric]) for p in common])
    persistent = [p for p in common if p["physical"]["persistence"][metric] is not None]
    persistence_values = [p["physical"]["persistence"][metric] for p in persistent]
    persistence = _hierarchy([entry(p, p["physical"]["persistence"][metric]) for p in persistent])
    persist_diff = {arm: _hierarchy([entry(p, p["errors"][arm][metric]-p["physical"]["persistence"][metric]) for p in persistent])["mean"] for arm in ARMS}
    a, b, base = _stats(values["A"]), _stats(values["B"]), _stats(persistence_values)
    return [metric, hierarchy["A"]["rows"], len(common)/len(pairs) if pairs else None, hierarchy["A"]["branches"], hierarchy["A"]["anchors"],
            a["p90"], a["max"], b["p90"], b["max"], hierarchy["A"]["mean"], hierarchy["B"]["mean"],
            difference["mean"], difference["p90"], difference["max"],
            persistence["rows"], persistence["branches"], persistence["anchors"], base["p90"], base["max"],
            persistence["mean"], persist_diff["A"], persist_diff["B"]]


def _small_stats(values):
    return [len(values), min(values) if values else None, max(values) if values else None,
            sum(values)/len(values) if values else None]


def _base_diagnostics(pairs, arm):
    reason_counts = {name: defaultdict(int) for name in ("state", "shape", "pusher", "coverage")}
    nuisance_fields = ("fitted_outline", "fitted_blur_sigma", "full_soft_iou", "area_ratio", "component_mass_fraction", "squared_residual")
    nuisance = {name: [] for name in nuisance_fields}
    raw = {name: [] for name in ("raw_min", "raw_max", "out_of_range_fraction")}
    for pair in pairs:
        row, view = pair["rows"][arm], pair["views"][arm]
        if row["status"] != "completed":
            continue
        measurement = row.get("measurement") if isinstance(row.get("measurement"), dict) else {}
        for part, flag in (("state", "object"), ("shape", "shape"), ("pusher", "pusher"), ("coverage", "coverage")):
            if not view["flags"][flag]:
                record = measurement.get(part) if isinstance(measurement.get(part), dict) else {}
                reason = record.get("reason")
                reason_counts[part][reason if isinstance(reason, str) and reason in REASONS else "reason_not_recorded" if reason is None else "other"] += 1
        shape = measurement.get("shape") if isinstance(measurement.get("shape"), dict) else {}
        for name in nuisance_fields:
            if _finite(shape.get(name)):
                nuisance[name].append(shape[name])
        for name, source in (("raw_min", "raw_min"), ("raw_max", "raw_max"), ("out_of_range_fraction", "clipped_fraction")):
            if _finite(row.get(source)):
                raw[name].append(row[source])
    return dict(invalid_reasons={name: dict(counts) for name, counts in reason_counts.items()},
                nuisance={name: _small_stats(values) for name, values in nuisance.items()},
                raw={name: _small_stats(values) for name, values in raw.items()})


def _cell(mode, shape, horizon, stratum, pairs):
    # The all cell retains the missing branch; posthoc strata cannot assign an
    # unknown physical label to it. Unknown-label counts are reported separately.
    chosen = pairs if stratum == "all" else [p for p in pairs if p["physical"].get(stratum) is True]
    available = [p for p in chosen if p["physical"]["available"]]
    statuses = {arm: [p["rows"][arm]["status"] for p in available] for arm in ARMS}
    counts = [len(chosen), len(available), len({(p["anchor"], p["candidate"]) for p in available}),
              len({p["anchor"] for p in available})]
    counts += [statuses[arm].count(status) for status in ("completed", "missing", "not_attempted") for arm in ARMS]
    completed_pairs = [p for p in available if all(p["rows"][arm]["status"] == "completed" for arm in ARMS)]
    counts += [len(completed_pairs), len(available)-len(completed_pairs)]
    counts += [len({(p["anchor"], p["candidate"]) for p in available if p["rows"][arm]["status"] == "completed"}) for arm in ARMS]
    counts += [len({p["anchor"] for p in available if p["rows"][arm]["status"] == "completed"}) for arm in ARMS]
    validity = {arm: {flag: sum(p["views"][arm]["flags"][flag] for p in available)
                      for flag in ("object", "shape", "object_shape", "pusher", "coverage")} for arm in ARMS}
    paired = {}
    for flag in ("object", "shape", "object_shape", "pusher", "coverage"):
        a, b = [p["views"]["A"]["flags"][flag] for p in completed_pairs], [p["views"]["B"]["flags"][flag] for p in completed_pairs]
        paired[flag] = [sum(y and not x for x, y in zip(a, b)), sum(x == y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b))]
    appearance = []
    for arm in ARMS:
        for metric in ("forecast_rgb_mse", "persistence_rgb_mse"):
            selected = []
            for p in available:
                row = p["rows"][arm]
                value = row.get("appearance", {}).get(metric) if isinstance(row.get("appearance"), dict) else None
                if row["status"] == "completed" and _finite(value) and value >= 0:
                    selected.append((p["anchor"], p["candidate"], value))
            stats, hierarchical = _stats([entry[2] for entry in selected]), _hierarchy(selected)
            appearance.append([arm, metric, len(selected), hierarchical["branches"], hierarchical["anchors"],
                               stats["p90"], stats["max"], hierarchical["mean"]])
    result = dict(mode=mode, shape=shape, horizon=horizon, stratum=stratum, counts=counts,
                  valid=validity, paired_validity_B_win_tie_loss=paired,
                  metrics=[_metric_row(metric, available) for metric in METRICS], appearance=appearance)
    if stratum == "all":
        result["base_only"] = {arm: _base_diagnostics(available, arm) for arm in ARMS}
    return result


def _screen(shape, horizon, pairs):
    available = [p for p in pairs if p["physical"]["available"]]
    completed = sum(p["rows"]["B"]["status"] == "completed" for p in available)
    joint = sum(p["views"]["B"]["flags"]["object_shape"] for p in available)
    required = 127 if shape == "L" else 44
    rows, checks = [], []
    for metric, threshold in (("position", 5), ("angle", 5), ("pusher", 5), ("coverage", .05)):
        values = [p["errors"]["B"][metric] for p in available if p["errors"]["B"][metric] is not None]
        stats = _stats(values)
        passed = stats["p90"] is not None and stats["p90"] <= threshold+1e-12
        checks.append(passed)
        rows.append([metric, len(values), len(available), len(values)/len(available) if available else None,
                     stats["p90"], stats["max"], threshold, passed if completed else None])
    labels_complete = all(all(p["physical"]["truth"][name] is not None for name in ("pose", "pusher", "coverage")) for p in available)
    shape_passed = joint >= required
    limited_passed = shape_passed and labels_complete and all(checks)
    return dict(mode="AR", arm="B", shape=shape, horizon=horizon,
                planned=len(pairs), available=len(available), original_missing=len(pairs)-len(available),
                completed=completed, object_shape_valid=joint, shape_required=required,
                shape_screen=dict(status="not_attempted" if not completed else "passed" if shape_passed else "failed",
                                  passed=shape_passed if completed else None),
                conditional_limited_screen=dict(status="not_attempted" if not completed else "passed" if limited_passed else "failed",
                                                passed=limited_passed if completed else None),
                truth_labels_complete=labels_complete, measurable_row_metrics=rows)


def forecast_diagnostics(rows, branches):
    """Compact development diagnostics; no ranking/stopping or prediction work.

    B-AR screens use its own measurable-row P90, not the cross-arm subset or
    anchor-mean P90. Every cross-arm error difference uses common measurable
    rows; persistence comparisons use the further common latest-label subset.
    No completed rows means not_attempted/unknown, not a failed experiment.
    """
    rows = list(rows)
    indexed, physical = _inventory(rows, branches)
    cells, screens, label_counts = [], [], []
    for mode, horizons in (("AR", PRIMARY_HORIZONS), ("TF", (2, 4))):
        for shape in ("L", "T"):
            for horizon in horizons:
                pairs = []
                for anchor in ANCHORS:
                    if anchor[0] != shape:
                        continue
                    for candidate in CANDIDATES:
                        label = physical[(anchor, candidate, horizon)]
                        for seed in SEEDS:
                            group = {}
                            for arm in ARMS:
                                key = (mode, arm, anchor, candidate, seed, horizon)
                                group[arm] = indexed.get(key, dict(mode=mode, arm=arm, anchor=anchor, candidate=candidate,
                                    seed=seed, horizon=horizon, shape=shape, status="missing", measurement=None))
                            views = {arm: _local_view(group[arm]) for arm in ARMS}
                            errors = {arm: _errors(dict(group[arm], truth=label["truth"]), views[arm]) for arm in ARMS}
                            pairs.append(dict(anchor=anchor, candidate=candidate, seed=seed, rows=group, views=views,
                                              errors=errors, physical=label))
                available = [p for p in pairs if p["physical"]["available"]]
                label_counts.append(dict(mode=mode, shape=shape, horizon=horizon,
                    labels={name: [sum(p["physical"][name] is value for p in available) for value in (True, False, None)]
                            for name in ("contact", "rotation_gt_5", "near_095")}))
                cells.extend(_cell(mode, shape, horizon, stratum, pairs) for stratum in STRATA)
                if mode == "AR":
                    screens.append(_screen(shape, horizon, pairs))
    return dict(schema_version=1, development_only=True, independent_validation=False,
        input_forecast_rows=len(rows), planned_forecast_rows=2304, available_forecast_rows=2268, original_missing_forecast_rows=36,
        primary_horizon_planned_rows=1920, primary_horizon_available_rows=1890, AR_H3_planned_rows_not_summarized=384,
        cells=cells, B_AR_screens=screens, label_counts=label_counts,
        columns=dict(counts=list(COUNT_COLUMNS), metrics=list(METRIC_COLUMNS), appearance=list(APPEARANCE_COLUMNS),
                     base_nuisance_and_raw=["rows", "min", "max", "mean"],
                     screen_metrics=["metric", "measured_rows", "available_rows", "measured_fraction", "row_p90", "row_max", "threshold", "passed"],
                     label_counts=["true_rows_per_arm", "false_rows_per_arm", "unknown_rows_per_arm"]),
        definitions=dict(primary_horizons=list(PRIMARY_HORIZONS), TF_horizons=[2, 4], TF_H1="AR first step reused; no independent TF call or missing cell",
            units=dict(position="world pixels", angle="degrees", pusher="world pixels", coverage="fraction"),
            strata=dict(contact="saved physical contact by H4", no_contact="saved physical no-contact through H4",
                        rotation_gt_5="strict wrapped actual rotation >5 degrees relative to branch.latest",
                        near_095=dict(lower=.92, upper=.98, inclusive=True, purpose="fixed development stratum; not terminal validation")),
            weighting="noise mean within branch, branch mean within anchor, equal anchor mean; reported row P90/max remain row statistics",
            differences="B minus A errors; negative favors B. Model minus privileged latest-state persistence; negative favors model, on the same further-common subset",
            validity="all available rows retain their denominator; B win/tie/loss only among paired completed records, with uncompleted pairs explicitly unknown",
            raw_range="out_of_range_fraction is the engine's clipped_fraction diagnostic; outputs were not clipped",
            shape_screen="B-AR >=127/141 L or >=44/48 T joint object/full-shape validity; no completed rows => not_attempted",
            limited_screen="conditional engineering screen: own measurable-row P90 <=5px position/pusher, <=5deg angle, <=.05 coverage; >=90% joint shape validity; fractions explicit",
            appearance="within each arm's color domain only; no cross-domain MSE motion claim",
            row_references="records/{key with slash replaced by hyphen}.json; absent chunks retain identity/status in planned-status.json",
            physical_labels="future truth only supplies posthoc errors/strata; no measurement, stopping, ranking, fitting or selection repair",
            excluded_claims=["precise terminal detection", "validated interval coverage", "general dynamics", "generalization", "policy gain"]))
