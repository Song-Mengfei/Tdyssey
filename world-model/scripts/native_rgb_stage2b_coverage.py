"""Rebuild stage2B-only and merged formal TRAIN contact evidence offline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_rgb_stage2a_coverage import (DEFINITIONS, LETTERS, WALLS, _reach_ids,
    _assert_same_bins, _sets_summary, _write_csv, _write_json, load_json, sha256)

V1 = "motion-room-actual-hit-prior-v1"
V2 = "motion-room-planned-attempts-v2"


def _blank_coverage():
    from tdyssey.native_rgb_data import contour_segments
    coverage = {letter: contour_segments(letter, 8) for letter in LETTERS}
    for rows in coverage.values():
        for row in rows:
            row.update(contact_control_count=0, moving_contact_control_count=0, trajectory_contact_controls={})
    return coverage


def _normalized_counts(counts, rows):
    if not isinstance(counts, dict):
        return None
    return {str(row["id"]): int(counts.get(row["id"], counts.get(str(row["id"]), 0))) for row in rows}


def _record_contact(letter, rows, points, angles, tid, moving, approach):
    from scripts.native_rgb_supervised_stage1 import contacted_bin_ids
    from tdyssey.native_rgb_data import _contact_coverage
    _contact_coverage(letter, rows, points, tid, moving=moving, direction=approach, angles=angles)
    hit = contacted_bin_ids(letter, rows, points)
    for arc in hit:
        row = rows[arc]
        row["contact_control_count"] += 1
        row["moving_contact_control_count"] += int(moving)
        row["trajectory_contact_controls"][tid] = row["trajectory_contact_controls"].get(tid, 0) + 1
    return hit


def _load_stage(stage, name):
    stage = Path(stage).resolve()
    if (stage / "COLLECTION_RUNNING.lock").exists():
        raise RuntimeError(f"{name} collection is still active; do not inspect partial data")
    manifest_path = stage / "data/manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("stage") != name or manifest.get("split") != "train":
        raise ValueError(f"Only {name} formal TRAIN is accepted")
    if any(entry.get("split") != "train" or entry.get("shape") not in LETTERS for entry in manifest["entries"]):
        raise ValueError("Probe, validation and held-out letters cannot enter formal TRAIN coverage")
    saved_coverage = load_json(stage / "contact-coverage.json")
    return stage, manifest, saved_coverage


def _historical_prior(stage2a):
    path = stage2a / "source-snapshot/historical-TRAIN-contact-coverage.json"
    prior = load_json(path)
    identity = load_json(stage2a / "source-identity.json")
    matches = [row for row in identity["files"] if row["snapshot"].endswith("historical-TRAIN-contact-coverage.json")]
    if len(matches) != 1 or matches[0]["sha256"] != sha256(path):
        raise ValueError("Historical prior must be the frozen historical TRAIN coverage")
    return prior, path


def _attempt_audit(entry, meta, rows, planned, actual):
    scene = meta.get("scene") or {}
    inputs = meta.get("selection_inputs") or {}
    expected_planned = _normalized_counts(dict(planned), rows)
    expected_actual = {str(row["id"]): int(row["contact_control_count"]) for row in actual}
    recorded_planned = _normalized_counts(inputs.get("planned_initial_attempts_before"), rows)
    recorded_actual = _normalized_counts(inputs.get("formal_train_contact_controls_before"), rows)
    evidence = scene.get("selection_evidence") or {}
    evidence_planned = _normalized_counts(evidence.get("planned_attempts_before"), rows)
    issues = []
    if scene.get("selection_version") != V2:
        issues.append("missing_or_incorrect_2B_selection_version")
    if recorded_planned != expected_planned:
        issues.append("planned_attempt_prefix_mismatch")
    if recorded_actual != expected_actual:
        issues.append("actual_contact_control_prefix_mismatch")
    if evidence_planned != expected_planned:
        issues.append("scene_planned_attempt_evidence_mismatch")
    if evidence.get("selected_arc") != scene.get("staging_segment"):
        issues.append("scene_selected_arc_evidence_mismatch")
    if entry.get("selected_initial_arc") != scene.get("staging_segment"):
        issues.append("entry_selected_arc_differs_from_scene")
    candidates = evidence.get("sorted_candidates") or []
    priorities = [tuple(row["priority"]) for row in candidates]
    if priorities != sorted(priorities):
        issues.append("saved_candidate_priorities_not_sorted")
    if candidates and candidates[0]["segment_id"] != scene.get("staging_segment"):
        issues.append("selected_arc_not_first_ranked")
    for candidate in candidates:
        arc = candidate["segment_id"]
        priority = candidate["priority"]
        if len(priority) != 8 or priority[1] != int(expected_planned[str(arc)]):
            issues.append("candidate_independent_attempt_factor_mismatch")
            break
        if priority[2] != int(expected_actual[str(arc)] > 0) or priority[4] != expected_actual[str(arc)]:
            issues.append("candidate_actual_contact_factor_mismatch")
            break
    selected = scene.get("staging_segment")
    return dict(trajectory_id=entry["trajectory_id"], letter=entry["shape"], selection_version=scene.get("selection_version"),
                selected_initial_arc=selected, selected_arc_attempts_before=expected_planned.get(str(selected)),
                selected_arc_actual_contact_controls_before=expected_actual.get(str(selected)),
                initial_scene_legal=scene.get("legal"), planned_attempt_increment=int(bool(scene.get("legal") and selected is not None)),
                matches=not issues, issues=issues, expected_planned_before=expected_planned,
                recorded_planned_before=recorded_planned, expected_actual_before=expected_actual,
                recorded_actual_before=recorded_actual)


def _coverage_differences(expected, saved, label):
    differences = []
    fields = ("contact_count", "moving_contact_count", "contact_control_count", "moving_contact_control_count",
              "requested_direction_counts", "actual_contact_angle_bins", "trajectory_contact_controls")
    for letter in LETTERS:
        _assert_same_bins(letter, expected[letter], saved[letter])
        for left, right in zip(expected[letter], saved[letter]):
            for field in fields:
                if left.get(field) != right.get(field):
                    differences.append(dict(comparison=label, letter=letter, arc=left["id"], field=field,
                                            recomputed=left.get(field), saved=right.get(field)))
            if set(left["trajectory_ids"]) != set(right["trajectory_ids"]):
                differences.append(dict(comparison=label, letter=letter, arc=left["id"], field="trajectory_ids",
                                        recomputed=sorted(left["trajectory_ids"]), saved=sorted(right["trajectory_ids"])))
    return differences


def _summarize(population, coverage, trajectories, prior):
    summaries, details, risks = [], {}, []
    for letter in LETTERS:
        rows = coverage[letter]
        _assert_same_bins(letter, rows, prior["segments"][letter])
        selected = [row for row in trajectories if row["letter"] == letter]
        H = {row["id"] for row in rows if row["contact_count"] > 0}
        R = _reach_ids(prior["canonical_reachability"][letter])
        E = {arc for trajectory in selected for arc in trajectory["initial_reachable_arc_ids"]}
        C = {row["segment_id"] for row in prior["canonical_reachability"][letter]
             if row["status"] == "proven_closed_hole_from_exterior"}
        counts, sets = _sets_summary(H, R, E, C, set(range(len(rows))))
        for evidence, conflicts in (("actual_hit", H & C), ("canonical_reachable", R & C), ("initial_ever_reachable", E & C)):
            if conflicts:
                risks.append(dict(population=population, letter=letter, issue=f"{evidence}_closed_hole_conflict", arcs=sorted(conflicts)))
        feature_counts = {}
        for feature in sorted({feature for row in rows for feature in row["features"]}):
            bins = {row["id"] for row in rows if feature in row["features"]}
            feature_counts[feature] = {name: len(bins & set(ids)) for name, ids in sets.items()}
            feature_counts[feature]["total"] = len(bins)
        actual_classes = Counter()
        actual_contact_classes = Counter()
        approaches = Counter()
        actual_gap_bands, actual_contact_gap_bands = Counter(), Counter()
        for trajectory in selected:
            actual_classes.update(trajectory["actual_control_spatial_class_counts"])
            actual_contact_classes.update(trajectory["actual_contact_control_spatial_class_counts"])
            approaches.update(trajectory["requested_approach_contact_control_counts"])
            actual_gap_bands.update(trajectory["actual_control_gap_band_counts"])
            actual_contact_gap_bands.update(trajectory["actual_contact_control_gap_band_counts"])
        summary = dict(population=population, letter=letter, **counts,
                       independent_trajectories=len(selected), independent_contact_trajectories=sum(t["actual_contact_controls"] > 0 for t in selected),
                       recorded_controls=sum(t["recorded_controls"] for t in selected), actual_contact_controls=sum(t["actual_contact_controls"] for t in selected),
                       actual_contact_control_arc_incidences=sum(row["contact_control_count"] for row in rows),
                       recorded_contact_points=sum(t["recorded_contact_points"] for t in selected), mapped_contact_points=sum(row["contact_count"] for row in rows),
                       planned_initial_arc_hit_trajectories=sum(t["planned_initial_arc_actually_hit"] for t in selected),
                       zero_contact_trajectories=[t["trajectory_id"] for t in selected if not t["actual_contact_controls"]],
                       actual_angle_point_counts=[sum(row["actual_contact_angle_bins"].get(str(index), 0) for row in rows) for index in range(8)],
                       actual_control_spatial_class_counts=dict(actual_classes), actual_contact_control_spatial_class_counts=dict(actual_contact_classes),
                       actual_control_gap_band_counts=dict(actual_gap_bands), actual_contact_control_gap_band_counts=dict(actual_contact_gap_bands),
                       historical_TRAIN_actual_hit_bins=sum(row["contact_count"] > 0 for row in prior["segments"][letter]),
                       requested_approach_contact_control_counts=dict(approaches))
        summaries.append(summary)
        details[letter] = dict(summary=summary, sets=sets, overlapping_feature_counts=feature_counts)
    return summaries, details, risks


def _focus(trajectories, coverages, details):
    result = {}
    for letter in ("O", "P"):
        arc = 55
        old_H = set(details["2A"][letter]["sets"]["actual_hit"])
        letter_result = {}
        for population in ("2A", "2B", "merged"):
            selected = [row for row in trajectories if row["letter"] == letter
                        and (population == "merged" or row["batch"] == population)]
            planned = [row for row in selected if row["planned_initial_arc"] == arc]
            hit = set(details[population][letter]["sets"]["actual_hit"])
            newly_hit_elsewhere = hit - old_H - {arc}
            planned55_other_hits = {bin_id for row in planned for bin_id in row["actual_hit_arc_ids"] if bin_id != arc}
            segment = coverages[population][letter][arc]
            letter_result[population] = dict(arc=arc, independent_trajectories=len(selected),
                planned_initial_attempts=len(planned), planned_initial_target_actually_hit=sum(row["planned_initial_arc_actually_hit"] for row in planned),
                any_trajectory_actual_hit_independent_trajectories=sum(arc in row["actual_hit_arc_ids"] for row in selected),
                actual_arc_contact_controls=segment["contact_control_count"], actual_arc_contact_points=segment["contact_count"],
                actual_total_hit_arcs=len(hit), actual_hit_arc_ids=sorted(hit),
                other_new_hit_arcs_beyond_2A=sorted(newly_hit_elsewhere),
                planned55_trajectories_actual_other_hit_arcs=sorted(planned55_other_hits),
                planned55_trajectories_actual_other_new_arcs_beyond_2A=sorted(planned55_other_hits-old_H),
                still_unconfirmed_arc_ids=details[population][letter]["sets"]["unconfirmed"],
                planned55_trajectory_ids=[row["trajectory_id"] for row in planned],
                planned55_actual_miss_trajectory_ids=[row["trajectory_id"] for row in planned if not row["planned_initial_arc_actually_hit"]])
        result[letter] = letter_result
    return dict(qualification="Arc55 is singled out only for the parent-requested report, never in the sampler. Planned attempts and true collision hits are distinct. New arc growth compares actual sets against2A, not target success rates.", per_letter=result)


def _markdown(report, rows):
    lines = ["# Formal TRAIN contact coverage: stage2B and merged2A+2B", "",
             "H=actual distinct collision-hit bins; R=frozen historical TRAIN canonical normal reachability; E=recorded legal initial-pose reachability union; K=R∪E∪H; U=all−K−closed holes.",
             "K is a global across observed poses reachable lower bound, not all physically reachable bins. Unknown bins remain unconfirmed. Repeated contacts do not enlarge H.", "",
             "| Batch | Letter | H/all | H∩R/R | H/K | U | Closed | Contact traj. | Contact controls | Planned initial arc hit |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['population']} | {row['letter']} | {row['actual_hit_bins']}/{row['all_bins']} | "
                     f"{row['canonical_hit_bins']}/{row['canonical_known_reachable_bins']} | {row['actual_hit_bins']}/{row['reachable_lower_bound_bins']} | "
                     f"{row['unconfirmed_bins']} | {row['proven_closed_hole_bins']} | {row['independent_contact_trajectories']}/{row['independent_trajectories']} | "
                     f"{row['actual_contact_controls']}/{row['recorded_controls']} | {row['planned_initial_arc_hit_trajectories']}/{row['independent_trajectories']} |")
    lines.extend(["", f"Prefix attempt audit mismatches: {report['attempt_prefix_audit']['mismatching_trajectories']}. Saved merged coverage differences: {len(report['saved_coverage_differences'])}. Reachability conflicts: {len(report['reachability_conflicts'])}.",
                  "", "O/P arc55 attempts, actual hits and other arc growth are in `O-P-arc55-focus.json/md`. The initial target is a planned contact arc, not pusher click arrival.",
                  "All actual per-control spatial classes and four signed wall gaps are retained in the trajectory/gap table and actual contact cube. No wall-arbiter points were stored, so wall-gap proximity is not a true wall-collision count.",
                  "Actual contact-normal angle bins count real pre-solve points; requested approach labels are reported separately. Feature flags overlap. Old TRAIN prior and independent initial planned attempts are never inserted into actual coverage.",
                  "The historical TRAIN prior had22/23 trajectories/letter;2A had9,2B adds13,merged has22 when complete. The independent scenes/budgets differ; old and new H are not a matched causal comparison.",
                  "", f"Postprocessor SHA256: `{report['source_sha256']}`. Zero environment construction, physics steps, renderer/model calls or source/data changes."])
    return "\n".join(lines)+"\n"


def build_coverage(stage2b, *, report_output=None, stage2a=None, svg=True):
    """Build fresh derived files after root signals final manifests are ready."""
    import numpy as np
    from tdyssey.native_rgb_motion_statistics import area_centroids, transition_rows

    stage2b = Path(stage2b).resolve()
    stage2a = Path(stage2a).resolve() if stage2a is not None else stage2b.parent / "stage2a"
    report_output = Path(report_output).resolve() if report_output is not None else stage2b / "report/coverage"
    if report_output.exists():
        raise FileExistsError("Do not overwrite derived evidence; use a new report directory")
    a_root, a_manifest, a_saved = _load_stage(stage2a, "2A")
    b_root, b_manifest, b_saved = _load_stage(stage2b, "2B")
    prior, prior_path = _historical_prior(a_root)
    coverages = {population: _blank_coverage() for population in ("2A", "2B", "merged")}
    trajectories, audits, cube = [], [], []
    planned = {letter: Counter() for letter in LETTERS}
    ids = set()
    grouped = defaultdict(lambda: dict(controls=set(), trajectories=set()))
    sources = {}
    for batch, root, manifest in (("2A", a_root, a_manifest), ("2B", b_root, b_manifest)):
        sources[batch] = dict(manifest_sha256=sha256(root/"data/manifest.json"),
                              saved_contact_coverage_sha256=sha256(root/"contact-coverage.json"),
                              source_identity_sha256=manifest.get("source_identity_sha256"))
        identity_path = root/"source-identity.json"
        if identity_path.exists() and manifest.get("source_identity_sha256") != sha256(identity_path):
            raise ValueError(f"{batch} source identity differs from manifest")
        for entry in manifest["entries"]:
            tid, letter = entry["trajectory_id"], entry["shape"]
            if tid in ids:
                raise ValueError("Formal TRAIN stages contain duplicate trajectory IDs")
            ids.add(tid)
            path, metadata_path = root/"data"/entry["file"], root/"data"/entry["metadata_file"]
            if sha256(path) != entry["sha256"] or sha256(metadata_path) != entry["metadata_sha256"]:
                raise ValueError(f"Immutable raw NPZ/metadata identity mismatch: {tid}")
            meta = load_json(metadata_path)
            scene, steps = meta.get("scene") or {}, meta.get("steps", [])
            if batch == "2B":
                audits.append(_attempt_audit(entry, meta, coverages[batch][letter], planned[letter], coverages["merged"][letter]))
            initial_arc = scene.get("staging_segment")
            if scene.get("legal") and initial_arc is not None:
                planned[letter][int(initial_arc)] += 1
            with np.load(path, allow_pickle=False) as arrays_file:
                arrays = {key: arrays_file[key] for key in arrays_file.files}
            n = len(arrays["actions"])
            points, angles, offsets = arrays["contact_points"], arrays["contact_angles"], arrays["contact_offsets"]
            if (len(steps) != n or len(offsets) != n+1 or int(offsets[-1]) != len(points) or len(angles) != len(points)):
                raise ValueError(f"Raw contact/metadata timeline mismatch: {tid}")
            transition_stats = transition_rows(entry, arrays, steps)
            centroids = area_centroids(letter, arrays["states"])
            hit_arcs = set()
            actual_classes, actual_contact_classes, approaches = Counter(), Counter(), Counter()
            actual_gap_bands, actual_contact_gap_bands = Counter(), Counter()
            for control, (stat, step) in enumerate(zip(transition_stats, steps)):
                start, end = int(offsets[control]), int(offsets[control+1])
                approach = str(step.get("approach_kind", "unknown"))
                moving = stat["motion_class"] == "moving"
                hit = _record_contact(letter, coverages[batch][letter], points[start:end], angles[start:end], tid, moving, approach)
                merged_hit = _record_contact(letter, coverages["merged"][letter], points[start:end], angles[start:end], tid, moving, approach)
                if hit != merged_hit:
                    raise AssertionError("Pure collision mapping changed between population counters")
                hit_arcs |= hit
                actual_classes[stat["actual_spatial_class_before"]] += 1
                actual_gap_bands[stat["gap_band_before"]] += 1
                if stat["contact"]:
                    actual_contact_classes[stat["actual_spatial_class_before"]] += 1
                    approaches[approach] += 1
                    actual_contact_gap_bands[stat["gap_band_before"]] += 1
                delta = centroids[control+1]-centroids[control]
                for arc in sorted(hit):
                    row = dict(row_id=f"{tid}:{control}:{arc}", batch=batch, split="train", letter=letter,
                        trajectory_id=tid, transition=control, actual_arc_bin=arc, planned_initial_arc=initial_arc,
                        requested_current_arc=step.get("target_bin"), requested_approach=approach,
                        requested_side_sign=step.get("approach_side_sign"), phase=step.get("phase"),
                        requested_signed_action_angle_rad=step.get("signed_action_angle_from_inward_normal_rad"),
                        target_distance_class=step.get("target_distance_class"), target_distance_world_px=stat["target_distance_world_px"],
                        motion_class=stat["motion_class"], area_centroid_dx_world_px=float(delta[0]), area_centroid_dy_world_px=float(delta[1]),
                        translation_world_px=stat["translation_world_px"], rotation_abs_deg=stat["rotation_abs_deg"], raw_rotation_rad=stat["raw_dtheta_rad"],
                        valid_transition=stat["valid_transition"], valid_h1_candidate=stat["valid_h1_candidate"],
                        requested_spatial_class=entry.get("spatial_class", entry.get("location")),
                        actual_spatial_class_before=stat["actual_spatial_class_before"], actual_spatial_class_after=stat["actual_spatial_class_after"],
                        gap_band_before=stat["gap_band_before"], gap_band_after=stat["gap_band_after"],
                        min_wall_gap_before=stat["min_wall_gap_before"], min_wall_gap_after=stat["min_wall_gap_after"],
                        recorded_contact_points_in_control=end-start)
                    for wall in WALLS:
                        row[f"initial_{wall}_gap_world_px"] = (scene.get("actual_wall_gaps") or {}).get(wall)
                        row[f"{wall}_gap_before"] = stat[f"{wall}_gap_before"]
                        row[f"{wall}_gap_after"] = stat[f"{wall}_gap_after"]
                    cube.append(row)
                    for population in (batch, "merged"):
                        key = (population, letter, arc, approach, stat["motion_class"], stat["gap_band_before"])
                        grouped[key]["controls"].add((tid, control))
                        grouped[key]["trajectories"].add(tid)
            real_contact = sum(row["contact"] is True for row in transition_stats)
            trajectory = dict(batch=batch, letter=letter, trajectory_id=tid, selected_initial_arc=entry.get("selected_initial_arc"),
                planned_initial_arc=initial_arc, planned_initial_arc_actually_hit=initial_arc in hit_arcs,
                actual_hit_arc_ids=sorted(hit_arcs), distinct_real_hit_bins=len(hit_arcs), actual_contact_controls=real_contact,
                any_real_contact=real_contact>0, recorded_controls=n, recorded_contact_points=len(points),
                selection_version=scene.get("selection_version", V1 if batch=="2A" else None),
                requested_spatial_class=entry.get("spatial_class", entry.get("location")), requested_gap_band=entry.get("gap_band_requested"),
                requested_gap_world_px=entry.get("gap_requested_world_px"), initial_scene_legal=scene.get("legal"),
                initial_min_wall_gap_world_px=scene.get("initial_min_wall_gap_world_px"),
                initial_reachable_arc_ids=sorted(_reach_ids(scene.get("reachability", []))) if scene.get("legal") else [],
                actual_control_spatial_class_counts=dict(actual_classes), actual_contact_control_spatial_class_counts=dict(actual_contact_classes),
                actual_initial_spatial_class=transition_stats[0]["actual_spatial_class_before"] if transition_stats else "unrecorded",
                actual_control_gap_band_counts=dict(actual_gap_bands), actual_contact_control_gap_band_counts=dict(actual_contact_gap_bands),
                requested_approach_contact_control_counts=dict(approaches), error=entry.get("error"))
            trajectory.update({f"initial_{wall}_gap_world_px": (scene.get("actual_wall_gaps") or {}).get(wall) for wall in WALLS})
            trajectories.append(trajectory)

    differences = _coverage_differences(coverages["2A"], a_saved["segments"], "2A reconstructed vs saved2A")
    differences += _coverage_differences(coverages["merged"], b_saved["segments"], "merged reconstructed vs saved2B inherited2A+2B")
    summary_rows, per_population, reachability_risks = [], {}, []
    for population in ("2A", "2B", "merged"):
        selected = trajectories if population == "merged" else [row for row in trajectories if row["batch"] == population]
        rows, detail, risks = _summarize(population, coverages[population], selected, prior)
        summary_rows.extend(rows)
        per_population[population] = detail
        reachability_risks.extend(risks)
    for population in ("2A", "2B", "merged"):
        for letter in LETTERS:
            H = set(per_population[population][letter]["sets"]["actual_hit"])
            H_a = set(per_population["2A"][letter]["sets"]["actual_hit"])
            per_population[population][letter]["actual_new_hit_arcs_beyond_2A"] = sorted(H-H_a)
    focus = _focus(trajectories, coverages, per_population)
    groups = [dict(population=key[0], letter=key[1], actual_arc_bin=key[2], requested_approach=key[3],
                   motion_class=key[4], actual_gap_band_before=key[5], distinct_contact_controls=len(value["controls"]),
                   independent_trajectories=len(value["trajectories"]), trajectory_ids=sorted(value["trajectories"]))
              for key, value in sorted(grouped.items())]
    report = dict(kind="native-rgb-stage2b-raw-reconstructed-coverage.v1", definitions=DEFINITIONS,
        source_sha256=sha256(__file__), source_path=str(Path(__file__).resolve()), sources=sources,
        frozen_historical_TRAIN_prior=dict(path=str(prior_path), sha256=sha256(prior_path)),
        helper_source_sha256={name: sha256(ROOT/name) for name in ("scripts/native_rgb_stage2a_coverage.py",
            "scripts/native_rgb_supervised_stage1.py", "tdyssey/native_rgb_data.py", "tdyssey/native_rgb_motion_statistics.py")},
        per_population=per_population, saved_coverage_differences=differences, reachability_conflicts=reachability_risks,
        attempt_prefix_audit=dict(trajectories=len(audits), mismatching_trajectories=sum(not row["matches"] for row in audits),
            planned_attempts_initialized_from="2A actual independent formal TRAIN initial targets only",
            per_started_legal_trajectory_increment=1, counts_not_hits=True),
        final_planned_attempts={letter: dict(sorted(values.items())) for letter, values in planned.items()},
        zero_contact_trajectories=[row["trajectory_id"] for row in trajectories if not row["actual_contact_controls"]],
        contact_control_arc_incidences=len(cube), independent_trajectories=len(trajectories),
        collection_failures={"2A": a_manifest.get("failure"), "2B": b_manifest.get("failure")},
        reconstruction_qualification="Saved float32 contact points/angles are replayed; any bin/angle edge quantization differences are explicitly listed, never hidden or used as a physical gate override.",
        physics_controls=0, environment_instances=0, renderer_calls=0, model_calls=0, original_data_or_controller_modified=False)
    report_output.mkdir(parents=True, exist_ok=False)
    _write_json(report_output/"coverage-summary.json", report)
    _write_csv(report_output/"coverage-summary.csv", summary_rows)
    _write_csv(report_output/"target-contact-comparison.csv", trajectories)
    _write_csv(report_output/"initial-scene-gaps.csv", [{key: value for key, value in row.items()
        if key in ("batch", "letter", "trajectory_id", "requested_spatial_class", "requested_gap_band", "requested_gap_world_px", "initial_scene_legal",
                   "initial_min_wall_gap_world_px", "initial_reachable_arc_ids") or key.startswith("initial_") and key.endswith("gap_world_px")} for row in trajectories])
    _write_csv(report_output/"contact-controls.csv", cube)
    _write_csv(report_output/"contact-cube-summary.csv", groups)
    _write_json(report_output/"attempt-prefix-audit.json", audits)
    _write_csv(report_output/"attempt-prefix-audit.csv", [{key: value for key, value in row.items()
        if not key.endswith("_before") or key.startswith("selected_arc_")} for row in audits])
    _write_json(report_output/"O-P-arc55-focus.json", focus)
    focus_lines = ["# O/P arc55: plans, real collisions and other arc growth", "", focus["qualification"], "",
                   "| Letter | Batch | Planned55 attempts | Planned55 real hit | Any traj55 hit | Arc55 controls | All hit arcs | Other new arcs vs2A |",
                   "|---|---|---:|---:|---:|---:|---:|---:|"]
    for letter, populations in focus["per_letter"].items():
        for population, value in populations.items():
            focus_lines.append(f"| {letter} | {population} | {value['planned_initial_attempts']} | {value['planned_initial_target_actually_hit']} | "
                               f"{value['any_trajectory_actual_hit_independent_trajectories']} | {value['actual_arc_contact_controls']} | "
                               f"{value['actual_total_hit_arcs']} | {value['other_new_hit_arcs_beyond_2A']} |")
    with (report_output/"O-P-arc55-focus.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(focus_lines)+"\n")
    with (report_output/"coverage-summary.md").open("x", encoding="utf-8") as stream:
        stream.write(_markdown(report, summary_rows))
    if svg:
        from tdyssey.native_rgb_data import _write_coverage
        for population in ("2B", "merged"):
            _write_coverage(report_output/f"svg-{population}", coverages[population], prior["canonical_reachability"])
    _write_json(report_output/"report-files.json", dict(source_sha256=sha256(__file__), files=[
        dict(file=path.relative_to(report_output).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))
        for path in sorted(report_output.rglob("*")) if path.is_file()]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Completed stage2B root")
    parser.add_argument("--output")
    parser.add_argument("--stage2a")
    parser.add_argument("--no-svg", action="store_true")
    args = parser.parse_args(argv)
    result = build_coverage(args.stage, report_output=args.output, stage2a=args.stage2a, svg=not args.no_svg)
    print(json.dumps(dict(independent_trajectories=result["independent_trajectories"],
        attempt_audit_mismatches=result["attempt_prefix_audit"]["mismatching_trajectories"],
        saved_coverage_differences=len(result["saved_coverage_differences"]), reachability_conflicts=len(result["reachability_conflicts"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
