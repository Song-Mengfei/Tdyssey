"""Offline TRAIN-only stage2A contact coverage; never constructs or steps an env."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LETTERS = tuple("ABCEFHIJLMOPSTUVXY")
WALLS = ("left", "right", "top", "bottom")
DEFINITIONS = {
    "H": "Distinct 8-world-pixel arc bins hit by this stage2A batch's actual collision points.",
    "R": "Canonical normal-approach reachable bins from the frozen historical TRAIN coverage only.",
    "E": "Union of normal-approach reachable bins in the recorded legal initial stage2A scenes.",
    "K": "R union E union H: global across observed poses reachable lower bound, not all physically reachable bins.",
    "C": "Geometrically closed hole bins, proven inaccessible to the pusher from the exterior.",
    "U": "All bins minus K minus C: unconfirmed, not proven unreachable.",
    "planned_target": "The initially selected contact arc, not arrival at an absolute pusher click target.",
    "angles": "Eight bins of actual pre-solve relative-velocity/contact-normal angle; point counts, not independent trials.",
    "cube": "One row per actual (trajectory, control, arc-bin) hit; one control may hit several bins.",
    "walls": "Signed object-bounding gaps to inner walls 7/504. No wall-arbiter points were retained; no actual wall-collision count is claimed.",
    "priors": "Historical TRAIN counts selected targets only; they are never added to stage2A actual contact counts.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def percent(numerator, denominator):
    value = fraction(numerator, denominator)
    return None if value is None else 100 * value


def _number(value):
    if value in (None, "", "None", "null"):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _bool(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    return None


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _assert_same_bins(letter, current, historical):
    if len(current) != len(historical):
        raise ValueError(f"{letter}: historical/current contour denominator differs")
    for index, (new, old) in enumerate(zip(current, historical)):
        for key in ("id", "ring", "ring_bin"):
            if new[key] != old[key] or new["id"] != index:
                raise ValueError(f"{letter}: fixed arc identity differs at {index}/{key}")
        for key in ("arc_start", "arc_end"):
            if abs(float(new[key]) - float(old[key])) > 1e-6:
                raise ValueError(f"{letter}: fixed arc geometry differs at {index}/{key}")
        for key in ("point", "normal"):
            if len(new[key]) != len(old[key]) or any(abs(float(a) - float(b)) > 1e-6
                                                    for a, b in zip(new[key], old[key])):
                raise ValueError(f"{letter}: fixed arc geometry differs at {index}/{key}")


def _reach_ids(rows):
    return {int(row["segment_id"]) for row in rows if row["status"].startswith("reachable")}


def _stats_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for row in rows:
        key = (row["trajectory_id"], int(row["transition"]))
        if key in output:
            raise ValueError(f"Duplicate transition statistics row: {key}")
        output[key] = row
    return output


def _sets_summary(H, R, E, C, all_bins):
    K = R | E | H
    U = all_bins - K - C
    sets = dict(actual_hit=H, canonical_known_reachable=R, initial_pose_ever_reachable=E,
                reachable_lower_bound=K, unconfirmed=U, proven_closed_hole=C,
                all_unhit=all_bins-H, canonical_unhit=R-H, initial_ever_unhit=E-H,
                known_lower_bound_unhit=K-H, old_canonical_unknown=all_bins-R-C)
    counts = {name + "_bins": len(values) for name, values in sets.items()}
    counts.update(all_bins=len(all_bins), actual_hit_percent_all=percent(len(H), len(all_bins)),
                  actual_hit_percent_canonical=percent(len(H & R), len(R)),
                  actual_hit_percent_initial_ever=percent(len(H & E), len(E)),
                  actual_hit_percent_known_lower_bound=percent(len(H), len(K)),
                  canonical_hit_bins=len(H & R), initial_ever_hit_bins=len(H & E),
                  unconfirmed_percent_all=percent(len(U), len(all_bins)),
                  proven_closed_percent_all=percent(len(C), len(all_bins)))
    return counts, {key: sorted(value) for key, value in sets.items()}


def derive(stage, statistics, historical_train, historical_manifest):
    """Read final immutable inputs only. This function never opens a ledger."""
    import numpy as np
    from scripts.native_rgb_supervised_stage1 import contacted_bin_ids
    from tdyssey.native_rgb_motion_statistics import area_centroids

    stage, statistics = Path(stage).resolve(), Path(statistics).resolve()
    if (stage / "COLLECTION_RUNNING.lock").exists():
        raise RuntimeError("Collection is still active; wait for the root's explicit final-manifest signal")
    data = stage / "data"
    manifest_path, coverage_path = data / "manifest.json", stage / "contact-coverage.json"
    manifest, coverage = load_json(manifest_path), load_json(coverage_path)
    report_marker = statistics.parent.parent / "stage2a-report.json"
    completed_statistics = load_json(report_marker)
    if completed_statistics.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Completed H1 report marker is missing or bound to another manifest")
    prior, old_manifest = load_json(historical_train), load_json(historical_manifest)
    if manifest.get("stage") != "2A" or manifest.get("split") != "train":
        raise ValueError("Only formal stage2A TRAIN is supported")
    if any(entry.get("split") != "train" or entry.get("shape") not in LETTERS
           for entry in manifest["entries"]):
        raise ValueError("No probes, validation or held-out letters are allowed")
    if set(coverage["segments"]) != set(LETTERS):
        raise ValueError("Expected fixed 18-letter coverage table")
    stats = _stats_rows(statistics)
    sources = {"report_script": dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
               "manifest": dict(path=str(manifest_path), sha256=sha256(manifest_path)),
               "contact_coverage": dict(path=str(coverage_path), sha256=sha256(coverage_path)),
               "historical_TRAIN_coverage": dict(path=str(Path(historical_train).resolve()), sha256=sha256(historical_train)),
               "historical_manifest": dict(path=str(Path(historical_manifest).resolve()), sha256=sha256(historical_manifest)),
               "transition_statistics": dict(path=str(statistics), sha256=sha256(statistics)),
               "completed_statistics_marker": dict(path=str(report_marker), sha256=sha256(report_marker)),
               "contact_mapping_helper": dict(path="scripts/native_rgb_supervised_stage1.py", sha256=sha256(ROOT/"scripts/native_rgb_supervised_stage1.py")),
               "area_centroid_helper": dict(path="tdyssey/native_rgb_motion_statistics.py", sha256=sha256(ROOT/"tdyssey/native_rgb_motion_statistics.py"))}
    identity_path = stage / "source-identity.json"
    if identity_path.exists():
        sources["collection_source_identity"] = dict(path=str(identity_path), sha256=sha256(identity_path))
        if manifest.get("source_identity_sha256") != sha256(identity_path):
            raise ValueError("Manifest/source identity mismatch")
        frozen = [row for row in load_json(identity_path)["files"]
                  if row["snapshot"].endswith("historical-TRAIN-contact-coverage.json")]
        if len(frozen) != 1 or frozen[0]["sha256"] != sha256(historical_train):
            raise ValueError("Historical TRAIN coverage is not the collection's frozen prior")
    by_letter = defaultdict(list)
    ids = set()
    for entry in manifest["entries"]:
        if entry["trajectory_id"] in ids:
            raise ValueError("Duplicate stage2A trajectory ID")
        ids.add(entry["trajectory_id"])
        by_letter[entry["shape"]].append(entry)
    if any(tid not in ids for tid, _ in stats):
        raise ValueError("Transition statistics contain a trajectory outside this batch")

    summaries, targets, gaps, cube, risks = [], [], [], [], []
    details, recomputed_bin_controls, recomputed_bin_trajs = {}, Counter(), defaultdict(set)
    control_group_counts = defaultdict(lambda: dict(controls=set(), trajectories=set(), incidences=0))
    expected_stats_keys = set()
    for letter in LETTERS:
        segments = coverage["segments"][letter]
        old_segments = prior["segments"][letter]
        _assert_same_bins(letter, segments, old_segments)
        all_bins = set(range(len(segments)))
        H = {row["id"] for row in segments if row["contact_count"] > 0}
        old_H = {row["id"] for row in old_segments if row["contact_count"] > 0}
        reach_rows = prior["canonical_reachability"][letter]
        R = _reach_ids(reach_rows)
        C = {row["segment_id"] for row in reach_rows if row["status"] == "proven_closed_hole_from_exterior"}
        ring_closed = {row["id"] for row in segments if row["ring"] > 0}
        if C != ring_closed:
            risks.append(dict(letter=letter, issue="closed_hole_ring_vs_frozen_proof_mismatch", bins=sorted(C ^ ring_closed)))
        E = set()
        contact_controls = controls = points_count = contact_trajectories = 0
        all_angles, actual_approaches, direction_sides = Counter(), Counter(), Counter()
        zero_contact, letter_gaps, letter_targets = [], [], []
        mapped_trajs = {tid for row in segments for tid in row["trajectory_ids"]}
        if not mapped_trajs <= ids:
            risks.append(dict(letter=letter, issue="coverage_contains_foreign_trajectory_ids", ids=sorted(mapped_trajs-ids)))
        for entry in by_letter[letter]:
            tid = entry["trajectory_id"]
            npz_path, metadata_path = data / entry["file"], data / entry["metadata_file"]
            if sha256(npz_path) != entry["sha256"] or sha256(metadata_path) != entry["metadata_sha256"]:
                raise ValueError(f"Saved trajectory identity mismatch: {tid}")
            meta = load_json(metadata_path)
            scene, steps = meta.get("scene") or {}, meta["steps"]
            reachable = _reach_ids(scene.get("reachability", [])) if scene.get("legal") else set()
            E |= reachable
            initial_gaps = scene.get("actual_wall_gaps") or {}
            gap_row = dict(letter=letter, trajectory_id=tid, requested_spatial_class=entry["spatial_class"],
                           requested_gap_band=entry["gap_band_requested"], requested_gap_world_px=entry["gap_requested_world_px"],
                           initial_scene_legal=scene.get("legal"), initial_min_wall_gap_world_px=scene.get("initial_min_wall_gap_world_px"),
                           initial_ever_reachable_bins=sorted(reachable))
            gap_row.update({f"initial_{wall}_gap_world_px": initial_gaps.get(wall) for wall in WALLS})
            gaps.append(gap_row)
            letter_gaps.append(gap_row)
            with np.load(npz_path, allow_pickle=False) as arrays:
                actions, states = arrays["actions"], arrays["states"]
                contacts, offsets = arrays["contact"], arrays["contact_offsets"]
                points, angles = arrays["contact_points"], arrays["contact_angles"]
                n = len(actions)
                if (len(states) != n+1 or len(steps) != n or len(contacts) != n or len(offsets) != n+1
                        or int(offsets[-1]) != len(points) or len(angles) != len(points)):
                    raise ValueError(f"Recorded timeline/contact offsets mismatch: {tid}")
                count = int(np.count_nonzero(contacts > 0))
                controls += n
                contact_controls += count
                contact_trajectories += int(count > 0)
                points_count += len(points)
                if not count:
                    zero_contact.append(tid)
                for angle in angles:
                    all_angles[int(((float(angle)+math.pi) % (2*math.pi))/(math.pi/4))] += 1
                centroids = area_centroids(letter, states)
                actual_bins = set()
                for transition, step in enumerate(steps):
                    key = (tid, transition)
                    expected_stats_keys.add(key)
                    if key not in stats:
                        raise ValueError(f"Missing frozen transition statistics: {key}")
                    stat = stats[key]
                    if stat["letter"] != letter or stat.get("split") != "train":
                        raise ValueError(f"Transition statistics split/letter mismatch: {key}")
                    hit = contacted_bin_ids(letter, segments, points[offsets[transition]:offsets[transition+1]])
                    actual_bins |= hit
                    if bool(hit) and not bool(contacts[transition] > 0):
                        risks.append(dict(letter=letter, issue="actual_mapped_points_but_contact_flag_false", trajectory_id=tid, transition=transition))
                    approach = step.get("approach_kind", "unknown")
                    if contacts[transition] > 0:
                        actual_approaches[approach] += 1
                        direction_sides[str(step.get("approach_side_sign", "unknown"))] += 1
                    delta = centroids[transition+1]-centroids[transition]
                    for arc in sorted(hit):
                        recomputed_bin_controls[(letter, arc, tid)] += 1
                        recomputed_bin_trajs[(letter, arc)].add(tid)
                        row = dict(row_id=f"{tid}:{transition}:{arc}", split="train", letter=letter,
                                   trajectory_id=tid, transition=transition, actual_arc_bin=arc,
                                   planned_initial_arc=scene.get("staging_segment"), requested_current_arc=step.get("target_bin"),
                                   requested_approach=approach, requested_side_sign=step.get("approach_side_sign"),
                                   requested_signed_action_angle_rad=step.get("signed_action_angle_from_inward_normal_rad"),
                                   phase=step.get("phase"), intent=step.get("intent"),
                                   requested_target_distance_class=step.get("target_distance_class"),
                                   actual_target_distance_world_px=step.get("actual_target_distance_world_px"),
                                   motion_class=stat["motion_class"], area_centroid_dx_world_px=float(delta[0]),
                                   area_centroid_dy_world_px=float(delta[1]), translation_world_px=_number(stat["translation_world_px"]),
                                   raw_rotation_rad=float(states[transition+1,4]-states[transition,4]),
                                   rotation_abs_rad=_number(stat["rotation_abs_rad"]), rotation_abs_deg=_number(stat["rotation_abs_deg"]),
                                   valid_transition=bool(arrays["valid_transition"][transition]),
                                   valid_h1_candidate=_bool(stat["valid_h1_candidate"]),
                                   requested_spatial_class=entry["spatial_class"], requested_gap_band=entry["gap_band_requested"],
                                   requested_gap_world_px=entry["gap_requested_world_px"],
                                   actual_spatial_class_before=stat.get("actual_spatial_class_before"),
                                   min_wall_gap_before=_number(stat["min_wall_gap_before"]),
                                   min_wall_gap_after=_number(stat["min_wall_gap_after"]),
                                   gap_band_before=stat["gap_band_before"], gap_band_after=stat["gap_band_after"],
                                   recorded_contact_points_in_control=int(offsets[transition+1]-offsets[transition]))
                        for wall in WALLS:
                            row[f"initial_{wall}_gap_world_px"] = initial_gaps.get(wall)
                            row[f"{wall}_gap_before"] = _number(stat[f"{wall}_gap_before"])
                            row[f"{wall}_gap_after"] = _number(stat[f"{wall}_gap_after"])
                        cube.append(row)
                        group = (letter, arc, approach, stat["motion_class"], stat["gap_band_before"])
                        value = control_group_counts[group]
                        value["controls"].add(key)
                        value["trajectories"].add(tid)
                        value["incidences"] += 1
                planned = scene.get("staging_segment")
                target = dict(split="train", letter=letter, trajectory_id=tid,
                              requested_spatial_class=entry["spatial_class"], planned_arc=planned,
                              any_real_contact=count > 0, planned_arc_actually_hit=planned in actual_bins,
                              planned_arc_is_proven_closed_hole=planned in C,
                              distinct_real_hit_bins=len(actual_bins), real_hit_bin_ids=sorted(actual_bins),
                              actual_contact_controls=count, recorded_controls=n, error=entry.get("error"))
                targets.append(target)
                letter_targets.append(target)

        counts, sets = _sets_summary(H, R, E, C, all_bins)
        for name, values in (("actual_hit_closed_hole_conflict", H & C),
                             ("canonical_reachable_closed_hole_conflict", R & C),
                             ("initial_ever_reachable_closed_hole_conflict", E & C)):
            if values:
                risks.append(dict(letter=letter, issue=name, bins=sorted(values)))
        for segment in segments:
            arc = segment["id"]
            expected = {str(tid): int(count) for tid, count in segment.get("trajectory_contact_controls", {}).items()}
            actual = {tid: count for (l, a, tid), count in recomputed_bin_controls.items() if l == letter and a == arc}
            if expected != actual or sum(actual.values()) != segment.get("contact_control_count"):
                risks.append(dict(letter=letter, issue="per_bin_contact_control_mapping_mismatch", arc=arc, recorded=expected, recomputed=actual))
            if set(segment["trajectory_ids"]) != recomputed_bin_trajs[(letter, arc)]:
                risks.append(dict(letter=letter, issue="per_bin_trajectory_mapping_mismatch", arc=arc))
        features = {}
        for feature in sorted({name for row in segments for name in row["features"]}):
            feature_bins = {row["id"] for row in segments if feature in row["features"]}
            features[feature] = dict(total=len(feature_bins), actual_hit=len(feature_bins & H),
                                     canonical_reachable=len(feature_bins & R), initial_ever_reachable=len(feature_bins & E),
                                     known_lower_bound=len(feature_bins & (R|E|H)), unconfirmed=len(feature_bins & set(sets["unconfirmed"])),
                                     proven_closed_hole=len(feature_bins & C))
        old_entries = [entry for entry in old_manifest["entries"] if entry["split"] == "train" and entry["shape"] == letter]
        actual_angle_counts = [sum(int(row.get("actual_contact_angle_bins", {}).get(str(index), 0)) for row in segments)
                               for index in range(8)]
        summary = dict(letter=letter, **counts, independent_trajectories=len(by_letter[letter]),
                       independent_contact_trajectories=contact_trajectories, recorded_controls=controls,
                       actual_contact_controls=contact_controls, recorded_contact_points=points_count,
                       mapped_contact_points=sum(row["contact_count"] for row in segments),
                       actual_contact_control_bin_incidences=sum(row.get("contact_control_count", 0) for row in segments),
                       planned_arc_hit_trajectories=sum(row["planned_arc_actually_hit"] for row in letter_targets),
                       planned_arc_hit_percent=percent(sum(row["planned_arc_actually_hit"] for row in letter_targets), len(letter_targets)),
                       zero_contact_trajectories=zero_contact, old_TRAIN_independent_trajectories=len(old_entries),
                       old_TRAIN_actual_hit_bins=len(old_H), old_TRAIN_actual_hit_percent=percent(len(old_H), len(all_bins)),
                       shared_old_new_hit_bins=len(old_H & H), new_only_hit_bins=len(H-old_H), old_only_hit_bins=len(old_H-H),
                       actual_angle_point_counts=actual_angle_counts,
                       all_recorded_angle_point_counts=[all_angles[index] for index in range(8)],
                       requested_approach_actual_contact_control_counts=dict(actual_approaches),
                       requested_side_actual_contact_control_counts=dict(direction_sides))
        summaries.append(summary)
        details[letter] = dict(summary=summary, sets=sets, old_TRAIN_actual_hit_bin_ids=sorted(old_H),
                               new_only_actual_hit_bin_ids=sorted(H-old_H), old_only_actual_hit_bin_ids=sorted(old_H-H),
                               features=features, initial_scenes=letter_gaps, planned_target_comparisons=letter_targets)
    if set(stats) != expected_stats_keys:
        raise ValueError("Statistics and raw recorded transition universes differ")
    group_rows = [dict(letter=key[0], actual_arc_bin=key[1], requested_approach=key[2], motion_class=key[3],
                       actual_gap_band_before=key[4], distinct_contact_controls=len(value["controls"]),
                       independent_trajectories=len(value["trajectories"]), trajectory_ids=sorted(value["trajectories"]))
                  for key, value in sorted(control_group_counts.items())]
    result = dict(kind="native-rgb-stage2a-offline-contact-coverage.v1", stage="2A", split="train",
                  definitions=DEFINITIONS, sources=sources, planned_slots=manifest.get("planned_slots"),
                  recorded_slots=len(manifest["entries"]), recorded_controls=sum(row["recorded_controls"] for row in summaries),
                  collection_failure=manifest.get("failure"), unexecuted_request_ids=manifest.get("unexecuted_request_ids", []),
                  per_letter=details, risks=risks, reachability_evidence_consistent=not risks,
                  actual_contact_controls=sum(row["actual_contact_controls"] for row in summaries),
                  contact_control_bin_incidences=len(cube), independent_contact_trajectories=sum(row["independent_contact_trajectories"] for row in summaries),
                  zero_contact_trajectories=[tid for row in summaries for tid in row["zero_contact_trajectories"]],
                  comparison_qualification="Historical TRAIN 22/23 independent trajectories per letter versus this first new 9-trajectory batch; budgets and scenes are not matched.",
                  features_qualification="Feature flags overlap; feature counts cannot be added.",
                  physics_controls_executed_by_report=0, environment_instances=0, renderer_calls=0, model_calls=0,
                  controller_or_data_modified=False, historical_counts_added_to_actual_counts=False)
    return result, summaries, targets, gaps, cube, group_rows, coverage, prior


def _markdown(report, summaries):
    lines = ["# Stage2A formal TRAIN contact coverage", "",
             "H=actual hit; R=frozen historical TRAIN canonical normal reachability; E=union over recorded legal initial poses; K=R∪E∪H.",
             "K is a global across observed poses reachable lower bound, not all physically reachable segments. U=all−K−closed holes remains unconfirmed.",
             "Historical TRAIN has 22/23 independent trajectories per letter; new first batch has 9 planned. These are different collection budgets and scenes.", "",
             "| Letter | H/all | H∩R/R | H∩E/E | H/K | U | Closed | Contact traj. | Contact controls | Planned arc hit | Old TRAIN H/all |", 
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['letter']} | {row['actual_hit_bins']}/{row['all_bins']} | {row['canonical_hit_bins']}/{row['canonical_known_reachable_bins']} | "
                     f"{row['initial_ever_hit_bins']}/{row['initial_pose_ever_reachable_bins']} | {row['actual_hit_bins']}/{row['reachable_lower_bound_bins']} | "
                     f"{row['unconfirmed_bins']} | {row['proven_closed_hole_bins']} | {row['independent_contact_trajectories']}/{row['independent_trajectories']} | "
                     f"{row['actual_contact_controls']}/{row['recorded_controls']} | {row['planned_arc_hit_trajectories']}/{row['independent_trajectories']} | "
                     f"{row['old_TRAIN_actual_hit_bins']}/{row['all_bins']} |")
    lines.extend(["", f"Evidence consistency risks: {len(report['risks'])}. Zero-contact trajectories: {report['zero_contact_trajectories']}.", "",
                  "`stage2a-reachability-summary.json` contains every unhit/known/unknown/closed bin ID and overlapping feature counts.",
                  "`target-contact-comparison.csv` compares the planned initial contact arc to actual collisions, not pusher click arrival.",
                  "`initial-scene-gaps.csv` lists all four true initial wall gaps. `contact-controls.csv` joins actual collision bins to continuous area-centroid dx/dy, rotation, motion class, requested approach and four before/after wall gaps.",
                  "`contact-cube-summary.csv` gives distinct controls and independent trajectory support per arc × requested approach × motion class × actual minimum-gap band. Actual contact-normal angle counts are separately reported; requested direction is not actual collision direction.",
                  "No wall-arbiter points were retained. Bounding-gap proximity does not establish an actual wall-collision count.",
                  "Repeated touches do not enlarge H. A control can hit multiple bins, so contact-control/bin incidences exceed distinct contact controls. Historical priors are not added to actual counts.", "",
                  f"Postprocessing source SHA256: `{report['sources']['report_script']['sha256']}`.",
                  "This report performed zero physics controls, environment construction, renderer/model calls or data/controller edits."])
    return "\n".join(lines) + "\n"


def build_report(stage, output, statistics, historical_train, historical_manifest, *, svg=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve prior reports; choose a new derived-output directory")
    report, rows, targets, gaps, cube, groups, coverage, prior = derive(stage, statistics, historical_train, historical_manifest)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/"stage2a-reachability-summary.json", report)
    _write_csv(output/"stage2a-reachability-summary.csv", rows)
    _write_csv(output/"target-contact-comparison.csv", targets)
    _write_csv(output/"initial-scene-gaps.csv", gaps)
    _write_csv(output/"contact-controls.csv", cube)
    _write_csv(output/"contact-cube-summary.csv", groups)
    with (output/"stage2a-reachability-summary.md").open("x", encoding="utf-8") as stream:
        stream.write(_markdown(report, rows))
    if svg:
        from tdyssey.native_rgb_data import _write_coverage
        _write_coverage(output/"svg-canonical", coverage["segments"], prior["canonical_reachability"])
    _write_json(output/"report-files.json", dict(source_sha256=sha256(__file__), files=[
        dict(file=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))
        for path in sorted(output.rglob("*")) if path.is_file()]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--statistics", help="Defaults to STAGE/report/h1/real-transition-statistics.csv")
    parser.add_argument("--historical-train", help="Defaults to collection's frozen historical TRAIN coverage snapshot")
    parser.add_argument("--historical-manifest", default=str(ROOT/"artifacts/native-rgb-20261028/data/manifest.json"))
    parser.add_argument("--svg", action="store_true", help="Write only pure geometric SVG contact heatmaps, no renderer")
    args = parser.parse_args(argv)
    stage = Path(args.stage)
    report = build_report(stage, args.output,
                          args.statistics or stage/"report/h1/real-transition-statistics.csv",
                          args.historical_train or stage/"source-snapshot/historical-TRAIN-contact-coverage.json",
                          args.historical_manifest, svg=args.svg)
    print(json.dumps({key: report[key] for key in ("recorded_slots", "recorded_controls", "actual_contact_controls",
                                                  "contact_control_bin_incidences", "reachability_evidence_consistent")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
