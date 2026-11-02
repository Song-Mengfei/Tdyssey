"""Cold R2A2 four-arm report from saved files; no renderer/model/environment.

The new phase reuses the old 16 scene families and 32 scene/action variants.
Contact-substep groups below are descriptive R2A2 groups, not revised R2A gates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.native_rgb_r2a_report import (
    _index_rows, _read, _safe, _sha, distribution, measure_slot, summarize,
)

ARMS = {
    "P0-frozen": 0.0,
    "P1-damping1-candidate": 1.0,
    "D-half-life-0.1s": 0.0009765625,
    "D-half-life-0.3s": 0.09921256574801246,
}
POPULATIONS = {"all_controls": 8, "h1_windows": 5, "h4_windows": 2}
GROUPS = ("all", "any_agent_object", "ao_at_least_80pct_future_substeps",
          "every_future_control_ao_ge8", "full_no_agent_object_no_object_wall", "nonzero_ao_impulse")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        return None
    return [float(v) for v in value]


def _kind(event):
    return str(event.get("kind", event.get("category", "unknown"))).replace("-", "_")


def _count(record, kind):
    value = record.get(kind+"_contact_substeps")
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10:
        return value
    return None


def _record_map(metadata):
    result = {}
    for i, row in enumerate(metadata.get("control_records", [])):
        control = int(row.get("control_index", row.get("index", i)))
        if control in result:
            raise ValueError("Duplicate control record index")
        result[control] = dict(row.get("info", {}), **row)
    return result


def _telemetry_controls(logs, metadata):
    """One record per control; impulses only post_solve, lever only pre_solve.

    The actual callback world point and post pose share the position established
    before pre_solve in the pinned engine. Signed lever is r x inward unit normal
    about the physical mass CoG, not the polygon's area centroid.
    """
    cog = _vector(metadata.get("initial_identity", {}).get("physics", {}).get("object", {}).get("center_of_gravity"), 2)
    result = {}
    for i, control in enumerate(logs):
        key = int(control.get("control_index", control.get("index", i)))
        if key in result:
            raise ValueError("Duplicate telemetry control index")
        impulse = defaultdict(list)
        unknown_impulses = 0
        signed_levers, reconstruction_errors = [], []
        contact_steps = {kind: set() for kind in ("agent_object", "object_wall", "agent_wall")}
        substeps = control.get("substeps", [])
        executed = 0
        for si, sub in enumerate(substeps):
            if sub.get("executed") is not True:
                continue
            executed += 1
            pose = _vector(sub.get("post_state"), 15)
            for event in sub.get("contacts", []):
                kind = _kind(event)
                if kind in contact_steps:
                    contact_steps[kind].add(si)
                if event.get("phase") == "post_solve":
                    vector = _vector(event.get("total_impulse"), 2)
                    if vector is None:
                        unknown_impulses += 1
                    else:
                        impulse[kind].append(math.hypot(*vector))
                if kind != "agent_object" or event.get("phase") != "pre_solve":
                    continue
                normal = _vector(event.get("normal"), 2)
                order = event.get("shapes_order", [])
                if normal is None or math.hypot(*normal) == 0 or pose is None or cog is None or not order:
                    continue
                length = math.hypot(*normal)
                sign = 1 if order[0] == "object" else -1
                normal = [sign*v/length for v in normal]
                c, s = math.cos(pose[4]), math.sin(pose[4])
                com = [pose[2]+c*cog[0]-s*cog[1], pose[3]+s*cog[0]+c*cog[1]]
                for point in event.get("points", []):
                    world, local = _vector(point.get("object_world"), 2), _vector(point.get("object_local"), 2)
                    if world is None:
                        continue
                    rx, ry = world[0]-com[0], world[1]-com[1]
                    signed_levers.append(rx*(-normal[1])-ry*(-normal[0]))
                    if local is not None:
                        reconstruction_errors.append(math.hypot(pose[2]+c*local[0]-s*local[1]-world[0],
                                                                pose[3]+s*local[0]+c*local[1]-world[1]))
        complete_log = control.get("completed") is True and executed == 10
        result[key] = dict(
            telemetry_executed_substeps=executed,
            telemetry_complete=complete_log,
            contact_substeps_from_events={k: len(v) if complete_log else None for k, v in contact_steps.items()},
            ao_post_solve_impulse_norm_sum=sum(impulse.get("agent_object", [])) if unknown_impulses == 0 else None,
            post_solve_impulse_norm_by_kind={k: distribution(v, ()) for k, v in impulse.items()},
            unknown_post_solve_impulse_events=unknown_impulses,
            signed_normal_levers_world_px=signed_levers,
            point_pose_reconstruction_error_max=None if not reconstruction_errors else max(reconstruction_errors))
    return result


def _annotate(row, records, telemetry):
    indices = range(row["anchor"], row["target_state_index"])
    complete = all(records.get(i, {}).get("completed") is True and
                   records.get(i, {}).get("substep_count", records.get(i, {}).get("substeps_recorded")) == 10 for i in indices)
    counts = {kind: [_count(records.get(i, {}), kind) for i in indices]
              for kind in ("agent_object", "object_wall", "agent_wall")}
    known = {kind: complete and all(v is not None for v in values) for kind, values in counts.items()}
    row["future_substeps_planned"] = 10*row["horizon"]
    for kind, values in counts.items():
        row[kind+"_contact_substeps"] = sum(values) if known[kind] else None
    ao, ow = row["agent_object_contact_substeps"], row["object_wall_contact_substeps"]
    row["ao_at_least_80pct_future_substeps"] = None if ao is None else ao >= 8*row["horizon"]
    row["every_future_control_ao_ge8"] = None if not known["agent_object"] else all(v >= 8 for v in counts["agent_object"])
    row["full_no_agent_object_no_object_wall"] = None if ao is None or ow is None else ao == 0 and ow == 0
    logged = [telemetry.get(i) for i in indices]
    if all(t is not None and t["telemetry_complete"] and t["ao_post_solve_impulse_norm_sum"] is not None for t in logged):
        row["ao_post_solve_impulse_norm_sum"] = sum(t["ao_post_solve_impulse_norm_sum"] for t in logged)
        row["nonzero_ao_impulse"] = row["ao_post_solve_impulse_norm_sum"] > 1e-12
    else:
        row["ao_post_solve_impulse_norm_sum"] = None
        row["nonzero_ao_impulse"] = None
    levers = [v for t in logged if t is not None for v in t["signed_normal_levers_world_px"]]
    row["signed_lever_measured_point_samples"] = len(levers)
    row["signed_lever_signs"] = sorted({"negative" if v < -1e-9 else "positive" if v > 1e-9 else "zero" for v in levers})
    row["signed_lever_min_world_px"] = min(levers) if levers else None
    row["signed_lever_max_world_px"] = max(levers) if levers else None
    row["telemetry_contact_count_matches_records"] = None if not all(t is not None and t["telemetry_complete"] for t in logged) else all(
        counts[kind][j] is not None and counts[kind][j] == logged[j]["contact_substeps_from_events"][kind]
        for kind in counts for j in range(row["horizon"]))
    return row


def _joint_targets(row):
    x, a = row["translation_image96_px"], row["absolute_rotation_deg"]
    if x is None or a is None:
        return {}
    return dict(translation_2_to_4=2 <= x < 4, translation_4_to_8=4 <= x < 8,
                rotation_5_to_15=5 <= a < 15,
                translation_2_to_4_angle_lt1=2 <= x < 4 and a < 1,
                translation_4_to_8_angle_lt1=4 <= x < 8 and a < 1,
                rotation_5_to_15_translation_lt1=5 <= a < 15 and x < 1,
                translation_ge2_rotation_ge5=x >= 2 and a >= 5)


def _summary(rows, planned=None):
    rows = list(rows)
    result = summarize(rows, planned)
    result["joint_targets"] = dict(Counter(k for r in rows for k, v in _joint_targets(r).items() if v))
    result["valid_joint_targets"] = dict(Counter(k for r in rows if r["valid_window"] for k, v in _joint_targets(r).items() if v))
    result["object_wall_contact_count"] = sum(r["object_wall_contact"] is True for r in rows)
    result["agent_wall_contact_count"] = sum(r["agent_wall_contact"] is True for r in rows)
    result["nonzero_ao_impulse_counts"] = dict(Counter(str(r["nonzero_ao_impulse"]) for r in rows))
    result["signed_lever_sign_control_counts"] = dict(Counter(sign for r in rows for sign in r["signed_lever_signs"]))
    return result


def _selected(rows, group):
    if group == "all":
        return rows
    key = "agent_object_contact" if group == "any_agent_object" else group
    return [r for r in rows if r[key] is True]


def _load_cohort(root, frozen, phase, allowed_arms):
    import numpy as np
    root = Path(root).resolve()
    index_path = root/"index.json"
    index = _read(index_path)
    if index.get("collection_closed") is not True:
        raise RuntimeError("Refusing a hot collector; collection_closed=true required")
    requests = [r for r in frozen["requests"] if r["phase"] == phase]
    if len(requests) != 64 or sum(r["max_controls"] for r in requests) != 512:
        raise ValueError("Each cohort must retain its frozen64 slots/512 planned controls")
    if Counter(r["protocol"] for r in requests) != Counter({arm: 32 for arm in allowed_arms}):
        raise ValueError("Unexpected cohort arm counts")
    families = {f["family_id"]: f for f in frozen["scene_families"]}
    by_id = {}
    for item in _index_rows(index):
        key = item.get("request_id", item.get("request", {}).get("request_id", item.get("slot_id")))
        if key in by_id:
            raise ValueError("Duplicate index request")
        by_id[key] = item
    if set(by_id)-{r["request_id"] for r in requests}:
        raise ValueError("Unexpected index request outside the frozen cohort")
    slots, populations, errors = [], defaultdict(list), []
    for req in requests:
        item = by_id.get(req["request_id"], {})
        slot_id = item.get("slot_id", req["request_id"])
        paths = item.get("paths", {})
        npz = _safe(root, paths.get("npz", item.get("trajectory_file", f"trajectories/{slot_id}/trajectory.npz")))
        meta = _safe(root, paths.get("metadata", item.get("metadata_file", f"trajectories/{slot_id}/metadata.json")))
        slot = dict(request_id=req["request_id"], source_request_id=req.get("source_request_id"),
                    family_id=req["family_id"], letter=req["letter"], action_variant=req["action_variant"],
                    protocol=req["protocol"], cohort=phase, status=item.get("status", "not_executed"),
                    planned_controls=8, planned_h1=5, planned_h4=2, complete_controls=0,
                    charged_controls=item.get("charged_controls"), actual_controls=item.get("actual_controls"))
        if npz.is_file() and meta.is_file():
            try:
                metadata = _read(meta)
                actual_npz_sha, actual_meta_sha = _sha(npz), _sha(meta)
                for expected, actual in ((item.get("trajectory_sha256"), actual_npz_sha),
                                         (item.get("metadata_sha256"), actual_meta_sha),
                                         (metadata.get("trajectory_sha256"), actual_npz_sha)):
                    if expected is not None and expected != actual:
                        raise ValueError("Index/metadata hash mismatch")
                if metadata.get("request") is not None and _canonical(metadata["request"]) != _canonical(req):
                    raise ValueError("Saved request differs from frozen request")
                with np.load(npz, allow_pickle=False) as source:
                    arrays = {k: source[k] for k in source.files}
                measured = measure_slot(req, families[req["family_id"]], arrays, metadata)
                measured_populations = measured.pop("populations")
                slot.update(measured, npz_sha256=actual_npz_sha, metadata_sha256=actual_meta_sha,
                            initial_identity=metadata.get("initial_identity"), protocol_sha256=metadata.get("protocol_sha256"))
                slot["actual_damping"] = metadata.get("initial_identity", {}).get("physics", {}).get("space", {}).get("damping")
                slot["damping_matches_requested_arm"] = slot["actual_damping"] == ARMS[req["protocol"]]
                logs = []
                if metadata.get("telemetry_file"):
                    path = _safe(root, metadata["telemetry_file"])
                    if metadata.get("telemetry_sha256") is not None and _sha(path) != metadata["telemetry_sha256"]:
                        raise ValueError("Telemetry SHA mismatch")
                    with gzip.open(path, "rt", encoding="utf-8") as stream:
                        logs = json.load(stream)
                records = _record_map(metadata)
                telemetry = _telemetry_controls(logs, metadata)
                slot["telemetry_control_records"] = len(logs)
                slot["telemetry_point_pose_reconstruction_error_max"] = max(
                    (t["point_pose_reconstruction_error_max"] for t in telemetry.values() if t["point_pose_reconstruction_error_max"] is not None), default=None)
                for name, values in measured_populations.items():
                    populations[name].extend(dict(_annotate(row, records, telemetry), cohort=phase) for row in values)
            except Exception as error:
                slot["data_error"] = type(error).__name__+": "+str(error)
                errors.append(dict(request_id=req["request_id"], error=slot["data_error"]))
        else:
            slot["data_missing"] = True
        slots.append(slot)
    return dict(root=str(root), index_sha256=_sha(index_path), slots=slots, populations=dict(populations),
                data_errors=errors, requests=requests)


def _validate_reuse(new, old):
    old_requests = [r for r in old["requests"] if r["phase"] == "R2A"]
    p0 = {r["request_id"]: r for r in old_requests if r["protocol"] == "P0-frozen"}
    new_requests = [r for r in new["requests"] if r["phase"] == "R2A2"]
    expected = {(r["family_id"], r["action_variant"]) for r in p0.values()}
    if len(expected) != 32 or len({k[0] for k in expected}) != 16:
        raise ValueError("Old baseline must contain16 independent families/32 variants")
    for arm in tuple(ARMS)[2:]:
        chosen = [r for r in new_requests if r["protocol"] == arm]
        if len(chosen) != 32 or {(r["family_id"], r["action_variant"]) for r in chosen} != expected:
            raise ValueError("New arm must reuse exactly the32 frozen old variants")
        for req in chosen:
            source = p0.get(req.get("source_request_id"))
            if source is None:
                raise ValueError("source_request_id must map to an old P0 request")
            for key in ("family_id", "action_variant", "letter", "initial_space", "initial_parameters_sha256", "actions_absolute"):
                if req.get(key) != source.get(key):
                    raise ValueError("New request changes old scene/action field: "+key)
    old_families = {f["family_id"]: f for f in old["scene_families"]}
    for family in new["scene_families"]:
        baseline_family = old_families.get(family["family_id"])
        if baseline_family is None or baseline_family.get("phase") != "R2A" or family.get("phase") != "R2A2":
            raise ValueError("Scene family phase migration must be exactly R2A -> R2A2")
        if {k: v for k, v in family.items() if k != "phase"} != {k: v for k, v in baseline_family.items() if k != "phase"}:
            raise ValueError("New scene family differs from old frozen family")
    if {f["family_id"] for f in new["scene_families"]} != {key[0] for key in expected}:
        raise ValueError("New phase must not add an independent scene family")
    return dict(reused_independent_families=16, reused_scene_action_variants=32,
                new_independent_families=0, source_request_mapping_matches=True)


def _contact_histogram(rows, planned):
    histogram = {str(i): 0 for i in range(11)}
    unknown = 0
    for row in rows:
        value = row["agent_object_contact_substeps"]
        if value is None:
            unknown += 1
        else:
            histogram[str(value)] += 1
    return dict(counts_0_to_10=histogram, unknown_observed=unknown,
                observed_complete=len(rows), missing_planned=max(0, planned-len(rows)), planned=planned,
                scope="0..10 distinct100Hz AO-contact substeps in one completed0.1s control; >=8 is descriptive, not a success gate")


def _coverage(rows, requests):
    result = {}
    for letter in sorted({r["letter"] for r in requests}):
        selected = [r for r in rows if r["letter"] == letter]
        arcs, signs = defaultdict(set), defaultdict(set)
        for row in selected:
            for arc in row["actual_contact_arc_ids"]:
                arcs[arc].add(row["family_id"])
            for sign in row["signed_lever_signs"]:
                signs[sign].add(row["family_id"])
        result[letter] = dict(
            planned_independent_families=len({r["family_id"] for r in requests if r["letter"] == letter}),
            observed_independent_families=len({r["family_id"] for r in selected}),
            actual_contact_independent_families=len({r["family_id"] for r in selected if r["agent_object_contact"] is True}),
            actual_arc_ids=sorted(arcs), independent_families_per_arc={k: len(v) for k, v in arcs.items()},
            independent_families_per_signed_lever_sign={k: len(v) for k, v in signs.items()},
            contact_controls_without_complete_arc_mapping=sum(r["agent_object_contact"] is True and not r["actual_arc_mapping_complete"] for r in selected),
            actual_boundary_process_counts=dict(Counter(r["boundary_process"] for r in selected)),
            actual_space_counts=dict(Counter(r["actual_spatial_class_before"] for r in selected)),
            actual_corner_two_gap_counts=dict(Counter(r["actual_corner_gap_pair_before"] for r in selected)),
            target_support={key: dict(controls=sum(_joint_targets(r).get(key, False) for r in selected),
                                     independent_families=len({r["family_id"] for r in selected if _joint_targets(r).get(key, False)}))
                            for key in _joint_targets(dict(translation_image96_px=0, absolute_rotation_deg=0))})
    return result


def _pairs(requests, slots, populations):
    by_slot = {s["request_id"]: s for s in slots}
    rows = {(r["request_id"], r["anchor"]): r for r in populations["all_controls"]}
    grouped = defaultdict(dict)
    for req in requests:
        grouped[(req["family_id"], req["action_variant"])][req["protocol"]] = req
    pairs = []
    for (family, variant), arms in sorted(grouped.items()):
        if set(arms) != set(ARMS):
            raise ValueError("Every frozen scene/action variant needs exactly four arms")
        actual = [by_slot[arms[a]["request_id"]] for a in ARMS]
        measured = all("actual_initial_state" in s for s in actual)
        initial_equal = None if not measured else all(
            s["actual_initial_state"] == actual[0]["actual_initial_state"] and s["initial_rgb_sha256"] == actual[0]["initial_rgb_sha256"] for s in actual)
        physicals = []
        for s in actual:
            physics = deepcopy((s.get("initial_identity") or {}).get("physics"))
            if physics is not None:
                physics.get("space", {}).pop("damping", None)
            physicals.append(physics)
        only_damping = None if any(p is None for p in physicals) else all(p == physicals[0] for p in physicals)
        actions_match = all(s.get("source_actions_equal_request") is True for s in actual)
        dampings_match = all(s.get("damping_matches_requested_arm") is True for s in actual)
        controls = []
        for i in range(8):
            observed = {a: rows.get((arms[a]["request_id"], i)) for a in ARMS}
            controls.append(dict(control_index=i, all_four_observed=all(r is not None for r in observed.values()),
                all_four_valid=all(r is not None and r["valid_window"] for r in observed.values()),
                by_arm={a: None if r is None else {k: r[k] for k in (
                    "translation_image96_px", "absolute_rotation_deg", "agent_object_contact_substeps", "object_wall_contact",
                    "full_no_agent_object_no_object_wall", "nonzero_ao_impulse")} for a, r in observed.items()}))
        comparisons = {}
        for left, right in combinations(ARMS, 2):
            both = [(c["by_arm"][left], c["by_arm"][right]) for c in controls if c["by_arm"][left] is not None and c["by_arm"][right] is not None]
            comparisons[left+" -> "+right] = dict(observed_control_pairs=len(both), missing_planned=8-len(both),
                translation_delta_right_minus_left=distribution((b["translation_image96_px"]-a["translation_image96_px"] if a["translation_image96_px"] is not None and b["translation_image96_px"] is not None else None for a, b in both), ()))
        pairs.append(dict(family_id=family, action_variant=variant, planned_four_arm_controls=8,
            actual_initial_state_and_rgb_equal=initial_equal, actual_physics_equal_except_damping=only_damping,
            actual_actions_match_frozen=actions_match, actual_damping_matches_all_arms=dampings_match,
            observed_four_arm_controls=sum(c["all_four_observed"] for c in controls),
            qualified_four_arm_controls=sum(c["all_four_valid"] for c in controls) if initial_equal and only_damping and actions_match and dampings_match else 0,
            controls=controls, pairwise_comparisons=comparisons))
    return pairs


def build_report(data_root, requests_path, baseline_root, baseline_requests_path):
    """Read closed new/old cohorts only, retaining their independent costs."""
    new_frozen, old_frozen = _read(requests_path), _read(baseline_requests_path)
    reuse = _validate_reuse(new_frozen, old_frozen)
    new = _load_cohort(data_root, new_frozen, "R2A2", tuple(ARMS)[2:])
    old = _load_cohort(baseline_root, old_frozen, "R2A", tuple(ARMS)[:2])
    requests, slots = old["requests"]+new["requests"], old["slots"]+new["slots"]
    populations = {name: old["populations"].get(name, [])+new["populations"].get(name, []) for name in POPULATIONS}
    by_arm = {}
    for arm in ARMS:
        reqs = [r for r in requests if r["protocol"] == arm]
        values = {name: [r for r in rows if r["protocol"] == arm] for name, rows in populations.items()}
        by_arm[arm] = dict(
            requested_damping=ARMS[arm], damping_half_life_seconds={tuple(ARMS)[2]: .1, tuple(ARMS)[3]: .3}.get(arm),
            populations={name: _summary(rows, 32*POPULATIONS[name]) for name, rows in values.items()},
            conditional_groups={name: {group: _summary(_selected(rows, group), 32*POPULATIONS[name] if group == "all" else None) for group in GROUPS} for name, rows in values.items()},
            ao_contact_substeps_histogram=_contact_histogram(values["all_controls"], 256),
            by_letter={letter: {name: _summary([r for r in rows if r["letter"] == letter], sum(q["letter"] == letter for q in reqs)*POPULATIONS[name]) for name, rows in values.items()} for letter in sorted({q["letter"] for q in reqs})},
            actual_contact_boundary_coverage=_coverage(values["all_controls"], reqs),
            actual_protocol_sha256s=sorted({s["protocol_sha256"] for s in slots if s["protocol"] == arm and s.get("protocol_sha256") is not None}))
    cohort_summaries = {phase: dict(
        planned_slots=64, status_counts=dict(Counter(s["status"] for s in cohort["slots"])),
        charged_controls_declared=sum(s["charged_controls"] for s in cohort["slots"] if s["charged_controls"] is not None),
        actual_controls_declared=sum(s["actual_controls"] for s in cohort["slots"] if s["actual_controls"] is not None),
        partial_attempt_records=sum(len(s.get("partial_attempt_records", [])) for s in cohort["slots"]),
        populations={name: _summary(cohort["populations"].get(name, []), 64*per) for name, per in POPULATIONS.items()})
        for phase, cohort in (("R2A", old), ("R2A2", new))}
    pairs = _pairs(requests, slots, populations)
    return dict(kind="native-rgb-R2A2-four-arm-cold-report.v1", requests_sha256=_sha(requests_path),
        baseline_requests_sha256=_sha(baseline_requests_path), index_sha256=new["index_sha256"], baseline_index_sha256=old["index_sha256"],
        collection_closed=True, reuse= dict(reuse), cohort_summaries=cohort_summaries, slots=slots, populations=populations, by_arm=by_arm,
        four_arm_pairs=pairs, planned_four_arm_pairs=32, planned_four_arm_control_comparisons=256,
        observed_four_arm_control_comparisons=sum(p["observed_four_arm_controls"] for p in pairs),
        qualified_four_arm_control_comparisons=sum(p["qualified_four_arm_controls"] for p in pairs),
        data_errors=old["data_errors"]+new["data_errors"],
        telemetry_contact_count_mismatches=sum(r["telemetry_contact_count_matches_records"] is False for r in populations["all_controls"]),
        scope=dict(new_authorized_planned_controls=512, reused_old_controls_not_reexecuted=512,
            new_independent_families=0, sampling_or_protocol_acceptance=False,
            contact_groups="R2A2 descriptive only; oldR2A recomputed into newreport without modifying its original acceptance/report",
            group_denominators="Conditional group observedN is not a frozen planned denominator; overall/arm/letter failures and missing stay in planned denominators",
            h4="Fourfuture-controls endpoint0.4s; >=32/40 is an auxiliary aggregate80pct descriptor, not parent preregistration, not everycontrol>=8 and not32 consecutive substeps; every_future_control_ao_ge8 is separate and still does not prove continuous contact; not H1 single-control2-4px target",
            targets="2<=translation<4 and4<=translation<8 image96px;5<=absangle<15deg; descriptive not an automatic acceptance gate",
            signed_lever="pre_solve points only;r cross inward unit objectnormal about actual physical massCoG; sign is geometric, not an active-motion causal conclusion",
            coasting="NoAO/noOW motion retained as measured dynamics; not invalid/bad data and not sustainedpushing success",
            validity="Unknowns never measuredzero; complete0.1s and partial attempts distinct; apparent amplitudes retain valid flags",
            old_outputs="Read-only baseline files; no oldreport/threshold/source rewritten"),
        new_environment_or_physics_steps=0, new_model_or_GPU_calls=0, renderer_calls=0)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"r2a2-report.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    for name, rows in report["populations"].items():
        fields = sorted({k for row in rows for k in row})
        with (output/(name+".csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    lines = ["# R2A2四臂离线幅度、接触与配对统计", "",
        "新增仅64槽/512计划控制；旧P0/P1只读复算、不重跑。16独立初始family、32场景/动作变体沿用，新增独立family为0。失败、未执行、缺测保留计划分母。", "",
        "|队列|计划raw/H1/H4|实测完整raw/H1/H4|有效raw/H1/H4|计账控制|", "|---|---|---|---|---:|"]
    for phase, summary in report["cohort_summaries"].items():
        pops = summary["populations"]
        lines.append(f"|{phase}|512/320/128|"+"/".join(str(pops[n]["observed_complete"]) for n in POPULATIONS)+"|"+"/".join(str(pops[n]["valid_windows"]) for n in POPULATIONS)+f"|{summary['charged_controls_declared']}|")
    lines += ["", "|臂/总体|计划|完整|有效|平移median/P95/max（96px图像px）|≥1/2/3/4/8px|转角median/P95/max（°）|≥5/15°|", "|---|---:|---:|---:|---|---|---|---|"]
    def fmt(value):
        return "缺测" if value is None else f"{value:.4f}"
    for arm, data in report["by_arm"].items():
        for name, summary in data["populations"].items():
            t, a = summary["translation_image96_px"], summary["rotation_deg"]
            lines.append(f"|{arm}/{name}|{summary['planned']}|{summary['observed_complete']}|{summary['valid_windows']}|"+
                "/".join(fmt(t[k]) for k in ("median", "p95", "max"))+"|"+"/".join(str(t["counts_gte"][str(k)]) for k in (1, 2, 3, 4, 8))+"|"+
                "/".join(fmt(a[k]) for k in ("median", "p95", "max"))+"|"+"/".join(str(a["counts_gte"][str(k)]) for k in (5, 15))+"|")
    lines += ["", "## AO接触子步分布（本阶段新增描述，不改旧验收）", "",
        "每个完整0.1s含10个100Hz子步，AO接触子步0–10全直方图如下。H1≥8/10为父明确的本阶段描述组，不等于主动推动成功或正式采样验收。H4单列0.4s端点：累计≥32/40只是本报告辅助聚合80%描述，并非父预注册条件，不等于四控制各≥8/10或32子步连续接触；every_future_control_ao_ge8另列各控制均≥8/10组，仍不证明连续接触。", "",
        "|臂|0|1|2|3|4|5|6|7|8|9|10|未知完整|计划缺项|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, data in report["by_arm"].items():
        h = data["ao_contact_substeps_histogram"]
        lines.append("|"+arm+"|"+"|".join(str(h["counts_0_to_10"][str(i)]) for i in range(11))+f"|{h['unknown_observed']}|{h['missing_planned']}|")
    lines += ["", "## H1条件组与联合幅度", "", "条件组分母为实测条件N，不能替代各臂计划160个H1。noAO/noOW滑行是合法实测动力学，单列、不贬为坏数据，也不称持续推移。", "",
        "|臂/条件|实测H1|接触family|平移median/P95/max|2–4px且角<1°|4–8px且角<1°|5–15°且平移<1px|≥2px且≥5°|", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for arm, data in report["by_arm"].items():
        for group, summary in data["conditional_groups"]["h1_windows"].items():
            t, targets = summary["translation_image96_px"], summary["joint_targets"]
            lines.append(f"|{arm}/{group}|{summary['observed_complete']}|{summary['actual_contact_independent_families']}|"+
                "/".join(fmt(t[k]) for k in ("median", "p95", "max"))+"|"+"|".join(str(targets.get(k, 0)) for k in (
                    "translation_2_to_4_angle_lt1", "translation_4_to_8_angle_lt1", "rotation_5_to_15_translation_lt1", "translation_ge2_rotation_ge5"))+"|")
    lines += ["", f"四臂计划32配对/256控制位置；实测四臂同时可观测{report['observed_four_arm_control_comparisons']}、同时有效且初态/RGB/非D参数/actions匹配{report['qualified_four_arm_control_comparisons']}。", "",
        "逐字母的真实arc、独立family、有符号法向力臂、四墙signed gaps/双墙gap和边界过程见JSON/CSV。不同协议/动作重复不增加独立family；实际接触/出墙/沿墙/旋转缺层不能靠平均幅度或初始位置标签补造。T面积质心与物理CoG保持分列，未作图像修补。", "",
        f"数据读取错误{len(report['data_errors'])}；telemetry与control_records接触子步不一致{report['telemetry_contact_count_mismatches']}。未知不当0，部分控制不伪作完整0.1s。非零冲量仅计post_solve，pre_solve不重复累加。", "",
        "仅保存文件离线统计：0新物理/renderer/model/GPU。D候选尚未自动获选，R2B/正式数据/训练须父另审；本报告不扩大执行授权。"]
    with (output/"REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")


def self_check():
    """Synthetic dictionaries only: no dataset, scene, renderer or real draw."""
    rec = {i: dict(completed=True, substep_count=10, agent_object_contact_substeps=i,
                   object_wall_contact_substeps=0, agent_wall_contact_substeps=0) for i in range(11)}
    row = dict(anchor=8, target_state_index=9, horizon=1)
    value = _annotate(dict(row), rec, {})
    assert value["ao_at_least_80pct_future_substeps"] is True and value["nonzero_ao_impulse"] is None
    assert value["every_future_control_ao_ge8"] is True
    h4records = {i: dict(completed=True, substep_count=10, agent_object_contact_substeps=v,
                        object_wall_contact_substeps=0, agent_wall_contact_substeps=0) for i, v in enumerate((10, 10, 10, 2))}
    h4 = _annotate(dict(anchor=0, target_state_index=4, horizon=4), h4records, {})
    assert h4["ao_at_least_80pct_future_substeps"] is True and h4["every_future_control_ao_ge8"] is False
    assert _annotate(dict(row, anchor=0, target_state_index=1), rec, {})["full_no_agent_object_no_object_wall"] is True
    missing = _annotate(dict(row), {8: dict(completed=True, substep_count=10)}, {})
    assert missing["ao_at_least_80pct_future_substeps"] is None and missing["full_no_agent_object_no_object_wall"] is None
    histogram = _contact_histogram([dict(agent_object_contact_substeps=i) for i in range(11)]+[dict(agent_object_contact_substeps=None)], 13)
    assert all(v == 1 for v in histogram["counts_0_to_10"].values()) and histogram["unknown_observed"] == 1 and histogram["missing_planned"] == 1
    event = dict(kind="agent_object", phase="post_solve", total_impulse=[3, 4])
    pre = dict(kind="agent_object", phase="pre_solve", total_impulse=[30, 40])
    logs = [dict(completed=True, substeps=[dict(executed=True, contacts=[event, pre]) for _ in range(10)])]
    measured = _telemetry_controls(logs, {})[0]
    assert measured["ao_post_solve_impulse_norm_sum"] == 50 and measured["contact_substeps_from_events"]["agent_object"] == 10
    assert _joint_targets(dict(translation_image96_px=4, absolute_rotation_deg=.5))["translation_4_to_8_angle_lt1"] is True
    return dict(checks=7, passed=True, synthetic_only=True, real_data_reads=0, new_physics_steps=0, model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-requests", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-after-root-cold", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
        return
    if not args.execute_after_root_cold or any(getattr(args, key) is None for key in (
            "data_root", "requests", "baseline_root", "baseline_requests", "output")):
        parser.error("All paths and explicit --execute-after-root-cold required; never inspect a live collector")
    report = build_report(args.data_root, args.requests, args.baseline_root, args.baseline_requests)
    write_report(report, args.output)
    print(json.dumps(dict(new_planned_controls=512, new_independent_families=0,
        counts_by_arm={arm: {name: v["observed_complete"] for name, v in data["populations"].items()} for arm, data in report["by_arm"].items()},
        four_arm_observed=report["observed_four_arm_control_comparisons"], data_errors=len(report["data_errors"]), new_physics_steps=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
