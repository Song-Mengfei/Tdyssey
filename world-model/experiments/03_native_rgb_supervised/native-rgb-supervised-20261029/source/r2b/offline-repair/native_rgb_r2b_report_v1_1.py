"""Selected-physics R2B report from closed saved data only.

Dynamic commands are checked against recorded policy decisions, not an old
frozen absolute action list. No simulator, renderer, model or iterator is run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent/"tdyssey/native_rgb_selected_physics.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_rgb_r2a_report import _geometry, _index_rows, _read, _safe, _sha, distribution, measure_slot
from scripts.native_rgb_r2a2_report import _annotate, _record_map, _summary, _telemetry_controls
from tdyssey.native_rgb_selected_physics import DISPLAY_NAME, SELECTED_ARM, SELECTED_PROTOCOL_SHA256, selected_metadata, verify_selected_identity

LETTERS = ("T", "L", "O", "C", "X", "B")
PLANNED = dict(all_controls=1296, h1_windows=1134, h4_windows=972, push_h4_windows=378, push_full_history_h4_windows=216)
PER_SLOT = dict(all_controls=24, h1_windows=21, h4_windows=18, push_h4_windows=7, push_full_history_h4_windows=4)
EPS = 1e-12


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def planned_phase(control):
    return "approach" if control < 4 else "push" if control < 14 else "withdraw" if control < 20 else "recontact"


def state_sha(state):
    import numpy as np
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes(order="C")).hexdigest()


def dynamic_decisions(arrays, metadata, external=None):
    """Validate executed and partial commands using the actual current-state chain."""
    import numpy as np
    states, actions = np.asarray(arrays["states"]), np.asarray(arrays["actions"])
    records = metadata.get("control_records", [])
    policy = metadata.get("policy_identity", {})
    policy_sha = policy.get("policy_sha256")
    rules_sha = policy.get("rules_sha256")
    if not isinstance(policy_sha, str) or len(policy_sha) != 64 or not isinstance(rules_sha, str) or len(rules_sha) != 64:
        raise ValueError("Missing frozen policy SHA identity")
    if external is not None and len(external) != len(records):
        raise ValueError("Independent decision log count differs")
    attempted = arrays.get("attempted_actions")
    for i, record in enumerate(records):
        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise ValueError("Missing actual dynamic decision")
        if external is not None:
            logged = external[i].get("decision", external[i])
            if canonical(logged) != canonical(decision):
                raise ValueError("Independent decision log differs from control record")
        if decision.get("control_index") != i or record.get("control_index", i) != i:
            raise ValueError("Decision/control order mismatch")
        if i >= len(states) or decision.get("state_before_sha256") != state_sha(states[i]):
            raise ValueError("Decision current-state prefix SHA mismatch")
        for alias in ("current_state15", "current_state", "state_before"):
            if alias in decision and not np.array_equal(np.asarray(decision[alias]), states[i]):
                raise ValueError("Decision actual current state differs from recorded prefix")
        target = np.asarray(decision.get("proposed_action"), dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all() or (target < 0).any() or (target > 511).any():
            raise ValueError("Invalid actual proposed target")
        if "action_absolute" in decision and not np.array_equal(target, np.asarray(decision["action_absolute"])):
            raise ValueError("Planner/root target aliases differ")
        if i < len(actions) and not np.array_equal(target, actions[i]):
            raise ValueError("Executed action differs from recorded dynamic target")
        if attempted is not None and (i >= len(attempted) or not np.array_equal(target, attempted[i])):
            raise ValueError("Attempted action differs from recorded dynamic target")
        if (decision.get("planned_phase") != planned_phase(i) or not isinstance(decision.get("actual_phase"), str)
                or not decision["actual_phase"] or decision.get("policy_sha256") != policy_sha
                or decision.get("policy_rules_sha256") != rules_sha):
            raise ValueError("Decision phase/policy binding differs from frozen rules")
    if len(records) < len(actions) or (attempted is not None and len(attempted) != len(records)):
        raise ValueError("Actual command/decision denominator differs")
    return dict(verified=True, completed_action_decisions=len(actions), all_attempted_decisions=len(records),
        independent_decision_log_verified=external is not None, policy_sha256=policy_sha,
        method="actual command and current-state prefix equality; no frozen absolute action list comparison")


def measure_dynamic(request, family, arrays, metadata, logs, decisions=None):
    """Pure saved-state measurement; the old geometry helper gets an adapter only."""
    import numpy as np
    validation = dynamic_decisions(arrays, metadata, decisions)
    if len(arrays["actions"]) > 24:
        raise ValueError("R2B control cap exceeded")
    adapter = dict(request, max_controls=24, action_variant=request.get("role", "feedback-policy"),
        protocol=SELECTED_ARM, actions_absolute=np.asarray(arrays["actions"]).tolist())
    adapter.setdefault("initial_space", request.get("location", "unknown"))
    measured = measure_slot(adapter, family, arrays, metadata)
    # This legacy helper field would imply a frozen-action test; remove it.
    measured.pop("source_actions_equal_request")
    measured["dynamic_decision_validation"] = validation
    records, telemetry = _record_map(metadata), _telemetry_controls(logs, metadata)
    for rows in measured["populations"].values():
        for row in rows:
            _annotate(row, records, telemetry)
            future = [records.get(i, {}).get("decision", {}) for i in range(row["anchor"], row["target_state_index"])]
            row.update(planned_role=request.get("role"), planned_process_goal=request.get("process_goal"),
                actual_phases=sorted({d.get("actual_phase", "unknown") for d in future}),
                planned_phases=sorted({d.get("planned_phase", "unknown") for d in future}),
                rotation_direction="positive" if row["signed_rotation_rad"] > 1e-10 else "negative" if row["signed_rotation_rad"] < -1e-10 else "near_zero")
    measured["populations"]["push_h4_windows"] = [dict(row, population="push_h4_windows")
        for row in measured["populations"]["h4_windows"] if 4 <= row["anchor"] <= 10]
    measured["populations"]["push_full_history_h4_windows"] = [dict(row, population="push_full_history_h4_windows")
        for row in measured["populations"]["h4_windows"] if 7 <= row["anchor"] <= 10]
    measured["release"] = release_observation(arrays["states"], logs, request["letter"])
    measured["actual_contact_summary"] = contact_observation(logs, telemetry, metadata)
    measured["telemetry_by_control"] = telemetry
    return measured


def contact_observation(logs, telemetry, metadata):
    """Actual100Hz contact runs and post-only impulses; no planned contact labels."""
    ao_hist = Counter({str(i): 0 for i in range(11)})
    impulse, history = defaultdict(list), []
    unknown_impulses = partial = 0
    signed_levers = []
    for log in logs:
        observed = []
        for sub in log.get("substeps", []):
            known = sub.get("executed") is True and not sub.get("engine_exception")
            contacts = sub.get("contacts", [])
            observed.append(any(e.get("kind") == "agent_object" for e in contacts) if known else None)
            for event in contacts:
                if event.get("phase") != "post_solve":
                    continue
                value = event.get("total_impulse")
                if value is None or len(value) != 2 or not all(isinstance(v, (int,float)) and math.isfinite(v) for v in value):
                    unknown_impulses += 1
                else:
                    impulse[event.get("kind", "unknown")].append(math.hypot(*value))
        history.extend(observed)
        if log.get("completed") is True and len(observed) == 10 and None not in observed:
            ao_hist[str(sum(observed))] += 1
        else:
            partial += 1
        signed_levers.extend(telemetry[int(log["control_index"])]["signed_normal_levers_world_px"])
    durations, run = [], 0
    for event in history+[False]:
        if event is True:
            run += 1
        elif run:
            durations.append(run*.01)
            run = 0
    return dict(AO_hit_substeps_histogram_complete_controls=dict(ao_hist), partial_or_unknown_controls=partial,
        AO_contact_episode_durations_seconds=durations, AO_contact_episodes=len(durations),
        AO_loss_transitions=sum(a is True and b is False for a,b in zip(history,history[1:])),
        AO_observed_recontacts=max(0,len(durations)-1),
        post_only_impulses={k: distribution(v, (1e-12,)) for k,v in impulse.items()}, unknown_post_impulse_events=unknown_impulses,
        signed_lever_actual_point_sign_counts=dict(Counter("positive" if v>1e-9 else "negative" if v < -1e-9 else "zero" for v in signed_levers)),
        signed_lever_world_px=distribution(signed_levers,()),
        actual_arc_ids=sorted({str(arc) for record in metadata.get("control_records",[]) for arc in record.get("actual_contact_arc_ids",[])}),
        note="Only actualevents; nativepost_solve impulse once, correlatedpointcounts and observedcontactepisodes are not independentfamilies or intentionalrecovery")


def release_observation(states, logs, letter):
    """Observe release outcomes; never fill crossing times from a damping formula."""
    import numpy as np
    states = np.asarray(states, dtype=float)
    result = dict(reference_state_index=14, planned_observation_controls=[14, 19],
        threshold_reference="speed at command release state14, never silently rebased to detached speed",
        data_available=len(states) > 14, free_segments=[], actual_detachment_time_seconds=None,
        actual_detachment_interval_seconds=None,
        detachment_time_convention="first no-AO/OW observed substep endpoint; continuous detachment only localized to its recorded10ms interval")
    if len(states) <= 14:
        result["missing_reason"] = "release reference state14 not observed"
        return result
    area, _, _ = _geometry(letter, states)
    end = min(20, len(states)-1)
    initial_v, initial_w = float(np.linalg.norm(states[14, 7:9])), abs(float(states[14, 9]))
    result.update(reference_linear_speed_world_s=initial_v, reference_angular_speed_rad_s=initial_w,
        observed_end_state_index=end, observed_duration_seconds=(end-14)*.1,
        complete_planned_release=end == 20,
        extra_area_translation_image96_px=float(np.linalg.norm(area[end]-area[14])*96/512),
        extra_signed_rotation_deg=float((states[end, 4]-states[14, 4])*180/math.pi),
        extra_absolute_rotation_deg=float(abs(states[end, 4]-states[14, 4])*180/math.pi))
    current, elapsed, previous_ao = None, 0., False
    log_by_control = {int(log["control_index"]): log for log in logs}
    if 13 in log_by_control and log_by_control[13]["substeps"]:
        previous_ao = any(e.get("kind") == "agent_object" for e in log_by_control[13]["substeps"][-1].get("contacts", []))
    result["AO_contact_immediately_before_release"] = previous_ao

    def start_segment(entry, start_time):
        segment = dict(start_time_since_release=start_time, end_time_since_release=start_time,
            after_observed_AO_contact=previous_ao, ended_by=None, thresholds={})
        for label, ref, value in (("linear", initial_v, float(np.linalg.norm(entry[7:9]))), ("angular", initial_w, abs(float(entry[9])))):
            for fraction in (.1, .05):
                key = f"{label}_{int(fraction*100)}pct"
                segment["thresholds"][key] = dict(reference_at_release=ref, threshold=ref*fraction,
                    status="not_assessable_zero_or_nearzero_release_speed" if ref <= EPS else
                        "already_below_at_segment_start_crossing_unobserved" if value <= ref*fraction else "not_reached_yet",
                    observed_time_since_release=None, crossing_interval_seconds=None,
                    first_below_observed_at_segment_start=start_time if ref > EPS and value <= ref*fraction else None)
        return segment

    def finish(reason, boundary, interval=None):
        nonlocal current
        if current is None:
            return
        current.update(ended_by=reason, end_time_since_release=boundary, censor_interval_seconds=interval)
        for threshold in current["thresholds"].values():
            if threshold["status"] == "not_reached_yet":
                threshold["status"] = "right_censored_"+reason
        result["free_segments"].append(current)
        current = None

    stop = False
    for control in range(14, 20):
        log = log_by_control.get(control)
        if log is None:
            finish("missing_or_ended_observation", elapsed)
            break
        for sub in log.get("substeps", []):
            dt = float(sub.get("dt", .01))
            entry, post = sub.get("integrator_pre_state"), sub.get("post_state")
            pen = sub.get("penetrations")
            valid = (sub.get("executed") is True and not sub.get("engine_exception")
                and entry is not None and post is not None and pen is not None
                and np.asarray(entry).shape == (15,) and np.asarray(post).shape == (15,)
                and np.isfinite(np.asarray(entry, dtype=float)).all() and np.isfinite(np.asarray(post, dtype=float)).all()
                and np.isfinite(np.asarray(pen, dtype=float)).all() and max(pen) <= 2.)
            kinds = {e.get("kind", e.get("category", "unknown")).replace("-", "_") for e in sub.get("contacts", [])}
            ao, wall = "agent_object" in kinds, "object_wall" in kinds
            if not valid:
                finish("invalid_or_unknown_substep", elapsed, [elapsed, elapsed+dt])
                stop = True
                break
            if ao or wall:
                finish("object_wall_collision" if wall else "agent_object_recontact", elapsed, [elapsed, elapsed+dt])
                if wall:
                    result.setdefault("wall_contact_intervals", []).append([elapsed, elapsed+dt])
                previous_ao = ao
                elapsed += dt
                continue
            entry, post = np.asarray(entry), np.asarray(post)
            if current is None:
                current = start_segment(entry, elapsed)
                if previous_ao and result["actual_detachment_time_seconds"] is None:
                    result["actual_detachment_time_seconds"] = elapsed+dt
                    result["actual_detachment_interval_seconds"] = [elapsed, elapsed+dt]
            for label, value in (("linear", float(np.linalg.norm(post[7:9]))), ("angular", abs(float(post[9])))):
                for fraction in (.1, .05):
                    threshold = current["thresholds"][f"{label}_{int(fraction*100)}pct"]
                    if threshold["status"] == "not_reached_yet" and value <= threshold["threshold"]:
                        threshold.update(status="observed_first_below_in_contact_free_segment",
                            observed_time_since_release=elapsed+dt, crossing_interval_seconds=[elapsed, elapsed+dt])
            elapsed += dt
            current["end_time_since_release"] = elapsed
            previous_ao = False
        if stop or log.get("completed") is not True:
            finish("partial_or_failed_control", elapsed)
            break
    finish("observation_end", elapsed)
    result["actual_100Hz_observed_span_seconds"] = elapsed
    result["all_segments_retained"] = True
    result["threshold_caveat"] = "Segment times use fixed release-state14 reference; pre-detachment contact, wall/recontact, missing endpoints and censoring are retained. No formula or later successful segment substitutes for first segment."
    return result


def build_report(root, requests_path=None):
    import numpy as np
    root = Path(root).resolve()
    frozen_path = Path(requests_path).resolve() if requests_path else root/"frozen-requests.json"
    index, frozen = _read(root/"index.json"), _read(frozen_path)
    config_path = root/"frozen-config.json"
    config = _read(config_path)
    policy_identity = config["policy_identity"]
    bundle = {key:policy_identity[key] for key in ("kind", "rules_sha256", "source_sha256")}
    if hashlib.sha256(canonical(bundle).encode()).hexdigest() != policy_identity["policy_sha256"]:
        raise ValueError("Frozen policy bundle SHA differs")
    rules_path = _safe(root, policy_identity["rules_file"])
    rules = _read(rules_path)
    if (_sha(rules_path) != policy_identity["rules_file_sha256"]
            or hashlib.sha256(canonical(rules).encode()).hexdigest() != policy_identity["rules_sha256"]
            or frozen.get("rules") != rules):
        raise ValueError("Frozen policy rule-file/plan identity differs")
    ledger = _read(root/"resource-ledger.json")
    if index.get("collection_closed") is not True or ledger.get("closed") is not True:
        raise RuntimeError("Root explicit cold notification and closed files required; no polling")
    requests = frozen.get("requests", frozen.get("slots", []))
    if len(requests) != 54 or any(r.get("phase", "R2B") != "R2B" or r["max_controls"] != 24 for r in requests):
        raise ValueError("Only the frozen54-slot/1296-control R2B is supported")
    if Counter(r["letter"] for r in requests) != Counter({letter: 9 for letter in LETTERS}):
        raise ValueError("R2B six letters by nine spaces differ")
    if any(r.get("protocol") != SELECTED_ARM or r.get("protocol_sha256") != SELECTED_PROTOCOL_SHA256 for r in requests):
        raise ValueError("Frozen R2B requests differ from selected protocol")
    source_path = root/"source-identity.json"
    source = _read(source_path)
    for entry in source["files"]:
        for path in (Path(entry["source"]), _safe(root, entry["snapshot"])):
            if _sha(path) != entry["sha256"]:
                raise ValueError("Frozen source/snapshot hash mismatch")
    if not any(Path(row["source"]).resolve() == frozen_path and row["sha256"] == _sha(frozen_path) for row in source["files"]):
        raise ValueError("Frozen request file not bound by source identity")
    for path, expected in ((config_path, _sha(config_path)), (rules_path, policy_identity["rules_file_sha256"]),
            (ROOT/policy_identity["source_file"], policy_identity["source_sha256"])):
        if not any(Path(row["source"]).resolve() == path.resolve() and row["sha256"] == expected for row in source["files"]):
            raise ValueError("Frozen config/rules/policy source missing from source identity")
    families = {f["family_id"]: f for f in frozen.get("scene_families", [])}
    registry = _index_rows(index)
    ids = [r.get("request_id", r.get("slot_id")) for r in registry]
    if len(ids) != len(set(ids)) or set(ids)-{r["request_id"] for r in requests}:
        raise ValueError("Unexpected or duplicate registry request")
    by_id = dict(zip(ids, registry))
    slots, populations, errors = [], defaultdict(list), []
    for request in requests:
        item = by_id.get(request["request_id"], {})
        identifier = item.get("slot_id", request["request_id"])
        slot = dict(request_id=request["request_id"], slot_id=identifier, family_id=request["family_id"],
            letter=request["letter"], initial_space=request.get("initial_space"), planned_role=request.get("role"),
            planned_process_goal=request.get("process_goal"), planned_static_missing=request.get("static_missing", request.get("static_missing_reasons")),
            planned_dispatchable=request.get("dispatchable"), status=item.get("status", "not_executed"),
            charged_controls=item.get("charged_controls", 0), planned=dict(PER_SLOT), complete_controls=0)
        npz = _safe(root, item.get("trajectory_file", f"trajectories/{identifier}/trajectory.npz"))
        meta_path = _safe(root, item.get("metadata_file", f"trajectories/{identifier}/metadata.json"))
        if not npz.exists() or not meta_path.exists():
            slot["missing_data"] = True
            slots.append(slot)
            continue
        try:
            metadata = _read(meta_path)
            if (canonical(metadata["request"]) != canonical(request) or _sha(meta_path) != item["metadata_sha256"]
                    or _sha(npz) != item["trajectory_sha256"] or _sha(npz) != metadata["trajectory_sha256"]):
                raise ValueError("Frozen request/NPZ/metadata binding differs")
            if metadata.get("policy_identity") != policy_identity:
                raise ValueError("Runtime policy identity differs from precollection frozen config")
            verify_selected_identity(metadata["initial_identity"])
            logs_path, decisions_path = _safe(root, metadata["telemetry_file"]), _safe(root, metadata["decisionlogs_file"])
            if _sha(logs_path) != metadata["telemetry_sha256"] or _sha(decisions_path) != metadata["decisionlogs_sha256"]:
                raise ValueError("Telemetry/independent decision log SHA differs")
            with gzip.open(logs_path, "rt", encoding="utf-8") as stream:
                logs = json.load(stream)
            with gzip.open(decisions_path, "rt", encoding="utf-8") as stream:
                decisions = json.load(stream)
            with np.load(npz, allow_pickle=False) as source_arrays:
                arrays = {name: source_arrays[name] for name in source_arrays.files}
            if len(logs) != slot["charged_controls"]:
                raise ValueError("Every charged control must retain telemetry including failure")
            measured = measure_dynamic(request, families.get(request["family_id"], request), arrays, metadata, logs, decisions)
            values = measured.pop("populations")
            measured.pop("telemetry_by_control")
            slot.update(measured, npz_sha256=_sha(npz), metadata_sha256=_sha(meta_path), telemetry_sha256=_sha(logs_path),
                decisionlogs_sha256=_sha(decisions_path), actual_initial_identity=metadata["initial_identity"])
            for name, rows in values.items():
                populations[name].extend(rows)
        except Exception as exc:
            slot["data_error"] = f"{type(exc).__name__}: {exc}"
            errors.append(dict(request_id=request["request_id"], error=slot["data_error"]))
        slots.append(slot)
    summaries = {name: _summary(populations[name], planned) for name, planned in PLANNED.items()}
    by_letter = {letter: {name: _summary([row for row in populations[name] if row["letter"] == letter], 9*PER_SLOT[name])
        for name in PLANNED} for letter in LETTERS}
    dimensions = ("family_id", "planned_role", "initial_space", "boundary_process", "actual_spatial_class_before", "actual_corner_gap_pair_before", "min_gap_band_before", "rotation_direction")
    grouped = {}
    for name, rows in populations.items():
        grouped[name] = {dimension: {str(value): _summary([row for row in rows if row.get(dimension) == value])
            for value in sorted({row.get(dimension) for row in rows}, key=str)} for dimension in dimensions}
        grouped[name]["actual_phase"] = {phase: _summary([row for row in rows if phase in row["actual_phases"]])
            for phase in sorted({phase for row in rows for phase in row["actual_phases"]})}
        grouped[name]["actual_arc"] = {f"{letter}:{arc}": _summary([row for row in rows if row["letter"] == letter and arc in row["actual_contact_arc_ids"]])
            for letter, arc in sorted({(row["letter"], arc) for row in rows for arc in row["actual_contact_arc_ids"]})}
        grouped[name]["signed_lever"] = {sign: _summary([row for row in rows if sign in row["signed_lever_signs"]]) for sign in ("positive", "negative", "zero")}
        grouped[name]["actual_contact_groups"] = {group: _summary([row for row in rows if row.get(key) is True])
            for group, key in (("any_AO", "agent_object_contact"), ("AO_ge80pct_future", "ao_at_least_80pct_future_substeps"),
                ("nonzero_AO_impulse", "nonzero_ao_impulse"), ("no_AO_no_OW_coast", "full_no_agent_object_no_object_wall"))}
    focus = {}
    for letter in LETTERS:
        rows = [r for r in populations["h1_windows"] if r["letter"] == letter and r["valid_window"]]
        focus[letter] = dict(valid_h1=len(rows), translation_ge2=sum(r["translation_image96_px"] >= 2 for r in rows),
            positive_rotation_ge5=sum(r["signed_rotation_rad"] >= math.radians(5) for r in rows),
            negative_rotation_le_minus5=sum(r["signed_rotation_rad"] <= -math.radians(5) for r in rows),
            actual_lever_signs=sorted({s for r in rows for s in r["signed_lever_signs"]}),
            actual_arc_ids=sorted({arc for r in rows for arc in r["actual_contact_arc_ids"]}))
    contact_summaries = [s["actual_contact_summary"] for s in slots if "actual_contact_summary" in s]
    contact_total = dict(AO_hit_substeps_histogram_complete_controls={str(i):sum(s["AO_hit_substeps_histogram_complete_controls"][str(i)] for s in contact_summaries) for i in range(11)},
        AO_episode_durations_seconds=distribution((v for s in contact_summaries for v in s["AO_contact_episode_durations_seconds"]),()),
        AO_contact_episodes=sum(s["AO_contact_episodes"] for s in contact_summaries),
        AO_loss_transitions=sum(s["AO_loss_transitions"] for s in contact_summaries),
        AO_observed_recontacts=sum(s["AO_observed_recontacts"] for s in contact_summaries),
        partial_or_unknown_controls=sum(s["partial_or_unknown_controls"] for s in contact_summaries))
    releases = [dict(request_id=s["request_id"], letter=s["letter"], family_id=s["family_id"], **s["release"]) for s in slots if "release" in s]
    release_summary = dict(planned_slots=54, observed_release_reference=sum(r["data_available"] for r in releases),
        complete_planned_release=sum(r.get("complete_planned_release",False) for r in releases),
        extra_translation_image96_px=distribution((r.get("extra_area_translation_image96_px") for r in releases), (2,4)),
        extra_absolute_rotation_deg=distribution((r.get("extra_absolute_rotation_deg") for r in releases), (5,15)),
        first_segment_thresholds={})
    for key in ("linear_10pct","linear_5pct","angular_10pct","angular_5pct"):
        statuses, times = Counter(), []
        for release in releases:
            if not release["data_available"]:
                statuses["missing_release_reference"] += 1
            elif not release["free_segments"]:
                statuses["no_contact_free_segment_observed"] += 1
            else:
                threshold = release["free_segments"][0]["thresholds"][key]
                statuses[threshold["status"]] += 1
                times.append(threshold["observed_time_since_release"])
        release_summary["first_segment_thresholds"][key] = dict(status_counts=dict(statuses),
            observed_times_since_release_seconds=distribution(times,()),
            missing_planned_release_slots=54-len(releases),
            note="First segment only for aggregate; all later segments retained individually, no success-selection or rebasing")
    return dict(kind="R2B-selected-physics-feedback-sampling-report.v1", display_name=DISPLAY_NAME,
        selected_physics=selected_metadata(), no_damping_comparison=True, collection_closed=True,
        index_sha256=_sha(root/"index.json"), ledger_sha256=_sha(root/"resource-ledger.json"),
        frozen_requests_sha256=_sha(frozen_path), source_identity_sha256=_sha(source_path),
        policy_identity=policy_identity, frozen_config_sha256=_sha(config_path),
        planned_slots=54, planned=dict(PLANNED), slots=slots, data_errors=errors,
        summaries=summaries, by_letter=by_letter, groups=grouped, focus_support=focus,
        actual_contact_summary=contact_total, release_summary=release_summary, release_observations=releases,
        missing_release_slots=[s["request_id"] for s in slots if "release" not in s or not s["release"]["data_available"]],
        populations=dict(populations), ledger=ledger, torch_imported="torch" in sys.modules,
        limits=["Planned intents/roles are never actual contact or movement support.", "Push-future H4 anchors4..10 (7/slot,378planned) contain futurecontrols entirely4..13; history may contain approach. Full-push-history H4 anchors7..10 (4/slot,216planned) contain all3past+4future transitions in push; actualphases/contact groups separate.",
            "Release threshold reference is state14 speed; all contact-free segments and censoring retained, no formula-filled times or rebasing.",
            "Actual signed-lever groups use pre_solve contact points about physical CoG; pre/post impulse not doubled.",
            "This report is not a damping validation,18-letter training set,model evaluation,or unseen-shape result."])


def markdown(result):
    lines = ["# R2B 接触跟随采样报告", "", result["display_name"], "",
        f"计划54槽/1296控制，数据错误{len(result['data_errors'])}；所有未执行、失败及缺测槽保留。", "",
        "|Population|Planned|Observed|Valid|≥2px|≥5°|", "|---|---:|---:|---:|---:|---:|"]
    for name, s in result["summaries"].items():
        lines.append(f"|{name}|{s['planned']}|{s['observed_complete']}|{s['valid_windows']}|{s['translation_image96_px']['counts_gte']['2']}|{s['rotation_deg']['counts_gte']['5']}|")
    lines.extend(["", "push_h4_windows为未来4控全在push的anchor4..10（7/槽，计划378），历史可含approach；push_full_history_h4_windows为3过去+4未来控全在push的anchor7..10（4/槽，计划216）。实际接触/冲量/阶段另列，计划推送不是已接触推送。", "",
        "|Letter|Valid H1|≥2px|+≥5°|−≤−5°|Actual lever signs|", "|---|---:|---:|---:|---:|---|"])
    for letter, s in result["focus_support"].items():
        lines.append(f"|{letter}|{s['valid_h1']}|{s['translation_ge2']}|{s['positive_rotation_ge5']}|{s['negative_rotation_le_minus5']}|{','.join(s['actual_lever_signs'])}|")
    lines.extend(["", "松开观测以state14实测速度为10%/5%参考，另列真实脱离时刻。所有自由段保留；碰撞、重接触、观察尾和缺测标删失，不套用半衰期公式填时间。", "",
        "逐family/真实弧段/有符号力臂/实际wallgap/边界过程/旋转方向及联合目标详见JSON与CSV。未运行环境、渲染、模型、GPU或训练。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = build_report(args.data, args.requests)
    out = args.out or args.data/"report"
    out.mkdir(parents=True, exist_ok=True)
    (out/"r2b-report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    (out/"REPORT.md").write_text(markdown(result), encoding="utf-8")
    for name, rows in result["populations"].items():
        fields = sorted({k for row in rows for k in row})
        with (out/(name+".csv")).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    print(json.dumps(dict(slots=len(result["slots"]), data_errors=len(result["data_errors"]),
        planned=result["planned"], observed={k:v["observed_complete"] for k,v in result["summaries"].items()})))
    return 1 if result["data_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
