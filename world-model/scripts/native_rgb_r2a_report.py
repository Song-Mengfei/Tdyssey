"""Cold, offline R2A amplitude/pair/contact report. Never runs an environment.

All complete0.1s controls and genuine four-frame-history H1/H4 candidates are
separate populations. A partial physics attempt is never a complete H1 target.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCALE = 96 / 512
WALLS = ("left", "right", "top", "bottom")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _safe(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("R2A data path escapes its root")
    return path


def _geometry(letter, states):
    import numpy as np
    from tdyssey.envs.official import official_shape
    shape = official_shape(letter)
    local = np.asarray(shape.centroid, dtype=float)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    area = states[:, 2:4] + np.column_stack((c*local[0]-s*local[1], s*local[0]+c*local[1]))
    vertices = np.asarray([v for part in shape.components for v in part.vertices], dtype=float)
    x = c[:, None]*vertices[:, 0]-s[:, None]*vertices[:, 1]+states[:, 2, None]
    y = s[:, None]*vertices[:, 0]+c[:, None]*vertices[:, 1]+states[:, 3, None]
    gaps = np.column_stack((x.min(1)-7, 504-x.max(1), y.min(1)-7, 504-y.max(1)))
    return area, gaps, local


def _contact(record, pair):
    aliases = {
        "agent-object": ("agent_object_contact_substeps", "agent_object_contact", "contact_count", "contact"),
        "object-wall": ("object_wall_contact_substeps", "object_wall_contact"),
        "agent-wall": ("agent_wall_contact_substeps", "agent_wall_contact", "pusher_wall_contact_substeps")}
    for key in aliases[pair]:
        if key in record and record[key] is not None:
            return bool(record[key] > 0)
    contacts = record.get("contact_substeps_by_kind", {})
    if pair in contacts:
        return bool(contacts[pair] > 0)
    return None


def _any_known(values):
    return True if True in values else False if values and all(v is False for v in values) else None


def _arc_ids(record):
    values = record.get("actual_contact_arc_ids", record.get("actual_contact_arcs"))
    if values is None:
        return None
    result = set()
    for value in values:
        value = value.get("segment_id", value.get("id")) if isinstance(value, dict) else value
        if value is None or isinstance(value, bool) or not str(value):
            return None
        result.add(str(value))
    return sorted(result)


def _penetrations(record):
    value = record.get("max_penetrations", record.get("max_substep_penetrations"))
    if isinstance(value, dict):
        value = [value.get("object_wall"), value.get("object_pusher", value.get("pusher_object")),
                 value.get("pusher_wall", value.get("agent_wall"))]
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(v is None for v in value):
        return None
    return [_finite(v) for v in value]


def _cog(metadata, family):
    identity = metadata.get("initial_identity", {})
    for source in (identity.get("physics", {}), identity, metadata.get("effective_protocol", {}), metadata):
        for key in ("object_local_cog", "object_center_of_gravity", "physical_local_cog"):
            if key in source:
                return source[key], "runtime_recorded_local_CoG"
        if isinstance(source.get("object"), dict) and "center_of_gravity" in source["object"]:
            return source["object"]["center_of_gravity"], "runtime_recorded_local_CoG"
    initial = family["initialization"]
    dx = initial["object_mass_center"][0]-initial["object_origin"][0]
    dy = initial["object_mass_center"][1]-initial["object_origin"][1]
    angle = initial["angle_rad"]
    return [math.cos(angle)*dx+math.sin(angle)*dy, -math.sin(angle)*dx+math.cos(angle)*dy], "frozen_planned_CoG_not_runtime_measured"


def measure_slot(request, family, arrays, metadata):
    """Derive metrics from saved states only; useful with analytic CPU fixtures."""
    import numpy as np
    states = np.asarray(arrays["states"], dtype=float)
    actions = np.asarray(arrays["actions"])
    n = len(actions)
    if states.shape != (n+1, 15) or actions.shape != (n, 2) or n > request["max_controls"]:
        raise ValueError("Complete control/state/action timeline differs")
    rgb = arrays.get("rgb")
    rgb_ok = rgb is not None and rgb.shape == (n+1, 3, 96, 96) and rgb.dtype == np.uint8
    frame_sha = metadata.get("frame_sha256", metadata.get("frameSHA", metadata.get("frame_sha")))
    frames_hash_match = None
    if frame_sha is not None and rgb_ok:
        expected = frame_sha if isinstance(frame_sha, list) else [frame_sha.get(str(i)) for i in range(n+1)]
        frames_hash_match = len(expected) == n+1 and all(hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() == value for frame,value in zip(rgb, expected))
    controls = metadata.get("control_records", [])
    record_by_index = {int(row.get("control_index", row.get("index", i))): dict(row.get("info",{}), **row) for i,row in enumerate(controls)}
    area, gaps, local_area = _geometry(request["letter"], states)
    local_cog, cog_scope = _cog(metadata, family)
    c, s = np.cos(states[:,4]), np.sin(states[:,4])
    physical = states[:,2:4] + np.column_stack((c*local_cog[0]-s*local_cog[1], s*local_cog[0]+c*local_cog[1]))
    valid = []
    for t in range(n):
        rec = record_by_index.get(t, {})
        penetration = _penetrations(rec)
        quality = (rec.get("completed") is True and rec.get("valid_transition", rec.get("valid")) is True
            and rec.get("substep_count", rec.get("substeps_recorded")) == 10 and penetration is not None
            and all(v is not None and v <= 2 for v in penetration)
            and np.isfinite(states[t:t+2]).all() and np.isfinite(actions[t]).all()
            and (actions[t]>=0).all() and (actions[t]<=511).all()
            and min(gaps[t:t+2].min(), 0) >= -2)
        valid.append(bool(quality))

    def row(anchor, horizon, population):
        end = anchor+horizon
        delta = area[end]-area[anchor]
        translation = float(np.linalg.norm(delta))
        theta = float(states[end,4]-states[anchor,4])
        future = [record_by_index.get(i, {}) for i in range(anchor,end)]
        contact = _any_known([_contact(v,"agent-object") for v in future])
        wall_contact = _any_known([_contact(v,"object-wall") for v in future])
        arc_values = [_arc_ids(v) for v in future]
        actual_arcs = sorted({a for values in arc_values if values is not None for a in values})
        before, after = gaps[anchor], gaps[end]
        nearest = int(np.argmin(before))
        gap_delta = float(after[nearest]-before[nearest])
        near_static = translation <= .5 and abs(theta) <= .005
        if near_static and contact is True and wall_contact is True:
            process = "observed_wall_contact_stall"
        elif near_static and contact is True and min(before) <= 2:
            process = "near_wall_contact_stall_proxy"
        elif min(before.min(),after.min()) > 64:
            process = "interior"
        elif gap_delta < -.5:
            process = "approaching_boundary_geometry"
        elif gap_delta > .5:
            process = "leaving_boundary_geometry"
        elif translation >= 2:
            process = "along_near_boundary_geometry"
        else:
            process = "boundary_other"
        start_valid = anchor if population == "all_controls" else anchor-3
        window_valid = bool(all(valid[start_valid:end]) and rgb_ok and frames_hash_match is not False)
        result = dict(request_id=request["request_id"], family_id=request["family_id"], letter=request["letter"],
            protocol=request["protocol"], action_variant=request["action_variant"], initial_space=request["initial_space"],
            population=population, anchor=anchor, horizon=horizon, target_state_index=end, actual_duration_seconds=.1*horizon,
            valid_window=window_valid, translation_world_px=_finite(translation), translation_image96_px=_finite(translation*SCALE),
            dx_image96_px=_finite(delta[0]*SCALE), dy_image96_px=_finite(delta[1]*SCALE),
            signed_rotation_rad=_finite(theta), absolute_rotation_deg=_finite(abs(theta)*180/math.pi), large_raw_rotation_step=bool(abs(theta)>math.pi),
            physical_CoG_translation_image96_px=_finite(np.linalg.norm(physical[end]-physical[anchor])*SCALE),
            physical_CoG_scope=cog_scope, agent_object_contact=contact, object_wall_contact=wall_contact,
            agent_wall_contact=_any_known([_contact(v,"agent-wall") for v in future]),
            actual_contact_arc_ids=actual_arcs, actual_arc_mapping_complete=all(v is not None and not (_contact(rec,"agent-object") is True and not v) for rec,v in zip(future,arc_values)),
            boundary_process=process, nearest_wall_before=WALLS[nearest], nearest_wall_gap_change_world_px=_finite(gap_delta),
            min_wall_gap_before_world_px=_finite(before.min()), min_wall_gap_after_world_px=_finite(after.min()),
            phase=str(record_by_index.get(anchor, {}).get("action_phase", request.get("action_schedule", ["unknown"]*n)[anchor] if anchor<len(request.get("action_schedule", [])) else "unknown")),
            near_static_original_definition=bool(near_static))
        for i,wall in enumerate(WALLS):
            result[wall+"_gap_before"] = _finite(before[i])
            result[wall+"_gap_after"] = _finite(after[i])
        result["two_nearest_walls_before"] = [{"wall":WALLS[i],"gap":_finite(before[i])} for i in np.argsort(before)[:2]]
        def gap_band(value):
            if not math.isfinite(float(value)):return "unknown"
            if value<0:return "negative"
            for upper,label in ((2,"0-2"),(8,"2-8"),(32,"8-32"),(64,"32-64")):
                if value<=upper:return label
            return ">64"
        if np.isfinite(before).all():
            horizontal=("left" if before[0]<=before[1] else "right") if min(before[:2])<=64 else None
            vertical=("top" if before[2]<=before[3] else "bottom") if min(before[2:])<=64 else None
            result["actual_spatial_class_before"]="-".join(v for v in (horizontal,vertical) if v) or "center"
            result["actual_corner_gap_pair_before"]=(horizontal+":"+gap_band(before[WALLS.index(horizontal)])+"|"+vertical+":"+gap_band(before[WALLS.index(vertical)])) if horizontal and vertical else "not_corner"
        else:
            result["actual_spatial_class_before"]="unknown"
            result["actual_corner_gap_pair_before"]="unknown"
            result["boundary_process"]="unknown_nonfinite_geometry"
        result["min_gap_band_before"]=gap_band(before.min())
        return result

    populations = dict(all_controls=[row(t,1,"all_controls") for t in range(n)],
        h1_windows=[row(a,1,"h1_windows") for a in range(3,n)],
        h4_windows=[row(a,4,"h4_windows") for a in range(3,n-3)])
    partial = [{k:v for k,v in rec.items() if k not in ("partial_rgb","rgb","partial_state")} for rec in controls if rec.get("completed") is False]
    return dict(populations=populations, complete_controls=n, partial_attempt_records=partial,
        valid_complete_controls=sum(valid), rgb_timeline_shape_valid=rgb_ok, frames_hash_match=frames_hash_match,
        local_area_centroid=local_area.tolist(), physical_local_CoG=local_cog, physical_CoG_scope=cog_scope,
        actual_initial_state=states[0].tolist(), initial_rgb_sha256=None if not rgb_ok else hashlib.sha256(np.ascontiguousarray(rgb[0]).tobytes()).hexdigest(),
        source_actions_equal_request=bool(n==0 or actions.dtype==np.float64 and np.array_equal(actions,np.asarray(request["actions_absolute"][:n],dtype=np.float64))))


def distribution(values, thresholds):
    import numpy as np
    all_values = list(values)
    finite = np.asarray([v for v in all_values if v is not None and math.isfinite(float(v))], dtype=float)
    return dict(total=len(all_values), measured=len(finite), missing=len(all_values)-len(finite),
        min=None if not len(finite) else float(finite.min()), median=None if not len(finite) else float(np.median(finite)),
        p95=None if not len(finite) else float(np.quantile(finite,.95)), max=None if not len(finite) else float(finite.max()),
        counts_gte={str(t):int((finite>=t).sum()) for t in thresholds})


def summarize(rows, planned):
    rows = list(rows)
    return dict(planned=planned, observed_complete=len(rows), missing_planned=None if planned is None else max(0,planned-len(rows)),
        valid_windows=sum(r["valid_window"] for r in rows), invalid_or_unverified_windows=sum(not r["valid_window"] for r in rows),
        translation_image96_px=distribution((r["translation_image96_px"] for r in rows),(1,2,3,4,8)),
        rotation_deg=distribution((r["absolute_rotation_deg"] for r in rows),(5,15)),
        contact_counts=dict(Counter(str(r["agent_object_contact"]) for r in rows)),
        object_wall_contact_counts=dict(Counter(str(r["object_wall_contact"]) for r in rows)),
        boundary_process_counts=dict(Counter(r["boundary_process"] for r in rows)),
        actual_independent_families=len({r["family_id"] for r in rows}),
        actual_contact_independent_families=len({r["family_id"] for r in rows if r["agent_object_contact"] is True}))


def _index_rows(index):
    rows = index.get("slots", index.get("entries", []))
    return [dict(value,slot_id=key) for key,value in rows.items()] if isinstance(rows,dict) else rows


def build_report(data_root, requests_path):
    """Read a closed root collection only. Failure/unexecuted requests stay rows."""
    import numpy as np
    root = Path(data_root).resolve()
    index_path, request_path = root/"index.json", Path(requests_path).resolve()
    index, frozen = _read(index_path), _read(request_path)
    if index.get("collection_closed") is not True:
        raise RuntimeError("Wait for explicit root cold notification and collection_closed=true")
    requests = [r for r in frozen["requests"] if r["phase"]=="R2A"]
    if len(requests)!=64 or sum(r["max_controls"] for r in requests)!=512:
        raise ValueError("Only the frozen64-slot512-control R2A is supported")
    families = {f["family_id"]:f for f in frozen["scene_families"]}
    rows = _index_rows(index)
    by_request = {}
    for item in rows:
        identifier = item.get("request_id", item.get("request",{}).get("request_id"))
        if identifier in by_request:
            raise ValueError("Duplicate index request")
        by_request[identifier] = item
    if set(by_request)-{r["request_id"] for r in requests}:
        raise ValueError("Index contains a request outside frozen R2A")
    slots, populations, loads = [], defaultdict(list), []
    for request in requests:
        item = by_request.get(request["request_id"], {})
        slot_id = item.get("slot_id",request["request_id"])
        paths = item.get("paths",{})
        npz_path = _safe(root,paths.get("npz",item.get("trajectory_file",item.get("npz_path",f"trajectories/{slot_id}/trajectory.npz"))))
        metadata_path = _safe(root,paths.get("metadata",item.get("metadata_file",item.get("metadata_path",f"trajectories/{slot_id}/metadata.json"))))
        slot = dict(request_id=request["request_id"],slot_id=slot_id,family_id=request["family_id"],letter=request["letter"],
            protocol=request["protocol"],action_variant=request["action_variant"],status=item.get("status","not_executed"),
            planned_controls=8,charged_controls=item.get("charged_controls"),actual_controls=item.get("actual_controls"),
            declared_completed_controls=item.get("completed_controls"),complete_controls=0,planned_h1=5,planned_h4=2)
        if npz_path.is_file() and metadata_path.is_file():
            try:
                metadata = _read(metadata_path)
                for expected,actual in ((item.get("trajectory_sha256"),_sha(npz_path)),(item.get("metadata_sha256"),_sha(metadata_path)),(metadata.get("trajectory_sha256"),_sha(npz_path))):
                    if expected is not None and expected!=actual:raise ValueError("Index/metadata artifact hash mismatch")
                with np.load(npz_path,allow_pickle=False) as source:
                    arrays = {key:source[key] for key in source.files}
                measured = measure_slot(request,families[request["family_id"]],arrays,metadata)
                slot_populations=measured.pop("populations")
                slot.update(measured,npz_sha256=_sha(npz_path),metadata_sha256=_sha(metadata_path),
                    initial_identity=metadata.get("initial_identity"),protocol_sha256=metadata.get("protocol_sha256",metadata.get("protocolSHA")))
                physics=metadata.get("initial_identity",{}).get("physics",{})
                slot["actual_damping"]=physics.get("space",{}).get("damping")
                slot["damping_matches_requested_arm"]=slot["actual_damping"]==(0. if request["protocol"]=="P0-frozen" else 1.)
                if metadata.get("telemetry_file"):
                    telemetry_path=_safe(root,metadata["telemetry_file"])
                    if metadata.get("telemetry_sha256") and _sha(telemetry_path)!=metadata["telemetry_sha256"]:raise ValueError("Telemetry SHA differs")
                    with gzip.open(telemetry_path,"rt",encoding="utf-8") as stream:telemetry=json.load(stream)
                    impulses=defaultdict(list);force_norms=[];unknown_force=0
                    for control in telemetry:
                        for sub in control.get("substeps",[]):
                            force=sub.get("clipped_body_force")
                            if force is not None and len(force)==2 and all(v is not None for v in force):force_norms.append(math.hypot(*force))
                            else:unknown_force+=1
                            for event in sub.get("contacts",[]):
                                if event.get("phase")=="post_solve" and event.get("total_impulse") is not None:
                                    impulse=event["total_impulse"]
                                    if len(impulse)==2 and all(v is not None for v in impulse):impulses[event.get("kind",event.get("category","unknown"))].append(math.hypot(*impulse))
                    slot["telemetry_summary"]=dict(charged_control_records=len(telemetry),actual_assigned_force_norm=distribution(force_norms,(10000,)),unknown_force_records=unknown_force,
                        post_solve_impulse_norm_by_kind={kind:distribution(values,()) for kind,values in impulses.items()},impulse_note="Native post_solve only; no doubled pre_solve impulses; unknown is not zero")
                for name,values in slot_populations.items():
                    populations[name].extend(values)
            except Exception as error:
                slot["data_error"] = type(error).__name__+": "+str(error)
                loads.append(dict(request_id=request["request_id"],error=slot["data_error"]))
        else:
            slot["data_missing"] = True
        slots.append(slot)
    planned = dict(all_controls=512,h1_windows=320,h4_windows=128)
    summaries = {name:summarize(populations[name],count) for name,count in planned.items()}
    by_protocol, by_letter = {}, {}
    for protocol in sorted({r["protocol"] for r in requests}):
        selected = [r for r in requests if r["protocol"]==protocol]
        by_protocol[protocol] = {name:summarize([r for r in populations[name] if r["protocol"]==protocol],len(selected)*per)
            for name,per in (("all_controls",8),("h1_windows",5),("h4_windows",2))}
    for letter in sorted({r["letter"] for r in requests}):
        selected = [r for r in requests if r["letter"]==letter]
        by_letter[letter] = {name:summarize([r for r in populations[name] if r["letter"]==letter],len(selected)*per)
            for name,per in (("all_controls",8),("h1_windows",5),("h4_windows",2))}
    raw_by_key = {(r["request_id"],r["anchor"]):r for r in populations["all_controls"]}
    slot_by_id = {r["request_id"]:r for r in slots}
    pairs = []
    for family in sorted({r["family_id"] for r in requests}):
        for action in sorted({r["action_variant"] for r in requests if r["family_id"]==family}):
            arms = {r["protocol"]:r for r in requests if r["family_id"]==family and r["action_variant"]==action}
            p0,p1 = arms["P0-frozen"],arms["P1-damping1-candidate"]
            s0,s1 = slot_by_id[p0["request_id"]],slot_by_id[p1["request_id"]]
            matched=[]
            for t in range(8):
                a,c = raw_by_key.get((p0["request_id"],t)),raw_by_key.get((p1["request_id"],t))
                if a is not None and c is not None:
                    matched.append(dict(control_index=t,P0=a["translation_image96_px"],P1=c["translation_image96_px"],
                        delta_translation_image96_px=None if a["translation_image96_px"] is None or c["translation_image96_px"] is None else c["translation_image96_px"]-a["translation_image96_px"],
                        delta_rotation_deg=None if a["absolute_rotation_deg"] is None or c["absolute_rotation_deg"] is None else c["absolute_rotation_deg"]-a["absolute_rotation_deg"],
                        both_valid=a["valid_window"] and c["valid_window"],P0_contact=a["agent_object_contact"],P1_contact=c["agent_object_contact"],P0_wall=a["object_wall_contact"],P1_wall=c["object_wall_contact"]))
            actual_initial_equal = None
            if "actual_initial_state" in s0 and "actual_initial_state" in s1:
                actual_initial_equal = s0["actual_initial_state"]==s1["actual_initial_state"] and s0["initial_rgb_sha256"]==s1["initial_rgb_sha256"]
            only_damping=None
            if s0.get("initial_identity",{}).get("physics") and s1.get("initial_identity",{}).get("physics"):
                physical0=deepcopy(s0["initial_identity"]["physics"]);physical1=deepcopy(s1["initial_identity"]["physics"])
                damping0=physical0["space"].pop("damping");damping1=physical1["space"].pop("damping")
                only_damping=damping0==0. and damping1==1. and physical0==physical1
            pairs.append(dict(family_id=family,action_variant=action,planned_control_pairs=8,observed_control_pairs=len(matched),
                missing_control_pairs=8-len(matched),frozen_scene_actions_equal=p0["initial_parameters_sha256"]==p1["initial_parameters_sha256"] and p0["actions_absolute"]==p1["actions_absolute"],
                actual_initial_state_and_rgb_equal=actual_initial_equal,controls=matched,
                actual_physics_only_damping_differs=only_damping,
                actual_actions_match_frozen=s0.get("source_actions_equal_request") is True and s1.get("source_actions_equal_request") is True,
                actual_damping_matches_both_arms=s0.get("damping_matches_requested_arm") is True and s1.get("damping_matches_requested_arm") is True,
                delta_translation=distribution((r["delta_translation_image96_px"] for r in matched),()),
                caveat="Paired observed response, not an ideal-contact/global-bound claim; contact/wall/coasting/state changes are reported separately"))
    coverage = {}
    for letter in by_letter:
        all_rows = [r for r in populations["all_controls"] if r["letter"]==letter]
        hit = defaultdict(set)
        for row in all_rows:
            for arc in row["actual_contact_arc_ids"]:
                hit[arc].add(row["family_id"])
        coverage[letter] = dict(planned_independent_families=len({r["family_id"] for r in requests if r["letter"]==letter}),
            initialized_independent_families=len({s["family_id"] for s in slots if s["letter"]==letter and "actual_initial_state" in s}),
            recorded_independent_families=len({r["family_id"] for r in all_rows}),actual_hit_arc_ids=sorted(hit),
            independent_families_per_actual_arc={k:len(v) for k,v in hit.items()},
            contact_controls_without_actual_arc_mapping=sum(r["agent_object_contact"] is True and not r["actual_arc_mapping_complete"] for r in all_rows),
            measured_boundary_processes=dict(Counter(r["boundary_process"] for r in all_rows)))
    groups = {key:{label:summarize([r for r in populations["all_controls"] if str(r[key])==label],None)
                for label in sorted({str(r[key]) for r in populations["all_controls"]})}
        for key in ("initial_space","phase","agent_object_contact","object_wall_contact","boundary_process","actual_spatial_class_before","min_gap_band_before","actual_corner_gap_pair_before")}
    return dict(kind="native-rgb-R2A-cold-offline-report.v1",requests_sha256=_sha(request_path),index_sha256=_sha(index_path),
        collection_closed=True,planned_slots=64,slot_status_counts=dict(Counter(r["status"] for r in slots)),slots=slots,
        populations=dict(populations),summaries=summaries,by_protocol=by_protocol,by_letter=by_letter,
        groups_all_controls=groups,protocol_pairs=pairs,planned_protocol_pairs=32,planned_control_pairs=256,
        observed_control_pairs=sum(p["observed_control_pairs"] for p in pairs),actual_contact_coverage=coverage,data_errors=loads,
        qualified_control_pairs=sum(sum(c["both_valid"] for c in p["controls"]) for p in pairs if p["actual_initial_state_and_rgb_equal"] is True and p["actual_actions_match_frozen"] and p["actual_damping_matches_both_arms"] and p["actual_physics_only_damping_differs"] is True),
        charged_controls_declared=sum(s["charged_controls"] for s in slots if s["charged_controls"] is not None),
        actual_controls_declared=sum(s["actual_controls"] for s in slots if s["actual_controls"] is not None),
        partial_attempt_records=sum(len(s.get("partial_attempt_records",[])) for s in slots),
        conditional_group_scope="Dynamic group planneddenominators are unknown/null; groupobservedN is conditional, global/perletter/protocol planneddenominators remain fixed",
        acceptance_scope="Amplitude/coverage/quality evidence only, no automaticprotocolR2Bortrainingrelease. Missing/failed/unexecuted slots keep planned denominators.",
        timing_scope="rawcompletecontrol=.1s;H1requires4historyframesandvalid4transitions;H4endpoint=.4srequires7transitions;partialattemptsseparate",
        new_environment_or_physics_steps=0,new_model_or_GPU_calls=0,renderer_calls=0)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"r2a-report.json").open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    for name,rows in report["populations"].items():
        fields=sorted({key for row in rows for key in row})
        with (output/(name+".csv")).open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    lines=["# R2A离线幅度、配对与覆盖统计", "", "仅分析保存记录，0新增仿真/渲染/模型。计划64槽/512控制；失败、未执行和缺测不缩分母。", "",
        "|总体|计划分母|完整可观测|有效窗口|缺少计划项|平移中位/P95/max(图像px)|角度中位/P95/max(度)|", "|---|---:|---:|---:|---:|---|---|"]
    def fmt(v):return "缺测" if v is None else f"{v:.4f}"
    for name,s in report["summaries"].items():
        t,a=s["translation_image96_px"],s["rotation_deg"]
        lines.append(f"|{name}|{s['planned']}|{s['observed_complete']}|{s['valid_windows']}|{s['missing_planned']}|"+"/".join(fmt(t[k]) for k in ("median","p95","max"))+"|"+"/".join(fmt(a[k]) for k in ("median","p95","max"))+"|")
    lines += ["", "8个完整控制只有5个真正H1窗(anchor3..7)、2个H4窗(anchor3..4)；H4为4步端点，不替代H1单步。部分物理尝试单列，不能伪作完整0.1s。位移为origin+R×面积质心；T面积局部CoG(0,40.7142857)与物理CoG(0,45)分列。", "",
        f"计划32个协议pair/256个控制pair，实际可观测{report['observed_control_pairs']}控制pair。必须同时检查配对初态/action、真实接触、no-contact与墙collision；P1−P0只是观测响应差，不是理想接触或唯一因果保证。", "",
        "1/2/3/4/8px与5/15°阈值计数、逐字母/协议/阶段/真实接触/动态墙角gap见JSON/CSV。近墙+静止只有proxy；没有实际arc映射或wall-arbiter记录就unknown，不用计划arc填充。初始四墙四角标签不等于实际过程覆盖。", "",
        "全部槽状态："+json.dumps(report['slot_status_counts'],ensure_ascii=False)+"。数据读取错误："+str(len(report['data_errors']))+"；任何缺测/无效都保留计划分母。R2B、协议选择和训练仍须父独立验收。"]
    with (output/"REPORT.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--requests",type=Path,default=ROOT/"artifacts/native-rgb-supervised-20261029/r1/r2-probe-requests.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute-after-root-cold",action="store_true")
    args=parser.parse_args()
    if not args.execute_after_root_cold:
        raise SystemExit("Explicit --execute-after-root-cold required; never inspect a live collector")
    report=build_report(args.data_root,args.requests)
    write_report(report,args.output)
    print(json.dumps(dict(planned_slots=64,all_controls=report['summaries']['all_controls']['observed_complete'],
        h1_windows=report['summaries']['h1_windows']['observed_complete'],h4_windows=report['summaries']['h4_windows']['observed_complete'],
        paired_controls=report['observed_control_pairs'],data_errors=len(report['data_errors']),new_physics_steps=0),ensure_ascii=False))


if __name__=="__main__":
    main()
