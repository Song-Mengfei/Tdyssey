"""Cold, recorded-only R2A telemetry audit. Never constructs or steps physics."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tdyssey.envs.official import official_shape
from tdyssey.native_rgb_data import contour_segments
from tdyssey.native_rgb_env import shape_geometry


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    values = list(values)
    finite = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    result = dict(n=len(values), finite=int(len(finite)), unknown_or_nonfinite=len(values)-len(finite))
    if len(finite):
        result.update(mean=float(finite.mean()), min=float(finite.min()), max=float(finite.max()),
            **{f"p{q}": float(np.percentile(finite, q)) for q in (50, 90, 95, 99)})
    return result


def vector(value, length):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if result.shape == (length,) and np.isfinite(result).all() else None


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def center(state, local):
    c, s = math.cos(state[4]), math.sin(state[4])
    return state[2:4] + np.asarray((c*local[0]-s*local[1], s*local[0]+c*local[1]))


def mapped_arcs(local, rings, segments):
    from shapely.geometry import Point
    point = Point(*local)
    ring_id = min(range(len(rings)), key=lambda i: rings[i].distance(point))
    arc = float(rings[ring_id].project(point))
    if rings[ring_id].distance(point) > 2.0+1e-6:
        return []
    ids = [row["id"] for row in segments if row["ring"] == ring_id and row["arc_start"] <= arc < row["arc_end"]]
    return ids or [row["id"] for row in segments if row["ring"] == ring_id and abs(row["arc_end"]-arc) <= 1e-9][-1:]


def category(event):
    return event.get("kind", event.get("category", "unknown")).replace("-", "_")


def process(base):
    base = Path(base).resolve()
    index, ledger = read(base/"index.json"), read(base/"resource-ledger.json")
    if index.get("collection_closed") is not True or ledger.get("closed") is not True:
        raise RuntimeError("Refusing hot or unfinished R2A data; no polling or waiting")
    metrics = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(Counter)
    impulses = defaultdict(lambda: defaultdict(list))
    groups = defaultdict(lambda: defaultdict(list))
    checks, anomalies, slot_rows, control_rows = {}, [], [], []
    pair_data = defaultdict(list)
    geometry_cache = {}
    arc_support = defaultdict(lambda: defaultdict(lambda: dict(point_samples=0, slots=set(), families=set())))
    source = read(base/"source-identity.json")
    source_bad = []
    for row in source["files"]:
        for key in ("source", "snapshot"):
            path = Path(row[key]) if key == "source" else base/row[key]
            if not path.is_file() or sha(path) != row["sha256"]:
                source_bad.append(dict(path=str(path), binding=key))
    checks["all_frozen_source_and_snapshot_hashes"] = not source_bad
    checks["model_gpu_optimizer_zero_in_collection_ledger"] = all(ledger.get(key) == 0
        for key in ("model_calls", "optimizer_attempts", "GPU_work"))
    total_logs = total_steps = total_completed = 0
    for slot in index["slots"]:
        if not slot.get("initialization_attempted"):
            slot_rows.append(dict(slot_id=slot["slot_id"], status=slot["status"], attempted=False))
            continue
        meta_path, npz_path = base/slot["metadata_file"], base/slot["trajectory_file"]
        if not meta_path.exists() or not npz_path.exists():
            anomalies.append(dict(slot_id=slot["slot_id"], reason="attempted initialization without saved metadata/NPZ"))
            continue
        meta = read(meta_path)
        telemetry_path = base/meta["telemetry_file"]
        for path, expected in ((meta_path, slot["metadata_sha256"]), (npz_path, meta["trajectory_sha256"]),
                               (telemetry_path, meta["telemetry_sha256"])):
            if sha(path) != expected:
                raise ValueError(f"Cold input hash mismatch: {path}")
        with gzip.open(telemetry_path, "rt", encoding="utf-8") as stream:
            logs = json.load(stream)
        with np.load(npz_path, allow_pickle=False) as z:
            states, actions, images = z["states"].copy(), z["actions"].copy(), z["rgb"].copy()
        identity, req = meta["initial_identity"], meta["request"]
        physics, arm, letter = identity["physics"], identity["arm"], req["letter"]
        damping, gravity = float(physics["space"]["damping"]), np.asarray(physics["space"]["gravity"])
        mass, object_mass = float(physics["agent"]["mass"]), float(physics["object"]["mass"])
        cog = np.asarray(physics["object"]["center_of_gravity"])
        area_local = np.asarray(official_shape(letter).centroid)
        if letter not in geometry_cache:
            geom = shape_geometry(letter)
            geometry_cache[letter] = ([geom.exterior, *geom.interiors], contour_segments(letter, 8))
        rings, segments = geometry_cache[letter]
        protocol = physics["base_control_protocol"]
        cap, speed_cap = float(protocol["force_limit"]), float(protocol["pre_step_speed_limit"])
        kp, kv = float(physics["kp"]), float(physics["kv"])
        counts[arm]["slots_initialized"] += 1
        counts[arm]["controls_charged"] += len(logs)
        counts[arm]["controls_completed"] += len(actions)
        total_logs += len(logs)
        total_completed += len(actions)
        checks[slot["slot_id"]+":frame_state_action_timing"] = (
            states.shape == (len(actions)+1, 15) and images.shape == (len(actions)+1, 3, 96, 96)
            and np.array_equal(actions, np.asarray(req["actions_absolute"][:len(actions)]))
            and len(logs) == slot["charged_controls"])
        checks[slot["slot_id"]+":frame_hashes"] = len(images) == len(meta["frame_sha256"]) and all(
            hashlib.sha256(im.tobytes()).hexdigest() == expected
            for im, expected in zip(images, meta["frame_sha256"]))
        pair_data[(req["family_id"], req["action_variant"])].append(dict(arm=arm,
            physics=physics, state=states[0], image=images[0], actions=req["actions_absolute"]))
        slot_count = Counter()
        last_post = None
        contact_history = []
        for log in logs:
            control, steps = int(log["control_index"]), log["substeps"]
            action = np.asarray(log["action"], dtype=np.float64)
            total_steps += len(steps)
            slot_count["substeps"] += len(steps)
            kinds_control = set()
            all_control_valid = True
            control_post_impulse = 0.
            control_force_toward_object = False
            for sub in steps:
                counts[arm]["substeps_recorded"] += 1
                kinds = {category(event) for event in sub.get("contacts", [])}
                kinds_control.update(kinds)
                contact_history.append("agent_object" in kinds)
                ref = dict(slot_id=slot["slot_id"], control=control, substep=sub.get("index"))
                pre, entry, post = (vector(sub.get(key), 15) for key in (
                    "pre_state", "integrator_pre_state", "post_state"))
                pen = vector(sub.get("penetrations"), 3)
                if any(item is None for item in (pre, entry, post, pen)):
                    counts[arm]["nonfinite_or_unknown_state_or_penetration"] += 1
                    anomalies.append(dict(**ref, reason="nonfinite or unknown recorded state/penetration"))
                    all_control_valid = False
                    last_post = None
                    continue
                if last_post is not None and not np.array_equal(pre, last_post):
                    anomalies.append(dict(**ref, reason="pre/post continuous-state mismatch"))
                last_post = post
                dt = float(sub["dt"])
                if dt != .01:
                    anomalies.append(dict(**ref, reason="non-100Hz step"))
                counts[arm]["penetration_gt2"] += int(np.max(pen) > 2.)
                metrics[arm]["penetration_world_px"].append(float(np.max(pen)))
                expected_raw = mass*(kp*(action-pre[:2])-kv*pre[5:7])
                expected_force = expected_raw.copy()
                norm = float(np.linalg.norm(expected_force))
                force_clipped = norm > cap
                if force_clipped:
                    expected_force *= cap/norm
                before_speed = float(np.linalg.norm(pre[5:7]))
                expected_velocity = pre[5:7].copy()
                speed_clipped = before_speed > speed_cap
                if speed_clipped:
                    expected_velocity *= speed_cap/before_speed
                recorded_raw = vector(sub.get("exact_formula_reconstruction", {}).get("unclipped_force"), 2)
                recorded_reconstructed = vector(sub.get("exact_formula_reconstruction", {}).get("clipped_force"), 2)
                measured_force = vector(sub.get("clipped_body_force"), 2)
                before_cap = vector(sub.get("velocity_before_cap"), 2)
                after_cap = vector(sub.get("velocity_after_cap"), 2)
                if any(item is None for item in (recorded_raw, recorded_reconstructed, measured_force, before_cap, after_cap)):
                    anomalies.append(dict(**ref, reason="missing finite force/cap telemetry"))
                    all_control_valid = False
                    continue
                errors = dict(unclipped_reconstruction=maxdiff(recorded_raw, expected_raw),
                    clipped_reconstruction=maxdiff(recorded_reconstructed, expected_force),
                    measured_clipped_force=maxdiff(measured_force, expected_force),
                    before_cap_velocity=maxdiff(before_cap, pre[5:7]),
                    after_cap_velocity=maxdiff(after_cap, expected_velocity),
                    integrator_entry_velocity=maxdiff(entry[5:7], after_cap),
                    integrator_entry_force=maxdiff(entry[10:12], measured_force))
                for key, value in errors.items():
                    metrics[arm][key+"_maxabs_error"].append(value)
                if max(errors.values()) > 1e-8:
                    anomalies.append(dict(**ref, reason="force/cap reconstruction mismatch", errors=errors))
                counts[arm]["force_clipped_substeps"] += int(force_clipped)
                counts[arm]["speed_clipped_substeps"] += int(speed_clipped)
                counts[arm]["force_reconstruction_bitwise_equal"] += int(np.array_equal(recorded_raw, expected_raw))
                for key, value in (("unclipped_force_norm", norm), ("measured_clipped_force_norm", np.linalg.norm(measured_force)),
                    ("agent_speed_before_cap", before_speed), ("agent_speed_after_cap", np.linalg.norm(after_cap)),
                    ("agent_speed_post", np.linalg.norm(post[5:7])), ("object_mass_COM_speed_post", np.linalg.norm(post[7:9])),
                    ("object_abs_omega_post_rad_s", abs(post[9]))):
                    metrics[arm][key].append(float(value))
                free_agent = not kinds.intersection(("agent_object", "agent_wall"))
                free_object = not kinds.intersection(("agent_object", "object_wall"))
                if free_agent:
                    expected_post = entry[5:7]*(damping**dt)+(gravity+measured_force/mass)*dt
                    other = 1. if damping == 0. else 0.
                    other_post = entry[5:7]*(other**dt)+(gravity+measured_force/mass)*dt
                    metrics[arm]["free_agent_actual_damping_velocity_residual_norm"].append(float(np.linalg.norm(post[5:7]-expected_post)))
                    metrics[arm]["free_agent_other_damping_counterfactual_residual_norm"].append(float(np.linalg.norm(post[5:7]-other_post)))
                    metrics[arm]["free_agent_entry_speed"].append(float(np.linalg.norm(entry[5:7])))
                    counts[arm]["free_agent_substeps"] += 1
                    counts[arm]["free_agent_nonzero_entry_velocity_substeps"] += int(np.linalg.norm(entry[5:7]) > 1e-8)
                if free_object:
                    expected_object = entry[7:9]*(damping**dt)+(gravity+entry[12:14]/object_mass)*dt
                    metrics[arm]["free_object_actual_damping_velocity_residual_norm"].append(float(np.linalg.norm(post[7:9]-expected_object)))
                    moment = physics["object"]["moment"]
                    expected_omega = entry[9]*(damping**dt) + (0. if moment == "infinity" else entry[14]/float(moment))*dt
                    metrics[arm]["free_object_actual_damping_omega_residual_abs"].append(float(abs(post[9]-expected_omega)))
                    counts[arm]["free_object_substeps"] += 1
                for kind in ("agent_object", "object_wall", "agent_wall"):
                    counts[arm][kind+"_contact_substeps"] += int(kind in kinds)
                for event in sub.get("contacts", []):
                    kind = category(event)
                    union_ids = set()
                    for point in event["points"]:
                        if "object_local" not in point:
                            continue
                        ids = mapped_arcs(point["object_local"], rings, segments)
                        if ids != point["arc_ids"]:
                            anomalies.append(dict(**ref, reason="independent canonical 8px point-arc mismatch"))
                        union_ids.update(ids)
                        counts[arm]["object_contact_point_arc_mappings_verified"] += 1
                        if kind == "agent_object" and event.get("phase") == "post_solve":
                            for arc_id in ids:
                                support = arc_support[arm][f"{letter}:{arc_id}"]
                                support["point_samples"] += 1
                                support["slots"].add(slot["slot_id"])
                                support["families"].add(req["family_id"])
                    if sorted(union_ids) != event["arc_ids"]:
                        anomalies.append(dict(**ref, reason="event arc union mismatch"))
                    counts[arm]["contact_events_"+event.get("phase", "unknown")+":"+kind] += 1
                    if event.get("phase") != "post_solve":
                        if "total_impulse" in event:
                            anomalies.append(dict(**ref, reason="impulse incorrectly recorded in pre_solve"))
                        continue
                    impulse = vector(event.get("total_impulse"), 2)
                    if impulse is None:
                        anomalies.append(dict(**ref, reason="nonfinite/missing post_solve impulse"))
                        continue
                    impulses[arm][kind+"_native_impulse_norm"].append(float(np.linalg.norm(impulse)))
                    impulses[arm][kind+"_total_ke"].append(float(event["total_ke"]))
                    if kind == "agent_object":
                        control_post_impulse += float(np.linalg.norm(impulse))
                # Source-ordered reconstruction, not a logged callback velocity:
                # Chipmunk updates positions before pre_solve, then velocities,
                # then post_solve; pose is unchanged by the velocity solver.
                # Thus callback pose is post pose and pre_solve velocities are
                # measured integrator-entry velocities. Pusher cannot spin.
                com_world = center(post, cog)
                for event in sub.get("contacts", []):
                    if category(event) != "agent_object" or event.get("phase") != "pre_solve":
                        continue
                    normal = np.asarray(event["normal"], dtype=float)
                    if event["shapes_order"][0] != "object":
                        normal = -normal
                    tangent = np.asarray((-normal[1], normal[0]))
                    for point in event["points"]:
                        world, local = np.asarray(point["object_world"]), np.asarray(point["object_local"])
                        lever = world-com_world
                        point_velocity = entry[7:9] + entry[9]*np.asarray((-lever[1], lever[0]))
                        relative = entry[5:7]-point_velocity
                        closing, tangential = float(-relative@normal), float(relative@tangent)
                        metrics[arm]["contact_approach_reconstructed_closing_speed"].append(closing)
                        metrics[arm]["contact_approach_reconstructed_tangential_speed"].append(tangential)
                        metrics[arm]["contact_approach_reconstructed_abs_tangential_speed"].append(abs(tangential))
                        metrics[arm]["contact_approach_reconstructed_angle_deg"].append(float(math.degrees(math.atan2(tangential, closing))))
                        force_toward = float(-measured_force@normal)
                        metrics[arm]["contact_applied_force_toward_object_normal"].append(force_toward)
                        counts[arm]["contact_force_toward_object_positive_point_samples"] += int(force_toward > 1e-8)
                        control_force_toward_object |= force_toward > 1e-8
                        metrics[arm]["contact_point_physical_COM_lever_length_world_px"].append(float(np.linalg.norm(local-cog)))
                        metrics[arm]["contact_normal_lever_length_world_px"].append(float(abs(lever[0]*normal[1]-lever[1]*normal[0])))
                        metrics[arm]["contact_world_local_pose_reconstruction_error"].append(float(np.linalg.norm(center(post, local)-world)))
                        counts[arm]["contact_pre_solve_point_samples"] += 1
                        counts[arm]["contact_reconstructed_closing_positive_point_samples"] += int(closing > 1e-8)
                        counts[arm]["contact_reconstructed_separating_point_samples"] += int(closing < -1e-8)
                        counts[arm]["contact_reconstructed_tangential_gt1_world_s_point_samples"] += int(abs(tangential) > 1.)
            if log["completed"] and control < len(actions):
                a, b = states[control:control+2]
                area_distance = float(np.linalg.norm(center(b, area_local)-center(a, area_local))*96/512)
                mass_distance = float(np.linalg.norm(center(b, cog)-center(a, cog))*96/512)
                angle = float((b[4]-a[4])*180/math.pi)
                control_row = dict(slot_id=slot["slot_id"], arm=arm, letter=letter,
                    family_id=req["family_id"], action_variant=req["action_variant"], control=control,
                    action_phase=req["action_schedule"][control], area_centroid_translation_image_px=area_distance,
                    physical_mass_COM_translation_image_px=mass_distance, signed_rotation_deg=angle,
                    agent_object_post_solve_impulse_norm_sum=control_post_impulse,
                    any_measured_force_toward_object_normal=bool(control_force_toward_object),
                    contact_kinds=sorted(kinds_control), all_substeps_finite=all_control_valid)
                control_rows.append(control_row)
                labels = ["all_raw_controls", "action_phase:"+str(control_row["action_phase"]), "letter:"+letter]
                labels.append("agent_object_contact" if "agent_object" in kinds_control else "no_agent_object_contact")
                if "agent_object" in kinds_control:
                    labels.append("contact_with_nonzero_impulse" if control_post_impulse > 1e-12 else "contact_without_measured_impulse")
                    if control_post_impulse > 1e-12 and control_force_toward_object:
                        labels.append("contact_impulse_and_force_toward_object_normal")
                if not kinds_control.intersection(("agent_object", "object_wall")):
                    labels.append("free_object_coasting_no_agent_or_wall_contact")
                for kind in ("object_wall", "agent_wall"):
                    if kind in kinds_control:
                        labels.append(kind+"_collision")
                for label in labels:
                    groups[arm][label].append(control_row)
        runs = []
        run_length = 0
        for is_contact in contact_history+[False]:
            if is_contact:
                run_length += 1
            elif run_length:
                runs.append(run_length)
                run_length = 0
        losses = sum(a and not b for a, b in zip(contact_history, contact_history[1:]))
        recontacts = max(0, len(runs)-1)
        counts[arm]["agent_object_contact_episodes"] += len(runs)
        counts[arm]["agent_object_contact_loss_transitions"] += losses
        counts[arm]["agent_object_recontacts_after_observed_episode"] += recontacts
        metrics[arm]["agent_object_contact_episode_seconds"].extend(length*.01 for length in runs)
        slot_rows.append(dict(slot_id=slot["slot_id"], status=slot["status"], arm=arm, letter=letter,
            charged_controls=len(logs), completed_controls=len(actions), **dict(slot_count),
            contact_episodes=len(runs), observed_loss_transitions=losses, recontacts_after_observed_episode=recontacts,
            initial_materials=[dict(role=row["role"], friction=row["friction"], elasticity=row["elasticity"])
                for row in physics["shapes"]]))
    pairs = []
    for key, rows in pair_data.items():
        by_arm = {row["arm"]: row for row in rows}
        result = dict(family_id=key[0], action_variant=key[1], arms=sorted(by_arm))
        if set(by_arm) == {"P0", "P1"}:
            a, b = by_arm["P0"], by_arm["P1"]
            pa, pb = json.loads(json.dumps(a["physics"])), json.loads(json.dumps(b["physics"]))
            da, db = pa["space"].pop("damping"), pb["space"].pop("damping")
            result.update(initial_states_bitwise_equal=np.array_equal(a["state"], b["state"]),
                initial_images_bitwise_equal=np.array_equal(a["image"], b["image"]),
                planned_absolute_actions_equal=a["actions"] == b["actions"],
                sole_physics_difference_damping=da == 0. and db == 1. and pa == pb)
        pairs.append(result)
    checks.update(recorded_substeps_match_ledger=total_steps == ledger["engine_step_calls"],
        charged_controls_match_ledger=total_logs == ledger["charged_controls"],
        completed_controls_match_ledger=total_completed == ledger["completed_controls"],
        every_available_pair_matches=all(all(value is True for name, value in row.items()
            if name.endswith(("equal", "damping"))) for row in pairs),
        no_telemetry_invariant_anomalies=not anomalies)
    group_stats = {}
    for arm, labels in groups.items():
        group_stats[arm] = {label: dict(controls=len(rows), slots=len({row["slot_id"] for row in rows}),
            independent_families=len({row["family_id"] for row in rows}),
            area_centroid_translation_image_px=stats(row["area_centroid_translation_image_px"] for row in rows),
            physical_mass_COM_translation_image_px=stats(row["physical_mass_COM_translation_image_px"] for row in rows),
            signed_rotation_deg=stats(row["signed_rotation_deg"] for row in rows),
            rotation_abs_deg=stats(abs(row["signed_rotation_deg"]) for row in rows))
            for label, rows in labels.items()}
    return dict(kind="R2A-independent-recorded-telemetry-audit.v1", passed=all(checks.values()),
        collection_closed=True, index_sha256=sha(base/"index.json"), ledger_sha256=sha(base/"resource-ledger.json"),
        frozen_source_identity_sha256=sha(base/"source-identity.json"), checks=checks,
        source_mismatches=source_bad, anomalies=anomalies, planned_slots=index["planned_slots"],
        planned_controls=index["planned_controls"], ledger=ledger,
        counts={arm: dict(value) for arm, value in counts.items()},
        substep_statistics={arm: {key: stats(values) for key, values in items.items()} for arm, items in metrics.items()},
        post_solve_impulse_statistics={arm: {key: stats(values) for key, values in items.items()} for arm, items in impulses.items()},
        actual_agent_object_post_solve_arc_support={arm: {key: dict(point_samples=row["point_samples"],
            independent_slots=len(row["slots"]), independent_initial_families=len(row["families"]),
            slots=sorted(row["slots"]), families=sorted(row["families"])) for key, row in items.items()}
            for arm, items in arc_support.items()},
        raw_control_motion_groups=group_stats, slots=slot_rows, pairs=pairs, controls=control_rows,
        measurement_boundaries=dict(unclipped_force="independent exact formula reconstruction from recorded pre-state/action; not original local-variable measurement",
            clipped_force="measured Body.force at original Space.step entry",
            free_velocity_equation="post=D**dt*entry+(gravity+measuredForce/mass)*dt, restricted to no recorded contacts involving the body",
            other_damping_residual="counterfactual algebra on same entry/force; not a new physical rollout or paired trajectory prediction",
            approach_and_slip="source-ordered reconstruction using measured integrator-entry velocities, callback contact points and post pose; not directly logged callback locals; tangential speed diagnostic is not proof of frictional sticking or sliding",
            lever="actual callback point relative to physical mass CoG; normal moment arm uses actual callback normal; no inference that larger lever causes larger rotation",
            contact_runs="100Hz any-agent-object-event episodes; a recontact requires an earlier observed episode and intervening no-contact substep; no selected episodes omitted",
            pushing_evidence="contact+nonzero post impulse+force toward actual object normal is joint observed evidence, not proof that every motion was actively initiated by the pusher; no-contact motion remains separate",
            impulse="native post_solve arbiter impulse norm once per post event; pre events excluded, no fabricated net-body sign",
            coasting="raw 0.1s controls with no object-agent/object-wall event across the whole control; not evidence of ideal sampling",
            motion="area centroid for visual displacement; physical mass center separate; 96/512 scale",
            no_new_work="no simulator construction/reset/step/render, no Torch/model/GPU/optimizer, no training iterator",
            interpretation="P1 is an authorized candidate, not selected final physics; contact and wall behavior remain outcomes to assess"))


def report(result):
    lines = ["# Independent R2A recorded-telemetry audit", "", f"Integrity checks passed: **{result['passed']}**.", "",
        f"Recorded {result['ledger']['charged_controls']} charged controls and {result['ledger']['executed_substeps']} completed physics substeps; "
        f"planned denominators remain {result['planned_slots']} slots / {result['planned_controls']} controls.", "",
        "The unclipped force is an exact formula reconstruction from recorded state and action. The clipped force and cap-entry velocity are engine-property measurements. "
        "Only post_solve impulses are aggregated; pre_solve events are excluded from impulse totals.", ""]
    for arm in ("P0", "P1"):
        counts = result["counts"].get(arm, {})
        metrics = result["substep_statistics"].get(arm, {})
        lines.extend([f"## {arm}", "", f"Recorded substeps: {counts.get('substeps_recorded',0)}; force clipping: {counts.get('force_clipped_substeps',0)}; "
            f"speed clipping: {counts.get('speed_clipped_substeps',0)}.", ""])
        for key in ("measured_clipped_force_maxabs_error", "after_cap_velocity_maxabs_error", "free_agent_actual_damping_velocity_residual_norm",
                    "free_agent_other_damping_counterfactual_residual_norm", "free_object_actual_damping_velocity_residual_norm", "agent_speed_post",
                    "object_mass_COM_speed_post", "penetration_world_px", "contact_approach_reconstructed_closing_speed",
                    "contact_approach_reconstructed_abs_tangential_speed", "contact_point_physical_COM_lever_length_world_px",
                    "agent_object_contact_episode_seconds"):
            row = metrics.get(key, {})
            lines.append(f"- {key}: n={row.get('finite',0)}, median={row.get('p50')}, max={row.get('max')}.")
        lines.append("")
        lines.append(f"Contact episodes={counts.get('agent_object_contact_episodes',0)}; observed contact-loss transitions={counts.get('agent_object_contact_loss_transitions',0)}; "
            f"recontacts after an earlier observed episode={counts.get('agent_object_recontacts_after_observed_episode',0)}.")
        lines.append("")
        for key in ("all_raw_controls", "contact_with_nonzero_impulse", "contact_impulse_and_force_toward_object_normal",
                "no_agent_object_contact", "free_object_coasting_no_agent_or_wall_contact", "object_wall_collision", "agent_wall_collision"):
            row = result["raw_control_motion_groups"].get(arm, {}).get(key, {})
            motion = row.get("area_centroid_translation_image_px", {})
            lines.append(f"- {key}: {row.get('controls',0)} controls / {row.get('independent_families',0)} initial families; "
                f"area-centroid image displacement median={motion.get('p50')}, max={motion.get('max')}.")
        lines.append("")
    lines.extend(["## Limits", "", "The no-contact velocity equation is tested only where neither pre- nor post-solve telemetry records a contact involving that body. "
        "The alternative-damping calculation uses the same recorded entry state and force; it is algebra, not an unexecuted trajectory. "
        "Approach and tangential speeds use a source-ordered reconstruction: entry velocities plus actual callback points and the same substep post pose. "
        "They are not direct callback-local measurements; tangential motion is not proof of a frictional sliding law. "
        "Actual contact shape friction and elasticity are zero in both arms; normal impulses still exchange momentum. "
        "A nonzero contact impulse together with force toward the object normal is observed pushing evidence, not a claim that every displacement was actively initiated by the pusher. "
        "P1 coasting and larger motion do not establish ideal sampling or select a final protocol. Wall impulses and peak speeds remain stability evidence. "
        "Raw-control coasting summaries are not H1/H4 window denominators. The main R2A report retains proper history windows and all failed/unexecuted slots.", "",
        f"Source mismatches: {len(result['source_mismatches'])}; telemetry anomalies: {len(result['anomalies'])}. Full counts, failures, pair checks and row references are in statistics.json.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collection-closed", action="store_true", required=True,
        help="Operator confirms root explicitly declared the collection cold; this does not override closed-file checks")
    args = parser.parse_args()
    result = process(args.root)
    out = args.root/"telemetry-audit"
    (out/"statistics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    (out/"REPORT.md").write_text(report(result), encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]), anomalies=len(result["anomalies"]))))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
