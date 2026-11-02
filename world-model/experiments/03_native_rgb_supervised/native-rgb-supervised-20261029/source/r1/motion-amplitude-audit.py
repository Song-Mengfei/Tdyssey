"""Read saved TRAIN states only; no images, models, environment creation or steps."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tdyssey.envs.official import official_shape
from tdyssey.native_rgb_motion_statistics import transition_rows

BASE = ROOT / "artifacts/native-rgb-supervised-20261029"
OUT = BASE / "r1"
SCALE = 96 / 512
STRATA = ("moving", "intermediate", "near_static_contact", "near_static_no_contact")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(values, thresholds=()):
    array = np.asarray(values, np.float64)
    if not len(array):
        return dict(n=0, min=None, median=None, p90=None, p95=None, max=None, mean=None,
                    at_least={str(t): 0 for t in thresholds})
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite saved diagnostic value")
    return dict(n=len(array), min=float(array.min()), median=float(np.median(array)),
        p90=float(np.percentile(array, 90)), p95=float(np.percentile(array, 95)), max=float(array.max()),
        mean=float(array.mean()), at_least={str(t): int((array >= t).sum()) for t in thresholds})


def summarize(rows):
    return dict(n=len(rows), independent_trajectories=len({r["trajectory_id"] for r in rows}),
        translation_image96_px=stats([r["translation_image96_px"] for r in rows], (1, 2, 3, 4, 8)),
        translation_world_px=stats([r["translation_world_px"] for r in rows]),
        physical_CoG_translation_image96_px=stats([r["physical_CoG_translation_image96_px"] for r in rows], (1, 2, 3)),
        rotation_deg=stats([r["rotation_deg"] for r in rows], (1, 3, 5, 10, 15)),
        target_distance_world_px=stats([r["target_distance_world_px"] for r in rows]),
        endpoint_pusher_speed_world_px_s=stats([r["endpoint_pusher_speed_world_px_s"] for r in rows]),
        endpoint_object_CoG_speed_world_px_s=stats([r["endpoint_object_CoG_speed_world_px_s"] for r in rows]),
        actual_min_wall_gap_world_px=stats([r["min_wall_gap_world_px"] for r in rows]),
        actual_second_perpendicular_wall_gap_world_px=stats([r["second_perpendicular_wall_gap_world_px"] for r in rows]),
        reconstructed_first_substep_preclip_force_norm=stats([r["reconstructed_first_substep_preclip_force_norm"] for r in rows]),
        reconstructed_first_substep_force_norm=stats([r["reconstructed_first_substep_force_norm"] for r in rows]),
        reconstructed_first_substep_force_saturated=sum(r["reconstructed_first_substep_force_saturated"] for r in rows))


def grouped(rows, field):
    return {str(value): summarize([r for r in rows if r[field] == value])
            for value in sorted({r[field] for r in rows}, key=str)}


def motion_bucket(distance, angle):
    trans_gate, rot_gate = distance >= 2., angle >= math.radians(1.)
    return ("mixed_threshold_motion" if trans_gate and rot_gate else
            "translation_threshold_only" if trans_gate else "rotation_threshold_only" if rot_gate else
            "near_static" if distance <= .5 and angle <= .005 else "intermediate")


def main():
    began = time.monotonic()
    manifest_path = BASE / "stage2b/report/merged-train-manifest.json"
    manifest = read(manifest_path)
    h1, h4, raw = [], [], []
    first_force, first_force_preclip, raw_pusher_speed, raw_block_speed, raw_omega = [], [], [], [], []
    force_zero, force_total = 0, 0
    largest_boundary_force = 0.
    definition_delta = 0.
    lookup, centroid_local, source_files = {}, {}, []
    for entry in manifest["entries"]:
        path = BASE / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Saved TRAIN NPZ identity differs")
        meta_path = BASE / entry["metadata_file"]
        if sha(meta_path) != entry["metadata_sha256"]:
            raise ValueError("Saved TRAIN metadata identity differs")
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        states, actions = arrays["states"], arrays["actions"]
        meta = read(meta_path)
        old_rows = transition_rows(entry, arrays, meta["steps"])
        letter = entry["shape"]
        local = np.asarray(official_shape(letter).centroid, np.float64)
        centroid_local[letter] = local.tolist()
        theta = states[:, 4]
        rotation = np.stack((np.stack((np.cos(theta), -np.sin(theta)), axis=-1),
                             np.stack((np.sin(theta), np.cos(theta)), axis=-1)), axis=1)
        centers = states[:, 2:4] + np.einsum("nij,j->ni", rotation, local)
        cog_local = np.asarray((0., 45.) if letter == "T" else (0., 0.))
        cog = states[:, 2:4] + np.einsum("nij,j->ni", rotation, cog_local)
        velocity_p = np.linalg.norm(states[:, 5:7], axis=1)
        velocity_b = np.linalg.norm(states[:, 7:9], axis=1)
        raw_pusher_speed.extend(velocity_p[1:].tolist())
        raw_block_speed.extend(velocity_b[1:].tolist())
        raw_omega.extend(np.degrees(np.abs(states[1:, 9])).tolist())
        forces = np.abs(states[:, 10:15])
        force_zero += int((forces == 0).all(axis=1).sum())
        force_total += len(forces)
        largest_boundary_force = max(largest_boundary_force, float(forces.max()))
        command = 100 * (actions.astype(np.float64) - states[:-1, :2]) - 20 * states[:-1, 5:7]
        command_norm = np.linalg.norm(command, axis=1)
        first_force_preclip.extend(command_norm.tolist())
        first_force.extend(np.minimum(command_norm, 10000.).tolist())
        for anchor, label in enumerate(old_rows):
            distance = float(np.linalg.norm(centers[anchor + 1] - centers[anchor]))
            angle = float(abs(theta[anchor + 1] - theta[anchor]))
            definition_delta = max(definition_delta, abs(distance - label["translation_world_px"]))
            decomposition = motion_bucket(distance, angle)
            row = dict(window_id=f"{entry['trajectory_id']}:{anchor}", trajectory_id=entry["trajectory_id"],
                letter=letter, anchor=anchor, stratum=label["index_stratum"],
                contact="contact" if label["contact"] else "noncontact", phase=label["phase"], intent=label["intent"],
                actual_boundary=label["actual_spatial_class_before"],
                actual_corner="corner" if "-" in label["actual_spatial_class_before"] else "not_corner",
                gap_band=label["gap_band_before"], boundary_behavior=label["boundary_behavior"],
                decomposition=decomposition, likely_wall_blocked_proxy=bool(label["likely_wall_blocked_proxy"]),
                observed_near_static_contact=label["motion_class"] == "near_static" and label["contact"],
                min_wall_gap_world_px=label["min_wall_gap_before"],
                second_perpendicular_wall_gap_world_px=max(min(label["left_gap_before"],label["right_gap_before"]),
                    min(label["top_gap_before"],label["bottom_gap_before"])),
                translation_world_px=distance, translation_image96_px=distance * SCALE,
                physical_CoG_translation_image96_px=float(np.linalg.norm(cog[anchor+1] - cog[anchor])) * SCALE,
                rotation_deg=math.degrees(angle), target_distance_world_px=float(np.linalg.norm(actions[anchor] - states[anchor, :2])),
                reconstructed_first_substep_force_norm=float(min(command_norm[anchor], 10000.)),
                reconstructed_first_substep_preclip_force_norm=float(command_norm[anchor]),
                reconstructed_first_substep_force_saturated=bool(command_norm[anchor] >= 10000.),
                endpoint_pusher_speed_world_px_s=float(velocity_p[anchor+1]),
                endpoint_object_CoG_speed_world_px_s=float(velocity_b[anchor+1]))
            raw.append(row)
            if label["valid_h1_candidate"]:
                h1.append(row)
                lookup[(entry["trajectory_id"], anchor)] = row
            if (3 <= anchor <= len(actions) - 4 and arrays["valid_transition"][anchor-3:anchor+4].all()
                    and all(r["valid_transition"] for r in old_rows[anchor-3:anchor+4])):
                end = anchor + 4
                end_distance = float(np.linalg.norm(centers[end] - centers[anchor]))
                end_angle = float(abs(theta[end] - theta[anchor]))
                record = dict(row, translation_world_px=float(np.linalg.norm(centers[end] - centers[anchor])),
                    translation_image96_px=float(np.linalg.norm(centers[end] - centers[anchor])) * SCALE,
                    physical_CoG_translation_image96_px=float(np.linalg.norm(cog[end] - cog[anchor])) * SCALE,
                    rotation_deg=math.degrees(end_angle),
                    decomposition=motion_bucket(end_distance, end_angle),
                    contact="path_contact" if (arrays["contact"][anchor:anchor+4] > 0).any() else "path_noncontact",
                    endpoint_pusher_speed_world_px_s=float(velocity_p[end]),
                    endpoint_object_CoG_speed_world_px_s=float(velocity_b[end]))
                h4.append(record)
        source_files.append(dict(file=entry["file"], sha256=entry["sha256"], metadata_sha256=entry["metadata_sha256"]))
    expected = dict(moving=14692, intermediate=1017, near_static_contact=1176, near_static_no_contact=7271)
    if len(h1) != 24156 or len(h4) != 22968 or dict(Counter(r["stratum"] for r in h1)) != expected:
        raise ValueError("Independent TRAIN candidate denominator or frozen strata differs")
    micro = []
    for selected in read(BASE / "stage3/frozen-windows.json")["train"]:
        row = lookup[(selected["trajectory_id"], selected["anchor"])]
        micro.append(dict(row, translation_ge1_image_px=row["translation_image96_px"] >= 1,
            rotation_ge3_deg=row["rotation_deg"] >= 3,
            selected_by_rotation_only=row["translation_image96_px"] < 1 and row["rotation_deg"] >= 3))
    fields = ("letter", "stratum", "contact", "actual_boundary", "actual_corner", "gap_band", "phase", "intent",
              "boundary_behavior", "decomposition", "observed_near_static_contact", "likely_wall_blocked_proxy")
    result = dict(kind="r1-independent-saved-TRAIN-amplitude-audit.v1", checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(manifest_path), source_files=source_files,
        method=dict(area_centroid="body origin + R(unwrapped Pymunk angle)*official_shape.area_centroid",
            world_to_image96=SCALE, H1_seconds=.1, H4_seconds=.4,
            H1_legal="a>=3 plus four valid history/current transitions", H4_legal="3<=a<=60 plus seven valid transitions",
            strata_scope="H4 stratum/actual-boundary/gap/phase are the measured anchor's H1 labels; H4 decomposition is endpoint-based and H4 contact means any recorded contact over the four future transitions",
            decomposition="Existing2worldpx/1degree thresholds only; 'threshold_only' does not mean mathematically zero other motion",
            confirmed_wall_blocked="Not identifiable: no object-wall arbiter impulse record; observed near-static contact and wall-blocked proxy are reported separately"),
        official_local_area_centroids=centroid_local, independent_vs_frozen_translation_max_abs_world_px=definition_delta,
        raw_controls=summarize(raw), H1=summarize(h1), H4=summarize(h4),
        largest_H1_translation_events=sorted(h1,key=lambda row:row["translation_image96_px"],reverse=True)[:12],
        H1_strata_counts=dict(Counter(r["stratum"] for r in h1)), H1_motion_decomposition=dict(Counter(r["decomposition"] for r in h1)),
        by_group={horizon: {field: grouped(records, field) for field in fields} for horizon, records in (("H1", h1), ("H4", h4))},
        per_letter_stratum={horizon: {letter: {name: summarize([r for r in records if r["letter"] == letter and r["stratum"] == name])
            for name in STRATA} for letter in sorted(centroid_local)} for horizon, records in (("H1", h1), ("H4", h4))},
        microfit18=micro, microfit18_summary=summarize(micro),
        recorded_control_boundary_dynamics=dict(samples=len(raw_pusher_speed), pusher_speed_world_px_s=stats(raw_pusher_speed, (100, 200)),
            object_physical_CoG_speed_world_px_s=stats(raw_block_speed, (50, 100, 200)), omega_deg_s=stats(raw_omega),
            state_rows_with_all_force_torque_zero=force_zero, state_rows_total=force_total, max_absolute_logged_force_torque=largest_boundary_force,
            force_scope="Control-boundary force/torque is cleared by engine integration, not an applied-force trace; no100Hz position/velocity/force logs were saved",
            first_substep_command_force_reconstructed_from_float32_target=stats(first_force, (9999.999, 10000)),
            first_substep_preclip_force_reconstructed_from_float32_target=stats(first_force_preclip, (10000, 20000)),
            reconstructed_force_scope="Source-formula inference for the first substep only; action float32 quantization and missing9later substeps forbid calling it measured full applied force"),
        proposed_new_physics_controls=0, actual_new_physics_controls=0, model_calls=0, optimizer_updates=0,
        Torch_imported="torch" in sys.modules, environments_created=0, render_calls=0,
        wall_clock_seconds=time.monotonic()-began)
    (OUT / "motion-amplitude-statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("H1", "H4", "H1_strata_counts", "H1_motion_decomposition",
        "recorded_control_boundary_dynamics", "microfit18_summary", "Torch_imported", "wall_clock_seconds")}))


if __name__ == "__main__":
    main()
