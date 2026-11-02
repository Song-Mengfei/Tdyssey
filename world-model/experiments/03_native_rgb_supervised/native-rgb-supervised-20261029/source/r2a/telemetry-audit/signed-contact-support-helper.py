"""Supplement only: signed actual contact support from closed R2A telemetry."""
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
LEVER_EPS = 1e-8  # world pixels; descriptive numerical-zero tolerance.
SPEED_EPS = 1e-8  # world pixels/second; descriptive normal-speed tolerance.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = list(values)
    a = np.asarray(values, dtype=float)
    result = dict(n=len(values))
    if len(a):
        result.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
            **{f"p{q}": float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)})
    return result


def aggregate(rows):
    def support(items):
        return dict(point_samples=len(items), independent_initial_families=len({r["family"] for r in items}),
            families=sorted({r["family"] for r in items}), independent_slots=len({r["slot"] for r in items}),
            slots=sorted({r["slot"] for r in items}), signed_lever_world_px=summary(r["signed_lever"] for r in items),
            closing_speed_world_s=summary(r["closing"] for r in items),
            reconstructed_angle_deg=summary(r["angle"] for r in items if r["angle"] is not None))
    result = support(rows)
    result["signed_lever_classes"] = {name: support([r for r in rows if r["lever_class"] == name])
        for name in ("positive", "negative", "near_zero")}
    result["normal_approach_classes"] = {name: support([r for r in rows if r["approach_class"] == name])
        for name in ("closing", "separating", "normal_stationary")}
    angle_edges = [-180., -135., -90., -45., 0., 45., 90., 135., 180.]
    result["exploratory_angle_bins"] = []
    for i, (lower, upper) in enumerate(zip(angle_edges, angle_edges[1:])):
        selected = [r for r in rows if r["angle"] is not None and lower <= r["angle"]
            and (r["angle"] < upper or (i == len(angle_edges)-2 and r["angle"] <= upper))]
        result["exploratory_angle_bins"].append(dict(lower_deg=lower, upper_deg=upper,
            upper_inclusive=i == len(angle_edges)-2, **support(selected)))
    result["undefined_zero_relative_speed_angles"] = sum(r["angle"] is None for r in rows)
    return result


def main():
    index, ledger = read(BASE/"index.json"), read(BASE/"resource-ledger.json")
    if index.get("collection_closed") is not True or ledger.get("closed") is not True:
        raise RuntimeError("Refusing unfinished collection")
    points, input_hashes = [], []
    normals, pose_errors = [], []
    for slot in index["slots"]:
        if not slot["initialization_attempted"]:
            continue
        metadata = BASE/slot["metadata_file"]
        if sha(metadata) != slot["metadata_sha256"]:
            raise ValueError("Metadata hash mismatch")
        meta = read(metadata)
        path = BASE/meta["telemetry_file"]
        if sha(path) != meta["telemetry_sha256"]:
            raise ValueError("Telemetry hash mismatch")
        identity, request = meta["initial_identity"], meta["request"]
        arm, letter, family = identity["arm"], request["letter"], request["family_id"]
        cog = np.asarray(identity["physics"]["object"]["center_of_gravity"], dtype=float)
        if identity["physics"]["agent"]["moment"] != "infinity":
            raise ValueError("Pusher no-spin premise differs from frozen physics")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            logs = json.load(stream)
        input_hashes.append(dict(slot=slot["slot_id"], metadata_sha256=sha(metadata), telemetry_sha256=sha(path)))
        for control in logs:
            for sub in control["substeps"]:
                entry, post = np.asarray(sub["integrator_pre_state"]), np.asarray(sub["post_state"])
                if entry.shape != (15,) or post.shape != (15,) or not np.isfinite(entry).all() or not np.isfinite(post).all():
                    raise ValueError("Supplement requires finite recorded state; failures must not be dropped")
                c, s = math.cos(post[4]), math.sin(post[4])
                rotation = np.asarray(((c, -s), (s, c)))
                mass_center = post[2:4]+rotation@cog
                for event in sub["contacts"]:
                    if event["kind"] != "agent_object" or event["phase"] != "pre_solve":
                        continue
                    normal = np.asarray(event["normal"], dtype=float)
                    if event["shapes_order"][0] != "object":
                        normal = -normal
                    norm = float(np.linalg.norm(normal))
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError("Invalid contact normal")
                    normals.append(norm)
                    normal /= norm  # Object -> agent, explicitly unit length.
                    inward, tangent = -normal, np.asarray((-normal[1], normal[0]))
                    for point in event["points"]:
                        world, local = np.asarray(point["object_world"]), np.asarray(point["object_local"])
                        lever = world-mass_center
                        signed = float(lever[0]*inward[1]-lever[1]*inward[0])
                        local_world_error = float(np.linalg.norm(post[2:4]+rotation@local-world))
                        pose_errors.append(local_world_error)
                        object_velocity_at_point = entry[7:9]+entry[9]*np.asarray((-lever[1], lever[0]))
                        relative = entry[5:7]-object_velocity_at_point
                        closing, tangential = float(-relative@normal), float(relative@tangent)
                        relative_speed = float(np.linalg.norm(relative))
                        angle = None if relative_speed <= SPEED_EPS else float(math.degrees(math.atan2(tangential, closing)))
                        points.append(dict(arm=arm, letter=letter, family=family, slot=slot["slot_id"],
                            control=int(control["control_index"]), substep=int(sub["index"]),
                            signed_lever=signed,
                            lever_class="positive" if signed > LEVER_EPS else "negative" if signed < -LEVER_EPS else "near_zero",
                            closing=closing, tangential=tangential, relative_speed=relative_speed, angle=angle,
                            approach_class="closing" if closing > SPEED_EPS else "separating" if closing < -SPEED_EPS else "normal_stationary",
                            object_local=point["object_local"], object_world=point["object_world"], arc_ids=point["arc_ids"]))
    arms = {arm: aggregate([r for r in points if r["arm"] == arm]) for arm in ("P0", "P1")}
    letters = sorted({row["letter"] for row in index["slots"]})
    per_letter = {arm: {letter: aggregate([r for r in points if r["arm"] == arm and r["letter"] == letter])
        for letter in letters} for arm in ("P0", "P1")}
    result = dict(kind="R2A-supplement-signed-real-contact-support.v1", collection_closed=True,
        index_sha256=sha(BASE/"index.json"), ledger_sha256=sha(BASE/"resource-ledger.json"),
        helper_sha256=sha(__file__), input_hashes=input_hashes,
        method=dict(lever="cross(r,-n), r=actual object contact world point minus physical mass CoG world; n=unit object-to-agent callback normal",
            sign="numeric 2D cross x1*y2-y1*x2; unit inward normal force moment arm, not measured net torque or a planned direction label",
            lever_near_zero_epsilon_world_px=LEVER_EPS,
            approach="source-ordered reconstruction: integrator-entry COM velocities/omega with actual callback contact point and same-substep post pose; no direct callback-local velocity measurement",
            approach_closing="-(v_agent-v_object_at_contact) dot n; positive closes, negative separates",
            approach_normal_stationary="abs(closing)<=epsilon; tangential motion may still be nonzero, so this is not total relative rest",
            approach_epsilon_world_s=SPEED_EPS,
            angle="atan2(tangential_relative_speed,closing_speed), continuous signed degrees; undefined for near-zero total relative speed",
            exploratory_bins="45-degree half-open bins [-180,180], final upper endpoint inclusive; descriptive after collection, not preregistered acceptance or planned labels",
            denominators="all actual pre_solve agent-object point samples including zero-impulse contacts; correlated within trajectory; independent families and slots listed separately",
            limitations="one normal sign is not net torque causation, sustained push quality, ideal protocol selection, or training/shape evidence",
            new_physics_controls=0, model_calls=0, new_renders=0),
        recorded_normal_norm=summary(normals), callback_point_pose_reconstruction_error_world_px=summary(pose_errors),
        per_arm=arms, per_arm_per_letter=per_letter)
    target = Path(__file__).with_name("signed-contact-support.json")
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({arm: {"points": rows["point_samples"],
        "signed": {name: group["point_samples"] for name, group in rows["signed_lever_classes"].items()},
        "approach": {name: group["point_samples"] for name, group in rows["normal_approach_classes"].items()}}
        for arm, rows in arms.items()}))


if __name__ == "__main__":
    main()
