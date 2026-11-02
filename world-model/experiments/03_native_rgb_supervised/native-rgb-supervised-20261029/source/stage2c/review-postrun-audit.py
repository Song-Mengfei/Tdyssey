"""Manual, artifact-only independent 2C audit; no environment/model/iterator.

Run raw only after root confirms collection and physical audit completion.
Run final only after root confirms that reports, indices, joint view and CPU
sample evidence are closed. This program never polls or waits for those files.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def require_cold(base):
    stage = base / "stage2c"
    if (stage / "COLLECTION_RUNNING.lock").exists():
        raise RuntimeError("Root collection is active; no hot validation reads")
    if not all((stage / p).is_file() for p in ("data/manifest.json", "physics-audit.json")):
        raise RuntimeError("Root must finish the final manifest and physical audit first")
    return stage


def protected_files(stage):
    problems = []
    rows = read(stage / "review-protected-train-files.json")["files"]
    for row in rows:
        path = Path(row["source"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            problems.append(row["source"])
    return len(rows), problems


def footprint(base):
    repo = base.parents[1]
    sources = set()
    for pattern in ("tdyssey/native_rgb*.py", "scripts/native_rgb*.py", "tests/test_native_rgb*.py",
                    "configs/native_rgb*.json", "run_native_rgb*", "requirements/native-rgb*", "docs/NATIVE_RGB_WORLD_MODEL.md"):
        sources.update(p for p in repo.glob(pattern) if p.is_file())
    result = dict(old_native_bytes=sum(p.stat().st_size for p in (repo / "artifacts/native-rgb-20261028").rglob("*") if p.is_file()),
                  supervised_bytes=sum(p.stat().st_size for p in base.rglob("*") if p.is_file()),
                  native_source_bytes=sum(p.stat().st_size for p in sources), cap_bytes=3221225472)
    result["combined_bytes"] = sum(result[k] for k in ("old_native_bytes", "supervised_bytes", "native_source_bytes"))
    return result


def audit_raw(base):
    stage = require_cold(base)
    import numpy as np
    from shapely.geometry import Point
    from tdyssey.envs.official import official_shape
    from tdyssey.native_rgb_env import shape_geometry
    from tdyssey.native_rgb_data import contour_segments
    from tdyssey.native_rgb_supervised_sampling import plan_scene, LOCATIONS, SIDE_SIGN_INDEX

    started = time.monotonic()
    errors = []

    def check(condition, name):
        if not condition:
            errors.append(name)
        return bool(condition)

    manifest = read(stage / "data/manifest.json")
    ledger = read(stage / "resource-ledger.json")
    plan = read(stage / "frozen-requests.json")
    dedup = read(stage / "dedup-reference.json")
    cfg = read(stage / "source-snapshot/frozen-config.json")
    baseline = read(stage / "counter-baseline.json")
    letters = cfg["letters"]
    bins = {l: contour_segments(l, 8) for l in letters}
    shapes = {l: shape_geometry(l) for l in letters}
    rings = {l: [shapes[l].exterior, *shapes[l].interiors] for l in letters}
    hit_counts = {l: {int(z["id"]): 0 for z in bins[l]} for l in letters}
    attempts = {l: Counter() for l in letters}
    trajectory_hits = defaultdict(Counter)
    prior = read(stage / "source-snapshot/frozen-oldTRAIN-prior.json")["segments"]
    historical = {l: {int(z["id"]): len(z["trajectory_ids"]) for z in prior[l]} for l in letters}
    check(all({int(k): int(v) for k, v in baseline["actual_hit_counts"][l].items()} == hit_counts[l]
              and baseline["planned_attempts"][l] == {} for l in letters), "zero_validation_baseline")
    check(not baseline["TRAIN_dynamic_counts_inherited"] and not baseline["validation_counters_may_flow_back_to_TRAIN"], "counter_isolation")
    check(baseline["old_TRAIN_prior_sha256"] == sha(stage / "source-snapshot/frozen-oldTRAIN-prior.json"), "fixed_oldTRAIN_prior")
    source_rows = []
    for z in read(stage / "source-identity.json")["files"]:
        source, snapshot = Path(z["source"]), stage / z["snapshot"]
        ok = source.stat().st_size == snapshot.stat().st_size == z["bytes"] and sha(source) == sha(snapshot) == z["sha256"]
        source_rows.append(dict(source=str(source), sha256=z["sha256"], matches=check(ok, "source:" + str(source))))
    check(len(source_rows) == 37, "source37")
    check(sha(stage / "source-snapshot/frozen-index-spec.json") == manifest["index_spec_sha256"] == plan["index_spec_sha256"]
          == "f55fad565e6cc013e165529f81f84cf1ef46a67985b2d3f1bdae391545f6c6a1", "prephysics_index_rules")
    count, changed = protected_files(stage)
    check(count == 1027 and not changed, "all1027_oldTRAIN_files")
    seen_hash = set(dedup["initial_state_hashes"])
    seen_seed = set(dedup["historical_seeds"])
    lookup = {z["request_id"]: z for z in plan["requests"]}
    ledger_rows = {z["trajectory_id"]: z for z in ledger["trajectories"]}
    per_letter = {l: Counter() for l in letters}
    slots = []
    maxima = np.zeros(3)
    raw_count = h1_count = h4_count = invalid_count = 0

    def mapped(letter, points):
        result = set()
        for xy in points:
            point = Point(float(xy[0]), float(xy[1]))
            ri = min(range(len(rings[letter])), key=lambda i: rings[letter][i].distance(point))
            ring = rings[letter][ri]
            if ring.distance(point) > 2:
                continue
            arc = ring.project(point)
            for row in bins[letter]:
                if row["ring"] == ri and row["arc_start"] <= arc <= row["arc_end"] + 1e-6:
                    result.add(int(row["id"]))
                    break
        return result

    for entry in manifest["entries"]:
        before_error = len(errors)
        tid, letter = entry["trajectory_id"], entry["shape"]
        q, lr = lookup[tid], ledger_rows[tid]
        path, mp = stage / "data" / entry["file"], stage / "data" / entry["metadata_file"]
        check(sha(path) == entry["sha256"] and sha(mp) == entry["metadata_sha256"], tid + ":npz_meta_hash")
        meta = read(mp)
        scene, inp = meta["scene"], meta["selection_inputs"]
        check(meta["request"] == q and entry["split"] == "seen_val" and entry["stage"] == "2C", tid + ":request_split")
        check(inp["counter_split"] == "seen_val" and "formal_train_contact_controls_before" not in inp, tid + ":own_split_scope")
        as_int = lambda d: {int(k): int(v) for k, v in d.items()}
        check(as_int(inp["seen_val_contact_controls_before"]) == hit_counts[letter], tid + ":actual_hit_prefix")
        check(as_int(inp["planned_initial_attempts_before"]) == dict(attempts[letter]), tid + ":planned_prefix")
        check(as_int(inp["historical_TRAIN_independent_trajectories"]) == historical[letter], tid + ":prior_prefix")
        replay = plan_scene(letter, q["spatial_class"], q["seed"], gap_band=q["gap_band"], gap_world_px=q["gap_world_px"],
                            sampling=q["sampling"], coverage_counts=hit_counts[letter], historical_counts=historical[letter],
                            planned_attempts=dict(attempts[letter]))
        check(replay["selection_evidence"] == scene["selection_evidence"] and replay["staging_segment"] ==
              entry["selected_initial_arc"] == scene["staging_segment"], tid + ":full_priority_choice_replay")
        check(replay["selection_version"] == scene["selection_version"] == "motion-room-planned-attempts-v2", tid + ":selector")
        with np.load(path, allow_pickle=False) as z:
            s, action, sub, pen = z["states"], z["actions"], z["substep_penetrations"], z["penetrations"]
            points, offsets = z["contact_points"], z["contact_offsets"]
            n = len(action)
            check(s.shape == (n + 1, 15) and action.shape == (n, 2) and sub.shape == (n, 10, 3) and pen.shape == (n + 1, 3), tid + ":timeline")
            check(s.dtype == sub.dtype == pen.dtype == np.float64, tid + ":float64")
            check(all(np.isfinite(z[k]).all() for k in ("states", "actions", "substep_penetrations", "penetrations",
                                                         "contact_points", "contact_angles")), tid + ":finite")
            check(not (action < 0).any() and not (action > 511).any(), tid + ":absolute_action")
            check(np.array_equal(sub.max(axis=1), pen[1:]), tid + ":ten_substep_maxima")
            valid = np.isfinite(sub).all(axis=(1, 2)) & (sub.max(axis=(1, 2)) <= 2) & np.isfinite(s[1:]).all(axis=1)
            check(np.array_equal(valid, z["valid_transition"]), tid + ":valid_flags")
            invalid_count += int((~valid).sum())
            digest = hashlib.sha256(s[0].tobytes()).hexdigest()
            check(digest == entry["initial_scene_hash"] and digest not in seen_hash and entry["seed"] not in seen_seed, tid + ":all_history_identity_disjoint")
            seen_hash.add(digest)
            seen_seed.add(entry["seed"])
            check(hashlib.sha256(np.asarray(entry["initial_scene"], np.float64).tobytes()).hexdigest() == digest
                  and hashlib.sha256(np.asarray(scene["initial_state"], np.float64).tobytes()).hexdigest() == digest, tid + ":initial_full_bytes")
            check(np.allclose(s[0, 2:4], replay["object_origin"], rtol=0, atol=1e-10)
                  and np.allclose(s[0, :2], replay["pusher_start"], rtol=0, atol=1e-10)
                  and s[0, 4] == replay["object_angle"] and np.all(s[0, 5:] == 0), tid + ":initial_pose")
            check(offsets.shape == (n + 1,) and offsets[0] == 0 and offsets[-1] == len(points)
                  and not (np.diff(offsets) < 0).any() and len(z["contact_angles"]) == len(points), tid + ":contacts")
            entered = n > 0 or bool((meta.get("exception_state") or {}).get("step_entered") and lr["charged_controls"])
            check(inp["planned_attempt_counted"] == entered, tid + ":actual_start")
            if entered:
                attempts[letter][int(scene["staging_segment"])] += 1
                check(as_int(inp["planned_initial_attempts_after_start"]) == dict(attempts[letter])
                      and lr.get("initial_arc_attempt_counted") == scene["staging_segment"], tid + ":one_start_increment")
            local = np.asarray(official_shape(letter).centroid)
            a = s[:, 4]
            center = s[:, 2:4] + np.column_stack((np.cos(a) * local[0] - np.sin(a) * local[1], np.sin(a) * local[0] + np.cos(a) * local[1]))
            delta, rot = np.linalg.norm(np.diff(center, axis=0), axis=1), np.abs(np.diff(a))
            moving, static = (delta >= 2) | (rot >= math.pi / 180), (delta <= .5) & (rot <= .005)
            check(len(meta["steps"]) == n and meta["hidden_physics_controls"] == 0, tid + ":continuous_history")
            for t in range(n):
                step = meta["steps"][t]
                sign = 1 if (LOCATIONS.index(entry["location"]) + SIDE_SIGN_INDEX[letter]) % 2 else -1
                check(step["step"] == t and not step["midtrajectory_reset"] and step["approach_side_sign"] == sign, tid + f":continuous:{t}")
                for arc in mapped(letter, points[offsets[t]:offsets[t + 1]]):
                    hit_counts[letter][arc] += 1
                    trajectory_hits[letter, arc][tid] += 1
                row = per_letter[letter]
                row["raw_controls"] += 1
                row["raw_moving"] += int(moving[t])
                row["raw_near_static"] += int(static[t])
                if t >= 3 and valid[t - 3:t + 1].all():
                    row["h1"] += 1
                    row["moving"] += int(moving[t])
                    row["near_static"] += int(static[t])
                    row["intermediate"] += int(not moving[t] and not static[t])
            raw_count += n
            h1_count += int(sum(valid[t - 3:t + 1].all() for t in range(3, n)))
            h4_count += int(sum(valid[t - 3:t + 4].all() for t in range(3, n - 3)))
            if n:
                maxima = np.maximum(maxima, sub.max(axis=(0, 1)))
            check(n == entry["controls"] == lr["completed_controls"] == lr["charged_controls"] == 64
                  and lr["status"] == "completed" and lr["pending_control"] is None, tid + ":ledger")
            slots.append(dict(trajectory_id=tid, selected_arc=scene["staging_segment"], controls=n, issues=errors[before_error:]))
    final_hits = read(stage / "contact-coverage.json")["segments"]
    final_attempts = read(stage / "planned-attempts-current.json")["counts"]
    for l in letters:
        check(dict(attempts[l]) == {int(k): int(v) for k, v in final_attempts[l].items()}, "final_attempts:" + l)
        for z in final_hits[l]:
            arc = int(z["id"])
            check(hit_counts[l][arc] == z["contact_control_count"] and dict(trajectory_hits[l, arc]) == z["trajectory_contact_controls"],
                  f"final_seen_val_hits:{l}:{arc}")
    check(len(slots) == len(ledger["trajectories"]) == 162 and raw_count == 10368 and h1_count == 9882 and h4_count == 9396
          and invalid_count == 0, "fixed_denominators")
    physical = read(stage / "physics-audit.json")
    check(physical["passed"] and physical["physics_collection_gate_passed"] and physical["manifest_sha256"] == sha(stage / "data/manifest.json")
          and physical["controls"] == 10368 and physical["audited_substeps"] == 103680 and physical["invalid_controls"] == 0
          and physical["valid_h1_candidates"] == h1_count and physical["valid_h4_candidates"] == h4_count
          and np.array_equal(np.asarray(physical["max_substep_penetration_world_px"]), maxima), "root_physics_alignment")
    check(ledger["charged_controls"] == ledger["completed_controls"] == 10368 and ledger["physics_substeps"] == 103680
          and ledger["recorded_partial_substeps_lower_bound"] == 0 and ledger["supervised_round_controls"] == 39680
          and ledger["historical_plus_new_native_controls"] == 74240 and not ledger["failure"], "costs")
    check(all(ledger[k] == 0 for k in ("model_inference_calls", "optimizer_attempts", "unseen_model_inference_calls", "server_jobs")), "no_model_optimizer_server")
    for l, z in per_letter.items():
        z["diagnostic_moving_fraction"] = z["moving"] / z["h1"] if z["h1"] else None
        z["diagnostic_near_static_fraction"] = z["near_static"] / z["h1"] if z["h1"] else None
    disk = footprint(base)
    check(disk["combined_bytes"] < disk["cap_bytes"], "3GiB")
    return dict(kind="stage2c-independent-raw-review.v1", checked_at_utc=datetime.now(timezone.utc).isoformat(), audit_passed=not errors,
                issues=errors, raw_controls=raw_count, valid_h1=h1_count, valid_h4=h4_count, invalid_controls=invalid_count,
                source_rows=source_rows, protected_train_files=count, changed_train_files=changed, per_letter=per_letter,
                manifest_sha256=sha(stage / "data/manifest.json"), physical_audit_sha256=sha(stage / "physics-audit.json"),
                source_identity_sha256=sha(stage / "source-identity.json"), full_priority_prefix_replays=len(slots),
                max_substep_penetrations=maxima.tolist(), slots=slots, footprint=disk, wall_seconds=time.monotonic() - started,
                reviewer_execution=dict(environment_constructions=0, physical_steps=0, model_calls=0, optimizer_updates=0, training_iterator_calls=0),
                diagnostic_targets_do_not_gate_validation=True)


def independent_catalog(base):
    """Truth-only saved arrays. Frozen proxy helper is used only for its proxy."""
    import numpy as np
    from tdyssey.native_rgb_motion_statistics import transition_rows, area_centroids, geometric_wall_gaps
    stage = base / "stage2c"
    manifest = read(stage / "data/manifest.json")
    candidates, trajectories = {}, {}
    for e in manifest["entries"]:
        tid = e["trajectory_id"]
        meta = read(stage / "data" / e["metadata_file"])
        with np.load(stage / "data" / e["file"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        rows = transition_rows(e, arrays, meta["steps"])
        centers = area_centroids(e["shape"], arrays["states"])
        gaps = geometric_wall_gaps(e["shape"], arrays["states"])
        trajectories[tid] = e
        for a in range(3, len(rows) - 3):
            if not all(rows[t]["valid_transition"] for t in range(a - 3, a + 4)):
                continue
            h1, future = rows[a], rows[a:a + 4]
            delta = float(np.linalg.norm(centers[a + 4] - centers[a]))
            rotation = abs(float(arrays["states"][a + 4, 4] - arrays["states"][a, 4]))
            moving = h1["motion_class"] == "moving"
            static = h1["motion_class"] == "near_static"
            conditions = dict(moving=moving, rotation=h1["rotation_abs_deg"] >= 1,
                              boundary_moving=moving and min(h1["min_wall_gap_before"], h1["min_wall_gap_after"]) <= 32,
                              likely_wall_blocked_proxy=h1["likely_wall_blocked_proxy"],
                              near_static_contact=static and h1["contact"] is True,
                              near_static_no_contact=static and h1["contact"] is False,
                              intermediate=h1["motion_class"] == "intermediate")
            candidates[f"{tid}:{a}"] = dict(window_id=f"{tid}:{a}", trajectory_id=tid, letter=e["shape"], anchor=a,
                conditions=conditions, h1=h1, h4=dict(endpoint_translation_world_px=delta, endpoint_rotation_abs_rad=rotation,
                    endpoint_moving=delta >= 2 or rotation >= math.pi / 180, endpoint_near_static=delta <= .5 and rotation <= .005,
                    path_any_step_moving=any(r["motion_class"] == "moving" for r in future),
                    path_all_steps_near_static=all(r["motion_class"] == "near_static" for r in future),
                    path_translation_sum_world_px=sum(r["translation_world_px"] for r in future),
                    path_rotation_abs_sum_rad=sum(r["rotation_abs_rad"] for r in future),
                    path_min_wall_gap_world_px=float(gaps[a:a + 5].min()),
                    path_actual_boundary_le32=bool(gaps[a:a + 5].min() <= 32)))
    return manifest, trajectories, candidates


def audit_final(base, cpu_path):
    stage = require_cold(base)
    raw = read(stage / "review-raw.json")
    paths = [stage / "report/indices/indices.json", stage / "report/seen-val-report.json", stage / "joint-data/manifest.json",
             stage / "joint-data/physics-audit.json", Path(cpu_path)]
    if not all(p.is_file() for p in paths):
        raise RuntimeError("Root must close indices/report/joint/CPU evidence before final audit")
    errors = []

    def check(condition, name):
        if not condition:
            errors.append(name)
        return bool(condition)

    check(raw["audit_passed"] and raw["manifest_sha256"] == sha(stage / "data/manifest.json"), "raw_audit_current")
    check(raw["source_identity_sha256"] == sha(stage / "source-identity.json"), "raw_source_identity_current")
    for row in read(stage / "source-identity.json")["files"]:
        check(sha(row["source"]) == row["sha256"] == sha(stage / row["snapshot"]), "final_source:" + row["source"])
    policy = read(stage / "source-snapshot/frozen-index-spec.json")
    indices = read(paths[0])
    manifest, trajectories, candidates = independent_catalog(base)
    check(len(candidates) == 9396, "all9396_H4legal_candidates")
    order = policy["extra"]["category_tie_order"]
    quota = policy["extra"]["quotas"]
    grouped = defaultdict(list)
    for w in candidates.values():
        grouped[w["trajectory_id"]].append(w)

    def rank(w, domain):
        e = trajectories[w["trajectory_id"]]
        payload = [policy["algorithm_version"], policy["algorithm_seed"], domain, "seen_val", w["letter"],
                   w["trajectory_id"], e["sha256"], e["metadata_sha256"], w["anchor"]]
        value = hashlib.sha256(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return value, w["trajectory_id"], w["anchor"]

    natural, extra, dev, missing = [], [], [], []
    used, loads = set(), Counter()
    for q in read(stage / "frozen-requests.json")["requests"]:
        choices = grouped[q["request_id"]]
        if not choices:
            missing.append(("natural", q["request_id"]))
            continue
        w = min(choices, key=lambda z: rank(z, "natural"))
        natural.append(w)
        used.add(w["window_id"])
        loads[w["trajectory_id"]] += 1
    expected_support, expected_roles = {}, []
    for letter in tuple(policy["letters"]):
        pool = [w for w in candidates.values() if w["letter"] == letter and w["window_id"] not in used]
        support = {c: [w for w in pool if w["conditions"][c]] for c in order}
        rarity = sorted(order, key=lambda c: (len(support[c]), order.index(c)))
        expected_support[letter] = dict(order=rarity, counts={c: len(support[c]) for c in order})
        for category in rarity:
            for slot in range(quota[category]):
                available = [w for w in support[category] if w["window_id"] not in used]
                if not available:
                    missing.append(("extra", letter, category, slot + 1))
                    continue
                w = min(available, key=lambda z: (loads[z["trajectory_id"]], *rank(z, "extra")))
                expected_roles.append((category, slot + 1, len(support[category]), loads[w["trajectory_id"]]))
                extra.append(w)
                used.add(w["window_id"])
                loads[w["trajectory_id"]] += 1
    for letter in tuple(policy["letters"]):
        for role, values, domain in (("natural", [w for w in natural if w["letter"] == letter], "development-natural"),
                                     ("extra_moving", [w for w in extra if w["letter"] == letter and w["conditions"]["moving"]], "development-extra-moving")):
            if values:
                dev.append((min(values, key=lambda z: rank(z, domain)), role, domain))
            else:
                missing.append(("development", letter, role))
    ids = lambda values: [w["window_id"] for w in values]
    check(ids(indices["natural"]) == ids(natural), "natural_uniform_hash_replay")
    check(ids(indices["extra"]) == ids(extra), "rare_once_leastload_extra_replay")
    check(ids(indices["development"]) == [w["window_id"] for w, _, _ in dev], "development36_replay")
    check(ids(indices["combined"]) == ids(natural + extra) and len(set(ids(indices["combined"]))) == len(indices["combined"]), "unique_combined_once")
    check(len(natural) == 162 and 162 <= len(natural + extra) <= 324 and len(dev) <= 36, "panel_caps")
    for letter, support in expected_support.items():
        actual = indices["extra_support_after_natural"][letter]
        check(actual["order"] == support["order"] and actual["counts"] == support["counts"], "rare_support:" + letter)
    for w, expected in zip(indices["extra"], expected_roles):
        reason = w["selection_reason"]
        check((reason["category"], reason["slot"], reason["initial_support"], reason["selected_trajectory_load_before"]) == expected,
              "extra_reason:" + w["window_id"])
    actual_missing = []
    for row in indices["missing_slots"]:
        if row["panel"] == "natural":
            actual_missing.append(("natural", row["trajectory_id"]))
        elif row["panel"] == "extra":
            actual_missing.append(("extra", row["letter"], row["category"], row["slot"]))
        else:
            actual_missing.append(("development", row["letter"], row["role"]))
    check(actual_missing == missing, "missing_slots_no_substitution")
    for w in indices["combined"]:
        expected, e = candidates[w["window_id"]], trajectories[w["trajectory_id"]]
        check(w["split"] == "seen_val" and 3 <= w["anchor"] <= 60 and w["conditions"] == expected["conditions"], "conditions:" + w["window_id"])
        check(w["source"]["npz_sha256"] == e["sha256"] and w["source"]["metadata_sha256"] == e["metadata_sha256"], "index_source:" + w["window_id"])
        check(all(w["h1"][k] == expected["h1"][k] for k in ("motion_class", "contact", "translation_world_px", "rotation_abs_rad",
                  "min_wall_gap_before", "min_wall_gap_after", "likely_wall_blocked_proxy")), "H1truth:" + w["window_id"])
        check(all(w["h4"][k] == v for k, v in expected["h4"].items()), "H4truth:" + w["window_id"])
    check(indices["policy_sha256"] == manifest["index_spec_sha256"] == sha(stage / "source-snapshot/frozen-index-spec.json"), "prephysics_policy_binding")

    def overlaps(windows):
        groups = defaultdict(list)
        counts = Counter()
        for w in windows:
            groups[w["trajectory_id"]].append(w["anchor"])
        for values in groups.values():
            values.sort()
            for i, a in enumerate(values):
                for b in values[i + 1:]:
                    d = b - a
                    counts.update(dict(H4_state_overlap=int(d <= 7), H4_transition_overlap=int(d <= 6),
                                       H4_future_transition_overlap=int(d <= 3), H1_state_overlap=int(d <= 4), same_trajectory_pairs=1))
        return dict(counts)

    check(indices["temporal_overlap"]["combined"]["counts"] == overlaps(natural + extra), "combined_temporal_overlap")
    check(indices["temporal_overlap"]["development"]["counts"] == overlaps([w for w, _, _ in dev]), "development_temporal_overlap")

    joint, ja = read(paths[2]), read(paths[3])
    check(joint["reference_only"] and joint["npz_files_copied"] == 0 and not joint["training_consumption_authorized"]
          and not list((stage / "joint-data").rglob("*.npz")), "zero_copy_joint")
    check(ja["physics_collection_gate_passed"] and ja["dataset_manifest_sha256"] == sha(paths[2]) and ja["split_isolation_verified"], "joint_gate")
    check(Path(joint["base_root"]).resolve() == base and joint["controller"] == "finite-force-pd-v1"
          and joint["protocol_sha256"] == manifest["protocol_sha256"], "joint_root_protocol")
    original, seen_ids, seen_seeds, seen_scenes = {}, set(), set(), set()
    for cohort in ("2A", "2B", "2C"):
        m = read(base / f"stage{cohort.lower()}/data/manifest.json")
        source = joint["reference_sources"][cohort]
        check(source["manifest_sha256"] == sha(base / source["manifest_file"])
              and source["audit_sha256"] == sha(base / source["audit_file"])
              and source["source_identity_sha256"] == m["source_identity_sha256"] == sha(base / source["source_identity_file"]), "joint_source:" + cohort)
        for e in m["entries"]:
            original[cohort, e["trajectory_id"]] = e
    joint_keys = set()
    for e in joint["entries"]:
        key = e["cohort"], e["trajectory_id"]
        check(key in original and key not in joint_keys, "joint_unique_reference:" + str(key))
        old = original[key]
        joint_keys.add(key)
        check(e["split"] == ("seen_val" if key[0] == "2C" else "train") and e["stage"] == key[0], "joint_split:" + str(key))
        check(all(e[k] == old[k] for k in ("shape", "controls", "seed", "initial_scene_hash")), "joint_original_identity:" + str(key))
        for field, hfield in (("file", "sha256"), ("metadata_file", "metadata_sha256")):
            prefix = f"stage{key[0].lower()}/data"
            target = (base / e[field]).resolve()
            check(target.is_relative_to((base / prefix).resolve()) and e[field] == (Path(prefix) / old[field]).as_posix()
                  and e[hfield] == old[hfield] == sha(target), "joint_hash:" + str(key) + ":" + field)
        for field, seen in (("trajectory_id", seen_ids), ("seed", seen_seeds), ("initial_scene_hash", seen_scenes)):
            check(e[field] not in seen and e[field] == old[field], "joint_disjoint:" + field + ":" + str(key))
            seen.add(e[field])
    check(joint_keys == set(original) and len(joint_keys) == 558 and len(joint["splits"]["train"]) == 396
          and len(joint["splits"]["seen_val"]) == 162 and not joint["splits"]["unseen_test"], "joint558_split_quotas")

    cpu = read(cpu_path)
    check(all(cpu[k] == 0 for k in ("physics_controls", "model_calls", "optimizer_updates", "index_draws"))
          and cpu["network_inputs_only"] == ["image", "action"], "CPU_no_extra_inputs_or_training")
    import numpy as np
    for sample in cpu["samples"]:
        e = next(e for e in joint["entries"] if e["trajectory_id"] == sample["trajectory_id"])
        a = sample["anchor"]
        with np.load(base / e["file"], allow_pickle=False) as z:
            actions = z["actions"][a - 3:a + 1].astype(np.float32) / 511 * 2 - 1
        check(sample["history_state_indices"] == list(range(a - 3, a + 1)) and sample["history_action_indices"] == list(range(a - 3, a))
              and sample["current_action_index"] == a and sample["target_state_index"] == a + 1
              and sample["action_sha256"] == hashlib.sha256(actions.tobytes()).hexdigest(), "CPU_timeline_normalization:" + sample["trajectory_id"])
    for batch in cpu["batches"].values():
        check(batch["image"][1:] == [5, 3, 96, 96] and batch["action"][1:] == [4, 2], "CPU_batch_shapes")
    count, changed = protected_files(stage)
    check(count == 1027 and not changed, "all_oldTRAIN_after_reference_and_indices")
    report = read(paths[1])
    check(report["recorded_slots"] == 162 and report["recorded_controls"] == 10368 and report["horizons"]["1"]["valid_candidates"] == 9882
          and report["horizons"]["4"]["valid_candidates"] == 9396 and report["distribution_targets_are_diagnostics_not_validation_acceptance_gates"], "report_denominators_diagnostic_scope")
    for letter, z in raw["per_letter"].items():
        actual = report["per_letter"][letter]["valid_h1"]
        check(actual["total"] == z["h1"] and all(actual["class_counts"].get(k, 0) == z[k] for k in ("moving", "near_static", "intermediate")), "report_H1:" + letter)
    disk = footprint(base)
    check(disk["combined_bytes"] < disk["cap_bytes"], "final3GiB")
    return dict(kind="stage2c-independent-final-review.v1", checked_at_utc=datetime.now(timezone.utc).isoformat(), audit_passed=not errors,
                issues=errors, raw_review_sha256=sha(stage / "review-raw.json"), indices_sha256=sha(paths[0]), joint_manifest_sha256=sha(paths[2]),
                CPU_evidence_sha256=sha(cpu_path), panel_counts=dict(natural=len(natural), extra=len(extra), combined=len(natural + extra), development=len(dev)),
                missing_slots=len(missing), natural_anchor_histogram=dict(Counter(w["anchor"] for w in natural)),
                temporal_overlap=dict(combined=overlaps(natural + extra), development=overlaps([w for w, _, _ in dev])),
                raw_controls=raw["raw_controls"], valid_h1=raw["valid_h1"], valid_h4=raw["valid_h4"], protected_train_files=count,
                joint_references=len(joint_keys), footprint=disk, reviewer_execution=dict(environment_constructions=0, physical_steps=0,
                    model_calls=0, optimizer_updates=0, training_iterator_calls=0),
                qualification="Seen-letter development validation; no real-world frequency or unbiased final-test claim. No stage3 approval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("raw", "final"))
    parser.add_argument("--after-root-completion", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cpu-evidence", type=Path)
    parser.add_argument("--output-stem")
    args = parser.parse_args()
    base = args.root.resolve()
    stage = require_cold(base)
    sys.path.insert(0, str(base.parents[1]))
    stem = args.output_stem or ("review-raw" if args.phase == "raw" else "review-postrun")
    if not stem.startswith("review-") or Path(stem).name != stem:
        raise ValueError("Only owned review-* artifact outputs are permitted")
    output = stage / (stem + ".json")
    if output.exists():
        raise FileExistsError("Preserve immutable review evidence; choose another review-* output stem")
    if args.phase == "final" and args.cpu_evidence is None:
        parser.error("final requires the root-confirmed closed --cpu-evidence file")
    result = audit_raw(base) if args.phase == "raw" else audit_final(base, args.cpu_evidence)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    (stage / (stem + ".md")).write_text("# Independent 2C " + args.phase + " review\n\nAudit passed: **" + str(result["audit_passed"])
        + "**. Issues: " + str(result["issues"]) + ".\n\nZero environment constructions, physics controls, model calls, optimizer updates or training-iterator calls. "
        "Validation distribution targets remain diagnostics; stage 3 is not authorized.\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in ("audit_passed", "issues", "panel_counts", "missing_slots", "raw_controls", "valid_h1", "valid_h4")}, indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
