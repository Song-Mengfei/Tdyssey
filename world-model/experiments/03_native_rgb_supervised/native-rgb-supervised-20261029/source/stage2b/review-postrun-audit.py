"""Independent saved-file audit. Run manually only after root confirms completion.

This artifact never constructs an environment, advances physics, loads a model,
draws training indices, changes source/data, waits for a job, or starts a job.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def audit(supervised_root, merged_path):
    base = Path(supervised_root).resolve()
    stage = base / "stage2b"
    if (stage / "COLLECTION_RUNNING.lock").exists():
        raise RuntimeError("Collection is active; do not read hot data")
    if not (stage / "data/manifest.json").is_file() or not Path(merged_path).is_file() or not (stage / "physics-audit.json").is_file():
        raise RuntimeError("Root must finish the final manifest and merged view first")

    # Imports below the manual/lock guards are geometry/array helpers only.
    import ast
    import math
    import numpy as np
    from shapely import affinity
    from shapely.geometry import Point
    from tdyssey.envs.official import official_shape
    from tdyssey.native_rgb_data import contour_segments
    from tdyssey.native_rgb_env import shape_geometry
    from tdyssey.native_rgb_supervised_sampling import LOCATIONS, SIDE_SIGN_INDEX, plan_scene

    started = time.monotonic()
    repo = base.parents[1]
    issues = []

    def check(condition, label):
        if not condition:
            issues.append(label)
        return bool(condition)

    manifests = {c: read_json(base / f"stage{c.lower()}/data/manifest.json") for c in ("2A", "2B")}
    plans = {c: read_json(base / f"stage{c.lower()}/frozen-requests.json") for c in ("2A", "2B")}
    ledger = read_json(stage / "resource-ledger.json")
    dedup = read_json(stage / "dedup-reference.json")
    cfg = read_json(stage / "source-snapshot/frozen-config.json")
    letters = cfg["letters"]
    source_rows = []
    for row in read_json(stage / "source-identity.json")["files"]:
        source, snapshot = Path(row["source"]), stage / row["snapshot"]
        ok = (source.is_file() and snapshot.is_file()
              and source.stat().st_size == snapshot.stat().st_size == row["bytes"]
              and digest(source) == digest(snapshot) == row["sha256"])
        source_rows.append(dict(source=str(source), sha256=row["sha256"], matches=check(ok, "source:" + str(source))))
    check(len(source_rows) == 32, "source_row_count")

    preserved2a = read_json(stage / "preserved2A-files.json")["files"]
    for row in preserved2a:
        path = base / "stage2a" / row["file"]
        check(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "changed2A:" + row["file"])
    check(len(preserved2a) == 417, "preserved2A_count")
    old_protected = read_json(base / "stage2a/baseline-preservation.json")["stage1_and_stage1b_protected_files"]
    for row in old_protected:
        check(Path(row["path"]).stat().st_size == row["bytes"] and digest(row["path"]) == row["sha256"],
              "changed_old_supervised:" + row["path"])

    prior = read_json(stage / "source-snapshot/historical-TRAIN-contact-coverage.json")["segments"]
    historical = {l: {int(z["id"]): len(z["trajectory_ids"]) for z in prior[l]} for l in letters}
    segments = {l: contour_segments(l, 8) for l in letters}
    geometry = {l: shape_geometry(l) for l in letters}
    rings = {l: [geometry[l].exterior, *geometry[l].interiors] for l in letters}
    counts = {l: {int(z["id"]): 0 for z in segments[l]} for l in letters}
    attempts = {l: Counter() for l in letters}
    trajectory_bin_counts = defaultdict(Counter)
    cohorts = {c: dict(slots=0, controls=0, valid_h1=0, valid_h4=0, invalid_controls=0,
                       per_letter={l: Counter() for l in letters}) for c in ("2A", "2B")}
    maxima = np.zeros(3)
    slot_rows = []
    seen_hashes = set(read_json(base / "stage2a/dedup-reference.json")["initial_state_hashes"])
    seen_seeds = set(read_json(base / "stage2a/dedup-reference.json")["historical_seeds"])
    q2b = {q["request_id"]: q for q in plans["2B"]["requests"]}
    ledger_rows = {z["trajectory_id"]: z for z in ledger["trajectories"]}

    def int_dict(value):
        return {int(k): int(v) for k, v in value.items()}

    def mapped_hits(letter, points):
        hit = set()
        for xy in points:
            point = Point(float(xy[0]), float(xy[1]))
            ri = min(range(len(rings[letter])), key=lambda i: rings[letter][i].distance(point))
            ring = rings[letter][ri]
            if ring.distance(point) > 2:
                continue
            arc = ring.project(point)
            for row in segments[letter]:
                if row["ring"] == ri and row["arc_start"] <= arc <= row["arc_end"] + 1e-6:
                    hit.add(int(row["id"]))
                    break
        return hit

    for cohort in ("2A", "2B"):
        data = base / f"stage{cohort.lower()}/data"
        if cohort == "2B":
            recorded_initial = read_json(stage / "initial-attempts-2a.json")["counts"]
            check(all(dict(attempts[l]) == int_dict(recorded_initial[l]) for l in letters), "initial2A_attempts")
            initial_counts = read_json(base / "stage2a/contact-coverage.json")["segments"]
            check(all(counts[l] == {int(z["id"]): int(z["contact_control_count"]) for z in initial_counts[l]}
                      for l in letters), "inherited2A_actual_control_counts")
            check(seen_hashes == set(dedup["initial_state_hashes"]), "dedup754_inheritance")
            check(seen_seeds == set(dedup["historical_seeds"]), "dedup736_seeds_inheritance")
        for entry in manifests[cohort]["entries"]:
            first_issue = len(issues)
            tid, letter = entry["trajectory_id"], entry["shape"]
            path, meta_path = data / entry["file"], data / entry["metadata_file"]
            check(digest(path) == entry["sha256"] and digest(meta_path) == entry["metadata_sha256"], tid + ":file_hashes")
            meta = read_json(meta_path)
            scene, inputs = meta["scene"], meta["selection_inputs"]
            check(entry["split"] == "train" and entry["stage"] == cohort and letter in letters, tid + ":cohort")
            prefix = int_dict(inputs.get("formal_train_contact_controls_before", inputs.get("new_2a_contact_controls_before", {})))
            check(prefix == counts[letter], tid + ":actual_count_prefix")
            check(int_dict(inputs["historical_TRAIN_independent_trajectories"]) == historical[letter], tid + ":historical_prior")
            check(not inputs["validation_feedback"] and not inputs["previous_probes_added_to_counts"], tid + ":forbidden_count_source")
            if cohort == "2B":
                q = q2b[entry["request_id"]]
                check(meta["request"] == q and (entry["seed"], letter, entry["location"]) ==
                      (q["seed"], q["letter"], q["spatial_class"]), tid + ":frozen_request")
                check(int_dict(inputs["planned_initial_attempts_before"]) == dict(attempts[letter]), tid + ":planned_attempt_prefix")
                replay = plan_scene(letter, q["spatial_class"], q["seed"], gap_band=q["gap_band"], gap_world_px=q["gap_world_px"],
                                    sampling=q["sampling"], coverage_counts=counts[letter], historical_counts=historical[letter],
                                    planned_attempts=dict(attempts[letter]))
                check(replay["selection_evidence"] == scene["selection_evidence"], tid + ":full_rank_replay")
                check(replay["selection_version"] == scene["selection_version"] == "motion-room-planned-attempts-v2"
                      and replay["staging_segment"] == scene["staging_segment"] == entry["selected_initial_arc"], tid + ":choice_replay")
                check(all(np.allclose(replay[k], scene[k], rtol=0, atol=1e-10)
                          for k in ("object_origin", "object_mass_center", "pusher_start")), tid + ":initial_geometry_replay")
            with np.load(path, allow_pickle=False) as arrays:
                s, actions, sub = arrays["states"], arrays["actions"], arrays["substep_penetrations"]
                n, points, offsets = len(actions), arrays["contact_points"], arrays["contact_offsets"]
                pen, valid = arrays["penetrations"], arrays["valid_transition"]
                check(s.shape == (n + 1, 15) and sub.shape == (n, 10, 3) and pen.shape == (n + 1, 3), tid + ":alignment")
                check(s.dtype == sub.dtype == pen.dtype == np.float64, tid + ":float64_audit")
                check(all(np.isfinite(arrays[k]).all() for k in ("states", "actions", "substep_penetrations", "penetrations",
                                                                 "contact_points", "contact_angles")), tid + ":finite")
                check(actions.shape == (n, 2) and not (actions < 0).any() and not (actions > 511).any(), tid + ":absolute_actions")
                check(np.array_equal(sub.max(axis=1), pen[1:]), tid + ":all_substep_maxima")
                computed = np.isfinite(sub).all(axis=(1, 2)) & (sub.max(axis=(1, 2)) <= 2) & np.isfinite(s[1:]).all(axis=1)
                check(np.array_equal(computed, valid), tid + ":valid_flags")
                invalid = int((~computed).sum())
                check(not invalid, tid + ":invalid_physics")
                state_hash = hashlib.sha256(s[0].tobytes()).hexdigest()
                check(state_hash == entry["initial_scene_hash"] and state_hash not in seen_hashes and entry["seed"] not in seen_seeds,
                      tid + ":unique_full_state_seed")
                seen_hashes.add(state_hash)
                seen_seeds.add(entry["seed"])
                check(hashlib.sha256(np.asarray(entry["initial_scene"], dtype=np.float64).tobytes()).hexdigest() == state_hash
                      and hashlib.sha256(np.asarray(scene["initial_state"], dtype=np.float64).tobytes()).hexdigest() == state_hash,
                      tid + ":initial_state_bytes")
                check(np.allclose(s[0, 2:4], scene["object_origin"], rtol=0, atol=1e-10)
                      and np.allclose(s[0, :2], scene["pusher_start"], rtol=0, atol=1e-10)
                      and s[0, 4] == scene["object_angle"] and np.all(s[0, 5:] == 0), tid + ":initial_pose_and_zero_dynamics")
                initial_geometry = affinity.translate(affinity.rotate(geometry[letter], float(s[0, 4]), origin=(0, 0),
                                                                    use_radians=True), *s[0, 2:4])
                xmin, ymin, xmax, ymax = initial_geometry.bounds
                wall_gaps = dict(left=xmin-7, right=504-xmax, top=ymin-7, bottom=504-ymax)
                if entry["location"] == "center":
                    check(min(wall_gaps.values()) > 64, tid + ":actual_center_gaps")
                else:
                    check(all(abs(wall_gaps[w] - entry["gap_requested_world_px"]) < 1e-10
                              for w in entry["location"].split("-")), tid + ":actual_requested_wall_gaps")
                check(offsets.shape == (n + 1,) and offsets[0] == 0 and offsets[-1] == len(points)
                      and not (np.diff(offsets) < 0).any() and len(arrays["contact_angles"]) == len(points), tid + ":contact_alignment")
                # The first actual env.step counts even when it throws. Failed initialization/charge does not.
                exception = meta.get("exception_state") or {}
                entered = n > 0 or bool(exception.get("step_entered") and ledger_rows.get(tid, {}).get("charged_controls", 0))
                if cohort == "2B":
                    check(bool(inputs["planned_attempt_counted"]) == entered, tid + ":actual_start_event")
                    if entered:
                        attempts[letter][int(scene["staging_segment"])] += 1
                        check(int_dict(inputs["planned_initial_attempts_after_start"]) == dict(attempts[letter]), tid + ":attempt_after_start")
                        check(ledger_rows[tid].get("initial_arc_attempt_counted") == scene["staging_segment"], tid + ":durable_attempt_event")
                elif n:
                    attempts[letter][int(scene["staging_segment"])] += 1
                local_center = np.asarray(official_shape(letter).centroid)
                theta = s[:, 4]
                center = s[:, 2:4] + np.column_stack((np.cos(theta) * local_center[0] - np.sin(theta) * local_center[1],
                                                       np.sin(theta) * local_center[0] + np.cos(theta) * local_center[1]))
                translation, rotation = np.linalg.norm(np.diff(center, axis=0), axis=1), np.abs(np.diff(theta))
                moving, static = (translation >= 2) | (rotation >= math.pi / 180), (translation <= .5) & (rotation <= .005)
                check(len(meta["steps"]) == n and not meta["hidden_physics_controls"], tid + ":continuous_step_metadata")
                for t in range(n):
                    step = meta["steps"][t]
                    sign = 1 if (LOCATIONS.index(entry["location"]) + SIDE_SIGN_INDEX[letter]) % 2 else -1
                    check(step["step"] == t and not step["midtrajectory_reset"] and step["approach_side_sign"] == sign,
                          tid + f":continuous_sign:{t}")
                    for arc in mapped_hits(letter, points[offsets[t]:offsets[t + 1]]):
                        counts[letter][arc] += 1
                        trajectory_bin_counts[letter, arc][tid] += 1
                    if t >= 3 and computed[t - 3:t + 1].all():
                        z = cohorts[cohort]["per_letter"][letter]
                        z["h1"] += 1
                        z["moving"] += int(moving[t])
                        z["near_static"] += int(static[t])
                        z["intermediate"] += int(not moving[t] and not static[t])
                        z["near_static_contact"] += int(static[t] and arrays["contact"][t] > 0)
                        z["near_static_no_contact"] += int(static[t] and arrays["contact"][t] == 0)
                c = cohorts[cohort]
                c["slots"] += 1
                c["controls"] += n
                c["invalid_controls"] += invalid
                c["valid_h1"] += int(sum(computed[t - 3:t + 1].all() for t in range(3, n)))
                c["valid_h4"] += int(sum(computed[t - 3:t + 4].all() for t in range(3, n - 3)))
                if n and cohort == "2B":
                    maxima = np.maximum(maxima, sub.max(axis=(0, 1)))
                if cohort == "2B":
                    row = ledger_rows[tid]
                    check(n == entry["controls"] == row["completed_controls"] and row["charged_controls"] >= n, tid + ":ledger_completed")
                slot_rows.append(dict(cohort=cohort, trajectory_id=tid, controls=n, selected_arc=scene["staging_segment"],
                                      issues=issues[first_issue:]))

    final_attempts = read_json(stage / "planned-attempts-current.json")["counts"]
    check(all(dict(attempts[l]) == int_dict(final_attempts[l]) for l in letters), "final_planned_attempts")
    final_coverage = read_json(stage / "contact-coverage.json")["segments"]
    for l in letters:
        for row in final_coverage[l]:
            arc = int(row["id"])
            check(counts[l][arc] == row["contact_control_count"] and dict(trajectory_bin_counts[l, arc]) ==
                  {tid: int(v) for tid, v in row["trajectory_contact_controls"].items()}, f"final_actual_coverage:{l}:{arc}")
    check(cohorts["2A"]["slots"] == 162 and cohorts["2B"]["slots"] == 234, "fixed_slot_denominators")
    check(cohorts["2A"]["controls"] == 10368 and cohorts["2B"]["controls"] == 14976, "fixed_control_denominators")
    check(cohorts["2A"]["valid_h1"] == 9882 and cohorts["2B"]["valid_h1"] == 14274, "fixed_h1_denominators")
    check(cohorts["2A"]["valid_h4"] == 9396 and cohorts["2B"]["valid_h4"] == 13572, "fixed_h4_denominators")
    check(ledger["charged_controls"] == ledger["completed_controls"] == cohorts["2B"]["controls"] == 14976
          and ledger["physics_substeps"] == 149760 and ledger["recorded_partial_substeps_lower_bound"] == 0, "cost_alignment")
    check(ledger["supervised_round_controls"] == 29312 and ledger["historical_plus_new_native_controls"] == 63872, "cumulative_cost")
    check(ledger["failure"] is None and manifests["2B"]["failure"] is None, "collection_failure")
    check(len(ledger["trajectories"]) == 234 and all(z["status"] == "completed" and z["pending_control"] is None
          and z["charged_controls"] == z["completed_controls"] == 64 for z in ledger["trajectories"]), "all234_ledger_slots")
    check(all(ledger[k] == 0 for k in ("model_inference_calls", "unseen_model_inference_calls", "optimizer_attempts", "server_jobs")),
          "forbidden_model_optimizer_server_work")
    physical = read_json(stage / "physics-audit.json")
    check(physical["passed"] and physical["physics_collection_gate_passed"] and physical["invalid_controls"] == 0
          and physical["controls"] == 14976 and physical["audited_substeps"] == 149760
          and physical["valid_h1_candidates"] == 14274 and physical["valid_h4_candidates"] == 13572
          and physical["manifest_sha256"] == digest(stage / "data/manifest.json")
          and np.array_equal(np.asarray(physical["max_substep_penetration_world_px"]), maxima), "root_physics_audit_alignment")

    merged = read_json(merged_path)
    check(merged["kind"] == "native-rgb-formal-train-reference-view.v1" and Path(merged["base_root"]).resolve() == base
          and merged["split"] == "train" and merged["stage"] == "2A+2B", "merged_base_and_scope")
    expected = {(c, e["trajectory_id"]): e for c, m in manifests.items() for e in m["entries"]}
    merged_seen, resolved_files = set(), set()
    check(merged["reference_only"] and merged["npz_files_copied"] == 0 and not merged["training_consumption_authorized"], "reference_only_view")
    for c in ("2A", "2B"):
        prefix = f"stage{c.lower()}/data"
        source = merged["source_manifests"][c]
        check(source["file"] == prefix + "/manifest.json" and source["sha256"] == digest(base / source["file"]), "merged_source:" + c)
        check(source["source_identity_sha256"] == manifests[c]["source_identity_sha256"] ==
              digest(base / f"stage{c.lower()}/source-identity.json"), "merged_source_identity:" + c)
        check(all(merged[k] == manifests[c][k] for k in ("state_columns", "source_root", "controller", "protocol_sha256")),
              "merged_header:" + c)
    versions = Counter()
    for entry in merged["entries"]:
        key = entry["cohort"], entry["trajectory_id"]
        check(key in expected and key not in merged_seen, "merged_unknown_or_duplicate:" + str(key))
        original = expected[key]
        merged_seen.add(key)
        prefix = f"stage{key[0].lower()}/data"
        for field, hfield in (("file", "sha256"), ("metadata_file", "metadata_sha256")):
            path = Path(entry[field])
            target = (base / path).resolve()
            check(not path.is_absolute() and target.is_relative_to((base / prefix).resolve())
                  and entry[field] == (Path(prefix) / original[field]).as_posix()
                  and entry[hfield] == original[hfield] == digest(target), "merged_reference:" + str(key) + ":" + field)
            if field == "file":
                resolved_files.add(str(target))
        check(all(entry[k] == original[k] for k in ("shape", "seed", "initial_scene_hash", "split", "stage", "controls")),
              "merged_entry_identity:" + str(key))
        version = "motion-room-actual-hit-prior-v1" if key[0] == "2A" else "motion-room-planned-attempts-v2"
        check(entry["selector_version"] == version, "merged_selector:" + str(key))
        versions[entry["selector_version"]] += 1
    check(merged_seen == set(expected) and len(resolved_files) == 396, "merged396_complete_unique_references")

    old_audit = read_json(repo / "artifacts/native-rgb-20261028/training-final-audit.json")
    weight_checks = []
    for role, row in old_audit["models"].items():
        ok = Path(row["path"]).stat().st_size == row["bytes"] and digest(row["path"]) == row["sha256"]
        weight_checks.append(dict(role=role, sha256=row["sha256"], matches=check(ok, "old_weight:" + role)))
    check(read_json(repo / "artifacts/native-rgb-20261028/native-rgb-cost-ledger.json") == old_audit["ledger"], "old_native_ledger")

    old_sampler = (stage / "baseline-source/native_rgb_supervised_sampling.py").read_text(encoding="utf-8")
    current_sampler = (repo / "tdyssey/native_rgb_supervised_sampling.py").read_text(encoding="utf-8")
    old_class = next(n for n in ast.parse(old_sampler).body if isinstance(n, ast.ClassDef))
    new_class = next(n for n in ast.parse(current_sampler).body if isinstance(n, ast.ClassDef))
    check(ast.get_source_segment(old_sampler, old_class) == ast.get_source_segment(current_sampler, new_class), "continuous_controller_bytes")

    merged_letters = {}
    for letter in letters:
        z = Counter(cohorts["2A"]["per_letter"][letter]) + Counter(cohorts["2B"]["per_letter"][letter])
        denom = z["h1"]
        merged_letters[letter] = dict(z, moving_fraction=z["moving"] / denom if denom else None,
                                     near_static_fraction=z["near_static"] / denom if denom else None,
                                     h1_gate_passed=bool(denom and z["moving"] / denom >= .4 and z["near_static"] / denom <= .45))
    native_sources = set()
    for pattern in ("tdyssey/native_rgb*.py", "scripts/native_rgb*.py", "tests/test_native_rgb*.py", "configs/native_rgb*.json",
                    "run_native_rgb*", "requirements/native-rgb*", "docs/NATIVE_RGB_WORLD_MODEL.md"):
        native_sources.update(p for p in repo.glob(pattern) if p.is_file())
    footprint = dict(old_native_bytes=sum(p.stat().st_size for p in (repo / "artifacts/native-rgb-20261028").rglob("*") if p.is_file()),
                     supervised_bytes=sum(p.stat().st_size for p in base.rglob("*") if p.is_file()),
                     native_source_bytes=sum(p.stat().st_size for p in native_sources), cap_bytes=3221225472)
    footprint["combined_bytes"] = sum(footprint[k] for k in ("old_native_bytes", "supervised_bytes", "native_source_bytes"))
    check(footprint["combined_bytes"] < footprint["cap_bytes"], "3GiB_cap")
    return dict(kind="native-rgb-supervised-stage2b-independent-postrun.v1", reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                audit_passed=not issues, issues=issues, cohorts=cohorts, merged_per_letter_h1=merged_letters,
                merged_h1_gate_failed_letters=[l for l, z in merged_letters.items() if not z["h1_gate_passed"]],
                source_rows=source_rows, protected2A_files=len(preserved2a), protected_earlier_files=len(old_protected),
                initial_attempt_and_full_rank_replay=dict(trajectories=cohorts["2B"]["slots"],
                    prefixes_and_rank_match=not any("prefix" in x or "replay" in x for x in issues),
                    actual_counts_inherited_from2A=True, planned_attempts_are_not_hits=True),
                merged_reference_versions=dict(versions), merged_manifest_sha256=digest(merged_path),
                source_identity_sha256=digest(stage / "source-identity.json"), manifest_sha256=digest(stage / "data/manifest.json"),
                resource_ledger_sha256=digest(stage / "resource-ledger.json"), max2b_substep_penetrations=maxima.tolist(),
                old_weight_checks=weight_checks, footprint=footprint, slots=slot_rows,
                supervised_controls=14336 + ledger["charged_controls"], all_native_controls=48896 + ledger["charged_controls"],
                reviewer_execution=dict(environment_constructions=0, physics_controls=0, model_calls=0, optimizer_updates=0,
                                        index_draws=0, server_jobs=0), wall_seconds=time.monotonic() - started,
                scope="Integrity/authorized-cost audit is separate from motion-quality targets. No next stage is authorized.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-after-root-completion", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--merged-manifest", type=Path)
    parser.add_argument("--output-stem", default="review-postrun")
    args = parser.parse_args()
    if not args.output_stem.startswith("review-") or Path(args.output_stem).name != args.output_stem:
        raise ValueError("Review outputs must stay in owned review-* artifact files")
    repo = args.root.resolve().parents[1]
    sys.path.insert(0, str(repo))
    stage = args.root / "stage2b"
    targets = [stage / (args.output_stem + suffix) for suffix in (".json", ".md")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Preserve existing review evidence; use another review-* output stem")
    result = audit(args.root, args.merged_manifest or stage / "report/merged-train-manifest.json")
    with targets[0].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    text = ("# Stage2B independent post-run review\n\n"
            f"Integrity/cost audit passed: **{result['audit_passed']}**. Issues: {len(result['issues'])}.\n\n"
            "Saved2A then2B collision points independently rebuild real control-bin coverage and one initial planned attempt per actual started trajectory. "
            "Every2B choice and complete ranked candidate evidence is replayed with pure geometry; no environment is constructed.\n\n"
            f"Cohorts: { {c: {k: v[k] for k in ('slots', 'controls', 'valid_h1', 'valid_h4', 'invalid_controls')} for c, v in result['cohorts'].items()} }.\n\n"
            f"Merged H1 motion-target failures: {result['merged_h1_gate_failed_letters']}. Quality remains separate from integrity.\n\n"
            f"Supervised controls {result['supervised_controls']}; all-native controls {result['all_native_controls']}. "
            f"Footprint {result['footprint']['combined_bytes']:,} bytes below3GiB.\n\n"
            "Reviewer executed0 environment constructions, controls, models, optimizer updates, training-index draws or server jobs. "
            "Root must report the fixed batch and stop for parent review.\n")
    targets[1].write_text(text, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("audit_passed", "issues", "merged_h1_gate_failed_letters", "supervised_controls", "all_native_controls", "wall_seconds")}, indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
