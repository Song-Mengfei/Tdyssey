"""Predeclared seen_val natural/conditional/development indices, without models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SPEC = ROOT/"configs/native_rgb_stage2c_indices.json"
LETTERS = tuple("ABCEFHIJLMOPSTUVXY")
PROTOCOL = "fd358dba8f26a616ec6c55b3f6d532681bf5036e2cf79fe5a467001e3a77ae96"
CATEGORIES = ("moving", "rotation", "boundary_moving", "likely_wall_blocked_proxy",
              "near_static_contact", "near_static_no_contact", "intermediate")
QUOTAS = dict(moving=3, rotation=1, boundary_moving=1, likely_wall_blocked_proxy=1,
              near_static_contact=1, near_static_no_contact=1, intermediate=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False,
                  indent=None if compact else 2, separators=(",", ":") if compact else None)
        stream.write("\n")


def load_spec(path=DEFAULT_SPEC):
    spec = read_json(path)
    if (spec["algorithm_version"] != "seen-val-panels-v1" or type(spec["algorithm_seed"]) is not int
            or spec["split"] != "seen_val" or tuple(spec["letters"]) != LETTERS
            or spec["history_transitions"] != 3 or spec["future_transitions"] != 4
            or spec["complete_64_control_anchor_range_inclusive"] != [3, 60]
            or tuple(spec["extra"]["category_tie_order"]) != CATEGORIES or spec["extra"]["quotas"] != QUOTAS):
        raise ValueError("Index policy differs from the predeclared protocol")
    thresholds = spec["truth_labels"]
    expected = dict(moving_translation_world_px_gte=2., moving_rotation_abs_deg_gte=1.,
                    near_static_translation_world_px_lte=.5, near_static_rotation_abs_rad_lte=.005,
                    rotation_abs_deg_gte=1.)
    if any(thresholds[key] != value for key, value in expected.items()):
        raise ValueError("Truth thresholds are fixed")
    return spec


def verify_frozen_inputs(data_root,spec_path):
    """Bind production indices to completed, audited data and pre-step rules."""
    data_root=Path(data_root).resolve();stage=data_root.parent
    manifest=read_json(data_root/"manifest.json");manifest_hash=sha(data_root/"manifest.json")
    if (manifest.get("stage")!="2C" or manifest.get("split")!="seen_val" or manifest.get("failure")
            or manifest.get("unexecuted_request_ids") or len(manifest.get("entries",[]))!=162
            or manifest.get("actual_controls")!=10368 or manifest.get("saved_controls")!=10368
            or any(row.get("controls")!=64 or row.get("error") or row.get("invalid_controls")!=0 for row in manifest["entries"])):
        raise ValueError("Evaluation indices require the complete, successful independent162-slot validation batch")
    identity_path=stage/"source-identity.json";identity=read_json(identity_path)
    if manifest.get("source_identity_sha256")!=sha(identity_path):raise ValueError("Source identity/manifest mismatch")
    for row in identity["files"]:
        snapshot=(stage/row["snapshot"]).resolve()
        if not snapshot.is_relative_to(stage) or sha(snapshot)!=row["sha256"]:
            raise ValueError("Frozen pre-step source snapshot differs")
    index_source=[row for row in identity["files"] if row["snapshot"].endswith("scripts/native_rgb_stage2c_indices.py")]
    if len(index_source)!=1 or index_source[0]["sha256"]!=sha(__file__):raise ValueError("Index source differs from pre-step freeze")
    policy_hash=sha(spec_path)
    if (manifest.get("index_spec_sha256")!=policy_hash
            or sha(stage/"source-snapshot/frozen-index-spec.json")!=policy_hash):
        raise ValueError("Index policy/seed differs from the pre-step frozen rules")
    audit_path=stage/"physics-audit.json";audit=read_json(audit_path)
    if (audit.get("passed") is not True or audit.get("physics_collection_gate_passed") is not True
            or audit.get("stage")!="2C" or audit.get("split")!="seen_val" or audit.get("manifest_sha256")!=manifest_hash
            or audit.get("threshold_world_px")!=2 or audit.get("threshold_relaxed") is not False
            or audit.get("invalid_controls")!=0 or audit.get("controls")!=10368 or audit.get("audited_substeps")!=103680):
        raise ValueError("Independent frozen-threshold physical audit has not passed for this manifest")
    return dict(manifest_sha256=manifest_hash,physics_audit_sha256=sha(audit_path),source_identity_sha256=sha(identity_path),policy_sha256=policy_hash)


def _rank(candidate, trajectory, spec, domain):
    values = [spec["algorithm_version"], spec["algorithm_seed"], domain, "seen_val", candidate["letter"],
              candidate["trajectory_id"], trajectory["npz_sha256"], trajectory["metadata_sha256"], int(candidate["anchor"])]
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), candidate["trajectory_id"], int(candidate["anchor"])


def catalog_from_records(records, *, planned_requests=None):
    """Pure saved-array geometry; no rendering, physics, model or sampling iterator."""
    import numpy as np
    from tdyssey.native_rgb_motion_statistics import area_centroids, geometric_wall_gaps, gap_band, transition_rows

    candidates, trajectories, seen = [], [], set()
    for record in records:
        entry, arrays = record["entry"], record["arrays"]
        tid, letter = entry["trajectory_id"], entry["shape"]
        if entry.get("split") != "seen_val" or letter not in LETTERS or tid in seen:
            raise ValueError("Only distinct seen_val trajectories of the fixed18 letters are allowed")
        seen.add(tid)
        rows = transition_rows(entry, arrays, record.get("steps"))
        states = np.asarray(arrays["states"], dtype=np.float64)
        centers = area_centroids(letter, states)
        gaps = geometric_wall_gaps(letter, states)
        n = len(rows)
        identity = dict(trajectory_id=tid, letter=letter, split="seen_val", npz_sha256=entry["sha256"],
            metadata_sha256=entry["metadata_sha256"], file=entry["file"], metadata_file=entry["metadata_file"],
            protocol_sha256=PROTOCOL, collector_version=entry.get("stage", "2C"),
            selection_version=(record.get("scene") or {}).get("selection_version", entry.get("selector_version")),
            requested_spatial_class=entry.get("location", entry.get("spatial_class", "unknown")),
            seed=entry.get("seed"), recorded_controls=n, raw_valid_controls=sum(row["valid_transition"] for row in rows),
            valid_h1_candidates=sum(row["valid_h1_candidate"] for row in rows), possible_h4_anchors=max(0,n-6),
            valid_h4_candidates=0, invalid_h4_anchor_ids=[], no_recorded_h4_reason=None)
        for anchor in range(3, n-3):
            if not all(rows[t]["valid_transition"] for t in range(anchor-3,anchor+4)):
                identity["invalid_h4_anchor_ids"].append(anchor)
                continue
            current = rows[anchor]
            future = rows[anchor:anchor+4]
            translation = float(np.linalg.norm(centers[anchor+4]-centers[anchor]))
            rotation = abs(float(states[anchor+4,4]-states[anchor,4]))
            moving = current["motion_class"] == "moving"
            static = current["motion_class"] == "near_static"
            boundary_min = min(current["min_wall_gap_before"], current["min_wall_gap_after"])
            conditions = dict(moving=moving, rotation=current["rotation_abs_deg"] >= 1.,
                boundary_moving=bool(moving and boundary_min <= 32.),
                likely_wall_blocked_proxy=bool(current["likely_wall_blocked_proxy"]),
                near_static_contact=bool(static and current["contact"] is True),
                near_static_no_contact=bool(static and current["contact"] is False),
                intermediate=current["motion_class"] == "intermediate")
            h1 = {key: current[key] for key in ("motion_class", "contact", "translation_world_px", "rotation_abs_rad", "rotation_abs_deg",
                "actual_spatial_class_before", "actual_spatial_class_after", "min_wall_gap_before", "min_wall_gap_after", "gap_band_before", "gap_band_after",
                "likely_wall_blocked_proxy", "requested_spatial_class", "requested_corner_max_gap_before", "requested_corner_max_gap_after",
                "requested_corner_gap_band_before", "requested_corner_gap_band_after")}
            h1.update(actual_boundary_min_world_px=boundary_min, actual_boundary_gap_band=gap_band(boundary_min),
                      actual_boundary_le32=boundary_min <= 32., moving=moving, near_static=static)
            for wall in ("left", "right", "top", "bottom"):
                h1[wall+"_gap_before"] = current[wall+"_gap_before"]
                h1[wall+"_gap_after"] = current[wall+"_gap_after"]
            h4 = dict(endpoint_translation_world_px=translation, endpoint_rotation_abs_rad=rotation,
                endpoint_rotation_abs_deg=math.degrees(rotation),
                endpoint_moving=bool(translation>=2. or rotation>=math.pi/180),
                endpoint_near_static=bool(translation<=.5 and rotation<=.005),
                path_any_step_moving=any(row["motion_class"]=="moving" for row in future),
                path_all_steps_near_static=all(row["motion_class"]=="near_static" for row in future),
                path_translation_sum_world_px=sum(row["translation_world_px"] for row in future),
                path_rotation_abs_sum_rad=sum(row["rotation_abs_rad"] for row in future),
                path_min_wall_gap_world_px=float(gaps[anchor:anchor+5].min()),
                path_actual_boundary_le32=bool(gaps[anchor:anchor+5].min()<=32),
                contact_any=any(row["contact"] is True for row in future),
                contact_every=all(row["contact"] is True for row in future),
                legal_transition_start=anchor-3, legal_transition_end_inclusive=anchor+3,
                history_state_start=anchor-3, future_state_end_inclusive=anchor+4)
            candidates.append(dict(window_id=f"{tid}:{anchor}", trajectory_id=tid, letter=letter, split="seen_val", anchor=anchor,
                                   conditions=conditions, h1=h1, h4=h4))
            identity["valid_h4_candidates"] += 1
        if not identity["valid_h4_candidates"]:
            identity["no_recorded_h4_reason"] = "no_H4_legal_history_and_future"
        trajectories.append(identity)
    requests = planned_requests or [dict(request_id=t["trajectory_id"], letter=t["letter"]) for t in trajectories]
    if any(row["letter"] not in LETTERS for row in requests):
        raise ValueError("Planned validation requests contain a held-out letter")
    return dict(kind="seen-val-H4-legal-candidate-catalog.v1", split="seen_val", candidates=candidates, trajectories=trajectories,
        planned_requests=[dict(trajectory_id=row.get("request_id",row.get("trajectory_id")),letter=row["letter"]) for row in requests],
        raw_controls=sum(t["recorded_controls"] for t in trajectories), valid_h1_candidates=sum(t["valid_h1_candidates"] for t in trajectories),
        valid_h4_candidates=len(candidates), geometry_labels_are_offline_only=True, model_inputs="Original RGB and actions only")


def _panel_summary(windows):
    return dict(unique_windows=len(windows), independent_trajectories=len({row["trajectory_id"] for row in windows}),
        per_letter=dict(Counter(row["letter"] for row in windows)), anchor_histogram=dict(sorted(Counter(row["anchor"] for row in windows).items())),
        h1_motion=dict(Counter(row["h1"]["motion_class"] for row in windows)),
        h1_contact=dict(Counter(str(row["h1"]["contact"]) for row in windows)),
        conditional_membership_counts={category:sum(row["conditions"][category] for row in windows) for category in CATEGORIES},
        actual_spatial_before=dict(Counter(row["h1"]["actual_spatial_class_before"] for row in windows)),
        actual_H1_boundary_gap_band=dict(Counter(row["h1"]["actual_boundary_gap_band"] for row in windows)),
        h4_endpoint_moving=sum(row["h4"]["endpoint_moving"] for row in windows),
        h4_endpoint_near_static=sum(row["h4"]["endpoint_near_static"] for row in windows),
        h4_path_any_step_moving=sum(row["h4"]["path_any_step_moving"] for row in windows),
        h4_path_all_steps_near_static=sum(row["h4"]["path_all_steps_near_static"] for row in windows))


def _overlap(windows):
    by_trajectory = defaultdict(list)
    for row in windows:
        by_trajectory[row["trajectory_id"]].append(row)
    counts = Counter()
    pairs = []
    for tid, rows in by_trajectory.items():
        rows = sorted(rows,key=lambda row:row["anchor"])
        for index, left in enumerate(rows):
            for right in rows[index+1:]:
                delta = right["anchor"]-left["anchor"]
                flags = dict(H4_state_overlap=delta<=7, H4_transition_overlap=delta<=6,
                             H4_future_transition_overlap=delta<=3, H1_state_overlap=delta<=4)
                counts.update({key:int(value) for key,value in flags.items()})
                counts["same_trajectory_pairs"] += 1
                if any(flags.values()):
                    pairs.append(dict(trajectory_id=tid,left_window_id=left["window_id"],right_window_id=right["window_id"],anchor_distance=delta,**flags))
    return dict(counts=dict(counts), correlated_pairs=pairs,
        qualification="Distinct anchors may share states/transitions; independent trajectory counts are reported separately.")


def select_indices(catalog, spec):
    """Deterministic evaluation-index instantiation, never training-index draws."""
    trajectories = {row["trajectory_id"]:row for row in catalog["trajectories"]}
    if len(trajectories) != len(catalog["trajectories"]):
        raise ValueError("Duplicate trajectory identity")
    candidates = catalog["candidates"]
    if len({row["window_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate legal candidate window")
    grouped = defaultdict(list)
    for row in candidates:
        if row["split"] != "seen_val" or row["letter"] not in LETTERS:
            raise ValueError("Only fixed seen_val letters are accepted")
        grouped[row["trajectory_id"]].append(row)
    natural, extra, development, missing, support = [], [], [], [], {}
    used, loads = set(), Counter()

    def selected(candidate,panel,reason,domain):
        item = copy.deepcopy(candidate)
        item.update(source=copy.deepcopy(trajectories[item["trajectory_id"]]), panels=[panel], selection_reason=reason,
                    selection_hash=_rank(candidate,trajectories[candidate["trajectory_id"]],spec,domain)[0],
                    conditional_panels=[category for category in CATEGORIES if item["conditions"][category]])
        used.add(item["window_id"])
        loads[item["trajectory_id"]] += 1
        return item

    for request in catalog["planned_requests"]:
        tid = request["trajectory_id"]
        available = grouped[tid]
        if not available:
            missing.append(dict(panel="natural",trajectory_id=tid,letter=request["letter"],reason="no_recorded_H4_legal_window"))
            continue
        choice = min(available,key=lambda row:_rank(row,trajectories[tid],spec,"natural"))
        natural.append(selected(choice,"natural","uniform_hash_one_per_trajectory","natural"))
    for letter in LETTERS:
        pool = [row for row in candidates if row["letter"]==letter and row["window_id"] not in used]
        initial_support = {category:[row for row in pool if row["conditions"][category]] for category in CATEGORIES}
        category_order = sorted(CATEGORIES,key=lambda category:(len(initial_support[category]),CATEGORIES.index(category)))
        support[letter] = dict(order=category_order, counts={category:len(rows) for category,rows in initial_support.items()},
                              independent_trajectories={category:len({row["trajectory_id"] for row in rows}) for category,rows in initial_support.items()})
        for category in category_order:
            for slot in range(QUOTAS[category]):
                available = [row for row in initial_support[category] if row["window_id"] not in used]
                if not available:
                    missing.append(dict(panel="extra",letter=letter,category=category,slot=slot+1,
                        reason="no_condition_support_after_natural" if not initial_support[category] else "support_consumed_by_prior_unique_slots"))
                    continue
                choice = min(available,key=lambda row:(loads[row["trajectory_id"]],*_rank(row,trajectories[row["trajectory_id"]],spec,"extra")))
                reason = dict(category=category,slot=slot+1,initial_support=len(initial_support[category]),
                              selected_trajectory_load_before=loads[choice["trajectory_id"]],rule="rarest_fixed_order_then_least_trajectory_load_then_hash")
                extra.append(selected(choice,"extra",reason,"extra"))
    combined = natural+extra
    for letter in LETTERS:
        for role, pool, domain in (("natural",[row for row in natural if row["letter"]==letter],"development-natural"),
                                  ("extra_moving",[row for row in extra if row["letter"]==letter and row["conditions"]["moving"]],"development-extra-moving")):
            if not pool:
                missing.append(dict(panel="development",letter=letter,role=role,reason="no_selected_natural" if role=="natural" else "no_selected_extra_H1_moving"))
                continue
            choice = min(pool,key=lambda row:_rank(row,trajectories[row["trajectory_id"]],spec,domain))
            choice["panels"].append("development")
            dev = copy.deepcopy(choice)
            dev.update(development_role=role, development_selection_hash=_rank(choice,trajectories[choice["trajectory_id"]],spec,domain)[0])
            development.append(dev)
    if len(used) != len(combined) or len({row["window_id"] for row in development}) != len(development):
        raise AssertionError("Index deduplication failure")
    return dict(kind="seen-val-predeclared-panels.v1",algorithm_version=spec["algorithm_version"],algorithm_seed=spec["algorithm_seed"],
        natural=natural,extra=extra,combined=combined,development=development,missing_slots=missing,extra_support_after_natural=support,
        panel_summaries={name:_panel_summary(values) for name,values in (("natural",natural),("extra",extra),("combined",combined),("development",development))},
        temporal_overlap={"combined":_overlap(combined),"development":_overlap(development)},
        unique_inference_windows=len(combined), panel_membership_count=sum(len(row["panels"]) for row in combined),
        conditional_membership_count=sum(len(row["conditional_panels"]) for row in combined),
        future_inference_deduplication_key="window_id", natural_real_world_frequency_claim=False,
        training_index_draws=0,model_calls=0,optimizer_updates=0,physics_controls=0)


def load_catalog(data_root):
    from tdyssey.native_rgb_motion_statistics import read_records
    data_root = Path(data_root).resolve()
    if (data_root.parent/"COLLECTION_RUNNING.lock").exists():
        raise RuntimeError("Do not read active partial seen_val data")
    manifest = read_json(data_root/"manifest.json")
    if manifest.get("split") != "seen_val" or manifest.get("protocol_sha256") != PROTOCOL:
        raise ValueError("Wrong validation split or frozen physics protocol")
    for entry in manifest["entries"]:
        if entry.get("split") != "seen_val" or entry["shape"] not in LETTERS:
            raise ValueError("No TRAIN/probe/unseen data allowed")
        for path_key,sha_key in (("file","sha256"),("metadata_file","metadata_sha256")):
            path=(data_root/entry[path_key]).resolve()
            if not path.is_relative_to(data_root) or sha(path) != entry[sha_key]:
                raise ValueError("Immutable NPZ/metadata hash differs")
    records = read_records(data_root,manifest)
    for record in records:
        metadata = read_json(data_root/record["entry"]["metadata_file"])
        record["scene"] = metadata.get("scene") or {}
    request_path = data_root.parent/"frozen-requests.json"
    planned = read_json(request_path)["requests"] if request_path.exists() else None
    catalog = catalog_from_records(records,planned_requests=planned)
    catalog.update(manifest_sha256=sha(data_root/"manifest.json"),data_root=str(data_root),
        collection_failure=manifest.get("failure"),planned_slots=manifest.get("planned_slots"),
        manifest_protocol_sha256=manifest["protocol_sha256"])
    return catalog


def build_indices(data_root, output, *, spec_path=DEFAULT_SPEC):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve an instantiated evaluation panel; choose a new output directory")
    spec = load_spec(spec_path)
    bindings=verify_frozen_inputs(data_root,spec_path)
    catalog = load_catalog(data_root)
    result = select_indices(catalog,spec)
    result.update(policy_sha256=sha(spec_path),policy_path=str(Path(spec_path).resolve()),
                  source_sha256=sha(__file__),manifest_sha256=catalog["manifest_sha256"],data_root=catalog["data_root"],
                  candidate_catalog_summary={key:catalog[key] for key in ("raw_controls","valid_h1_candidates","valid_h4_candidates","planned_slots","collection_failure")})
    result["frozen_input_bindings"]=bindings
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"candidate-catalog.json",catalog,compact=True)
    write_json(output/"indices.json",result)
    with (output/"policy.json").open("xb") as stream:stream.write(Path(spec_path).read_bytes())
    from scripts.native_rgb_stage2a_coverage import _write_csv
    _write_csv(output/"evaluation-window-indices.csv",[dict(window_id=row["window_id"],trajectory_id=row["trajectory_id"],
        letter=row["letter"],split=row["split"],anchor=row["anchor"],npz_sha256=row["source"]["npz_sha256"],
        metadata_sha256=row["source"]["metadata_sha256"],panels=row["panels"],conditional_panels=row["conditional_panels"],
        selection_reason=row["selection_reason"],selection_hash=row["selection_hash"],h1_motion=row["h1"]["motion_class"],
        h1_contact=row["h1"]["contact"],actual_boundary_gap_band=row["h1"]["actual_boundary_gap_band"],
        h4_endpoint_moving=row["h4"]["endpoint_moving"],h4_path_any_step_moving=row["h4"]["path_any_step_moving"]) for row in result["combined"]])
    lines=["# Frozen seen-letter evaluation indices","",f"Policy seed={spec['algorithm_seed']}; policySHA256 `{result['policy_sha256']}`.",
        "Natural is uniform pseudorandom per-trajectory H4-legal anchor ranking, under balanced scene sampling rather than real-world operation frequencies.","",
        "| Panel | Unique windows | Independent trajectories | H1 motion | H4 endpoint moving | H4 path any moving |",
        "|---|---:|---:|---|---:|---:|"]
    for name,value in result["panel_summaries"].items():
        lines.append(f"| {name} | {value['unique_windows']} | {value['independent_trajectories']} | {value['h1_motion']} | {value['h4_endpoint_moving']} | {value['h4_path_any_step_moving']} |")
    lines += ["",f"Missing requested slots: {len(result['missing_slots'])}; exact reasons in indices.json. No category replacements.",
        "H1 conditional labels cross categories. Each (trajectory,anchor) is inferred once despite multiple labels/panels. H4 endpoint and four-step path labels remain separate.",
        "Distinct windows can share states/transitions. Exact correlated pairs and per-trajectory support are in temporal_overlap; no independent-window claim.",
        "Complete64 controls: anchors3..60; every history/future transition a−3..a+3 must be physically valid. No padding or cross-trajectory history.",
        "Development is a subset of these seen-letter development-validation indices, not an unbiased final test. No models, optimizers or training sampling iterators were run.",
        "",f"Index sourceSHA256 `{result['source_sha256']}`."]
    with (output/"INDEX_REPORT.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    write_json(output/"files.json",[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--spec",default=str(DEFAULT_SPEC))
    args=parser.parse_args(argv)
    result=build_indices(args.data,args.output,spec_path=args.spec)
    print(json.dumps(dict(unique_windows=result["unique_inference_windows"],natural=len(result["natural"]),extra=len(result["extra"]),
                          development=len(result["development"]),missing_slots=len(result["missing_slots"]))))
    return 0


if __name__=="__main__":raise SystemExit(main())
