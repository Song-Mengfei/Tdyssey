"""Formal TRAIN stage2A recorded-state audit; no model, physics step, or index draw."""

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tdyssey import native_rgb_motion_statistics as stats
from tdyssey.native_rgb_data import TRAIN_LETTERS

NEW_LETTERS=tuple("AEFHIJMPSUVY")
CATEGORIES=("moving","rotation","near_static_contact","failure")
FROZEN_STATS_SHA256="bf5641fd269a69d02db4332987da4794a0df5e2fd47cbcefb9766a276dfc2c49"
EXAMPLE_RULES=dict(new_letters=list(NEW_LETTERS),desired_categories_cycle=list(CATEGORIES),
    within_category="earliest trajectory_id then anchor; near_static_contact prefers likely_wall_blocked_proxy",
    missing_category="explicitly mark unavailable; fallback moving,rotation,near_static_contact,failure,then earliest H1 anchor",
    extras="one earliest case for each still-unrepresented available category",max_cases=16,
    max_sheets=2,max_rows_per_sheet=8,max_gifs=4,max_consecutive_saved_frames=8,
    failure="independently invalid transition or recorded recovery failed; no inferred physical failure from staging")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def _finite(value):
    import math
    return float(value) if math.isfinite(float(value)) else None


def _counts(rows):
    rows=list(rows)
    return dict(total=len(rows),valid=sum(row["valid_transition"] for row in rows),
        motion=dict(Counter(row["motion_class"] for row in rows)),
        contact_by_motion={contact:dict(Counter(row["motion_class"] for row in rows if row["contact"] is contact))
                           for contact in (True,False)},
        near_static_contact=sum(row["near_static_contact"] for row in rows),
        likely_wall_blocked_proxy=sum(row["likely_wall_blocked_proxy"] for row in rows))


def _support(rows):
    rows=list(rows)
    return dict(feasibility_analysis_only=stats.proposed_index_feasibility(rows),
        strata={name:dict(unique_windows=sum(row["index_stratum"]==name for row in rows),
            independent_trajectories=len({row["trajectory_id"] for row in rows if row["index_stratum"]==name}))
            for name in stats.STRATA},index_draws=0,index_materialized=False,
        overlapping_windows_are_not_independent=True)


def _invalid_reasons(row):
    reasons=[]
    if not row["declared_valid"]:reasons.append("collector_declared_invalid")
    if not row["independent_valid"]:
        if not row["substep_evidence_present"]:reasons.append("missing_substep_evidence")
        maximum=row["max_penetration_world_px"]
        if maximum is not None and maximum>2:reasons.append("penetration_over_fixed_2px")
        if any(row[name] is None for name in ("translation_world_px","rotation_abs_rad","max_penetration_world_px")):
            reasons.append("missing_or_nonfinite_measurement")
        if not reasons:reasons.append("independent_physics_or_action_validity_false")
    return reasons


def horizon_audit(record,horizon):
    """H1 t-3:t+1; H4 t-3:t+4. All real anchors retained in exclusions."""
    import numpy as np
    if horizon not in (1,4):raise ValueError("Only H1 and H4 recorded-window audits")
    rows=stats.transition_rows(record["entry"],record["arrays"],record.get("steps"));n=len(rows)
    valid=np.asarray([row["valid_transition"] for row in rows],bool)
    states=np.asarray(record["arrays"]["states"],float);centers=stats.area_centroids(record["entry"]["shape"],states)
    motion=stats.motion_components(record["entry"]["shape"],states)
    exclusions=Counter();invalid_source=Counter();universe=[];selected=[]
    for anchor in range(n):
        if anchor<3:reason="insufficient_recorded_history"
        elif anchor+horizon>n:reason="insufficient_recorded_future"
        else:
            history_ok=bool(valid[anchor-3:anchor].all());future_ok=bool(valid[anchor:anchor+horizon].all())
            reason=("invalid_history_and_future" if not history_ok and not future_ok else
                    "invalid_history" if not history_ok else "invalid_future" if not future_ok else "valid")
            universe.append(anchor)
            if reason!="valid":
                source_reasons={name for t in range(anchor-3,anchor+horizon) for name in _invalid_reasons(rows[t])}
                invalid_source.update(source_reasons)
            else:selected.append(anchor)
        exclusions[reason]+=1
    metrics={key:[] for key in ("endpoint_translation_world_px","endpoint_translation_image96_px",
        "endpoint_rotation_abs_rad","endpoint_rotation_abs_deg","endpoint_wrapped_rotation_abs_rad",
        "path_translation_world_px","path_translation_image96_px","path_rotation_abs_rad")}
    for anchor in selected:
        distance=float(np.linalg.norm(centers[anchor+horizon]-centers[anchor]));delta=float(states[anchor+horizon,4]-states[anchor,4])
        path=float(sum(motion["translation_world_px"][anchor:anchor+horizon]))
        metrics["endpoint_translation_world_px"].append(_finite(distance))
        metrics["endpoint_translation_image96_px"].append(_finite(distance*96/512))
        metrics["endpoint_rotation_abs_rad"].append(_finite(abs(delta)))
        metrics["endpoint_rotation_abs_deg"].append(_finite(abs(delta)*180/np.pi))
        metrics["endpoint_wrapped_rotation_abs_rad"].append(_finite(abs(np.arctan2(np.sin(delta),np.cos(delta)))))
        metrics["path_translation_world_px"].append(_finite(path));metrics["path_translation_image96_px"].append(_finite(path*96/512))
        metrics["path_rotation_abs_rad"].append(_finite(sum(motion["rotation_abs_rad"][anchor:anchor+horizon])))
    return dict(horizon=horizon,elapsed_seconds=.1*horizon,recorded_anchor_controls=n,
        planned_candidates_per_complete_64=64-3-(horizon-1),candidate_universe=len(universe),valid_candidates=len(selected),
        invalid_candidate_windows=len(universe)-len(selected),exclusive_anchor_outcomes=dict(exclusions),
        invalid_source_reasons_nonexclusive=dict(invalid_source),valid_anchors=selected,continuous_values=metrics,
        contact_any_future=sum(any(rows[t]["contact"] is True for t in range(a,a+horizon)) for a in selected),
        contact_every_future=sum(all(rows[t]["contact"] is True for t in range(a,a+horizon)) for a in selected),
        scope="Recorded truth endpoint/path only; H4 is .4s, not the .1s H1 gate or a new training objective")


def _combine_horizons(values,horizon):
    exclusions=Counter();sources=Counter();metrics=defaultdict(list)
    for value in values:
        exclusions.update(value["exclusive_anchor_outcomes"]);sources.update(value["invalid_source_reasons_nonexclusive"])
        for name,numbers in value["continuous_values"].items():metrics[name].extend(numbers)
    return dict(horizon=horizon,elapsed_seconds=.1*horizon,
        recorded_anchor_controls=sum(value["recorded_anchor_controls"] for value in values),
        candidate_universe=sum(value["candidate_universe"] for value in values),
        valid_candidates=sum(value["valid_candidates"] for value in values),
        invalid_candidate_windows=sum(value["invalid_candidate_windows"] for value in values),
        exclusive_anchor_outcomes=dict(exclusions),invalid_source_reasons_nonexclusive=dict(sources),
        continuous={name:stats.continuous_distribution(numbers) for name,numbers in metrics.items()},
        contact_any_future=sum(value["contact_any_future"] for value in values),
        contact_every_future=sum(value["contact_every_future"] for value in values),
        h1_gate_changed=False,model_calls=0,training_target_changed=False)


def _corner_cross(rows):
    groups=defaultdict(list)
    for row in rows:
        walls=[wall for wall in stats.WALLS if wall in row["requested_spatial_class"].split("-")]
        if len(walls)!=2:continue
        key="|".join((row["requested_spatial_class"],stats.gap_band(row[walls[0]+"_gap_before"]),stats.gap_band(row[walls[1]+"_gap_before"])))
        groups[key].append(row)
    return {key:_counts(values) for key,values in sorted(groups.items())}


def analyze_records(records,*,planned_requests=None,core=None):
    """Formal train only; fixed162-slot denominator survives partial/failed batches."""
    records=list(records)
    if any(record["entry"].get("split")!="train" for record in records):raise ValueError("Stage2A report reads formal train only")
    keys=[(record["entry"]["shape"],record["entry"]["location"]) for record in records]
    expected={(letter,location) for letter in TRAIN_LETTERS for location in stats.SPATIAL_CLASSES}
    if len(keys)!=len(set(keys)) or set(keys)-expected:raise ValueError("Repeated or unexpected formal letter/location slot")
    if planned_requests is not None:
        planned_requests=planned_requests.get("requests",[]) if isinstance(planned_requests,dict) else list(planned_requests)
        requested={(row["letter"],row.get("spatial_class",row.get("location"))) for row in planned_requests}
        if len(planned_requests)!=162 or requested!=expected:raise ValueError("Frozen requests must declare all162 distinct TRAIN letter/location slots")
    core=stats.summarize_records(records,expected_letters=TRAIN_LETTERS) if core is None else core
    rows=core["rows"];candidates=[row for row in rows if row["valid_h1_candidate"]]
    horizon_values={h:[horizon_audit(record,h) for record in records] for h in (1,4)}
    trajectories=[]
    for index,record in enumerate(records):
        entry=record["entry"];trajectory_rows=[row for row in rows if row["trajectory_id"]==entry["trajectory_id"]]
        h1,h4=horizon_values[1][index],horizon_values[4][index]
        trajectories.append(dict(trajectory_id=entry["trajectory_id"],letter=entry["shape"],location=entry["location"],
            raw_controls=len(trajectory_rows),raw=_counts(trajectory_rows),valid_h1_candidates=h1["valid_candidates"],
            valid_h4_candidates=h4["valid_candidates"],h1_exclusive_anchor_outcomes=h1["exclusive_anchor_outcomes"],
            h4_exclusive_anchor_outcomes=h4["exclusive_anchor_outcomes"],
            missing_planned_controls=max(0,64-len(trajectory_rows)),
            no_contact=bool(trajectory_rows and all(row["contact"] is False for row in trajectory_rows)),
            metadata_error=record.get("metadata",{}).get("error")))
    per_letter={}
    for letter in TRAIN_LETTERS:
        subset=[row for row in candidates if row["letter"]==letter];indices=[i for i,r in enumerate(records) if r["entry"]["shape"]==letter]
        per_letter[letter]=dict(planned_trajectories=9,recorded_trajectories=len(indices),planned_controls=576,
            raw=stats.summarize_rows([row for row in rows if row["letter"]==letter]),h1=core["per_letter"][letter]["valid_h1"],
            h1_gate=core["per_letter_gate"][letter],h4=_combine_horizons([horizon_values[4][i] for i in indices],4),
            planned_h1_candidates=549,planned_h4_candidates=522,strata_support=_support(subset))
    initial={}
    for location in stats.SPATIAL_CLASSES:
        actual=core["initial_scene_distribution"][location]
        expected_bands={">64":18} if location=="center" else {"0-8":6,"8-32":6,"32-64":6}
        initial[location]=dict(expected_trajectories=18,expected_actual_min_gap_bands=expected_bands,**actual,
            exact_expected_gap_distribution=actual["actual_min_gap_bands"]==expected_bands)
    report=dict(kind="native-rgb-stage2a-formal-train-report.v1",split="train",planned_slots=162,planned_controls=10368,
        recorded_slots=len(records),recorded_controls=len(rows),missing_planned_controls=max(0,10368-len(rows)),
        missing_slots=[dict(letter=l,location=p) for l,p in sorted(expected-set(keys))],
        planned_h1_candidates=9882,planned_h4_candidates=9396,
        missing_unrecorded_h1_candidate_windows=max(0,9882-sum(v["candidate_universe"] for v in horizon_values[1])),
        missing_unrecorded_h4_candidate_windows=max(0,9396-sum(v["candidate_universe"] for v in horizon_values[4])),
        all_real_transitions=core["all_real_transitions"],valid_h1_candidates=core["valid_h1_candidates"],
        horizons={str(h):_combine_horizons(horizon_values[h],h) for h in (1,4)},per_letter=per_letter,trajectories=trajectories,
        complete_fixed_batch=bool(len(records)==162 and all(t["raw_controls"]==64 for t in trajectories)),
        all_h1_letter_gates_passed=core["all_expected_letters_raw_gate_passed"],
        no_contact_trajectories=[t for t in trajectories if t["no_contact"]],
        strata_support=_support(candidates),initial_scene_distribution=initial,
        four_signed_wall_gaps={population:{name:stats.continuous_distribution(row[name] for row in subset)
            for name in [f"{wall}_gap_{side}" for wall in stats.WALLS for side in ("before","after")]}
            for population,subset in (("all_real",rows),("valid_h1",candidates))},
        requested_actual_space_cross={population:{f"{requested}|{actual}":_counts([r for r in subset
            if r["requested_spatial_class"]==requested and r["actual_spatial_class_before"]==actual])
            for requested in stats.SPATIAL_CLASSES for actual in stats.SPATIAL_CLASSES}
            for population,subset in (("all_real",rows),("valid_h1",candidates))},
        requested_corner_two_gap_cross={population:_corner_cross(subset) for population,subset in (("all_real",rows),("valid_h1",candidates))},
        h1_detail_report="h1/motion-statistics.json",example_selection_rules=dict(EXAMPLE_RULES),
        h4_scope="Truth-only .4s endpoint and path distributions. No .1s H1 gate, sampler, loss, or training objective changes.",
        index_draws=0,physics_controls_executed_by_report=0,model_calls=0,optimizer_updates=0,
        statistics_source_sha256=core["statistics_source_sha256"])
    return report


def _failed(step):
    before=step.get("recovery") or {};after=step.get("observed",{}).get("recovery") or {};event={**before,**after}
    return event.get("status")=="failed" or bool(event.get("failure_reason"))


def select_examples(records):
    """Frozen rules, independent of any model output; unfilled categories disclosed."""
    pool=[]
    for record in records:
        steps=record.get("steps",[])
        for row in stats.transition_rows(record["entry"],record["arrays"],steps):
            if not row["h1_anchor_eligible"]:continue
            row=dict(row);row["categories"]=[];anchor=row["transition"]
            if row["valid_h1_candidate"]:
                if row["motion_class"]=="moving":row["categories"].append("moving")
                if row["rotation_abs_deg"]>=1:row["categories"].append("rotation")
                if row["near_static_contact"]:row["categories"].append("near_static_contact")
            if not row["valid_transition"] or (anchor<len(steps) and _failed(steps[anchor])):row["categories"].append("failure")
            pool.append(row)
    def ordered(values,category):return sorted(values,key=lambda r:(not r["likely_wall_blocked_proxy"] if category=="near_static_contact" else False,r["trajectory_id"],r["transition"]))
    selected=[];missing=[]
    for index,letter in enumerate(NEW_LETTERS):
        desired=CATEGORIES[index%4];subset=[row for row in pool if row["letter"]==letter]
        matched=[row for row in subset if desired in row["categories"]];actual=desired
        if not matched:
            actual=next((name for name in CATEGORIES if any(name in row["categories"] for row in subset)),"unclassified")
            matched=[row for row in subset if actual in row["categories"]] if actual!="unclassified" else subset
        if not matched:missing.append(dict(letter=letter,desired_category=desired,reason="no_recorded_H1_anchor"));continue
        chosen=dict(ordered(matched,actual)[0],desired_category=desired,selected_category=actual,
                    desired_category_unavailable=actual!=desired,selection_role="new_letter")
        selected.append(chosen)
    for category in CATEGORIES:
        if any(row["selected_category"]==category for row in selected):continue
        matches=[row for row in pool if category in row["categories"] and row["row_id"] not in {r["row_id"] for r in selected}]
        if matches:selected.append(dict(ordered(matches,category)[0],desired_category=category,selected_category=category,
            desired_category_unavailable=False,selection_role="category_extra"))
        else:missing.append(dict(desired_category=category,reason="no_unselected_recorded_support"))
    return dict(rules=dict(EXAMPLE_RULES),cases=selected[:16],missing=missing,selection_uses_model_outputs=False)


def render_examples(records,physics_root,output,*,selection=None):
    """Bounded sheets/GIFs from consecutive saved states; never calls step()."""
    import numpy as np
    from PIL import Image,ImageDraw
    from tdyssey.native_rgb_env import make_env,CONTROLLER
    selection=select_examples(records) if selection is None else selection
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lookup={record["entry"]["trajectory_id"]:record for record in records};envs={};sheets={};evidence=[];gif_categories=set()
    for index,case in enumerate(selection["cases"]):
        record=lookup[case["trajectory_id"]];n=len(record["arrays"]["actions"]);count=min(8,n+1)
        start=max(0,min(case["transition"]-3,n+1-count));frames=list(range(start,start+count));letter=case["letter"]
        if letter not in envs:envs[letter]=make_env(physics_root,letter,CONTROLLER);envs[letter].reset_scene()
        sheet_index=index//8;row=index%8;remaining=min(8,len(selection["cases"])-sheet_index*8)
        if sheet_index not in sheets:sheets[sheet_index]=Image.new("RGB",(1024,remaining*160),"white")
        canvas=sheets[sheet_index];draw=ImageDraw.Draw(canvas)
        draw.text((4,row*160),f"{letter} {case['requested_spatial_class']} t={case['transition']} {case['selected_category']} desired={case['desired_category']}",fill="black")
        animation=[]
        for column,frame in enumerate(frames):
            envs[letter].set_render_state(record["arrays"]["states"][frame]);pixels=np.moveaxis(envs[letter].render_clean(),0,-1)
            tile=Image.fromarray(np.asarray(pixels,dtype=np.uint8)).resize((128,128),Image.Resampling.NEAREST)
            canvas.paste(tile,(column*128,row*160+28));draw.text((column*128+4,row*160+14),f"frame {frame}",fill="black")
            gif=Image.new("RGB",(256,280),"white");ImageDraw.Draw(gif).text((4,2),f"{letter} {case['selected_category']} frame {frame}",fill="black")
            gif.paste(tile.resize((256,256),Image.Resampling.NEAREST),(0,24));animation.append(gif)
        category=case["selected_category"];gif_file=None
        if category in CATEGORIES and category not in gif_categories:
            gif_file=f"example-{category}.gif";animation[0].save(output/gif_file,save_all=True,append_images=animation[1:],duration=100,loop=0)
            gif_categories.add(category)
        evidence.append(dict(letter=letter,trajectory_id=case["trajectory_id"],anchor=case["transition"],frames=frames,
            desired_category=case["desired_category"],selected_category=category,desired_category_unavailable=case["desired_category_unavailable"],
            raw_rgb_unmodified=True,gif=gif_file,sheet=f"examples-{sheet_index+1}.png"))
    for index,canvas in sheets.items():canvas.save(output/f"examples-{index+1}.png")
    result=dict(selection_rules=dict(EXAMPLE_RULES),sequences=evidence,missing=selection["missing"],physics_controls_executed=0,model_calls=0)
    _write(output/"examples.json",result);return result


def _markdown(report):
    lines=["# 阶段2A正式TRAIN记录审计","","仅保存的真实状态/动作/接触；0新增物理控制、0模型调用、0训练索引抽样。", "",
        f"计划162槽/10368控制；记录{report['recorded_slots']}槽/{report['recorded_controls']}控制。有效H1={report['horizons']['1']['valid_candidates']}/计划9882；有效H4={report['horizons']['4']['valid_candidates']}/计划9396。", "",
        "H1为0.1s，原40%moving/45%near_static目标保持。H4仅0.4s真实端点/路径审计，不改变H1门槛或训练目标。", "",
        "| 字母 | raw控制 | H1有效 | H4有效 | H1 moving% | H1 near_static% | H1门槛 |", "|---|---:|---:|---:|---:|---:|---|"]
    for letter,value in report["per_letter"].items():
        gate=value["h1_gate"];m=gate["moving_fraction"];s=gate["near_static_fraction"]
        lines.append(f"| {letter} | {value['raw']['total']} | {value['h1']['total']} | {value['h4']['valid_candidates']} | {None if m is None else round(m*100,2)} | {None if s is None else round(s*100,2)} | {gate['passed']} |")
    lines.extend(["","排除原因、四层独特窗口/独立轨迹/重复倍率、无接触轨迹、四个有符号墙间隙与双墙交叉见JSON；原始与H1详细分布见h1/motion-statistics.json。缺失窗口保留计划分母，静止/失败不删除。","",
        "此报告不证明模型已学会运动，不授权阶段2B–6；即使某字母不达，也不自动补采、换seed或改阈值。"])
    return "\n".join(lines)+"\n"


def build_report(data_root,output,*,planned_requests=None,physics_root=None,examples=False):
    if sha(stats.__file__)!=FROZEN_STATS_SHA256:raise RuntimeError("Frozen H1 statistics source changed")
    data_root,output=Path(data_root),Path(output)
    manifest=json.loads((data_root/"manifest.json").read_text(encoding="utf-8"))
    if any(entry.get("split")!="train" for entry in manifest["entries"]):raise ValueError("No probe/validation/test data allowed")
    records=stats.read_records(data_root,manifest);output.mkdir(parents=True,exist_ok=False)
    core=stats.summarize_records(records,output=output/"h1",expected_letters=TRAIN_LETTERS)
    report=analyze_records(records,planned_requests=planned_requests,core=core)
    report.update(source_sha256=sha(__file__),manifest_sha256=sha(data_root/"manifest.json"),
        frozen_requests_sha256=None if planned_requests is None else hashlib.sha256(json.dumps(planned_requests,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest())
    _write(output/"stage2a-report.json",report)
    with (output/"stage2a-report.md").open("x",encoding="utf-8") as stream:stream.write(_markdown(report))
    selection=select_examples(records);_write(output/"example-selection.json",selection)
    if examples:
        if physics_root is None:raise ValueError("Render-only examples require physics source root")
        render_examples(records,physics_root,output/"examples",selection=selection)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--requests",required=True)
    parser.add_argument("--physics-root");parser.add_argument("--examples",action="store_true")
    args=parser.parse_args(argv)
    if args.examples and not args.physics_root:parser.error("--examples requires --physics-root")
    requests=json.loads(Path(args.requests).read_text(encoding="utf-8"))
    result=build_report(args.data,args.output,planned_requests=requests,physics_root=args.physics_root,examples=args.examples)
    print(json.dumps({key:result[key] for key in ("recorded_slots","recorded_controls","complete_fixed_batch","all_h1_letter_gates_passed")}))
    return 0


if __name__=="__main__":raise SystemExit(main())
