"""Three-cohort formal TRAIN report and hash-bound, reference-only merged view."""

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts import native_rgb_stage2a_report as a
from tdyssey import native_rgb_motion_statistics as stats
from tdyssey.native_rgb_data import TRAIN_LETTERS

COHORTS={"2A":("stage2a/data",9),"2B":("stage2b/data",13)}
LEGACY_SELECTOR="motion-room-actual-hit-prior-v1"
NEW_SELECTOR="motion-room-planned-attempts-v2"
EXAMPLE_PLAN=tuple(zip(("M","M","O","O","P","P","V","X"),
    ("moving","failure","moving","failure","rotation","failure","near_static_contact","rotation")))
EXAMPLE_RULES=dict(plan=[dict(letter=l,category=c) for l,c in EXAMPLE_PLAN],
    source="new2B only",order="earliest trajectory_id then anchor; static-contact prefers wall-blocked proxy",
    first_M_moving="prefer requested corner if supported; missing corner support explicitly disclosed",
    fallback="desired category absent: moving,rotation,near_static_contact,failure,then earliest recorded H1 anchor",
    max_cases=8,max_sheets=1,max_gifs=4,max_consecutive_saved_frames=8)


def _json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe(root,relative,prefix):
    path=Path(relative)
    if path.is_absolute():raise ValueError("Absolute entry path is forbidden")
    resolved=(root/path).resolve()
    if not resolved.is_relative_to((root/prefix).resolve()):raise ValueError("Entry path escapes its formal cohort")
    if not resolved.is_relative_to(root.resolve()):raise ValueError("Entry path escapes supervised root")
    return resolved


def _selector(metadata,cohort):
    value=metadata.get("scene",{}).get("selection_version") or metadata.get("selector_version")
    if cohort=="2A":return value or LEGACY_SELECTOR
    if value!=NEW_SELECTOR:raise ValueError("2B needs explicit planned-attempts selector version")
    return value


def _source_identity(root,cohort,manifest):
    digest=manifest.get("source_identity_sha256")
    if digest is not None:
        relative=COHORTS[cohort][0].split("/")[0]+"/source-identity.json"
        if a.sha(_safe(root,relative,relative.split("/")[0]))!=digest:raise ValueError("Original cohort source identity file changed")
    return digest


def _verify_entry(root,entry,cohort):
    prefix,_=COHORTS[cohort]
    if entry.get("split")!="train" or entry.get("stage")!=cohort or entry.get("shape") not in TRAIN_LETTERS:
        raise ValueError("Merged view accepts only formal2A/2B TRAIN, no probes/validation/unseen")
    for key,hash_key in (("file","sha256"),("metadata_file","metadata_sha256")):
        path=_safe(root,entry[key],prefix)
        if not entry.get(hash_key) or a.sha(path)!=entry[hash_key]:raise ValueError("Original NPZ/metadata hash mismatch")
    metadata=_json(_safe(root,entry["metadata_file"],prefix))
    if entry.get("selector_version")!=_selector(metadata,cohort):raise ValueError("Selector version differs from original metadata")
    return metadata


def make_merged_manifest(supervised_root):
    """Read and verify originals. No files are copied or changed here."""
    root=Path(supervised_root).resolve();entries=[];sources={};headers=[]
    for cohort,(prefix,_) in COHORTS.items():
        manifest_path=_safe(root,prefix+"/manifest.json",prefix);manifest=_json(manifest_path);headers.append(manifest)
        if manifest.get("split")!="train" or manifest.get("stage")!=cohort:raise ValueError("Source manifest must be correct formal TRAIN cohort")
        sources[cohort]=dict(file=prefix+"/manifest.json",sha256=a.sha(manifest_path),
            source_identity_sha256=_source_identity(root,cohort,manifest))
        for original in manifest["entries"]:
            if original.get("split")!="train" or original.get("stage")!=cohort or original.get("shape") not in TRAIN_LETTERS:
                raise ValueError("Merged view accepts only formal2A/2B TRAIN, no probes/validation/unseen")
            entry=dict(original,cohort=cohort,original_file=original["file"],original_metadata_file=original["metadata_file"])
            entry["file"]=(Path(prefix)/original["file"]).as_posix()
            entry["metadata_file"]=(Path(prefix)/original["metadata_file"]).as_posix()
            metadata=_json(_safe(root,entry["metadata_file"],prefix));entry["selector_version"]=_selector(metadata,cohort)
            _verify_entry(root,entry,cohort);entries.append(entry)
    for key in ("state_columns","source_root","controller","protocol_sha256"):
        if key not in headers[0] or any(header.get(key)!=headers[0][key] for header in headers[1:]):
            raise ValueError("Cohort physical/state header differs: "+key)
    for key in ("trajectory_id","seed","initial_scene_hash"):
        values=[entry.get(key) for entry in entries]
        if any(value is None for value in values) or len(values)!=len(set(values)):raise ValueError("Missing/duplicate formal identity: "+key)
    return dict(kind="native-rgb-formal-train-reference-view.v1",base_root=str(root),split="train",stage="2A+2B",
        **{key:headers[0][key] for key in ("state_columns","source_root","controller","protocol_sha256")},
        entries=entries,source_manifests=sources,planned_slots=396,recorded_slots=len(entries),
        planned_controls=25344,actual_controls=sum(int(entry["controls"]) for entry in entries),
        selector_versions=dict(Counter(entry["selector_version"] for entry in entries)),
        reference_only=True,npz_files_copied=0,training_consumption_authorized=False,
        parent_release_required_before_training=True,no_probe_or_validation_entries=True)


def read_merged_records(manifest_path,supervised_root):
    """Explicit base-root adapter; no NativeRGBDataset construction or index draw."""
    root=Path(supervised_root).resolve();manifest=_json(manifest_path)
    if manifest.get("kind")!="native-rgb-formal-train-reference-view.v1" or Path(manifest["base_root"]).resolve()!=root:
        raise ValueError("Merged base root/kind differs")
    originals={}
    for cohort,(prefix,_) in COHORTS.items():
        source=manifest["source_manifests"][cohort]
        if source["file"]!=prefix+"/manifest.json" or a.sha(_safe(root,source["file"],prefix))!=source["sha256"]:
            raise ValueError("Original cohort manifest changed")
        source_manifest=_json(root/source["file"])
        if source_manifest.get("split")!="train" or source_manifest.get("stage")!=cohort:
            raise ValueError("Original manifest formal cohort changed")
        if source.get("source_identity_sha256")!=_source_identity(root,cohort,source_manifest):
            raise ValueError("Merged source identity differs from original manifest")
        for key in ("state_columns","source_root","controller","protocol_sha256"):
            if manifest.get(key)!=source_manifest.get(key):raise ValueError("Merged physical/state header differs from source")
        originals[cohort]={entry["trajectory_id"]:entry for entry in source_manifest["entries"]}
    seen=set()
    for entry in manifest["entries"]:
        cohort=entry.get("cohort")
        if cohort not in COHORTS:raise ValueError("Unknown merged cohort")
        original=originals[cohort].get(entry["trajectory_id"])
        if original is None or entry["trajectory_id"] in seen:raise ValueError("Unknown/duplicate merged trajectory")
        seen.add(entry["trajectory_id"])
        for key in ("sha256","metadata_sha256","shape","seed","initial_scene_hash","split","stage","controls"):
            if entry.get(key)!=original.get(key):raise ValueError("Merged identity differs from source entry: "+key)
        prefix=COHORTS[cohort][0]
        if entry["file"]!=(Path(prefix)/original["file"]).as_posix() or entry["metadata_file"]!=(Path(prefix)/original["metadata_file"]).as_posix():
            raise ValueError("Merged paths differ from source entries")
        _verify_entry(root,entry,cohort)
    if seen!={identifier for values in originals.values() for identifier in values}:raise ValueError("Merged view silently omitted original TRAIN records")
    for key in ("seed","initial_scene_hash"):
        values=[entry.get(key) for entry in manifest["entries"]]
        if any(value is None for value in values) or len(set(values))!=len(values):raise ValueError("Duplicate merged scene/seed identity")
    return stats.read_records(root,manifest)


def _support(rows):
    rows=list(rows);value=a._support(rows)
    for name,subset in value["strata"].items():
        selected=[row for row in rows if row["index_stratum"]==name]
        subset["requested_space_windows"]={location:sum(r["requested_spatial_class"]==location for r in selected) for location in stats.SPATIAL_CLASSES}
        subset["requested_space_independent_trajectories"]={location:len({r["trajectory_id"] for r in selected if r["requested_spatial_class"]==location}) for location in stats.SPATIAL_CLASSES}
    return value


def cohort_report(records,name,per_letter_planned):
    records=list(records);core=stats.summarize_records(records,expected_letters=TRAIN_LETTERS);rows=core["rows"]
    h1=[row for row in rows if row["valid_h1_candidate"]];h4=[a.horizon_audit(record,4) for record in records]
    per_letter={}
    for letter in TRAIN_LETTERS:
        positions=[i for i,record in enumerate(records) if record["entry"]["shape"]==letter]
        per_letter[letter]=dict(planned_trajectories=per_letter_planned,recorded_trajectories=len(positions),
            planned_controls=per_letter_planned*64,planned_h1=per_letter_planned*61,planned_h4=per_letter_planned*58,
            all_real=core["per_letter"][letter]["all_real"],h1=core["per_letter"][letter]["valid_h1"],
            h1_gate=core["per_letter_gate"][letter],h4=a._combine_horizons([h4[i] for i in positions],4),
            strata_support=_support([row for row in h1 if row["letter"]==letter]))
    m_scenes={}
    for location in stats.SPATIAL_CLASSES:
        subset=[r for r in rows if r["letter"]=="M" and r["requested_spatial_class"]==location]
        selected=[r for r in subset if r["valid_h1_candidate"]]
        m_scenes[location]=dict(all_real=a._counts(subset),h1=a._counts(selected),
            trajectories=[dict(trajectory_id=record["entry"]["trajectory_id"],seed=record["entry"]["seed"],
                requested_gap_world_px=record["entry"].get("gap_requested_world_px"),
                h1=a._counts([r for r in selected if r["trajectory_id"]==record["entry"]["trajectory_id"]]))
                for record in records if record["entry"]["shape"]=="M" and record["entry"]["location"]==location])
    planned=18*per_letter_planned
    return dict(cohort=name,planned_slots=planned,recorded_slots=len(records),planned_controls=planned*64,
        actual_controls=len(rows),missing_planned_controls=max(0,planned*64-len(rows)),
        planned_h1=planned*61,actual_h1=len(h1),missing_unrecorded_h1=max(0,planned*61-sum(max(0,len(r["arrays"]["actions"])-3) for r in records)),
        planned_h4=planned*58,h4=a._combine_horizons(h4,4),
        all_real=core["all_real_transitions"],h1=core["valid_h1_candidates"],per_letter=per_letter,
        all_h1_letter_gates_passed=core["all_expected_letters_raw_gate_passed"],strata_support=_support(h1),M_scenes=m_scenes,
        initial_scene_distribution=core["initial_scene_distribution"],
        spatial_groups={key:core["groups"][key] for key in ("requested_spatial_class","actual_spatial_class_before","gap_band_before","requested_corner_gap_band_before","boundary_behavior")},
        four_signed_gaps={side:{f"{wall}_gap_{when}":stats.continuous_distribution(r[f"{wall}_gap_{when}"] for r in selected)
            for wall in stats.WALLS for when in ("before","after")} for side,selected in (("all_real",rows),("h1",h1))},
        requested_corner_two_gap_cross={side:a._corner_cross(selected) for side,selected in (("all_real",rows),("h1",h1))},
        zero_contact_trajectories=[record["entry"]["trajectory_id"] for record in records if len(record["arrays"]["actions"]) and
            not (record["arrays"]["contact"]>0).any()],
        initial_target_arc_counts={letter:dict(Counter(str(r["entry"].get("selected_initial_arc")) for r in records if r["entry"]["shape"]==letter)) for letter in TRAIN_LETTERS},
        initial_target_scope="Independent initial plans only, never actual hits or geometric reachability proof",
        h4_scope=".4s recorded endpoint/path audit only, not H1 gate or training objective",index_draws=0,model_calls=0,physics_controls=0)


def select_examples(records):
    pool=[]
    for record in records:
        if record["entry"].get("stage")!="2B":raise ValueError("Examples must be new2B only")
        steps=record.get("steps",[])
        for row in stats.transition_rows(record["entry"],record["arrays"],steps):
            if not row["h1_anchor_eligible"]:continue
            row=dict(row);row["categories"]=[]
            if row["valid_h1_candidate"]:
                if row["motion_class"]=="moving":row["categories"].append("moving")
                if row["rotation_abs_deg"]>=1:row["categories"].append("rotation")
                if row["near_static_contact"]:row["categories"].append("near_static_contact")
            t=row["transition"]
            if not row["valid_transition"] or (t<len(steps) and a._failed(steps[t])):row["categories"].append("failure")
            pool.append(row)
    chosen=[];missing=[]
    for index,(letter,desired) in enumerate(EXAMPLE_PLAN):
        values=[r for r in pool if r["letter"]==letter and r["row_id"] not in {c["row_id"] for c in chosen}]
        matching=[r for r in values if desired in r["categories"]];corner_missing=False
        if index==0:
            corner=[r for r in matching if r["requested_spatial_class"] in ("left-top","right-top","left-bottom","right-bottom")]
            corner_missing=not corner
            if corner:matching=corner
        actual=desired
        if not matching:
            actual=next((name for name in a.CATEGORIES if any(name in r["categories"] for r in values)),"unclassified")
            matching=[r for r in values if actual in r["categories"]] if actual!="unclassified" else values
        if not matching:missing.append(dict(letter=letter,desired_category=desired,reason="no_unselected_recorded_H1_anchor"));continue
        row=sorted(matching,key=lambda r:(not r["likely_wall_blocked_proxy"] if actual=="near_static_contact" else False,r["trajectory_id"],r["transition"]))[0]
        chosen.append(dict(row,desired_category=desired,selected_category=actual,desired_category_unavailable=actual!=desired,
            desired_corner_moving_unavailable=corner_missing,selection_role="frozen2Bcase"))
    return dict(rules=dict(EXAMPLE_RULES),cases=chosen,missing=missing,selection_uses_model_outputs=False)


def render_examples(records,physics_root,output,selection):
    """One bounded sheet and four GIFs; clean RGB from saved states only."""
    import numpy as np
    from PIL import Image,ImageDraw
    from tdyssey.native_rgb_env import make_env,CONTROLLER
    if len(selection["cases"])>8:raise ValueError("Stage2B examples cap is8")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lookup={r["entry"]["trajectory_id"]:r for r in records};envs={};evidence=[];gif_categories=set()
    canvas=Image.new("RGB",(1024,max(1,len(selection["cases"]))*160),"white");draw=ImageDraw.Draw(canvas)
    for index,case in enumerate(selection["cases"]):
        record=lookup[case["trajectory_id"]];n=len(record["arrays"]["actions"]);count=min(8,n+1)
        start=max(0,min(case["transition"]-3,n+1-count));frames=list(range(start,start+count));letter=case["letter"]
        if letter not in envs:envs[letter]=make_env(physics_root,letter,CONTROLLER);envs[letter].reset_scene()
        draw.text((4,index*160),f"2B {letter} {case['requested_spatial_class']} t={case['transition']} {case['selected_category']}",fill="black")
        animation=[]
        for column,frame in enumerate(frames):
            envs[letter].set_render_state(record["arrays"]["states"][frame]);rgb=np.moveaxis(envs[letter].render_clean(),0,-1)
            tile=Image.fromarray(np.asarray(rgb,dtype=np.uint8)).resize((128,128),Image.Resampling.NEAREST)
            canvas.paste(tile,(column*128,index*160+28));draw.text((column*128+4,index*160+14),f"frame {frame}",fill="black")
            gif=Image.new("RGB",(256,280),"white");ImageDraw.Draw(gif).text((4,2),f"2B {letter} {case['selected_category']} frame {frame}",fill="black")
            gif.paste(tile.resize((256,256),Image.Resampling.NEAREST),(0,24));animation.append(gif)
        category=case["selected_category"];gif_file=None
        if category in a.CATEGORIES and category not in gif_categories:
            gif_file=f"example-{category}.gif";animation[0].save(output/gif_file,save_all=True,append_images=animation[1:],duration=100,loop=0)
            gif_categories.add(category)
        evidence.append(dict(letter=letter,trajectory_id=case["trajectory_id"],anchor=case["transition"],frames=frames,
            desired_category=case["desired_category"],selected_category=category,desired_category_unavailable=case["desired_category_unavailable"],
            desired_corner_moving_unavailable=case["desired_corner_moving_unavailable"],gif=gif_file,raw_rgb_unmodified=True))
    if selection["cases"]:canvas.save(output/"examples-1.png")
    result=dict(selection_rules=dict(EXAMPLE_RULES),sequences=evidence,missing=selection["missing"],physics_controls_executed=0,model_calls=0)
    a._write(output/"examples.json",result);return result


def _verify_requests(requests,records):
    values=requests["requests"] if isinstance(requests,dict) else requests
    if len(values)!=234 or Counter(row["letter"] for row in values)!={letter:13 for letter in TRAIN_LETTERS}:
        raise ValueError("2B plan must contain234requests,13perTRAINletter")
    if Counter(row.get("spatial_class",row.get("location")) for row in values)!={location:26 for location in stats.SPATIAL_CLASSES}:
        raise ValueError("2B global spatial quota must be26each")
    lookup={row["request_id"]:row for row in values}
    if len(lookup)!=234 or any(row.get("split")!="train" or row.get("controls")!=64 for row in values):raise ValueError("Invalid frozen TRAIN request identity")
    seen=set()
    for record in records:
        entry=record["entry"];request=lookup.get(entry.get("request_id"))
        if request is None or entry["request_id"] in seen:raise ValueError("Unexpected/duplicate executed request")
        seen.add(entry["request_id"])
        if entry["shape"]!=request["letter"] or entry["location"]!=request.get("spatial_class",request.get("location")) or entry["seed"]!=request["seed"]:
            raise ValueError("Executed TRAIN entry differs from frozen request")
    return sorted(set(lookup)-seen)


def _markdown(report):
    lines=["# 阶段2B三cohort报告","","全部正式TRAIN，2A原批、2B新批与只读合并视图分列；0模型/物理步/训练索引抽样。", "",
        "| cohort | 实际/计划控制 | 有效/计划H1 | 有效/计划H4 | 每字母H1目标全部通过 |", "|---|---:|---:|---:|---|"]
    for name,v in report["cohorts"].items():lines.append(f"| {name} | {v['actual_controls']}/{v['planned_controls']} | {v['actual_h1']}/{v['planned_h1']} | {v['h4']['valid_candidates']}/{v['planned_h4']} | {v['all_h1_letter_gates_passed']} |")
    lines.extend(["","| 字母 | 2A moving/static% | 2B moving/static% | 合并moving/static% | 合并通过 |", "|---|---:|---:|---:|---|"])
    for letter in TRAIN_LETTERS:
        cells=[]
        for cohort in report["cohorts"].values():
            gate=cohort["per_letter"][letter]["h1_gate"];m,s=gate["moving_fraction"],gate["near_static_fraction"]
            cells.append(f"{None if m is None else round(m*100,2)}/{None if s is None else round(s*100,2)}")
        passed=report["cohorts"]["merged"]["per_letter"][letter]["h1_gate"]["passed"]
        lines.append(f"| {letter} | {' | '.join(cells)} | {passed} |")
    lines.extend(["","M困难空间及每条独立轨迹在各cohort的M_scenes中分开保留，不能用容易场景均值覆盖原批失败。四层独特窗口、独立轨迹与空间支持、重复倍率和缺失分母见JSON。", "",
        "merged-train-manifest.json仅引用原NPZ和metadata/hash，base_root为监督根目录，未复制数据或覆盖2A。该视图不授权训练；H4仅0.4s真值审计，不改变H1目标。后续验证/训练仍须父指令。"])
    return "\n".join(lines)+"\n"


def build_report(supervised_root,output,*,requests_2b,physics_root=None,examples=False):
    if a.sha(stats.__file__)!=a.FROZEN_STATS_SHA256:raise RuntimeError("Frozen H1 statistics changed")
    root=Path(supervised_root).resolve();output=Path(output).resolve()
    if not output.is_relative_to((root/"stage2b").resolve()):raise ValueError("New reports must stay inside stage2b")
    manifest=make_merged_manifest(root);output.mkdir(parents=True,exist_ok=False)
    manifest_path=output/"merged-train-manifest.json";a._write(manifest_path,manifest)
    records=read_merged_records(manifest_path,root);groups={cohort:[r for r in records if r["entry"]["cohort"]==cohort] for cohort in COHORTS}
    missing=_verify_requests(requests_2b,groups["2B"])
    report=dict(kind="native-rgb-stage2b-three-cohort-report.v1",cohorts={
        "2A":cohort_report(groups["2A"],"2A",9),"2B":cohort_report(groups["2B"],"2B",13),
        "merged":cohort_report(records,"merged",22)},unexecuted_2b_request_ids=missing,
        merged_manifest="merged-train-manifest.json",merged_manifest_sha256=a.sha(manifest_path),
        source_sha256=a.sha(__file__),stage2a_report_source_sha256=a.sha(a.__file__),statistics_source_sha256=a.sha(stats.__file__),
        frozen_requests_sha256=hashlib.sha256(json.dumps(requests_2b,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
        example_rules=dict(EXAMPLE_RULES),physics_controls_executed=0,model_calls=0,optimizer_updates=0,index_draws=0,
        no_probe_validation_old_native_entries=True,parent_release_required_before_training=True)
    a._write(output/"stage2b-report.json",report)
    with (output/"stage2b-report.md").open("x",encoding="utf-8") as stream:stream.write(_markdown(report))
    selection=select_examples(groups["2B"]);a._write(output/"example-selection.json",selection)
    if examples:
        if physics_root is None:raise ValueError("Saved-state examples require physics source")
        render_examples(groups["2B"],physics_root,output/"examples",selection)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--requests",required=True)
    parser.add_argument("--physics-root");parser.add_argument("--examples",action="store_true");args=parser.parse_args(argv)
    if args.examples and not args.physics_root:parser.error("--examples requires --physics-root")
    report=build_report(args.root,args.output,requests_2b=_json(args.requests),physics_root=args.physics_root,examples=args.examples)
    print(json.dumps({name:dict(raw=v["actual_controls"],h1=v["actual_h1"],h4=v["h4"]["valid_candidates"],all_letter_gates=v["all_h1_letter_gates_passed"]) for name,v in report["cohorts"].items()}));return 0


if __name__=="__main__":raise SystemExit(main())
