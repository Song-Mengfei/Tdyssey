"""Independent seen_val truth statistics, frozen panels and bounded saved frames."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tdyssey import native_rgb_motion_statistics as stats
from scripts.native_rgb_stage2a_report import horizon_audit,_combine_horizons
from scripts.native_rgb_stage2c_indices import (DEFAULT_SPEC,LETTERS,PROTOCOL,build_indices,
    load_spec,read_json,sha,write_json)

FROZEN_STATS_SHA256="bf5641fd269a69d02db4332987da4794a0df5e2fd47cbcefb9766a276dfc2c49"


def select_examples(records,rules):
    """Earliest same-letter valid H1; missing desired category is disclosed."""
    cases=[];missing=[]
    for request in rules["cases"]:
        letter,desired=request["letter"],request["desired_category"]
        pool=[]
        for record in records:
            if record["entry"]["shape"]!=letter:continue
            for row in stats.transition_rows(record["entry"],record["arrays"],record.get("steps")):
                if not row["valid_h1_candidate"]:continue
                conditions=dict(moving=row["motion_class"]=="moving",rotation=row["rotation_abs_deg"]>=1,
                    likely_wall_blocked_proxy=row["likely_wall_blocked_proxy"],near_static_contact=row["near_static_contact"])
                pool.append(dict(row,conditions=conditions))
        pool.sort(key=lambda row:(row["trajectory_id"],row["transition"]))
        matched=[row for row in pool if row["conditions"].get(desired,False)]
        if not pool:
            missing.append(dict(letter=letter,desired_category=desired,reason="no_saved_valid_H1_anchor"));continue
        chosen=matched[0] if matched else pool[0]
        if not matched:missing.append(dict(letter=letter,desired_category=desired,reason="desired_category_unavailable_same_letter_earliest_valid_fallback"))
        cases.append(dict(letter=letter,trajectory_id=chosen["trajectory_id"],anchor=chosen["transition"],
            requested_spatial_class=chosen["requested_spatial_class"],desired_category=desired,
            desired_category_unavailable=not bool(matched),fallback_used=not bool(matched),
            actual_h1_motion_class=chosen["motion_class"],actual_h1_rotation_abs_deg=chosen["rotation_abs_deg"],
            actual_h1_contact=chosen["contact"],actual_likely_wall_blocked_proxy=chosen["likely_wall_blocked_proxy"],
            conditions=chosen["conditions"]))
    return dict(rules=rules,cases=cases[:rules["max_cases"]],missing=missing,model_outputs_used=False,
                fallback_rule="If requested condition absent, earliest valid H1 in the same letter; otherwise explicitly missing")


def render_examples(records,physics_root,output,selection):
    """At most4x8 saved states and2 GIFs; no simulator step method is called."""
    import numpy as np
    from PIL import Image,ImageDraw
    from tdyssey.native_rgb_env import make_env,CONTROLLER
    rules=selection["rules"]
    if (rules["max_cases"]>4 or rules["max_sheet_rows"]>4 or rules["max_gifs"]>2
            or rules["max_consecutive_frames_per_case"]>8):raise ValueError("Bounded frozen examples only")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lookup={record["entry"]["trajectory_id"]:record for record in records};envs={};evidence=[]
    cases=selection["cases"]
    canvas=Image.new("RGB",(1024,max(1,len(cases))*160),"white");draw=ImageDraw.Draw(canvas)
    for index,case in enumerate(cases):
        record=lookup[case["trajectory_id"]];states=record["arrays"]["states"];n=len(states)
        count=min(rules["max_consecutive_frames_per_case"],n);start=max(0,min(case["anchor"]-3,n-count))
        frames=list(range(start,start+count));letter=case["letter"]
        if letter not in envs:
            envs[letter]=make_env(physics_root,letter,CONTROLLER);envs[letter].reset_scene()
        draw.text((4,index*160),f"{letter} {case['requested_spatial_class']} a={case['anchor']} desired={case['desired_category']} fallback={case['fallback_used']}",fill="black")
        animation=[]
        for column,frame in enumerate(frames):
            envs[letter].set_render_state(states[frame]);pixels=np.moveaxis(envs[letter].render_clean(),0,-1)
            tile=Image.fromarray(np.asarray(pixels,dtype=np.uint8)).resize((128,128),Image.Resampling.NEAREST)
            canvas.paste(tile,(column*128,index*160+28));draw.text((column*128+4,index*160+14),f"s{frame}",fill="black")
            gif=Image.new("RGB",(256,280),"white");ImageDraw.Draw(gif).text((4,2),f"{letter} saved s{frame} {case['desired_category']}",fill="black")
            gif.paste(tile.resize((256,256),Image.Resampling.NEAREST),(0,24));animation.append(gif)
        gif_file=None
        if index<rules["max_gifs"] and animation:
            gif_file=f"saved-case-{index+1}-{letter}.gif"
            animation[0].save(output/gif_file,save_all=True,append_images=animation[1:],duration=100,loop=0)
        evidence.append(dict(case,frames=frames,gif=gif_file,sheet="saved-continuous-examples.png",raw_rgb_unmodified=True))
        if envs[letter].control_step!=0:raise AssertionError("Rendering advanced physics")
    canvas.save(output/"saved-continuous-examples.png")
    result=dict(sequences=evidence,missing=selection["missing"],rules=rules,physics_controls=0,model_calls=0,
                qualification="Consecutive raw saved-state RGB, resized only for display; offline state restoration is not a physical trajectory reset")
    write_json(output/"examples.json",result)
    return result


def build_report(data_root,output,*,config_path=ROOT/"configs/native_rgb_supervised_stage2c.json",physics_root=None,examples=False):
    if sha(stats.__file__)!=FROZEN_STATS_SHA256:raise RuntimeError("Frozen statistics source identity differs")
    data_root,output=Path(data_root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError("Preserve old validation reports")
    if (data_root.parent/"COLLECTION_RUNNING.lock").exists():raise RuntimeError("Collection is still running")
    config=read_json(config_path);manifest=read_json(data_root/"manifest.json")
    spec_path=ROOT/config["index_spec"]
    if sha(spec_path)!=config["index_spec_sha256"]:raise ValueError("Pre-physics frozen index policy changed")
    load_spec(spec_path)
    if manifest.get("split")!="seen_val" or manifest.get("protocol_sha256")!=PROTOCOL:raise ValueError("Independent seen_val protocol required")
    for entry in manifest["entries"]:
        if entry.get("split")!="seen_val" or entry["shape"] not in LETTERS:raise ValueError("No train/probe/unseen records")
        if sha(data_root/entry["file"])!=entry["sha256"] or sha(data_root/entry["metadata_file"])!=entry["metadata_sha256"]:
            raise ValueError("Immutable saved input identity differs")
    records=stats.read_records(data_root,manifest)
    output.mkdir(parents=True,exist_ok=False)
    core=stats.summarize_records(records,output=output/"h1",expected_letters=LETTERS)
    audits={horizon:[horizon_audit(record,horizon) for record in records] for horizon in (1,4)}
    per_letter={}
    for letter in LETTERS:
        indices=[i for i,record in enumerate(records) if record["entry"]["shape"]==letter]
        raw=[row for row in core["rows"] if row["letter"]==letter]
        h1=[row for row in raw if row["valid_h1_candidate"]]
        per_letter[letter]=dict(independent_trajectories=len(indices),raw=stats.summarize_rows(raw),valid_h1=stats.summarize_rows(h1),
            h4=_combine_horizons([audits[4][i] for i in indices],4),
            distribution_target_diagnostic=core["per_letter_gate"][letter],
            target_is_diagnostic_only=True,validation_resampling_or_frame_deletion=False)
    panel_indices=build_indices(data_root,output/"indices",spec_path=spec_path)
    selection=select_examples(records,config["examples"])
    write_json(output/"example-selection.json",selection)
    report=dict(kind="independent-seen-val-stage2c-report.v1",split="seen_val",planned_slots=162,planned_controls=10368,
        recorded_slots=len(records),recorded_controls=len(core["rows"]),collection_failure=manifest.get("failure"),
        planned_h1_candidates=9882,planned_h4_candidates=9396,
        raw=stats.summarize_rows(core["rows"]),valid_h1=stats.summarize_rows([row for row in core["rows"] if row["valid_h1_candidate"]]),
        horizons={str(horizon):_combine_horizons(values,horizon) for horizon,values in audits.items()},per_letter=per_letter,
        panel_summaries=panel_indices["panel_summaries"],missing_index_slots=panel_indices["missing_slots"],
        indices_path="indices/indices.json",indices_policy_sha256=sha(spec_path),
        manifest_sha256=sha(data_root/"manifest.json"),source_sha256=sha(__file__),frozen_statistics_sha256=sha(stats.__file__),
        examples_selection=selection,distribution_targets_are_diagnostics_not_validation_acceptance_gates=True,
        no_training_data_or_strategy_changed=True,model_calls=0,optimizer_updates=0,training_index_draws=0,physics_controls=0)
    write_json(output/"seen-val-report.json",report)
    lines=["# Independent seen-letter validation data and frozen indices","",
        f"Recorded {report['recorded_slots']}/162 trajectories, {report['recorded_controls']}/10368 controls; H1={report['horizons']['1']['valid_candidates']}/9882, H4={report['horizons']['4']['valid_candidates']}/9396.",
        "40% moving /45% near-static thresholds describe validation distribution only; no frames are removed and no validation resampling is allowed.","",
        "| Letter | Raw controls | H1 windows | H4 windows | H1 moving | H1 near-static | Diagnostic target |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for letter,value in per_letter.items():
        gate=value["distribution_target_diagnostic"]
        lines.append(f"| {letter} | {value['raw']['total']} | {value['valid_h1']['total']} | {value['h4']['valid_candidates']} | {gate['moving_fraction']} | {gate['near_static_fraction']} | {gate['passed']} |")
    lines += ["","Natural/conditional/combined/development panel distributions, missing slots and temporal correlation are in indices/INDEX_REPORT.md and indices.json.",
        "All raw contact×motion, actual spaces, four signed wall gaps and dual-wall requested-corner gaps are retained in h1/real-transition-statistics.csv and motion-statistics.json. H4 endpoint/path motion remains separate from H1 labels.",
        "No optimizer, denoiser, training iterator or model prediction was run. Saved-state example rendering, when requested, performs zero physics controls."]
    with (output/"SEEN_VAL_REPORT.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    if examples:
        if physics_root is None:raise ValueError("Saved-state-only rendering requires physics source root")
        render_examples(records,physics_root,output/"examples",selection)
    write_json(output/"report-files.json",[dict(file=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.rglob("*")) if p.is_file()])
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--config",default=str(ROOT/"configs/native_rgb_supervised_stage2c.json"))
    parser.add_argument("--physics-root");parser.add_argument("--examples",action="store_true")
    args=parser.parse_args(argv)
    result=build_report(args.data,args.output,config_path=args.config,physics_root=args.physics_root,examples=args.examples)
    print(json.dumps({key:result[key] for key in ("recorded_slots","recorded_controls","panel_summaries")}))
    return 0


if __name__=="__main__":raise SystemExit(main())
