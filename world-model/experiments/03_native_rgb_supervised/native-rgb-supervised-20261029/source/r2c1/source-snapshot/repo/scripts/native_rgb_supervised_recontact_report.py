"""Offline stage1B paired recontact report. Never steps physics or calls a model."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdyssey.native_rgb_motion_statistics import read_records, summarize_rows, transition_rows

OLD_RECOVERY_PHASES={"contact_recovery_route","recontact","blocked_escape_recontact"}
PROTOCOL=dict(kind="native-rgb-stage1b-recontact-report.v1",planned_controls_per_slot=64,
    planned_h1_candidates_per_slot=61,control_seconds=.1,post_contact_followup_controls=8,
    sustained_contact_diagnostic_controls=3,pusher_no_displacement_diagnostic_world_px=.05,
    old_episode_definition="contiguous recovery/recontact phases; lost-contact streak is context only; approximate, not new state machine",
    route_or_staging_is_physical_contact=False,model_calls=0,physics_controls_executed=0)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recovery(step):
    before=step.get("recovery",{})
    after=step.get("observed",{}).get("recovery",{})
    return {**(before if isinstance(before,dict) else {}),**(after if isinstance(after,dict) else {})}


def _max_streak(values):
    best=current=0
    for value in values:
        current=current+1 if value else 0;best=max(best,current)
    return best


def _flag(events,name):
    values=[value[name] for value in events if name in value]
    return any(value is True for value in values) if values else None


def _finite(value):
    try:return float(value) if math.isfinite(float(value)) else None
    except (TypeError,ValueError):return None


def recovery_episodes(record,*,version):
    """Latest post-step logger state plus independent recorded contact/motion."""
    import numpy as np
    arrays=record["arrays"];states=np.asarray(arrays["states"],float);actions=np.asarray(arrays["actions"],float)
    rows=transition_rows(record["entry"],arrays,record.get("steps"));n=len(rows)
    steps=record.get("steps",[]);groups=[]
    if version=="new":
        by_id={}
        for t,step in enumerate(steps[:n]):
            event=_recovery(step);identifier=event.get("episode_id")
            if identifier is None:continue
            key=str(identifier)
            if key not in by_id:by_id[key]=[];groups.append((key,by_id[key]))
            by_id[key].append((t,event))
    elif version=="old":
        active=None
        for t in range(n):
            step=steps[t] if t<len(steps) else {}
            if step.get("phase") in OLD_RECOVERY_PHASES:
                if active is None:active=[];groups.append((f"approx-{len(groups)+1}",active))
                active.append((t,{}))
            else:active=None
    else:raise ValueError("Declare old or new episode semantics")
    result=[]
    for position,(identifier,group) in enumerate(groups):
        indices=[t for t,_ in group];events=[event for _,event in group];start,end=indices[0],indices[-1]
        next_start=groups[position+1][1][0][0] if position+1<len(groups) else n
        contact_indices=[t for t in indices if rows[t]["contact"] is True]
        first=contact_indices[0] if contact_indices else None
        follow_end=min(n,first+8,next_start) if first is not None else None
        follow=list(range(first,follow_end)) if first is not None else []
        contact=[rows[t]["contact"] is True for t in indices]
        pusher_displacements=[float(np.linalg.norm(states[t+1,:2]-states[t,:2])) for t in indices]
        target_distances=[float(np.linalg.norm(actions[t]-states[t,:2])) for t in indices]
        latest=events[-1] if events else {}
        reported_first=latest.get("first_real_contact_step")
        reported_max=latest.get("max_consecutive_contact_controls")
        independent_max=_max_streak(rows[t]["contact"] is True if t in indices else False for t in range(start,end+1))
        discrepancies=[]
        if version=="new" and reported_first is not None and reported_first!=first:discrepancies.append("first_real_contact_step")
        if version=="new" and reported_max is not None and reported_max!=independent_max:discrepancies.append("max_consecutive_contact_controls")
        reason=next((event.get("failure_reason") for event in reversed(events) if event.get("failure_reason")),None)
        status=latest.get("status","unlogged_old_approximation" if version=="old" else "unknown")
        result.append(dict(episode_id=identifier,semantics="new_logged_machine" if version=="new" else "old_phase_approximation",
            first_step=start,last_step=end,observed_controls=len(indices),control_seconds=.1*len(indices),
            nonconsecutive_episode_metadata=indices!=list(range(start,end+1)),
            phases=dict(Counter(steps[t].get("phase","unknown") for t in indices if t<len(steps))),
            initial_lost_contact_streak=steps[start].get("lost_contact_streak") if start<len(steps) else None,
            route_completed=_flag(events,"route_completed"),staging_arrived=_flag(events,"staging_arrived"),
            staging_arrival_step=next((event.get("staging_arrival_step") for event in events if event.get("staging_arrival_step") is not None),None),
            first_real_contact_step=first,first_real_contact_seconds_from_episode_start=None if first is None else .1*(first-start+1),
            physically_recontacted=first is not None,contact_controls=sum(contact),moving_controls=sum(rows[t]["motion_class"]=="moving" for t in indices),
            contact_moving_controls=sum(rows[t]["contact"] is True and rows[t]["motion_class"]=="moving" for t in indices),
            max_consecutive_contact_controls=independent_max,sustained_contact_3_controls=independent_max>=3,
            contact_followup_first_step=first,contact_followup_exclusive_end=follow_end,
            contact_followup_controls=len(follow),contact_followup_max_streak=_max_streak(rows[t]["contact"] is True for t in follow),
            post_contact_moving_controls=sum(rows[t]["motion_class"]=="moving" for t in follow if t>first),
            post_contact_contact_moving_controls=sum(rows[t]["contact"] is True and rows[t]["motion_class"]=="moving" for t in follow if t>first),
            actual_pusher_path_world_px=sum(pusher_displacements),
            pusher_displacement_lte_0_05_controls=sum(value<=.05 for value in pusher_displacements),
            actual_target_distance_first_world_px=target_distances[0],actual_target_distance_last_world_px=target_distances[-1],
            absolute_target_path_world_px=sum(float(np.linalg.norm(actions[b]-actions[a])) for a,b in zip(indices,indices[1:])),
            route_remaining_first_world_px=next((_finite(event["route_remaining_world_px"]) for event in events if "route_remaining_world_px" in event),None),
            route_remaining_last_world_px=_finite(latest.get("route_remaining_world_px")),
            reported_no_progress_controls_max=max([int(event.get("no_progress_controls",0)) for event in events],default=0) if version=="new" else None,
            status=status,failure_reason=reason,
            contact_not_observed_in_episode=first is None,end_at_trajectory_boundary=end==n-1,
            logger_discrepancies=discrepancies,
            logged_latest={key:latest.get(key) for key in ("state","stage","status","start_step","end_step","controls",
                "progress_world_px","consecutive_contact_controls","max_consecutive_contact_controls",
                "post_contact_moving_controls","post_contact_contact_moving_controls","alternative_switches")} if version=="new" else None))
    return result


def summarize_record(record,*,version):
    rows=transition_rows(record["entry"],record["arrays"],record.get("steps"))
    universe=[row for row in rows if row["h1_anchor_eligible"]];valid=[row for row in universe if row["valid_h1_candidate"]]
    actual=len(rows)
    return dict(trajectory_id=record["entry"].get("trajectory_id"),letter=record["entry"]["shape"],
        location=record["entry"].get("location"),all_real=summarize_rows(rows),raw_h1_universe=summarize_rows(universe),
        valid_h1=summarize_rows(valid),invalid_history_windows=len(universe)-len(valid),
        planned_controls=64,missing_planned_controls=max(0,64-actual),planned_h1_candidates=61,
        missing_planned_h1_candidates=max(0,61-len(universe)),
        complete_64_controls=actual==64,metadata_controls=len(record.get("steps",[])),
        phase_all_real={phase:summarize_rows([row for row in rows if row["phase"]==phase]) for phase in sorted({row["phase"] for row in rows})},
        recovery_episodes=recovery_episodes(record,version=version))


def _aggregate(records,planned_slots):
    rows=[row for record in records for row in transition_rows(record["entry"],record["arrays"],record.get("steps"))]
    universe=[row for row in rows if row["h1_anchor_eligible"]];valid=[row for row in universe if row["valid_h1_candidate"]]
    return dict(planned_slots=planned_slots,recorded_slots=len(records),planned_controls=64*planned_slots,
        recorded_controls=len(rows),missing_planned_controls=max(0,64*planned_slots-len(rows)),
        planned_h1_candidates=61*planned_slots,missing_planned_h1_candidates=max(0,61*planned_slots-len(universe)),
        all_real=summarize_rows(rows),raw_h1_universe=summarize_rows(universe),valid_h1=summarize_rows(valid),
        invalid_history_windows=len(universe)-len(valid))


def compare_records(baseline_records,variant_records,slots):
    """Explicit frozen slot matching, including missing or failed executions."""
    import numpy as np
    slots=list(slots)
    if len(slots)!=8 or sum(slot.get("role")=="paired" for slot in slots)!=6 or sum(slot.get("role")=="new" for slot in slots)!=2:
        raise ValueError("Stage1B requires exactly six paired and two new frozen slots")
    ids=[str(slot["slot_id"]) for slot in slots];new_ids=[str(slot["new_trajectory_id"]) for slot in slots]
    if len(set(ids))!=8 or len(set(new_ids))!=8:raise ValueError("Duplicate slot or new trajectory ID")
    def lookup(records):
        values={record["entry"]["trajectory_id"]:record for record in records}
        if len(values)!=len(records):raise ValueError("Duplicate recorded trajectory ID")
        return values
    old,new=lookup(baseline_records),lookup(variant_records)
    if set(new)-set(new_ids):raise ValueError("Unexpected variant trajectory outside frozen eight slots")
    paired_old=[];paired_new=[];new_only=[];results=[]
    for slot in slots:
        role=slot["role"];before=old.get(slot.get("baseline_trajectory_id"));after=new.get(slot["new_trajectory_id"])
        if role=="paired" and not slot.get("baseline_trajectory_id"):raise ValueError("Paired slot needs explicit baseline ID")
        if role=="new" and slot.get("baseline_trajectory_id"):raise ValueError("New slot cannot claim a paired baseline")
        match=None
        planned_match=None
        planned_state=slot.get("initial_state",slot.get("scene",{}).get("initial_state"))
        if after is not None and planned_state is not None:
            planned_match=bool(np.array_equal(np.asarray(planned_state,dtype=float),after["arrays"]["states"][0]))
        if before is not None and after is not None:
            match=bool(before["entry"]["shape"]==after["entry"]["shape"] and
                       np.array_equal(before["arrays"]["states"][0],after["arrays"]["states"][0]))
        if role=="paired":
            if before is not None:paired_old.append(before)
            if after is not None:paired_new.append(after)
        elif after is not None:new_only.append(after)
        old_summary=summarize_record(before,version="old") if before is not None else None
        new_summary=summarize_record(after,version="new") if after is not None else None
        delta={}
        if old_summary is not None and new_summary is not None:
            for population in ("all_real","valid_h1"):
                delta[population]={name:new_summary[population]["class_counts"].get(name,0)-old_summary[population]["class_counts"].get(name,0)
                                   for name in ("moving","intermediate","near_static")}
        slot_identity={key:slot.get(key) for key in ("slot_id","new_trajectory_id","baseline_trajectory_id","role",
                                                  "letter","spatial_class","seed","initial_state_sha256")}
        results.append(dict(**slot_identity,baseline_present=before is not None,variant_present=after is not None,
            paired_initial_full_state_exact=match,paired_comparison_valid=bool(role=="paired" and match),
            frozen_initial_full_state_exact=planned_match,
            baseline=old_summary,variant=new_summary,new_minus_baseline_counts=delta,
            delta_is_unmatched_diagnostic=bool(delta and not match)))
    return dict(protocol=dict(PROTOCOL),all_eight_slots_recorded_complete=all(row["variant"] is not None and row["variant"]["complete_64_controls"] for row in results),
        all_six_pair_initial_states_exact=all(row["paired_initial_full_state_exact"] is True for row in results if row["role"]=="paired"),
        all_recorded_frozen_initial_states_exact=bool(variant_records and all(row["frozen_initial_full_state_exact"] is True for row in results if row["variant_present"])),
        slots=results,stage1_original_all54=_aggregate(baseline_records,54),stage1b_all8=_aggregate(variant_records,8),
        paired_six_old=_aggregate(paired_old,6),paired_six_new=_aggregate(paired_new,6),new_two_x=_aggregate(new_only,2),
        scientific_scope="Six paired initial scenes and two predeclared new X scenes, not statistical significance or authorization for later stages",
        no_phase1_and_phase1b_pooled_gate=True,physics_controls_executed_by_report=0,model_calls=0)


def _slots_from_manifest(manifest):
    slots=[]
    for entry in manifest["entries"]:
        baseline=entry.get("baseline_trajectory_id",entry.get("paired_trajectory_id"))
        slots.append(dict(entry,slot_id=entry.get("slot_id",entry["trajectory_id"]),new_trajectory_id=entry["trajectory_id"],
                          baseline_trajectory_id=baseline,role=entry.get("role","paired" if baseline else "new")))
    return slots


def _markdown(report):
    text=["# 阶段1B重接触配对报告","","仅汇总保存的状态、动作、接触与恢复日志；0物理控制、0模型调用。原阶段1与1B分开，两个新X场景单列。", "",
          "| 槽 | 角色 | 原始步 old/new | H1有效窗 old/new | moving old/new | near_static old/new | 初始完整状态相同 |",
          "|---|---|---:|---:|---:|---:|---|"]
    for row in report["slots"]:
        old,new=row["baseline"],row["variant"]
        def counts(key,field):return "/".join(str(value[key].get(field,0)) if value else "missing" for value in (old,new))
        moving="/".join(str(value["valid_h1"]["class_counts"].get("moving",0)) if value else "missing" for value in (old,new))
        static="/".join(str(value["valid_h1"]["class_counts"].get("near_static",0)) if value else "missing" for value in (old,new))
        text.append(f"| {row['slot_id']} | {row['role']} | {counts('all_real','total')} | {counts('valid_h1','total')} | {moving} | {static} | {row['paired_initial_full_state_exact']} |")
    text.extend(["","所有实际64步和有效61窗的接触×运动表、连续分布、各恢复事件的真实首次接触/连续接触/接触后运动/路径距离及失败见同名JSON。缺失或部分槽保留计划分母64/61，不能当成完成或替代采样。", "",
        "新版route_completed/staging_arrived仅表示日志中的路径/准备点事件，不能等同真实接触。真实接触和运动由NPZ独立重算；旧版episode只按连续恢复/重接触phase近似切分，不冒充新版状态机。接触后额外观察最多8控制并在下一episode前截止，和状态机成功条件分列。", "",
        "本报告不合并旧54轨迹稀释1B缺陷，不做显著性声明，也不授权正式采集、训练、模型推理或服务器运行。"])
    text.extend(["","## 恢复事件逐项证据","",
        "old为连续phase近似；new为记录的状态机episode。时间单位秒，first为实际接触的全局控制索引；post moving不含首次接触控制。", "",
        "| 槽/版本/episode | 控制/秒 | route/staging | first contact | 最大连续接触 | post contact moving | 最终状态/失败 |",
        "|---|---:|---|---:|---:|---:|---|"])
    for row in report["slots"]:
        for side,key in (("old","baseline"),("new","variant")):
            summary=row[key]
            if summary is None:continue
            for ep in summary["recovery_episodes"]:
                text.append(f"| {row['slot_id']}/{side}/{ep['episode_id']} | {ep['observed_controls']}/{ep['control_seconds']:.1f} | {ep['route_completed']}/{ep['staging_arrived']} | {ep['first_real_contact_step']} | {ep['max_consecutive_contact_controls']} | {ep['post_contact_contact_moving_controls']} | {ep['status']}/{ep['failure_reason']} |")
    return "\n".join(text)+"\n"


def build_report(baseline_root,variant_root,output,*,slots=None):
    baseline_root,variant_root,output=map(Path,(baseline_root,variant_root,output))
    manifest=json.loads((variant_root/"manifest.json").read_text(encoding="utf-8"))
    slots=_slots_from_manifest(manifest) if slots is None else slots
    report=compare_records(read_records(baseline_root),read_records(variant_root,manifest),slots)
    report["source_sha256"]=sha(__file__)
    report["baseline_manifest_sha256"]=sha(baseline_root/"manifest.json")
    report["variant_manifest_sha256"]=sha(variant_root/"manifest.json")
    report["frozen_slots_sha256"]=hashlib.sha256(json.dumps(slots,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    with (output/"recontact-report.json").open("x",encoding="utf-8") as stream:json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
    with (output/"recontact-report.md").open("x",encoding="utf-8") as stream:stream.write(_markdown(report))
    return report


def render_pairs(baseline_records,variant_records,report,physics_root,output,*,max_pairs=2):
    """At most two pairs, five consecutive saved frames per side, no step()."""
    import numpy as np
    from PIL import Image,ImageDraw
    from tdyssey.native_rgb_env import make_env,CONTROLLER
    if type(max_pairs) is not int or not 1<=max_pairs<=2:raise ValueError("Render one or two bounded pairs")
    old={row["entry"]["trajectory_id"]:row for row in baseline_records};new={row["entry"]["trajectory_id"]:row for row in variant_records}
    selected=[row for row in report["slots"] if row["paired_comparison_valid"]][:max_pairs]
    output=Path(output);output.mkdir(parents=True,exist_ok=True);envs={};evidence=[]
    canvas=Image.new("RGB",(640,max(1,len(selected))*320),"white");draw=ImageDraw.Draw(canvas)
    for pair_index,pair in enumerate(selected):
        for side_index,(side,record) in enumerate((("old",old[pair["baseline_trajectory_id"]]),("new",new[pair["new_trajectory_id"]]))):
            n=len(record["arrays"]["actions"])
            if n<4:continue
            summary=pair["baseline" if side=="old" else "variant"];episodes=summary["recovery_episodes"]
            start=max(0,min((episodes[0]["first_step"]-1) if episodes else 0,n-4));frames=list(range(start,start+5))
            letter=record["entry"]["shape"]
            if letter not in envs:envs[letter]=make_env(physics_root,letter,CONTROLLER);envs[letter].reset_scene()
            y=(pair_index*2+side_index)*160;draw.text((4,y),f"{pair['slot_id']} {side}: consecutive saved frames {frames}",fill="black")
            for column,frame in enumerate(frames):
                envs[letter].set_render_state(record["arrays"]["states"][frame]);pixels=np.moveaxis(envs[letter].render_clean(),0,-1)
                canvas.paste(Image.fromarray(np.asarray(pixels,dtype=np.uint8)).resize((128,128),Image.Resampling.NEAREST),(column*128,y+28))
            evidence.append(dict(slot_id=pair["slot_id"],side=side,frames=frames,source_trajectory=record["entry"]["trajectory_id"]))
    target=output/"recontact-paired-saved-frames.png"
    if target.exists():raise FileExistsError("Preserve existing paired render")
    if selected:canvas.save(target)
    result=dict(sequences=evidence,physics_controls_executed=0,model_calls=0,render_only=True)
    with (output/"recontact-paired-saved-frames.json").open("x",encoding="utf-8") as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write("\n")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-data",required=True);parser.add_argument("--variant-data",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--slots",help="Frozen JSON list or object with slots")
    parser.add_argument("--physics-root");parser.add_argument("--render-pairs",type=int,default=0,choices=(0,1,2))
    args=parser.parse_args(argv)
    if args.render_pairs and not args.physics_root:parser.error("--render-pairs requires --physics-root")
    slots=json.loads(Path(args.slots).read_text(encoding="utf-8")) if args.slots else None
    if isinstance(slots,dict):slots=slots["slots"]
    report=build_report(args.baseline_data,args.variant_data,args.output,slots=slots)
    if args.render_pairs:render_pairs(read_records(args.baseline_data),read_records(args.variant_data),report,args.physics_root,args.output,max_pairs=args.render_pairs)
    print(json.dumps(dict(report=str(Path(args.output)/"recontact-report.json"),all_slots_complete=report["all_eight_slots_recorded_complete"],pair_initial_states_exact=report["all_six_pair_initial_states_exact"])))
    return 0 if report["all_eight_slots_recorded_complete"] and report["all_six_pair_initial_states_exact"] and report["all_recorded_frozen_initial_states_exact"] else 1


if __name__=="__main__":raise SystemExit(main())
