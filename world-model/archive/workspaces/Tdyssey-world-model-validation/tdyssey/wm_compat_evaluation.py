"""M -> T -> L orchestration, with injectable measurement and model factories.

The public worker verifies external execution authority/guardian first. This
module itself never starts a process, simulator, policy or optimizer.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .wm_compat_engine import (SavedBranch, ChunkSink, compact_measurement,
    run_paired_branch, REAL_RECORD_BYTES)
from .wm_compat_transform import transform_for_arm, T1_SHA256


REPORT_SHA256 = "c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795"


def gate_evidence(result, evidence_in="summary.json"):
    """Bind full report evidence without unbounded identity lists in the journal."""
    data={key:value for key,value in result.items() if key!="rows"}
    encoded=json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return dict(passed=result["passed"],report_evidence_sha256=hashlib.sha256(encoded).hexdigest(),
                evidence_in=evidence_in,development_only=True)


def load_saved_inputs(run_dir):
    """Read existing arrays/labels; no environment is constructed or restored."""
    from .wm_interface_diagnostics import build_inventory, load_rgb
    run_dir = Path(run_dir).resolve()
    content = (run_dir/"report.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != REPORT_SHA256:
        raise ValueError("Saved report differs from the frozen input identity")
    report = json.loads(content)
    inventory = build_inventory(report, run_dir)
    real_items = []
    for item in inventory:
        if item["source"] == "forecast":
            continue
        real_items.append(dict(source_key=item["id"], source=item["source"], anchor=item["anchor"],
            candidate=item["candidate"], shape=item["shape"], horizon=item["horizon"],
            rgb=load_rgb(item), goal=item.get("goal"), truth=item.get("truth")))
    expected = {(row["anchor"],row["candidate"]) for row in report["branches"] if row["status"]=="complete"}
    from .wm_compat_budget import canonical_plan
    fixed = {(row["anchor"],row["candidate"]) for row in canonical_plan()["branches"] if row["status"]=="available"}
    if expected != fixed or len(real_items)!=536 or sum(row["rgb"] is not None for row in real_items)!=528:
        raise ValueError("Saved branch/real-image inventory differs from the fixed plan")
    branches = []
    metadata = {(row["anchor"],row["candidate"]):row for row in report["branches"]}
    for planned in canonical_plan()["branches"]:
        if planned["status"] == "missing":
            continue
        anchor,candidate = planned["anchor"],planned["candidate"]
        row=metadata[anchor,candidate]
        if [step["horizon"] for step in row["actual"]] != [1,2,3,4] or any(step.get("absorbing") for step in row["actual"]):
            raise ValueError("Saved physical future is not the four actually executed controls")
        with np.load(run_dir/f"{anchor}-{candidate}-physical.npz",allow_pickle=False) as arrays:
            branch=SavedBranch(anchor,candidate,planned["shape"],
                arrays["history"].copy(),arrays["past_actions"].copy(),arrays["future_actions"].copy(),
                arrays["actual"].copy(),goal=row["initial_state"]["goal"],
                truth=[step["truth"] for step in row["actual"]],latest=row["latest"],
                contact_by_h4=any(step.get("contacts",0)>0 for step in row["actual"]))
        branches.append(branch)
    return dict(branches=branches,real_items=real_items)


def accumulate_transform(aggregate, diag):
    """Bounded totals over M rows, which cover every real AR/TF input frame."""
    arm=diag["arm"]
    row=aggregate.setdefault(arm,dict(t1_spec_sha256=T1_SHA256,rows=0,pixels=0,
        counts={},winning_segment_counts={},accepted_segment_counts={},residual_by_winning_segment={},
        input_range=[1.,0.],output_range=[1.,0.],all_output_finite=True,all_output_in_range=True,
        any_input_mutated=False,residual_rms_rgb_255=None))
    row["rows"]+=1
    row["pixels"]+=diag["pixel_count"]
    for name in ("changed_pixels","accepted_pixels","unmatched_pixels","converted_pixels",
                 "identity_preserved_pixels","unchanged_pixels"):
        if diag.get(name) is not None:
            row["counts"][name]=row["counts"].get(name,0)+diag[name]
    for name in ("input_range","output_range"):
        row[name]=[min(row[name][0],diag[name][0]),max(row[name][1],diag[name][1])]
    row["all_output_finite"] &= diag["output_finite"]
    row["all_output_in_range"] &= diag["output_in_range"]
    row["any_input_mutated"] |= diag["input_mutated"]
    for name in ("winning_segment_counts","accepted_segment_counts"):
        for key,value in diag[name].items():
            row[name][key]=row[name].get(key,0)+value
    residual=diag.get("residual_rms_rgb_255")
    if residual is not None:
        record=row["residual_rms_rgb_255"]
        if record is None:
            record=row["residual_rms_rgb_255"]=dict(min=residual["min"],max=residual["max"],
                pixel_weighted_sum=0.,per_frame_quantile_ranges={},
                quantile_interpretation="ranges of frame quantiles; not pooled pixel quantiles")
        record["min"]=min(record["min"],residual["min"])
        record["max"]=max(record["max"],residual["max"])
        record["pixel_weighted_sum"]+=residual["mean"]*diag["pixel_count"]
        for key in ("p50","p90","p99"):
            prior=record["per_frame_quantile_ranges"].get(key,[residual[key],residual[key]])
            record["per_frame_quantile_ranges"][key]=[min(prior[0],residual[key]),max(prior[1],residual[key])]
    for key,value in diag.get("residual_by_winning_segment",{}).items():
        record=row["residual_by_winning_segment"].setdefault(key,
            dict(winner_pixels=0,accepted_pixels=0,pixel_weighted_sum=0.,max_rms_rgb_255=None))
        record["winner_pixels"]+=value["winner_pixels"]
        record["accepted_pixels"]+=value["accepted_pixels"]
        if value["winner_pixels"]:
            record["pixel_weighted_sum"]+=value["mean_rms_rgb_255"]*value["winner_pixels"]
            record["max_rms_rgb_255"]=max(record["max_rms_rgb_255"] or 0.,value["max_rms_rgb_255"])


def measure_real_items(items, *, measure, budget, sink=None, progress=None, transform_diagnostics=None):
    rows=[]
    for item in items:
        for arm in ("A","B"):
            budget.check()
            row={key:value for key,value in item.items() if key not in ("rgb","goal")}
            row.update(key=item["source_key"]+"/"+arm,arm=arm,mode="REAL",
                       status="missing" if item["rgb"] is None else "completed",measurement=None)
            if item["rgb"] is not None:
                image,diag=transform_for_arm(item["rgb"],arm)
                if transform_diagnostics is not None:
                    accumulate_transform(transform_diagnostics,diag)
                row["measurement"]=compact_measurement(measure(image,item["shape"],item["goal"]))
                row["transform"]={key:diag.get(key) for key in (
                    "input_sha256","output_sha256","changed_pixels","unmatched_pixels")}
            if sink is not None:
                sink.record("real/"+item["source_key"]+"-"+arm+".json",row,maximum=REAL_RECORD_BYTES)
            rows.append(row)
            if progress:
                progress(row)
    return rows


def planned_forecast_rows(plan, rows, branches, *, reason=None):
    """Preserve 2304 planned rows, including 36 missing and all blocked stages."""
    measured={row["key"]:row for row in rows}
    if len(measured)!=len(rows):
        raise ValueError("Duplicate forecast output")
    physical={(row.anchor,row.candidate):row for row in branches}
    result=[]
    for planned in plan["calls"]:
        if planned["key"] in measured:
            result.append(measured[planned["key"]])
        else:
            row=dict(planned,shape=planned["phase"],status="missing" if planned["status"]=="missing" else "not_attempted",
                     reason="original_L01_hold_missing" if planned["status"]=="missing" else reason or "not_reached",
                     measurement=None)
            branch=physical.get((row["anchor"],row["candidate"]))
            row.update(truth=branch.truth[row["horizon"]-1] if branch and branch.truth else None,
                       contact_by_h4=branch.contact_by_h4 if branch else None)
            result.append(row)
    return result


def run_evaluation(inputs, *, budget, backend_factory, measure, sink=None,
                   m_gate=None, t_gate=None, heartbeat=None, binding_check=None):
    """Run only when invoked by an authorized worker; fake tests inject all work."""
    from .wm_compat_reporting import measurement_screen, t_stage_gate, paired_rankings
    from .wm_compat_diagnostics import forecast_diagnostics
    m_gate=m_gate or measurement_screen
    t_gate=t_gate or t_stage_gate
    branches,real_items=inputs["branches"],inputs["real_items"]
    rows,real_rows=[],[]
    transform_diagnostics={}
    backend=None
    m_result=t_result=None
    def check(row=None):
        budget.check()
        if heartbeat:
            heartbeat.check()
    try:
        if binding_check:
            binding_check()
        real_rows=measure_real_items(real_items,measure=measure,budget=budget,sink=sink,progress=check,
                                    transform_diagnostics=transform_diagnostics)
        m_result=m_gate(real_rows)
        state=budget.advance_m(m_result["passed"],gate_evidence(m_result,"M.json"))
        if state["status"]=="active":
            if binding_check:
                binding_check()
            budget.reserve_model_load()
            if heartbeat:
                heartbeat.update(phase="model_loading",model_instances=1,model_loads=1,
                                 cuda_allocated_bytes=None,cuda_reserved_bytes=None)
            backend=backend_factory()
            budget.complete_model_load()
            if heartbeat:
                heartbeat.refresh=backend.stats
                heartbeat.update(phase="running",**backend.stats())
            for phase in ("T","L"):
                for branch in branches:
                    if branch.shape!=phase:
                        continue
                    for seed in (5478,5479,5480):
                        check()
                        if binding_check:
                            binding_check()
                        outcome=run_paired_branch(branch,seed,sampler=backend,rng=backend,
                            budget=budget,sink=sink,measure=measure,progress=check)
                        rows.extend(outcome["rows"])
                if phase=="T":
                    t_result=t_gate(rows)
                    if budget.advance_t(t_result["passed"],gate_evidence(t_result))["status"]!="active":
                        break
        state=budget.snapshot()
        all_rows=planned_forecast_rows(budget.plan,rows,branches,reason=state["stop_reason"])
        # TF is never passed to sequence/ranking/task-score APIs.
        ranks=paired_rankings([row for row in all_rows if row["mode"]=="AR"])
        return dict(kind="tdyssey.wm_input_compatibility.v1",development_only=True,
            independent_validation=False,M=m_result,T=t_result,rows=all_rows,real_rows=real_rows,
            diagnostics=forecast_diagnostics(all_rows,branches),
            ranking=ranks,budget=dict(status=state["status"],counts=state["counts"],
                stage_status=state["stage_status"],stop_reason=state["stop_reason"],
                started_monotonic=state["started_monotonic"],deadline_monotonic=state["deadline_monotonic"]),
            transform_diagnostics=transform_diagnostics,
            real_operations=0,policy_sampling=0,optimizer_updates=0)
    except BaseException as exc:
        try:
            budget.stop(type(exc).__name__+": "+str(exc))
        except BaseException:
            pass
        raise
    finally:
        primary_error=sys.exc_info()[1]
        if backend is not None:
            try:
                backend.close()
            except BaseException as exc:
                try:
                    budget.stop("backend_close_failed: "+str(exc))
                except BaseException:
                    pass
                if primary_error is None:
                    raise
                primary_error.add_note("Secondary backend close failure: "+type(exc).__name__+": "+str(exc))


def save_summary(result, sink, budget, *, branches=None):
    """Bounded final exports reference immutable row chunks rather than duplicate them."""
    summary={key:value for key,value in result.items() if key not in ("rows","real_rows","M","ranking","diagnostics")}
    # Detailed real measurements live in real/*.json, not repeated in M summary.
    if isinstance(result.get("M"),dict):
        measurement={key:value for key,value in result["M"].items() if key!="rows"}
        sink.record("M.json",measurement,maximum=512*1024,final=True)
        summary["M"]=dict(passed=measurement["passed"],report="M.json")
    if "ranking" in result:
        sink.record("ranking.json",result["ranking"],maximum=1024*1024,final=True)
        summary["ranking"]=dict(report="ranking.json")
    if "diagnostics" in result:
        sink.record("forecast-diagnostics.json",result["diagnostics"],maximum=192*1024,final=True)
        summary["diagnostics"]=dict(report="forecast-diagnostics.json")
    if branches is not None:
        from .wm_compat_appearance import export_case_plates
        summary["case_plates"]=export_case_plates(result["rows"],branches,sink)
    summary.update(expected_forecast_rows=len(result["rows"]),
        completed_forecast_rows=sum(row["status"]=="completed" for row in result["rows"]),
        missing_forecast_rows=sum(row["status"]=="missing" for row in result["rows"]),
        not_attempted_forecast_rows=sum(row["status"]=="not_attempted" for row in result["rows"]))
    sink.record("summary.json",summary,maximum=64*1024,final=True)
    statuses=dict(rows=[[row["key"],row["status"]] for row in result["rows"]],
        not_attempted_reason=result["budget"]["stop_reason"],missing_reason="original_L01_hold_missing")
    sink.record("planned-status.json",statuses,maximum=256*1024,final=True)
    if result["budget"]["status"]=="active":
        budget.finish(dict(development_only=True,report_exports_complete=True))
    state=budget.snapshot()
    completion={key:state[key] for key in ("status","counts","stage_status","stop_reason","deadline_monotonic")}
    sink.record("completion.json",completion,maximum=8*1024,final=True)
    result["budget"].update(completion)
