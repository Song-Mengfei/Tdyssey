"""Numeric-only screens for both weights; TF never contributes action ranking."""
from collections import defaultdict
from copy import deepcopy
import math

from .wm_compat_reporting import _distance, _errors, _measurement, _pose, _stats
from .wm_measurement_contract import sequence_endpoint


RETENTION_RULES = {
    "allowed_validity_drop_count": 0, "allowed_new_invalid_count": 0,
    "allowed_error_increase": 0.,
    "numeric_tolerance": {"position": 1e-6, "angle": 1e-6, "pusher": 1e-6, "coverage": 1e-9},
    "minimum_common_valid_fraction": {"position": .9, "angle": .9, "pusher": .95, "coverage": .9},
    "ranking_regret_numeric_tolerance": 1e-9, "allowed_ranking_accuracy_drop": 0.,
    "minimum_rank_comparable_anchors": 1, "minimum_rank_separable_pairs": 1,
    "require_complete_planned_ranking_grid": True,
    "require_all_informative_rank_anchors": True,
    "required_strata": ["regression", "independent"],
    "required_noise_seeds": [5478, 5479, 5480],
    "modes": {"AR": [1, 2, 3, 4], "TF": [2, 4]},
}


def _decision(failed, sufficient):
    return dict(status="fail" if failed else "pass" if sufficient else "insufficient",
                passed=False if failed else True if sufficient else None)


def _combine(records, required=True):
    records=list(records)
    return _decision(any(row["status"]=="fail" for row in records),
        required and bool(records) and all(row["status"]=="pass" for row in records))


def _row_key(row):
    return tuple(row[name] for name in ("stratum", "shape", "anchor", "candidate", "seed", "mode", "horizon"))


def _paired_cell(keys, indexed, contact, declared, plans):
    pairs=[];quality=dict(both_completed=0,matched_noise=0,matched_truth=0,matched_contact=0,declared_truth_consistent=0)
    completion={role:dict(completed=0,missing=0,not_completed=0) for role in ("frozen","adapted")}
    for key in keys:
        rows=[indexed.get((role,key)) for role in ("frozen","adapted")]
        for role,row in zip(("frozen","adapted"),rows):
            completion[role]["missing" if row is None else "completed" if row.get("status")=="completed" else "not_completed"]+=1
        views=[_measurement(row or {}) for row in rows]
        errors=[_errors(row or {},view) for row,view in zip(rows,views)]
        both=all(row is not None and row.get("status")=="completed" for row in rows)
        noise=both and all(isinstance(rows[0].get(name),str) and bool(rows[0][name]) and rows[0][name]==rows[1].get(name)
            for name in ("rng_entry","rng_exit"))
        expected=plans[key[:4]].get("truth",[])
        known=len(expected)==4
        source_match=both and known and all(row.get("truth")==expected[key[-1]-1] for row in rows)
        truth=both and all(error["truth_complete"] for error in errors) and rows[0]["truth"]==rows[1]["truth"] and (not known or source_match)
        labels=both and type(contact) is bool and all(row.get("contact_by_h4") is contact for row in rows)
        for name,value in (("both_completed",both),("matched_noise",noise),("matched_truth",truth),("matched_contact",labels),
                ("declared_truth_consistent",source_match)):
            quality[name]+=bool(value)
        pairs.append((key,views,errors,bool(noise and truth and labels)))
    n=len(keys);validity={};failed=False
    def ids(selected):return [list(key[2:5]) for key in selected]
    for flag in ("object_shape","pusher","coverage"):
        transitions=dict(both_valid=0,frozen_only_valid=0,adapted_only_valid=0,both_invalid=0)
        lost=[]
        for key,views,_,_ in pairs:
            f,a=(view["flags"][flag] for view in views)
            transitions["both_valid" if f and a else "frozen_only_valid" if f else "adapted_only_valid" if a else "both_invalid"]+=1
            if f and not a:lost.append(key)
        f=transitions["both_valid"]+transitions["frozen_only_valid"]
        a=transitions["both_valid"]+transitions["adapted_only_valid"]
        failed |= a<f or bool(lost)
        validity[flag]=dict(transitions,frozen_valid=f,adapted_valid=a,planned=n,
            frozen_fraction=f/n,adapted_fraction=a/n,delta_fraction=(a-f)/n,
            new_invalid_rows=ids(lost),new_invalid_branches=len({key[2:4] for key in lost}),
            new_invalid_anchors=len({key[2] for key in lost}))
    sufficient=declared and all(value==n for value in quality.values())
    metrics={};all_anchors={key[2] for key in keys}
    for metric,flag in (("position","object_shape"),("angle","object_shape"),("pusher","pusher"),("coverage","coverage")):
        common=[(key,errors) for key,views,errors,qualified in pairs if qualified
            and all(view["flags"][flag] for view in views) and all(error[metric] is not None for error in errors)]
        f=[errors[0][metric] for _,errors in common];a=[errors[1][metric] for _,errors in common]
        deltas=[av-fv for fv,av in zip(f,a)];tolerance=RETENTION_RULES["numeric_tolerance"][metric]
        deteriorated=[key for (key,_),delta in zip(common,deltas) if delta>tolerance]
        supported=(len(common)>=math.ceil(RETENTION_RULES["minimum_common_valid_fraction"][metric]*n)
            and {key[2] for key,_ in common}==all_anchors and bool(common))
        sufficient &= supported;failed |= bool(deteriorated)
        fs,ads=_stats(f),_stats(a)
        metrics[metric]=dict(common_valid=len(common),planned=n,common_fraction=len(common)/n,
            common_branches=len({key[2:4] for key,_ in common}),common_anchors=len({key[2] for key,_ in common}),
            minimum_common_fraction=RETENTION_RULES["minimum_common_valid_fraction"][metric],
            frozen_on_common=fs,adapted_on_common=ads,paired_error_increase=_stats(deltas),
            difference_of_p90=ads["p90"]-fs["p90"] if common else None,
            new_error_regressions=ids(deteriorated),numeric_tolerance=tolerance,
            **_decision(bool(deteriorated),supported and all(value==n for value in quality.values()) and declared))
    return dict(planned=n,planned_physical_branches=len({key[2:4] for key in keys}),
        planned_anchors=len(all_anchors),completion=completion,pair_quality=quality,validity=validity,
        errors_on_same_common_subset=metrics,**_decision(failed,sufficient))


def paired_retention(rows, rankings, seeds, planned_cases):
    """Compare the complete declared grid; absent rows never shrink denominators."""
    declared=planned_cases is not None
    if planned_cases is None:
        inferred={(r["stratum"],r["shape"],r["anchor"],r["candidate"]):dict(
            role=r["stratum"],shape=r["shape"],anchor=r["anchor"],candidate=r["candidate"],contact_by_h4=r.get("contact_by_h4")) for r in rows}
        planned_cases=list(inferred.values())
    plans={};groups=defaultdict(list);expected=set()
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("Retention needs unique integer noise seeds")
    protocol_seeds=RETENTION_RULES["required_noise_seeds"]
    protocol_matches=list(seeds)==protocol_seeds
    for case in planned_cases:
        case_key=(case["role"],case["shape"],case["anchor"],case["candidate"])
        if case_key in plans:raise ValueError("Duplicate planned physical branch")
        plans[case_key]=case
        for mode,horizons in RETENTION_RULES["modes"].items():
            for horizon in horizons:
                group=(case["role"],case["shape"],mode,horizon,case.get("contact_by_h4"))
                for seed in protocol_seeds:
                    key=(*case_key,seed,mode,horizon);expected.add(key);groups[group].append(key)
    indexed={}
    for row in rows:
        key=_row_key(row);identity=(row["role"],key)
        if key not in expected or type(row["seed"]) is not int or row["role"] not in ("frozen","adapted"):
            raise ValueError("Unplanned retention row/seed/mode/horizon")
        if identity in indexed:raise ValueError("Duplicate prediction in paired retention grid")
        indexed[identity]=row
    cells=[]
    for group,keys in sorted(groups.items(),key=lambda item:str(item[0])):
        cells.append(dict(stratum=group[0],shape=group[1],mode=group[2],horizon=group[3],contact_by_h4=group[4],
            **_paired_cell(keys,indexed,group[4],declared and protocol_matches,plans)))
    return dict(rules=deepcopy(RETENTION_RULES),denominator_source="declared_cases" if declared else "observed_union_unverified",
        declared_noise_seeds=list(protocol_seeds),requested_noise_seeds=list(seeds),noise_protocol_matches=protocol_matches,
        cells=cells,**_paired_rankings(rankings,plans,declared and protocol_matches,indexed,protocol_seeds))


def _paired_rankings(rankings, plans, declared, indexed, seeds):
    candidates=("hold","toward","side","away")
    records={(row["role"],row["stratum"],row["anchor"]):row for row in rankings}
    by_anchor=defaultdict(dict)
    for (stratum,shape,anchor,candidate),case in plans.items():by_anchor[(stratum,shape,anchor)][candidate]=case
    grouped=defaultdict(list)
    for (stratum,shape,anchor),cases in sorted(by_anchor.items()):
        f=records.get(("frozen",stratum,anchor));a=records.get(("adapted",stratum,anchor))
        eligible=[bool(row and row["eligible"]) for row in (f,a)]
        physical={};truth_complete=set(cases)==set(candidates);grid_complete=True;noise_complete=True
        expected_rows=completed_frozen=completed_adapted=matched_rows=0
        for candidate in candidates:
            case=cases.get(candidate,{})
            labels=case.get("truth",[])
            value=labels[-1].get("coverage") if len(labels)==4 else None
            # Information is fixed by declared physical outcomes, never by
            # either predictor's validity, scores or observed-row availability.
            truth_complete &= len(labels)==4 and type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
            physical[candidate]=value
            for seed in seeds:
                for mode,horizons in RETENTION_RULES["modes"].items():
                    for horizon in horizons:
                        key=(stratum,shape,anchor,candidate,seed,mode,horizon)
                        pair=[indexed.get((role,key)) for role in ("frozen","adapted")]
                        expected_rows+=1
                        complete=[row is not None and row.get("status")=="completed" for row in pair]
                        completed_frozen+=complete[0];completed_adapted+=complete[1]
                        same_noise=all(complete) and all(isinstance(pair[0].get(name),str) and bool(pair[0][name]) and pair[0][name]==pair[1].get(name)
                            for name in ("rng_entry","rng_exit"))
                        if mode=="AR":noise_complete &= bool(same_noise)
                        matched=all(complete) and len(labels)==4 and type(case.get("contact_by_h4")) is bool
                        if matched:
                            matched=all(row.get("truth")==labels[horizon-1] and row.get("contact_by_h4") is case["contact_by_h4"] for row in pair)
                            matched &= bool(same_noise)
                        matched_rows+=bool(matched);grid_complete &= bool(matched)
        separable=[]
        if truth_complete:
            separable=[(first,second) for i,first in enumerate(candidates) for second in candidates[i+1:]
                if abs(physical[first]-physical[second])+1e-12>=.05]
        information="informative" if separable else "uninformative" if truth_complete else "unknown"
        common=all(eligible) and grid_complete and information=="informative"
        f_correct=[];a_correct=[];f_regret=a_regret=None
        if all(eligible) and truth_complete and grid_complete:
            f_regret=max(physical.values())-physical[f["selected"]]
            a_regret=max(physical.values())-physical[a["selected"]]
        if common:
            for row,correct in ((f,f_correct),(a,a_correct)):
                for first,second in separable:
                    gap=row["scores"][first]-row["scores"][second]
                    if abs(gap)>1e-6 and gap*(physical[first]-physical[second])>0:correct.append((first,second))
        lost=[pair for pair in f_correct if pair not in a_correct]
        failed=information=="informative" and ((eligible[0] and not eligible[1]) or bool(lost) or (
            common and a_regret-f_regret>RETENTION_RULES["ranking_regret_numeric_tolerance"]))
        decision=_decision(failed,declared and common)
        if information=="uninformative" and grid_complete:decision=dict(status="uninformative",passed=None)
        grouped[(stratum,shape)].append(dict(anchor=anchor,planned=True,frozen_eligible=eligible[0],adapted_eligible=eligible[1],
            new_ineligible=eligible[0] and not eligible[1],truth_complete=truth_complete,information=information,
            matched_noise_complete=bool(noise_complete),
            grid_complete=bool(grid_complete),planned_rows_per_role=expected_rows,completed_frozen_rows=completed_frozen,
            completed_adapted_rows=completed_adapted,matched_grid_rows=matched_rows,
            actual_separable_pairs=len(separable),common_comparable=bool(common),common_pair_ids=[list(pair) for pair in separable] if common else [],
            frozen_correct_pairs=len(f_correct) if common else None,adapted_correct_pairs=len(a_correct) if common else None,
            lost_correct_pairs=[list(pair) for pair in lost],frozen_selected_regret=f_regret,adapted_selected_regret=a_regret,
            **decision))
    output=[]
    for (stratum,shape),anchors in sorted(grouped.items()):
        informative=[row for row in anchors if row["information"]=="informative"]
        uninformative=[row for row in anchors if row["information"]=="uninformative"]
        common=[row for row in anchors if row["common_comparable"]]
        pair_count=sum(row["actual_separable_pairs"] for row in common)
        complete=declared and all(row["grid_complete"] and row["truth_complete"] for row in anchors)
        support=(len(common)>=RETENTION_RULES["minimum_rank_comparable_anchors"]
            and pair_count>=RETENTION_RULES["minimum_rank_separable_pairs"] and len(common)==len(informative))
        output.append(dict(stratum=stratum,shape=shape,planned_anchors=len(anchors),
            grid_complete=bool(complete),incomplete_grid_anchors=[row["anchor"] for row in anchors if not row["grid_complete"]],
            planned_rows_per_role=sum(row["planned_rows_per_role"] for row in anchors),
            informative_anchors=len(informative),informative_anchor_ids=[row["anchor"] for row in informative],
            uninformative_anchors=len(uninformative),uninformative_anchor_ids=[row["anchor"] for row in uninformative],
            unknown_truth_anchors=[row["anchor"] for row in anchors if row["information"]=="unknown"],
            planned_informative_pairs=sum(row["actual_separable_pairs"] for row in informative),
            frozen_eligible_anchors=sum(row["frozen_eligible"] for row in anchors),adapted_eligible_anchors=sum(row["adapted_eligible"] for row in anchors),
            new_ineligible_anchors=[row["anchor"] for row in anchors if row["new_ineligible"]],
            common_comparable_anchors=len(common),common_separable_pairs=pair_count,
            frozen_accuracy_on_common=sum(row["frozen_correct_pairs"] for row in common)/pair_count if pair_count else None,
            adapted_accuracy_on_common=sum(row["adapted_correct_pairs"] for row in common)/pair_count if pair_count else None,
            frozen_regret_on_common=_stats([row["frozen_selected_regret"] for row in common]),
            adapted_regret_on_common=_stats([row["adapted_selected_regret"] for row in common]),
            anchors=anchors,**_combine(informative,complete and support)))
    return dict(ranking=output,ranking_scope="Complete identity-matched grid for every planned anchor; rank every physical-truth-informative anchor/pair; complete uninformative anchors are neutral, never successes",
        termination_certified=False)


def summarize(rows, seeds=(5478, 5479, 5480), *, planned_cases=None):
    rows=list(rows)
    cells, sequences = defaultdict(list), defaultdict(list)
    for row in rows:
        cells[(row["role"], row["stratum"], row["shape"], row["mode"], row["horizon"], row.get("contact_by_h4"))].append(row)
        if row["mode"] == "AR":
            sequences[(row["role"], row["stratum"], row["anchor"], row["candidate"], row["seed"])].append(row)
    measurements = []
    for key, group in sorted(cells.items(), key=lambda pair:str(pair[0])):
        views = [_measurement(row) for row in group]
        errors = [_errors(row, view) for row, view in zip(group, views)]
        n = len(group)
        valid = {name:sum(view["flags"][name] for view in views) for name in ("object_shape", "pusher", "coverage")}
        stats = {name:_stats([e[name] for e in errors if e[name] is not None]) for name in ("position", "angle", "pusher", "coverage")}
        limited = (stats["position"]["p90"] is not None and stats["position"]["p90"] <= 5
            and stats["angle"]["p90"] <= 5 and stats["pusher"]["p90"] is not None and stats["pusher"]["p90"] <= 5
            and stats["coverage"]["p90"] is not None and stats["coverage"]["p90"] <= .05)
        physical = {(r["anchor"],r["candidate"]) for r in group}
        measurements.append(dict(role=key[0], stratum=key[1], shape=key[2], mode=key[3], horizon=key[4],
            contact_by_h4=key[5], total=n, anchors=len({r["anchor"] for r in group}), physical_branches=len(physical),
            valid=valid, conditional_errors=stats, invalid_count=n-valid["object_shape"],
            shape_screen=valid["object_shape"] >= math.ceil(.9*n), pusher_screen=valid["pusher"] >= math.ceil(.95*n),
            limited_error_screen=limited, all_error_labels_present=all(e["truth_complete"] for e in errors),
            error_statistics_are_conditional=True))
    scores, actual = {}, {}
    for key, sequence in sequences.items():
        sequence = sorted(sequence,key=lambda row:row["horizon"])
        if len({row["horizon"] for row in sequence})!=len(sequence):raise ValueError("Duplicate AR horizon")
        endpoint = sequence_endpoint([_measurement(row)["scoring"] for row in sequence]) if [row["horizon"] for row in sequence]==[1,2,3,4] else None
        scores[key] = endpoint["stopped_score"] if endpoint and endpoint["stopped_score_valid"] else None
        last=next((row for row in sequence if row["horizon"]==4),{})
        value = last.get("truth",{}).get("coverage")
        actual_key=key[:4]
        if value is not None and actual.get(actual_key) is not None and actual[actual_key] != value:
            raise ValueError("Noise replicates cannot have different physical truth")
        if value is not None or actual_key not in actual:actual[actual_key] = value
    rankings=[]
    candidates=("hold","toward","side","away")
    for role, stratum, anchor in sorted({key[:3] for key in scores}):
        means={}
        truth={candidate:actual.get((role,stratum,anchor,candidate)) for candidate in candidates}
        for candidate in candidates:
            values=[scores.get((role,stratum,anchor,candidate,seed)) for seed in seeds]
            means[candidate]=sum(values)/len(values) if all(v is not None for v in values) else None
        complete=all(v is not None for v in means.values())
        truth_complete=all(isinstance(v,(int,float)) and 0 <= v <= 1 for v in truth.values())
        chosen=None; regret=None
        if complete:
            best=max(means.values()); chosen=next(c for c in candidates if means[c] >= best-1e-6)
            if truth_complete:regret=max(truth.values())-truth[chosen]
        pairs=correct=ties=unscored=0
        for i,first in enumerate(candidates):
            for second in candidates[i+1:]:
                if truth[first] is None or truth[second] is None or abs(truth[first]-truth[second])+1e-12 < .05:continue
                pairs+=1
                if means[first] is None or means[second] is None:unscored+=1; continue
                difference=means[first]-means[second]
                if abs(difference) <= 1e-6:ties+=1
                elif difference*(truth[first]-truth[second]) > 0:correct+=1
        rankings.append(dict(role=role,stratum=stratum,anchor=anchor,eligible=complete,selected=chosen,
            scores=means,actual=truth,selected_regret=regret,actual_separable_pairs=pairs,
            correct_pairs=correct,tied_pairs=ties,abstained_pairs=unscored,
            accuracy_including_abstentions=correct/pairs if pairs else None,
            hold_regret=max(truth.values())-truth["hold"] if truth_complete else None,
            uniform_expected_regret=max(truth.values())-sum(truth.values())/4 if truth_complete else None))
    dynamics=[]
    h4=defaultdict(list)
    for row in rows:
        if row["mode"] == "AR" and row["horizon"] == 4:
            h4[(row["role"],row["stratum"],row["shape"])].append(row)
    def turn(a,b):return math.degrees(math.atan2(math.sin(a-b),math.cos(a-b)))
    for key, group in sorted(h4.items()):
        contact_errors=[]; persistence=[]; drift=[]; angle_errors=[]; zero_rotation=[]
        rotation=[]; correct=0; positive_anchors=set(); tp=fp=unknown=0
        for row in group:
            view=_measurement(row); actual_pose=_pose(row.get("truth",{}).get("pose"))
            latest=_pose((row.get("latest") or {}).get("pose"))
            truth=row.get("truth",{}).get("coverage")
            decision=view["termination"]["status"]
            if decision=="success" and isinstance(truth,(int,float)):
                if truth>.95:tp+=1; positive_anchors.add(row["anchor"])
                else:fp+=1
            elif decision=="unknown":unknown+=1
            if actual_pose is None or latest is None:continue
            if row.get("contact_by_h4") is True and view["flags"]["object"]:
                contact_errors.append(_distance(view["pose"]["position"],actual_pose["position"]))
                persistence.append(_distance(latest["position"],actual_pose["position"]))
            if row.get("contact_by_h4") is False and view["flags"]["object"]:
                drift.append(_distance(view["pose"]["position"],latest["position"]))
            actual_turn=turn(actual_pose["angle"],latest["angle"])
            if abs(actual_turn)>5:
                rotation.append(row); zero_rotation.append(abs(actual_turn))
                if view["flags"]["object"]:
                    correct+=turn(view["pose"]["angle"],latest["angle"])*actual_turn>0
                    angle_errors.append(abs(turn(view["pose"]["angle"],actual_pose["angle"])))
        rotation_branches=len({(r["anchor"],r["candidate"]) for r in rotation})
        rotation_anchors=len({r["anchor"] for r in rotation})
        direction=correct/len(rotation) if rotation else None
        rotation_enough=rotation_branches>=4 and rotation_anchors>=2
        contact_count=len({(r["anchor"],r["candidate"]) for r in group if r.get("contact_by_h4") is True})
        dynamics.append(dict(role=key[0],stratum=key[1],shape=key[2],contact_physical_branches=contact_count,
            contact_position_error=_stats(contact_errors),same_valid_subset_persistence_error=_stats(persistence),
            contact_better_than_persistence=(sum(contact_errors)<sum(persistence)) if contact_count>=8 and
                len(contact_errors)==sum(r.get("contact_by_h4") is True for r in group) else None,
            noncontact_drift=_stats(drift),noncontact_drift_screen=sum(drift)/len(drift)<=2 if drift and
                len(drift)==sum(r.get("contact_by_h4") is False for r in group) else None,
            rotation_physical_branches=rotation_branches,rotation_anchors=rotation_anchors,
            rotation_direction_fraction_including_abstentions=direction,conditional_rotation_angle_error=_stats(angle_errors),
            rotation_screen=(direction>=.8 and len(angle_errors)==len(rotation) and sum(angle_errors)/len(angle_errors)<=5
                and sum(angle_errors)<sum(zero_rotation)) if rotation_enough else None,
            interval_success_tp=tp,interval_success_fp=fp,interval_success_precision=tp/(tp+fp) if tp+fp else None,
            interval_unknown=unknown,positive_anchors=len(positive_anchors),
            independent_termination_precision_validated=False,
            termination_reason="Existing +/- .03 development interval is an assumption; sparse/unknown positives cannot certify stopping"))
    retention=paired_retention(rows,rankings,seeds,planned_cases)
    t={}
    for mode in ("AR","TF"):
        relative=[row for row in retention["cells"] if row["shape"]=="T" and row["mode"]==mode]
        required=set(RETENTION_RULES["required_strata"])=={row["stratum"] for row in relative}
        t[mode]=_combine(relative,required)
    t["ranking"]=_combine([row for row in retention["ranking"] if row["shape"]=="T"],
        {row["stratum"] for row in retention["ranking"] if row["shape"]=="T"}==set(RETENTION_RULES["required_strata"]))
    observed={(row["stratum"],row["horizon"],row["contact_by_h4"]):row for row in measurements
        if row["role"]=="adapted" and row["shape"]=="T" and row["mode"]=="AR"}
    absolute=[]
    for grid in retention["cells"]:
        if grid["shape"]!="T" or grid["mode"]!="AR" or grid["horizon"] not in (1,2,4):continue
        row=observed.get((grid["stratum"],grid["horizon"],grid["contact_by_h4"]))
        complete=retention["denominator_source"]=="declared_cases" and row is not None and row["total"]==grid["planned"] and all(
            count==grid["planned"] for count in grid["pair_quality"].values())
        gates=bool(row and all(row[field] for field in ("shape_screen","pusher_screen","limited_error_screen","all_error_labels_present")))
        absolute.append(dict(stratum=grid["stratum"],horizon=grid["horizon"],contact_by_h4=grid["contact_by_h4"],
            planned=grid["planned"],observed_adapted=row["total"] if row else 0,
            **_decision(complete and not gates,complete and gates)))
    t["absolute_AR"]=dict(_combine(absolute,{row["stratum"] for row in absolute}==set(RETENTION_RULES["required_strata"])),cells=absolute)
    t["overall"]=_combine([t["AR"],t["absolute_AR"],t["ranking"]])
    retention["T"]=t
    return dict(kind="tdyssey.gpc_adaptation_screens.v1", cells=measurements, ranking=rankings, dynamics=dynamics,paired_retention=retention,
        ranking_uses="AR stopping-consistent scores; all declared noises per candidate; declared order breaks 1e-6 ties",
        noise_is_not_an_independent_physical_sample=True,
        acceptance="Engineering screens only; contact/rotation/near-success coverage needs separate review; no imagined-training authorization")
