"""Offline L attribution from immutable numeric records; no decoder or model.

Truth labels define errors/strata and physical references only. They never
repair a rejected measurement, stopping score, or candidate selection.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .wm_compat_reporting import (
    ARMS, CANDIDATES, SEEDS, TIE_TOLERANCE, PAIR_SEPARATION,
    _coverage, _errors, _finite, _mean, _measurement, _pair_summary, _stats,
    paired_rankings,
)
from .wm_compat_diagnostics import _persistence_errors, _same_truth, _truth
from .wm_measurement_contract import angle_error_degrees

ANCHORS = tuple(f"L{i:02d}" for i in range(12))
STRATA = ("all", "contact", "noncontact", "rotation_gt_5", "rotation_le_5",
          "unknown_contact", "unknown_rotation")
THRESHOLDS = dict(position=5., angle=5., pusher=5., coverage=.05)
FLAGS = ("object", "shape", "object_shape", "pusher", "coverage_independent", "coverage")


def _hierarchy(entries):
    """Three noises are repeated predictions, not independent physical cases."""
    branches = defaultdict(list)
    for anchor, candidate, seed, value in entries:
        branches[(anchor, candidate)].append((seed, value))
    anchors = defaultdict(list)
    branch_rows = []
    for (anchor, candidate), values in sorted(branches.items()):
        value = _mean([v for _, v in values])
        branch_rows.append(dict(anchor=anchor, candidate=candidate, measured_seeds=len(values), mean=value))
        anchors[anchor].append(value)
    anchor_rows = [dict(anchor=k, measured_branches=len(v), mean=_mean(v)) for k, v in sorted(anchors.items())]
    return dict(rows=len(entries), branches=len(branch_rows), anchors=len(anchor_rows),
        row_statistics=_stats([v for _, _, _, v in entries]),
        equal_anchor_statistics=_stats([r["mean"] for r in anchor_rows]),
        branch_values=branch_rows, anchor_values=anchor_rows)


def _view(row):
    result = _measurement(row)
    coverage = result["measurement"].get("coverage", {})
    result["flags"]["coverage_independent"] = bool(row["status"] == "completed"
        and coverage.get("valid") is True and _coverage(coverage.get("estimate")) is not None)
    return result


def _inventory(rows, branches):
    indexed, by_branch = {}, {}
    for branch in branches:
        key = (branch.get("anchor"), branch.get("candidate"))
        if key[0] not in ANCHORS:
            continue
        if key[1] not in CANDIDATES or key in by_branch:
            raise ValueError("Duplicate or unknown L physical branch")
        by_branch[key] = branch
    for value in rows:
        if value.get("shape") != "L":
            continue
        row = deepcopy(value)
        mode, horizon = row.get("mode"), row.get("horizon")
        key = (row.get("arm"), row.get("anchor"), row.get("candidate"), row.get("seed"), mode, horizon)
        if (key[0] not in ARMS or key[1] not in ANCHORS or key[2] not in CANDIDATES
                or type(key[3]) is not int or key[3] not in SEEDS or mode not in ("AR", "TF")
                or type(horizon) is not int or horizon not in ((1, 2, 3, 4) if mode == "AR" else (2, 4))
                or row.get("status") not in ("completed", "missing", "not_attempted") or key in indexed):
            raise ValueError("Duplicate or noncanonical L forecast row")
        if key[1:3] == ("L01", "hold") and row["status"] == "completed":
            raise ValueError("Original L01/hold cannot acquire a replacement forecast")
        indexed[key] = row
    physical = {}
    for anchor in ANCHORS:
        for candidate in CANDIDATES:
            branch = by_branch.get((anchor, candidate), {})
            available = (anchor, candidate) != ("L01", "hold")
            for h in range(1, 5):
                group = [r for k, r in indexed.items() if k[1:3] == (anchor, candidate) and k[-1] == h]
                labels = [_truth(r.get("truth")) for r in group]
                saved = branch.get("truth", [])
                if isinstance(saved, list) and len(saved) >= h:
                    labels.insert(0, _truth(saved[h-1]))
                truth = dict.fromkeys(("pose", "pusher", "coverage"))
                for label in (v for v in labels if v is not None):
                    if not _same_truth(truth, label):
                        raise ValueError("Conflicting physical labels across noise/arm/conditioning")
                    truth.update({k: v for k, v in label.items() if v is not None})
                contacts = {r["contact_by_h4"] for r in group if type(r.get("contact_by_h4")) is bool}
                if type(branch.get("contact_by_h4")) is bool:
                    contacts.add(branch["contact_by_h4"])
                if len(contacts) > 1:
                    raise ValueError("Conflicting physical contact labels")
                latest = branch.get("latest")
                prior = _truth(latest)
                rotation = (angle_error_degrees(truth["pose"]["angle"], prior["pose"]["angle"])
                    if truth["pose"] is not None and prior and prior["pose"] is not None else None)
                if not available:
                    truth, rotation, contacts = dict.fromkeys(truth), None, set()
                physical[(anchor, candidate, h)] = dict(available=available, truth=truth, latest=latest,
                    contact=next(iter(contacts)) if contacts else None, rotation_degrees=rotation,
                    persistence=_persistence_errors(latest, truth))
                for arm in ARMS:
                    for seed in SEEDS:
                        for mode in ("AR", "TF") if h in (2, 4) else ("AR",):
                            key = (arm, anchor, candidate, seed, mode, h)
                            row = indexed.setdefault(key, dict(arm=arm, anchor=anchor, candidate=candidate,
                                seed=seed, shape="L", mode=mode, horizon=h,
                                status="not_attempted" if available else "missing", measurement=None,
                                absence_reason="numeric_record_missing" if available else "original_L01_hold_missing"))
                            row["truth"] = deepcopy(truth)
                            row["contact_by_h4"] = physical[(anchor, candidate, h)]["contact"]
    return indexed, physical


def _chosen(label, stratum):
    if stratum == "all": return True
    if stratum == "contact": return label["contact"] is True
    if stratum == "noncontact": return label["contact"] is False
    if stratum == "unknown_contact": return label["contact"] is None
    rotation = label["rotation_degrees"]
    if stratum == "unknown_rotation": return rotation is None
    return rotation is not None and ((rotation > 5) if stratum == "rotation_gt_5" else rotation <= 5)


def _cell(index, physical, mode, horizon, stratum):
    source_mode = "AR" if mode == "TF" and horizon == 1 else mode
    identities = [(a, c, s) for a in ANCHORS for c in CANDIDATES for s in SEEDS
                  if _chosen(physical[(a, c, horizon)], stratum)]
    pairs = []
    for a, c, s in identities:
        label = physical[(a, c, horizon)]
        rows = {arm:index[(arm, a, c, s, source_mode, horizon)] for arm in ARMS}
        views = {arm:_view(row) for arm, row in rows.items()}
        pairs.append(dict(anchor=a, candidate=c, seed=s, physical=label, rows=rows, views=views,
            errors={arm:_errors(rows[arm], views[arm]) for arm in ARMS}))
    available = [p for p in pairs if p["physical"]["available"]]
    result = dict(mode=mode, horizon=horizon, stratum=stratum, source_mode=source_mode,
        H1_reused_from_AR=mode == "TF" and horizon == 1,
        denominators=dict(planned_rows_per_arm=len(pairs), available_rows_per_arm=len(available),
            planned_branches=len({(p['anchor'],p['candidate']) for p in pairs}),
            available_branches=len({(p['anchor'],p['candidate']) for p in available}),
            planned_anchors=len({p['anchor'] for p in pairs}), available_anchors=len({p['anchor'] for p in available})),
        arms={}, paired_metrics={})
    entry = lambda p,v:(p["anchor"],p["candidate"],p["seed"],v)
    for arm in ARMS:
        metrics = {}
        for metric in THRESHOLDS:
            measured = [p for p in available if p["errors"][arm][metric] is not None]
            persistent = [p for p in measured if p["physical"]["persistence"][metric] is not None]
            truth_key = 'pose' if metric in ('position','angle') else metric
            truth_missing = sum(p['physical']['truth'][truth_key] is None for p in available)
            metrics[metric] = dict(threshold=THRESHOLDS[metric],
                measured_fraction=len(measured)/len(available) if available else None,
                truth_missing_rows=truth_missing, measurement_rejected_or_unavailable_rows=len(available)-len(measured)-truth_missing,
                error=_hierarchy([entry(p,p["errors"][arm][metric]) for p in measured]),
                persistence_same_subset=_hierarchy([entry(p,p["physical"]["persistence"][metric]) for p in persistent]),
                model_minus_persistence_same_subset=_hierarchy([entry(p,p["errors"][arm][metric]-p["physical"]["persistence"][metric]) for p in persistent]))
        result["arms"][arm] = dict(status_counts=dict(Counter(p["rows"][arm]["status"] for p in pairs)),
            validity={flag:dict(valid_rows=sum(p["views"][arm]["flags"][flag] for p in available),
                available_rows=len(available), complete_three_seed_branches=sum(all(p["views"][arm]["flags"][flag]
                    for p in available if (p["anchor"],p["candidate"])==branch) for branch in {(p["anchor"],p["candidate"]) for p in available}),
                anchors_with_valid_rows=len({p["anchor"] for p in available if p["views"][arm]["flags"][flag]})) for flag in FLAGS},
            by_seed={str(s):dict(available_rows=sum(p["seed"]==s for p in available),
                valid={flag:sum(p["views"][arm]["flags"][flag] for p in available if p["seed"]==s) for flag in FLAGS},
                errors={m:_stats([p["errors"][arm][m] for p in available if p["seed"]==s and p["errors"][arm][m] is not None]) for m in THRESHOLDS}) for s in SEEDS},
            metrics=metrics)
    for metric in THRESHOLDS:
        common = [p for p in available if all(p["errors"][arm][metric] is not None for arm in ARMS)]
        result["paired_metrics"][metric] = dict(common_rows=len(common),available_rows=len(available),
            B_minus_A=_hierarchy([entry(p,p["errors"]["B"][metric]-p["errors"]["A"][metric]) for p in common]))
    return result


def _first_failures(index, physical):
    streams=[]
    for arm in ARMS:
        for mode, horizons in (("AR",(1,2,3,4)),("TF",(1,2,4))):
            for a in ANCHORS:
                for c in CANDIDATES:
                    for s in SEEDS:
                        failures={}
                        for flag in FLAGS:
                            failures['validity_'+flag]=None
                        failures.update({'physical_'+m:None for m in THRESHOLDS})
                        if physical[(a,c,1)]["available"]:
                            for h in horizons:
                                row=index[(arm,a,c,s,"AR" if h==1 else mode,h)];view=_view(row)
                                errors=_errors(row,view)
                                for flag in FLAGS:
                                    if failures['validity_'+flag] is None and not view['flags'][flag]:
                                        part='state' if flag in ('object','object_shape') else 'coverage' if flag=='coverage_independent' else flag
                                        detail=view['measurement'].get(part,{})
                                        failures['validity_'+flag]=dict(horizon=h,kind='invalid_measurement' if row['status']=='completed' else 'unavailable',
                                            status=row['status'],reason=detail.get('reason') or row.get('absence_reason') or 'invalid_or_missing_measurement')
                                for metric,threshold in THRESHOLDS.items():
                                    error=errors[metric]
                                    if failures['physical_'+metric] is None and (error is None or error>threshold+1e-12):
                                        failures['physical_'+metric]=dict(horizon=h,kind='unavailable' if error is None else 'threshold_exceeded',
                                            error=error,threshold=threshold)
                        streams.append(dict(arm=arm,mode=mode,anchor=a,candidate=c,seed=s,
                            physical_available=physical[(a,c,1)]['available'],evaluated_horizons=list(horizons),
                            TF_H1_reuses_AR=mode=='TF',first_failures=failures))
    summary=[]
    for arm in ARMS:
        for mode in ('AR','TF'):
            group=[r for r in streams if r['arm']==arm and r['mode']==mode]
            for metric in group[0]['first_failures']:
                failed=[r for r in group if r['first_failures'][metric] is not None]
                summary.append(dict(arm=arm,mode=mode,component=metric,planned_seed_streams=len(group),
                    available_seed_streams=sum(r['physical_available'] for r in group),planned_branches=48,available_branches=47,planned_anchors=12,
                    failed_seed_streams=len(failed),affected_branches=len({(r['anchor'],r['candidate']) for r in failed}),
                    affected_anchors=len({r['anchor'] for r in failed}),
                    first_event_kind_counts=dict(Counter(r['first_failures'][metric]['kind'] for r in failed)),
                    first_horizon_counts=dict(Counter(str(r['first_failures'][metric]['horizon']) for r in failed)),
                    by_seed={str(s):dict(planned_branches=48,available_branches=47,planned_anchors=12,
                        failed_streams=sum(r['seed']==s for r in failed),
                        first_horizon_counts=dict(Counter(str(r['first_failures'][metric]['horizon']) for r in failed if r['seed']==s))) for s in SEEDS}))
    return dict(streams=streams,summary=summary,TF_H3='not sampled; first failure only over H1 reuse/H2/H4')


def _ranking(rows):
    old=paired_rankings([r for r in rows if r['mode']=='AR'])
    anchor_rows=[r for r in old['anchor_rows'] if r['shape']=='L']
    physical={(r['anchor'],r['horizon']):r for r in old['physical_anchor_rows'] if r['shape']=='L'}
    for row in anchor_rows:
        actual=physical[row['anchor'],row['horizon']]['actual_coverage']
        ties=0
        for i,a in enumerate(CANDIDATES):
            for b in CANDIDATES[i+1:]:
                if actual[a] is None or actual[b] is None or abs(actual[a]-actual[b])+1e-12<PAIR_SEPARATION:
                    continue
                first,second=row['candidates'][a],row['candidates'][b]
                if first['eligible'] and second['eligible'] and abs(first['score']-second['score'])<=TIE_TOLERANCE:
                    ties+=1
        row['pairs']['predicted_ties_among_scored_separable_pairs']=ties
        values=[v['score'] for v in row['candidates'].values() if v['eligible']]
        row['maximizer_tie_candidates']=([c for c in CANDIDATES if max(values)-row['candidates'][c]['score']<=TIE_TOLERANCE]
            if row['selection_eligible'] else [])
    subsets=[]
    for h in range(1,5):
        for mode in ('raw_endpoint','stopping'):
            group={arm:{r['anchor']:r for r in anchor_rows if r['arm']==arm and r['horizon']==h and r['scoring_mode']==mode} for arm in ARMS}
            for label,arms in (('A_own',('A',)),('B_own',('B',)),('A_B_common',ARMS)):
                chosen=[a for a in ANCHORS if physical[a,h]['complete'] and all(group[arm][a]['selection_eligible'] for arm in arms)]
                omitted=[dict(anchor=a,reasons=([] if physical[a,h]['complete'] else ['incomplete_physical_candidates'])+
                    [arm+'_candidate_noise_abstention' for arm in arms if not group[arm][a]['selection_eligible']]) for a in ANCHORS if a not in chosen]
                item=dict(subset=label,horizon=h,scoring_mode=mode,planned_anchors=12,included_anchors=chosen,
                    included_anchor_count=len(chosen),omitted_anchors=omitted,
                    physical_baselines_same_anchors={k:_mean([physical[a,h][k] for a in chosen]) for k in
                        ('hold_reward','uniform_expected_reward','hold_regret','uniform_expected_regret')},arms={})
                for arm in arms:
                    eligible=[group[arm][a] for a in chosen]
                    item['arms'][arm]=dict(mean_reward=_mean([r['selected_actual_reward'] for r in eligible]),
                        mean_regret=_mean([r['regret'] for r in eligible]),pairs=_pair_summary(eligible),
                        predicted_ties_among_scored_separable_pairs=sum(r['pairs']['predicted_ties_among_scored_separable_pairs'] for r in eligible))
                subsets.append(item)
    return dict(anchor_rows=anchor_rows,subsets=subsets,
        all_planned_pairs=[dict(arm=arm,horizon=h,scoring_mode=mode,**_pair_summary([r for r in anchor_rows
            if r['arm']==arm and r['horizon']==h and r['scoring_mode']==mode]))
            for arm in ARMS for h in range(1,5) for mode in ('raw_endpoint','stopping')],
        independent_physical_baselines=[r for r in old['independent_physical_baselines'] if r['shape']=='L'],
        TF_used=False,pair_separation=PAIR_SEPARATION,tie_tolerance=TIE_TOLERANCE,
        tie_rule='First candidate in hold/toward/side/away within 1e-6 of maximum; pair prediction ties are incorrect scored pairs.',
        uniform_reference='Exact mean of four saved physical outcomes; no new random sampling.',
        weighting='Three-noise mean inside each candidate, then equal complete anchors; baselines use identical anchors in each subset.')


def _contact_onset_timing(index, branches):
    """Compare first B-AR shape rejection to exact saved contact onset only."""
    lookup={(r['anchor'],r['candidate']):r for r in branches if r.get('anchor') in ANCHORS}
    streams,branch_rows=[],[]
    for a in ANCHORS:
        for c in CANDIDATES:
            available=(a,c)!=('L01','hold')
            supplied=lookup.get((a,c),{}).get('contacts_by_horizon')
            supplied=supplied if isinstance(supplied,(list,tuple)) else ()
            contacts=[(float(supplied[h]) if h<len(supplied) and _finite(supplied[h]) and supplied[h]>=0 else None)
                      for h in range(4)]
            positive=next((h+1 for h,v in enumerate(contacts) if v is not None and v>0),None)
            contact_h=(positive if positive is not None and all(v is not None for v in contacts[:positive-1]) else None)
            contact_status=('physical_missing' if not available else 'known' if contact_h is not None else
                            'never' if all(v is not None and v==0 for v in contacts) else 'unknown')
            group=[]
            for seed in SEEDS:
                shape=[]
                for h in range(1,5):
                    row=index[('B',a,c,seed,'AR',h)]
                    part=(row.get('measurement') or {}).get('shape') or {}
                    shape.append(part.get('valid') if row['status']=='completed' and type(part.get('valid')) is bool else None)
                observed=next((h+1 for h,v in enumerate(shape) if v is False),None)
                failure_h=(observed if observed is not None and all(v is True for v in shape[:observed-1]) else None)
                failure_status=('physical_missing' if not available else 'known' if failure_h is not None else
                                'never' if all(v is True for v in shape) else 'unknown')
                relation=('physical_missing' if not available else
                          'unknown' if 'unknown' in (contact_status,failure_status) else
                          'never' if failure_status=='never' else 'no_contact' if contact_status=='never' else
                          'before' if failure_h<contact_h else 'same' if failure_h==contact_h else 'after')
                item=dict(anchor=a,candidate=c,seed=seed,arm='B',mode='AR',physical_available=available,
                    contacts_by_horizon=contacts,first_contact_status=contact_status,first_contact_h=contact_h,
                    first_observed_positive_contact_h=positive,first_shape_invalid_status=failure_status,
                    first_shape_invalid_h=failure_h,first_observed_shape_invalid_h=observed,relation=relation)
                streams.append(item);group.append(item)
            relations=[r['relation'] for r in group]
            branch_rows.append(dict(anchor=a,candidate=c,physical_available=available,contacts_by_horizon=contacts,
                first_contact_status=contact_status,first_contact_h=contact_h,
                relation_by_seed={str(r['seed']):r['relation'] for r in group},
                unanimous_relation=relations[0] if len(set(relations))==1 else 'mixed'))
    labels=('before','same','after','never','no_contact','unknown','physical_missing')
    return dict(streams=streams,branches=branch_rows,
        denominators=dict(planned_seed_streams=144,available_seed_streams=141,planned_branches=48,
            available_branches=47,planned_anchors=12,available_anchors=12),
        seed_stream_relation_counts=dict(Counter(r['relation'] for r in streams)),
        branch_unanimous_relation_counts=dict(Counter(r['unanimous_relation'] for r in branch_rows)),
        branches_with_relation={label:len({(r['anchor'],r['candidate']) for r in streams if r['relation']==label}) for label in labels},
        anchors_with_relation={label:len({r['anchor'] for r in streams if r['relation']==label}) for label in labels},
        by_seed={str(s):dict(planned_branches=48,available_branches=47,
            relation_counts=dict(Counter(r['relation'] for r in streams if r['seed']==s))) for s in SEEDS},
        definitions=dict(contact='First actual[].contacts>0 preceded by known zero-contact steps; missing values never mean zero.',
            failure='First explicit shape.valid=False preceded by completed shape.valid=True frames; missing predecessors make onset unknown.',
            before_same_after='Compare exact H1-H4 horizons, not the future contact_by_H4 stratum.',
            never='All four B-AR shape frames valid and contact timing known or known never.',
            no_contact='Shape rejection occurs but every physical H1-H4 contact count is known zero.',
            unknown='Contact or shape onset cannot be determined; not an absence-of-contact assertion.',
            branch_anchor_counts='with_relation fields are nonexclusive across seeds/candidates; unanimous branch counts partition 48 branches.',
            posthoc_only=True,future_truth_used_for_prediction=False))


def analyze_records(rows, branches=()):
    """Pure numeric fixture/public helper; preserve all 48 L physical identities."""
    rows,branches=list(rows),list(branches)
    index,physical=_inventory(rows,branches)
    cells=[_cell(index,physical,mode,h,stratum) for mode,hs in (('AR',(1,2,3,4)),('TF',(1,2,4)))
           for h in hs for stratum in STRATA]
    branch_table=[]
    for a in ANCHORS:
        for c in CANDIDATES:
            branch_table.append(dict(anchor=a,candidate=c,status='available' if physical[a,c,1]['available'] else 'original_missing',
                horizons=[dict(horizon=h,contact_by_h4=physical[a,c,h]['contact'],rotation_degrees=physical[a,c,h]['rotation_degrees'],
                    truth=physical[a,c,h]['truth'],truth_complete=all(v is not None for v in physical[a,c,h]['truth'].values())) for h in range(1,5)]))
    return dict(schema_version=1,development_only=True,independent_validation=False,
        numeric_statistics=dict(branch_table=branch_table,cells=cells,definitions=dict(
            planned_branches=48,available_branches=47,planned_anchors=12,seeds=list(SEEDS),
            strata='Physical contact through H4; strict wrapped actual rotation >5deg relative to latest, separately at each horizon.',
            TF='H1 references AR first step; only H2/H4 independently sampled. No TF H3.',
            coverage='Independent coverage flag is displayed separately; physical coverage errors and ranking retain old object+shape+coverage eligibility.',
            errors='Existing world-pixel/degree/fraction metrics; missing truth/measurement remains None, never zero.',
            weighting='Row statistics disclose noise repeats; seed mean within branch, candidate mean within anchor, anchors equally weighted.',
            thresholds=THRESHOLDS)),
        first_failures=_first_failures(index,physical),ranking_analysis=_ranking(list(index.values())),
        contact_onset_timing=_contact_onset_timing(index,branches))


def analyze_run(run_dir):
    """Read saved numeric JSON only, following the run's already-bound report."""
    root=Path(run_dir)
    results=root/'results' if (root/'results').is_dir() else root
    binding=json.loads((results/'binding.json').read_bytes())
    reports=[Path(p) for p in binding.get('asset_sha256',{}) if Path(p).name=='report.json']
    if len(reports)!=1:
        raise ValueError('One bound original physical report is required')
    content=reports[0].read_bytes()
    if hashlib.sha256(content).hexdigest()!=binding['asset_sha256'][str(reports[0])]:
        raise ValueError('Bound original physical report changed')
    report=json.loads(content)
    branches=[]
    for row in report['branches']:
        contacts=[x.get('contacts') for x in row.get('actual',[])]
        contact_by_h4=(True if any(_finite(v) and v>0 for v in contacts) else
            False if len(contacts)==4 and all(_finite(v) and v==0 for v in contacts) else None)
        branches.append(dict(anchor=row['anchor'],candidate=row['candidate'],latest=row.get('latest'),
            truth=[x.get('truth') for x in row.get('actual',[])],
            contacts_by_horizon=contacts,
            contact_by_h4=contact_by_h4 if row.get('status')=='complete' else None))
    rows=[json.loads(path.read_bytes()) for path in sorted((results/'records').glob('*.json'))]
    result=analyze_records(rows,branches)
    result['source']=dict(run_dir=str(root.resolve()),numeric_records=len(rows),
        original_report=str(reports[0]),original_report_sha256=hashlib.sha256(content).hexdigest(),
        model_loads=0,forwards=0,decoder_calls=0,physics_operations=0,
        note='Existing forecast measurements only. No NPY decoding or image refit was necessary.')
    return result


def save_artifacts(result, destination):
    """Exclusive writes into a new analysis directory; never overwrite evidence."""
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    names={'numeric_statistics':'numeric-statistics.json','first_failures':'first-failures.json',
           'ranking_analysis':'ranking-analysis.json','contact_onset_timing':'contact-onset-timing.json'}
    paths={key:destination/name for key,name in names.items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Offline analysis artifacts already exist')
    written={}
    for key,path in paths.items():
        value=dict(schema_version=result['schema_version'],source=result.get('source'),analysis=result[key])
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        with path.open('xb') as stream:
            stream.write(data)
        written[key]=dict(path=str(path.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return written
