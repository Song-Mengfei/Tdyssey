"""Prepare CONTACT-PILOT2 scenes without physics; freeze only after regression release.

All CP1 initial scenes/targets repeat unchanged. Only twelve cases are fresh.
Validation states are read solely for duplicate exclusion, never outcome selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Point

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r3a_collection import (
    LETTERS,LOCATIONS,RULES,scene_candidate,segments,geometry_at,wall_gaps,
    near_initial,rot,_candidate,_free,
)
from tdyssey.native_rgb_selected_physics import SELECTED_PROTOCOL_SHA256

BASE=ROOT/'artifacts/native-rgb-supervised-20261029'
R3A=BASE/'r3a'
CP1=BASE/'contact-pilot1-20261030'
AUDIT=BASE/'contact-audit-20261030'
STAGE=BASE/'contact-pilot2-20261030'
OUT=STAGE/'plan'
SEED_BASE=90521030
CONTACT_KEYS=('arc_id','point_local','outward_normal_local','staging_world',
    'staging_distance_world_px','signed_lever_world_px','resultant_signed_lever_world_px',
    'physical_local_CoG','drive_tangent_ratio')
PRIORITY=['CP1-A-X-03','CP1-A-T-03','CP1-A-B-04']+[f'CP1-B-{letter}-04'for letter in LETTERS]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def state_sha(state):return hashlib.sha256(np.asarray(state,np.float64).tobytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)
def requested(contact):
    return dict(arc_id=int(contact['arc_id']),ring=0,point_local=deepcopy(contact['point_local']),
                outward_normal_local=deepcopy(contact['outward_normal_local']))


def target_weak_contact(letter,candidate,coverage):
    """Rank finite legal exterior candidates by observed TRAIN/CP1 arc support."""
    state=np.asarray(candidate['initial_state15'],np.float64)
    geometry=geometry_at(letter,state);free=_free(geometry)
    for sequence,segment in sorted(enumerate(segments(letter)),key=lambda pair:(coverage[int(pair[1]['id'])],int(pair[1]['id']))):
        contact,_=_candidate(letter,state,geometry,segment,'generic','center',free)
        if contact is None or contact['translation_room_world_px']<16:continue
        # The controller must acquire the requested normal first. This frozen
        # task tangent is applied only after its independently measured arc hit.
        contact['drive_tangent_ratio']=.8*(1 if sequence%2==0 else -1)
        result=deepcopy(candidate)
        result['selected_contact']={k:deepcopy(contact[k])for k in CONTACT_KEYS}
        result['arc_start']=sequence
        result['initial_state15'][:2]=contact['staging_world']
        result['initialization']['pusher']=contact['staging_world']
        return result
    return None


def challenge_direction(letter,candidate):
    """Current geometry criterion, not a prediction of a future physical outcome."""
    state=np.asarray(candidate['initial_state15'],float);geometry=geometry_at(letter,state)
    gaps=wall_gaps(geometry);wall=min(gaps,key=gaps.get)
    inward={'left':np.array((1.,0.)),'right':np.array((-1.,0.)),
            'top':np.array((0.,1.)),'bottom':np.array((0.,-1.))}[wall]
    contact=candidate['selected_contact'];normal=rot(state[4])@np.asarray(contact['outward_normal_local'])
    tangent=np.array((-normal[1],normal[0]));direction=-normal+float(contact['drive_tangent_ratio'])*tangent
    direction/=np.linalg.norm(direction)
    return dict(nearest_wall=wall,initial_wall_gap_world_px=float(gaps[wall]),
                selected_drive_inward_dot=float(direction@inward))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if(OUT/'cases.json').exists()or(OUT/'cases-draft.json').exists():
        raise FileExistsError('CP2 draft/frozen plan already exists; never silently resample cases')
    cp1_plan=read(CP1/'plan/cases.json');cp1_manifest=read(CP1/'manifest.json')
    r3a_manifest=read(R3A/'manifest.json');r3a_plan=read(R3A/'plan.json')
    if not cp1_manifest['closed']or len(cp1_plan['cases'])!=48 or len(cp1_manifest['entries'])!=48:
        raise ValueError('Closed one-pass CP1 evidence required')
    cp1_entries={entry['case_id']:entry for entry in cp1_manifest['entries']}
    if Counter(c['group']for c in cp1_plan['cases'])!=Counter(A_matched=24,B_fresh=24):
        raise ValueError('Original CP1 group identities changed')
    old_initial={letter:[]for letter in LETTERS};old_seeds=set();verified_sources=0
    for entry in r3a_manifest['entries']:
        path=R3A/entry['file']
        if sha(path)!=entry['sha256']:raise ValueError('R3A source changed: '+entry['trajectory_id'])
        with np.load(path,allow_pickle=False)as archive:state=archive['states'][0].copy()
        old_initial[entry['letter']].append(('R3A:'+entry['trajectory_id'],state))
        old_seeds.add(entry['seed']);verified_sources+=1
    for slot in r3a_plan['slots']:old_seeds.update(c['seed']for c in slot['attempts'])
    repeats=[]
    for old in cp1_plan['cases']:
        entry=cp1_entries[old['case_id']];path=CP1/entry['file'];meta=CP1/entry['metadata_file']
        if sha(path)!=entry['sha256']or sha(meta)!=entry['metadata_sha256']:
            raise ValueError('CP1 source changed: '+old['case_id'])
        with np.load(path,allow_pickle=False)as archive:
            initial=archive['states'][0].copy();rgb0=bytes(archive['rgb_frame_sha256'][0]).hex()
        if not np.allclose(initial,old['candidate']['initial_state15'],rtol=0,atol=1e-8)or np.any(initial[5:]!=0):
            raise ValueError('CP1 initial state differs from its legal original plan')
        candidate=deepcopy(old['candidate']);candidate['initial_state15']=initial.tolist()
        group='CP1_A_repeat'if old['group']=='A_matched'else'CP1_B_repeat'
        cid=old['case_id'].replace('CP1-','CP2-R-',1)
        cp1_baseline=dict(trajectory_id=old['case_id'],file=str(path.resolve()),sha256=entry['sha256'],
            metadata_file=str(meta.resolve()),metadata_sha256=entry['metadata_sha256'],status=entry['status'],
            charged_controls=entry['charged_controls'],completed_controls=entry['completed_controls'])
        row=dict(case_id=cid,source_cp1_case_id=old['case_id'],group=group,letter=old['letter'],split='train',
            scenario=old['scenario'],family_id=old['family_id'],role=old['role'],
            independence='repeated_existing_TRAIN_family',new_independent_family=False,
            slot=deepcopy(old['slot']),candidate=candidate,
            baseline=cp1_baseline,baseline_npz=cp1_baseline['file'],cp1_baseline=deepcopy(cp1_baseline),
            r3a_baseline=deepcopy(old['baseline']),expected_initial_state_sha256=state_sha(initial),
            expected_initial_rgb_sha256=rgb0,initial_requested_arc=requested(candidate['selected_contact']),
            original_cp1_candidate_state_sha256=state_sha(old['candidate']['initial_state15']),
            task_recontact_start_control=old['task_recontact_start_control'],
            task_withdraw_start_control=old['task_withdraw_start_control'],
            task_withdraw_end_control_exclusive=old['task_withdraw_end_control_exclusive'],
            maximum_controls=48,maximum_physical_attempts=1,
            comparison_versions=['R3A','CP1','CP2']if old['baseline']is not None else['CP1','CP2'],
            source_cp1_group=old['group'])
        repeats.append(row)
        old_initial[old['letter']].append(('CP1:'+old['case_id'],initial))
        old_seeds.add(candidate['seed'])
    # Actual arc memberships, counted once per control. CP1 repeated cases are
    # additional observed time, not new independent-family support.
    coverage={letter:Counter()for letter in LETTERS}
    with np.load(AUDIT/'core/controls.npz',allow_pickle=False)as archive:
        index=archive['control_arc_offsets'];arcs=archive['control_arc_ids'];split=archive['split'];letters=archive['letter']
        for row in np.flatnonzero(split==0):coverage[LETTERS[int(letters[row])]].update(map(int,arcs[index[row]:index[row+1]]))
    for case in cp1_plan['cases']:
        entry=cp1_entries[case['case_id']]
        with np.load(CP1/entry['file'],allow_pickle=False)as archive:
            tags=archive['contact_tags'];offsets=archive['contact_arc_offsets'];arcs=archive['contact_arc_ids']
            by_control={}
            for point in np.flatnonzero((tags[:,1]==0)&(tags[:,2]==1)):
                by_control.setdefault(int(tags[point,0])//10,set()).update(map(int,arcs[offsets[point]:offsets[point+1]]))
        for ids in by_control.values():coverage[case['letter']].update(ids)
    new_initial={letter:[]for letter in LETTERS};new_seeds=set();rejections=[];fresh=[]
    regions={'T':('left-top','right'),'L':('right-top','bottom'),'O':('right-bottom','left-top'),
             'C':('left-bottom','top'),'X':('right-bottom','left-top'),'B':('right-top','bottom')}
    for li,letter in enumerate(LETTERS):
        for position in range(2):
            role='fresh_nearwall_recovery_challenge'if position==0 else'fresh_requested_weak_exterior'
            scenario=('rotation_negative'if letter=='X'else'release_recontact')if position==0 else'grazing'
            location=regions[letter][position];found=None
            eligible=[index for index in range(2400+li*144+position*36,2400+li*144+position*36+240)
                      if LOCATIONS[(index//12+(index%12)*4)%9]==location]
            for index in eligible:
                for static_try in range(64):
                    candidate,reason=scene_candidate(letter,'train',index,0,static_try,SEED_BASE,scenario)
                    if candidate is not None and position==1:
                        candidate=target_weak_contact(letter,candidate,coverage[letter])
                        if candidate is None:reason='no_feasible_exterior_weak_target'
                    challenge=None
                    if candidate is not None and position==0:
                        challenge=challenge_direction(letter,candidate)
                        if challenge['selected_drive_inward_dot']>-.25:
                            candidate=None;reason='challenge_drive_not_towards_nearest_wall'
                    if candidate is None:
                        rejections.append(dict(letter=letter,role=role,index=index,static_try=static_try,reason=reason));continue
                    state=np.asarray(candidate['initial_state15'],np.float64)
                    duplicate=next((tid for tid,other in old_initial[letter]+new_initial[letter]
                                    if state_sha(state)==state_sha(other)or near_initial(letter,state,other)),None)
                    reason=('seed_collision'if candidate['seed']in old_seeds|new_seeds else
                            'exact_or_near_initial:'+duplicate if duplicate is not None else None)
                    if reason:
                        rejections.append(dict(letter=letter,role=role,index=index,static_try=static_try,reason=reason));continue
                    geometry=geometry_at(letter,state)
                    if np.any(state[5:]!=0)or not _free(geometry).covers(Point(*state[:2])):
                        raise ValueError('Fresh scene is not a zero-dynamics legal pusher placement')
                    found=(index,candidate,challenge);break
                if found:break
            if found is None:raise RuntimeError('Finite static scene catalogue exhausted for '+letter+role)
            index,candidate,challenge=found;cid=f'CP2-N-{letter}-{position+1:02d}';family='F-'+cid
            state=np.asarray(candidate['initial_state15'],np.float64);new_initial[letter].append((cid,state));new_seeds.add(candidate['seed'])
            fresh.append(dict(case_id=cid,source_cp1_case_id=None,group='CP2_fresh',letter=letter,split='train',
                scenario=scenario,family_id=family,role=role,independence='new_independent_TRAIN_family',new_independent_family=True,
                slot=dict(slot_id=cid,family_id=family,letter=letter,split='train',index=index,scenario=scenario),candidate=candidate,
                baseline=None,baseline_npz=None,cp1_baseline=None,r3a_baseline=None,
                expected_initial_state_sha256=state_sha(state),expected_initial_rgb_sha256=None,
                initial_requested_arc=requested(candidate['selected_contact']),
                task_recontact_start_control=26,task_withdraw_start_control=18 if scenario=='release_recontact'else 22,
                task_withdraw_end_control_exclusive=26,maximum_controls=48,maximum_physical_attempts=1,
                comparison_versions=['CP2'],selection_evidence=dict(initial_region_rule=location,geometry_challenge=challenge,
                    initial_target_arc_id=candidate['selected_contact']['arc_id'],
                    observed_TRAIN_CP1_arc_control_support=coverage[letter][candidate['selected_contact']['arc_id']],
                    targeted_scene_distribution_not_natural=True)))
    ordered=[];by_source={c['source_cp1_case_id']:c for c in repeats}
    for source_id in PRIORITY:ordered.append(by_source[source_id])
    ordered.extend(c for c in repeats if c['source_cp1_case_id']not in PRIORITY);ordered.extend(fresh)
    if len(ordered)!=60 or len({c['case_id']for c in ordered})!=60 or len(new_seeds)!=12:raise AssertionError('CP2 case counts changed')
    if Counter(c['group']for c in ordered)!=Counter(CP1_A_repeat=24,CP1_B_repeat=24,CP2_fresh=12):raise AssertionError('CP2 groups changed')
    for letter in LETTERS:
        if sum(c['letter']==letter and c['group']=='CP2_fresh'for c in ordered)!=2:raise AssertionError('Two new families per letter required')
    sources=[CP1/'plan/cases.json',CP1/'manifest.json',CP1/'controller-rules.json',R3A/'manifest.json',R3A/'plan.json',
        AUDIT/'core/controls.npz',ROOT/'configs/native_rgb_selected_physics.json',ROOT/'tdyssey/native_rgb_r3a_collection.py',Path(__file__)]
    payload=dict(kind='CONTACT-PILOT2-regression-gated-60case-draft.v1',draft_created_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen=False,physical_execution_released=False,requires_saved_state_regression_release=True,cases=ordered,case_count=60,
        repeated_existing_CP1_cases=48,new_independent_TRAIN_families=12,maximum_charged_controls_including_failures=2880,
        controls_per_case=48,maximum_attempts_per_case=1,automatic_replacements=False,add_to_R3A_training_pool=False,
        model_work_authorized=False,physical_protocol_sha256=SELECTED_PROTOCOL_SHA256,physical_half_life_seconds=.1,
        stop_entire_pass_on=['any_existing2px_physical_guard','unknown_engine_advance','engineering_failure'],
        frozen_failure_priority_source_cases=PRIORITY,
        predeclared_paired_media_cases=['CP2-R-A-X-03','CP2-R-A-T-03','CP2-R-B-B-04'],
        selection_rules=dict(repeats='all48 exact CP1 saved initial states and original selected contacts/tasks; no outcome-based replacement',
            fresh=['one current-geometry boundary recovery/nearwall challenge; Xnegative rotation',
                   'one feasible low-observed-support exterior requested arc'],
            fresh_seed_base=SEED_BASE,fresh_regions=regions,
            challenge_geometry_filter='initial selected drive inward dot<=-.25 relative to nearest current wall',
            weak_arc_rank='actual postsolve control-arc memberships over R3A TRAIN plus CP1; repeats are observed time, not independent support',
            duplicate_thresholds=RULES['initial_near_duplicate'],validation_states_for_duplicate_exclusion_only=True),
        engineering_targets=dict(contact_and_obvious_fraction_ge=.4,moving_pusher_no_contact_fraction_le=.25,
            observed_recontact_median_seconds_le=.5,observed_recontact_p90_seconds_le=1.,unrecovered_fraction_le=.05,
            physical_guard_failures_eq=0,meaning='predeclared mechanism targets, not population or model-quality guarantees'),
        recovery_definition='verified at least one complete no-AO interval during fixed withdrawal before first actual postsolve after task26; nonseparation is not0delay success',
        caveats=['Only12 new independent families; CP1 fresh24 are now repeated seen scenes.',
            'All original cases and fresh challenges are targeted/bias-disclosed mechanism diagnostics.',
            'No future physics outcome is used in scene choice; no validation/heldout outcome used.',
            'Guard/unknown/system failure stops all later planned cases; unused budget never replaces a failure.',
            'Requested-arc hit must be measured from actual postsolve membership, impulse support and movement, not proximity or neighboring arc.',
            'Closed O/B inner holes are outside this pilot; zero coverage is not proof of unreachability.'],
        source_bindings=[bind(p)for p in sources],prephysics=True,new_physics_controls=0,new_environment_instances=0,
        new_renderer_calls=0,new_model_calls=0,Torch_imported='torch'in sys.modules)
    if payload['Torch_imported']:raise RuntimeError('Planner unexpectedly imported Torch')
    write(OUT/'cases-draft.json',payload)
    write(OUT/'static-rejections.json',dict(count=len(rejections),rejections=rejections,physics_advance=0))
    write(OUT/'preflight-draft.json',dict(kind='CONTACT-PILOT2-static-scene-preflight.v1',passed=True,frozen=False,
        draft_cases_sha256=sha(OUT/'cases-draft.json'),planned_cases=60,repeated_CP1_cases=48,new_independent_families=12,
        source_R3A_NPZ_verified=verified_sources,source_CP1_NPZ_metadata_verified=48,repeat_initial_state_and_targets_preserved=True,
        fresh_unique_seeds=12,fresh_exact_or_declared_near_R3A_TRAIN_VAL_CP1_or_new=0,
        all_initial_dynamics_zero=True,all_initial_pusher_positions_geometry_legal=True,
        static_rejections=len(rejections),regression_release_still_required=True,physics_controls=0,model_calls=0))
    (OUT/'SELECTION_NOTES.md').write_text('''# CONTACT-PILOT2 draft case catalogue

The original48 CP1 cases repeat exact saved initial full states and original
targets/tasks. Their prior results are baselines, not new independent data. The
first9 are X03,T03,B04 and the six old weak-arc cases; the remainder of the
original48 and twelve new families follow in a fixed order. No failure is replaced.

Twelve fresh TRAIN families have zero dynamics and current-geometry feasible
pusher placements. One per letter is a boundary recovery/nearwall challenge with
an initial drive directed partly towards the nearest wall; X uses negative
rotation. One per letter requests a low-support exterior arc using R3A TRAIN and
CP1 actual postsolve memberships. These geometric priors do not guarantee future
arc hit, pushing or physical safety. Validation initial states only exclude exact
or declared near duplicates; validation outcomes and heldout letters are unused.

cases-draft.json is not a physics release. --freeze requires a root regression
release binding this draft, final controller and safety sources and passed real
saved-state regression evidence. Only after that gate is cases.json written.
''',encoding='utf-8')
    size=sum(p.stat().st_size for p in OUT.iterdir()if p.is_file())
    if size>300*1024:raise RuntimeError('Draft plan exceeds300KiB: '+str(size))
    print(json.dumps(dict(draft=True,frozen=False,cases=60,repeats=48,fresh=12,bytes=size,
        draft_cases_sha256=sha(OUT/'cases-draft.json'),static_rejections=len(rejections),physics_controls=0)))


def freeze(receipt_path):
    draft=OUT/'cases-draft.json';target=OUT/'cases.json';receipt_path=Path(receipt_path).resolve()
    if target.exists():raise FileExistsError('Final CP2 plan already frozen')
    receipt=read(receipt_path)
    if not(receipt.get('passed')is True and receipt.get('all_required_regressions_passed')is True
           and receipt.get('physical_plan_freeze_authorized')is True):raise PermissionError('Passed real-state regression release required')
    if receipt.get('draft_cases_sha256')!=sha(draft):raise ValueError('Regression release does not bind this exact draft')
    if not receipt.get('bindings'):raise ValueError('Regression release must bind final sources and evidence')
    for item in receipt['bindings']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Regression binding changed: '+item['path'])
    payload=read(draft)
    for item in payload['source_bindings']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Draft historical input changed: '+item['path'])
    payload.update(kind='CONTACT-PILOT2-frozen-60case-plan.v1',frozen=True,physical_execution_released=True,
        regression_release=bind(receipt_path),frozen_at_utc=datetime.now(timezone.utc).isoformat())
    # Same-directory file replacement removes the draft duplicate; no old stage
    # is modified. The case list and ordering are unchanged by this operation.
    write(draft,payload);draft.replace(target)
    preflight=read(OUT/'preflight-draft.json');preflight.update(frozen=True,cases_sha256=sha(target),
        regression_release=bind(receipt_path),regression_release_still_required=False)
    write(OUT/'preflight.json',preflight)
    print(json.dumps(dict(frozen=True,cases=60,cases_sha256=sha(target),regression_release_sha256=sha(receipt_path),physics_controls=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--draft',action='store_true')
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--regression-receipt',type=Path);args=parser.parse_args()
    if args.draft and not args.freeze:prepare();return
    if args.freeze and not args.draft and args.regression_receipt:freeze(args.regression_receipt);return
    parser.error('Choose --draft or --freeze --regression-receipt PATH')


if __name__=='__main__':main()
