"""Freeze biased matched diagnostics and fresh TRAIN scenes without physics.

Only existing TRAIN outcomes choose difficult matched scenes/weak exterior arcs.
Seen validation initial states are read solely to reject scene duplicates.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tdyssey.native_rgb_r3a_collection import (
    LETTERS, LOCATIONS, RULES, identity, near_initial, scene_candidate, segments,
    geometry_at, _candidate, _free,
)
from tdyssey.native_rgb_selected_physics import SELECTED_PROTOCOL_SHA256

BASE = ROOT/'artifacts/native-rgb-supervised-20261029'
R3A = BASE/'r3a'
AUDIT = BASE/'contact-audit-20261030'
OUT = BASE/'contact-pilot1-20261030/plan'
SEED_BASE = 90481030
CONTACT_KEYS = ('arc_id','point_local','outward_normal_local','staging_world',
    'staging_distance_world_px','signed_lever_world_px','resultant_signed_lever_world_px',
    'physical_local_CoG','drive_tangent_ratio')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def state_sha(state):
    return hashlib.sha256(np.asarray(state, np.float64).tobytes()).hexdigest()


def target_weak_contact(letter, candidate, coverage):
    """Finite exterior geometry ranked by TRAIN-only observed arc support."""
    state = np.asarray(candidate['initial_state15'], np.float64)
    geometry = geometry_at(letter, state)
    free = _free(geometry)
    arcs = segments(letter)
    for sequence, segment in sorted(enumerate(arcs), key=lambda pair: (coverage[int(pair[1]['id'])], int(pair[1]['id']))):
        contact, _ = _candidate(letter, state, geometry, segment, 'generic', 'center', free)
        if contact is None or contact['translation_room_world_px'] < 16:
            continue
        contact['drive_tangent_ratio'] = .8*(1 if sequence % 2 == 0 else -1)
        result = deepcopy(candidate)
        result['selected_contact'] = {key:deepcopy(contact[key]) for key in CONTACT_KEYS}
        result['arc_start'] = int(sequence)
        result['initial_state15'][:2] = contact['staging_world']
        result['initialization']['pusher'] = contact['staging_world']
        return result
    return None


def make_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'cases.json').exists():
        raise FileExistsError('Frozen pilot cases already exist; do not replace them.')
    manifest = read(R3A/'manifest.json')
    old_plan = read(R3A/'plan.json')
    metadata = read(AUDIT/'core/trajectories.json')['trajectories']
    with np.load(AUDIT/'core/controls.npz', allow_pickle=False) as archive:
        core = {key:archive[key] for key in archive.files}
    entries = manifest['entries']
    if len(entries) != 912 or not manifest['collection_closed']:
        raise ValueError('Closed 912-scene source required')
    source_by_id = {entry['trajectory_id']:entry for entry in entries}
    slots = {slot['slot_id']:slot for slot in old_plan['slots']}
    old_initial = {letter:[] for letter in LETTERS}
    old_seeds = set()
    for entry in entries:
        if sha(R3A/entry['file']) != entry['sha256']:
            raise ValueError('Old source NPZ changed: '+entry['trajectory_id'])
        with np.load(R3A/entry['file'], allow_pickle=False) as archive:
            state = archive['states'][0].copy()
        old_initial[entry['letter']].append((entry['trajectory_id'], state))
        old_seeds.add(entry['seed'])
    for slot in old_plan['slots']:
        old_seeds.update(c['seed'] for c in slot['attempts'])
    coverage = {letter:Counter() for letter in LETTERS}
    for row in np.flatnonzero(core['split'] == 0):
        lo, hi = core['control_arc_offsets'][row:row+2]
        letter = LETTERS[int(core['letter'][row])]
        coverage[letter].update(map(int, core['control_arc_ids'][lo:hi]))

    cases = []
    used_source = set()
    for letter in LETTERS:
        train = [record for record in metadata if record['split'] == 'train' and record['letter'] == letter]
        release = [record for record in train if record['scenario'] == 'release_recontact']
        # Prefer one unrecovered release and one slow observed release. Frozen
        # ranks do not consult any new rollout or seen-val outcome.
        first = sorted(release, key=lambda record: (
            not record['recontact_censored'], -(record['recontact_latency_seconds'] or 0), record['trajectory_id']))[0]
        observed = [record for record in release if not record['recontact_censored'] and record['trajectory_id'] != first['trajectory_id']]
        second = sorted(observed, key=lambda record: (-record['recontact_latency_seconds'], record['trajectory_id']))[0]
        chosen = [('slow_or_unrecovered_recontact', first), ('slow_observed_recontact', second)]
        used_source.update(record['trajectory_id'] for _,record in chosen)
        def wallstall(record):
            ti=record['trajectory_index'];rows=slice(ti*48,(ti+1)*48)
            return int(((core['contact_substeps'][rows]>0)&(core['object_wall_substeps'][rows]>0)&(core['motion_class'][rows]==0)).sum())
        wall = sorted((record for record in train if record['trajectory_id'] not in used_source),
                      key=lambda record:(-wallstall(record),record['trajectory_id']))[0]
        chosen.append(('contact_wall_nearstatic', wall));used_source.add(wall['trajectory_id'])
        boundary = [record for record in train if record['trajectory_id'] not in used_source
                    and record['initial_space'] != 'center' and record['scenario'].startswith('rotation_')
                    and (letter != 'X' or record['scenario']=='rotation_negative')]
        def source_candidate(record):
            entry=source_by_id[record['trajectory_id']]
            attempt=read(R3A/entry['metadata_file'])['attempt_index']
            return slots[entry['slot_id']]['attempts'][attempt]
        weak = sorted(boundary, key=lambda record:(
            coverage[letter][source_candidate(record)['selected_contact']['arc_id']], record['trajectory_id']))[0]
        chosen.append(('weak_exterior_boundary_rotation', weak));used_source.add(weak['trajectory_id'])
        for index, (role, record) in enumerate(chosen, 1):
            entry=source_by_id[record['trajectory_id']]
            original_slot=slots[entry['slot_id']]
            candidate=deepcopy(source_candidate(record))
            actual=next(state for tid,state in old_initial[letter] if tid==record['trajectory_id'])
            if not np.allclose(candidate['initial_state15'],actual,rtol=0,atol=1e-8) or np.any(actual[5:]!=0):
                raise ValueError('Saved matched initial state differs from its frozen candidate')
            original_candidate_state_sha=state_sha(candidate['initial_state15'])
            candidate['initial_state15']=actual.tolist()
            slot={key:deepcopy(original_slot[key])for key in ('slot_id','family_id','letter','split','index','scenario')}
            baseline=dict(trajectory_id=entry['trajectory_id'],file=str((R3A/entry['file']).resolve()),sha256=entry['sha256'],
                          metadata_file=str((R3A/entry['metadata_file']).resolve()),metadata_sha256=entry['metadata_sha256'])
            cases.append(dict(case_id=f'CP1-A-{letter}-{index:02d}',group='A_matched',letter=letter,
                split='train',scenario=slot['scenario'],family_id=slot['family_id'],role=role,
                independence='matched_existing_TRAIN_family',slot=slot,candidate=candidate,baseline=baseline,
                baseline_npz=baseline['file'],expected_initial_state_sha256=state_sha(actual),
                original_candidate_initial_state_sha256=original_candidate_state_sha,
                task_recontact_start_control=26,task_withdraw_start_control=18 if slot['scenario']=='release_recontact' else 22,
                task_withdraw_end_control_exclusive=26,maximum_controls=48,maximum_physical_attempts=1,
                selection_evidence=dict(old_recontact_censored=record['recontact_censored'],
                    old_observed_recontact_delay_seconds=record['recontact_latency_seconds'],
                    old_contact_wall_nearstatic_controls=wallstall(record),
                    initial_target_arc_id=candidate['selected_contact']['arc_id'],
                    TRAIN_actual_arc_control_support=coverage[letter][candidate['selected_contact']['arc_id']],
                    biased_hard_case_selection=True)))

    fresh_initial={letter:[] for letter in LETTERS};rejections=[];new_seeds=set()
    desired_regions={
        'T':('center','left-top','left','bottom'), 'L':('center','right-top','right','top'),
        'O':('center','right-bottom','bottom','left'), 'C':('center','left-bottom','top','right'),
        'X':('left-bottom','right-top','right-bottom','left-top'),
        'B':('center','right-bottom','right','bottom')}
    for li,letter in enumerate(LETTERS):
        for position in range(4):
            role=('fresh_recontact_1','fresh_recontact_2','fresh_boundary_rotation','fresh_weak_exterior')[position]
            scenario='release_recontact' if position<2 else ('rotation_negative' if letter=='X' or li%2==0 else 'rotation_positive') if position==2 else 'grazing'
            location=desired_regions[letter][position]
            found=None
            # Finite pre-physics deterministic candidates. Both scene index and
            # static_try affect independent seed; no simulated outcome is read.
            eligible=[index for index in range(300+li*96+position*12,300+li*96+position*12+180)
                      if LOCATIONS[(index//12+(index%12)*4)%9]==location]
            for index in eligible:
                for static_try in range(64):
                    candidate,reason=scene_candidate(letter,'train',index,0,static_try,SEED_BASE,scenario)
                    if candidate is not None and position==3:
                        candidate=target_weak_contact(letter,candidate,coverage[letter])
                        if candidate is None:reason='no_weak_exterior_geometry_feasible'
                    if candidate is None:
                        rejections.append(dict(letter=letter,role=role,index=index,static_try=static_try,reason=reason));continue
                    state=np.asarray(candidate['initial_state15'],np.float64)
                    duplicate=next((tid for tid,old in old_initial[letter]+fresh_initial[letter]
                                    if state_sha(old)==state_sha(state) or near_initial(letter,state,old)),None)
                    reason=('seed_collision' if candidate['seed'] in old_seeds|new_seeds else
                            'exact_or_declared_near_initial:'+duplicate if duplicate is not None else None)
                    if reason:
                        rejections.append(dict(letter=letter,role=role,index=index,static_try=static_try,reason=reason));continue
                    if np.any(state[5:]!=0) or not _free(geometry_at(letter,state)).covers(Point(*state[:2])):
                        raise ValueError('Fresh legal initial placement failed')
                    found=(index,candidate);break
                if found:break
            if found is None:raise RuntimeError('Finite fresh pre-physics catalogue exhausted: '+letter+role)
            index,candidate=found;cid=f'CP1-B-{letter}-{position+1:02d}';family='F-'+cid
            fresh_initial[letter].append((cid,np.asarray(candidate['initial_state15'],np.float64)))
            new_seeds.add(candidate['seed'])
            slot=dict(slot_id=cid,family_id=family,letter=letter,split='train',index=index,scenario=scenario)
            cases.append(dict(case_id=cid,group='B_fresh',letter=letter,split='train',scenario=scenario,
                family_id=family,role=role,independence='new_independent_TRAIN_family',slot=slot,candidate=candidate,
                baseline=None,baseline_npz=None,expected_initial_state_sha256=state_sha(candidate['initial_state15']),
                task_recontact_start_control=26,task_withdraw_start_control=18 if scenario=='release_recontact' else 22,
                task_withdraw_end_control_exclusive=26,maximum_controls=48,maximum_physical_attempts=1,
                selection_evidence=dict(initial_region_rule=location,
                    initial_target_arc_id=candidate['selected_contact']['arc_id'],
                    TRAIN_actual_arc_control_support=coverage[letter][candidate['selected_contact']['arc_id']],
                    targeted_scene_distribution_not_natural=True)))
    assert len(cases)==48 and len(new_seeds)==24 and len(used_source)==24
    assert Counter((case['group'],case['letter'])for case in cases)==Counter({(group,letter):4 for group in('A_matched','B_fresh')for letter in LETTERS})
    # Explicit duplicate-check sanity: each matched source is recognized as its
    # own near initial scene, whereas all24new scenes pass the same rule.
    assert all(near_initial(case['letter'],case['candidate']['initial_state15'],case['candidate']['initial_state15'])for case in cases)
    sources=[R3A/'manifest.json',R3A/'plan.json',AUDIT/'core/manifest.json',AUDIT/'core/controls.npz',AUDIT/'core/trajectories.json',
             ROOT/'configs/native_rgb_selected_physics.json',ROOT/'tdyssey/native_rgb_r3a_collection.py',Path(__file__)]
    payload=dict(kind='CONTACT-PILOT1-prephysics-48cases.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases,case_count=48,paired_existing_TRAIN_cases=24,new_independent_TRAIN_families=24,
        maximum_charged_controls_including_failures=2304,controls_per_case=48,maximum_attempts_per_case=1,
        automatic_replacements=False,add_to_R3A_training_pool=False,model_work_authorized=False,
        physical_protocol_sha256=SELECTED_PROTOCOL_SHA256,physical_half_life_seconds=.1,
        selection_rules=dict(A=['one censored-preferred release_recontact','one slow observed release_recontact',
            'maximum actual AO+OW+nearstatic controls','minimum TRAIN initial-target arc support among boundary rotations; Xnegative'],
            B=['two release_recontact','one boundary rotation; Xnegative','one geometry-feasible weakest TRAIN exterior arc, grazing'],
            tie_break='lexicographic original trajectory ID / smallest arc ID',
            existing_near_thresholds=RULES['initial_near_duplicate'],fresh_seed_base=SEED_BASE,
            regions=desired_regions,weak_arc_coverage_source='TRAIN actual postsolve control-arc membership only; count each arc at most once percontrol',
            no_validation_outcome_used=True),
        engineering_targets=dict(contact_and_obvious_fraction_ge=.4,moving_pusher_no_contact_fraction_le=.25,
            observed_recontact_median_seconds_le=.5,observed_recontact_p90_seconds_le=1.,unrecovered_fraction_le=.05,
            meaning='pre-frozen small mechanism pilot goals; no population guarantee or retrospective R3A gate'),
        predeclared_paired_media_cases=['CP1-A-T-01','CP1-A-X-04'],
        caveats=['A restarts existing TRAIN families and is not24new independent support.',
            'A difficult-case selection and B targeted scene distribution are biased; report separately.',
            'A uses each exact old48control baseline without rerunning old physics.',
            'Every case retains its withdraw/recontact task and48control maximum; failures/partials remain charged.',
            'Uncovered exterior or inner-ring contact is not evidence of physical unreachability.'],
        source_bindings=[dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)for path in sources],
        prephysics=True,new_physics_controls=0,new_environment_instances=0,new_renderer_calls=0,new_model_calls=0,
        Torch_imported='torch' in sys.modules)
    if payload['Torch_imported']:raise RuntimeError('Planning imported Torch')
    write(OUT/'cases.json',payload)
    write(OUT/'static-rejections.json',dict(count=len(rejections),rejections=rejections,physical_advance=0))
    write(OUT/'preflight.json',dict(kind='CONTACT-PILOT1-cases-preflight.v1',passed=True,cases_sha256=sha(OUT/'cases.json'),
        planned_cases=48,matched_existing_families=24,new_families=24,new_seeds_unique=True,
        B_exact_or_declared_near_existing_TRAIN_or_VAL=0,B_exact_or_declared_near_other_pilot_initial=0,
        A_matched_state_and_target_checked=True,source_NPZ_hashes_verified=912,
        matched_zero_initial_dynamics=True,fresh_zero_initial_dynamics=True,
        static_rejections=len(rejections),new_physics_controls=0,model_calls=0,
        original_R3A_manifest_sha256=sha(R3A/'manifest.json')))
    (OUT/'old-controller-snapshot.py').write_bytes((ROOT/'tdyssey/native_rgb_r3a_collection.py').read_bytes())
    (OUT/'selected-physics-snapshot.json').write_bytes((ROOT/'configs/native_rgb_selected_physics.json').read_bytes())
    size=sum(path.stat().st_size for path in OUT.iterdir()if path.is_file())
    if size>300*1024:raise RuntimeError('Plan exceeded300KiB estimate; preserve and report')
    print(json.dumps(dict(frozen=True,cases=48,matched=24,fresh=24,cases_sha256=sha(OUT/'cases.json'),plan_bytes=size,
                          static_rejections=len(rejections),model_calls=0,physics_controls=0)))


if __name__=='__main__':make_plan()
