"""Cold numeric integrity audit of the completed R3A independent-scene library.

Read each immutable scene once. No environment, physics, RGB rendering, model,
training or data selection. The live resource ledger is never polled: one read
is permitted only after manifest and index declare collection closed.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r3a_collection import (LETTERS,ContactPolicy,near_initial,
    phase_for,PHASES,rot,sha)
from tdyssey.native_rgb_training import atomic_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def safe(base,name):
    path=(base/name).resolve()
    if not path.is_relative_to(base.resolve()):raise ValueError('Reference escapes collection root: '+name)
    return path


def audit(manifest_path,*,out):
    started=time.perf_counter();manifest_path=Path(manifest_path).resolve();base=manifest_path.parent
    manifest=read(manifest_path);index=read(base/'index.json')
    if manifest.get('collection_closed') is not True or index.get('collection_closed') is not True:
        raise RuntimeError('Wait for the completed library; refusing a hot audit')
    if manifest.get('kind')!='R3A-compact-state-reference.v1':raise ValueError('Only actual R3A compact state library')
    # Exactly one ledger read, after closed manifests, not during collection.
    ledger=read(base/'resource-ledger.json');plan=read(base/'plan.json')
    issues=[];checks=Counter();counts=Counter();signatures=defaultdict(lambda:defaultdict(list));initials={l:[] for l in LETTERS};per_letter={}
    slots={row['slot_id']:row for row in plan['slots']};registry={row['slot_id']:row for row in index['slots']}
    details=[];near_pairs=[]
    def check(ok,label,context=None):
        checks['total']+=1
        if not bool(ok):issues.append(dict(check=label,context=context));checks['failed']+=1
        else:checks['passed']+=1
    entries=manifest['entries'];check(len(entries)==912,'successful_scene_count912',len(entries))
    check(ledger.get('closed') is True and ledger.get('pending_control') is None,'resource_ledger_closed_no_pending')
    check(len({e['family_id'] for e in entries})==len(entries),'unique_independent_successful_families')
    check(len({e['seed'] for e in entries})==len(entries),'unique_successful_scene_seeds')
    check(len({e['trajectory_id'] for e in entries})==len(entries),'unique_successful_trajectory_ids')
    check(manifest.get('protocol_sha256')=='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6','selected_protocol_unchanged')
    meta_by_id={};file_by_id={};success_ids=set();cached_attempt_counts={};cached_trajectory_hashes={}
    for entry in entries:
        tid=entry['trajectory_id'];success_ids.add(tid);letter=entry['letter'];sid=entry['slot_id'];scene=slots[sid]
        counts[(letter,entry['split'])]+=1
        path=safe(base,entry['file']);mp=safe(base,entry['metadata_file']);meta=read(mp);meta_by_id[tid]=meta;file_by_id[tid]=path
        cached_trajectory_hashes[tid]=sha(path)
        check(cached_trajectory_hashes[tid]==entry['sha256'],'trajectory_SHA',tid);check(sha(mp)==entry['metadata_sha256'],'metadata_SHA',tid)
        check(meta['status']=='completed' and meta['completed_controls']==48 and entry['controls']==48,'successful_status48',tid)
        check(all(meta[key]==entry[key] for key in ('trajectory_id','slot_id','family_id','letter','split','seed','protocol_sha256')),'manifest_metadata_identity',tid)
        check(scene['family_id']==entry['family_id'] and scene['split']==entry['split'] and scene['letter']==letter,'prephysics_family_split_unchanged',tid)
        candidate=scene['attempts'][meta['attempt_index']]
        check(candidate['seed']==entry['seed'],'frozen_candidate_seed',tid)
        ref=read(safe(base,meta['initial_identity_reference']));physics=ref['physics'];cog=np.asarray(physics['object']['center_of_gravity'],np.float64)
        check(ref['protocol_sha256']==entry['protocol_sha256'] and ref['physics_sha256']==meta['physics_sha256'],'fixed_physics_reference',tid)
        with np.load(path,allow_pickle=False) as z:
            states=z['states'];actions=z['actions'];sub=z['substep_states'];forces=z['substep_forces'];capvel=z['substep_velocity_after_cap'];penetrations=z['substep_penetrations'];flags=z['contact_flags']
            offsets=z['control_substep_offsets'];tags=z['contact_tags'];values=z['contact_values'];arcoff=z['contact_arc_offsets'];arcids=z['contact_arc_ids'];rgb_hash=z['rgb_frame_sha256']
            check(states.dtype==np.float64 and states.shape==(49,15),'states49_full15_float64',tid)
            check(actions.dtype==np.float64 and actions.shape==(48,2),'actions48_absolute_float64',tid)
            check(sub.dtype==np.float64 and sub.shape==(481,15),'substep_states481_full15_float64',tid)
            check(np.isfinite(states).all() and np.isfinite(sub).all() and np.isfinite(actions).all(),'finite_complete_states_actions',tid)
            check(np.array_equal(states,sub[::10]),'control_actual100Hz_state_chain_exact',tid)
            check(np.array_equal(z['attempted_actions'],actions),'attempted_executed_action_alignment',tid)
            check(z['control_valid'].shape==(48,) and z['control_valid'].dtype==bool and z['control_valid'].all(),'all_successful_controls_valid',tid)
            check(np.array_equal(z['control_time_seconds'],np.arange(49,dtype=np.float64)*.1),'10Hz_saved_timestamps',tid)
            check(np.array_equal(z['substep_control_index'],np.repeat(np.arange(48,dtype=np.uint8),10)),'100Hz_control_indices',tid)
            check(np.array_equal(z['substep_local_index'],np.tile(np.arange(10,dtype=np.uint8),48)),'100Hz_local_indices',tid)
            check(np.array_equal(offsets,np.arange(49,dtype=np.uint16)*10),'ten_substeps_per_complete_control',tid)
            check(z['substep_status'].shape==(480,) and (z['substep_status']==1).all() and z['substep_state_available'].all(),'all_success_substeps_known_available',tid)
            check(np.allclose(states[0],candidate['initial_state15'],rtol=0,atol=1e-8) and np.count_nonzero(states[0,5:])==0,'sole_frozen_initial_state_zero_velocity_force',tid)
            check((actions>=0).all() and (actions<=511).all(),'actions_legal_absolute_domain',tid)
            check((np.linalg.norm(actions-states[:-1,:2],axis=1)<=64+1e-8).all(),'actual_target_distance64_bound',tid)
            check((np.linalg.norm(np.diff(actions,axis=0),axis=1)<=96+1e-8).all(),'actual_target_change96_bound',tid)
            check(np.array_equal(z['control_phase'],np.asarray([PHASES.index(phase_for(i,entry['scenario'])) for i in range(48)],np.uint8)),'frozen48_phase_schedule',tid)
            check(((z['control_mode']>=0)&(z['control_mode']<=2)).all(),'command_mode_codes',tid)
            check(np.all(z['control_geometry_verified']|(z['control_mode']==0)),'moving_command_geometry_verified',tid)
            check(np.isfinite(forces).all() and forces.shape==(480,6),'compact_force_evidence_finite',tid)
            check(np.array_equal(forces[:,2:4],forces[:,4:6]),'saved_formula_and_engine_force_float32_equal',tid)
            previous=sub[:-1];targets=actions[z['substep_control_index']]
            reconstructed=physics['agent']['mass']*(physics['kp']*(targets-previous[:,:2])-physics['kv']*previous[:,5:7])
            norms=np.linalg.norm(reconstructed,axis=1);ratio=np.ones_like(norms);np.divide(10000.,norms,out=ratio,where=norms>10000.);clipped=reconstructed*ratio[:,None]
            check(np.allclose(forces[:,:2],reconstructed,rtol=5e-7,atol=.005),'independent_PD_unclipped_reconstruction_float32_tolerance',tid)
            check(np.allclose(forces[:,2:4],clipped,rtol=5e-7,atol=.005),'independent_PD_force_cap_reconstruction_float32_tolerance',tid)
            check((np.linalg.norm(forces[:,4:6].astype(np.float64),axis=1)<=10000.01).all(),'actual_engine_force_cap_float32_tolerance',tid)
            velocity=previous[:,5:7];speeds=np.linalg.norm(velocity,axis=1);velocity_ratio=np.ones_like(speeds);np.divide(200.,speeds,out=velocity_ratio,where=speeds>200.);expected_cap=velocity*velocity_ratio[:,None]
            check(np.allclose(capvel,expected_cap,rtol=5e-7,atol=.0001),'independent_prestep_velocity_cap_reconstruction',tid)
            check((np.linalg.norm(capvel.astype(np.float64),axis=1)<=200.001).all(),'prestep_speed_cap200_float32_tolerance',tid)
            check(penetrations.shape==(480,3) and np.isfinite(penetrations).all() and (penetrations>=0).all() and np.max(penetrations)<=2.,'all_100Hz_penetration_guard2',tid)
            check(flags.shape==(480,3) and ((flags>=0)&(flags<=3)).all(),'contact_pre_post_event_flag_codes',tid)
            k=len(tags);check(tags.shape==(k,5) and values.shape==(k,11) and np.isfinite(values).all(),'contact_numeric_ragged_shape_finite',tid)
            check(arcoff.shape==(k+1,) and arcoff[0]==0 and np.diff(arcoff).min(initial=0)>=0 and arcoff[-1]==len(arcids),'actual_contact_arc_offsets',tid)
            if k:
                check(((tags[:,0]>=0)&(tags[:,0]<480)).all() and ((tags[:,1]>=0)&(tags[:,1]<3)).all() and ((tags[:,2]>=0)&(tags[:,2]<2)).all(),'contact_substep_kind_phase_domains',tid)
                check(((tags[:,3]>=0)&(tags[:,3]<=1)).all() and ((tags[:,4]>=0)&(tags[:,4]<=31)).all(),'object_order_and_known_field_masks',tid)
                check(((flags[tags[:,0],tags[:,1]]&(1<<tags[:,2]))!=0).all(),'each_contact_point_matches_recorded_event_flag',tid)
                check(np.allclose(np.linalg.norm(values[:,4:6].astype(np.float64),axis=1),1.,atol=1e-5),'actual_contact_normal_unit',tid)
                haslever=(tags[:,4]&16)!=0
                check(((tags[haslever,1]==0)&(tags[haslever,2]==1)&((tags[haslever,4]&1)!=0)).all(),'actual_lever_only_postsolve_agentobject_knownpoint',tid)
                if haslever.any():
                    ss=sub[tags[haslever,0]+1];cc=np.cos(ss[:,4]);sn=np.sin(ss[:,4]);com=ss[:,2:4]+np.column_stack((cc*cog[0]-sn*cog[1],sn*cog[0]+cc*cog[1]))
                    relative=values[haslever,:2].astype(np.float64)-com;normal=values[haslever,4:6].astype(np.float64)
                    lever=-relative[:,0]*normal[:,1]+relative[:,1]*normal[:,0]
                    check(np.allclose(lever,values[haslever,10],rtol=1e-5,atol=.002),'independent_signed_normal_lever_float32_tolerance',tid)
            check(rgb_hash.dtype==np.uint8 and rgb_hash.shape==(49,32) and 'rgb' not in z.files and 'images' not in z.files,'49_collector_RGB_hashes_no_persisted_images',tid)
            check(array_sha(rgb_hash)==meta['integrity']['rgb_frame_hash_sequence_sha256'],'collector_RGB_frame_hash_sequence_identity',tid)
            for key,value in (('states',states),('actions',actions),('initial_state',states[0])):signatures[key][array_sha(value)].append(tid)
            signatures['RGB_sequence'][meta['rgb_sequence_sha256']].append(tid)
            signatures['RGB_initial'][rgb_hash[0].tobytes().hex()].append(tid)
            cached_attempt_counts[tid]=dict(saved_attempted_controls=len(z['attempted_actions']),known_substeps=int(np.count_nonzero(z['substep_status']==1)),unknown_substeps=int(np.count_nonzero(z['substep_status']==2)))
            initials[letter].append((tid,entry['split'],states[0].copy()))
            # Independent replay of the frozen state-only controller, not
            # dynamics or action search; after-state is only observe feedback.
            class View:
                def state(self):return self.value.copy()
            view=View();policy=ContactPolicy(view,scene,candidate);replay_ok=True
            for i in range(48):
                view.value=states[i];issued,decision=policy.next_action(i)
                if not np.array_equal(issued,actions[i]) or decision['arc_id']!=int(z['control_arc_id'][i]) or decision['mode']!=int(z['control_mode'][i]):replay_ok=False
                policy.observe(states[i],issued,states[i+1],{'agent_object_contact_substeps':int(np.count_nonzero(flags[i*10:(i+1)*10,0]))})
            check(replay_ok,'deterministic_current_state_action_controller_replay48',tid)
            details.append(dict(trajectory_id=tid,letter=letter,split=entry['split'],checks_failed=sum(x.get('context')==tid for x in issues),
                max_penetration=float(np.max(penetrations)),max_measured_force_norm_float32=float(np.max(np.linalg.norm(forces[:,4:6].astype(np.float64),axis=1))),
                max_prestep_speed_float32=float(np.max(np.linalg.norm(capvel.astype(np.float64),axis=1))),contact_points=k,substeps=480))
    for letter in LETTERS:
        check(counts[(letter,'train')]==128 and counts[(letter,'seen_val')]==24,'perletter128TRAIN24newval',letter)
        data=initials[letter]
        for first,split_a,a in data:
            if split_a!='train':continue
            for second,split_b,b in data:
                if split_b=='seen_val' and near_initial(letter,a,b):near_pairs.append(dict(first=first,second=second,letter=letter))
        per_letter[letter]=dict(train=counts[(letter,'train')],seen_val=counts[(letter,'seen_val')],
            completed_controls=sum(48 for d in details if d['letter']==letter),substeps=sum(d['substeps'] for d in details if d['letter']==letter),
            max_penetration=max((d['max_penetration'] for d in details if d['letter']==letter),default=None))
    check(not near_pairs,'no_actual_cross_split_declared_initial_near_scenes',len(near_pairs))
    duplicates={kind:[dict(sha256=digest,trajectories=tids) for digest,tids in groups.items() if len(tids)>1] for kind,groups in signatures.items()}
    check(not duplicates['states'],'no_exact_duplicate_complete_state_sequences')
    check(not duplicates['initial_state'],'no_exact_duplicate_initial_states')
    # Failed candidates are outside the main manifest but their actual physics
    # and the reserved-not-started charge must still reconcile, exactly once.
    all_attempt_counts=Counter();failed_details=[]
    for row in index['slots']:
        check(row['status']=='completed' and 1<=len(row['attempts'])<=3,'all_slots_success_and_max_two_replacements',row['slot_id'])
        check(sum(a['status']=='completed' for a in row['attempts'])==1,'completed_success_never_recollected',row['slot_id'])
        for attempt in row['attempts']:
            all_attempt_counts['attempts']+=1;all_attempt_counts['charged_controls']+=attempt['charged_controls'];all_attempt_counts['completed_controls']+=attempt['completed_controls']
            if attempt.get('trajectory_file'):
                ap=safe(base,attempt['trajectory_file'])
                tid=attempt['trajectory_id']
                actual_sha=cached_trajectory_hashes[tid] if tid in cached_trajectory_hashes else sha(ap)
                check(actual_sha==attempt['trajectory_sha256'],'all_attempt_including_failure_NPZ_SHA',tid)
                if tid in cached_attempt_counts:
                    for key,value in cached_attempt_counts[tid].items():all_attempt_counts[key]+=value
                else:
                    with np.load(ap,allow_pickle=False) as z:
                        all_attempt_counts['saved_attempted_controls']+=len(z['attempted_actions'])
                        all_attempt_counts['known_substeps']+=int(np.count_nonzero(z['substep_status']==1));all_attempt_counts['unknown_substeps']+=int(np.count_nonzero(z['substep_status']==2))
                        failed_details.append(dict(trajectory_id=tid,charged=attempt['charged_controls'],completed=len(z['actions']),saved_attempted=len(z['attempted_actions']),known_substeps=int(np.count_nonzero(z['substep_status']==1)),status=attempt['status']))
                    if attempt.get('metadata_file'):
                        am=safe(base,attempt['metadata_file']);check(sha(am)==attempt['metadata_sha256'],'failed_attempt_metadata_SHA_preserved',tid)
    check(all_attempt_counts['charged_controls']==ledger['charged_controls'],'all_failed_successful_attempt_charges_preserved')
    check(all_attempt_counts['completed_controls']==ledger['completed_controls'],'all_attempt_complete_controls_reconcile')
    check(all_attempt_counts['known_substeps']==ledger['executed_substeps'],'all_actual_known100Hz_steps_reconcile')
    check(all_attempt_counts['unknown_substeps']==ledger['engine_unknown_advance_substeps'],'unknown_engine_advance_separately_preserved')
    check(all_attempt_counts['charged_controls']-all_attempt_counts['saved_attempted_controls']==ledger.get('reserved_not_started_controls',0),'reserved_not_started_charge_retained_not_refunded')
    check(ledger['supervised_plus_charged_controls']==41976+ledger['charged_controls'] and ledger['all_native_plus_charged_controls']==76536+ledger['charged_controls'],'historical_physics_counts_not_reset')
    result=dict(kind='R3A-cold-numeric-state-force-action-integrity.v1',passed=not issues,checks=dict(checks),issues=issues,
        successful_scenes=len(entries),per_letter=per_letter,trajectory_details=details,duplicates=duplicates,
        duplicate_RGB_or_action_signatures_require_interpretation_not_automatic_scene_recollection=True,
        cross_split_initial_near_pairs=near_pairs,all_attempt_counts=dict(all_attempt_counts),failed_attempts=failed_details,
        reserved_not_started_controls=ledger.get('reserved_not_started_controls',0),
        manifest_sha256=sha(manifest_path),plan_sha256=sha(base/'plan.json'),index_sha256=sha(base/'index.json'),
        ledger_sha256=sha(base/'resource-ledger.json'),resource_ledger_reads=1,
        RGB_sequence_hashes_are_collector_observations_not_recomputed_cold=True,
        float32_audit_tolerances=dict(force_atol=.005,force_rtol=5e-7,speed_atol=.0001,lever_atol=.002),
        post_collision_velocity_is_not_the_protocol_prestep_speed_cap=True,
        new_physics_controls=0,new_environment_constructions=0,RGB_render_calls=0,model_calls=0,
        elapsed_seconds=time.perf_counter()-started)
    atomic_json(out,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default=str(Path(__file__).parent/'manifest.json'));p.add_argument('--out',default=str(Path(__file__).parent/'numeric-integrity-review.json'));args=p.parse_args()
    result=audit(args.manifest,out=args.out)
    print(json.dumps(dict(passed=result['passed'],checks=result['checks'],successful_scenes=result['successful_scenes'],issues=len(result['issues']),seconds=result['elapsed_seconds']),ensure_ascii=False))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
