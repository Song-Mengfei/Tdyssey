"""Cold-only saved-data audit. Never creates an environment or advances physics."""
import sys,json,gzip,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.native_rgb_r2b_run import (OUT,EXPERIMENT,read,sha,atomic_json,utc,
    verify_sources,check_saved,verify_selected_parameters,storage_check)

verify_sources()
idx=read(OUT/'index.json'); ledger=read(OUT/'resource-ledger.json')
assert idx['collection_closed'] and ledger['closed']
assert idx['status']=='completed_frozen54_with_static_gaps'
assert len(idx['slots'])==54 and not ledger['failures'] and ledger['pending_control'] is None
attempted=[r for r in idx['slots'] if r['attempted']]
blocked=[r for r in idx['slots'] if not r['dispatchable']]
assert len(attempted)==53 and len(blocked)==1 and blocked[0]['slot_id']=='R2B-new-41-X-left-top'
assert blocked[0]['charged_controls']==0 and blocked[0]['status']=='static_blocked'
reviews=[]; maxpen=np.zeros(3); contacthist=np.zeros(11,dtype=int); controlcount=0
selected_shas=set(); frames=0; physical_substeps=0; all_zero_initial=True
for row in attempted:
    assert row['status']=='completed' and row['actual_controls']==row['charged_controls']==row['completed_controls']==24
    assert row['partial_substeps']==0
    for field,key in [('trajectory_file','trajectory_sha256'),('metadata_file','metadata_sha256')]:
        assert sha(OUT/row[field])==row[key]
    m=read(OUT/row['metadata_file'])
    for field,key in [('telemetry_file','telemetry_sha256'),('decisionlogs_file','decisionlogs_sha256')]:
        assert sha(OUT/m[field])==m[key]
    verify_selected_parameters(m['initial_identity'],row['letter'])
    selected_shas.add(m['protocol_sha256'])
    assert m['midtrajectory_resets']==0 and m['actual_initial_parameters_match']
    with np.load(OUT/row['trajectory_file'],allow_pickle=False) as z:
        states=z['states'].copy(); actions=z['actions'].copy(); rgb=z['rgb'].copy()
        assert states.dtype==np.float64 and rgb.dtype==np.uint8
        assert np.isfinite(states).all() and np.isfinite(actions).all()
        assert z['partial_states'].size==0 and z['partial_rgb'].size==0
    assert np.array_equal(states[0],m['actual_initial_state'])
    assert np.array_equal(states[0,5:],np.zeros(10))
    frames+=len(rgb)
    logs=json.load(gzip.open(OUT/m['telemetry_file'],'rt',encoding='utf-8'))
    for i,(log,rec) in enumerate(zip(logs,m['control_records'])):
        assert log['completed'] and rec['completed'] and rec['valid_transition'] and rec['valid']
        assert len(log['substeps'])==rec['substep_count']==10
        assert np.array_equal(actions[i],rec['action_absolute']) and np.array_equal(actions[i],log['action'])
        last=states[i]
        kinds={'agent_object':set(),'object_wall':set(),'agent_wall':set()}
        for j,s in enumerate(log['substeps']):
            assert s['executed'] and not s.get('engine_exception')
            assert np.array_equal(s['pre_state'],last)
            last=np.asarray(s['post_state'],dtype=float)
            assert last.shape==(15,) and np.isfinite(last).all()
            pen=np.asarray(s['penetrations'],dtype=float)
            assert pen.shape==(3,) and np.isfinite(pen).all() and np.all(pen<=2)
            maxpen=np.maximum(maxpen,pen)
            for ev in s.get('contacts',[]):
                if ev.get('kind') in kinds:kinds[ev['kind']].add(j)
            physical_substeps+=1
        assert np.array_equal(last,states[i+1])
        for kind in kinds:assert rec[kind+'_contact_substeps']==len(kinds[kind])
        assert rec['contact']==len(kinds['agent_object'])
        contacthist[rec['contact']]+=1;controlcount+=1
    reviews.append(check_saved(row))
    if len(reviews)%9==0:print(json.dumps({'cold_slots_verified':len(reviews),'physics_steps_added':0}),flush=True)
assert controlcount==ledger['charged_controls']==ledger['completed_controls']==1272
assert physical_substeps==ledger['executed_substeps']==ledger['engine_step_calls']==12720
assert frames==1325 and ledger['engine_call_errors_with_unknown_advance']==0
assert selected_shas=={'58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'}
prior=read(OUT/'prior-resource-identities.json')
model=[r for r in prior['files'] if 'cost' in r['path'] or 'model' in r['path']]
assert model and all(sha(r['path'])==r['sha256'] for r in model)
result=dict(kind='R2B-root-cold-integrity.v1',at_utc=utc(),passed=True,
    planned_slots=54,attempted_completed_slots=53,static_blocked_slots=blocked,
    charged_controls=1272,completed_controls=1272,recorded_physics_substeps=12720,
    raw_frames=frames,policy_replayed_controls=sum(r['sampler_replayed_controls'] for r in reviews),
    new_physics_in_this_audit=0,environment_constructions=0,model_calls=0,
    all_substep_state_chains_exact=True,all_actions_decisions_bound=True,
    all_raw_frame_hashes_exact=True,all_initial_velocities_zero=True,
    max_penetrations_world_px=maxpen.tolist(),agent_object_contact_substeps_histogram=contacthist.tolist(),
    selected_protocol_sha256=list(selected_shas)[0],frozen_sources_and_prior_ledgers_unchanged=True,
    prior_model_ledger_files=model,slot_reviews=reviews,storage=storage_check('root-cold-integrity',1024**2))
atomic_json(OUT/'root-integrity-review.json',result)
state=read(EXPERIMENT/'stage-state.json')
state.update(status='R2B_cold_offline_evidence',r2b_execution_authorized=False,
    r2b_physics_authorization_spent=True,r2b_collection_closed=True,
    r2b_collection_complete=True,parent_review_received=False,
    next_action='Finish cold evidence, actual authorized parent callback, save receipt, waiting_parent and STOP')
state['resources'].update(r2b_charged_controls=1272,r2b_completed_controls=1272,
    r2b_executed_substeps=12720,supervised_controls=41976,all_native_controls=76536)
atomic_json(EXPERIMENT/'stage-state.json',state)
print(json.dumps({k:result[k] for k in ['passed','charged_controls','raw_frames','policy_replayed_controls','max_penetrations_world_px','agent_object_contact_substeps_histogram']}),flush=True)
