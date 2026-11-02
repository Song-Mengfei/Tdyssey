"""Read stopped R3A evidence for handoff accounting; never mutate collection."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def review(stage,out):
    ledger=read(stage/'resource-ledger.json');index=read(stage/'index.json');manifest=read(stage/'manifest.json')
    if ledger.get('closed') is not True or index.get('collection_closed') is not True:
        raise RuntimeError('Wait until the previous collector is stopped and evidence is closed')
    issues=[];facts=[];totals=Counter();successful={e['trajectory_id']:e for e in manifest['entries']}
    def check(value,label,tid=None):
        if not bool(value):issues.append(dict(check=label,trajectory_id=tid))
    for row in index['slots']:
        for record in row['attempts']:
            tid=record['trajectory_id'];totals['attempts']+=1;totals['charged']+=record['charged_controls']
            totals['record_completed']+=record['completed_controls']
            raw=record.get('trajectory_file','trajectories/'+tid+'.npz');path=(stage/raw).resolve()
            check(path.is_relative_to(stage.resolve()),'trajectory_path_inside_stage',tid)
            if not path.exists():
                facts.append(dict(trajectory_id=tid,status=record['status'],file_exists=False,
                    charged_controls=record['charged_controls'],record_completed_controls=record['completed_controls']))
                check(record['completed_controls']==0,'missing_file_only_before_any_completed_control',tid)
                totals['charged_without_saved_attempt']+=record['charged_controls'];continue
            digest=sha(path)
            if record.get('trajectory_sha256'):check(digest==record['trajectory_sha256'],'index_trajectory_SHA',tid)
            with np.load(path,allow_pickle=False) as z:
                states=z['states'];actions=z['actions'];attempted=z['attempted_actions'];sub=z['substep_states']
                offsets=z['control_substep_offsets'];status=z['substep_status'];available=z['substep_state_available']
                n=len(actions);c=len(attempted);m=len(status);known=int(np.count_nonzero(status==1));unknown=int(np.count_nonzero(status==2))
                check(states.dtype==np.float64 and states.shape==(n+1,15),'complete_state15_float64_shape',tid)
                check(actions.dtype==np.float64 and actions.shape==(n,2) and attempted.dtype==np.float64 and attempted.shape==(c,2),'action_float64_shapes',tid)
                check(n<=c<=n+1 and len(offsets)==c+1 and int(offsets[0])==0 and int(offsets[-1])==m,'attempt_and_substep_offsets',tid)
                check(sub.dtype==np.float64 and sub.shape==(m+1,15) and available.shape==(m+1,),'actual_substep_state15_shape',tid)
                check(np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(attempted).all(),'complete_state_action_finite',tid)
                check(np.array_equal(actions,attempted[:n]),'completed_action_is_attempt_prefix',tid)
                check(np.array_equal(states,sub[np.asarray(offsets[:n+1],dtype=int)]),'complete_control_endpoints_equal_actual_substep_chain',tid)
                check(np.array_equal(offsets[:n+1],np.arange(n+1)*10),'ten_substeps_each_completed_control',tid)
                check(np.array_equal(z['control_time_seconds'],np.arange(n+1,dtype=np.float64)*.1),'saved_10Hz_timestamps',tid)
                check(record['completed_controls']==n,'index_completed_equals_saved_actions',tid)
                gap=record['charged_controls']-c
                check(gap>=0,'charged_never_below_saved_attempts',tid)
                totals.update(saved_completed=n,saved_attempted=c,known_substeps=known,unknown_substeps=unknown,
                    recorded_substeps=m,charged_without_saved_attempt=gap,incomplete_attempted_controls=c-n)
                fact=dict(trajectory_id=tid,status=record['status'],file=str(path),sha256=digest,
                    states_shape=list(states.shape),actions_shape=list(actions.shape),attempted_actions_shape=list(attempted.shape),
                    substep_states_shape=list(sub.shape),known_substeps=known,unknown_substeps=unknown,
                    substep_status_counts={str(int(k)):int(v)for k,v in zip(*np.unique(status,return_counts=True))},
                    charged_controls=record['charged_controls'],completed_controls=n,saved_attempted_controls=c,
                    charged_without_saved_attempt=gap,substep_offsets=list(map(int,offsets)),
                    state_chain_finite_on_available_rows=bool(np.isfinite(sub[available]).all()),
                    available_poststates=int(np.count_nonzero(available)),last_completed_state_sha256=hashlib.sha256(states[-1].tobytes()).hexdigest())
                if tid in successful:
                    entry=successful[tid];totals['successful_entries']+=1
                    check(record['status']=='completed' and n==c==48 and m==known==480 and unknown==0,'successful_complete48_actual480_chain',tid)
                    check(entry['sha256']==digest and entry['file']==raw,'successful_manifest_binding',tid)
                    check(sha(stage/entry['metadata_file'])==entry['metadata_sha256'],'successful_metadata_binding',tid)
                elif n==48:
                    fact['full48_missing_manifest_requires_rollforward_not_recollection']=True
                    issues.append(dict(check='full48_requires_success_manifest_rollforward',trajectory_id=tid))
                facts.append(fact)
    for key,total_key in [('charged_controls','charged'),('completed_controls','saved_completed'),('attempted_scenes','attempts'),
            ('successful_scenes','successful_entries'),('executed_substeps','known_substeps'),('engine_unknown_advance_substeps','unknown_substeps')]:
        check(ledger[key]==totals[total_key],'ledger_'+key+'_equals_saved_attempt_totals')
    check(ledger['engine_step_calls']==totals['known_substeps']+totals['unknown_substeps'],'engine_calls_known_plus_unknown')
    check(len(successful)==ledger['successful_scenes'],'all_successes_bound_in_manifest')
    check(len(successful)==len(manifest['entries']),'no_duplicate_successful_trajectory_ID')
    check(totals['unknown_substeps']==0,'no_unknown_engine_advance')
    check(ledger.get('reserved_not_started_controls',0)<=totals['charged_without_saved_attempt'],'historical_unstarted_reservations_retained')
    result=dict(kind='R3A-independent-stopped-handoff-accounting-review.v1',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=not issues,issues=issues,stage=str(stage),source_snapshots={n:sha(stage/n)for n in ('resource-ledger.json','index.json','manifest.json')},
        ledger_snapshot=ledger,observed_numeric_totals=dict(totals),attempt_facts=facts,
        pending_control_requires_traceback_resolution=ledger.get('pending_control'),
        historical_reserved_not_started_controls=ledger.get('reserved_not_started_controls',0),
        new_unresolved_charged_without_attempt=max(0,totals['charged_without_saved_attempt']-ledger.get('reserved_not_started_controls',0)),
        successful_trajectory_inventory=[dict(trajectory_id=e['trajectory_id'],file=e['file'],sha256=e['sha256'])for e in manifest['entries']],
        successful_trajectories_must_be_skipped_without_environment_or_new_charges=True,
        partial_files_must_remain_immutable=True,refund_controls=0,collection_mutations=0,
        physics_steps=0,models_loaded=0,Torch_imported='torch'in sys.modules)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k in ('passed','issues','observed_numeric_totals','historical_reserved_not_started_controls','new_unresolved_charged_without_attempt','pending_control_requires_traceback_resolution')}))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(review(args.stage.resolve(),args.out.resolve()))
