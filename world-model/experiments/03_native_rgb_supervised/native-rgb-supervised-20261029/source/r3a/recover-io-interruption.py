"""Audited I/O-only recovery: preserve successful data, failures and all costs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone

import numpy as np

stage=Path(__file__).resolve().parent;repo=stage.parents[2]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
    archive=stage/'io-interruption-v1';receipt=stage/'io-recovery.json'
    if not a.activate:
        archive.mkdir(exist_ok=False)
        names=['resource-ledger.json','index.json','manifest.json','halt.json','first-six-completion.json','first-six-once-claim.json','source-identity.json']
        records=[]
        for name in names:
            source=stage/name;target=archive/name;shutil.copyfile(source,target)
            records.append(dict(file=name,sha256=sha(source),bytes=source.stat().st_size))
        shutil.copyfile(repo/'configs/native_rgb_r3a.json',archive/'execution-config-v1.json')
        ledger=read(stage/'resource-ledger.json');index=read(stage/'index.json');manifest=read(stage/'manifest.json')
        if [ledger[k]for k in ('charged_controls','completed_controls','engine_step_calls','executed_substeps','successful_scenes','failed_scenes')]!=[67,66,660,660,1,1]:
            raise ValueError('Unexpected physical history; do not infer or refund costs')
        if ledger['pending_control']['trajectory_id']!='R3A-train-L-000-a0'or ledger['pending_control']['control_index']!=18:
            raise ValueError('Unexpected unresolved reservation')
        if ledger['engine_unknown_advance_substeps']!=0:raise ValueError('Unknown engine work cannot be replayed')
        with np.load(stage/'trajectories/R3A-train-L-000-a0.npz',allow_pickle=False)as z:
            if z['states'].shape!=(19,15)or z['actions'].shape!=(18,2)or len(z['substep_control_index'])!=180:
                raise ValueError('Partial evidence differs from completed L18/180')
        if len(manifest['entries'])!=1 or manifest['entries'][0]['trajectory_id']!='R3A-train-T-000-a0':raise ValueError('Preserve original successful T exactly')
        halt=read(stage/'halt.json')
        if 'ledger[\'pending_control\']'not in halt['traceback']or 'persist()'not in halt['traceback']or 'PermissionError: [WinError 5]'not in halt['traceback']:
            raise ValueError('The halt must prove the unresolved reservation failed at persistence before engine entry')
        value=dict(kind='R3A-audited-Windows-atomic-I/O-repair-authorization.v1',authorized=True,mode='repair-first-six',
            at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(stage/'plan.json'),
            failed_slot_next_attempts={'R3A-train-L-000':1},old_run_evidence=records,
            costs_preserved=dict(charged_controls=67,completed_controls=66,actual_executed_substeps=660,
                failed_partial_controls=18,failed_partial_substeps=180,reserved_not_started_controls=1,refunded_controls=0),
            pending_resolution='L control18 reserve persist failed before env.step; traceback plus exact18complete/180substeps proves no engine entry. Charge retained as reserved_not_started.',
            repair='Versioned v2 I/O only: existing robust atomic JSON and UUID NPZ temporary with bounded Windows same-file replace retries. No physics, plan or controller change.',
            successful_trajectory_untouched=manifest['entries'][0],partial_attempt_retained='trajectories/R3A-train-L-000-a0.npz',
            no_replay_or_state_continuation=True,no_blind_retry=True,required_v2_contracts_before_activation=True)
        atomic_json(receipt,value);print(json.dumps(dict(archived=len(records),receipt_sha256=sha(receipt),physical_work=0)))
        return
    config=read(repo/'configs/native_rgb_r3a.json')
    if not config.get('repair_first_six_authorized')or config.get('recovery_receipt_sha256')!=sha(receipt):raise ValueError('Explicit frozen v2 receipt required')
    if not Path(config['source_identity']).exists()or not(stage/'collector-v2-contracts.json').exists():raise ValueError('V2 code identity/contracts required')
    if not read(stage/'collector-v2-contracts.json').get('passed'):raise ValueError('V2 contracts must pass')
    if (stage/'io-recovery-activated.json').exists():raise FileExistsError('Recovery activation is one-time')
    ledger=read(stage/'resource-ledger.json');index=read(stage/'index.json');manifest=read(stage/'manifest.json')
    if not ledger['closed']or ledger['charged_controls']!=67:raise ValueError('Only the reviewed stopped state may be activated')
    ledger.update(closed=False,pending_control=None,reserved_not_started_controls=1,io_recovery_receipt_sha256=sha(receipt))
    index.update(status='io_repair_ready',collection_closed=False,gate_passed=False)
    for row in index['slots']:
        if row['slot_id']=='R3A-train-L-000':row['status']='audited_repair_pending'
    manifest['collection_closed']=False
    for name,value in [('resource-ledger.json',ledger),('index.json',index),('manifest.json',manifest)]:atomic_json(stage/name,value)
    atomic_json(stage/'io-recovery-activated.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(receipt),
        charged_controls=67,completed_controls=66,engine_steps=660,successful_trajectory_replayed=False))
    print('Audited I/O recovery activated; no physics work was performed')


if __name__=='__main__':main()
