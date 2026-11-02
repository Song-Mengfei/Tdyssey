"""Archive audited terminal stop, freeze v3 and activate one detached owner."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

stage=Path(__file__).resolve().parent;repo=stage.parents[2];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json
from tdyssey.native_rgb_stage_budget import native_storage_paths,storage_footprint


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--activate',action='store_true');args=parser.parse_args()
    directory=stage/'background';receipt=directory/'handoff-receipt.json';config_path=repo/'configs/native_rgb_r3a.json'
    if not args.activate:
        if receipt.exists():raise FileExistsError('Handoff already prepared; no silent rebase')
        review=read(directory/'handoff-independent-review.json');conclusion=read(directory/'handoff-independent-conclusion.json')
        if not review['passed']:raise ValueError('Independent stopped-state accounting must pass')
        ledger=read(stage/'resource-ledger.json');index=read(stage/'index.json');manifest=read(stage/'manifest.json')
        if not ledger['closed']or ledger['pending_control']is not None or ledger['engine_unknown_advance_substeps']!=0:raise ValueError('Only a closed, fully reconciled collector may hand off')
        if ledger['successful_scenes']!=587 or len(manifest['entries'])!=587:raise ValueError('The587 preserved successes must match the actual reviewed handoff')
        if ledger['charged_controls']!=28282 or ledger['completed_controls']!=28278 or ledger['executed_substeps']!=282801:raise ValueError('Do not infer/refund the reviewed physical costs')
        for name,digest in review['source_snapshots'].items():
            if sha(stage/name)!=digest:raise ValueError('Stopped state changed after independent review: '+name)
        stopped=next(r for r in index['slots']if r['slot_id']=='R3A-train-X-074')
        if len(stopped['attempts'])!=1 or stopped['attempts'][0]['status']!='failed_system':raise ValueError('Only the reviewed X074a0 may use the frozena1')
        archive=directory/'handoff-archive-v2';archive.mkdir(exist_ok=False)
        rows=[]
        for name in ['resource-ledger.json','index.json','manifest.json','halt.json','remaining-completion.json','remaining-once-claim.json','source-identity-v2.json']:
            source=stage/name;shutil.copyfile(source,archive/name);rows.append(dict(file=name,sha256=sha(source),bytes=source.stat().st_size))
        shutil.copyfile(config_path,archive/'execution-config-v2.json')
        value=dict(kind='R3A-human-authorized-detached-continuation-handoff.v1',authorized=True,mode='continue-unfinished',
            authorized_at_utc=utc(),human_request='有没有办法让我停止当前的对话并关闭codex后采集继续的',
            plan_sha256=sha(stage/'plan.json'),previous_owner_pid=28400,
            previous_owner_exited=True,exit_evidence='Original tools.write_stdin session45695 completed after handled durable-write stop.',
            failed_slot_next_attempts={'R3A-train-X-074':1},successful_scenes_preserved=587,remaining_successful_target=325,
            successful_manifest_archive=str(archive/'manifest.json'),successful_manifest_sha256=sha(archive/'manifest.json'),
            independent_review_sha256=sha(directory/'handoff-independent-review.json'),
            independent_conclusion_sha256=sha(directory/'handoff-independent-conclusion.json'),
            stop_request_sha256=sha(directory/'handoff-request.json'),archived_stopped_state=rows,
            costs_preserved=dict(charged_controls=28282,completed_controls=28278,executed_substeps=282801,
                reserved_not_started_controls=1,new_handoff_reserved_not_started_controls=0,unknown_engine_advance_substeps=0,refunded_controls=0),
            handoff_partial='X074a0 completed1control/10substeps; stopped at post-save ledger persistence. Preserve partial and use pre-frozena1, never continue physics from a restored state.',
            no_successful_trajectory_recollection=True,no_source_physics_or_controller_change=True,
            model_work_authorized=False,offline_callback_implemented=False,OS_reboot_resume_implemented=False)
        atomic_json(receipt,value)
        config=read(config_path);previous=read(stage/'source-identity-v2.json');files=previous['files'][:]
        for name in ['scripts/native_rgb_r3a_collect_v3.py','artifacts/native-rgb-supervised-20261029/r3a/background-worker.py','artifacts/native-rgb-supervised-20261029/r3a/launch-background.py']:
            source=repo/name;snapshot=stage/'frozen-source-v3'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,snapshot)
            files.append(dict(source=str(source),snapshot=snapshot.relative_to(stage).as_posix(),sha256=sha(source)))
        config.update(entry_script='scripts/native_rgb_r3a_collect_v3.py',entry_version=3,
            source_identity=str(stage/'source-identity-v3.json'),continue_unfinished_authorized=True,
            handoff_receipt=str(receipt),handoff_receipt_sha256=sha(receipt),
            detached_run_directory=str(directory/'run'),background_survives_Codex_close_verified_by_dummy=True,
            background_launch_method='Existing Explorer desktop Document.Application.ShellExecute, approved outside tool lifecycle job')
        config['input_hashes'][str(stage/'background-shellbroker-smoke-proof.json')]=sha(stage/'background-shellbroker-smoke-proof.json')
        config['storage_roots']=list(dict.fromkeys(map(str,native_storage_paths(repo,stage.parent,config['storage_roots']))))
        config['new_stage_source_files']=sorted(set(str(q)for folder,pattern in [('tdyssey','*r3a*.py'),('scripts','*r3a*.py'),('tests','*r3a*.py'),('configs','*r3a*.json')]for q in (repo/folder).glob(pattern)))
        atomic_json(config_path,config)
        atomic_json(config['source_identity'],dict(kind='R3A-background-continuation-source.v3',files=files,
            previous_identity='source-identity-v2.json',config_sha256=sha(config_path),
            plan_sha256=sha(stage/'plan.json'),handoff_receipt_sha256=sha(receipt),
            v3_contracts_sha256=sha(stage/'collector-v3-contracts.json'),
            no_physics_or_controller_change=True))
        print(json.dumps(dict(prepared=True,preserved_successes=587,handoff_sha256=sha(receipt),config_sha256=sha(config_path))))
        return
    config=read(config_path);value=read(receipt)
    if sha(receipt)!=config['handoff_receipt_sha256']or not read(stage/'collector-v3-contracts.json')['passed']:raise ValueError('Frozen receipt and focusedv3contracts required')
    if(directory/'handoff-activated.json').exists():raise FileExistsError('No repeated activation')
    ledger=read(stage/'resource-ledger.json');index=read(stage/'index.json');manifest=read(stage/'manifest.json')
    if not ledger['closed']or ledger['charged_controls']!=28282 or ledger['pending_control']is not None:raise ValueError('Reviewed stopped ledger differs')
    ledger.update(closed=False,background_handoff_receipt_sha256=sha(receipt))
    index.update(status='background_handoff_ready',collection_closed=False,gate_passed=True,
        previous_halt_is_historical=True,background_handoff_receipt_sha256=sha(receipt))
    manifest['collection_closed']=False
    for name,item in [('resource-ledger.json',ledger),('index.json',index),('manifest.json',manifest)]:atomic_json(stage/name,item)
    atomic_json(directory/'handoff-activated.json',dict(at_utc=utc(),handoff_receipt_sha256=sha(receipt),
        successful_scenes=587,charged_controls=28282,completed_controls=28278,executed_substeps=282801,physics_work_by_activation=0))
    print('Audited detached continuation activated; no physics replay and no refunded costs')


if __name__=='__main__':main()
