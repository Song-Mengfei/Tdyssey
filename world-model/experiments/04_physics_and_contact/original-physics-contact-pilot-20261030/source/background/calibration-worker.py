"""Exactly18 frozen calibration-v2 attempts. No formal release, retry or model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import (CHECKPOINT, KIND, LETTERS, PROTOCOL, actual_python_pe, atomic_json,
    child_environment, exclusive_json, in_job, read_json, sha, storage_guard, utc,
    validate_config, wait_launcher_exit)
from worker import merge_ledgers, own_paths


class Calibration:
    def __init__(self,args):
        self.args=args
        self.config,self.paths,self.by_id=validate_config(args.config,args.config_sha,require_release=True,mode='calibration-v2')
        self.runtime=Path(self.config['artifact_root'])/'background/calibration-v2'
        self.own=own_paths(self.config,self.config['run_label'])
        self.process=None
        self.started=utc()
        self.next_heartbeat=0
        self.state='preflight'
        self.failure=None
        self.ledger=None
        self.merge_error=None

    def snapshot(self,force=False):
        if not force and time.monotonic()<self.next_heartbeat:return
        self.next_heartbeat=time.monotonic()+30
        record=dict(kind=KIND,mode='calibration-v2',run_id=self.config['run_id'],config_sha256=self.args.config_sha,
            state=self.state,pid=os.getpid(),started_at_utc=self.started,updated_at_utc=utc(),failure=self.failure,
            child_pid=self.process.pid if self.process else None,child_exit_code=self.process.poll() if self.process else None,
            progress=read_json(self.own['progress']) if self.own['progress'].is_file() else None,
            model_calls=0,optimizer_updates=0,formal_released=False,auto_retry=False)
        atomic_json(self.runtime/'status.json',record);atomic_json(self.runtime/'heartbeat.json',record)

    def stop(self):
        if self.process and self.process.poll() is None and not self.own['stop'].exists():
            exclusive_json(self.own['stop'],dict(run_id=self.config['run_id'],run_label=self.config['run_label'],
                reason=self.failure,requested_at_utc=utc(),contract='stop at next trajectory boundary; preserve all costs; no retries'))

    def observed_failure(self):
        if self.own['ledger'].is_file():
            rows=read_json(self.own['ledger'])['attempts']
            bad=[row for row in rows if row['status'] not in ('reserved','completed')]
            if bad:return 'Substantive calibration physical failure: '+bad[0]['slot_id']
        if self.own['completion'].is_file():
            record=read_json(self.own['completion'])
            if record.get('status')=='failed' or record.get('completed') is False or record.get('graceful_stop'):
                return 'Calibration collector reported failed/stopped completion'
        if self.process and self.process.poll() not in (None,0):
            return 'Calibration CLI exit '+str(self.process.returncode)
        return None

    def start(self):
        validate_config(self.args.config,self.args.config_sha,require_release=True,mode='calibration-v2')
        for slot in self.config['calibration_slot_ids']:
            if (Path(self.config['collection'])/'trajectories'/slot/'claim.json').exists():
                raise RuntimeError('Existing cal2 slot claim; no retry/skip: '+slot)
        if self.own['root'].exists() or self.own['stop'].exists():
            raise RuntimeError('Existing cal2 run label or stop request; no automatic retry')
        postprocess=read_json(self.paths['resource_policy'])['postprocess_reserve_bytes']
        atomic_json(self.runtime/'storage-preflight.json',storage_guard(self.config,extra_bytes=postprocess))
        self.own['root'].mkdir(parents=True,exist_ok=False)
        command=[actual_python_pe(),'-X','utf8','-B',str(self.paths['collector_script']),'collect',
            '--output',self.config['collection'],'--plan',str(self.paths['plan']),'--phase','calibration-v2',
            '--physics-gate',str(self.paths['physics_gate']),'--source-root',self.config['source_root'],
            '--storage-roots-json',str(self.paths['storage_roots']),'--policy-revision',str(self.paths['calibration_policy']),
            '--resource-policy',str(self.paths['resource_policy']),'--letters',LETTERS,
            '--run-label',self.config['run_label'],'--stop-file',str(self.own['stop'])]
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        with (self.own['root']/'stdout.log').open('ab') as out,(self.own['root']/'stderr.log').open('ab') as err:
            self.process=subprocess.Popen(command,cwd=self.config['repo'],env=child_environment(self.config),
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,startupinfo=startup,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
        job=in_job(int(self.process._handle))
        atomic_json(self.own['root']/'child-process.json',dict(run_id=self.config['run_id'],
            run_label=self.config['run_label'],pid=self.process.pid,command=command,is_process_in_job=job,at_utc=utc()))
        if job:raise RuntimeError('Calibration child is inside a Windows Job')
        self.state='collecting-calibration-v2';self.snapshot(True)

    def verify(self):
        if not self.process or self.process.returncode!=0:
            raise RuntimeError('Calibration CLI did not exit successfully')
        completion=read_json(self.own['completion']);identity=read_json(self.own['identity'])
        expected=dict(phase='calibration-v2',run_label=self.config['run_label'],
            plan_sha256=self.config['plan']['sha256'],physics_protocol_sha256=PROTOCOL,
            physics_gate_sha256=self.config['physics_gate']['sha256'],
            resource_policy_sha256=self.config['resource_policy']['sha256'],
            policy_revision_sha256=self.config['calibration_policy']['sha256'])
        for record in (identity,completion):
            if any(record.get(key)!=value for key,value in expected.items()):
                raise ValueError('Calibration shard identity differs from frozen release')
            if record.get('model_calls',record.get('models',0))!=0 or record.get('optimizer_updates',0)!=0:
                raise ValueError('Unexpected calibration model/optimizer activity')
        if completion.get('completed') is not True or completion.get('graceful_stop',False):
            raise ValueError('Calibration completion is partial; not a successful18-slot phase')
        rows=read_json(self.own['ledger'])['attempts']
        if len(rows)!=18 or {row['slot_id'] for row in rows}!=set(self.config['calibration_slot_ids']):
            raise ValueError('Calibration actual denominator differs from original18-slot release')
        if any(row['status']!='completed' or row['actual_controls']!=64 for row in rows):
            raise ValueError('Calibration has incomplete/rejected actual physical attempts')

    def merge(self):
        sources=[self.paths['calibration_ledger']]
        if self.own['ledger'].is_file():sources.append(self.own['ledger'])
        # The helper's formal requirements are deliberately disabled. No formal
        # slot is authorized or invoked anywhere in this worker.
        self.ledger=merge_ledgers(self.config,self.by_id,sources,require_formal=False)
        atomic_json(Path(self.config['collection'])/'ledger.json',self.ledger)
        return sources

    def finish(self,success):
        self.state='waiting_calibration_review' if success else 'failed_waiting_calibration_review'
        self.snapshot(True)
        sources=[self.paths['calibration_ledger']]+([self.own['ledger']] if self.own['ledger'].is_file() else [])
        originals=[];files=[self.args.config,self.runtime/'worker-startup.json',self.runtime/'launcher-receipt.json',*sources]
        for source in sources:
            item=dict(path=str(source),sha256=sha(source))
            try:
                original=read_json(source)
                item['reported_actual_costs_unverified']=dict(actual_controls=original.get('actual_controls'),
                    actual_substeps=original.get('actual_substeps'),actual_physics_seconds=original.get('actual_physics_seconds'),
                    attempts=len(original.get('attempts',[])))
                for row in original.get('attempts',[]):
                    directory=Path(self.config['collection'])/'trajectories'/row['slot_id']
                    files += [directory/'claim.json',directory/'metadata.json',directory/'raw.npz']
            except Exception as error:item['read_error']=repr(error)
            originals.append(item)
        if not success:
            atomic_json(self.runtime/'failure-evidence.json',dict(run_id=self.config['run_id'],failure=self.failure,
                merge_error=self.merge_error,totals_unresolved=self.ledger is None,sources=originals,at_utc=utc()))
            (self.runtime/'FAILED.md').write_text('# Calibration-v2 stopped\n\n'+str(self.failure)+
                '\n\nNo retry or formal collection was started. Actual partial raw/cost evidence is retained.\n',encoding='utf-8')
            files += [self.runtime/'failure-evidence.json',self.runtime/'FAILED.md']
        files += [Path(item['path']) for item in self.config['source_bindings']]
        files += [self.paths[key] for key in ('plan','physics_gate','storage_roots','resource_policy',
                    'calibration_policy','collector_script','notification')]
        if self.own['root'].is_dir():files+=list(self.own['root'].rglob('*'))
        unique={str(Path(path).resolve()).casefold():Path(path).resolve() for path in files if Path(path).is_file()}
        manifest=self.runtime/'final-manifest.json'
        atomic_json(manifest,dict(run_id=self.config['run_id'],status='success' if success else 'failed',at_utc=utc(),
            checkpoint=dict(path=str(self.paths['checkpoint']),sha256=sha(self.paths['checkpoint']),expected_sha256=CHECKPOINT),
            files=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in unique.values()],
            model_calls=0,optimizer_updates=0,formal_released=False))
        current=[row for row in self.ledger['attempts'] if row['phase']=='calibration-v2'] if self.ledger else None
        valid_controls=sum(read_json(Path(self.config['collection'])/'trajectories'/row['slot_id']/'metadata.json')['actual_valid_controls']
                           for row in current) if current is not None else None
        actual_phase_costs=None
        if self.own['ledger'].is_file():
            try:
                phase_ledger=read_json(self.own['ledger'])
                actual_phase_costs={key:phase_ledger.get(key) for key in ('actual_controls','actual_substeps','actual_physics_seconds')}
            except Exception:
                pass  # Original malformed source receipt is retained in manifest/evidence.
        atomic_json(self.runtime/'completion.json',dict(kind=KIND,mode='calibration-v2',run_id=self.config['run_id'],
            config_sha256=self.args.config_sha,status='success' if success else 'failed',finished_at_utc=utc(),
            failure=self.failure,merge_error=self.merge_error,totals_unresolved=self.ledger is None,
            expected_attempts=18,expected_controls=1152,
            actual_calibration_attempts=len(current) if current is not None else None,
            actual_calibration_complete=sum(row['status']=='completed' for row in current) if current is not None else None,
            actual_calibration_controls=sum(row['actual_controls'] for row in current) if current is not None else None,
            actual_calibration_valid_controls=valid_controls,
            calibration_phase_costs=actual_phase_costs,phase_costs_verified_by_merge=self.ledger is not None,
            merged_total_controls=self.ledger['actual_controls'] if self.ledger else None,
            merged_total_substeps=self.ledger['actual_substeps'] if self.ledger else None,
            merged_total_physics_seconds=self.ledger['actual_physics_seconds'] if self.ledger else None,
            cost_ledger_paths=[str(path) for path in sources],manifest=str(manifest),
            FAILED=str(self.runtime/'FAILED.md') if not success else None,REPORT=None,
            checkpoint_sha256=sha(self.paths['checkpoint']),model_calls=0,optimizer_updates=0,
            no_auto_retry=True,formal_released=False,training_authorized=False,
            callback=dict(sent=False,performed=False,required_actor='Codex app / official monitor'),
            next_action='Root/monitor reads saved evidence, may build saved-only report and reviews calibration; no automatic formal release'))

    def run(self):
        exclusive_json(self.runtime/'worker.lock.json',dict(run_id=self.config['run_id'],config_sha256=self.args.config_sha,
            pid=os.getpid(),at_utc=utc(),no_automatic_recovery=True))
        try:
            if not self.args.launcher_pid or self.args.launcher_pid==os.getpid():
                raise RuntimeError('Independent calibration launcher PID required')
            job=in_job()
            receipt=dict(run_id=self.config['run_id'],config_sha256=self.args.config_sha,pid=os.getpid(),
                observed_parent_pid=os.getppid(),launcher_pid=self.args.launcher_pid,is_process_in_job=job,
                interpreter_pe=actual_python_pe(),sys_prefix=sys.prefix,sys_executable=sys.executable,at_utc=utc())
            atomic_json(self.runtime/'worker-startup.json',receipt)
            if job:raise RuntimeError('Calibration worker is inside a Windows Job; no control executed')
            if Path(sys.prefix).resolve()!=Path(self.config['python_executable']).parent.parent.resolve():
                raise RuntimeError('Calibration worker venv mismatch')
            receipt['launcher_exit']=wait_launcher_exit(self.args.launcher_pid)
            receipt['controls_before_parent_exit']=0
            atomic_json(self.runtime/'worker-startup.json',receipt)
            self.start()
            while self.process.poll() is None:
                if (self.runtime/'STOP.json').exists():self.failure=self.failure or 'Explicit calibration boundary-stop request'
                self.failure=self.failure or self.observed_failure()
                if self.failure:self.state='stopping-at-trajectory-boundary';self.stop()
                self.snapshot();time.sleep(1)
            self.failure=self.failure or self.observed_failure()
            if self.failure:raise RuntimeError(self.failure)
            self.verify();self.merge()
            validate_config(self.args.config,self.args.config_sha,require_release=True,mode='calibration-v2')
            atomic_json(self.runtime/'storage-postflight.json',storage_guard(self.config))
            self.finish(True)
            return 0
        except Exception as error:
            self.failure=self.failure or repr(error)
            (self.runtime/'worker-traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            self.state='stopping-at-trajectory-boundary';self.stop()
            while self.process and self.process.poll() is None:self.snapshot();time.sleep(1)
            try:self.merge()
            except Exception as merge_error:self.ledger=None;self.merge_error=repr(merge_error)
            self.finish(False)
            return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha',required=True)
    parser.add_argument('--launcher-pid',type=int);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        config,paths,by_id=validate_config(args.config,args.config_sha,require_release=False,mode='calibration-v2')
        print(json.dumps(dict(passed=True,run_id=config['run_id'],config_sha256=args.config_sha,
            planned_calibration_attempts=18,planned_controls=1152,formal_released=False,
            actual_physics_steps=0,model_calls=0,optimizer_updates=0,processes_started=0,files_written=0),ensure_ascii=False))
        return 0
    if not args.launcher_pid:parser.error('Use calibration-launcher.py; check-only never starts collection')
    return Calibration(args).run()


if __name__=='__main__':
    raise SystemExit(main())
