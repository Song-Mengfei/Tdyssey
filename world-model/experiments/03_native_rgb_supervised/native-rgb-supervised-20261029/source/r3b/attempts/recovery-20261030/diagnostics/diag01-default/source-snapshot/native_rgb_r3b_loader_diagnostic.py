"""One bounded CPU-only diagnostic of the actual detached two-worker loader."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey import native_rgb_r3b_loader_diagnostics as diag

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with(output/'once-claim.json').open('x',encoding='utf-8')as stream:
        json.dump(dict(pid=os.getpid(),run_id=args.run_id,at_utc=diag.utc()),stream)
    os.environ['TDYSSEY_LOADER_DIAGNOSTIC_DIR']=str(output)
    os.environ['TDYSSEY_LOADER_DIAGNOSTIC_PARENT_PID']=str(os.getpid())
    for key,value in(('SDL_VIDEODRIVER','dummy'),('SDL_AUDIODRIVER','dummy'),('PYGAME_HIDE_SUPPORT_PROMPT','1'),('PYTHONDONTWRITEBYTECODE','1')):os.environ.setdefault(key,value)
    stderr=(output/'parent.stderr.log').open('a',encoding='utf-8',buffering=1);sys.stderr=stderr
    proof=dict(kind='R3B-loader-diagnostic-startup.v1',run_id=args.run_id,pid=os.getpid(),parent_pid=os.getppid(),
        actual_process_executable=diag.actual_pe(),executable=sys.executable,prefix=sys.prefix,
        working_directory=str(Path.cwd().resolve()),job=diag.job(),torch_imported='torch'in sys.modules)
    diag.json_write(output/'startup-proof.json',proof)
    result=dict(kind='R3B-two-worker-loader-diagnostic.v1',run_id=args.run_id,phase=args.phase,
        started_at_utc=diag.utc(),passed=False,parent_identity=diag.identity(),workers=2,prefetch_batches=2,
        model_loaded=False,cuda_initialized=False,verified_batches=0,verified_worker_pids=[],new_physics_controls=0,
        launcher_exited_before_progress=False,batches=[],failure=None)
    loader=reference=None;started=time.monotonic()
    try:
        if diag.job()['is_in_job']is not False:raise RuntimeError('Diagnostic must use the actual independent background path')
        deadline=time.monotonic()+45
        while not diag.parent_has_exited(args.launcher_pid):
            if time.monotonic()>deadline:raise RuntimeError('Launcher has not exited; do not claim independence')
            time.sleep(.1)
        result['launcher_exited_before_progress']=True
        diag.event('after_launcher_exit',identity=diag.identity(),launcher_pid=args.launcher_pid)
        if not args.without_site_hook:result['site_hook']=diag.install_site_hook(output)
        diag.install_spawn_observation()
        if args.phase=='bootstrap':
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=diag.bootstrap_initializer)as pool:
                rows=list(pool.map(diag.bootstrap_task,range(4)))
            result['bootstrap_results']=rows
            result['verified_worker_pids']=sorted({r['identity']['pid']for r in rows})
            result['passed']=len(result['verified_worker_pids'])==2
        else:
            import numpy as np
            from tdyssey import native_rgb_r3a_data as data
            from tdyssey.native_rgb_r3b_training import normalize_config,training_inputs
            config=normalize_config(json.loads(args.config.read_text(encoding='utf-8')))
            kwargs,order=training_inputs(config)
            result.update(data_manifest_sha256=sha(config['data_manifest']),loader_source_sha256=sha(data.__file__),
                dataset_contract=dict(physics_source_root=config['physics_source_root'],frame_cache_bytes=config['frame_cache_bytes'],state_cache_bytes=config['state_cache_bytes']))
            # Wrappers add phase logs only; the actual dataset, initializer and
            # worker batch operations remain the production implementations.
            data._initialize_worker=diag.diagnostic_initializer
            data._worker_batch=diag.diagnostic_batch
            loader=data.PrefetchBatchLoader(kwargs,workers=2,prefetch_batches=2)
            reference=data.R3AStateDataset(**kwargs)
            for number,batch in enumerate(loader.iterate(order.batches[:args.batches])):
                values=batch['model_inputs'];indices=batch['indices']
                expected={k:np.stack([reference[int(i)][k]for i in indices])for k in('image','action')}
                matches={k:bool(np.array_equal(values[k],expected[k]))for k in expected}
                if set(values)!=set(('image','action'))or not all(matches.values()):raise AssertionError('Original indexed RGB/action content differs')
                row=dict(number=number,worker_pid=batch['worker_pid'],indices=list(indices),
                    families=[order.rows[int(i)]['family_id']for i in indices],exact_reference_matches=matches,
                    model_input_keys=sorted(values),shapes={k:list(v.shape)for k,v in values.items()},
                    dtypes={k:str(v.dtype)for k,v in values.items()},
                    hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()for k,v in values.items()},
                    worker_statistics=batch['worker_statistics'],worker_identity=batch['worker_identity'])
                result['batches'].append(row);result['verified_batches']+=1
                result['verified_worker_pids']=sorted(set(result['verified_worker_pids']+[batch['worker_pid']]))
                diag.event('parent_batch_verified',number=number,worker_pid=batch['worker_pid'],matches=matches)
                diag.json_write(output/'progress.json',result)
            result['maximum_in_flight']=loader.maximum_in_flight
            result['reference_statistics']=reference.statistics()
            result['passed']=result['verified_batches']==args.batches and len(result['verified_worker_pids'])==2 and loader.maximum_in_flight<=2
        if 'torch'in sys.modules:raise RuntimeError('CPU diagnostic unexpectedly imported Torch')
    except BaseException as exc:
        result['passed']=False;result['failure']=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        traceback.print_exc(file=sys.stderr)
    finally:
        if loader is not None:
            try:loader.close()
            except BaseException as exc:result['loader_close_failure']=repr(exc);result['passed']=False
        if reference is not None:reference.close()
        result.update(exitcode_records=diag.child_exitcodes(),finished_at_utc=diag.utc(),wall_seconds=time.monotonic()-started,
            torch_imported='torch'in sys.modules,completed=True)
        diag.event('diagnostic_finished',passed=result['passed'],exitcode_records=result['exitcode_records'])
        diag.json_write(output/'completion.json',result)
        gate={k:result.get(k)for k in('passed','run_id','data_manifest_sha256','loader_source_sha256','workers','prefetch_batches','model_loaded','cuda_initialized','launcher_exited_before_progress','verified_batches','verified_worker_pids','dataset_contract')}
        gate['kind']='R3B-detached-two-worker-loader-gate.v1';gate['completion_path']=str(output/'completion.json')
        diag.json_write(output/'gate.json',gate)
        for restore in reversed(diag.HOOK_RESTORE):restore()
    return 0 if result['passed']else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--run-id',required=True);parser.add_argument('--launcher-pid',required=True,type=int)
    parser.add_argument('--phase',choices=('bootstrap','data'),default='data')
    parser.add_argument('--batches',type=int,choices=(2,3,4),default=4)
    parser.add_argument('--without-site-hook',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);return run(args)

if __name__=='__main__':raise SystemExit(main())
