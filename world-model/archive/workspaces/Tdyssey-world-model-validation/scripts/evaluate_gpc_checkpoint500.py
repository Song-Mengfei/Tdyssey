"""Read-only plan by default; future execution needs exact reviewed authority."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey import gpc_evaluation_entry as entry


def worker(path):
    from tdyssey import gpc_adaptation_guard as guard
    from tdyssey.gpc_evaluation_status import Writer
    from tdyssey.gpc_checkpoint_evaluation import run_evaluation
    from tdyssey.gpc_adaptation import OutputWriter,NativeSampler
    handoff,prepared=entry.verify_worker(path)
    cfg=prepared['config'];runtime=dict(torch=None,model_instances=0)
    trace=guard.StartupTrace(path)
    def refresh():
        values=dict(model_instances=runtime['model_instances'])
        if runtime['torch'] is not None:values.update(guard.cuda_allocator_sample(runtime['torch']))
        return values
    with trace, Writer(Path(handoff['status_path']),pid=handoff['actual_worker_pid'],nonce=handoff['nonce']) as journal:
        with guard.Heartbeat(path,guard.status_template(eval_only=True),refresh=refresh,
                publisher=lambda _path,value:journal.publish(value)) as heartbeat:
            try:
                torch,_=guard.initialize_sdk(heartbeat,trace,cpu_threads=cfg['limits']['cpu_threads'])
                runtime['torch']=torch
                from tdyssey.gpc_supervised_adaptation import make_real_denoiser
                def memory_check():
                    heartbeat.check();torch.cuda.synchronize(0)
                    if (torch.cuda.max_memory_allocated(0)>cfg['limits']['cuda_allocated_bytes']
                            or torch.cuda.max_memory_reserved(0)>cfg['limits']['cuda_reserved_bytes']):
                        raise MemoryError('Fixed eval-only CUDA ceiling exceeded')
                def progress(phase,step,state):
                    heartbeat.update(phase=phase,optimizer_updates=state['spent_reserved_updates'],
                        model_loads=state['spent_reserved_model_loads'],forecasts=state['spent_reserved_forecasts'],
                        euler_steps=state['spent_reserved_euler_steps'],eval_batches=state['spent_reserved_eval_batches'],
                        eval_examples=state['spent_eval_examples'])
                def close_model():
                    gc.collect();torch.cuda.empty_cache();runtime['model_instances']=0
                    heartbeat.update()
                def factory(role,*,state_dict):
                    runtime['model_instances']=1;heartbeat.update(phase='model_loading')
                    return make_real_denoiser(handoff['gpc_root'],handoff['checkpoint'],'cuda:0',
                        dict(cfg,execute=True,evaluation_only=True,validated_checkpoint_identity=cfg['sources'][role]),
                        state_dict=state_dict)
                result=run_evaluation(prepared,handoff['binding'],factory=factory,close_model=close_model,
                    make_sampler=lambda model:NativeSampler(model,memory_check),
                    writer=OutputWriter(Path(handoff['output_dir']),cfg['limits']),progress=progress,memory_check=memory_check)
            except BaseException as error:
                heartbeat.record_error(error,step='evaluation.worker')
                raise
    print(json.dumps(dict(status=result['status'],forecast_records=result['forecast_records'])))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',type=Path,default=entry.CONFIG)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--authorization-id',default='')
    parser.add_argument('--reviewed-commit',default='')
    parser.add_argument('--authorized-config-sha256',default='')
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--gpc-root',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.worker:return worker(args.worker)
        prepared=entry.plan(args.config)
        if not args.execute:
            print(json.dumps({k:prepared[k] for k in ('kind','mode','ready','gaps','config_sha256','assets_sha256',
                'real_operations','native_resource_queries','execution_authorized')},indent=2));return 0
        binding=entry.verify_execution(prepared,authorization_id=args.authorization_id,
            reviewed_commit=args.reviewed_commit,authorized_config_sha256=args.authorized_config_sha256)
        if args.gpc_root is None or args.checkpoint is None:raise PermissionError('Explicit pinned source and initial checkpoint required')
        from tdyssey.gpc_adaptation_guard import supervise
        result=supervise(prepared,binding,python_executable=args.python,gpc_root=args.gpc_root,
            checkpoint=args.checkpoint,script_path=Path(__file__),config_path=args.config,eval_only=True)
        print(json.dumps(result,indent=2));return 0 if result['status']=='complete' else 2
    except Exception as error:
        print(json.dumps(dict(status='refused_or_stopped',reason=type(error).__name__+': '+str(error))));return 2


if __name__=='__main__':raise SystemExit(main())
