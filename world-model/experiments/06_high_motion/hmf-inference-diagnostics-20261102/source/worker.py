"""One service-owned low-priority inference process. No child pool or retry."""
from pathlib import Path
import sys,os,argparse,time,traceback,json
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(REPO),str(REPO/'scripts')]
from resource_limits import constrain_cpu
CPU=constrain_cpu()
from common import read,sha,atomic_json,check_storage,utc
import direct_target_background as detached

def validate(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen runtime identity mismatch')
    c=read(path)
    if c['kind']!='hmf-inference-diagnostics.v1' or Path(c['artifact_root']).resolve()!=ROOT:raise ValueError('Scope/root mismatch')
    if c['limits']!=dict(max_predictions=3672,max_image_denoiser_forwards=11016,A_predictions=3456,B_predictions=216,optimizer_updates=0,backward_calls=0,new_physical_steps=0):raise ValueError('Approved count budget differs')
    if c['training_authorized'] or c['formal_sampler_steps']!=3 or c['precision']!='CUDA_FP32' or c['new_root_cap_bytes']!=500000000:raise ValueError('Approved scientific/resource recipe differs')
    observed={}
    for b in c['bindings']:
        p=Path(b['path']).resolve();s=p.stat();before=(s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if sha(p)!=b['sha256']:raise ValueError('Actually referenced frozen source changed: '+str(p))
        s=p.stat()
        if before!=(s.st_size,s.st_mtime_ns,s.st_ctime_ns):raise ValueError('Source changed during validation')
        observed[str(p)]=dict(sha256=b['sha256'],bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
    c['_validated_source_stats']=observed;c['_runtime_config_sha256']=digest;c['_cpu_limits']=CPU
    return c

def status(c,digest,phase,**extra):
    v=dict(run_id=c['run_id'],config_sha256=digest,phase=phase,pid=os.getpid(),updated_at_utc=utc(),**extra)
    atomic_json(ROOT/'background/status.json',v);atomic_json(ROOT/'background/heartbeat.json',v)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--launcher-pid',type=int)
    a=p.parse_args();started=time.perf_counter();c=validate(a.config,a.config_sha)
    if a.mode=='check':print(dict(passed=True,bindings=len(c['bindings']),model_calls=0,physics_calls=0,CPU=CPU));return 0
    bg=ROOT/'background';bg.mkdir(exist_ok=True)
    if (bg/'completion.json').exists()or(ROOT/'execution-claim.json').exists():raise RuntimeError('No repeated task or automatic resume')
    registration=read(bg/'task-registration.json')
    if registration['run_id']!=c['run_id']or registration['task_name']!=c['windows_task_name']or registration['config_sha256']!=a.config_sha:raise ValueError('Actual Windows task registration identity differs')
    launch=detached.wait_launcher_exit(a.launcher_pid)
    atomic_json(ROOT/'execution-claim.json',dict(run_id=c['run_id'],pid=os.getpid(),single_compute_process=True,automatic_retry=False,at_utc=utc()),exclusive=True)
    proof=dict(run_id=c['run_id'],config_sha256=a.config_sha,pid=os.getpid(),actual_interpreter=detached.actual_python(),actual_is_process_in_job=detached.in_job(),launcher_exit=launch,
        service_lifecycle=dict(mechanism='Windows Task Scheduler directly owns this one inference process',task_name=c['windows_task_name'],python_supervisor_processes=0,child_compute_processes=0),cpu_limits=CPU,started_at_utc=utc())
    atomic_json(bg/'worker-startup.json',proof,exclusive=True);atomic_json(bg/'validated-references.json',dict(run_id=c['run_id'],config_sha256=a.config_sha,files=c['_validated_source_stats'],at_utc=utc()))
    status(c,a.config_sha,'inference')
    results={};failure=None
    try:
        check_storage(ROOT,c)
        from engine import run
        results['inference']=run(ROOT,c)
        status(c,a.config_sha,'saved_only_report')
        from report import generate_report
        results['report']=generate_report(ROOT,c,results['inference'])
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc())
        atomic_json(bg/'failure.json',failure)
        (ROOT/'FAILED.md').write_text('# Inference diagnostics stopped\n\n'+failure['message']+'\n\nActual partial calls/reservations/raw artifacts remain. No automatic retry, training or physics.\n',encoding='utf-8')
    try:resource=check_storage(ROOT,c)
    except Exception as exc:
        resource=dict(passed=False,error=repr(exc));failure=failure or dict(type=type(exc).__name__,message=str(exc),at_utc=utc())
    final=dict(run_id=c['run_id'],config_sha256=a.config_sha,completed=failure is None,failed=failure is not None,failure=failure,results=results,wall_seconds=time.perf_counter()-started,
        finished_at_utc=utc(),CPU=CPU,own_CPU_seconds=dict(user=os.times().user,system=os.times().system),actual_extra_child_compute_processes=0,
        optimizer_updates=0,backward_calls=0,new_physical_steps=0,automatic_retry=False,callback_sent=False,state='waiting_callback')
    atomic_json(bg/'completion.json',final);atomic_json(bg/'resource-final.json',resource);status(c,a.config_sha,'waiting_callback',failed=failure is not None)
    return int(failure is not None)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='run':
        (ROOT/'background').mkdir(exist_ok=True);sys.stdout=(ROOT/'background/worker-stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(ROOT/'background/worker-stderr.log').open('a',encoding='utf-8',buffering=1)
    try:raise SystemExit(main())
    except Exception:
        atomic_json(ROOT/'background/bootstrap-failure.json',dict(error=traceback.format_exc(),at_utc=utc(),CPU=CPU,run_id=read(ROOT/'runtime-config.json').get('run_id'),config_sha256=sha(ROOT/'runtime-config.json')));raise

