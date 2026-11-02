"""ONE root-executed native append diagnostic. No SDK/model/resource queries.
Base Python -B -S native-journal.py --mode journal --plan native-plan.json.
Plan exactly: mode, harness_sha256, source_sha256 for the fixed SOURCES mapping.
Normal2048 + intentional partial2 = at most2050 publications. No retry/run reuse.
"""
import argparse, ctypes, errno, hashlib, json, os, subprocess, sys, threading, time, traceback, uuid
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).absolute().parents[2]
REVIEW=ROOT/'artifacts/gpc-checkpoint-evaluation-engineering-20261026'
SELF=REVIEW/'native-journal.py';OUT=REVIEW/'native-journal';SLOT=65536
SOURCES=('tdyssey/gpc_evaluation_status.py','tdyssey/gpc_evaluation_entry.py','tdyssey/gpc_adaptation_guard.py',
    'tdyssey/gpc_adaptation.py','tdyssey/wm_compat_guard.py','tdyssey/wm_compat_storage.py',
    'tdyssey/policy_memory.py','tdyssey/wm_compat_identity.py','scripts/guarded_c_child.py',
    'tdyssey/gpc_evaluation_assets.py','tdyssey/gpc_checkpoint_evaluation.py','tdyssey/gpc_supervised_adaptation.py',
    'scripts/evaluate_gpc_checkpoint500.py','configs/gpc_checkpoint500_evaluation.json')
FORBIDDEN={'torch','numpy','scipy','pygame','pymunk','transformers','huggingface_hub'}
def enc(v):return (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan(p):
    p=Path(os.path.abspath(p));raw=p.read_bytes()
    if p!=REVIEW/'native-plan.json' or len(raw)>SLOT:raise ValueError('Only fixed bounded native-plan.json')
    expected=dict(mode='journal',harness_sha256=sha(SELF),source_sha256={n:sha(ROOT/n) for n in SOURCES})
    if json.loads(raw)!=expected:raise ValueError('Frozen mode/harness/source mismatch')
    return raw,hashlib.sha256(raw).hexdigest()
def writeall(f,b):
    v=memoryview(b)
    while v:
        n=f.write(v)
        if type(n) is not int or not 0<n<=len(v):raise OSError('Invalid/no write progress')
        v=v[n:]
def new(p,b):
    if len(b)>SLOT:raise ValueError('Evidence exceeds64KiB')
    with p.open('xb',buffering=0) as f:writeall(f,b);os.fsync(f.fileno())
def slot(f,v):
    b=enc(v)
    if len(b)>SLOT:raise ValueError('Evidence slot overflow')
    f.seek(0);writeall(f,b+b' '*(SLOT-len(b)));os.fsync(f.fileno())
def error(e,op):return dict(op=op,type=type(e).__name__,errno=getattr(e,'errno',None),winerror=getattr(e,'winerror',None),message=str(e)[:1200],traceback=''.join(traceback.format_exception(e))[-2400:],at=time.monotonic())
def left(t,n):return max(0.,min(n,t-time.monotonic()))
def deadline(t):
    if time.monotonic()>=t:raise TimeoutError('Original45s work deadline')
class DenyImports:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:raise ImportError('Forbidden neural/SDK import: '+fullname)
def no_neural():
    if FORBIDDEN.intersection(sys.modules):raise RuntimeError('Forbidden neural/SDK module loaded')
def job():
    from tdyssey import policy_memory
    k,_=policy_memory._libraries();inside=wintypes.BOOL()
    k.GetCurrentProcess.argtypes=[];k.GetCurrentProcess.restype=wintypes.HANDLE
    k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];k.IsProcessInJob.restype=wintypes.BOOL
    if not k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(inside)) or not inside.value:raise RuntimeError('Worker outside owned Job')
    k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p];k.QueryInformationJobObject.restype=wintypes.BOOL;v=policy_memory._JobExtendedLimits()
    if not k.QueryInformationJobObject(None,9,ctypes.byref(v),ctypes.sizeof(v),None) or v.JobMemoryLimit!=3*1024**3 or v.BasicLimitInformation.LimitFlags&0x2200!=0x2200:raise RuntimeError('Job limits differ')
def payload(g,n):
    v=g.status_template(eval_only=True);v.update(generation=n,mirror=n,body=str(n)+':'+'0123456789abcdef'*8);return v

def normal(h,j,g,e):
    d=OUT/'normal';d.mkdir();p=d/j.STATUS_NAME;stop=threading.Event();finished=threading.Event();go=threading.Event()
    ready=[threading.Event(),threading.Event()];errs=[];lock=threading.Lock();seen=[];stats=[{},{}];threads=[]
    def fail(exc,op):
        with lock:
            if not errs:errs.append(error(exc,op))
        stop.set();finished.set();go.set()
    with j.Writer(p,pid=os.getpid(),nonce=h['nonce'],max_records=2048) as writer:
        writer.publish(payload(g,0));first=p.read_bytes();new(d/'first-frame.bin',first)
        def validate(v):
            e.validate_status_record(v);n=v.get('generation')
            if n!=len(seen) or v.get('mirror')!=n or v.get('body')!=payload(g,n)['body']:raise ValueError('Every-frame JSON generation/mirror/body mismatch')
            seen.append(n);size=len(json.dumps(v,separators=(',',':')).encode())
            stats[0]['payload_min']=min(stats[0].get('payload_min',size),size);stats[0]['payload_max']=max(stats[0].get('payload_max',size),size)
        def production_reader():
            try:
                with j.Reader(p,pid=os.getpid(),nonce=h['nonce'],validate=validate) as reader:
                    ready[0].set()
                    if not go.wait(left(h['until'],45)):raise TimeoutError('Reader start deadline')
                    for poll in range(8192):
                        deadline(h['until'])
                        if stop.is_set():break
                        reader.poll()
                        if reader.sequence==2048:break
                        if finished.is_set() and reader.read_offset==p.stat().st_size:break
                        stop.wait(left(h['until'],.005))
                    stats[0].update(polls=poll+1,sequence=reader.sequence,read_offset=reader.read_offset,committed_bytes=reader.committed_bytes,partial_bytes=reader.partial_bytes,file_id=reader.file_id)
                    if reader.sequence!=2048 or reader.partial_bytes or seen!=list(range(2048)):raise ValueError('Incomplete production incremental sequence')
            except BaseException as exc:fail(exc,'normal.production_reader');ready[0].set()
        def ordinary_reader():
            try:
                with p.open('rb') as held:
                    prefix=held.read(7);ready[1].set()
                    if not finished.wait(left(h['until'],45)):raise TimeoutError('Ordinary-reader hold deadline')
                    rest=held.read(j.MAX_BYTES+1);whole=prefix+rest
                    if len(whole)>j.MAX_BYTES or whole[:len(first)]!=first:raise ValueError('Held original record/prefix changed')
                    stats[1].update(reads=2,bytes=len(whole),sha256=hashlib.sha256(whole).hexdigest(),old_prefix_complete=True)
            except BaseException as exc:fail(exc,'normal.ordinary_reader');ready[1].set()
        threads=[threading.Thread(target=production_reader,daemon=True),threading.Thread(target=ordinary_reader,daemon=True)]
        for thread in threads:thread.start()
        try:
            for event in ready:
                if not event.wait(left(h['until'],2)):raise TimeoutError('Both readers must hold real handles before writing')
            go.set()
            for n in range(1,2048):
                deadline(h['until'])
                if stop.is_set():break
                writer.publish(payload(g,n))
        except BaseException as exc:fail(exc,'normal.writer')
        finally:
            finished.set();go.set()
            for thread in threads:thread.join(left(h['until'],2))
        if any(thread.is_alive() for thread in threads):raise RuntimeError('Normal readers remain; no later threads/case allowed')
        raw=p.read_bytes()
        if len(raw)>j.MAX_BYTES:raise ValueError('Journal exceeds16MiB')
        if not errs and (stats[1].get('bytes')!=len(raw) or stats[1].get('sha256')!=hashlib.sha256(raw).hexdigest()):fail(ValueError('Ordinary full-file SHA/bytes differ'),'normal.final_compare')
        result=dict(case='normal',status='failed' if errs else 'passed',error=errs[0] if errs else None,
            readers=stats,writer_reserved=writer.reserved_records,writer_committed=writer.committed_records,
            writer_bytes=writer.committed_bytes,sequence_sha256=hashlib.sha256(enc(seen)).hexdigest(),no_retry=True)
    new(d/'case-result.json',enc(result));return result

def partial(h,j,g,e):
    d=OUT/'partial';d.mkdir();p=d/j.STATUS_NAME;hp=d/'handoff.json';new(hp,enc(dict(pid=os.getpid(),nonce=h['nonce'])))
    writer=j.Writer(p,pid=os.getpid(),nonce=h['nonce'],max_records=2)
    heartbeat=g.Heartbeat(hp,payload(g,0),publisher=lambda ignored,value:writer.publish(value))
    # Freeze exactly TWO synchronous callback publications. No periodic writer.
    heartbeat.stop.set();reader=None;entered=False;first_error=None;original=j._write_all
    try:
        heartbeat.__enter__();entered=True;heartbeat.thread.join(left(h['until'],1))
        if heartbeat.thread.is_alive():raise RuntimeError('Stopped heartbeat thread failed to quiesce')
        first=p.read_bytes();new(d/'first-frame.bin',first)
        reader=j.Reader(p,pid=os.getpid(),nonce=h['nonce'],validate=e.validate_status_record)
        old=reader.poll()
        fault=OSError(errno.EIO,'INTENTIONAL native-file partial: seven bytes then EIO')
        def seven(stream,raw):
            if stream.write(raw[:7])!=7:raise OSError('Native partial injection wrote fewer than seven bytes')
            raise fault
        j._write_all=seven
        try:heartbeat.update(generation=1,mirror=1,body=payload(g,1)['body'])
        except OSError as exc:first_error=exc
        finally:j._write_all=original
        if first_error is not fault:raise AssertionError('Original injected EIO was not preserved')
        retained=p.read_bytes();value=reader.poll()
        if value is not None or reader.latest!=old or reader.partial_bytes!=7 or reader.sequence!=1:raise AssertionError('Partial tail refreshed status')
        if retained[:len(first)]!=first or len(retained)!=len(first)+7:raise AssertionError('Initial frame/partial bytes differ')
        if writer.reserved_records!=2 or writer.committed_records!=1 or writer.first_error is not fault:raise AssertionError('Reserved/durable split differs')
        record=g.heartbeat_failure(hp)
        if record is None or record.get('errno')!=errno.EIO or record.get('step')!='status.append.write':raise AssertionError('First-error emergency slot missing/mismatched')
        refused=False
        try:heartbeat.update(generation=2)
        except RuntimeError as exc:refused=exc.__cause__ is fault
        if not refused or p.read_bytes()!=retained or writer.reserved_records!=2:raise AssertionError('Failed heartbeat retried or changed bytes')
        new(d/'partial-tail.bin',retained[len(first):])
        result=dict(case='partial',status='passed',injected=True,no_retry=True,
            attempted=writer.reserved_records,durable=writer.committed_records,reserved_bytes=writer.reserved_bytes,
            durable_bytes=writer.committed_bytes,physical_bytes=len(retained),partial_bytes=reader.partial_bytes,
            old_at_monotonic=old['at_monotonic'],latest_at_monotonic=reader.latest['at_monotonic'],
            refused_after_error=refused,first_error=record,sha256=hashlib.sha256(retained).hexdigest())
    finally:
        j._write_all=original
        if reader is not None:reader.close()
        if entered:
            heartbeat.stop.set();heartbeat.thread.join(left(h['until'],1))
            if heartbeat.thread.is_alive():raise RuntimeError('Heartbeat thread still active; no unbounded shutdown')
            heartbeat.__exit__(OSError,first_error,None)  # Thread proven exited; its internal join is immediate.
        writer.close()
    new(d/'case-result.json',enc(result));return result

def worker(hp):
    h=json.loads(hp.read_bytes())
    if hp!=OUT/'run-handoff.json' or h['pid']!=os.getpid() or h['parent']!=os.getppid() or os.environ.get('JOURNAL_DIAG_NONCE')!=h['nonce']:raise RuntimeError('Exact worker PID/nonce differs')
    if json.loads((OUT/'ack.json').read_bytes())!=dict(pid=os.getpid(),nonce=h['nonce']):raise RuntimeError('No matching ownership ACK')
    _,ps=plan(h['plan'])
    if ps!=h['plan_sha256']:raise RuntimeError('Frozen plan changed')
    sys.path.insert(0,str(ROOT));sys.meta_path.insert(0,DenyImports());no_neural();job()
    from tdyssey import gpc_evaluation_status as j,gpc_adaptation_guard as g,gpc_evaluation_entry as e
    from tdyssey import wm_compat_guard as wm
    replacements=[]
    def no_replace(*args,**kwargs):
        replacements.append('forbidden rename/replace/delete status path')
        raise AssertionError('No replacement/deletion during append diagnostic')
    # Bootstrap's immutable handshake precedes this gate. No tested status
    # operation may call either legacy publisher or filesystem replacement.
    os.rename=os.replace=os.unlink=no_replace
    g.publish_status=g._publish_json=wm._publish_json=no_replace
    result=dict(cases=[],status='running',planned_cases=['normal','partial'],pid=os.getpid(),nonce=h['nonce'],
        sdk_operations=0,neural_operations=0,resource_queries=0,physics_operations=0,network_operations=0,
        max_publications=2050,max_reader_polls=8192,max_reader_threads=2,max_evidence_bytes=32*1024**2)
    with (OUT/'worker-result.json').open('xb+',buffering=0) as report:
        slot(report,result)
        try:
            for name,fn in (('normal',normal),('partial',partial)):
                deadline(h['until'])
                try:case=fn(h,j,g,e)
                except BaseException as exc:case=dict(case=name,status='failed',error=error(exc,'case.'+name))
                result['cases'].append(case);slot(report,result)
                if threading.active_count()!=1:raise RuntimeError('Case leaked threads; later case prohibited')
            plan(h['plan']);no_neural();result['status']='collected'
        except BaseException as exc:result.update(status='stopped',error=error(exc,'worker'))
        result.update(ended=time.monotonic(),replacement_calls=replacements,all_cases_passed=not replacements and len(result['cases'])==2 and all(c['status']=='passed' for c in result['cases']))
        slot(report,result)
    return 0 if result['status']=='collected' and result['all_cases_passed'] else 2

def parent(pp):
    began=time.monotonic();work=began+45;cleanup=began+55;never=threading.Event();state=dict(child=None,owner=None)
    def watchdog():
        if not never.wait(58):
            if state['owner'] is None and state['child'] is not None:
                try:state['child'].kill()
                except BaseException:pass
            os._exit(124)  # OS closes retained noninherited kill-on-close Job.
    threading.Thread(target=watchdog,daemon=True).start()
    if sys.platform!='win32' or not sys.flags.no_site or not sys.dont_write_bytecode:raise RuntimeError('Use native base Python -B -S')
    base=Path(os.path.abspath(getattr(sys,'_base_executable',sys.executable)))
    if os.path.normcase(str(base))!=os.path.normcase(os.path.abspath(sys.executable)):raise RuntimeError('No venv launcher layer')
    raw,ps=plan(pp);sys.path.insert(0,str(ROOT));sys.meta_path.insert(0,DenyImports());no_neural()
    from tdyssey.wm_compat_guard import WindowsWorker
    from tdyssey.wm_compat_storage import unlinked_path
    out=unlinked_path(OUT);out.mkdir();new(out/'frozen-plan.json',raw);nonce=uuid.uuid4().hex;child=owner=None
    r=dict(mode='journal',status='running',nonce=nonce,parent_pid=os.getpid(),plan_sha256=ps,start=began,
        work_deadline=work,cleanup_deadline=cleanup,total_deadline=began+60,hard_exit=began+58,hard_exit_code=124,
        python_processes=2,external_powershell_expected=1,max_total_processes=3,
        max_publications=2050,max_reader_polls=8192,max_reader_threads=2,max_evidence_bytes=32*1024**2,
        sdk_operations=0,neural_operations=0,resource_queries=0,physics_operations=0,network_operations=0,
        job_active_processes_after_cleanup=None,cleanup_errors=[])
    with (out/'parent-report.json').open('xb+',buffering=0) as report:
        slot(report,r)
        try:
            deadline(work);hs=out/'worker.json';ack=out/'ack.json';hp=out/'run-handoff.json'
            command=[str(base),'-B','-S',str(ROOT/'scripts/guarded_c_child.py'),'--handshake',str(hs),'--ack',str(ack),'--nonce',nonce,'--',str(SELF),'--worker',str(hp)]
            env=os.environ.copy();env['JOURNAL_DIAG_NONCE']=nonce
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['child']=child
            gate=min(work,began+5)
            while not hs.exists():
                if child.poll() is not None:raise RuntimeError('Bootstrap exited before ownership')
                if time.monotonic()>=gate:raise TimeoutError('Original handshake deadline')
                time.sleep(left(gate,.02))
            if json.loads(hs.read_bytes())!=dict(pid=child.pid,parent_pid=os.getpid(),nonce=nonce):raise RuntimeError('Actual PID/parent/nonce differs')
            owner=WindowsWorker(child.pid,child.pid);state['owner']=owner;r.update(worker_pid=child.pid,launcher_pid=child.pid,job_memory_bytes=3*1024**3,kill_on_close=True)
            new(hp,enc(dict(plan=str(Path(os.path.abspath(pp))),plan_sha256=ps,pid=child.pid,parent=os.getpid(),nonce=nonce,until=work)))
            deadline(work);new(ack,enc(dict(pid=child.pid,nonce=nonce)))
            while not owner.poll_exit():deadline(work);time.sleep(left(work,.02))
            r['worker_exit_code']=owner.exit_code()
            if (out/'worker-result.json').exists():r['worker_result']=json.loads((out/'worker-result.json').read_bytes())
            if r['worker_exit_code']!=0:raise RuntimeError('Native worker stopped/failed; no retry')
            plan(pp);no_neural();r['status']='collected'
        except BaseException as exc:r.update(status='stopped',error=error(exc,'parent'))
        finally:
            try:
                if owner is not None:
                    timeout=left(cleanup,5)
                    if timeout<=0:raise TimeoutError('Original cleanup deadline')
                    r['job_active_processes_after_cleanup']=owner.quiesce(timeout=timeout)
                    if not owner.poll_exit():raise RuntimeError('Exact handle not signaled after empty Job')
                    r['worker_exit_code']=owner.exit_code();owner.close();owner=None;state['owner']=None
                elif child is not None and child.poll() is None:child.kill()
                if child is not None:
                    timeout=left(cleanup,1)
                    if timeout<=0:raise TimeoutError('Original launcher cleanup deadline')
                    r['launcher_exit_code']=child.wait(timeout=timeout)
            except BaseException as exc:r['cleanup_errors'].append(error(exc,'cleanup'));r['status']='stopped'
            r.update(end=time.monotonic(),wall_seconds_before_final_report=time.monotonic()-began)
            slot(report,r)
    if owner is not None:os._exit(2)  # Teardown closes uncertain Job; no new timeout.
    # Watchdog stays armed through final stdout/OS teardown.
    print(json.dumps(dict(status=r['status'],report=str(out/'parent-report.json'),wall_seconds_before_final_report=r['wall_seconds_before_final_report'])))
    return 0 if r['status']=='collected' else 2

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('journal',));p.add_argument('--plan',type=Path);p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:return worker(Path(os.path.abspath(a.worker)))
    if a.mode!='journal' or a.plan is None:p.error('Explicit --mode journal --plan native-plan.json required')
    return parent(a.plan)
if __name__=='__main__':raise SystemExit(main())
