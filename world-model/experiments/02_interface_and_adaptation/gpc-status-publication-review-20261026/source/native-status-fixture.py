"""Root-only one-off Windows FILE diagnostic. Base Python -B -S; never SDK/model.
Plan JSON: mode, harness_sha256, source_sha256 (exact SOURCES mapping).
CLI --mode original|fixed --plan review/plan.json (or NEW plan-fixed.json).
"""
import argparse, ctypes, hashlib, json, os, stat, subprocess, sys, threading, time, traceback, uuid
from ctypes import wintypes
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).absolute().parents[2]
REVIEW=ROOT/'artifacts/gpc-status-publication-review-20261026'
SELF=REVIEW/'native-status-fixture.py'
SOURCES=('tdyssey/wm_compat_guard.py','tdyssey/wm_compat_storage.py','tdyssey/gpc_adaptation_guard.py',
    'tdyssey/gpc_adaptation.py','tdyssey/policy_memory.py','tdyssey/wm_compat_identity.py','scripts/guarded_c_child.py')
FORBIDDEN={'torch','numpy','scipy','pygame','pymunk','transformers','huggingface_hub'}
SLOT=65536; PAYLOAD=2048; TOTAL_BYTES=32*1024**2
def enc(v):return (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_plan(p,mode):
    p=Path(os.path.abspath(p));raw=p.read_bytes()
    if p.parent!=REVIEW or p.name not in ('plan.json','plan-fixed.json') or len(raw)>SLOT:raise ValueError('Fixed bounded NEW plan required')
    expected=dict(mode=mode,harness_sha256=sha(SELF),source_sha256={n:sha(ROOT/n) for n in SOURCES})
    if json.loads(raw)!=expected:raise ValueError('Frozen mode/source/harness SHA mismatch')
    return raw,hashlib.sha256(raw).hexdigest()
def writeall(f,b):
    v=memoryview(b)
    while v:
        n=f.write(v)
        if type(n) is not int or not 0<n<=len(v):raise OSError('No/invalid write progress')
        v=v[n:]
def new(p,b):
    if len(b)>SLOT:raise ValueError('Evidence exceeds 64KiB')
    with p.open('xb',buffering=0) as f:writeall(f,b);os.fsync(f.fileno())
def slot(f,v):
    b=enc(v)
    if len(b)>SLOT:raise ValueError('Report exceeds prepaid slot')
    f.seek(0);writeall(f,b+b' '*(SLOT-len(b)));os.fsync(f.fileno())
def err(e,op):
    c=e;seen=set()
    while id(c) not in seen and getattr(c,'errno',None) is None and getattr(c,'winerror',None) is None:
        seen.add(id(c));n=c.__cause__ or c.__context__
        if n is None:break
        c=n
    return dict(op=op,type=type(e).__name__,errno=getattr(c,'errno',None),winerror=getattr(c,'winerror',None),
        message=str(e)[:900],traceback=''.join(traceback.format_exception(e))[-1800:],at=time.monotonic(),
        step=getattr(e,'heartbeat_step',None),temporary_path=getattr(e,'heartbeat_temporary_path',None),details=getattr(e,'details',None))
def left(t,n):return max(0.,min(n,t-time.monotonic()))
def deadline(t):
    if time.monotonic()>=t:raise TimeoutError('Original work deadline')
def no_neural():
    if FORBIDDEN.intersection(sys.modules):raise RuntimeError('Forbidden neural/SDK module loaded')
class DenyImports:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:raise ImportError('Forbidden SDK/neural import: '+fullname)
@contextmanager
def explicit(p,attrs=False):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];k.CreateFileW.restype=wintypes.HANDLE
    k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
    h=k.CreateFileW(str(p),0x80 if attrs else 0x80000000,0 if attrs else 3,None,3,0x00200000,None);f=None
    if h in (None,ctypes.c_void_p(-1).value):raise ctypes.WinError(ctypes.get_last_error())
    try:
        if attrs:yield None
        else:
            import msvcrt
            fd=msvcrt.open_osfhandle(int(h),os.O_RDONLY|os.O_BINARY|os.O_NOINHERIT);h=None;f=os.fdopen(fd,'rb',buffering=0);yield f
    finally:
        primary=sys.exc_info()[1]
        try:
            if f is not None:f.close()
            elif h is not None and not k.CloseHandle(h):raise ctypes.WinError(ctypes.get_last_error())
        except BaseException as e:
            if primary is None:raise
            primary.add_note('Secondary reader close: '+repr(e))

def worker(hp):
    h=json.loads(hp.read_bytes());out=REVIEW/('native-'+h['mode'])
    if hp!=out/'run-handoff.json' or h['pid']!=os.getpid() or h['parent']!=os.getppid() or os.environ.get('STATUS_FILE_NONCE')!=h['nonce']:raise RuntimeError('Worker PID/nonce gate differs')
    if json.loads((out/'ack.json').read_bytes())!=dict(pid=os.getpid(),nonce=h['nonce']):raise RuntimeError('Ownership ACK differs')
    _,ps=check_plan(h['plan'],h['mode'])
    if ps!=h['plan_sha256']:raise RuntimeError('Plan changed after ownership')
    sys.path.insert(0,str(ROOT));sys.meta_path.insert(0,DenyImports());no_neural()
    from tdyssey import wm_compat_guard as wm, gpc_adaptation_guard as gpc, policy_memory
    k,_=policy_memory._libraries();inside=wintypes.BOOL()
    k.GetCurrentProcess.argtypes=[];k.GetCurrentProcess.restype=wintypes.HANDLE
    k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];k.IsProcessInJob.restype=wintypes.BOOL
    if not k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(inside)) or not inside.value:raise RuntimeError('Worker outside owned Job')
    k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p];k.QueryInformationJobObject.restype=wintypes.BOOL
    ji=policy_memory._JobExtendedLimits()
    if not k.QueryInformationJobObject(None,9,ctypes.byref(ji),ctypes.sizeof(ji),None) or ji.JobMemoryLimit!=3*1024**3 or ji.BasicLimitInformation.LimitFlags&0x2200!=0x2200:raise RuntimeError('Job limits differ')
    counts=dict(attempted=0,success=0,charged=0);results=[]
    def charge(n):
        if counts['charged']+n>12*1024**2:raise OSError('Conservative worker allowance exhausted')
        counts['charged']+=n
    def save(p,b):charge(len(b));new(p,b)
    def payload(c,n):
        v=gpc.status_template();v.update(generation=n,mirror=n,body=c+':'+str(n)+':'+'0123456789abcdef'*8);return v
    def valid(v,c):
        n=v.get('generation')
        if type(n) is not int or not 0<=n<=1024 or v.get('mirror')!=n or v.get('body')!=payload(c,n)['body'] or v.get('nonce')!=h['nonce'] or v.get('actual_worker_pid')!=os.getpid():raise ValueError('Mixed generation/body/nonce/PID')
        return n
    def publish(c,n,d):
        deadline(h['until']);counts['attempted']+=1
        if counts['attempted']>2060 or counts['attempted']>5000:raise RuntimeError('Publication cap exceeded')
        counts['last_write_operation']=c+'.publish.'+str(n);charge(PAYLOAD);v=payload(c,n)
        if len(enc(v))+256>PAYLOAD:raise ValueError('Input exceeds 2048-byte bound')
        if c!='G':v=gpc.publish_status(d/'handoff.json',v)
        else:
            v.update(nonce=h['nonce'],actual_worker_pid=os.getpid(),at_monotonic=time.monotonic())
            tmp=d/('worker-status.json.'+uuid.uuid4().hex+'.tmp');new(tmp,enc(v))
            wm._publish_json(tmp,d/'worker-status.json',shared_status=True)
        if len(enc(v))>PAYLOAD:raise ValueError('Published status exceeds 2048 bytes')
        valid(v,c);counts['success']+=1;return v
    def bounded(p):
        s=p.stat()
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size>SLOT:raise ValueError('Evidence not ordinary bounded file')
        with p.open('rb') as f:b=f.read(SLOT+1)
        if len(b)>SLOT:raise ValueError('Evidence grew beyond bound')
        return b
    with (out/'worker-result.json').open('xb+',buffering=0) as report:
        summary=dict(mode=h['mode'],pid=os.getpid(),nonce=h['nonce'],cases=results,status='running',limits=dict(publications=2060,reads_per_reader=4096,reader_threads=2,payload=PAYLOAD,total_bytes=TOTAL_BYTES))
        slot(report,summary)
        try:
            for c in 'ABCDEFG':
                deadline(h['until']);d=out/c;d.mkdir();save(d/'handoff.json',enc(dict(status_path=str(d/'worker-status.json'),actual_worker_pid=os.getpid(),nonce=h['nonce'])))
                r=dict(case=c,start=time.monotonic(),no_retry=True);start=counts.copy();old=None;stop=threading.Event();errors=[];elock=threading.Lock();threads=[]
                def note(e,op):
                    with elock:
                        if not errors:errors.append(err(e,op))
                    stop.set()
                try:
                    publish(c,0,d);old=bounded(d/'worker-status.json');p=d/'worker-status.json'
                    if c in 'ABCDE':
                        ready=[threading.Event() for _ in range(2 if c=='B' else 1)];release=[threading.Event() for _ in ready]
                        if c=='B':r['old_reader_complete']=[None,None]
                        def hold(index=0):
                            opener=wm._open_shared_read(p) if c in 'AB' else p.open('rb') if c=='C' else explicit(p,c=='E')
                            try:
                                with opener as f:
                                    prefix=f.read(len(old)//2) if f is not None else None;ready[index].set()
                                    try:
                                        if c=='B':
                                            if not release[index].wait(left(h['until'],45)):raise TimeoutError('Held reader release deadline')
                                        else:publish(c,1,d)
                                    except BaseException as e:note(e,'writer.publish.1')
                                    finally:
                                        if f is not None:
                                            complete=prefix+f.read()==old
                                            if c=='B':r['old_reader_complete'][index]=complete
                                            else:r['old_reader_complete']=complete
                                            if not complete:raise ValueError('Held reader mixed old/new bytes')
                                        else:r['old_reader_complete']=None;r['note']='Attributes-only handle cannot read JSON'
                            except BaseException as e:note(e,'held_reader'+str(index));ready[index].set()
                        if c=='B':
                            threads=[threading.Thread(target=hold,args=(i,),daemon=True) for i in range(2)]
                            for t in threads:t.start()
                            try:
                                for event in ready:
                                    if not event.wait(left(h['until'],2)):raise TimeoutError('Both held readers must be ready')
                                if not errors:
                                    try:publish(c,1,d)
                                    except BaseException as e:note(e,'writer.publish.1')
                            finally:
                                for event in release:event.set()
                                for t in threads:t.join(left(h['until'],2))
                        else:hold()
                        if not errors:
                            r['new_reader_generation']=valid(wm._read(p),c)
                            if r['new_reader_generation']!=1:raise ValueError('New name did not contain new JSON')
                    else:
                        rs=[dict(reads=0,min=None,max=None) for _ in range(2)];hashes=[hashlib.sha256(),hashlib.sha256()];barrier=threading.Barrier(3);read_times=[[],[]]
                        def reader(i):
                            try:
                                barrier.wait(timeout=left(h['until'],2))
                                for _ in range(4096):
                                    if stop.is_set():break
                                    deadline(h['until']);read_began=time.monotonic()
                                    if c=='F':n=valid(wm._read(p),c)
                                    else:
                                        from tdyssey.wm_compat_storage import unlinked_path
                                        q=unlinked_path(p)
                                        if not q.exists():raise FileNotFoundError(2,'Status absent during metadata query',str(q))
                                        s=q.lstat()
                                        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size>SLOT:raise ValueError('Metadata target contract changed')
                                        n=-1
                                    read_times[i].append((read_began,time.monotonic()))
                                    v=rs[i];v['reads']+=1;v['min']=n if v['min'] is None else min(v['min'],n);v['max']=n if v['max'] is None else max(v['max'],n);hashes[i].update(str(n).encode()+b'\n')
                            except BaseException as e:note(e,'reader'+str(i))
                        threads=[threading.Thread(target=reader,args=(i,),daemon=True) for i in range(2)]
                        for t in threads:t.start()
                        seq=hashlib.sha256();writer_began=None
                        try:
                            barrier.wait(timeout=left(h['until'],2))
                            writer_began=time.monotonic()
                            for n in range(1,1025):
                                if stop.is_set():break
                                seq.update(enc(publish(c,n,d)))
                        finally:
                            writer_ended=time.monotonic();r['writer_started_monotonic']=writer_began;r['writer_ended_monotonic']=writer_ended
                            stop.set()
                            for t in threads:t.join(left(h['until'],2))
                            for i,v in enumerate(rs):
                                spans=read_times[i];v.update(first_read_monotonic=spans[0][0] if spans else None,last_read_monotonic=spans[-1][1] if spans else None,
                                    successful_reads_overlapping_writer=sum(writer_began is not None and a<writer_ended and b>writer_began for a,b in spans))
                            r['readers']=[dict(v,sequence_sha256=z.hexdigest()) for v,z in zip(rs,hashes)];r['writer_sequence_sha256']=seq.hexdigest()
                        if not errors and any(v['reads']<1 or v['successful_reads_overlapping_writer']<1 for v in rs):raise ValueError('Concurrency coverage insufficient: each reader needs a successful read overlapping publication')
                except BaseException as e:note(e,'case.'+c)
                if any(t.is_alive() for t in threads):raise RuntimeError('Readers not quiescent; no later case can start')
                r.update(attempted=counts['attempted']-start['attempted'],success=counts['success']-start['success'],last_write_operation=counts.get('last_write_operation'),end=time.monotonic(),error=errors[0] if errors else None)
                try:
                    if old is not None:save(d/'old.bin',old)
                    copies=[]
                    for i,p in enumerate([d/'worker-status.json',*sorted(d.glob('worker-status.json.*.tmp'))]):
                        if not p.exists():copies.append(dict(source=p.name,missing=True));continue
                        b=bounded(p);name='final.bin' if i==0 else 'failed-temp-'+str(i)+'.bin';save(d/name,b);copies.append(dict(source=p.name,copy=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
                    r['copies']=copies
                except BaseException as e:r['error']=r['error'] or err(e,'evidence.'+c)
                r['status']='observed_failure' if r['error'] else 'passed';results.append(r);save(d/'case-result.json',enc(r));charge(SLOT);slot(report,summary)
            check_plan(h['plan'],h['mode']);no_neural();summary['status']='collected'
        except BaseException as e:summary.update(status='stopped',error=err(e,'worker'))
        summary.update(counts=counts,ended=time.monotonic(),sdk_operations=0,neural_operations=0,resource_queries=0,physics_operations=0,network_operations=0)
        slot(report,summary)
    return 0 if summary['status']=='collected' else 2

def parent(mode,plan):
    began=time.monotonic();work=began+45;cleanup=began+55;done=threading.Event();state=dict(child=None,owner=None)
    def watchdog():
        if not done.wait(58):
            # OS teardown closes the noninherited KILL_ON_JOB_CLOSE handle.
            if state['owner'] is None and state['child'] is not None:
                try:state['child'].kill()  # Retained direct child, still at ACK if unowned.
                except BaseException:pass
            os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    if sys.platform!='win32' or not sys.flags.no_site or not sys.dont_write_bytecode:raise RuntimeError('Use Windows base Python -B -S')
    base=Path(os.path.abspath(getattr(sys,'_base_executable',sys.executable)))
    if os.path.normcase(str(base))!=os.path.normcase(os.path.abspath(sys.executable)):raise RuntimeError('No venv launcher layer allowed')
    raw,ps=check_plan(plan,mode);sys.path.insert(0,str(ROOT));sys.meta_path.insert(0,DenyImports());no_neural()
    from tdyssey.wm_compat_guard import WindowsWorker
    from tdyssey.wm_compat_storage import unlinked_path
    out=unlinked_path(REVIEW/('native-'+mode));out.mkdir();new(out/'frozen-plan.json',raw)
    nonce=uuid.uuid4().hex;child=owner=None
    r=dict(mode=mode,status='running',nonce=nonce,parent_pid=os.getpid(),plan_sha256=ps,start=began,
        work_deadline=work,cleanup_deadline=cleanup,total_deadline=began+60,hard_exit=began+58,hard_exit_code=124,
        sdk_operations=0,neural_operations=0,resource_queries=0,physics_operations=0,network_operations=0,
        max_processes=3,actual_processes=2,job_active_processes_after_cleanup=None,cleanup_errors=[],
        two_run_byte_ceiling=TOTAL_BYTES,worker_lifetime_byte_allowance=12*1024**2,parent_fixed_allowance=1024**2,
        total_publication_ceiling=2060,payload_ceiling=PAYLOAD,slot_ceiling=SLOT)
    with (out/'parent-report.json').open('xb+',buffering=0) as report:
        slot(report,r)
        try:
            deadline(work);handshake=out/'worker.json';ack=out/'ack.json';hp=out/'run-handoff.json'
            command=[str(base),'-B','-S',str(ROOT/'scripts/guarded_c_child.py'),'--handshake',str(handshake),'--ack',str(ack),'--nonce',nonce,'--',str(SELF),'--worker',str(hp)]
            env=os.environ.copy();env['STATUS_FILE_NONCE']=nonce
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['child']=child
            gate=min(work,began+5)
            while not handshake.exists():
                if child.poll() is not None:raise RuntimeError('Bootstrap exited before ownership')
                if time.monotonic()>=gate:raise TimeoutError('Bootstrap handshake deadline')
                time.sleep(left(gate,.02))
            if json.loads(handshake.read_bytes())!=dict(pid=child.pid,parent_pid=os.getpid(),nonce=nonce):raise RuntimeError('Exact base-worker PID/parent/nonce differs')
            owner=WindowsWorker(child.pid,child.pid);state['owner']=owner;r.update(worker_pid=child.pid,launcher_pid=child.pid,job_memory_bytes=3*1024**3,job_kill_on_close=True)
            new(hp,enc(dict(mode=mode,plan=str(Path(os.path.abspath(plan))),plan_sha256=ps,pid=child.pid,parent=os.getpid(),nonce=nonce,until=work)))
            deadline(work);new(ack,enc(dict(pid=child.pid,nonce=nonce)))
            while not owner.poll_exit():deadline(work);time.sleep(left(work,.02))
            r['worker_exit_code']=owner.exit_code()
            if (out/'worker-result.json').exists():r['worker_result']=json.loads((out/'worker-result.json').read_bytes())
            if r['worker_exit_code']!=0:raise RuntimeError('Worker stopped; no retry')
            check_plan(plan,mode);no_neural();r['status']='collected'
        except BaseException as e:r.update(status='stopped',error=err(e,'parent'))
        finally:
            try:
                if owner is not None:
                    timeout=left(cleanup,5)
                    if timeout<=0:raise TimeoutError('Original cleanup allowance exhausted')
                    r['job_active_processes_after_cleanup']=owner.quiesce(timeout=timeout)
                    if not owner.poll_exit():raise RuntimeError('Exact process not signaled after empty Job')
                    r['worker_exit_code']=owner.exit_code();owner.close();owner=None;state['owner']=None
                elif child is not None and child.poll() is None:child.kill()
                if child is not None:
                    timeout=left(cleanup,1)
                    if timeout<=0:raise TimeoutError('Original launcher cleanup allowance exhausted')
                    r['launcher_exit_code']=child.wait(timeout=timeout)
            except BaseException as e:r['cleanup_errors'].append(err(e,'cleanup'));r['status']='stopped'
            r.update(end=time.monotonic(),wall_seconds=time.monotonic()-began)
            if time.monotonic()>=began+60:r['status']='stopped';r.setdefault('error',dict(op='final',message='Original total deadline exceeded'))
            slot(report,r)
    if owner is not None:os._exit(2)  # Uncertain cleanup: Job closes on process exit, no fresh timeout.
    # Keep the watchdog armed through final stdout/teardown; no fresh IO deadline.
    print(json.dumps(dict(status=r['status'],report=str(out/'parent-report.json'),wall_seconds_before_final_report=r['wall_seconds'])))
    return 0 if r['status']=='collected' else 2

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('original','fixed'));p.add_argument('--plan',type=Path);p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:return worker(Path(os.path.abspath(a.worker)))
    if a.mode is None or a.plan is None:p.error('Explicit frozen --mode and --plan required')
    return parent(a.mode,a.plan)
if __name__=='__main__':raise SystemExit(main())
