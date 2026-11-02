"""Source/IO recovery checks only; no Torch or numerical model work."""
import ast,ctypes,json,os,sys
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parent
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
from continuation_io import RetryBudgetIO,replace_with_retry,install
from continuation_ledger import ResumeLedger
from common import read,sha,write

def main():
    checks=[]
    for name in ('continuation_io.py','continuation_ledger.py','continuation_worker.py',__file__):ast.parse((ROOT/name).read_text(encoding='utf-8') if not Path(name).is_absolute() else Path(name).read_text(encoding='utf-8'))
    checks.append('continuation sources AST parsed')
    q=ROOT/'continuation-checks';q.mkdir(exist_ok=True);io=RetryBudgetIO(q,[],root_cap=10000,shared_cap=100000,atomic=128,growth=32,metadata=128)
    io.write_bytes('source.txt',b'exact unchanged bytes');source=io.checked('source.txt');target=io.checked('target.txt')
    assert source.is_relative_to(q.resolve()) and target.is_relative_to(q.resolve())
    calls=[0]
    def transient(s,t):
        calls[0]+=1
        if calls[0]<3:raise PermissionError('controlled sharing violation')
        os.replace(s,t)
    retry_count=replace_with_retry(source,target,replace=transient,sleep=lambda x:None)
    assert retry_count==2 and calls[0]==3 and target.read_bytes()==b'exact unchanged bytes'
    checks.append('two transient replace failures retry same already-written bytes without rewriting payload/model work')
    clock=[0.]
    def tick():clock[0]+=.1;return clock[0]
    def permanent(*x):raise PermissionError('controlled persistent denial')
    try:replace_with_retry(source,target,replace=permanent,sleep=lambda x:None,clock=tick,timeout=.2)
    except PermissionError:pass
    else:raise AssertionError('Persistent denial did not stop')
    checks.append('persistent replace denial remains bounded')
    original_sha=sha(ROOT/'completion.json');old=read(ROOT/'completion.json')['ledger'];paths=[]
    class MemoryIO:
        def write_json(self,p,v,**kw):assert kw.get('exclusive');paths.append(p)
    ledger=ResumeLedger(MemoryIO(),old);assert not ledger.inflight and ledger.extra['original_reservation_cost_retained_images']==4
    with ledger.phase('eval/SNR'):
        ledger.before_forward(None,([0]*4,));ledger.after_forward(None,([0]*4,),None)
    assert len(paths)==2 and paths[0]!=paths[1] and ledger.counts['eval/SNR']['images_attempted']==420 and ledger.counts['eval/SNR']['images_completed']==416
    assert sha(ROOT/'completion.json')==original_sha
    checks.append('hydrate old cost without erasure; immutable distinct snapshots replace hot-file overwrite')
    attempted=sum(x['images_attempted'] for x in old['counts'].values());completed=sum(x['images_completed'] for x in old['counts'].values())
    remaining=3360+2*2048+2*3744;assert attempted==8276 and completed==8272 and remaining==14944 and attempted+remaining==23220<=23296
    checks.append('all remaining original work including replay fits original total image-forward cap')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    install();write(ROOT/'continuation-static-checks.json',dict(passed=True,checks=checks,model_calls=0,GPU_calls=0,physics_calls=0,simulated_replace_retries=2,expected_cumulative_charged_images=23220,expected_cumulative_completed_images=23216))
    print(json.dumps(dict(passed=True,checks=checks,model_calls=0)))

if __name__=='__main__':main()

