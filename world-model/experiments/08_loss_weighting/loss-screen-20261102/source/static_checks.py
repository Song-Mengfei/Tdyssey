"""Stdlib-only checks for prior engineering failures; zero Torch/GPU calls."""
import ast,ctypes,json,os,re,sys
from ctypes import wintypes
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[name]='1'
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
ROOT=Path(__file__).resolve().parent
from bounded_io import BudgetIO,measured_bytes
from safe_names import panel_name
from ledger import Ledger
from authorization import issue_permit
import native_forward,state_restore,scoring_adapter,objectives

def main():
    checks=[]
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    checks.append('all Python sources AST parsed')
    names=[]
    for s in ('AC-O-01-08:AC-O-01-08:a3:H16','CON','..\\aux:stream?*','A/B','A:B'):
        name,record=panel_name(s,'real_history');assert re.fullmatch('[A-Za-z0-9_.-]+',name) and len(name)<100
        assert record['identity']['sample_id']==s;names.append(name)
    assert len(set(names))==len(names);assert panel_name('x','real_history')[0]!=panel_name('x','autoregressive')[0]
    checks.append('colon/reserved/traversal-like sample IDs map to safe distinct bounded basenames')
    q=ROOT/'stdlib-checks';q.mkdir(exist_ok=True)
    io=BudgetIO(q,[],root_cap=4096,shared_cap=100000,atomic=128,growth=32,metadata=128)
    io.write_bytes('atomic.txt',b'x'*1000);io.write_bytes('atomic.txt',b'y'*1000)
    assert io.own==measured_bytes([q]);checks.append('atomic overwrite accounting includes old+temp peak')
    before={p.name:p.stat().st_size for p in q.iterdir()}
    try:io.write_bytes('over-limit.txt',b'z'*4096)
    except RuntimeError:pass
    else:raise AssertionError('Budget exceeded without rejection')
    assert before=={p.name:p.stat().st_size for p in q.iterdir()};checks.append('over-cap write rejected before temp/file mutation')
    try:io.write_bytes('../outside.txt',b'x')
    except ValueError:pass
    else:raise AssertionError('Out-of-root write accepted')
    checks.append('outputs remain within new root')
    class MemoryIO:
        def write_json(self,*args,**kwargs):pass
    ledger=Ledger(MemoryIO(),{'preflight/test':4},4,category_caps={'preflight':4},batch_cap=1)
    phase=ledger.phase('preflight/test')
    with phase:
        with phase:assert ledger.stack==['preflight/test','preflight/test']
    try:
        with phase:raise ValueError('expected')
    except ValueError:pass
    assert ledger.stack==[];checks.append('same Phase supports nested/repeated/exception exits')
    with phase:
        ledger.before_forward(None,([0,0,0,0],));ledger.after_forward(None,([0,0,0,0],),None)
        try:ledger.before_forward(None,([0],))
        except RuntimeError:pass
        else:raise AssertionError('Forward over cap accepted')
    assert ledger.counts['preflight/test']['images_attempted']==4;checks.append('attempted/completed model accounting rejects excess before call')
    try:issue_permit()
    except RuntimeError:pass
    else:raise AssertionError('Unreleased execution permitted')
    try:state_restore.load_parent(None)
    except RuntimeError:pass
    else:raise AssertionError('Parent load without permit')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules;checks.append('unreleased heavy entry rejects before Torch/NumPy/model import')
    text='\\u4e2d\\u6587 UTF8'
    # Literal source remains ASCII; JSON unescape produces a real UTF8 proof.
    text=json.loads('"'+text+'"');io.write_json('unicode.json',dict(text=text));assert json.loads((q/'unicode.json').read_text(encoding='utf-8'))['text']==text
    checks.append('UTF8 Chinese JSON byte roundtrip')
    result=dict(passed=True,checks=checks,model_calls=0,Torch_imported=False,GPU_calls=0,physics_calls=0,numerical_runtime_not_tested=True)
    (ROOT/'STATIC_CHECKS.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()

