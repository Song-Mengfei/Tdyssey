"""Bounded Windows-safe output scoped to this new execution root."""
from pathlib import Path
import json,hashlib,importlib.util,sys,io
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];HMF=ROOT.parent/'high-motion-fit-20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
_IO=None
def output_io():
    global _IO
    if _IO is None:
        from continuation_io import RetryBudgetIO
        _IO=RetryBudgetIO(ROOT,read(ROOT/'storage-plan.json')['shared_roots'],root_cap=650_000_000,metadata=24*1024**2)
    return _IO
def write(p,v,*,emergency=False):return output_io().write_json(p,v,final_metadata=emergency)
def write_bytes(p,v,*,emergency=False):return output_io().write_bytes(p,v,final_metadata=emergency)
def save_raw(path,raw):
    import numpy as np
    if raw.dtype!=np.float32 or raw.shape!=(16,3,96,96):raise ValueError('Original FP32 saved16-frame sample required')
    buffer=io.BytesIO();np.savez(buffer,raw=raw)
    return output_io().write_bytes(path,buffer.getvalue(),exclusive=True)
