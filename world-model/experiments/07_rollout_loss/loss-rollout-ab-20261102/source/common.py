"""Run-local, bounded writes. Never writes into referenced experiment roots."""
from pathlib import Path
import os, json, hashlib, importlib.util, sys, shutil
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
HMF=ROOT.parent/'high-motion-fit-20261101'
CAP=1_200_000_000
SHARED_CAP=10_000_000_000
RESERVE=256*1024**2
FINAL_RESERVE=16*1024**2
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def size(roots):
    top=[]
    for p in sorted({Path(x).resolve() for x in roots},key=lambda x:len(x.parts)):
        if not any(p==q or p.is_relative_to(q) for q in top):top.append(p)
    total=0
    for p in top:
        if p.is_file():total+=p.stat().st_size
        elif p.is_dir():
            for d,_,fs in os.walk(p,followlinks=False):
                for f in fs:total+=(Path(d)/f).stat().st_size
    return total
_external=None
def guard(prospective=0,full=False,emergency=False):
    global _external
    own=size([ROOT]);cfg=read(ROOT/'config.json') if (ROOT/'config.json').exists() else None
    if full or _external is None:
        old=read(ROOT.parent/'hmf-inference-diagnostics-20261102/runtime-config.json')
        roots=old['shared_storage_roots']+[str(ROOT.parent/'hmf-inference-diagnostics-20261102'),str(ROOT.parent/'retrospective-analysis-20261101'),str(ROOT)]
        _external=size(roots)-own
    extra=0 if emergency else FINAL_RESERVE
    result=dict(new_root_bytes=own,external_bytes=_external,shared_bytes_estimate=own+_external,new_root_cap=CAP,shared_cap=SHARED_CAP,atomic_reserve=RESERVE,final_metadata_reserve=extra,prospective_bytes=prospective)
    if own+prospective+extra>CAP or own+_external+prospective+RESERVE+64*1024**2>SHARED_CAP or shutil.disk_usage(ROOT).free<prospective+RESERVE:raise RuntimeError('Storage pre-write guard: '+json.dumps(result))
    return result
def checked(p):
    p=Path(p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Output outside new root')
    p.parent.mkdir(parents=True,exist_ok=True);return p
def write_bytes(p,b,emergency=False):
    p=checked(p);guard(len(b),emergency=emergency);tmp=p.with_suffix(p.suffix+'.tmp')
    if tmp.exists():raise FileExistsError(tmp)
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def write(p,v,emergency=False):write_bytes(p,(json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode(),emergency)
def save_raw(p,raw):
    import numpy as np
    p=checked(p);guard(raw.nbytes+65536)
    if p.exists():raise FileExistsError(p)
    with p.open('xb') as f:np.savez(f,raw=raw)
    return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
