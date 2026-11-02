"""Stdlib-only scoped output; atomic peak, metadata and shared reserves included."""
import hashlib,json,os,shutil,uuid
from pathlib import Path

def measured_bytes(roots):
    tops=[]
    for p in sorted({Path(x).resolve() for x in roots},key=lambda x:len(x.parts)):
        if not any(p==q or p.is_relative_to(q) for q in tops):tops.append(p)
    total=0
    for p in tops:
        if p.is_file():total+=p.stat().st_size
        elif p.is_dir():
            for base,dirs,files in os.walk(p,followlinks=False):
                dirs[:]=[x for x in dirs if not Path(base,x).is_symlink() and not Path(base,x).is_junction()]
                total+=sum(Path(base,f).stat().st_size for f in files)
    return total

class BudgetIO:
    def __init__(self,root,shared_roots,*,root_cap=900_000_000,shared_cap=10_000_000_000,
                 atomic=256*1024**2,growth=64*1024**2,metadata=16*1024**2):
        self.root=Path(root).resolve();self.roots=list(shared_roots)+[self.root]
        self.root_cap=root_cap;self.shared_cap=shared_cap
        self.atomic=atomic;self.growth=growth;self.metadata=metadata;self.external=None;self.own=None
    def guard(self,n=0,*,final_metadata=False,refresh=False):
        if not isinstance(n,int) or n<0:raise ValueError('Nonnegative byte reservation required')
        if self.own is None or refresh:self.own=measured_bytes([self.root])
        own=self.own
        if self.external is None or refresh:self.external=measured_bytes(self.roots)-own
        reserve=0 if final_metadata else self.metadata
        if own+n+reserve>self.root_cap or own+self.external+n+self.atomic+self.growth>self.shared_cap:
            raise RuntimeError('Storage pre-write limit; no write performed')
        if shutil.disk_usage(self.root if self.root.exists() else self.root.parent).free<n+self.atomic:
            raise RuntimeError('Filesystem free reserve unavailable')
        return dict(own=own,external=self.external,shared=own+self.external,prospective=n,
                    root_cap=self.root_cap,shared_cap=self.shared_cap,atomic=self.atomic,growth=self.growth,metadata=reserve)
    def checked(self,path):
        p=Path(path);p=(self.root/p if not p.is_absolute() else p).resolve()
        if not p.is_relative_to(self.root):raise ValueError('Output outside this new root')
        return p
    def write_bytes(self,path,data,*,final_metadata=False,exclusive=False):
        p=self.checked(path)
        if exclusive and p.exists():raise FileExistsError(p)
        self.guard(len(data),final_metadata=final_metadata)
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.part-'+uuid.uuid4().hex)
        old_size=p.stat().st_size if p.exists() else 0
        # Charge the complete reserved temp before mutation. A failed write
        # remains conservatively charged; successful replacement removes old.
        self.own+=len(data)
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
        self.own-=old_size
        return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    def write_json(self,path,value,**kwargs):
        data=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
        return self.write_bytes(path,data,**kwargs)

