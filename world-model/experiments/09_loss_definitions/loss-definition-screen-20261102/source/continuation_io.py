"""Windows-safe bounded writes; retries IO only, never model work."""
import os,time,uuid
from bounded_io import BudgetIO

def replace_with_retry(source,target,*,replace=os.replace,sleep=time.sleep,clock=time.monotonic,timeout=2.):
    deadline=clock()+timeout;failures=0
    while True:
        try:replace(source,target);return failures
        except PermissionError:
            failures+=1
            if clock()>=deadline:raise
            sleep(max(0.,min(.025,deadline-clock())))

class RetryBudgetIO(BudgetIO):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.io_retry_count=0
    def write_bytes(self,path,data,*,final_metadata=False,exclusive=False):
        import hashlib
        p=self.checked(path)
        if exclusive and p.exists():raise FileExistsError(p)
        self.guard(len(data),final_metadata=final_metadata);p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.part-'+uuid.uuid4().hex);old=p.stat().st_size if p.exists() else 0
        self.own+=len(data)
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        self.io_retry_count+=replace_with_retry(tmp,p);self.own-=old
        return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def install():
    import common
    value=RetryBudgetIO(common.ROOT,common.read(common.ROOT/'storage-plan.json')['shared_roots']);common._IO=value;return value

