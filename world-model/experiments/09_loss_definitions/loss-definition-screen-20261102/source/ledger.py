"""Immutable cost snapshots. IO retries never retry a numerical operation."""
from collections import defaultdict
class Phase:
    def __init__(self,l,name):self.l=l;self.name=name
    def __enter__(self):self.l.stack.append(self.name);return self
    def __exit__(self,*exc):
        if not self.l.stack or self.l.stack[-1]!=self.name:raise RuntimeError('Phase stack mismatch')
        self.l.stack.pop();return False
class Ledger:
    def __init__(self,io,arms):
        self.io=io;self.caps={**{'preflight/'+a:128 for a in ['CALIBRATION','REFERENCE','PARITY']},**{'train/'+a:2048 for a in arms},**{'eval/'+a:3744 for a in arms}}
        self.counts=defaultdict(lambda:dict(images_attempted=0,images_completed=0,batches_attempted=0,batches_completed=0));self.attempts=defaultdict(int);self.updates=defaultdict(int);self.exposures=defaultdict(int);self.backward=defaultdict(int);self.stack=[];self.inflight=[];self.active=None;self.extra={};self.sequence=0
    def snapshot(self):return dict(counts=dict(self.counts),attempts=dict(self.attempts),updates=dict(self.updates),exposures=dict(self.exposures),backward=dict(self.backward),stack=list(self.stack),inflight=list(self.inflight),active=self.active,extra=self.extra,sequence=self.sequence)
    def persist(self):
        self.sequence+=1;self.io.write_json(f'events/{self.sequence:06d}.json',self.snapshot(),exclusive=True)
    def phase(self,name):
        if name not in self.caps:raise ValueError('Unknown phase')
        return Phase(self,name)
    def before_forward(self,module,args):
        if not self.stack:raise RuntimeError('Unaccounted forward')
        scope=self.stack[-1];n=len(args[0]);q=self.counts[scope];category=scope.split('/')[0]
        totals=sum(x['images_attempted'] for x in self.counts.values());batches=sum(x['batches_attempted'] for x in self.counts.values());cat=sum(x['images_attempted'] for k,x in self.counts.items() if k.split('/')[0]==category)
        if n<=0 or totals+n>17504 or batches>=4544 or cat+n>{'preflight':128,'train':6144,'eval':11232}[category] or q['images_attempted']+n>self.caps[scope]:raise RuntimeError('Forward cap before body')
        q['images_attempted']+=n;q['batches_attempted']+=1;self.inflight.append((scope,n));self.persist()
    def after_forward(self,module,args,out):
        if not self.inflight:raise RuntimeError('Unreserved forward completion')
        scope,n=self.inflight.pop()
        if n!=len(args[0]):raise RuntimeError('Batch changed')
        self.counts[scope]['images_completed']+=n;self.counts[scope]['batches_completed']+=1;self.persist()
    def grad_begin(self,key,preflight=False):
        prefix='preflight' if preflight else 'train';used=sum(v for k,v in self.backward.items() if k.startswith(prefix+'/') and k.endswith('/attempted'))
        if used>=(32 if preflight else 1536):raise RuntimeError('Backward cap')
        self.backward[prefix+'/'+key+'/attempted']+=1;self.persist()
    def grad_end(self,key,preflight=False):self.backward[('preflight' if preflight else 'train')+'/'+key+'/completed']+=1;self.persist()
