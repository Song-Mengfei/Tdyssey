"""Reusable stack scopes and durable attempted/completed forward accounting."""
from collections import defaultdict

class Phase:
    def __init__(self,ledger,name):self.ledger=ledger;self.name=name
    def __enter__(self):self.ledger.stack.append(self.name);return self
    def __exit__(self,*exc):
        if not self.ledger.stack or self.ledger.stack[-1]!=self.name:raise RuntimeError('Phase stack mismatch')
        self.ledger.stack.pop();return False

class Ledger:
    def __init__(self,io,phase_caps,total_cap,*,category_caps=None,batch_cap=6016):
        self.io=io;self.caps=dict(phase_caps);self.total_cap=total_cap;self.stack=[]
        self.counts=defaultdict(lambda:dict(images_attempted=0,images_completed=0,batches_attempted=0,batches_completed=0))
        self.attempts=defaultdict(int);self.updates=defaultdict(int);self.exposures=defaultdict(int)
        self.backward=defaultdict(int);self.inflight=[]
        self.category_caps=dict(category_caps or {});self.batch_cap=batch_cap
        self.extra={};self.active=None
    def phase(self,name):
        if name not in self.caps:raise ValueError('Unfrozen phase')
        return Phase(self,name)
    def snapshot(self):
        return dict(counts=dict(self.counts),attempts=dict(self.attempts),updates=dict(self.updates),
                    exposures=dict(self.exposures),backward=dict(self.backward),stack=list(self.stack),inflight=list(self.inflight),extra=self.extra,active=self.active)
    def persist(self):self.io.write_json('ledger.json',self.snapshot())
    def before_forward(self,module,args):
        if not self.stack:raise RuntimeError('Unaccounted model forward')
        n=len(args[0]);scope=self.stack[-1];q=self.counts[scope]
        category=scope.split('/')[0]
        category_used=sum(v['images_attempted'] for k,v in self.counts.items() if k.split('/')[0]==category)
        if category in self.category_caps and category_used+n>self.category_caps[category]:raise RuntimeError('Category image budget exhausted')
        if sum(v['batches_attempted'] for v in self.counts.values())>=self.batch_cap:raise RuntimeError('Batch-forward budget exhausted')
        if n<=0 or q['images_attempted']+n>self.caps[scope] or sum(x['images_attempted'] for x in self.counts.values())+n>self.total_cap:
            raise RuntimeError('Forward budget exhausted before model call')
        q['images_attempted']+=n;q['batches_attempted']+=1;self.inflight.append((scope,n));self.persist()
    def after_forward(self,module,args,output):
        if not self.inflight:raise RuntimeError('Completion without reserved call')
        scope,n=self.inflight.pop();q=self.counts[scope]
        if n!=len(args[0]):raise RuntimeError('Forward batch changed')
        q['images_completed']+=n;q['batches_completed']+=1;self.persist()

    def grad_begin(self,key,preflight=False):
        prefix='preflight' if preflight else 'train'
        used=sum(v for k,v in self.backward.items() if k.startswith(prefix+'/') and k.endswith('/attempted'))
        if used>=(32 if preflight else 2048):raise RuntimeError('Gradient/backward call cap exhausted')
        self.backward[prefix+'/'+key+'/attempted']+=1;self.persist()

    def grad_end(self,key,preflight=False):
        self.backward[('preflight' if preflight else 'train')+'/'+key+'/completed']+=1;self.persist()

