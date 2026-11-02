"""Create an explicitly authorized repair revision; preserve original failure."""
from pathlib import Path
import json,hashlib,sys,difflib
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'loss-rollout-ab-20261102'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    if (ROOT/'config.json').exists():raise FileExistsError('Revision already frozen')
    old=read(OLD/'config.json')
    for b in old['bindings']:
        if sha(b['path'])!=b['sha256']:raise RuntimeError('Previously frozen source changed: '+b['path'])
    for name in ('worker.py','common.py','rollout_loss.py','report.py','PLAN.txt','schedule.json','evaluation-spec.json'):
        (ROOT/name).write_bytes((OLD/name).read_bytes())
    p=ROOT/'worker.py';s=p.read_text(encoding='utf8');before=s
    a='''    @contextmanager
    def phase(self,key):
        old=self.scope;self.scope=key
        try:yield
        finally:self.scope=old
'''
    b='''    def phase(self,key):
        return ReusablePhase(self,key)
'''
    cls='''class ReusablePhase:
    def __init__(self,ledger,key):self.ledger=ledger;self.key=key;self.stack=[]
    def __enter__(self):
        self.stack.append(self.ledger.scope);self.ledger.scope=self.key
        return self
    def __exit__(self,typ,value,tb):
        self.ledger.scope=self.stack.pop()
        return False

'''
    assert s.count(a)==1;s=s.replace(a,b).replace('class Ledger:',cls+'class Ledger:',1)
    a="        if not finite or grad_to_first<=0:raise RuntimeError('No finite H4->predictedH1 gradient')"
    assert s.count(a)==1;s=s.replace(a,"        write(ROOT/'preflight-cross-step-gradient.json',dict(finite=finite,norm=grad_to_first,passed=finite and grad_to_first>0))\n"+a)
    p.write_text(s,encoding='utf8',newline='\n')
    (ROOT/'applied-repair.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='original/worker.py',tofile='r1/worker.py')),encoding='utf8')
    p=ROOT/'common.py';s=p.read_text(encoding='utf8');a="str(ROOT.parent/'retrospective-analysis-20261101'),str(ROOT)]";assert s.count(a)==1
    s=s.replace(a,"str(ROOT.parent/'retrospective-analysis-20261101'),str(ROOT.parent/'loss-rollout-ab-20261102'),str(ROOT)]");p.write_text(s,encoding='utf8',newline='\n')
    sys.path.insert(0,str(ROOT))
    from common import write,guard,utc
    write(ROOT/'human-authorization.json',dict(source='direct human message in implementing chat',date='2026-11-02 Asia/Singapore',text='可以，把反传计账错误修好，先通过预检，再按原定预算跑完A/B对照。loss配方、数据和原物理都别改，完成后主动把结果回传给独立分析对话，让他判断，等我审批再继续。',analysis_chat='__LOCAL_CHAT__'))
    previous=read(OLD/'completion.json')
    write(ROOT/'prior-failure-cost.json',dict(source=str(OLD/'completion.json'),sha256=sha(OLD/'completion.json'),ledger=previous['ledger'],wall_seconds=previous['wall_seconds'],own_CPU_seconds=previous['own_CPU_seconds'],scientific_results=None,erased_or_reclassified=False))
    write(ROOT/'revision.json',dict(reason='explicitly approved reusable backward accounting context repair',old_config_sha256=sha(OLD/'config.json'),changes=['reusable logging context','immediate cross-step gradient receipt','count prior failure root in shared storage','new output root/authorization/bindings'],scientific_formula_data_physics_sampler_unchanged=True,retry_authorized=True,automatic_extra_retries=False,prior_forward_cost=33))
    c=dict(old);c['run_id']='LOSS-AR4-20261102-R1';c['created_utc']=utc();c['revision']=read(ROOT/'revision.json');c['retry_authorization']=str(ROOT/'human-authorization.json')
    paths=[]
    for binding in old['bindings']:
        p=Path(binding['path'])
        if p.is_relative_to(OLD):
            q=ROOT/p.relative_to(OLD)
            if q.exists():paths.append(q)
        else:paths.append(p)
    paths += [ROOT/'prepare_revision.py',ROOT/'human-authorization.json',ROOT/'revision.json',ROOT/'prior-failure-cost.json',ROOT.parent/'hmf-inference-diagnostics-20261102/resource_limits.py']
    c['bindings']=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths))]
    # Preserve every pre-existing scientific/config field byte-for-value.
    exceptions={'run_id','created_utc','bindings'}
    unchanged={k:c[k]==v for k,v in old.items() if k not in exceptions};assert all(unchanged.values())
    assert all(sha(ROOT/name)==sha(OLD/name) for name in ('rollout_loss.py','report.py','PLAN.txt','schedule.json','evaluation-spec.json'))
    write(ROOT/'config.json',c);write(ROOT/'unchanged-protocol-proof.json',dict(original_fields=unchanged,identical_files={n:sha(ROOT/n) for n in ('rollout_loss.py','report.py','PLAN.txt','schedule.json','evaluation-spec.json')},no_numerical_calls=True))
    write(ROOT/'freeze.json',dict(config_sha256=sha(ROOT/'config.json'),storage=guard(full=True),at_utc=utc()))
    for p in ROOT.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
    print(json.dumps(dict(config_sha256=sha(ROOT/'config.json'),bindings=len(c['bindings']),unchanged_protocol=all(unchanged.values()),storage=guard())))
if __name__=='__main__':main()
