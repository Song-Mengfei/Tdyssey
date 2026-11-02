"""Freeze a separate finite CPU collection phase; never launches a process."""
from pathlib import Path
import argparse,ast,datetime,hashlib,importlib.util,json,uuid
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/original-physics-contact-pilot-20261030'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def closure(starts):
    found=set();pending=list(starts)
    def candidate(module):
        p=ROOT.joinpath(*module.split('.'))
        return p.with_suffix('.py') if p.with_suffix('.py').is_file() else p/'__init__.py' if (p/'__init__.py').is_file() else None
    while pending:
        p=Path(pending.pop()).resolve()
        if p in found:continue
        found.add(p);package=list(p.relative_to(ROOT).parts[:-1])
        for n in ast.walk(ast.parse(p.read_text(encoding='utf8'))):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                base=package[:len(package)-n.level+1] if n.level else []
                parts=base+(n.module.split('.') if n.module else [])
                names=['.'.join(parts)]+['.'.join(parts+[a.name]) for a in n.names]
            for name in names:
                c=candidate(name)
                if c and c.resolve() not in found:pending.append(c)
    return found
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('calibration-v1','calibration-v2','formal','supplement'),required=True)
    p.add_argument('--policy',type=Path);p.add_argument('--review',type=Path);p.add_argument('--letters',default='TLOCXB')
    a=p.parse_args();phase=a.phase
    runtime=OUT/'background'/phase
    if (runtime/'runtime-config.json').exists():raise RuntimeError('Phase already frozen; no retry/alternate labels')
    if not (OLD/'OLD_MODIFIED_PHYSICS_REVOKED.json').is_file():raise RuntimeError('Old authorization must first be revoked')
    for q in (OUT/'validation/collector-readiness.json',OUT/'validation/background-audit/run-v2/receipt.json',OUT/'validation/strict-ui-original-v1/summary.json'):
        v=read(q)
        if not (v.get('passed') is True or v.get('source_freeze_ready') is True):raise RuntimeError('Actual engineering evidence not passed')
    ready=read(OUT/'validation/collector-readiness.json')
    for path,digest in ready['source_bindings'].items():
        if sha(path)!=digest:raise RuntimeError('Collector changed after actual checks '+path)
    ui=read(OUT/'ui-restored-live-health.json');actual=read(OUT/'ui-restored-public-identity.json')
    protocol=read(OUT/'validation/strict-ui-original-v1/summary.json')['protocol_sha256']
    controller=actual['physical_state']['controller']
    if (ui['physics_sha256']!=protocol or controller['physics_hz']!=100 or controller['solver_iterations']!=10 or controller['kp']!=100 or controller['kv']!=20):raise RuntimeError('Actual restored service identity differs')
    if phase!='calibration-v1':
        if a.review is None or read(a.review).get('released_next_phase')!=phase:raise RuntimeError('Separate actual preceding phase review required')
    plan=OUT/'collection/plan.json';policy=(a.policy or OUT/'policies/requested-target-v1.json').resolve()
    if sha(plan)!=ready['plan_sha256']:raise RuntimeError('Frozen scenes changed')
    slots=[s for s in read(plan)['slots'] if s['phase']==phase and s['letter'] in a.letters]
    expected={'calibration-v1':18,'calibration-v2':18,'formal':360,'supplement':12*len(a.letters)}[phase]
    if len(slots)!=expected or (phase!='supplement' and a.letters!='TLOCXB'):raise RuntimeError('Finite scene denominator changed')
    rootsfile=OUT/'storage-roots.json';resource=OUT/'policies/resource-policy.json'
    if not rootsfile.exists():
        roots=read(OLD/'storage-roots.json');roots['roots']=list(dict.fromkeys(roots['roots']+[str(OUT),str(ROOT/'tdyssey/strict_original_contact_pilot.py'),str(ROOT/'tdyssey/native_rgb_data_eligibility.py'),str(ROOT/'scripts/strict_original_contact_pilot.py')]))
        write(rootsfile,roots)
    if not resource.exists():
        res=read(OLD/'policies/resource-policy-v2.json');res.update(kind='strict-original-same-administrative-storage.v1',original_limits_unchanged=True,old_data_still_counted=True)
        write(resource,res)
    runtime.mkdir(parents=True,exist_ok=True)
    run='SOPC-'+phase+'-20261031-'+uuid.uuid4().hex
    starts=[ROOT/'scripts/strict_original_contact_pilot.py',ROOT/'tdyssey/strict_original_contact_pilot.py',ROOT/'tdyssey/native_rgb_data_eligibility.py']
    sources=closure(starts)
    source_root=ROOT.parent/'Tdyssey/third_party/diffusion_policy'
    pin=read(ROOT/'manifests/official_dp.json')
    for rel in ('diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'):
        path=source_root/rel
        if sha(path)!=pin['source_sha256'][rel]:raise RuntimeError('Actual upstream source differs')
        sources.add(path.resolve())
    evidence=[plan,policy,resource,rootsfile,ROOT/'manifests/official_dp.json',OUT/'validation/collector-readiness.json',
        OUT/'validation/background-audit/run-v2/receipt.json',OUT/'validation/strict-ui-original-v1/summary.json',
        OUT/'ui-restored-live-health.json',OUT/'ui-restored-public-identity.json',OUT/'ui-deployment.json',
        OLD/'quarantine/strict-original-exclusion-manifest.json',OLD/'quarantine/guard-check.json',
        OLD/'STRICT_ORIGINAL_CORRECTION-20261031.txt',OLD/'OLD_MODIFIED_PHYSICS_REVOKED.json',
        Path(__file__).resolve(),OUT/'background/phase.py']
    if a.review:evidence.append(a.review.resolve())
    sources.update(v.resolve() for v in evidence)
    bindings={str(v):sha(v) for v in sorted(sources)}
    gatefile=runtime/'execution-gate.json'
    gate=dict(kind='strict-original-verified-execution-release.v1',passed=True,strict_original_restored_verified=True,
        training_authorized=False,phase=phase,run_id=run,protocol_sha256=protocol,plan_sha256=sha(plan),policy_sha256=sha(policy),
        selected_slot_ids=[s['slot_id'] for s in slots],sampling_quality_verified=False,bindings=bindings,
        no_modified_physics=True,model_calls=0,optimizer_updates=0)
    write(gatefile,gate)
    c=dict(kind='strict-original-cpu-phase.v1',run_id=run,phase=phase,released=True,artifact_root=str(OUT),repo=str(ROOT),
        runtime=str(runtime),collection=str(OUT/'collection'),plan=str(plan),policy=str(policy),physics_gate=str(gatefile),
        resource_policy=str(resource),storage_roots=str(rootsfile),source_root=str(source_root),protocol_sha256=protocol,
        run_label=phase,letters=a.letters,expected_attempts=expected,expected_controls=expected*64,
        python_executable=str(ROOT.parent/'Tdyssey/.venv/Scripts/python.exe'),
        collector_progress=str(OUT/'collection/runs'/phase/'progress.json'),
        collector_completion=str(OUT/'collection/runs'/phase/'completion.json'),stop_file=str(OUT/'collection/runs'/phase/'STOP.json'),
        physics_settings=dict(physics_hz=100,dt=.01,iterations=10,control_hz=10,kp=100,kv=20,damping=0,pusher='KINEMATIC'),
        training_authorized=False,model_calls=0,optimizer_updates=0,auto_retry=False,
        next_phase_released=False,bindings=[dict(path=k,sha256=v) for k,v in bindings.items()]+[dict(path=str(gatefile),sha256=sha(gatefile))])
    spec=importlib.util.spec_from_file_location('strict_phase',OUT/'background/phase.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c['collector_command']=m.collector_command(c,'{CONFIG_SHA}')
    config=runtime/'runtime-config.json';write(config,c);digest=sha(config)
    (runtime/'runtime-config.sha256').write_text(digest+'\n',encoding='ascii')
    m.validate(config,digest)
    write(runtime/'notification.json',dict(run_id=run,config_sha256=digest,phase=phase,runtime=str(runtime),
        parent_thread_id='__LOCAL_CHAT__',host='local',old_modified_physics_never_released=True,
        unfinished='One identity check then quiet end; no wait/poll/model/GPU/restart',
        terminal='Read actual completion/ledger/raw hashes; saved-only guarded report permitted. All-attempt denominator and valid legalprefixes separate. Actual callback with receipt required.',
        next_phase='Root must review actual original-phase results then freeze separate bounded next gate; no third calibration or replacement/retry. No model/training/server.',
        final='Actual parent report and MCP receipt; supervision+stage waiting_user_approval; pause existing tdyssey formally; STOP.'))
    print(json.dumps(dict(run_id=run,config=str(config),sha256=digest,bindings=len(c['bindings']),attempts=expected,physics_steps=0,model_calls=0)))
if __name__=='__main__':main()
