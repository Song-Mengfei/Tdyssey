"""One bounded CPU pass: fresh pinned-source parity6letters + staged UI checks."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback,unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(output,stage):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    with(output/'once-claim.json').open('x',encoding='utf-8')as f:json.dump(dict(pid=os.getpid()),f)
    from tdyssey.native_rgb_original_physics import OriginalPhysicsEnv,PROTOCOL,PROTOCOL_SHA256
    v=load('strict_original_base_validation',ROOT/'scripts/validate_original_playground_physics.py')
    began=time.perf_counter();cost=dict(attempted_controls=0,completed_controls=0,known_substeps=0,unknown_substeps=0,model_calls=0,optimizer_updates=0,GPU_calls=0)
    result=dict(kind='strict-original-UI-restoration-validation.v1',started_at_utc=v.utc(),passed=False,protocol=PROTOCOL,protocol_sha256=PROTOCOL_SHA256,parity=[],cost=cost,sources={},staged_not_deployed=True)
    try:
        for p in [Path(__file__),ROOT/'scripts/validate_original_playground_physics.py',ROOT/'tdyssey/native_rgb_original_physics.py',ROOT/'tdyssey/native_rgb_env.py',ROOT/'tdyssey/envs/official.py',ROOT/'manifests/official_dp.json',*sorted(stage.iterdir())]:
            if p.is_file():result['sources'][str(p.resolve())]=sha(p)
        source=ROOT.parent/'Tdyssey/third_party/diffusion_policy'
        for rel,expected in PROTOCOL['source_sha256'].items():
            actual=sha(source/rel)
            if actual!=expected:raise AssertionError('Pinned original source differs: '+rel)
            result['sources'][str((source/rel).resolve())]=actual
        for letter in'TLOCXB':result['parity'].append(v.parity(source,letter,output,cost))
        if cost['attempted_controls']!=96 or cost['completed_controls']!=96 or cost['known_substeps']!=960:raise AssertionError('Bounded six-letter reference cost differs')
        sys.path.insert(0,str(stage))
        server=load('server',stage/'server.py')
        server.REPO=ROOT;server.WORKSPACE=ROOT.parent;server.PHYSICS=source
        server.GPC=ROOT.parent/'Tdyssey/third_party/gpc_code'
        server.RUN=ROOT/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030'
        server.CHECKPOINT=server.RUN/'run/last.pt'
        server.RUNTIME=ROOT/'artifacts/r3b-playground';server.DIAGNOSTICS=ROOT/'artifacts/r3b-playground-diagnostics-20261030'
        tests=load('strict_original_staged_ui_tests',stage/'test_playground.py')
        ui_cost=dict(attempted_controls=0,completed_controls=0,known_substeps=0,synthetic_predictor_calls=0,model_calls=0,GPU_calls=0,optimizer_updates=0)
        observe=OriginalPhysicsEnv._observe_step;step=OriginalPhysicsEnv.step;spy=tests.Spy.__call__
        def counted_observe(self,dt):
            count=len(self.last_substeps)
            try:return observe(self,dt)
            finally:ui_cost['known_substeps']+=sum(bool(r.get('executed'))for r in self.last_substeps[count:])
        def counted_step(self,action):
            ui_cost['attempted_controls']+=1
            value=step(self,action);ui_cost['completed_controls']+=1;return value
        def counted_spy(self,*args,**kwargs):ui_cost['synthetic_predictor_calls']+=1;return spy(self,*args,**kwargs)
        OriginalPhysicsEnv._observe_step=counted_observe;OriginalPhysicsEnv.step=counted_step;tests.Spy.__call__=counted_spy
        try:
            test_began=time.perf_counter()
            with(output/'ui-test-output.txt').open('w',encoding='utf-8')as stream:
                ran=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
            ui_cost['wall_seconds']=time.perf_counter()-test_began
            result['UI_tests']=dict(passed=ran.wasSuccessful(),tests_run=ran.testsRun,failures=len(ran.failures),errors=len(ran.errors),skipped=len(ran.skipped),cost=ui_cost)
        finally:OriginalPhysicsEnv._observe_step=observe;OriginalPhysicsEnv.step=step;tests.Spy.__call__=spy
        if not ran.wasSuccessful():raise AssertionError('Staged original UI tests failed; see ui-test-output.txt')
        result['passed']=True
        result['interpretation']='All6 letters execute pinned PushTEnv.step at100Hz/solver10 with original controller; fresh construction and complete identical action prefix match state15 and RGB at every measured substep/control. UI tests use only synthetic predictor arrays. No arbitrary mid-contact snapshot equivalence or penetration repair is claimed.'
    except BaseException as exc:result['failure']=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        cost['wall_seconds_total']=time.perf_counter()-began;cost['simulated_seconds_both_worlds']=cost['known_substeps']/100
        cost['torch_imported']='torch'in sys.modules;result['finished_at_utc']=v.utc()
        v.write(output/'summary.json',result)
        v.write(output/'inventory.json',dict(files=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(output.rglob('*'))if p.is_file()and p.name!='inventory.json']))
    print(json.dumps(dict(passed=result['passed'],parity_letters=len(result['parity']),cost=cost,UI_tests=result.get('UI_tests'),failure=result.get('failure')),ensure_ascii=False))
    return 0 if result['passed']else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required for bounded CPU physics verification')
    raise SystemExit(run(args.output,args.stage))
