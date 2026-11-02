"""Stdlib preparation only; original sources are read-only references."""
import ast, json, sys, hashlib, importlib.util, ctypes
from pathlib import Path
from ctypes import wintypes
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]; OLD=ROOT.parent/'loss-screen-20261102'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    assert not (ROOT/'frozen-config.json').exists()
    # Copy reviewed infrastructure, never the original evidence or model assets.
    for n in ['bounded_io.py','continuation_io.py','authorization.py','native_forward.py','state_restore.py','scoring_adapter.py','safe_names.py']:
        p=ROOT/n
        if p.exists():raise FileExistsError(p)
        p.write_bytes((OLD/n).read_bytes())
    sys.path.insert(0,str(ROOT))
    from continuation_io import RetryBudgetIO
    roots=read(OLD/'storage-plan.json')['shared_roots']+[str(ROOT)]
    io=RetryBudgetIO(ROOT,roots,root_cap=650_000_000,metadata=24*1024**2)
    snapshot=io.guard(refresh=True)
    if snapshot['shared']+650_000_000+snapshot['atomic']+snapshot['growth']>10_000_000_000:raise RuntimeError('Entire approved root will not fit shared budget')
    io.write_json('storage-plan.json',dict(shared_roots=roots,root_cap=650000000,shared_cap=10000000000,atomic=268435456,growth=67108864,metadata=25165824,initial_snapshot=snapshot))
    for n in ['schedule.json','evaluation-spec.json','scoring-parity-inputs.json']:
        io.write_bytes(n,(OLD/n).read_bytes(),exclusive=True)
    spec=ROOT.parent/'retrospective-analysis-20261101/loss-definition-review-20261102/TEST_SPEC.txt'
    io.write_bytes('SOURCE_TEST_SPEC.txt',spec.read_bytes(),exclusive=True)
    previous=read(OLD/'frozen-config.json')
    # All inherited bindings remain scientific source identities; previous owner
    # scripts are harmless references and are not new execution instructions.
    bindings={b['path']:b for b in previous['bindings']}
    for p in [spec,OLD/'evaluation/BASE/metrics.json',OLD/'training/BASE-log.json',OLD/'BASE-vs-R1-A128.json',REPO/'tdyssey/native_rgb_env.py',REPO/'tdyssey/appearance.py',*ROOT.glob('*.py'),ROOT/'launch.ps1',ROOT/'schedule.json',ROOT/'evaluation-spec.json',ROOT/'scoring-parity-inputs.json',ROOT/'SOURCE_TEST_SPEC.txt']:
        bindings[str(p.resolve())]=dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    config={key:previous[key] for key in ['repo','python_executable','source_root','parent','parent_sha256','E2_metrics','copy_reference','hid_metrics_path','cpu_limits_path','sampler']}
    config.update(run_id='LOSS-DEFINITION-20261102',arms=['G_ONLY','NATIVE_G','NATIVE_T'],BASE_metrics=str(OLD/'evaluation/BASE/metrics.json'),BASE_log=str(OLD/'training/BASE-log.json'),BASE_parity=str(OLD/'BASE-vs-R1-A128.json'),test_spec_path=str(spec),test_spec_sha256=sha(spec),
        palette=dict(white=[255,255,255],gray=[119,136,153],blue=[65,105,225],source=str(REPO/'tdyssey/native_rgb_env.py'),source_sha256=sha(REPO/'tdyssey/native_rgb_env.py'),source_lines=[227,228,229,230]),
        caps=dict(model_image_forwards=17504,model_batch_forwards=4544,training_image_forwards=6144,evaluation_image_forwards=11232,preflight_image_forwards=128,eval_predictions=3744,training_attempts=384,training_updates=384,training_backward_calls=1536,preflight_gradient_calls=32,tensor_only_gradient_calls=64,model_constructions=1,preflight_model_only_loads=1,full_arm_restores=3),
        windows_task_name='Tdyssey-LDEF-20261102',callback_thread_id='__LOCAL_CHAT__',callback_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        equations='SOURCE_TEST_SPEC sections2-5: fixed RGB unmix, squared Dice G,16 fixed spatial-projection W1 plus area T; cG/cT median first4 trainB4 complete parameter norm ratios; G_ONLY=cG*G,NATIVE_G=native+cG*G,NATIVE_T=native+cT*T',
        no_new_physics=True,no_new_data=True,diagnostic_images_cap=288,diagnostic_definitions_cap=12,diagnostic_panels_cap=12,report_panels_cap=12,
        no_next_experiment=True,bindings=sorted(bindings.values(),key=lambda x:x['path']))
    io.write_json('frozen-config.json',config,exclusive=True);digest=sha(ROOT/'frozen-config.json')
    io.write_json('freeze-receipt.json',dict(run_id=config['run_id'],config_sha256=digest,test_spec_sha256=sha(spec),binding_count=len(bindings),preparation_model_calls=0,preparation_Torch_imports=0,storage=io.guard(refresh=True)),exclusive=True)
    io.write_json('execution-release.json',dict(run_id=config['run_id'],config_sha256=digest,execution_authorized=True,source='Actual human loss-definition review/testing authorization; evidence USER_AUTHORIZATION.json; finalized TEST_SPEC'),exclusive=True)
    print(json.dumps(dict(run_id=config['run_id'],config_sha256=digest,test_spec_sha256=sha(spec),storage=io.guard())))
if __name__=='__main__':main()
