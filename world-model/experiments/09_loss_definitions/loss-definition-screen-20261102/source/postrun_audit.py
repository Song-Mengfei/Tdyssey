"""One saved-only audit. No Torch, model, physical replay or new rendering."""
import sys,ctypes,json,struct
from ctypes import wintypes
from pathlib import Path
from common import ROOT,read,sha,write,output_io,utc
def main():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    if (ROOT/'saved-audit.json').exists():raise RuntimeError('Audit already completed')
    c=read(ROOT/'frozen-config.json');done=read(ROOT/'completion.json');stages=read(ROOT/'model-stages-complete.json');ledger=done['ledger'];arms=stages['executed_arms']
    assert done['complete'] and not done['failed'] and done['config_sha256']==sha(ROOT/'frozen-config.json') and done['run_id']==c['run_id']
    assert ledger['attempts']==ledger['updates']=={a:128 for a in arms} and ledger['exposures']=={a:2048 for a in arms}
    assert ledger['stack']==ledger['inflight']==[]
    pre=images=batches=0
    for name,v in ledger['counts'].items():
        assert v['images_attempted']==v['images_completed'] and v['batches_attempted']==v['batches_completed']
        images+=v['images_completed'];batches+=v['batches_completed']
        if name.startswith('preflight/'):pre+=v['images_completed']
        elif name.startswith('train/'):assert v['images_completed']==2048 and v['batches_completed']==512
        elif name.startswith('eval/'):assert v['images_completed']==3744 and v['batches_completed']==960
        else:raise AssertionError('Unexpected phase')
    assert images==len(arms)*(2048+3744)+pre and pre<=128 and images<=17504 and batches<=4544
    assert sum(v for k,v in ledger['backward'].items() if k.startswith('train/') and k.endswith('/attempted'))==len(arms)*512
    assert all(ledger['backward'][k.replace('/attempted','/completed')]==v for k,v in ledger['backward'].items() if k.endswith('/attempted'))
    pgrad=sum(v for k,v in ledger['backward'].items() if k.startswith('preflight/') and k.endswith('/attempted'));assert pgrad<=32
    assert ledger['extra']['model_constructions_attempted']==ledger['extra']['model_constructions_completed']==1
    assert ledger['extra']['preflight_model_only_loads']==1 and ledger['extra']['full_arm_restores']==len(arms)
    preflight=read(ROOT/'preflight.json');assert preflight['native_parity_passed'] and preflight['coefficients_sha256']==sha(ROOT/'coefficients.json')
    assert read(ROOT/'scoring-parity.json')['passed']
    base=read(c['BASE_log']);reference=read(c['E2_metrics']);raws=raw_bytes=predictions=0
    for arm in arms:
        assert all(read(ROOT/f'restore-{arm}.json').values())
        log=read(ROOT/f'training/{arm}-log.json');assert len(log)==128 and all(x['scaler']['applied']==1 for x in log)
        assert all(x['branches']==y['branches'] and x['conditioning_rng_hashes']==y['conditioning_rng_hashes'] for x,y in zip(log,base))
        weight=read(ROOT/f'training/{arm}-128.json');assert sha(weight['path'])==weight['sha256'] and Path(weight['path']).stat().st_size==weight['bytes']
        value=read(ROOT/f'evaluation/{arm}/metrics.json');assert value['predictions']==len(value['rows'])==len(value['auxiliary_RGB_mask_diagnostics'])==1248
        assert len(value['raw_files'])==78 and value['evidence']==reference['evidence'];predictions+=1248
        for item in value['raw_files']:
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'];raws+=1;raw_bytes+=item['bytes']
    diagnosis=read(ROOT/'diagnosis/summary.json');assert diagnosis['complete'] and diagnosis['model_calls']==0 and diagnosis['constructed_images']<=288 and diagnosis['tensor_only_gradient_calls']<=64
    manifest=read(ROOT/'report/final-manifest.json');assert manifest['complete'] and len(manifest['panels'])==12 and len(manifest['diagnostic_panels'])==12
    mapping=read(ROOT/'report/basename-map.json')
    for panel in manifest['panels']+manifest['diagnostic_panels']:
        p=Path(panel['path']);assert p.is_relative_to(ROOT) and ':' not in p.name
        header=p.read_bytes()[:24];assert header[:8]==bytes.fromhex('89504e470d0a1a0a')
        width,height=struct.unpack('>II',header[16:24])
        if 'width' in panel:assert (width,height)==(panel['width'],panel['height']) and p.name in mapping
        else:assert (width,height)==(624,488)
    for b in c['bindings']:
        assert sha(b['path'])==b['sha256']
    # Immutable journal sequence is complete and its final snapshot matches costs.
    events=sorted((ROOT/'events').glob('*.json'));assert len(events)==ledger['sequence']
    assert all(p.name==f'{i:06d}.json' for i,p in enumerate(events,1));last=read(events[-1])
    for field in ['counts','attempts','updates','exposures','backward']:assert last[field]==ledger[field]
    result=dict(passed=True,run_id=c['run_id'],config_sha256=sha(ROOT/'frozen-config.json'),executed_arms=arms,skipped=stages['skipped'],all_three_arms_complete=len(arms)==3,
        image_forwards=images,preflight_images=pre,batch_forwards=batches,formal_backwards=len(arms)*512,preflight_gradient_calls=pgrad,tensor_only_gradient_calls=diagnosis['tensor_only_gradient_calls'],attempts_updates_each=128,native_exposures_each=2048,
        raw_archives_SHA_verified=raws,raw_bytes=raw_bytes,predictions=predictions,lightweight_weights_SHA_verified=len(arms),panel_headers_verified=24,immutable_cost_events=len(events),AMP_skips=0,
        coefficients_sha256=sha(ROOT/'coefficients.json'),E2_BASE_reused=True,BASE_R1_previous_parity=read(c['BASE_parity']),new_model_calls=0,new_physics=0,new_renders=0,storage=output_io().guard(refresh=True),at_utc=utc())
    write(ROOT/'saved-audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
