"""Single-thread saved-only audit, zero Torch/model/new render calls."""
import ctypes,sys,json,struct
from ctypes import wintypes
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
from common import ROOT,read,sha,write,utc,output_io

def main():
    c=read(ROOT/'frozen-config.json');done=read(ROOT/'completion.json');ledger=done['ledger'];arms=c['arms']
    assert done['config_sha256']==sha(ROOT/'frozen-config.json') and done['run_id']==c['run_id']
    # Model stages must be complete; a recovered saved-only report may coexist
    # with an original engineering failure record. Never infer missing frames.
    assert ledger['attempts']==ledger['updates']=={a:128 for a in arms}
    assert ledger['exposures']=={a:2048 for a in arms} and ledger['stack']==ledger['inflight']==[]
    pre=0;images=0;batches=0
    for name,x in ledger['counts'].items():
        assert x['images_attempted']==x['images_completed'] and x['batches_attempted']==x['batches_completed']
        images+=x['images_completed'];batches+=x['batches_completed']
        if name.startswith('preflight/'):pre+=x['images_completed']
        elif name.startswith('train/'):assert x['images_completed']==2048 and x['batches_completed']==512
        elif name.startswith('eval/'):assert x['images_completed']==3744 and x['batches_completed']==960
        else:raise AssertionError('Unexpected model phase')
    assert images==8192+14976+pre and pre<=128 and images<=23296 and batches<=6016
    assert sum(v for k,v in ledger['backward'].items() if k.startswith('train/') and k.endswith('/attempted'))==2048
    assert all(ledger['backward'][k.replace('/attempted','/completed')]==v for k,v in ledger['backward'].items() if k.endswith('/attempted'))
    pgrad=sum(v for k,v in ledger['backward'].items() if k.startswith('preflight/') and k.endswith('/attempted'));assert pgrad<=32
    assert ledger['extra']['model_constructions_attempted']==ledger['extra']['model_constructions_completed']==1
    assert ledger['extra']['preflight_model_only_loads']==1 and ledger['extra']['full_arm_restores']==4
    assert read(ROOT/'preflight.json')['passed'] and read(ROOT/'scoring-parity.json')['passed']
    logs={a:read(ROOT/f'training/{a}-log.json') for a in arms}
    assert all(len(x)==128 and all(r['scaler']['applied']==1 for r in x) for x in logs.values())
    assert all(all(x['conditioning_rng_hashes']==y['conditioning_rng_hashes'] and x['branches']==y['branches'] for x,y in zip(logs[arms[0]],log)) for log in logs.values())
    reference=read(c['E2_metrics']);raw_files=0;raw_bytes=0;predictions=0
    for arm in arms:
        assert all(read(ROOT/f'restore-{arm}.json').values())
        checkpoint=read(ROOT/f'training/{arm}-128.json');assert sha(checkpoint['path'])==checkpoint['sha256'] and Path(checkpoint['path']).stat().st_size==checkpoint['bytes']
        value=read(ROOT/f'evaluation/{arm}/metrics.json');assert len(value['rows'])==value['predictions']==1248 and len(value['raw_files'])==78
        assert value['evidence']==reference['evidence'];predictions+=value['predictions']
        for item in value['raw_files']:
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
            raw_files+=1;raw_bytes+=item['bytes']
    assert raw_files==312 and predictions==4992
    manifest=read(ROOT/'report/final-manifest.json');assert manifest['complete'] and len(manifest['panels'])==12
    mapping=read(ROOT/'report/basename-map.json')
    for panel in manifest['panels']:
        path=Path(panel['path']);assert path.name in mapping and ':' not in path.name
        with path.open('rb') as f:header=f.read(24)
        assert header[:8]==bytes.fromhex('89504e470d0a1a0a')
        assert struct.unpack('>II',header[16:24])==(900,1362)
    summary=dict(passed=True,run_id=c['run_id'],original_worker_failed=done['failed'],current_image_forwards=images,preflight_image_forwards=pre,
        batch_forwards=batches,formal_backward_calls=2048,preflight_model_gradient_calls=pgrad,tensor_only_gradient_calls=read(ROOT/'tensor-only-preflight.json')['tensor_only_gradient_calls'],
        attempts_updates_each=128,unique_native_targets=len(set(i for b in read(ROOT/'schedule.json')['native_h1_batches'] for i in b)),native_exposures_each=2048,
        AMP_skips=0,predictions=predictions,raw_archives_SHA_verified=raw_files,raw_bytes=raw_bytes,lightweight_checkpoints_SHA_verified=4,
        panel_headers_verified=12,paired_native_training=True,E2_reused=True,old_R1_source_cost_not_recharged=True,new_model_calls=0,new_physics=0,new_renders=0,
        storage=output_io().guard(refresh=True),BASE_crosscheck=read(ROOT/'BASE-vs-R1-A128.json'),at_utc=utc())
    write(ROOT/'saved-audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='BASE_crosscheck'}))

if __name__=='__main__':main()

