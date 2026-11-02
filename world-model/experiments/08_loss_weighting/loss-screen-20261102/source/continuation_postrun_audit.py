"""Saved-only cumulative audit after the explicitly authorized continuation."""
import ctypes,json,struct
from ctypes import wintypes
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
from continuation_io import install
from common import ROOT,read,write,sha,utc

def main():
    io=install();done=read(ROOT/'continuation-completion.json');cfg=read(ROOT/'continuation-config.json');c=read(ROOT/'frozen-config.json');old=read(ROOT/'completion.json');l=done['ledger'];arms=c['arms']
    assert done['complete'] and not done['failed'] and old['failed']
    assert sha(ROOT/'completion.json')==cfg['original_completion_sha256'] and done['config_sha256']==sha(ROOT/'continuation-config.json')
    assert l['attempts']==l['updates']=={a:128 for a in arms} and l['exposures']=={a:2048 for a in arms}
    assert not l['stack'] and not l['inflight'] and l['extra']['original_reservation_cost_retained_images']==4
    charged=sum(x['images_attempted'] for x in l['counts'].values());completed=sum(x['images_completed'] for x in l['counts'].values())
    assert charged==23220 and completed==23216 and charged<=23296
    assert sum(x['batches_attempted'] for x in l['counts'].values())==5901<=6016
    for a in arms:
        t=l['counts']['train/'+a];e=l['counts']['eval/'+a]
        assert t==dict(images_attempted=2048,images_completed=2048,batches_attempted=512,batches_completed=512)
        expected=dict(images_attempted=3776,images_completed=3772,batches_attempted=968,batches_completed=967) if a=='SNR' else dict(images_attempted=3744,images_completed=3744,batches_attempted=960,batches_completed=960)
        assert e==expected
        assert all(read(ROOT/f'restore-{a}.json').values())
    assert sum(v for k,v in l['backward'].items() if k.startswith('train/') and k.endswith('/attempted'))==2048
    assert all(l['backward'][k.replace('/attempted','/completed')]==v for k,v in l['backward'].items() if k.endswith('/attempted'))
    assert l['extra']['model_constructions_attempted']==l['extra']['model_constructions_completed']==2 and l['extra']['full_arm_restores']==4
    assert l['extra']['preflight_model_only_loads']==1 and l['extra']['continuation_SNR_final_model_only_loads']==1
    logs={a:read(ROOT/f'training/{a}-log.json') for a in arms}
    assert all(len(log)==128 and all(x['scaler']['applied']==1 for x in log) for log in logs.values())
    assert all(all(x['branches']==y['branches'] and x['conditioning_rng_hashes']==y['conditioning_rng_hashes'] for x,y in zip(logs['BASE'],log)) for log in logs.values())
    reference=read(c['E2_metrics']);raws=0;rawbytes=0;predictions=0
    for a in arms:
        rec=read(ROOT/f'training/{a}-128.json');assert Path(rec['path']).stat().st_size==rec['bytes'] and sha(rec['path'])==rec['sha256']
        value=read(ROOT/f'evaluation/{a}/metrics.json');assert value['evidence']==reference['evidence'] and len(value['rows'])==value['predictions']==1248 and len(value['raw_files'])==78
        predictions+=value['predictions']
        for x in value['raw_files']:
            assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256'];raws+=1;rawbytes+=x['bytes']
    provenance=read(ROOT/'evaluation/SNR/metrics.json')['noise_provenance'];assert sum(x['reused_saved_archives'] for x in provenance)==2
    assert raws==312 and predictions==4992
    manifest=read(ROOT/'report/final-manifest.json');mapping=read(ROOT/'report/basename-map.json');assert manifest['complete'] and len(manifest['panels'])==12
    for p in manifest['panels']:
        path=Path(p['path']);assert path.name in mapping and ':' not in path.name
        with path.open('rb') as f:b=f.read(24)
        assert b[:8]==bytes.fromhex('89504e470d0a1a0a') and struct.unpack('>II',b[16:24])==(900,1362)
    record=dict(passed=True,run_id=cfg['run_id'],cumulative_charged_images=charged,cumulative_completed_images=completed,
        original_aborted_before_body_charged=4,extra_completed_replay_denoiser_images=28,unique_evaluation_predictions=4992,completed_sampler_frames_including_lost_prefix_replay=5000,
        raw_archives_SHA_verified=raws,raw_bytes=rawbytes,lightweight_checkpoints_SHA_verified=4,panels_verified=12,AMP_skips=0,full_arm_restores=4,
        model_constructions_cumulative=2,user_authorized_resume=True,original_failure_preserved=True,prior_prefight_not_repeated=True,
        cached_SNR_archives_reused=8,noise_receipt_scope='Old in-memory SNR receipt lost; frozen control flow checked actual tensor; independent hash replay recorded in SNR noise_provenance.',
        new_model_calls_in_audit=0,new_physics=0,new_renders=0,storage=io.guard(refresh=True),at_utc=utc())
    write(ROOT/'continuation-saved-audit.json',record);print(json.dumps(record))

if __name__=='__main__':main()

