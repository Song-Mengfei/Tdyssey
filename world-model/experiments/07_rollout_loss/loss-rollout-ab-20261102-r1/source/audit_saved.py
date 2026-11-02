"""Audit saved results only, with no Torch import, models or inference."""
from common import *
from collections import Counter
import math
def main():
    done=read(ROOT/'completion.json');assert done['complete'] and not done['failed'], 'Execution not complete'
    c=read(ROOT/'config.json');ledger=done['ledger'];schedule=read(ROOT/'schedule.json')
    assert ledger['attempts']=={'A':256,'B':256} and ledger['successful_updates']=={'A':256,'B':256}
    assert ledger['target_exposures']=={'A_native':4096,'B_native':4096,'B_aux':1024}
    expected={'preflight/native':(12,12),'preflight/surrogate':(12,12),'preflight/recompute':(21,21),
        'train/A/native':(1024,4096),'train/B/native':(1024,4096),'train/B/aux':(3072,3072),'train/B/aux_recompute':(3072,3072)}
    for label in ('E2','A128','A256','B128','B256'):expected['eval/'+label]=(960,3744)
    for key,(batches,images) in expected.items():
        actual=ledger['counts'][key]
        assert actual==dict(images_attempted=images,batches_attempted=batches,images_completed=images,batches_completed=batches),(key,actual)
    assert set(ledger['counts'])==set(expected)
    assert ledger['backward_calls']=={'preflight_gradient_attempted':1,'preflight_gradient_completed':1,'preflight_backward_attempted':1,'preflight_backward_completed':1,
        'A_native_attempted':1024,'A_native_completed':1024,'B_native_attempted':1024,'B_native_completed':1024,'B_aux_attempted':256,'B_aux_completed':256}
    a,b=read(ROOT/'training/A-log.json'),read(ROOT/'training/B-log.json')
    assert len(a)==len(b)==256
    assert all(x['conditioning_rng_hashes']==y['conditioning_rng_hashes'] and x['branches']==y['branches'] for x,y in zip(a,b))
    assert all(x['scaler']['applied']==1 for x in a+b)
    for arm in ('A','B'):assert all(read(ROOT/f'restore-{arm}.json').values())
    counts=Counter(i for batch in schedule['native_h1_batches'] for i in batch)
    targets=read(HMF/'selection/index.json')['targets'];native_letters=Counter()
    for idx,n in counts.items():native_letters[targets[idx]['letter']]+=n
    aux_blocks=Counter(schedule['aux_block_indices']);aux_letters=Counter()
    for idx,n in aux_blocks.items():aux_letters[schedule['aux_blocks'][idx]['letter']]+=4*n
    raw=0;pred=0;score_rows=0;verified=0;all_raw_keys=[]
    for label in ('E2','A128','A256','B128','B256'):
        value=read(ROOT/f'evaluation/{label}/metrics.json');pred+=value['predictions'];score_rows+=len(value['rows']);assert len(value['rows'])==1248 and len(value['raw_files'])==78
        keys={(r['sample_id'],r['input_mode'],r['horizon']) for r in value['rows']};assert len(keys)==1248;all_raw_keys.append(keys)
        for r in value['raw_files']:
            # Full SHA verification scans saved bytes once, no decompression/inference.
            assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'];verified+=1;raw+=r['bytes']
    assert all(k==all_raw_keys[0] for k in all_raw_keys) and pred==6240
    assert read(ROOT/'report/evaluation-pairing.json')['passed']
    prior=read(ROOT/'prior-failure-cost.json');prior_forwards=sum(v['images_completed'] for v in prior['ledger']['counts'].values());now=sum(v['images_completed'] for v in ledger['counts'].values())
    checks=read(ROOT/'report/comparison-checks.json')
    summary=dict(passed=True,read_only_model_audit=True,new_model_calls=0,new_physics=0,paired_updates=256,native_exposures_each=4096,aux_exposures_B=1024,
        native_unique_targets=len(counts),native_repeated_exposures_per_arm=4096-len(counts),native_letter_exposures_each=dict(native_letters),aux_unique_blocks=len(aux_blocks),aux_letter_exposures=dict(aux_letters),
        current_image_forwards=now,prior_failed_image_forwards=prior_forwards,cumulative_image_forwards=now+prior_forwards,
        prediction_images=pred,scored_images=score_rows,raw_files_verified=verified,raw_bytes=raw,
        preflight=read(ROOT/'preflight.json'),protocol_unchanged=read(ROOT/'unchanged-protocol-proof.json'),
        numerical_support={x['letter']:x['numeric_directional_support'] for x in checks},original_seen_val_acceptance=read(ROOT/'report/summary.json')['original_seen_val_acceptance'],
        no_automatic_next=True,at_utc=utc())
    write(ROOT/'saved-audit.json',summary);write(ROOT/'storage-after-audit.json',guard(full=True))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('preflight','protocol_unchanged')},ensure_ascii=False))
if __name__=='__main__':main()
