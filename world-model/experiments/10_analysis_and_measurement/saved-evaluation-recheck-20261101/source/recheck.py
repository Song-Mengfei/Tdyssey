"""Read-only checkpoint/output recheck; no model construction/inference/training.

CPU torch is used only to read locally produced checkpoint tensors. Predictions
are saved arrays; display quantization is measured without editing any image.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()

dirs={a:ROOT/'artifacts'/('generated-history-ab-20261101' if a in 'AB' else 'region-weighted-cd-20261101') for a in 'ABCD'}
metrics={a:read(p/f'evaluation/{a}/formal/metrics.json') for a,p in dirs.items()}
identities={};states={};prediction_data={}
for arm,p in dirs.items():
    m=metrics[arm];identity=read(p/f'evaluation/{arm}/identity-evidence.json')
    cp=m['model_identity'];actual_file_sha=sha(cp['path']);assert actual_file_sha==cp['sha256']
    payload=torch.load(cp['path'],map_location='cpu',weights_only=False)
    st=payload['model'];actual_state_sha=state_sha(st)
    assert actual_state_sha==identity['model_state_sha256_before']==identity['model_state_sha256_after']
    identities[arm]=dict(checkpoint=cp['path'],checkpoint_sha256=actual_file_sha,
        actual_checkpoint_model_state_sha256=actual_state_sha,
        matches_runtime_evaluated_model=True,formal_prediction_files=len(m['raw_files']),
        formal_case_count=m['cases'],formal_prediction_frames=sum(16 for _ in m['raw_files']),
        formal_sampler_batch_calls=m['sampler_batch_calls'],formal_evaluation_seconds=m['wall_seconds'])
    states[arm]={k:v.detach().cpu().numpy() for k,v in st.items()}
    del payload,st
    prediction_data[arm]={}
    for row in m['raw_files']:
        assert sha(row['path'])==row['sha256']
        with np.load(row['path'],allow_pickle=False) as z:arr=z['raw'].copy()
        assert arr.shape==(16,3,96,96) and arr.dtype==np.float32 and np.isfinite(arr).all()
        prediction_data[arm][row['sample_id']]=arr

assert len({v['actual_checkpoint_model_state_sha256'] for v in identities.values()})==4
comparisons={}
for old,new in [('A','C'),('B','D'),('C','D')]:
    assert set(prediction_data[old])==set(prediction_data[new])
    old_params=states[old];new_params=states[new];assert set(old_params)==set(new_params)
    rows=[]
    for sid,x in prediction_data[old].items():
        y=prediction_data[new][sid]
        raw_eq=np.all(x==y,axis=(1,2,3))
        # This exactly matches report.py's display-only conversion.
        u=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
        v=np.rint(np.clip(y,0,1)*255).astype(np.uint8)
        visible_changed=np.any(u!=v,axis=1).sum(axis=(1,2))
        rows.append(dict(sample_id=sid,sequence_raw_identical=bool(np.array_equal(x,y)),
            frame_raw_identical=raw_eq.tolist(),display_changed_pixels=visible_changed.tolist(),
            H16_raw_mean_absolute_difference=float(np.abs(x[-1]-y[-1]).mean()),
            H16_raw_max_absolute_difference=float(np.abs(x[-1]-y[-1]).max())))
    comparisons[f'{old}_vs_{new}']=dict(cases=len(rows),frames=len(rows)*16,
        state_tensor_count=len(old_params),changed_state_tensor_count=sum(not np.array_equal(old_params[k],new_params[k]) for k in old_params),
        whole_16_frame_sequences_identical=sum(r['sequence_raw_identical'] for r in rows),
        raw_identical_frames=sum(sum(r['frame_raw_identical']) for r in rows),
        display_identical_frames=sum(sum(n==0 for n in r['display_changed_pixels']) for r in rows),
        highlight_counts={str(h):dict(raw_identical=sum(r['frame_raw_identical'][h-1] for r in rows),
            display_identical=sum(r['display_changed_pixels'][h-1]==0 for r in rows),
            median_changed_display_pixels=float(np.median([r['display_changed_pixels'][h-1] for r in rows]))) for h in (1,4,8,16)},rows=rows)

examples=[]
for family in ('AC-O-01-08','AC-C-01-24','AC-B-00-08'):
    entries=[]
    for arm,m in metrics.items():
        r=next(r for r in m['rows'] if r['family_id']==family and r['horizon']==16 and r['method']==arm)
        entries.append(dict(arm=arm,iou=r['metrics']['raw_mask_iou'],position_px=r['metrics']['object_position_error_image_px'],holes=r['metrics']['predicted_holes']))
    examples.append(dict(family=family,H16=entries))
result=dict(kind='parent-saved-checkpoint-and-output-recheck.v1',model_constructions=0,model_inference_calls=0,
    optimizer_updates=0,physics_steps=0,GPU_calls=0,checkpoint_CPU_deserialization_only=True,
    checkpoint_identities=identities,comparisons=comparisons,fixed_examples=examples,
    limits=['Different weights/pixels prove different saved outputs, not meaningful quality improvement.',
            'Runtime provenance is checked against recorded identity and source; no new inference replay was performed.',
            'The ground-truth/epoch2/A/B columns are deliberately reused; C/D are new.',
            'These are repeatedly consulted development cases, not a fresh unseen-shape final test.'])
(OUT/'RECHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(checkpoint_identities=identities,
    comparisons={k:{q:v for q,v in d.items() if q!='rows'} for k,d in comparisons.items()},fixed_examples=examples),ensure_ascii=False,indent=2))
