"""Visible-axis diagnostic on saved O RGB, calibrated against saved truth angles.

No model calls or physical steps. PCA is not a rigid-pose certificate and can be
ambiguous if the predicted shape loses its long axis. Both grey and hole axes
are retained, with eigenvalue ratios and true-image calibration error.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_evaluation import palette_masks
RUN=ROOT/'artifacts/generated-history-ab-20261101'
OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
cal=read(ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
spec=read(RUN/'formal-spec.json')
evaluations={a:read(RUN/f'evaluation/{a}/formal/metrics.json') for a in ('baseline','A','B')}

def axis(mask):
    y,x=np.nonzero(mask)
    if len(x)<3:return None
    xy=np.column_stack([x,y]).astype(float)
    eig,vec=np.linalg.eigh(np.cov(xy,rowvar=False,bias=True))
    angle=float(np.arctan2(vec[1,-1],vec[0,-1])*180/np.pi)
    return dict(angle=angle,ratio=float(eig[-1]/max(eig[0],1e-10)),pixels=len(x))

def axes(image):
    gray,blue=palette_masks(image,cal)
    labels,count=ndimage.label(~np.pad(gray,1),structure=ndimage.generate_binary_structure(2,1))
    outside=labels[0,0];sizes=np.bincount(labels.ravel())
    ids=[i for i in range(1,count+1) if i!=outside and sizes[i]>=2]
    hole=labels[1:-1,1:-1]==ids[0] if len(ids)==1 else np.zeros_like(gray)
    return dict(gray=axis(gray),hole=axis(hole),hole_count=len(ids))

out=[]
for row in spec['rows']:
    if row['letter']!='O':continue
    a=row['anchor'];sid=row['sample_id']
    with np.load(row['raw_path'],allow_pickle=False) as z:
        truth=z['rgb'][a:a+17].copy();states=z['states'][a:a+17].copy()
    recorded=np.degrees(np.unwrap(states[:,4])-states[0,4])
    tracks={'truth':truth}
    for arm,evaluation in evaluations.items():
        rf=next(r for r in evaluation['raw_files'] if r['sample_id']==sid)
        with np.load(rf['path'],allow_pickle=False) as z:pred=z['raw'].copy()
        tracks[arm]=np.concatenate((truth[:1].astype(np.float32)/255,pred),axis=0)
    measures={}
    for name,images in tracks.items():
        vals=[axes(im) for im in images];measures[name]={}
        for region in ('gray','hole'):
            if any(v[region] is None for v in vals):
                measures[name][region]={'status':'unavailable'};continue
            angles=np.array([v[region]['angle'] for v in vals])
            delta=np.degrees(np.unwrap(np.radians(2*angles)))/2
            delta-=delta[0]
            measures[name][region]=dict(status='visible_axis_only',clockwise_positive_image_y_down=True,
                delta_deg=delta.tolist(),axis_ratios=[v[region]['ratio'] for v in vals],
                min_axis_ratio=min(v[region]['ratio'] for v in vals))
    truth_error={}
    for region in ('gray','hole'):
        m=measures['truth'][region]
        if 'delta_deg' in m:
            error=np.asarray(m['delta_deg'])-recorded
            truth_error[region]=dict(max_abs_error_deg=float(abs(error).max()),mean_abs_error_deg=float(abs(error).mean()))
    out.append(dict(sample_id=sid,recorded_physical_delta_deg=recorded.tolist(),
        truth_axis_calibration_error=truth_error,visible_axes=measures))
result=dict(kind='saved-O-visible-axis-direction-diagnostic.v1',model_calls=0,physics_steps=0,
    method='PCA long axes of visible grey pixels and of the single visible enclosed white hole; 180deg unwrapping in time; compare true RGB with recorded true angles.',
    limits=['Post-hoc diagnostic, not the original formal angle score.',
            'Deformed or nearly circular predictions may not admit a rigid angle.',
            'PCA can swap axes by 90deg when anisotropy changes; inspect both regions, ratios and frame continuity.',
            'Truth states only calibrate measurement; no output repair or model input.'],rows=out)
(OUT/'O_DIRECTION_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for r in out:
    if 'AC-O-01-08:' not in r['sample_id']:continue
    print('recorded physical clockwise degrees',[round(r['recorded_physical_delta_deg'][h],2) for h in (1,4,8,16)])
    print('true RGB calibration',r['truth_axis_calibration_error'])
    for name,vals in r['visible_axes'].items():
        for region,v in vals.items():
            print(name,region,'deltas',[round(v['delta_deg'][h],2) for h in (1,4,8,16)] if 'delta_deg' in v else 'unknown','minimum axis ratio',round(v.get('min_axis_ratio',0),3))
