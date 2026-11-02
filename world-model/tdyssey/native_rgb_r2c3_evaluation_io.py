"""Lossless endpoint references into saved raw steps; no model or scoring calls."""
from pathlib import Path
import hashlib,json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def safe(root,name):
    root=Path(root).resolve();p=(root/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root):raise ValueError('Raw reference escapes cohort folder')
    return p

def load_referenced_predictions(folder,cohort,prepared,calibration=None):
    """Rebuild the original float64 endpoint containers with exact float32 values.

    The original runtime still scores its float64 rollout endpoints. This loader
    avoids storing those same pixels twice, and uses no future RGB for feedback.
    Missing endpoints remain NaN with their recorded unattempted/failure status.
    """
    import numpy as np
    folder=Path(folder).resolve();side=folder/f'cohort-{cohort}.json'
    ids=list(prepared['window_ids']);n=len(ids);h={1:np.full((n,3,96,96),np.nan),4:np.full((n,3,96,96),np.nan)}
    raw_steps=[[None]*4 for _ in ids]
    if not side.exists():
        result=dict(window_ids=ids,planned_windows=n,h1=h[1],h4=h[4],status_h1=['missing_file']*n,status_h4=['missing_file']*n,hard_stop=True)
        return result,raw_steps,dict(files_missing=True,raw_steps_expected=4*n,raw_steps_present=0)
    s=read(side)
    if s['window_ids']!=ids or s['cohort_identity']!=prepared['identity']:raise ValueError('Endpoint reference cohort identity differs')
    pos={key:i for i,key in enumerate(ids)};loaded={};inventory={};present=0
    for item in s['raw_step_files']:
        path=safe(folder,item['file'])
        if item['file'] in inventory:raise ValueError('Duplicate raw file reference')
        inventory[item['file']]=item
        if sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Raw source SHA/bytes differs')
        with np.load(path,allow_pickle=False) as z:raw=z['raw'].copy();keys=z['window_ids'].tolist()
        if keys!=item['window_ids'] or raw.shape!=(len(keys),3,96,96) or raw.dtype!=np.float32:raise ValueError('Raw batch identity/shape/dtype differs')
        step=item['step']
        if type(step)is not int or not 0<=step<4:raise ValueError('Invalid raw step')
        loaded[item['file']]=(raw,keys)
        for row,key in enumerate(keys):
            if key not in pos or raw_steps[pos[key]][step] is not None:raise ValueError('Duplicate/unknown raw slot')
            raw_steps[pos[key]][step]=raw[row];present+=1
    refs=s['raw_endpoint_refs']
    if set(refs)!=set(ids):raise ValueError('Endpoint reference denominator differs')
    for index,key in enumerate(ids):
        for horizon,step in ((1,0),(4,3)):
            ref=refs[key][f'H{horizon}']
            if ref is None:continue
            if (ref['step']!=step or ref['file'] not in loaded or ref['sha256']!=inventory[ref['file']]['sha256']
                    or inventory[ref['file']]['step']!=step):raise ValueError('Endpoint reference step/hash differs')
            raw,keys=loaded[ref['file']];row=ref['row']
            if type(row)is not int or not 0<=row<len(raw) or keys[row]!=key:raise ValueError('Endpoint reference row/window differs')
            if raw_steps[index][step] is None or not np.array_equal(raw_steps[index][step],raw[row],equal_nan=True):raise ValueError('Endpoint not its actual first/fourth raw prediction')
            h[horizon][index]=raw[row]
    for horizon in (1,4):
        if len(s[f'status_h{horizon}'])!=n:raise ValueError('Endpoint status denominator differs')
    result=dict(window_ids=ids,planned_windows=n,h1=h[1],h4=h[4],status_h1=s['status_h1'],status_h4=s['status_h4'],
        hard_stop=s['hard_stop'],step_records=s['step_records'],h1_reuses_first_h4_call=True)
    return result,raw_steps,dict(sidecar=str(side),sha256=sha(side),raw_steps_expected=4*n,raw_steps_present=present,
        endpoint_storage='lossless raw step references; original float64 value containers reconstructed',scoring_calls=0)
