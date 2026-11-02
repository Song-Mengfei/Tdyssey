"""One new-data epoch and paired raw recursive predictions; no import-time Torch.

The worker owns execution. Original R3B model/objective/noise policy/Adam and
sampler are reused. Only recorded new RGB/action arrays cross the model boundary.
"""
from __future__ import annotations
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PARENT = REPO/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt'
PARENT_SHA = '23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
SAMPLER = dict(num_steps_denoising=3, sigma_min=.002, sigma_max=5., rho=7, order=1, s_churn=0., s_cond=0.)
HORIZONS = (1, 4, 8, 16)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def atomic_json(path, value):
    from tdyssey.native_rgb_training import atomic_json as write
    write(path, value)
def tree_bytes(root): return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())

def storage_guard(root, config, additional=0):
    if 'shared_storage_roots' in config:
        from common import check_storage
        # Atomic reserve remains present in addition to the prospective output.
        return check_storage(root,config,reserve_bytes=int(additional))
    roots = config.get('shared_roots', config.get('storage_roots', [str(root)]))
    if isinstance(roots, (str, Path)):
        value = read(roots); roots = value if isinstance(value, list) else value.get('roots', value.get('shared_roots', []))
    normalized = []
    for item in roots:
        p = Path(item['path'] if isinstance(item, dict) else item).resolve()
        if not any(p.is_relative_to(x) for x in normalized):
            normalized = [x for x in normalized if not x.is_relative_to(p)]; normalized.append(p)
    if not any(Path(root).resolve().is_relative_to(p) for p in normalized): normalized.append(Path(root).resolve())
    used = sum(tree_bytes(p) for p in normalized)
    cap = int(config.get('shared_cap_bytes', config.get('storage_cap_bytes', 10_000_000_000)))
    reserve = int(config.get('atomic_reserve_bytes', 256*1024**2))
    if used + int(additional) + reserve > cap: raise RuntimeError('Shared10G storage including256MiB write reserve unavailable')
    if shutil.disk_usage(root).free < reserve + int(additional): raise RuntimeError('Actual disk write reserve unavailable')
    return dict(used_bytes=used, cap_bytes=cap, atomic_reserve_bytes=reserve)

def checkpoint_preflight(path=PARENT, expected_sha=PARENT_SHA):
    """Explicit trusted full checkpoint CPU inspection; never builds a model."""
    import torch
    torch.set_num_threads(2)
    path = Path(path).resolve()
    actual = sha(path)
    if actual != expected_sha: raise ValueError('Latest complete7499 parent checkpoint hash differs')
    p = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    required = {'model','optimizer','scaler','global_rng','conditioning_rngs','order','progress','identity','legacy_scheduler_provenance'}
    if p.get('kind') != 'native_gpc_rgb_R3B_training.v1' or not required.issubset(p): raise ValueError('Latest parent is not the complete R3B checkpoint')
    q = p['progress']
    if q.get('lineage_updates') != 7499 or not q.get('epoch_complete') or not q.get('complete_optimizer_boundary') or q.get('pending_attempt') is not None: raise ValueError('Latest7499 parent boundary incomplete')
    if any(not bool(torch.isfinite(t).all()) for t in p['model'].values() if torch.is_floating_point(t)): raise ValueError('Nonfinite checkpoint weights')
    if any(abs(g['lr']-1e-5)>1e-14 for g in p['optimizer']['param_groups']): raise ValueError('Parent held learning rate differs')
    return dict(path=str(path), sha256=actual, bytes=path.stat().st_size, kind=p['kind'], progress=deepcopy(q),
                model_tensors=len(p['model']), conditioning_policy_sha256=p['conditioning_rngs']['policy_sha256'],
                optimizer='AdamW held1e-5', scaler=deepcopy(p['scaler']), full_state_readable=True, cpu_only=True)

class RawStore:
    """Small bounded saved-array cache; no simulator or renderer."""
    def __init__(self, max_bytes=48*1024**2): self.cache=OrderedDict(); self.bytes=0; self.cap=max_bytes; self.loads=0
    def arrays(self, row):
        key = str(Path(row['raw_path']).resolve())
        if key in self.cache: self.cache.move_to_end(key); return self.cache[key]
        if row.get('raw_sha256') and sha(key) != row['raw_sha256']: raise ValueError('Frozen selected raw changed')
        with np.load(key, allow_pickle=False) as z:
            value = {k:z[k].copy() for k in ('rgb','states','requested_actions','passed_actions','control_valid') if k in z}
        rgb=value['rgb']; actions=value['passed_actions']
        if rgb.dtype != np.uint8 or rgb.shape[1:] != (3,96,96) or len(rgb)!=len(actions)+1: raise ValueError('Saved RGB/action timeline mismatch')
        if not np.array_equal(actions, value['requested_actions']): raise ValueError('Requested/executed targets differ')
        size=sum(v.nbytes for v in value.values()); self.loads+=1
        while self.cache and self.bytes+size>self.cap:
            _,gone=self.cache.popitem(last=False); self.bytes-=sum(v.nbytes for v in gone.values())
        if size<=self.cap: self.cache[key]=value; self.bytes+=size
        return value
    def training(self, row):
        from tdyssey.native_rgb_r3a_data import assert_model_inputs
        d=self.arrays(row); a=int(row['anchor']); actions=d['passed_actions'][a-3:a+1]
        if a<3 or a+1>=len(d['rgb']) or not d['control_valid'][a-3:a+1].all(): raise ValueError('Incomplete/invalid selected training window')
        if not np.isfinite(actions).all() or actions.min()<0 or actions.max()>511: raise ValueError('Selected window outside unchanged model0..511 action contract')
        return assert_model_inputs(dict(image=d['rgb'][a-3:a+2].astype(np.float32)/np.float32(255),
            action=actions.astype(np.float32)/np.float32(255.5)-np.float32(1)))
    def batch(self, rows):
        items=[self.training(r) for r in rows]
        return {k:np.stack([x[k] for x in items]) for k in ('image','action')}

def validate_indices(train, val):
    ids=[r['sample_id'] for r in train]
    if not ids or len(set(ids))!=len(ids): raise ValueError('Empty or repeated selected training samples')
    if {r['family_id'] for r in train}&{r['family_id'] for r in val}: raise ValueError('Family leaked across training/seen-validation')
    if any(r['letter'] not in 'TLOCXB' for r in train+val): raise ValueError('Reserved/unapproved letter entered model inputs')
    if any(int(r['anchor'])<3 for r in train+val): raise ValueError('Four real history frames required')

def epoch_batches(rows, seed=20261031):
    """Family/event round-robin order; every already-selected sample exactly once."""
    rng=random.Random(seed); pools=defaultdict(list)
    for i,r in enumerate(rows): pools[(r['family_id'],str(r.get('event','unknown')))].append(i)
    keys=sorted(pools); rng.shuffle(keys)
    for p in pools.values(): rng.shuffle(p)
    flat=[]
    while keys:
        remaining=[]
        for key in keys:
            flat.append(pools[key].pop())
            if pools[key]: remaining.append(key)
        keys=remaining
    if sorted(flat)!=list(range(len(rows))): raise AssertionError('Epoch repeats/drops a sample')
    return [flat[i:i+16] for i in range(0,len(flat),16)]

def rollout_raw(sample, histories, absolute_actions, *, seeds):
    """Future truth is absent. Each raw generated frame becomes history unchanged."""
    history=np.asarray(histories, dtype=np.float32).copy(); actions=np.asarray(absolute_actions)
    if history.shape[1:]!=(4,3,96,96) or actions.shape!=(len(history),19,2) or len(seeds)!=16: raise ValueError('Recursive H16 input shape mismatch')
    if not np.isfinite(history).all() or history.min()<0 or history.max()>1: raise ValueError('Initial real history outside0..1')
    if not np.isfinite(actions).all() or actions.min()<0 or actions.max()>511: raise ValueError('Unchanged action0..511 contract required')
    normalized=actions.astype(np.float32)/np.float32(255.5)-np.float32(1); frames=[]
    for step in range(16):
        raw=np.asarray(sample(history.copy(), normalized[:,step:step+4].copy(), seed=int(seeds[step])))
        if raw.dtype!=np.float32 or raw.shape!=(len(history),3,96,96) or not np.isfinite(raw).all(): raise FloatingPointError('Nonfinite/wrong-shape native raw sampler output')
        frames.append(raw.copy()); history=np.concatenate((history[:,1:],raw[:,None]),axis=1)
    return np.stack(frames,axis=1)

def eval_spec(rows, cap=48, seed=20261031):
    """One seeded uniform family per letter/task first; no motion-based choice."""
    eligible=[r for r in rows if int(r.get('horizon',16))>=16 and r.get('committed_future',True)]
    groups=defaultdict(list)
    for r in eligible: groups[(r['letter'],r['task'])].append(r)
    rng=random.Random(seed); chosen=[]; families=set()
    for key in sorted(groups):
        values=sorted(groups[key],key=lambda r:r['sample_id']); rng.shuffle(values)
        found=next((r for r in values if r['family_id'] not in families),None)
        if found is not None: chosen.append(found); families.add(found['family_id'])
    remaining=[r for r in sorted(eligible,key=lambda r:r['sample_id']) if r['family_id'] not in families]; rng.shuffle(remaining)
    for r in remaining:
        if len(chosen)>=cap: break
        if r['family_id'] not in families: chosen.append(r); families.add(r['family_id'])
    chosen=chosen[:cap]; batches=[]
    for at in range(0,len(chosen),4):
        rs=chosen[at:at+4]; ids=[r['sample_id'] for r in rs]
        seeds=[int(digest(dict(domain='accelerated-paired-H16.v1',seed=seed,ids=ids,step=k))[:8],16) for k in range(16)]
        batches.append(dict(rows=rs,seeds=seeds))
    return dict(rows=chosen,batches=batches,selection='Seeded uniform family/letter/task, no prediction or future-motion filtering',
        cap=cap,seed=seed,sampler=SAMPLER,horizons=list(HORIZONS),truth_feedback=False,output_repairs=False)

def raw_npz(path, config, root, **arrays):
    storage_guard(root,config,sum(x.nbytes for x in arrays.values())+1024**2)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    try:
        with tmp.open('xb') as stream: np.savez_compressed(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def _summary(rows):
    fields=('raw_mask_iou','raw_contour_error_image_px','object_position_error_image_px','pusher_mask_iou',
        'pusher_position_error_image_px','rgb_full_mae','holes_difference','area_fraction_error','motion_vector_error_image_px','visible_angle_error_deg')
    result=dict(denominator=len(rows),hard_generation_failures=sum(r['metrics']['hard_generation_failure'] for r in rows),means={})
    for f in fields:
        v=[r['metrics'].get(f) for r in rows]; vals=[float(x) for x in v if x is not None and math.isfinite(float(x))]
        result['means'][f]=dict(mean=float(np.mean(vals)) if vals else None,available=len(vals),unavailable=len(rows)-len(vals))
    result['shape_assessable']=sum(r['metrics'].get('legacy_shape_assessable',False) for r in rows)
    result['shape_failed']=sum(r['metrics'].get('legacy_shape_deformation_failure') is True for r in rows)
    result['pusher_missing']=sum('empty_pusher' in r['metrics'].get('failure_reasons',[]) for r in rows)
    result['object_missing']=sum('empty_object' in r['metrics'].get('failure_reasons',[]) for r in rows)
    return result

def _metric(raw, truth, last, letter, calibration):
    from scripts.native_rgb_stage3_score import score_image
    from tdyssey.native_rgb_evaluation import palette_masks,_centroid
    m=score_image(raw,truth,last,calibration,letter=letter)
    m['motion_vector_error_image_px']=m.get('object_position_error_image_px')
    # Both vectors share the real anchor, so their endpoint difference is this
    # visible-centroid error; apparent rotations are separately marked ambiguous.
    m['visible_angle_error_deg']=m.get('legacy',{}).get('angle_error_deg')
    m['predicted_raw_corner_counts']=None; m['truth_raw_corner_counts']=None
    if letter in 'OC' and not m['hard_generation_failure']:
        import cv2
        def corners(image):
            mask=palette_masks(image,calibration)[0]
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            return [len(cv2.approxPolyDP(c,.75,True)) for c in contours if cv2.contourArea(c)>=2]
        m['predicted_raw_corner_counts']=corners(raw); m['truth_raw_corner_counts']=corners(truth)
    m['angle_scope']='Visible-mask angular alignment only; symmetry, occlusion, deformation can make it unavailable/ambiguous'
    return m

def evaluate(root, config, model, source, spec, phase, store):
    from tdyssey.native_rgb_r2c1r1_training import torch_native_sample
    from selection import centers
    calibration=read(REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    folder=root/'evaluation'/phase; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists(): raise RuntimeError('Evaluation is once-only; no silent repetition')
    sample=torch_native_sample(model,source,SAMPLER); mode=model.training; model.eval(); scored=[]; records=[]; started=time.monotonic()
    try:
        for bi,b in enumerate(spec['batches']):
            history=[]; actions=[]
            for r in b['rows']:
                d=store.arrays(r); a=int(r['anchor'])
                if a+16>=len(d['rgb']) or not d['control_valid'][a-3:a+16].all(): raise ValueError('Selected seen-val H16 not complete/clean')
                history.append(d['rgb'][a-3:a+1].astype(np.float32)/np.float32(255)); actions.append(d['passed_actions'][a-3:a+16])
            predictions=rollout_raw(sample,np.stack(history),np.stack(actions),seeds=b['seeds'])
            for j,r in enumerate(b['rows']):
                path=folder/f'raw-{bi:03d}-{j}.npz'; raw_npz(path,config,root,raw=predictions[j])
                records.append(dict(sample_id=r['sample_id'],path=str(path),sha256=sha(path),bytes=path.stat().st_size,
                    seeds=b['seeds'],raw_min=float(predictions[j].min()),raw_max=float(predictions[j].max())))
                d=store.arrays(r); a=int(r['anchor']); last=d['rgb'][a]
                for h in HORIZONS:
                    truth=d['rgb'][a+h]
                    for method,image in ((phase,predictions[j,h-1]),('copy_last',last)):
                        if phase=='after' and method=='copy_last': continue
                        area_centers=centers(r['letter'],d['states'][[a,a+h]])
                        delta=area_centers[1]-area_centers[0]
                        angle=math.degrees(float(d['states'][a+h,4]-d['states'][a,4]))
                        displacement=float(np.linalg.norm(delta)*96/512)
                        thresholds={1:(1,3),4:(2,5),8:(4,10),16:(8,20)}
                        scored.append(dict(sample_id=r['sample_id'],family_id=r['family_id'],letter=r['letter'],task=r['task'],
                            event=r.get('event'),horizon=h,method=method,truth_translation_image_px=displacement,
                            truth_signed_rotation_deg=angle,obvious_motion=displacement>=thresholds[h][0] or abs(angle)>=thresholds[h][1],
                            metrics=_metric(image,truth,last,r['letter'],calibration)))
            atomic_json(folder/'progress.json',dict(completed_batches=bi+1,total_batches=len(spec['batches']),model_calls=(bi+1)*16,sampler=SAMPLER))
        result=dict(complete=True,phase=phase,rows=scored,raw_files=records,wall_seconds=time.monotonic()-started,
            sampler_batch_calls=len(spec['batches'])*16,denoiser_batch_forwards=len(spec['batches'])*48,
            physical_steps=0,true_future_feedback=False,raw_feedback_unclipped=True,planned_windows=len(spec['rows']))
        atomic_json(folder/'metrics.json',result); return result
    finally: model.train(mode)

def _display(raw):
    """Display quantization only; saved float predictions remain untouched."""
    return np.rint(np.clip(np.asarray(raw).transpose(1,2,0),0,1)*255).astype(np.uint8)

def make_report(root, config, spec, before, after, training, store):
    import cv2
    import imageio_ffmpeg
    report=root/'report'; report.mkdir(exist_ok=True)
    by_before={x['sample_id']:x for x in before['raw_files']}; by_after={x['sample_id']:x for x in after['raw_files']}
    allrows=before['rows']+after['rows']; groups=[]
    for method in ('before','after','copy_last'):
        for h in HORIZONS:
            for letter in ('ALL',*'TLOCXB'):
                rs=[r for r in allrows if r['method']==method and r['horizon']==h and (letter=='ALL' or r['letter']==letter)]
                groups.append(dict(method=method,horizon=h,letter=letter,**_summary(rs)))
            for subset in ('obvious_motion','below_showcase_motion'):
                rs=[r for r in allrows if r['method']==method and r['horizon']==h and r['obvious_motion']==(subset=='obvious_motion')]
                groups.append(dict(method=method,horizon=h,letter=subset,**_summary(rs)))
    panels=[]; videos=[]
    for letter in 'TLOCXB':
        row=next((r for r in spec['rows'] if r['letter']==letter),None)
        if row is None: continue
        with np.load(by_before[row['sample_id']]['path'],allow_pickle=False) as z: bp=z['raw'].copy()
        with np.load(by_after[row['sample_id']]['path'],allow_pickle=False) as z: ap=z['raw'].copy()
        d=store.arrays(row); a=int(row['anchor']); last=d['rgb'][a]; frames=[]
        for k in range(16):
            frame=np.full((236,768,3),255,np.uint8)
            for col,(title,im) in enumerate((('TRUTH',d['rgb'][a+k+1].astype(np.float32)/255),('BEFORE',bp[k]),('AFTER 1 EPOCH',ap[k]),('COPY LAST',last.astype(np.float32)/255))):
                frame[36:228,col*192:(col+1)*192]=cv2.resize(_display(im),(192,192),interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame,title,(col*192+4,29),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
            cv2.putText(frame,f'{letter} | {row["sample_id"]} | t+{k+1}={(k+1)/10:.1f}s',(4,12),cv2.FONT_HERSHEY_SIMPLEX,.34,(20,20,20),1)
            frames.append(frame)
        sheet=np.concatenate([frames[h-1] for h in HORIZONS],axis=0); path=report/f'{letter}-H1-H4-H8-H16.png'
        ok,png=cv2.imencode('.png',cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
        if not ok: raise RuntimeError('Prediction sheet encode failed')
        storage_guard(root,config,len(png)); path.write_bytes(png.tobytes()); panels.append(dict(letter=letter,path=str(path),sample_id=row['sample_id']))
        video=report/f'{letter}-paired-10fps.mp4'; storage_guard(root,config,4*1024**2)
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','768x236','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        kw={}
        if os.name=='nt': kw['creationflags']=subprocess.CREATE_NO_WINDOW
        proc=subprocess.run(cmd,input=b''.join(x.tobytes() for x in frames),stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
        if proc.returncode: raise RuntimeError('Prediction video failed:'+proc.stderr.decode(errors='replace')[-500:])
        cap=cv2.VideoCapture(str(video)); n=0; fps=cap.get(cv2.CAP_PROP_FPS)
        while cap.read()[0]: n+=1
        cap.release()
        if n!=16 or abs(fps-10)>1e-3: raise RuntimeError('Prediction video decode/timing mismatch')
        videos.append(dict(letter=letter,path=str(video),fps=fps,frames=n,sha256=sha(video)))
    summary=dict(training=training,groups=groups,panels=panels,videos=videos,evaluation_windows=len(spec['rows']),
        independent_seen_val_families=len({r['family_id'] for r in spec['rows']}),
        no_clean_committed_h16_seen_val_support=[l for l in 'TLOCXB' if not any(r['letter']==l for r in spec['rows'])],
        scope='New original-physics data, six exposed development letters only; no unseen-letter or policy improvement claim',
        metric_limits='Visible gray/blue masks only; occlusion/viewport ambiguity retained. Hole/corner counts are diagnostics, not proof of rigid shapes. No privileged state enters predictions.',
        display='Only PNG/video display clips/quantizes; lossless saved rawfloat32 feeds recursion and scoring without repair.',
        storage=storage_guard(root,config))
    atomic_json(report/'summary.json',summary)
    atomic_json(root/'training/summary.json',dict(training=training,evaluation_groups=groups,report=str(root/'REPORT.md')))
    lines=['# One new-data RGB epoch and paired predictions','',f'Full epoch: {training["epoch_complete"]}; unique windows {training["unique_samples_exposed"]}; optimizer successes {training["successful_updates"]}; AMP skips {training["AMP_skips"]}.',
        f'Independent seen-validation families: {summary["independent_seen_val_families"]}. Horizons1/4/8/16 share anchors/actions/noise across before/after. Copy-last uses the real anchor.',
        '',summary['scope'],'',summary['metric_limits'],'',summary['display'],'', '| Method | H | Gray IoU | Gray centroid error(px) | Blue centroid error(px) | Missing blue / denominator |','|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['letter']!='ALL': continue
        def fmt(k):
            v=g['means'][k]['mean']; return 'unavailable' if v is None else f'{v:.4f}'
        lines.append(f'| {g["method"]} | {g["horizon"]} | {fmt("raw_mask_iou")} | {fmt("object_position_error_image_px")} | {fmt("pusher_position_error_image_px")} | {g["pusher_missing"]}/{g["denominator"]} |')
    for p in panels: lines+=['',f'[{p["letter"]} comparison]({p["path"]})']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    chunks=['<!doctype html><meta charset="utf-8"><title>One new-data RGB epoch</title><style>body{font:16px sans-serif;max-width:1100px;margin:24px auto}img,video{max-width:100%}pre{white-space:pre-wrap}</style>',f'<h1>One new-data RGB epoch</h1><pre>{html.escape(chr(10).join(lines[:8]))}</pre>']
    for p in panels:
        v=next(x for x in videos if x['letter']==p['letter']); chunks+=[f'<h2>{p["letter"]}</h2><img src="{Path(p["path"]).name}"><video controls src="{Path(v["path"]).name}"></video>']
    chunks.append('<p>Full grouped metrics: <a href="summary.json">summary.json</a></p>'); (report/'index.html').write_text('\n'.join(chunks),encoding='utf-8')
    (root/'evaluation/index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../report/index.html"><a href="../report/index.html">Paired prediction report</a>',encoding='utf-8')
    return dict(report=str(root/'REPORT.md'),index=str(report/'index.html'),summary=str(report/'summary.json'),panels=panels,videos=videos)

def train_and_evaluate(root:Path, config:dict)->dict:
    """Called once by the detached stage worker after selection is frozen."""
    import torch
    from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
    from tdyssey.raw_world_model_training import OPTIMIZER,make_generator
    from tdyssey.native_rgb_training import _cpu_tree,capture_rng,restore_rng
    from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs,conditioning_residual_loss,POLICY_SHA
    from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
    root=Path(root).resolve(); run=root/'training'; run.mkdir(exist_ok=True)
    claim=run/'claim.json'
    if claim.exists(): raise RuntimeError('Training stage already claimed; no automatic replay or extra epoch')
    atomic_json(claim,dict(pid=os.getpid(),started_unix=time.time(),epoch_limit=1,automatic_retry=False))
    train=read(root/'selection/train-index.json'); val=read(root/'selection/seen-val-index.json')
    selection=read(root/'selection/summary.json')
    for filename,expected in selection['index_sha256'].items():
        if sha(root/'selection'/filename)!=expected: raise ValueError('Frozen selected index hash changed')
    if isinstance(train,dict): train=train['rows']
    if isinstance(val,dict): val=val['rows']
    validate_indices(train,val); batches=epoch_batches(train,int(config.get('order_seed',20261031)))
    spec=eval_spec(val,int(config.get('evaluation_max_windows',48)),int(config.get('evaluation_seed',20261031)))
    atomic_json(root/'evaluation-spec.json',spec)
    preflight=checkpoint_preflight(config.get('starting_checkpoint',PARENT),config.get('starting_checkpoint_sha256',PARENT_SHA))
    atomic_json(run/'parent-checkpoint.json',preflight)
    if not torch.cuda.is_available(): raise RuntimeError('Authorized local3060Ti CUDA unavailable')
    torch.set_num_threads(2); torch.cuda.reset_peak_memory_stats(); storage_guard(root,config)
    source=load_world_model_source(config.get('source_root',str(REPO.parent/'Tdyssey/third_party/gpc_code')))
    inner=deepcopy(world_model_spec()['model']); outer={k:inner.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
    with torch.random.fork_rng(devices=[]): model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**outer)).float()
    model.to('cuda').requires_grad_(True); model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.)); model.train()
    optimizer=torch.optim.AdamW(model.parameters(),**dict(OPTIMIZER,lr=1e-5)); scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128.)
    parent=torch.load(preflight['path'],map_location='cpu',weights_only=False); model.load_state_dict(parent['model'],strict=True)
    optimizer.load_state_dict(parent['optimizer']); scaler.load_state_dict(parent['scaler'])
    generator=make_generator(5478,next(model.parameters()).device); restore_rng(parent['global_rng'],generator)
    rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=next(model.parameters()).device); rngs.load_state_dict(parent['conditioning_rngs'])
    progress=dict(epoch_complete=False,unique_samples_exposed=0,successful_updates=0,AMP_skips=0,batches_consumed=0,
        planned_samples=len(train),planned_batches=len(batches),lineage_updates=7499,complete_optimizer_boundary=True,pending_attempt=None,
        model_forward_attempts=0,model_forward_completed=0,backward_microbatches=0,clean_microbatches=0,weak_microbatches=0)
    identity=dict(kind='accelerated-new-original-data-one-epoch.v1',parent_checkpoint=preflight,
        train_index_sha256=sha(root/'selection/train-index.json'),seen_val_index_sha256=sha(root/'selection/seen-val-index.json'),
        order_sha256=digest([[train[i]['sample_id'] for i in b] for b in batches]),conditioning_policy_sha256=POLICY_SHA,
        microbatch=4,effective_batch=16,final_short_batch='All remaining distinct samples once; sample-weighted micro means, no padding/replay',
        learning_rate=1e-5,AMP=True,sampler=SAMPLER,action_normalization='Actual R3A float32 targets/255.5-1; no clipping; incompatible>511windows excluded',
        transition='New dynamics/data stage initialized from full7499state; NOT same-distribution exact continuation',
        adam_rng_source='Latest full R3B7499 Adam/scaler/global+conditioning RNG restored; old order retained as provenance, new unique order frozen')
    atomic_json(run/'identity.json',identity); store=RawStore(); application=[0]
    atomic_json(run/'source-recipe.json',identity)
    hook=optimizer.register_step_post_hook(lambda *args:application.__setitem__(0,application[0]+1)); began=time.monotonic(); clean=True
    def save(reason):
        storage_guard(root,config,preflight['bytes']+8*1024**2)
        if not clean or not progress['complete_optimizer_boundary']: raise RuntimeError('Refusing checkpoint with uncertain optimizer application')
        payload=dict(kind='accelerated_native_rgb_complete.v1',model=_cpu_tree(model.state_dict()),optimizer=_cpu_tree(optimizer.state_dict()),
            scaler=_cpu_tree(scaler.state_dict()),global_rng=capture_rng(generator),conditioning_rngs=rngs.state_dict(),
            order=dict(batches=batches,cursor=progress['batches_consumed'],sample_ids=[r['sample_id'] for r in train]),
            progress=deepcopy(progress),identity=identity,legacy_scheduler_provenance=deepcopy(parent['legacy_scheduler_provenance']),
            migration_provenance=dict(parent_identity=parent['identity'],parent_progress=parent['progress'],parent_order=parent['order'],parent_global_rng=parent['global_rng']),
            reason=reason)
        path=run/'last.pt'; tmp=path.with_name(path.name+'.tmp')
        try:
            torch.save(payload,tmp)
            with tmp.open('rb+') as stream: os.fsync(stream.fileno())
            os.replace(tmp,path)
        finally: tmp.unlink(missing_ok=True)
        return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,consumed=progress['unique_samples_exposed'])
    before=None; checkpoint=None
    try:
        # Forked sampler RNG and explicit restoration preserve every training stream.
        old_rng=capture_rng(generator); old_cond=rngs.state_dict()
        try: before=evaluate(root,config,model,source,spec,'before',store)
        finally: restore_rng(old_rng,generator); rngs.load_state_dict(old_cond)
        atomic_json(run/'progress.json',progress)
        for bi,indices in enumerate(batches):
            if bi%25==0: storage_guard(root,config)
            arrays=store.batch([train[i] for i in indices]); n=len(indices); optimizer.zero_grad(set_to_none=True)
            clean=False; progress['complete_optimizer_boundary']=False
            progress['pending_attempt']=dict(batch_index=bi,sample_ids=[train[i]['sample_id'] for i in indices],optimizer_application='not_started')
            # Draws are consumed durably before forward; an interruption is never silently replayed.
            progress['batches_consumed']=bi+1; progress['unique_samples_exposed']+=n; atomic_json(run/'progress.json',progress)
            loss_total=0.; micros=[]
            for start in range(0,n,4):
                size=min(4,n-start); batch={k:torch.as_tensor(v[start:start+size],device='cuda') for k,v in arrays.items()}
                progress['model_forward_attempts']+=1
                with torch.autocast('cuda',dtype=torch.float16,enabled=True): loss,details=conditioning_residual_loss(model,batch,rngs=rngs)
                progress['model_forward_completed']+=1; scaler.scale(loss*(size/n)).backward(); loss_total+=float(loss.detach())*(size/n)
                progress['backward_microbatches']+=1; progress[details['history_branch']+'_microbatches']+=1; micros.append(details)
            scaler.unscale_(optimizer); params=list(model.parameters())
            if any(p.grad is None for p in params): raise FloatingPointError('Missing native full-network gradient')
            if any(not bool(torch.isfinite(p).all()) for p in params): raise FloatingPointError('Nonfinite model weights')
            finite=all(bool(torch.isfinite(p.grad).all()) for p in params)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)) if finite else None
            progress['pending_attempt']['optimizer_application']='started'; atomic_json(run/'progress.json',progress)
            outcome=scaled_optimizer_step(scaler,optimizer,application_count=lambda:application[0])
            if outcome['applied']:
                progress['successful_updates']+=1; progress['lineage_updates']+=1; consecutive=0
            else:
                progress['AMP_skips']+=1; consecutive=progress.get('consecutive_AMP_overflows',0)+1
            progress['consecutive_AMP_overflows']=consecutive
            optimizer.zero_grad(set_to_none=True); clean=True; progress['complete_optimizer_boundary']=True; progress['pending_attempt']=None
            if any(not bool(torch.isfinite(p).all()) for p in params): raise FloatingPointError('Nonfinite parameter after optimizer')
            progress['wall_seconds']=time.monotonic()-began; progress['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(); progress['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
            atomic_json(run/'progress.json',progress)
            with (run/'losses.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(dict(batch=bi,loss=loss_total,gradient_norm=norm,draws=n,**outcome,branches=[m['history_branch'] for m in micros]))+'\n')
            if progress['peak_cuda_reserved_bytes']>int(config.get('max_cuda_bytes',6*1024**3)): raise RuntimeError('Existing6GiB CUDA envelope exceeded')
            if consecutive>=16: raise FloatingPointError('16consecutive actualAMPoverflows; noautomaticretry')
            if outcome['applied'] and progress['successful_updates']%100==0: checkpoint=save('existing100-success-boundary')
        if progress['unique_samples_exposed']!=len(train) or progress['batches_consumed']!=len(batches): raise RuntimeError('Full unique epoch did not complete')
        progress['epoch_complete']=True; checkpoint=save('one_new_data_epoch_complete')
        atomic_json(run/'result.json',dict(**progress,checkpoint=checkpoint,unique_families=len({r['family_id'] for r in train}),physical_steps=0,loader=dict(cache_bytes=store.bytes,loads=store.loads)))
        after=evaluate(root,config,model,source,spec,'after',store)
        report=make_report(root,config,spec,before,after,progress,store)
        return dict(complete=True,training=dict(**progress,checkpoint=checkpoint),before=str(root/'evaluation/before/metrics.json'),after=str(root/'evaluation/after/metrics.json'),**report)
    except BaseException as error:
        if clean:
            try: checkpoint=save('failure_complete_boundary')
            except BaseException as saving: checkpoint=dict(saved=False,error=repr(saving))
        atomic_json(run/'failure.json',dict(error_type=type(error).__name__,error=str(error),progress=progress,checkpoint=checkpoint,
            optimizer_state_uncertain=not clean,automatic_restart=False,wall_seconds=time.monotonic()-began))
        raise
    finally:
        hook.remove(); optimizer.zero_grad(set_to_none=True); store.cache.clear()
        del parent,model,optimizer,scaler,rngs,generator; torch.cuda.empty_cache()
