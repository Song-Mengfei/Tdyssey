"""Read-only retrospective. No torch, model, simulator, network or package work.
Writes only beside this script; never imports project execution modules.
"""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, os
import numpy as np
from scipy import ndimage

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
ART=REPO/'artifacts'
SOURCES={}
def mark(p):
    p=Path(p).resolve(); st=p.stat()
    SOURCES[str(p)]=dict(bytes=st.st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mtime_ns=st.st_mtime_ns)
    return p
def read(p): return json.loads(mark(p).read_text(encoding='utf-8-sig'))
def save(n,j): (OUT/n).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def distribution(rows):
    t=np.array([x['translation_imagepx'] for x in rows]);r=np.array([abs(x['signed_rotation_deg']) for x in rows]);n=len(rows)
    q=lambda a:dict(zip(['P0','P10','P50','P90','P95','P99','max'],map(float,np.quantile(a,[0,.1,.5,.9,.95,.99,1]))))
    return dict(n=n,families=len({x['family_id'] for x in rows}),translation_image96_px=q(t),abs_rotation_deg=q(r),
        below_1px_and_3deg=int(((t<1)&(r<3)).sum()),strong=int(((t>=1)|(r>=3)).sum()),
        fully_static_RGB=sum(bool(x.get('bitstatic_future')) for x in rows),contact_and_move=sum(x.get('event')=='contact_and_move' for x in rows),
        events=dict(Counter(x.get('event') for x in rows)),CW_ge3=sum(x['signed_rotation_deg']>=3 for x in rows),CCW_le_minus3=sum(x['signed_rotation_deg']<=-3 for x in rows))

idx=read(ART/'accelerated-rgb-run-20261031/selection/train-index.json')
pool=read(ART/'generated-history-ab-20261101/pool-index.json')
lookup={x['sample_id']:x for x in idx}
ids=[v for seg in pool['segments'] for v in seg['supervised_H1_ids']]
subset=[lookup[k] for k in ids]
formal=read(ART/'generated-history-ab-20261101/formal-spec.json')
datasets=dict(AR1_full=distribution(idx),ABCD_fourstep_subset=distribution(subset),ABCD_by_letter={l:distribution([x for x in subset if x['letter']==l]) for l in 'TLOCXB'},
    source_unique_ids=len(lookup),fourstep_unique_ids=len(set(ids)),excluded=len(set(lookup)-set(ids)),segments=len(pool['segments']),
    formal48_families=len({x['family_id'] for x in formal['rows']}),
    train_vs_formal_family_overlap=sorted({x['family_id'] for x in subset}&{x['family_id'] for x in formal['rows']}))
hm=read(ART/'high-motion-fit-20261101/selection/index.json');order=read(ART/'high-motion-fit-20261101/selection/order.json')
targets=hm['targets'];draws=[i for b in order['batches'] for i in b];cnt=Counter(draws)
datasets['HMF_frozen_plan']=dict(train48=len(hm['train48']),seenval24=len(hm['seen_val24']),failures3=len(hm['failures3']),
    distinct_families=len({x['family_id']for k in ['train48','seen_val24','failures3']for x in hm[k]}),
    train_val_overlap=sorted({x['family_id']for x in hm['train48']}&{x['family_id']for x in hm['seen_val24']}),
    target_count=len(targets),strong=sum(x['strong']for x in targets),planned_exposures=len(draws),
    planned_strong_exposures=sum(targets[i]['strong'] for i in draws),
    planned_target_exposures_min_max=[min(cnt.values()),max(cnt.values())],
    planned_strong_exposures_min_max=[min(cnt[i] for i,x in enumerate(targets)if x['strong']),max(cnt[i]for i,x in enumerate(targets)if x['strong'])],
    planned_transition_exposures_min_max=[min(cnt[i] for i,x in enumerate(targets)if not x['strong']),max(cnt[i]for i,x in enumerate(targets)if not x['strong'])],
    note='Frozen schedule statistics only. Actual progress is separately frozen in CURRENT_SNAPSHOT.json; this script does not reread the running state.')
save('INDEPENDENT_DATA_AUDIT.json',datasets)

# Reaggregate actual per-case saved scores; do not resample or refit model states.
metrics={}
for label,root in [('A','generated-history-ab-20261101'),('B','generated-history-ab-20261101'),('C','region-weighted-cd-20261101'),('D','region-weighted-cd-20261101')]:
    j=read(ART/root/'evaluation'/label/'formal/metrics.json');metrics[label]=j
aggregates={}
for label,j in metrics.items():
    aggregates[label]={}
    for h in [1,4,8,16]:
        rows=[x for x in j['rows']if x['horizon']==h]
        vals=lambda k:[x['metrics'][k]for x in rows if x['metrics'].get(k)is not None]
        aggregates[label][str(h)]=dict(n=len(rows),IoU=float(np.mean(vals('raw_mask_iou'))),position_px=float(np.mean(vals('object_position_error_image_px'))),
            blue_position_px=float(np.mean(vals('pusher_position_error_image_px'))),position_P90=float(np.quantile(vals('object_position_error_image_px'),.9)),
            rows_motion_equals_position=sum(x['metrics'].get('motion_vector_error_image_px')==x['metrics'].get('object_position_error_image_px')for x in rows))
save('INDEPENDENT_METRIC_AUDIT.json',aggregates)

# Verify six selected C/D prediction NPZs against their source records, and
# recompute topology using the documented frozen palette; do not repair RGB.
cal=read(ART/'native-rgb-20261028/frozen-renderer-calibration.json')
def masks(image):
    pixels=np.asarray(image,dtype=np.float64).transpose(1,2,0);v=1-np.array(cal['object_rgb']);a=((1-pixels)*v).sum(-1)/(v@v)
    residual=np.linalg.norm(pixels-(1-a[...,None]*v),axis=-1)
    grey=(a>=cal['palette']['minimum_alpha'])&(a<=cal['palette']['maximum_alpha'])&(residual<=cal['palette']['colour_residual'])
    red,green,blue=np.asarray(image);b=(blue>.6)&(red<.45)&(green<.65)&(blue-np.maximum(red,green)>.25)
    return grey,b
def holes(mask):
    lab,n=ndimage.label(~np.pad(mask,1),structure=ndimage.generate_binary_structure(2,1));sizes=np.bincount(lab.ravel());ex=lab[0,0]
    return int(sum(sizes[i]>=2 for i in range(1,n+1)if i!=ex))
rawchecks=[]
for label in ['C','D']:
    for letter,fam in [('O','AC-O-01-08'),('C','AC-C-01-24'),('B','AC-B-00-08')]:
        rec=next(x for x in metrics[label]['raw_files']if x['family_id']==fam)
        path=mark(rec['path']);raw=np.load(path,allow_pickle=False)['raw'];p=mark(rec['raw_path'])
        with np.load(p,allow_pickle=False)as z:gt=z['rgb'][rec['anchor']+1:rec['anchor']+17].astype(np.float32)/255
        hm=[];gt_hm=[];ious=[];blue_missing=[]
        for pred,true in zip(raw,gt):
            gm,bm=masks(pred);tm,tb=masks(true);hm.append(holes(gm));gt_hm.append(holes(tm));ious.append(float((gm&tm).sum()/(gm|tm).sum()));blue_missing.append(not bool(bm.any()))
        rawchecks.append(dict(label=label,letter=letter,family=fam,prediction=str(path),source=str(p),raw_shape=list(raw.shape),dtype=str(raw.dtype),
            prediction_file_sha_match=SOURCES[str(path)]['sha256']==rec['sha256'],source_file_sha_match=SOURCES[str(p)]['sha256']==rec['raw_sha256'],
            holes_by_H1_to16=hm,truth_holes_by_H1_to16=gt_hm,IoU_H1_H4_H8_H16=[ious[i-1]for i in [1,4,8,16]],blue_missing_count=sum(blue_missing),
            limitation='Visible topology under frozen palette, not complete shape/rigidity/physical-contact certification.'))
save('INDEPENDENT_RAW_AUDIT.json',rawchecks)

# Filesystem accounting is metadata only. It is not an atomic live storage audit.
disk=[]
for d in sorted(ART.iterdir()):
    if not d.is_dir()or d==OUT:continue
    counts=Counter();sizes=Counter();total=0;n=0
    for root,dirs,files in os.walk(d):
        for f in files:
            p=Path(root)/f
            try:sz=p.stat().st_size
            except FileNotFoundError:continue
            n+=1;total+=sz;counts[p.suffix.lower()]+=1;sizes[p.suffix.lower()]+=sz
    disk.append(dict(root=str(d),files=n,bytes=total,NPZ_files=counts['.npz'],PT_files=counts['.pt'],PT_bytes=sizes['.pt']))
save('DISK_INVENTORY.json',dict(read_at_singapore=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    note='Per-directory metadata snapshot, changing files may differ; excludes environments, teammates and other shared roots. Do not equate with global reservation guard.',directories=disk))
deletions=read(ART/'original-physics-contact-pilot-20261030/cleanup/checkpoint-deletion-receipt.json')
save('DELETED_CHECKPOINTS.json',[dict(path=x['path'],bytes=x['bytes'],historical_sha256=x['sha256'],exists_now=Path(x['path']).exists())for x in deletions['deleted']])
for rel in ['AGENTS.md','docs/PROJECT.md','docs/NATIVE_RGB_WORLD_MODEL.md','tdyssey/gpc_world_model.py','tdyssey/native_rgb_r3a_conditioning.py','tdyssey/native_rgb_evaluation.py','tdyssey/direct_target_collection.py','artifacts/accelerated-rgb-run-20261031/selection.py','artifacts/accelerated-rgb-run-20261031/train_eval.py','artifacts/accelerated-rgb-run-20261031/training/source-recipe.json','artifacts/generated-history-ab-20261101/training_core.py','artifacts/region-weighted-cd-20261101/training_core.py','artifacts/region-weighted-cd-20261101/masks.py','artifacts/high-motion-fit-20261101/fit-standard.json']:
    mark(REPO/rel)
for root in ['native-rgb-20261028','accelerated-rgb-run-20261031','oc-continuation-20261031','generated-history-ab-20261101','region-weighted-cd-20261101','saved-evaluation-recheck-20261101']:
    for name in ['REPORT.md','FINAL_REVIEW.md','REPORT.txt']:
        p=ART/root/name
        if p.exists():mark(p)
for p in (ART/'native-rgb-supervised-20261029').glob('*/REPORT.md'):mark(p)
for rel in ['third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py','third_party/gpc_code/gpc_rank_evaluation/diffusion/inner_model.py','third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py','third_party/gpc_code/world_model_train_phase_two/utils.py']:
    mark(REPO.parent/'Tdyssey'/rel)
for n in ['audit.json','physics-contact-audit/summary.json','physics-contact-audit/final-candidate-summary.json','physics-contact-audit/audit-receipt.json']:
    mark(REPO.parent/'tmp/pushing-data-review-20261030'/n)
save('EVIDENCE_MANIFEST.json',dict(method='Small text/JSON and six selected raw predictions plus three reused source NPZ hashes. No checkpoint loading or full-dataset rehash.',files=SOURCES))
print(json.dumps(dict(data=datasets,metrics=aggregates,raw=[{k:v for k,v in x.items()if k not in ['prediction','source']}for x in rawchecks],evidence_files=len(SOURCES)),ensure_ascii=False,indent=2))
