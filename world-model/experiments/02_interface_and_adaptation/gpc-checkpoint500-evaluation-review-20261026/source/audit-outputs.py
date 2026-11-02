"""Post-exit stdlib evidence audit only; no model, array runtime or new score."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import time

ROOT=Path(__file__).absolute().parents[2]
REVIEW=Path(__file__).absolute().parent
OUT=ROOT/'artifacts/gpc-checkpoint500-evaluation-20261026'
started=time.monotonic();inventory={}
def ordinary(p):
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or getattr(s,'st_file_attributes',0)&0x400:
        raise ValueError('Nonordinary evidence: '+str(p))
    return s
def record(p,raw):
    before=ordinary(p)
    if before.st_size!=len(raw):raise ValueError('Evidence byte length changed')
    inventory[p.relative_to(OUT).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):
    raw=p.read_bytes();record(p,raw);return json.loads(raw)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
cfg=json.loads((ROOT/'configs/gpc_checkpoint500_evaluation.json').read_bytes())
bm=json.loads((ROOT/cfg['benchmark_manifest']).read_bytes())
reg=json.loads((ROOT/bm['regression']['path']/'report.json').read_bytes())
case_keys={(r['anchor'],r['candidate']) for r in reg['branches'] if r['status']=='complete'}
if len(case_keys)!=63:raise ValueError('Regression branch count differs')
independent={(r['anchor'],r['candidate']) for r in bm['cases']}
if len(independent)!=48 or case_keys&independent:raise ValueError('Independent grid differs')
case_keys |= independent
modes=[(m,h) for m,hs in cfg['evaluation']['modes'].items() for h in hs]
expected={(a,c,s,m,h) for a,c in case_keys for s in cfg['evaluation']['seeds'] for m,h in modes}
if len(expected)!=1998:raise ValueError('Fixed paired denominator differs')
rows={};raw_paths=set();counts=Counter();extrema={role:[math.inf,-math.inf] for role in ('frozen','adapted')}
for p in sorted((OUT/'measurements').glob('*.json')):
    r=read(p);role=r['role'];key=(r['anchor'],r['candidate'],r['seed'],r['mode'],r['horizon'])
    if role not in ('frozen','adapted') or key not in expected or (role,key) in rows or r['status']!='completed':
        raise ValueError('Duplicate/unplanned/incomplete measurement')
    if r['stratum']!=('independent' if key[:2] in independent else 'regression'):
        raise ValueError('Physical stratum changed')
    relative=Path(r['raw_path'])
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=2 or relative.parts[0]!='predictions' or relative.stem!=p.stem:
        raise ValueError('Raw path differs from immutable measurement')
    raw_path=OUT/relative
    if raw_path in raw_paths:raise ValueError('Raw image was reused')
    raw_paths.add(raw_path);data=raw_path.read_bytes();record(raw_path,data)
    if data[:6]!=b'\x93NUMPY' or data[6:8] not in (b'\x01\x00',b'\x02\x00'):
        raise ValueError('Unsupported NumPy file header')
    width=2 if data[6]==1 else 4;length=int.from_bytes(data[8:8+width],'little');offset=8+width+length
    header=ast.literal_eval(data[8+width:offset].decode('latin1').strip())
    payload=data[offset:]
    if header!={'descr':'<f4','fortran_order':False,'shape':(3,96,96)} or len(payload)!=110592:
        raise ValueError('Float32 CHW prediction bytes differ')
    if hashlib.sha256(payload).hexdigest()!=r['raw_sha256']:raise ValueError('Saved raw array hash differs')
    for field in ('action_sha256','input_sha256','rng_entry','rng_exit','raw_sha256'):
        value=r[field]
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Missing SHA field')
    if not all(type(r[k]) in (int,float) and math.isfinite(r[k]) for k in ('raw_min','raw_max','rgb_mae')) or r['raw_min']>r['raw_max']:
        raise ValueError('Raw finite summary differs')
    extrema[role][0]=min(extrema[role][0],r['raw_min']);extrema[role][1]=max(extrema[role][1],r['raw_max'])
    rows[role,key]=r;counts[role]+=1;counts[(role,r['shape'],r['stratum'],r['mode'],r['horizon'])]+=1
if len(rows)!=3996 or counts['frozen']!=1998 or counts['adapted']!=1998:
    raise ValueError('Incomplete actual forecast denominator')
checks=Counter();failures=[]
for key in sorted(expected):
    a,b=rows['frozen',key],rows['adapted',key]
    for field in ('rng_entry','rng_exit','action_sha256','truth','latest','contact_by_h4','shape','stratum'):
        if canonical(a[field])!=canonical(b[field]):failures.append(dict(key=key,field=field))
        else:checks[field]+=1
    fixed=key[3]=='TF' or key[4]==1
    same=a['input_sha256']==b['input_sha256']
    checks['fixed_input_expected' if fixed else 'autoregressive_input_expected']+=1
    checks[('fixed_input_equal' if fixed else 'autoregressive_input_equal') if same else ('fixed_input_different' if fixed else 'autoregressive_input_different')]+=1
    if fixed and not same:failures.append(dict(key=key,field='fixed_input_sha256'))
for role in ('frozen','adapted'):
    for anchor,candidate in sorted(case_keys):
        for seed in cfg['evaluation']['seeds']:
            for horizon in (2,4):
                ar=rows[role,(anchor,candidate,seed,'AR',horizon)];tf=rows[role,(anchor,candidate,seed,'TF',horizon)]
                for field in ('rng_entry','rng_exit','action_sha256'):
                    if ar[field]!=tf[field]:failures.append(dict(role=role,anchor=anchor,candidate=candidate,seed=seed,horizon=horizon,field='AR_TF_'+field))
                    else:checks['AR_TF_'+field]+=1
if failures:raise ValueError('Matched protocol differs: '+str(failures[:3]))
completion=read(OUT/'completion.json');loss=completion['loss_rows'];loss_summary={}
if [r['noise'] for r in loss['frozen']]!=[r['noise'] for r in loss['adapted']] or [r['index'] for r in loss['frozen']]!=[r['index'] for r in loss['adapted']]:
    raise ValueError('Loss sample/noise pairing changed')
for role,items in loss.items():
    if len(items)!=16 or not all(math.isfinite(r['loss']) for r in items):raise ValueError('Loss denominator/finite values differ')
    loss_summary[role]=dict(count=16,mean=sum(r['loss'] for r in items)/16,
        T=dict(count=8,mean=sum(r['loss'] for r in items[::2])/8),
        L=dict(count=8,mean=sum(r['loss'] for r in items[1::2])/8))
predictions=set((OUT/'predictions').glob('*.npy'))
if predictions!=raw_paths:raise ValueError('Unreferenced or missing prediction files')
for p in sorted(OUT.rglob('*')):
    if not p.is_file():continue
    relative=p.relative_to(OUT).as_posix()
    if relative not in inventory:record(p,p.read_bytes())
directory_bytes=sum(v['bytes'] for v in inventory.values())
value=dict(kind='gpc_eval_output_integrity.v1',forecast_rows=3996,roles=dict(frozen=1998,adapted=1998),
    physical_branches=111,regression_branches=63,independent_branches=48,noise_seeds=cfg['evaluation']['seeds'],
    array_header_and_raw_hash_verified=3996,raw_payload_bytes=3996*110592,
    array_file_bytes=sum(v['bytes'] for k,v in inventory.items() if k.startswith('predictions/')),
    pair_checks=dict(checks),pair_failures=failures,loss_pair_count=16,loss_summary=loss_summary,
    loss_adapted_frozen_ratio=loss_summary['adapted']['mean']/loss_summary['frozen']['mean'],
    raw_extrema=extrema,output_files=len(inventory),output_bytes=directory_bytes,seconds=time.monotonic()-started,
    runtime_before_export_finite_check_bound_by_raw_hash=True,extra_model_or_measurement_operations=0)
with (REVIEW/'output-integrity.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
with (REVIEW/'output-inventory.json').open('x',encoding='utf-8') as f:json.dump(inventory,f,indent=2)
print(json.dumps(value))
