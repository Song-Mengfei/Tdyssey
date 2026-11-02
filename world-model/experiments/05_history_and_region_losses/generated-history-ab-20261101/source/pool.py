"""Saved-only four-target pool and byte-equivalent bounded uint8 disk cache.

No physics, Torch or model imports. Never modifies source data or selection.
"""
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import os
import random
import time
import numpy as np

REPO=Path(__file__).resolve().parents[2]
AR1=REPO/'artifacts/accelerated-rgb-run-20261031'
PREVIEW=REPO/'artifacts/generated-history-four-step-review-20261101/POOL_PREVIEW.json'
PREVIEW_SHA='311a8982c5e2a4eed5e5b8fc2733c70ca9b5eabc91383064f57c3c7c16790732'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def fingerprint(p):
    s=Path(p).stat(); return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def normalized(a): return a.astype(np.float32)/np.float32(255.5)-np.float32(1)

def group_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['split']!='train' or r['horizon']!=1 or r['letter'] not in 'TLOCXB': raise ValueError('Invalid training source')
        groups[(r['family_id'],r['raw_path'])].append(r)
    segments=[]; leftovers=[]
    for (family,path),rs in sorted(groups.items()):
        rs=sorted(rs,key=lambda r:r['anchor']); runs=[]; current=[]
        for r in rs:
            if current and r['anchor']!=current[-1]['anchor']+1: runs.append(current);current=[]
            current.append(r)
        if current:runs.append(current)
        for run in runs:
            end=len(run)//4*4
            for i in range(0,end,4):
                block=run[i:i+4]; a=block[0]['anchor']
                if len({r['raw_sha256'] for r in block})!=1: raise ValueError('Source hash mismatch inside block')
                segments.append(dict(segment_id=f'{family}:a{a}:four-step-review',family_id=family,
                    letter=block[0]['letter'],task=block[0]['task'],event=block[0].get('event','unknown'),
                    raw_path=path,raw_sha256=block[0]['raw_sha256'],anchor=a,split='train',
                    supervised_H1_ids=[x['sample_id'] for x in block],
                    contact_motion_targets=sum(x['diagnostic_contact_and_motion'] for x in block)))
            leftovers.extend(r['sample_id'] for r in run[end:])
    ids=[x for s in segments for x in s['supervised_H1_ids']]
    if len(ids)!=len(set(ids)) or len(ids)+len(leftovers)!=len(rows): raise ValueError('Repeated/dropped source targets')
    return segments,leftovers

def validate_arrays(d,m,segments):
    rgb=d['rgb']; actions=d['passed_actions']; valid=d['control_valid']
    if rgb.dtype!=np.uint8 or rgb.shape[1:]!=(3,96,96) or len(rgb)!=len(actions)+1:raise ValueError('Raw 100ms RGB/control timeline differs')
    if not np.array_equal(actions,d['requested_actions']):raise ValueError('Requested/passed action mismatch')
    if valid.dtype!=np.bool_ or len(valid)!=len(actions):raise ValueError('Raw validity timeline differs')
    for s in segments:
        a=int(s['anchor']); blocks=m.get('segments',m.get('blocks',[]))
        if m['split']!='train' or m['family_id']!=s['family_id'] or a<3 or a+4>int(m['clean_control_count']):raise ValueError('Clean prefix/split mismatch')
        if len(rgb[a-3:a+5])!=8 or len(actions[a-3:a+4])!=7 or not valid[a-3:a+4].all():raise ValueError('8 image/7 action timing incomplete/invalid')
        act=actions[a-3:a+4]
        if not np.isfinite(act).all() or act.min()<0 or act.max()>511:raise ValueError('Unchanged model action contract violated')
        block=next((b for b in blocks if int(b.get('start_control',-1))<=a and a+4<=int(b.get('end_control',-1))),None)
        if block is None or int(block.get('committed_before_control',a))>a:raise ValueError('Four-target future crosses replanning boundary')
        start=int(block['start_control']); committed=np.asarray(block['committed_actions'])
        if not np.array_equal(committed[a-start:a+4-start],actions[a:a+4]):raise ValueError('Committed future differs from raw passed actions')
        s['committed_future']=True

def build(root,config):
    from common import atomic_json,check_storage,replace_atomic,utc
    root=Path(root); began=time.perf_counter()
    if (root/'pool-index.json').exists():raise FileExistsError('Pool already frozen; no regeneration')
    if sha(PREVIEW)!=PREVIEW_SHA:raise ValueError('Approved preview identity changed')
    source=AR1/'selection/train-index.json'; rows=read(source); val=read(AR1/'selection/seen-val-index.json')
    if len(rows)!=30935 or len({r['sample_id'] for r in rows})!=30935 or {r['family_id'] for r in rows}&{r['family_id'] for r in val}:raise ValueError('Original selected pool/split changed')
    selection=read(AR1/'selection/summary.json')
    if sha(source)!=selection['index_sha256']['train-index.json']:raise ValueError('Original frozen train-index differs')
    segments,leftovers=group_rows(rows); preview=read(PREVIEW)
    if [s['segment_id'] for s in segments]!=[s['segment_id'] for s in preview['segments']] or leftovers!=preview['uncovered_H1_ids']:raise ValueError('Recomputed grouping differs from approved preview')
    if len(segments)!=7568 or len({s['family_id'] for s in segments})!=1313 or len(leftovers)!=663:raise ValueError('Approved pool counts differ')
    n=len(segments); shape=(n,8,3,96,96); cache_bytes=int(np.prod(shape))+n*7*2*4
    check_storage(root,config,reserve_bytes=cache_bytes+8*1024**2,full=True)
    cache=root/'cache'; cache.mkdir(exist_ok=True)
    fp=cache/'frames.u8'; ap=cache/'actions.f32'; ft=fp.with_suffix('.u8.tmp'); at=ap.with_suffix('.f32.tmp')
    if any(p.exists() for p in (fp,ap,ft,at)):raise FileExistsError('Cache exists; do not silently overwrite')
    frames=np.memmap(ft,dtype=np.uint8,mode='w+',shape=shape); actions=np.memmap(at,dtype=np.float32,mode='w+',shape=(n,7,2))
    grouped=defaultdict(list)
    for i,s in enumerate(segments):grouped[s['raw_path']].append((i,s))
    files=[]; source_bytes=0
    for path,items in sorted(grouped.items()):
        p=Path(path); mp=p.with_suffix('.json'); expected={s['raw_sha256'] for _,s in items}
        before=fingerprint(p); mbefore=fingerprint(mp); actual=sha(p); mh=sha(mp)
        if expected!={actual} or preview['metadata_sha256'][str(mp)]!=mh:raise ValueError('Frozen raw/metadata changed')
        with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in ('rgb','passed_actions','requested_actions','control_valid')}
        m=read(mp); validate_arrays(d,m,[s for _,s in items])
        for i,s in items:
            a=s['anchor']; frames[i]=d['rgb'][a-3:a+5]; actions[i]=normalized(d['passed_actions'][a-3:a+4])
            # Byte equality for each h versus the original loader contract.
            for h in range(4):
                if not np.array_equal(frames[i,h:h+5],d['rgb'][a+h-3:a+h+2]) or not np.array_equal(actions[i,h:h+4],normalized(d['passed_actions'][a+h-3:a+h+1])):raise ValueError('Cache/H1 timing differs')
        if before!=fingerprint(p) or mbefore!=fingerprint(mp):raise ValueError('Source mutated during verification')
        for f,digest,stat in ((p,actual,before),(mp,mh,mbefore)):files.append(dict(path=str(f),sha256=digest,stat=stat))
        source_bytes+=sum(v.nbytes for v in d.values())
    frames.flush(); actions.flush(); del frames,actions
    for temp,final in ((ft,fp),(at,ap)):
        with temp.open('rb+') as stream:os.fsync(stream.fileno())
        replace_atomic(temp,final)
    cache_records=[dict(path=str(p),sha256=sha(p),stat=fingerprint(p)) for p in (fp,ap)]
    summary=dict(segments=n,unique_targets=4*n,independent_families=1313,excluded_short_tail_H1=663,
        per_letter=dict(Counter(s['letter'] for s in segments)),all_committed=True,all_clean=True,physics_calls=0,model_calls=0,
        original_index_sha256=sha(source),preview_sha256=PREVIEW_SHA,cache_bytes=cache_bytes,
        frames_dtype='uint8',frames_shape=list(shape),actions_dtype='float32',actions_shape=[n,7,2],
        action_normalization='absolute.astype(float32)/float32(255.5)-float32(1)',
        cache_policy='read-only mmap; full hashes at entry, stat change checks in loader; no floating-point full-library copy',
        source_array_bytes_decoded_once=source_bytes,build_wall_seconds=time.perf_counter()-began,at_utc=utc())
    atomic_json(root/'pool-index.json',dict(summary=summary,segments=segments,excluded_H1_ids=leftovers,sources=files,cache=cache_records),exclusive=True)
    pools=defaultdict(list); rng=random.Random(20261101)
    for i,s in enumerate(segments):pools[(s['family_id'],s['event'])].append(i)
    keys=sorted(pools);rng.shuffle(keys)
    for values in pools.values():rng.shuffle(values)
    flat=[]
    while keys:
        remaining=[]
        for key in keys:
            flat.append(pools[key].pop())
            if pools[key]:remaining.append(key)
        keys=remaining
    batches=[flat[i:i+4] for i in range(0,n,4)]
    if sorted(flat)!=list(range(n)) or len(batches)!=1892 or any(len(b)!=4 for b in batches):raise ValueError('Incomplete paired epoch')
    atomic_json(root/'order.json',dict(seed=20261101,batches=batches,segment_ids=[segments[i]['segment_id'] for i in flat],order='family/event round robin, seeded within pools; shared A/B'),exclusive=True)
    return summary

class SegmentLoader:
    def __init__(self,root,config):
        self.root=Path(root);self.index=read(self.root/'pool-index.json');self.rows=self.index['segments'];self.loads=0
        self.cache={};self.bytes=0; self._sources={s['raw_path'] for s in self.rows}
        self._records={r['path']:r for r in self.index['sources']+self.index['cache']}
        for r in self._records.values():
            if sha(r['path'])!=r['sha256'] or fingerprint(r['path'])!=r['stat']:raise ValueError('Frozen pool source/cache changed: '+r['path'])
        self.frames=np.memmap(self.root/'cache/frames.u8',mode='r',dtype=np.uint8,shape=(7568,8,3,96,96))
        self.actions=np.memmap(self.root/'cache/actions.f32',mode='r',dtype=np.float32,shape=(7568,7,2))
    def batch(self,indices):
        # All selected sources checked at entry; only touched sources and cache
        # metadata are checked here. Changed files cause stop rather than stale use.
        paths={str(self.root/'cache/frames.u8'),str(self.root/'cache/actions.f32')}
        for i in indices:
            paths.add(self.rows[i]['raw_path']);paths.add(str(Path(self.rows[i]['raw_path']).with_suffix('.json')))
        for p in paths:
            if fingerprint(p)!=self._records[p]['stat']:raise ValueError('Frozen input changed during run: '+p)
        self.loads+=1
        return dict(image=np.asarray(self.frames[indices]).copy(),action=np.asarray(self.actions[indices]).copy())
    def close(self):
        self.frames._mmap.close();self.actions._mmap.close()
