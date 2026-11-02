"""Saved-state RGB rendering and CPU-only R3A training data interfaces.

State/geometry/letter labels are renderer and audit data only. They are never
returned as model features. No physics stepping, model, Torch or training lives
in this module.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

LETTERS=tuple('TLOCXB')
PROTOCOL_SHA256='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class RecordedStateRenderer:
    """One reusable headless original renderer per seen letter; never step it."""
    def __init__(self,source_root,*,maximum_frame_cache_bytes=32*1024**2):
        if type(maximum_frame_cache_bytes)is not int or maximum_frame_cache_bytes<0:
            raise ValueError('Nonnegative bounded RAM cache required')
        self.source_root=Path(source_root).resolve();self.maximum_frame_cache_bytes=maximum_frame_cache_bytes
        self.cache=OrderedDict();self.cache_bytes=0;self.environments={}
        self.stats=dict(environment_constructions=0,initial_setup_render_calls=0,rendered_frames=0,cache_hits=0,physics_steps=0,physics_controls=0)

    def _environment(self,letter,first_state):
        if letter not in LETTERS:raise ValueError('Only the six declared seen letters may be rendered')
        if letter not in self.environments:
            from .native_rgb_env import make_env
            from .envs.official import T_REFERENCE_COG
            env=make_env(self.source_root,letter=letter,render_size=96)
            # The sole setup uses this saved pose, not a fresh random scene.
            # Velocities/forces do not affect RGB and are set immediately below.
            # reset_scene takes the physical body CoG, distinct from the
            # geometric area centroid used solely for amplitude statistics.
            cog=np.asarray(T_REFERENCE_COG if letter=='T'else(0.,0.),dtype=np.float64)
            angle=float(first_state[4]);c,s=math.cos(angle),math.sin(angle)
            center=np.asarray(first_state[2:4])+np.array(((c,-s),(s,c)))@cog
            env.reset_scene(center=center,angle=angle,pusher=first_state[:2])
            self.environments[letter]=env;self.stats['environment_constructions']+=1;self.stats['initial_setup_render_calls']+=1
        return self.environments[letter]

    def render(self,letter,state,*,cache_key=None):
        state=np.asarray(state)
        if state.dtype!=np.float64 or state.shape!=(15,)or not np.isfinite(state).all():
            raise ValueError('Exact saved finite float64 state15 required; no pose reduction')
        key=None if cache_key is None else(letter,str(cache_key),hashlib.sha256(state.tobytes()).hexdigest())
        if key is not None and key in self.cache:
            self.cache.move_to_end(key);self.stats['cache_hits']+=1
            return self.cache[key].copy()
        env=self._environment(letter,state);env.set_render_state(state)
        before=env.state().copy();rgb=env.render_clean()
        if not np.array_equal(env.state(),before):raise RuntimeError('Recorded-state RGB rendering changed physics state')
        if rgb.dtype!=np.uint8 or rgb.shape!=(3,96,96):raise ValueError('Original96 CHW uint8 RGB layout changed')
        self.stats['rendered_frames']+=1
        if key is not None and rgb.nbytes<=self.maximum_frame_cache_bytes:
            while self.cache and self.cache_bytes+rgb.nbytes>self.maximum_frame_cache_bytes:
                _,removed=self.cache.popitem(last=False);self.cache_bytes-=removed.nbytes
            self.cache[key]=rgb.copy();self.cache_bytes+=rgb.nbytes
        return rgb.copy()

    def close(self):
        self.cache.clear();self.cache_bytes=0
        for env in self.environments.values():env.close()
        self.environments.clear()


def assert_model_inputs(value,*,batch=False):
    """Only RGB and normalized actions cross the network boundary."""
    if set(value)!={'image','action'}:raise ValueError('Model inputs may contain only image/action')
    image,action=np.asarray(value['image']),np.asarray(value['action'])
    shape=(None,5,3,96,96)if batch else(5,3,96,96)
    good_image=image.shape[1:]==shape[1:]if batch else image.shape==shape
    good_action=action.shape==(len(image),4,2)if batch else action.shape==(4,2)
    if (not good_image or not good_action or image.dtype!=np.float32 or action.dtype!=np.float32
            or not np.isfinite(image).all()or not np.isfinite(action).all()
            or image.min()<0 or image.max()>1 or action.min()<-1 or action.max()>1):
        raise ValueError('Expected float32 fiveRGB[0,1] and fouractions[-1,1]')
    return value


def _safe(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if Path(name).is_absolute()or not path.is_relative_to(root):raise ValueError('Trajectory reference escapes declared source root')
    return path


def load_reference(manifest_path,*,require_complete_library=True,allow_legacy_benchmark=False):
    path=Path(manifest_path).resolve();manifest=read(path)
    production=manifest.get('kind')=='R3A-compact-state-reference.v1'
    benchmark=manifest.get('kind')=='R3A-loader-benchmark-R2B-reference.v1'
    if not production and not(benchmark and allow_legacy_benchmark):
        raise ValueError('Fresh R3A compact reference required; old library is benchmark-only')
    if manifest.get('protocol_sha256')!=PROTOCOL_SHA256 or manifest.get('control_dt')!=.1:
        raise ValueError('Selected physics identity and0.1s control timeline required')
    source=(path.parent/manifest.get('source_root','.')).resolve();entries=manifest['entries']
    if not entries or len({e['trajectory_id']for e in entries})!=len(entries):raise ValueError('Unique nonempty trajectories required')
    if len({e['family_id']for e in entries})!=len(entries)or len({e['seed']for e in entries})!=len(entries):
        raise ValueError('Independent families and scene seeds must not repeat across splits')
    if any(e['letter']not in LETTERS or e['split']not in('train','seen_val')or e['protocol_sha256']!=PROTOCOL_SHA256 for e in entries):
        raise ValueError('Only selected-physics sixseen TRAIN/newseen-validation references allowed')
    if production and any(e['controls']!=48 for e in entries):raise ValueError('Only complete successful48-control trajectories enter the new library')
    if require_complete_library:
        from collections import Counter
        counts=Counter((e['letter'],e['split'])for e in entries)
        if len(entries)!=912 or counts!={(l,s):n for l in LETTERS for s,n in(('train',128),('seen_val',24))}:
            raise ValueError('Fresh full library requires768 TRAIN+144 val,128/24 perletter; no silent small-library fallback')
    return manifest,source,entries


def _trajectory_arrays(path,entry,*,legacy_benchmark=False):
    if sha(path)!=entry['sha256']:raise ValueError('Compact trajectory SHA differs')
    with np.load(path,allow_pickle=False)as archive:
        if not legacy_benchmark and any(k in archive.files for k in('rgb','images','frames')):
            raise ValueError('New compact main library must not persist an RGB frame bank')
        states=archive['states'].copy();actions=archive['actions'].copy();n=len(actions)
        valid=archive['control_valid'].copy()if 'control_valid'in archive.files else(np.ones(n,dtype=bool)if legacy_benchmark else None)
        times=archive['control_time_seconds'].copy()if 'control_time_seconds'in archive.files else(np.arange(n+1,dtype=np.float64)*.1 if legacy_benchmark else None)
    if (n!=entry['controls']or states.dtype!=np.float64 or states.shape!=(n+1,15)
            or actions.dtype!=np.float64 or actions.shape!=(n,2)or not np.isfinite(states).all()or not np.isfinite(actions).all()
            or (actions<0).any()or(actions>511).any()or valid is None or valid.dtype!=bool or valid.shape!=(n,)
            or times is None or times.dtype!=np.float64 or times.shape!=(n+1,)
            or not np.allclose(times,np.arange(n+1)*.1,rtol=0,atol=1e-12)):
        raise ValueError('Exact state15/action/valid/time compact control timeline differs')
    return dict(states=states,actions=actions,control_valid=valid,control_time_seconds=times)


def build_window_catalogue(manifest_path,*,require_complete_library=True,allow_legacy_benchmark=False):
    """Full legal/valid H1/H4 catalogue from saved state, no renderer or physics."""
    from .envs.official import official_shape
    manifest,source,entries=load_reference(manifest_path,require_complete_library=require_complete_library,allow_legacy_benchmark=allow_legacy_benchmark)
    output=dict(kind='R3A-full-legal-state-window-catalogue.v1',manifest_sha256=sha(manifest_path),
        protocol_sha256=PROTOCOL_SHA256,h1=[],h4=[],entries=entries,
        moving_rule='H1 translation>=2 image96 pixels OR abs rotation>=5degrees',
        validity='all saved transitions from anchor-3 through anchor+horizon-1 must be valid',
        no_RGB_persisted=True,no_model_calls=True,no_physics_steps=True)
    for trajectory_index,entry in enumerate(entries):
        path=_safe(source,entry['file']);arrays=_trajectory_arrays(path,entry,legacy_benchmark=manifest['kind']=='R3A-loader-benchmark-R2B-reference.v1')
        if entry.get('metadata_file'):
            meta=_safe(source,entry['metadata_file'])
            if sha(meta)!=entry['metadata_sha256']:raise ValueError('Trajectory metadata SHA differs')
        states=arrays['states'];cog=np.asarray(official_shape(entry['letter']).centroid)
        c,s=np.cos(states[:,4]),np.sin(states[:,4]);rotated=np.column_stack((c*cog[0]-s*cog[1],s*cog[0]+c*cog[1]))
        area_centers=states[:,2:4]+rotated
        for horizon,key in((1,'h1'),(4,'h4')):
            for anchor in range(3,len(states)-horizon):
                valid=bool(arrays['control_valid'][anchor-3:anchor+horizon].all())
                if not valid:continue
                translation=float(np.linalg.norm(area_centers[anchor+horizon]-area_centers[anchor])*96/512)
                rotation=float((states[anchor+horizon,4]-states[anchor,4])*180/math.pi)
                trans=translation>=2;rot=abs(rotation)>=5
                output[key].append(dict(dataset_index=len(output[key]),window_id=f"{entry['trajectory_id']}:{anchor}:H{horizon}",
                    trajectory_index=trajectory_index,trajectory_id=entry['trajectory_id'],family_id=entry['family_id'],
                    letter=entry['letter'],split=entry['split'],anchor=anchor,horizon=horizon,
                    translation_image96_px=translation,signed_rotation_deg=rotation,abs_rotation_deg=abs(rotation),
                    translation=trans,rotation=rot,moving=trans or rot,
                    category='both'if trans and rot else'translation_only'if trans else'rotation_only'if rot else'other',
                    valid_window=True,control_dt=.1))
    output['identity']=dict(kind=output['kind'],manifest_sha256=output['manifest_sha256'],
        h1_sha256=canonical_sha(output['h1']),h4_sha256=canonical_sha(output['h4']))
    return output


def select_training_pool(catalogue,*,seed=5478):
    """Keep all moving windows and balanced distinct negatives, no replacement."""
    from collections import defaultdict,Counter,deque
    train=[r for r in catalogue['h1']if r['split']=='train'];moving=[r for r in train if r['moving']]
    other=[r for r in train if not r['moving']]
    if not moving:raise ValueError('No distinct obvious-motion support; collect/report rather than lower thresholds or repeat windows')
    limit=min(len(other),len(moving));groups=defaultdict(lambda:defaultdict(list))
    for row in other:groups[row['letter']][row['family_id']].append(row)
    for letter in groups:
        for family,rows in groups[letter].items():rows.sort(key=lambda r:canonical_sha(['R3A-negative-pool-v1',seed,r['window_id']]))
    queues={l:deque(sorted(groups[l],key=lambda f:canonical_sha([seed,l,f])))for l in groups}
    selected=[]
    while len(selected)<limit:
        changed=False
        for letter in LETTERS:
            q=queues.get(letter)
            if not q:continue
            family=q.popleft();selected.append(groups[letter][family].pop());changed=True
            if groups[letter][family]:q.append(family)
            if len(selected)==limit:break
        if not changed:break
    indices=sorted(r['dataset_index']for r in moving+selected);selected_rows={r['dataset_index']:r for r in train}
    result=dict(kind='R3A-distinct-selected-TRAIN-pool.v1',catalogue_identity=catalogue['identity'],seed=seed,
        selected_h1_training_indices=indices,complete_legal_TRAIN_windows=len(train),selected_windows=len(indices),
        selected_obvious_motion=len(moving),selected_other=len(selected),
        selected_motion_fraction=len(moving)/len(indices),complete_legal_library_preserved=True,
        rare_both_sampling='all selected windows once per epoch, no category repetition quota',
        unique_trajectories=len({selected_rows[i]['trajectory_id']for i in indices}),
        category_counts=dict(Counter(selected_rows[i]['category']for i in indices)))
    result['identity_sha256']=canonical_sha(result)
    return result


def export_compact_catalogue(catalogue,selected_pool,out):
    """Persist numeric references, not repeated verbose window metadata/RGB."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    arrays={};categories=['other','translation_only','rotation_only','both']
    for key in('h1','h4'):
        rows=catalogue[key]
        arrays[key+'_trajectory_index']=np.asarray([r['trajectory_index']for r in rows],dtype=np.int16)
        arrays[key+'_anchor']=np.asarray([r['anchor']for r in rows],dtype=np.int16)
        arrays[key+'_translation_image96_px']=np.asarray([r['translation_image96_px']for r in rows],dtype=np.float64)
        arrays[key+'_signed_rotation_deg']=np.asarray([r['signed_rotation_deg']for r in rows],dtype=np.float64)
        arrays[key+'_category']=np.asarray([categories.index(r['category'])for r in rows],dtype=np.uint8)
    arrays['selected_h1_training_indices']=np.asarray(selected_pool['selected_h1_training_indices'],dtype=np.int32)
    path=out/'window-catalogue.npz'
    with path.open('wb')as stream:np.savez_compressed(stream,**arrays)
    metadata=dict(kind='R3A-compact-numeric-window-reference.v1',catalogue_identity=catalogue['identity'],
        file=path.name,sha256=sha(path),bytes=path.stat().st_size,h1_windows=len(catalogue['h1']),h4_windows=len(catalogue['h4']),
        trajectory_entry_order='same manifest.entries',category_codes=dict(enumerate(categories)),
        array_dtypes={k:str(v.dtype)for k,v in arrays.items()},
        selected_pool={k:v for k,v in selected_pool.items()if k!='selected_h1_training_indices'},
        no_images_or_model_inputs_persisted=True)
    (out/'window-catalogue.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return metadata


class R3AStateDataset:
    """On-demand saved-state image/action data; getitem contains exactly2 keys."""
    def __init__(self,manifest_path,source_root,*,split='train',catalogue=None,selected_pool=None,
                 frame_cache_bytes=16*1024**2,state_cache_bytes=8*1024**2,
                 require_complete_library=True,allow_legacy_benchmark=False,renderer_factory=RecordedStateRenderer):
        self.manifest_path=Path(manifest_path).resolve();self.physics_source_root=Path(source_root).resolve();self.split=split
        if split not in('train','seen_val')or type(state_cache_bytes)is not int or state_cache_bytes<0:raise ValueError('Explicit split and bounded state RAM required')
        self.manifest,self.source_root,self.entries=load_reference(self.manifest_path,require_complete_library=require_complete_library,allow_legacy_benchmark=allow_legacy_benchmark)
        self.legacy_benchmark=self.manifest['kind']=='R3A-loader-benchmark-R2B-reference.v1';self.frame_cache_bytes=frame_cache_bytes;self.state_cache_bytes=state_cache_bytes
        self.renderer_factory=renderer_factory;self.renderer=None;self.state_cache=OrderedDict();self.state_cache_nbytes=0
        self.catalogue=catalogue if catalogue is not None else build_window_catalogue(self.manifest_path,require_complete_library=require_complete_library,allow_legacy_benchmark=allow_legacy_benchmark)
        if self.catalogue['manifest_sha256']!=sha(self.manifest_path):raise ValueError('Catalogue belongs to another manifest')
        rows=[r for r in self.catalogue['h1']if r['split']==split]
        if selected_pool is not None:
            if split!='train'or selected_pool['catalogue_identity']!=self.catalogue['identity']:raise ValueError('Selected pool must be this TRAIN catalogue')
            requested=selected_pool['selected_h1_training_indices'];selected=set(requested)
            if len(selected)!=len(requested)or not selected<={r['dataset_index']for r in rows}:raise ValueError('Selected pool repeats indices or includes validation/unknown windows')
            rows=[r for r in rows if r['dataset_index']in selected]
        self.window_rows=rows;self.getitem_calls=0

    def __len__(self):return len(self.window_rows)
    def identity(self):return dict(kind='R3A-state-to-RGB-array-interface.v1',manifest_sha256=sha(self.manifest_path),
        split=self.split,window_ids_sha256=canonical_sha([r['window_id']for r in self.window_rows]),
        model_inputs=['image','action'],RGB='5CHW96 float32/255 once',action='4absoluteXY float32/255.5-1 once')
    def metadata(self,index):return dict(self.window_rows[index])
    def _trajectory(self,index):
        if index in self.state_cache:self.state_cache.move_to_end(index);return self.state_cache[index]
        entry=self.entries[index];arrays=_trajectory_arrays(_safe(self.source_root,entry['file']),entry,legacy_benchmark=self.legacy_benchmark)
        nbytes=sum(v.nbytes for v in arrays.values())
        while self.state_cache and self.state_cache_nbytes+nbytes>self.state_cache_bytes:
            _,removed=self.state_cache.popitem(last=False);self.state_cache_nbytes-=sum(v.nbytes for v in removed.values())
        if nbytes<=self.state_cache_bytes:self.state_cache[index]=arrays;self.state_cache_nbytes+=nbytes
        return arrays
    def __getitem__(self,index):
        if type(index)is not int or not 0<=index<len(self):raise IndexError(index)
        row=self.window_rows[index];entry=self.entries[row['trajectory_index']];arrays=self._trajectory(row['trajectory_index']);anchor=row['anchor']
        if self.renderer is None:self.renderer=self.renderer_factory(self.physics_source_root,maximum_frame_cache_bytes=self.frame_cache_bytes)
        frames=[self.renderer.render(entry['letter'],arrays['states'][k],cache_key=entry['sha256']+':'+str(k))for k in range(anchor-3,anchor+2)]
        value=dict(image=np.stack(frames).astype(np.float32)/np.float32(255),
            action=arrays['actions'][anchor-3:anchor+1].astype(np.float32)/np.float32(255.5)-np.float32(1))
        self.getitem_calls+=1
        return assert_model_inputs(value)
    def statistics(self):return dict(getitem_calls=self.getitem_calls,state_cache_bytes=self.state_cache_nbytes,
        maximum_state_cache_bytes=self.state_cache_bytes,frame_cache_bytes=0 if self.renderer is None else self.renderer.cache_bytes,
        maximum_frame_cache_bytes=self.frame_cache_bytes,renderer={}if self.renderer is None else dict(self.renderer.stats))
    def close(self):
        self.state_cache.clear();self.state_cache_nbytes=0
        if self.renderer is not None:self.renderer.close();self.renderer=None


class StratifiedEpochSampler:
    """Each selected window once/epoch; same-family windows never share a batch."""
    def __init__(self,window_rows,*,seed=5478,batch_size=16):
        if type(batch_size)is not int or batch_size<1 or not window_rows:raise ValueError('Nonempty windows and positive batch size required')
        if any(r['split']!='train'for r in window_rows):raise ValueError('Only TRAIN can be sampled')
        self.rows=[dict(r)for r in window_rows];self.batch_size=batch_size;self.rng=np.random.default_rng(seed)
        self.seed=seed;self.epoch=0;self.batch_cursor=0;self.total_exposures=0;self.exposures={}
        self.identity_payload=dict(kind='R3A-trajectory-letter-amplitude-epoch-order.v1',seed=seed,batch_size=batch_size,
            windows_sha256=canonical_sha(self.rows),without_replacement=True,one_family_per_batch=True,
            amplitude_choice='natural remaining-window proportions; no rare-both oversampling')
        self.batches=self._build_epoch()
    def identity(self):return dict(self.identity_payload)
    def _build_epoch(self):
        from collections import defaultdict,deque
        groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,row in enumerate(self.rows):groups[row['letter']][row['family_id']][row['category']].append(i)
        for families in groups.values():
            for categories in families.values():
                for category,values in list(categories.items()):categories[category]=self.rng.permutation(values).tolist()
        queues={letter:deque(self.rng.permutation(list(families)).tolist())for letter,families in groups.items()}
        remaining=len(self.rows);batches=[];letter_order=[]
        while remaining:
            batch=[];used=set();misses=0
            while len(batch)<self.batch_size and remaining:
                if not letter_order:letter_order=self.rng.permutation(LETTERS).tolist()
                letter=letter_order.pop();queue=queues.get(letter)
                if not queue:misses+=1
                else:
                    candidates=[family for family in queue if family not in used]
                    if not candidates:misses+=1
                    else:
                        family=candidates[0];queue.remove(family);categories=groups[letter][family]
                        keys=[k for k,v in categories.items()if v];weights=np.asarray([len(categories[k])for k in keys],dtype=float)
                        category=str(self.rng.choice(keys,p=weights/weights.sum()));index=int(categories[category].pop())
                        batch.append(index);used.add(family);remaining-=1;misses=0
                        if any(categories.values()):queue.append(family)
                if misses>len(LETTERS)*2:break
            if not batch:raise RuntimeError('Cannot allocate remaining windows without fabricating samples')
            batches.append(batch)
        flat=[i for batch in batches for i in batch]
        if len(flat)!=len(self.rows)or len(set(flat))!=len(flat):raise RuntimeError('Epoch repeated or dropped windows')
        return batches
    def take_batch(self):
        if self.batch_cursor==len(self.batches):self.epoch+=1;self.batch_cursor=0;self.batches=self._build_epoch()
        batch=tuple(self.batches[self.batch_cursor]);self.batch_cursor+=1;self.total_exposures+=len(batch)
        for index in batch:self.exposures[index]=self.exposures.get(index,0)+1
        return batch
    def state_dict(self):
        value=dict(kind='R3A-epoch-sampler-state.v1',identity=self.identity(),epoch=self.epoch,batch_cursor=self.batch_cursor,
            batches=self.batches,rng=self.rng.bit_generator.state,total_exposures=self.total_exposures,
            exposures={str(k):v for k,v in self.exposures.items()})
        value['state_sha256']=canonical_sha(value);return json.loads(json.dumps(value))
    def load_state_dict(self,state):
        value=dict(state);digest=value.pop('state_sha256',None)
        if digest!=canonical_sha(value)or value['identity']!=self.identity():raise ValueError('Epoch sampler data/policy/checksum differs')
        flat=[i for batch in value['batches']for i in batch]
        if sorted(flat)!=list(range(len(self.rows)))or not 0<=value['batch_cursor']<=len(value['batches']):raise ValueError('Saved epoch must cover each selected window once')
        for batch in value['batches']:
            if len({self.rows[i]['family_id']for i in batch})!=len(batch):raise ValueError('Saved batch contains neighbouring same-family windows')
        epoch=value['epoch'];exposures={int(k):v for k,v in value['exposures'].items()}
        delivered={i for batch in value['batches'][:value['batch_cursor']]for i in batch}
        if(type(epoch)is not int or epoch<0 or any(type(v)is not int or v<1 for v in exposures.values())
                or set(exposures)-set(range(len(self.rows)))or sum(exposures.values())!=value['total_exposures']
                or any(exposures.get(i,0)!=epoch+int(i in delivered)for i in range(len(self.rows)))):
            raise ValueError('Unique/repeated exposure counters do not match complete epochs/current prefix')
        self.epoch=value['epoch'];self.batch_cursor=value['batch_cursor'];self.batches=value['batches'];self.rng.bit_generator.state=value['rng']
        self.total_exposures=value['total_exposures'];self.exposures=exposures
    def statistics(self):
        observed=list(self.exposures.values());values=[self.exposures.get(i,0)for i in range(len(self.rows))]
        families={self.rows[i]['family_id']for i in self.exposures}
        return dict(epoch=self.epoch,total_exposures=self.total_exposures,unique_windows=len(observed),unique_trajectories=len(families),
            repeated_exposures=self.total_exposures-len(observed),minimum_exposures=min(values)if values else 0,
            median_exposures=float(np.median(values))if values else 0,maximum_exposures=max(values)if values else 0,
            observed_window_exposure_min=min(observed)if observed else 0,
            observed_window_exposure_median=float(np.median(observed))if observed else 0,
            observed_window_exposure_max=max(observed)if observed else 0,
            exposure_summary_scope='all selected windows including unexposed zeros; observed-only fields are explicitly separate',
            unexposed_windows=len(values)-len(observed),pool_size=len(self.rows),without_replacement_each_epoch=True)


WORKER_DATASET=None
def _initialize_worker(kwargs):
    global WORKER_DATASET
    import atexit
    WORKER_DATASET=R3AStateDataset(**kwargs)
    atexit.register(WORKER_DATASET.close)

def _worker_batch(indices):
    import os
    began=time.perf_counter();items=[WORKER_DATASET[int(i)]for i in indices]
    model_inputs=assert_model_inputs({key:np.stack([item[key]for item in items])for key in('image','action')},batch=True)
    return dict(model_inputs=model_inputs,indices=tuple(indices),worker_pid=os.getpid(),
        worker_seconds=time.perf_counter()-began,worker_statistics=WORKER_DATASET.statistics())


class PrefetchBatchLoader:
    """Exactly2 spawn workers, at most2 in-flight CPU batches, bounded caches."""
    def __init__(self,dataset_kwargs,*,workers=2,prefetch_batches=2):
        if workers!=2 or prefetch_batches!=2:raise ValueError('Initial R3A loader fixes2workers/prefetch2')
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        self.executor=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker,initargs=(dict(dataset_kwargs),))
        self.prefetch_batches=2;self.worker_statistics={};self.maximum_in_flight=0;self.closed=False
    def iterate(self,batches):
        from collections import deque
        source=iter(batches);pending=deque()
        for _ in range(2):
            try:pending.append(self.executor.submit(_worker_batch,tuple(next(source))))
            except StopIteration:break
        while pending:
            self.maximum_in_flight=max(self.maximum_in_flight,len(pending));result=pending.popleft().result()
            self.worker_statistics[result['worker_pid']]=result['worker_statistics']
            try:pending.append(self.executor.submit(_worker_batch,tuple(next(source))))
            except StopIteration:pass
            yield result
    def close(self):
        if not self.closed:self.executor.shutdown(wait=True,cancel_futures=True);self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
