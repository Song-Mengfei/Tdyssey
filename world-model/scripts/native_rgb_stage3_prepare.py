"""Freeze visible TRAIN18 and existing DEV36 before any model work."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_training import atomic_json,sha256
from tdyssey.native_rgb_data import NativeRGBDataset,TRAIN_LETTERS
from tdyssey.native_rgb_motion_statistics import transition_rows


def authorize(output):
    parent=Path(output).resolve().parent;(parent/'SUPERVISION.md').read_text(encoding='utf-8')
    state=json.loads((parent/'stage-state.json').read_text())
    if state['current_stage']!='3' or state['authorized_stages']!=['3'] or state['status']!='in_progress' or state['callback']['sent']:
        raise RuntimeError('Only the released single stage3 microfit is allowed')
    return state


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def select_train(dataset):
    """No rendering, learned score, random draw or training iterator."""
    import numpy as np
    lookup={(int(i),int(anchor)):index for index,(i,anchor) in enumerate(dataset.windows)}
    selected={};support={l:0 for l in TRAIN_LETTERS}
    for i,entry in enumerate(dataset.entries):
        if entry['split']!='train':raise ValueError('Only formal TRAIN may be fitted')
        _,arrays=dataset.trajectory(i);valid=arrays['valid_transition'];n=len(valid)
        meta=json.loads((dataset.data_root/entry['metadata_file']).read_text()) if hasattr(dataset,'data_root') else None
        # Reference entries are rooted at the supervised base, not joint-data/.
        if meta is None:
            from tdyssey.native_rgb_reference import resolve_reference_entry
            meta=json.loads(resolve_reference_entry(dataset.root,dataset.manifest,entry,'metadata_file').read_text())
        stats=transition_rows(entry,arrays,meta['steps'])
        for anchor in range(3,n-3):
            row=stats[anchor]
            if (not valid[anchor-3:anchor+4].all() or int(arrays['contact'][anchor])<=0
                or not(row['translation_world_px']*96/512>=1 or row['rotation_abs_deg']>=3)):
                continue
            letter=entry['shape'];support[letter]+=1
            rank=canonical_hash(['stage3-visible-microfit-v1',2026102904,letter,entry['trajectory_id'],entry['sha256'],entry['metadata_sha256'],anchor])
            candidate=dict(window_id=f"{entry['trajectory_id']}:{anchor}",letter=letter,trajectory_id=entry['trajectory_id'],
                anchor=int(anchor),dataset_index=int(lookup[i,anchor]),entry_index=i,split='train',selection_hash=rank,
                npz_sha256=entry['sha256'],metadata_sha256=entry['metadata_sha256'],npz_file=entry['file'],metadata_file=entry['metadata_file'],
                seed=entry['seed'],source_stage=entry['stage'],translation_world_px=row['translation_world_px'],
                translation_image_px=row['translation_world_px']*96/512,rotation_abs_deg=row['rotation_abs_deg'],
                contact_count=int(arrays['contact'][anchor]),h1_motion_class=row['motion_class'],motion_class=row['motion_class'],
                actual_spatial_before=row['actual_spatial_class_before'],actual_spatial_after=row['actual_spatial_class_after'],
                min_wall_gap_before=row['min_wall_gap_before'],min_wall_gap_after=row['min_wall_gap_after'],
                boundary_min_world_px=min(row['min_wall_gap_before'],row['min_wall_gap_after']),
                gap_band_before=row['gap_band_before'],gap_band_after=row['gap_band_after'],h4_legal=True,
                selection_rule='min SHA256 among H4-legal visible-motion+contact candidates per letter')
            if letter not in selected or (rank,entry['trajectory_id'],anchor)<(selected[letter]['selection_hash'],selected[letter]['trajectory_id'],selected[letter]['anchor']):selected[letter]=candidate
    missing=[l for l in TRAIN_LETTERS if l not in selected]
    if missing:raise ValueError('No permitted visible-motion support; no threshold relaxation: '+','.join(missing))
    return [selected[l] for l in TRAIN_LETTERS],support


def select_dev(dataset,index_path):
    frozen=json.loads(Path(index_path).read_text());dev=frozen['development']
    if len(dev)!=36:raise ValueError('Existing development panel must remain36')
    entries={e['trajectory_id']:(i,e) for i,e in enumerate(dataset.entries)}
    lookup={(int(i),int(a)):index for index,(i,a) in enumerate(dataset.windows)};result=[]
    for row in dev:
        i,entry=entries[row['trajectory_id']];anchor=int(row['anchor'])
        _,arrays=dataset.trajectory(i)
        if entry['split']!='seen_val' or not 3<=anchor<=len(arrays['valid_transition'])-4 or not arrays['valid_transition'][anchor-3:anchor+4].all():raise ValueError('Frozen development H4 split/timeline changed')
        if row['source']['npz_sha256']!=entry['sha256'] or row['source']['metadata_sha256']!=entry['metadata_sha256']:raise ValueError('Frozen development source identities changed')
        result.append(dict(row,dataset_index=int(lookup[i,anchor]),entry_index=i,split='seen_val',npz_file=entry['file'],metadata_file=entry['metadata_file'],
            npz_sha256=entry['sha256'],metadata_sha256=entry['metadata_sha256'],h1_motion_class=row['h1']['motion_class'],
            letter=entry['shape'],source_stage=entry['stage'],selection_rule='exact existing stage2C development order; no resampling'))
    return result


def batch_plan(groups):
    plans={}
    for cohort,windows in groups.items():
        batches=[]
        for start in range(0,len(windows),4):
            ids=[r['window_id'] for r in windows[start:start+4]]
            seed=int(canonical_hash(['stage3-paired-native-H1-v1',2026102905,cohort,start//4,ids])[:8],16)
            batches.append(dict(batch_index=start//4,positions=list(range(start,min(start+4,len(windows)))),window_ids=ids,batch_seed=seed,batch_size=len(ids)))
        plans[cohort]=batches
    return dict(kind='stage3-fixed-paired-native-H1-batches.v1',batch_size=4,cohorts=plans,
        noise_pairing='same cohort/member order/batch shape/seed for frozen and fitted; not a per-window sequential RNG claim',
        sampler='all unchanged published sampler settings',predictions_allowed=['frozen_gpc','final_fitted'],horizon=1)


def save_truth(dataset,windows,path):
    import numpy as np
    histories=[];actions=[];last=[];truth=[];last_masks=[];true_masks=[]
    for row in windows:
        i,anchor=dataset.windows[row['dataset_index']]
        if dataset.entries[i]['trajectory_id']!=row['trajectory_id'] or anchor!=row['anchor']:raise ValueError('Exact dataset-index binding changed')
        item=dataset[row['dataset_index']];entry,arrays=dataset.trajectory(i)
        if item['image'].shape!=(5,3,96,96) or item['action'].shape!=(4,2):raise ValueError('Original batch layout changed')
        histories.append(item['image'][:4]);last.append(item['image'][3]);truth.append(item['image'][4]);actions.append(arrays['actions'][anchor-3:anchor+1].copy())
        last_masks.append(dataset.render_layers(entry,arrays['states'][anchor])['full_object_mask'])
        true_masks.append(dataset.render_layers(entry,arrays['states'][anchor+1])['full_object_mask'])
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,histories=np.asarray(histories,dtype=np.float32),actions_absolute=np.asarray(actions,dtype=np.float32),
            last=np.asarray(last,dtype=np.float32),true=np.asarray(truth,dtype=np.float32),hold=np.asarray(last,dtype=np.float32),
            last_full_mask=np.asarray(last_masks,dtype=np.float32),true_full_mask=np.asarray(true_masks,dtype=np.float32),
            window_ids=np.asarray([r['window_id'] for r in windows]))


def prepare(output,joint,physics_root):
    output=Path(output).resolve();authorize(output);started=time.monotonic()
    if (output/'frozen-windows.json').exists():raise FileExistsError('Preserve first fixed18/36 selection')
    train=NativeRGBDataset(joint,'train',physics_source_root=physics_root,cache_bytes=1024**2)
    dev=NativeRGBDataset(joint,'seen_val',physics_source_root=physics_root,cache_bytes=1024**2)
    selected,support=select_train(train);dev_path=output.parent/'stage2c/report/indices/indices.json';validation=select_dev(dev,dev_path)
    if len({r['trajectory_id'] for r in selected})!=18:raise ValueError('TRAIN18 must be independent trajectories')
    result=dict(kind='stage3-frozen-visible-TRAIN18-development36.v1',train=selected,dev=validation,visible_candidate_support=support,
        train_hash_seed=2026102904,source_joint_manifest_sha256=sha256(Path(joint)/'manifest.json'),
        source_development_indices_sha256=sha256(dev_path),new_physics_controls=0,model_calls=0,training_index_draws=0,
        dataset_index_scope='exact joint NativeRGBDataset H1 catalog; metadata never becomes model input')
    atomic_json(output/'frozen-windows.json',result);atomic_json(output/'paired-batches.json',batch_plan({'train18':selected,'dev36':validation}))
    truth_dir=output/'truth';truth_dir.mkdir(exist_ok=False)
    save_truth(train,selected,truth_dir/'train18.npz');save_truth(dev,validation,truth_dir/'dev36.npz')
    atomic_json(output/'CPU-preparation.json',dict(wall_seconds=time.monotonic()-started,new_physics_controls=0,model_calls=0,
        training_iterator_calls=0,renderer='saved real states only',train_windows=18,dev_windows=36,
        train_truth_sha256=sha256(truth_dir/'train18.npz'),dev_truth_sha256=sha256(truth_dir/'dev36.npz')))
    return dict(train=18,dev=36,visible_candidate_support=support,wall_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage3')
    p.add_argument('--joint',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage2c/joint-data')
    p.add_argument('--physics-root',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output,a.joint,a.physics_root)))
