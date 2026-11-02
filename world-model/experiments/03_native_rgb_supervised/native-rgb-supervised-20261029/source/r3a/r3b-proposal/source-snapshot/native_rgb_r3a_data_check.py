"""Cold compact-library/catalogue/epoch/2worker gate; no model or physics."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r3a_data import (build_window_catalogue,select_training_pool,export_compact_catalogue,
    R3AStateDataset,StratifiedEpochSampler,sha,canonical_sha)
from scripts.native_rgb_r3a_renderer_check import benchmark_loader


def run(manifest_path,source_root,out,batches=32):
    import numpy as np
    out.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    catalogue=build_window_catalogue(manifest_path,require_complete_library=True)
    pool=select_training_pool(catalogue);compact=export_compact_catalogue(catalogue,pool,out)
    dataset=R3AStateDataset(manifest_path,source_root,split='train',catalogue=catalogue,selected_pool=pool)
    sampler=StratifiedEpochSampler(dataset.window_rows,batch_size=16);epoch=[];histogram=Counter();family_checks=0
    for _ in range(len(sampler.batches)):
        batch=sampler.take_batch();epoch.extend(batch);histogram[str(len(batch))]+=1
        if len({dataset.metadata(i)['family_id']for i in batch})!=len(batch):raise ValueError('Same-family/adjacent-window collision within training batch')
        family_checks+=1
    if sorted(epoch)!=list(range(len(dataset))):raise ValueError('Selected epoch repeated/dropped windows')
    before=sampler.state_dict();recovered=StratifiedEpochSampler(dataset.window_rows,batch_size=16);recovered.load_state_dict(before)
    resumed_equal=[]
    for _ in range(3):resumed_equal.append(sampler.take_batch()==recovered.take_batch())
    if not all(resumed_equal)or sampler.state_dict()!=recovered.state_dict():raise ValueError('Epoch RNG/exposure state resume differs')
    full_indices=np.asarray([dataset.window_rows[i]['dataset_index']for i in epoch],dtype=np.int32)
    np.savez_compressed(out/'one-epoch-index-audit.npz',selected_local_indices=np.asarray(epoch,dtype=np.int32),full_h1_indices=full_indices)
    complete_train=[r for r in catalogue['h1']if r['split']=='train'];complete_val=[r for r in catalogue['h1']if r['split']=='seen_val']
    selected=dataset.window_rows
    group_counts=lambda rows:dict(Counter(r['letter']+'/'+r['category']for r in rows))
    proof=dict(kind='R3A-fresh-main-library-data-interface-gate.v1',passed=True,manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),
        trajectories=len(catalogue['entries']),TRAIN_trajectories=sum(e['split']=='train'for e in catalogue['entries']),
        seen_val_trajectories=sum(e['split']=='seen_val'for e in catalogue['entries']),
        fulllegal_H1_windows=len(catalogue['h1']),fulllegal_H4_windows=len(catalogue['h4']),
        fulllegal_TRAIN_H1=len(complete_train),fulllegal_seen_val_H1=len(complete_val),
        complete_TRAIN_moving=sum(r['moving']for r in complete_train),
        complete_TRAIN_motion_fraction=sum(r['moving']for r in complete_train)/len(complete_train),
        selected_pool={k:v for k,v in pool.items()if k!='selected_h1_training_indices'},
        fulllegal_TRAIN_letter_category_counts=group_counts(complete_train),selected_letter_category_counts=group_counts(selected),
        epoch_samples=len(epoch),epoch_unique_windows=len(set(epoch)),epoch_unique_trajectories=len({selected[i]['trajectory_id']for i in epoch}),
        epoch_repeated_exposures=0,epoch_window_exposure_min=1,epoch_window_exposure_median=1,epoch_window_exposure_max=1,
        same_batch_family_separation_checked=family_checks,batch_size_histogram=dict(histogram),
        partial_batch_policy='No dropping, duplication or cross-epoch padding; caller must explicitly accumulate a partial tail if it exists',
        rare_both_exposures_in_epoch=sum(selected[i]['category']=='both'for i in epoch),
        rare_both_distinct_windows=sum(r['category']=='both'for r in selected),
        sampler_rng_and_exposure_resume_exact=True,serialized_sampler_state_sha256=before['state_sha256'],
        first_epoch_window_order_sha256=canonical_sha([selected[i]['window_id']for i in epoch]),
        compact_window_reference=compact,index_archive_sha256=sha(out/'one-epoch-index-audit.npz'),
        model_inputs=['image','action'],state_letter_geometry_masks_are_not_model_inputs=True,
        model_calls=0,checkpoint_loads=0,physics_controls=0,renderer_calls_for_catalogue=0,
        elapsed_catalogue_audit_seconds=time.perf_counter()-began,Torch_imported='torch'in sys.modules)
    dataset.close();(out/'interface-gate.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    code=benchmark_loader(source_root,out/'two-worker-throughput',manifest_path=manifest_path,batches=batches,selected_pool=pool)
    if code:raise RuntimeError('New-library2worker throughput failed')
    proof['two_worker_throughput_file']='two-worker-throughput/throughput.json'
    proof['two_worker_throughput_sha256']=sha(out/'two-worker-throughput/throughput.json')
    (out/'interface-gate.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:proof[k]for k in('passed','trajectories','TRAIN_trajectories','seen_val_trajectories','fulllegal_TRAIN_H1','selected_pool','epoch_unique_windows','epoch_unique_trajectories','batch_size_histogram','sampler_rng_and_exposure_resume_exact')}))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batches',type=int,default=32)
    a=p.parse_args();return run(a.manifest.resolve(),a.source_root.resolve(),a.out.resolve(),a.batches)

if __name__=='__main__':raise SystemExit(main())
