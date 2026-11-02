"""Compare saved R2B state15 renders to their exact original RGB; no physics."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r3a_data import RecordedStateRenderer,read,sha


def run(source_root,r2b_root,out):
    import numpy as np
    import pymunk
    from PIL import Image
    out.mkdir(parents=True,exist_ok=False);index=read(r2b_root/'index.json')
    renderer=RecordedStateRenderer(source_root,maximum_frame_cache_bytes=0)
    rows=[];mismatches=[];examples=[];began=time.perf_counter();angle_bins=Counter();contacts=Counter();spaces=Counter();letters=Counter()
    guards=dict(physics_step_calls=0)
    def forbidden_step(*args,**kwargs):
        guards['physics_step_calls']+=1
        raise AssertionError('Offline renderer may never advance pymunk.Space.step')
    try:
        with patch.object(pymunk.Space,'step',forbidden_step):
            for slot in index['slots']:
                if slot['status']!='completed':continue
                path=r2b_root/slot['trajectory_file'];metadata_path=r2b_root/slot['metadata_file']
                if sha(path)!=slot['trajectory_sha256']or sha(metadata_path)!=slot['metadata_sha256']:
                    raise ValueError('Original R2B data identity changed')
                metadata=read(metadata_path)
                with np.load(path,allow_pickle=False)as z:states=z['states'].copy();rgb=z['rgb'].copy()
                if states.dtype!=np.float64 or states.shape!=(25,15)or rgb.shape!=(25,3,96,96)or rgb.dtype!=np.uint8:
                    raise ValueError('Original complete R2B state/RGB timeline differs')
                for frame in range(25):
                    actual=renderer.render(slot['letter'],states[frame]);expected=rgb[frame]
                    difference=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
                    record=dict(trajectory_id=slot['request_id'],letter=slot['letter'],initial_space=slot['initial_space'],frame_index=frame,
                        time_seconds=frame*.1,state15_sha256=hashlib.sha256(states[frame].tobytes()).hexdigest(),
                        expected_RGB_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),rendered_RGB_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                        exact=bool(np.array_equal(actual,expected)),different_channel_values=int(np.count_nonzero(difference)),maximum_absolute_channel_error=int(difference.max()))
                    rows.append(record);letters[slot['letter']]+=1;spaces[slot['initial_space']]+=1
                    angle_bins[str(int(np.floor((states[frame,4]%(2*np.pi))/(np.pi/4))))]+=1
                    if frame:
                        r=metadata['control_records'][frame-1]
                        contact=r.get('agent_object_contact',r.get('contact_substeps',r.get('contact',None)))
                        contacts[str(contact)]+=1
                    if not record['exact']:mismatches.append(record)
                    if frame in(0,12,24)and slot['letter']not in {x['letter']for x in examples}:
                        name=f"{slot['letter']}-frame{frame:02d}.png"
                        tile=np.concatenate((np.moveaxis(expected,0,-1),np.moveaxis(actual,0,-1)),axis=1)
                        Image.fromarray(tile).resize((768,384),Image.Resampling.NEAREST).save(out/name)
                        examples.append(dict(letter=slot['letter'],file=name,frame_index=frame,trajectory=slot['request_id'],sha256=sha(out/name)))
    finally:renderer.close()
    result=dict(kind='R3A-stored-state15-to-original-R2B-RGB-consistency.v1',passed=not mismatches,
        exact_frames=sum(r['exact']for r in rows),frames=len(rows),trajectories=len({r['trajectory_id']for r in rows}),
        mismatches=mismatches,maximum_absolute_channel_error=max((r['maximum_absolute_channel_error']for r in rows),default=0),
        letter_counts=dict(letters),angle_octant_counts=dict(angle_bins),initial_space_counts=dict(spaces),contact_audit=dict(contacts),
        original_RGB_geometry_palette_resolution_changed=False,timeline='saved state[k] compared only with original RGB[k], control_dt0.1s',
        source_root=str(source_root),R2B_index_sha256=sha(r2b_root/'index.json'),renderer_source_sha256=sha(ROOT/'tdyssey/native_rgb_env.py'),
        offline_data_source_sha256=sha(ROOT/'tdyssey/native_rgb_r3a_data.py'),renderer_statistics=renderer.stats,
        forbidden_physics_step_guard=guards,physics_controls=0,model_calls=0,checkpoint_loads=0,Torch_imported='torch'in sys.modules,
        elapsed_seconds=time.perf_counter()-began,comparisons=rows,examples=examples)
    (out/'consistency.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>原存RGB / state15离线渲染</h1><p>每图左列为原始存RGB，右列为同帧state15渲染。无physics step、无模型；全量结果见consistency.json。</p>'+''.join(f'<h2>{e["letter"]}</h2><img src="{e["file"]}">'for e in examples),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k not in('comparisons','mismatches','examples')},ensure_ascii=False))
    return 0 if result['passed']else 2


def benchmark_loader(source_root,out,*,manifest_path=None,r2b_root=None,batches=32,allow_partial=False,selected_pool=None):
    import numpy as np
    from tdyssey.native_rgb_r3a_data import (R3AStateDataset,StratifiedEpochSampler,PrefetchBatchLoader,assert_model_inputs,PROTOCOL_SHA256)
    out.mkdir(parents=True,exist_ok=False)
    legacy=manifest_path is None
    if legacy:
        if r2b_root is None:raise ValueError('Legacy benchmark requires explicit R2B source')
        index=read(r2b_root/'index.json');splits=read(r2b_root.parent/'r2c0/data/split-plan.json')['assignments'];entries=[]
        for slot in index['slots']:
            if slot['status']!='completed':continue
            metadata=read(r2b_root/slot['metadata_file'])
            entries.append(dict(trajectory_id=slot['request_id'],family_id=slot['family_id'],slot_id=slot['slot_id'],
                letter=slot['letter'],split=splits[slot['request_id']],seed=metadata['request']['seed'],
                file=slot['trajectory_file'],sha256=slot['trajectory_sha256'],metadata_file=slot['metadata_file'],metadata_sha256=slot['metadata_sha256'],
                controls=24,protocol_sha256=PROTOCOL_SHA256))
        manifest=dict(kind='R3A-loader-benchmark-R2B-reference.v1',source_root=str(r2b_root),entries=entries,
            protocol_sha256=PROTOCOL_SHA256,control_dt=.1,benchmark_only=True,not_new_R3A_training_data=True)
        manifest_path=out/'legacy-benchmark-reference.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    kwargs=dict(manifest_path=str(manifest_path),source_root=str(source_root),split='train',
        require_complete_library=not(legacy or allow_partial),allow_legacy_benchmark=legacy,
        frame_cache_bytes=16*1024**2,state_cache_bytes=8*1024**2)
    if selected_pool is not None:kwargs['selected_pool']=selected_pool
    prepare_started=time.perf_counter();dataset=R3AStateDataset(**kwargs)
    order=StratifiedEpochSampler(dataset.window_rows,batch_size=16);selected=[order.take_batch()for _ in range(batches)]
    prepare_seconds=time.perf_counter()-prepare_started;began=time.perf_counter();first=None;count=0;batch_times=[]
    with PrefetchBatchLoader(kwargs,workers=2,prefetch_batches=2)as loader:
        for i,result in enumerate(loader.iterate(selected)):
            assert_model_inputs(result['model_inputs'],batch=True)
            if first is None:first=time.perf_counter()-began
            count+=len(result['indices']);batch_times.append(result['worker_seconds'])
            if set(result['model_inputs'])!={'image','action'}:raise ValueError('Privileged state/letter leaked to features')
        wall=time.perf_counter()-began;stats=loader.worker_statistics;max_inflight=loader.maximum_in_flight
    elapsed=time.perf_counter()-began
    result=dict(kind='R3A-two-worker-on-demand-loader-throughput.v1',source_manifest=str(manifest_path),source_manifest_sha256=sha(manifest_path),
        legacy_R2B_benchmark_only=legacy,new_R3A_library_admission_claimed=False if legacy else not allow_partial,
        windows=len(dataset),selected_pool_identity_sha256=None if selected_pool is None else selected_pool['identity_sha256'],
        batch_size=16,batches=batches,samples=count,workers_observed=len(stats),worker_pids=list(stats),
        bounded_prefetch_batches=2,maximum_in_flight_observed=max_inflight,
        frame_RAM_limit_per_worker=16*1024**2,state_RAM_limit_per_worker=8*1024**2,
        worker_statistics=stats,model_input_keys=['image','action'],image_shape='B,5,3,96,96 float32[0,1]',action_shape='B,4,2 float32[-1,1]',
        model_calls=0,checkpoint_loads=0,physics_controls=0,RGB_persistent_frame_bank_written=False,
        preparation_seconds=prepare_seconds,cold_start_to_first_batch_seconds=first,consumption_seconds=wall,
        inclusive_worker_shutdown_seconds=elapsed,samples_per_second_including_start=count/wall,
        steady_samples_per_second_excluding_first_batch=(count-len(selected[0]))/(wall-first)if wall>first else None,
        worker_batch_seconds_mean=float(np.mean(batch_times)),dataset_identity=dataset.identity(),sampler_statistics=order.statistics(),
        renderer_reuse='each spawned process keeps at most six headless original renderer objects; never recreated perbatch',
        measured_scope='saved-state rendering+CPU feature validation+prefetch transport, no training or physics evolution',
        Torch_imported='torch'in sys.modules)
    if len(stats)!=2 or max_inflight>2:raise RuntimeError('Two-worker/bounded-prefetch contract not observed')
    if any(s['frame_cache_bytes']>16*1024**2 or s['state_cache_bytes']>8*1024**2 for s in stats.values()):raise RuntimeError('RAM cache bound exceeded')
    dataset.close();(out/'throughput.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--r2b-root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--benchmark-loader',action='store_true');p.add_argument('--manifest',type=Path);p.add_argument('--batches',type=int,default=32)
    p.add_argument('--allow-partial-library-benchmark',action='store_true')
    a=p.parse_args()
    if a.benchmark_loader:return benchmark_loader(a.source_root.resolve(),a.out.resolve(),manifest_path=a.manifest.resolve()if a.manifest else None,
        r2b_root=a.r2b_root.resolve()if a.r2b_root else None,batches=a.batches,allow_partial=a.allow_partial_library_benchmark)
    if a.r2b_root is None:p.error('--r2b-root required for exact RGB consistency')
    return run(a.source_root.resolve(),a.r2b_root.resolve(),a.out.resolve())

if __name__=='__main__':raise SystemExit(main())
