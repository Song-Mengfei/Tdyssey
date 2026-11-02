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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--r2b-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();return run(a.source_root.resolve(),a.r2b_root.resolve(),a.out.resolve())

if __name__=='__main__':raise SystemExit(main())
