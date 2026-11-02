"""Focused CPU data contracts; fake renderer only, no physics/model/Torch."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
from tdyssey.native_rgb_r3a_data import (RecordedStateRenderer,R3AStateDataset,StratifiedEpochSampler,
    build_window_catalogue,select_training_pool,export_compact_catalogue,assert_model_inputs,canonical_sha,sha,PROTOCOL_SHA256)
from tdyssey.envs.official import official_shape

ROOT=Path(__file__).resolve().parents[1]


class FakeRenderer:
    def __init__(self,*args,**kwargs):self.cache_bytes=0;self.stats={'rendered_frames':0}
    def render(self,letter,state,**kwargs):
        self.stats['rendered_frames']+=1
        return np.full((3,96,96),int(round(state[0]))%255,dtype=np.uint8)
    def close(self):pass


class DataContracts(unittest.TestCase):
    def fixture(self,folder):
        folder=Path(folder);entries=[]
        for letter in 'TLOCXB':
            for family in range(4):
                number=len(entries);states=np.zeros((49,15),np.float64);states[:,0]=np.arange(49)
                states[:,1]=50+number;states[:,2:4]=(230+number,250)
                if family==0:
                    states[:,4]=np.arange(49)*.15
                    cog=np.asarray(official_shape(letter).centroid);angle=states[:,4]
                    states[:,2]=256-np.cos(angle)*cog[0]+np.sin(angle)*cog[1]
                    states[:,3]=256-np.sin(angle)*cog[0]-np.cos(angle)*cog[1]
                elif family==1:states[:,2]+=np.arange(49)*14
                elif family==2:states[:,2]+=np.arange(49)*.01
                actions=np.column_stack((np.arange(48),np.full(48,300+number))).astype(np.float64)
                path=folder/f'{letter}-{family}.npz'
                np.savez_compressed(path,states=states,actions=actions,control_valid=np.ones(48,bool),control_time_seconds=np.arange(49,dtype=np.float64)*.1)
                entries.append(dict(trajectory_id=f't{number}',family_id=f'f{number}',slot_id=f's{number}',letter=letter,
                    split='seen_val'if family==3 else'train',seed=1000+number,file=path.name,sha256=sha(path),controls=48,protocol_sha256=PROTOCOL_SHA256))
        manifest=dict(kind='R3A-compact-state-reference.v1',source_root='.',entries=entries,protocol_sha256=PROTOCOL_SHA256,control_dt=.1)
        p=folder/'manifest.json';p.write_text(json.dumps(manifest));return p,manifest

    def test_48control_h1_h4_timeline_and_area_centroid_rotation(self):
        with tempfile.TemporaryDirectory(dir=ROOT)as tmp:
            p,m=self.fixture(tmp);catalogue=build_window_catalogue(p,require_complete_library=False)
            self.assertEqual(len(catalogue['h1']),24*45);self.assertEqual(len(catalogue['h4']),24*42)
            first=[r for r in catalogue['h1']if r['trajectory_id']=='t0']
            self.assertEqual([r['anchor']for r in first],list(range(3,48)))
            self.assertTrue(all(r['category']=='rotation_only'and r['translation_image96_px']<1e-10 for r in first))
            with self.assertRaisesRegex(ValueError,'768 TRAIN'):build_window_catalogue(p)

    def test_loader_only_rgb_actions_once_normalized_and_correct_five_frames(self):
        with tempfile.TemporaryDirectory(dir=ROOT)as tmp:
            p,m=self.fixture(tmp);d=R3AStateDataset(p,tmp,require_complete_library=False,renderer_factory=FakeRenderer)
            value=d[0];self.assertEqual(set(value),{'image','action'})
            self.assertEqual(value['image'].shape,(5,3,96,96));self.assertEqual(value['action'].shape,(4,2))
            self.assertEqual(value['image'].dtype,np.float32);self.assertEqual(value['action'].dtype,np.float32)
            np.testing.assert_array_equal(value['image'][:,0,0,0],np.arange(5,dtype=np.float32)/np.float32(255))
            np.testing.assert_array_equal(value['action'][:,0],np.arange(4,dtype=np.float32)/np.float32(255.5)-np.float32(1))
            with self.assertRaisesRegex(ValueError,'only image/action'):assert_model_inputs(dict(value,letter='T'))
            self.assertLessEqual(d.statistics()['state_cache_bytes'],8*1024**2);d.close()
        self.assertNotIn('torch',sys.modules)

    def test_valid_transition_filter_no_cross_split_and_pool_no_rare_both_repeats(self):
        with tempfile.TemporaryDirectory(dir=ROOT)as tmp:
            p,m=self.fixture(tmp);path=Path(tmp)/m['entries'][0]['file']
            with np.load(path)as z:arrays={k:z[k].copy()for k in z.files}
            arrays['control_valid'][5]=False;np.savez_compressed(path,**arrays);m['entries'][0]['sha256']=sha(path);p.write_text(json.dumps(m))
            catalogue=build_window_catalogue(p,require_complete_library=False)
            self.assertTrue(all(not(r['trajectory_id']=='t0'and 5 in range(r['anchor']-3,r['anchor']+r['horizon']))for r in catalogue['h1']+catalogue['h4']))
            pool=select_training_pool(catalogue);indices=pool['selected_h1_training_indices']
            self.assertEqual(len(indices),len(set(indices)));self.assertGreaterEqual(pool['selected_motion_fraction'],.5)
            self.assertTrue(all(catalogue['h1'][i]['split']=='train'for i in indices))
            compact=export_compact_catalogue(catalogue,pool,Path(tmp)/'compact')
            self.assertLess(compact['bytes'],100000)
            with np.load(Path(tmp)/'compact/window-catalogue.npz')as z:self.assertEqual(len(z['selected_h1_training_indices']),pool['selected_windows'])
            bad=deepcopy(pool);bad['selected_h1_training_indices'].append(next(r['dataset_index']for r in catalogue['h1']if r['split']=='seen_val'))
            with self.assertRaisesRegex(ValueError,'validation'):R3AStateDataset(p,tmp,require_complete_library=False,catalogue=catalogue,selected_pool=bad,renderer_factory=FakeRenderer)
            m['entries'][-1]['seed']=m['entries'][0]['seed'];p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'seeds'):build_window_catalogue(p,require_complete_library=False)

    def test_epoch_without_replacement_family_separation_and_exact_resume(self):
        rows=[]
        for family in range(24):
            for anchor in range(7):rows.append(dict(split='train',family_id=f'f{family}',letter='TLOCXB'[family%6],category='both'if family==0 and anchor==0 else'other',anchor=anchor))
        sampler=StratifiedEpochSampler(rows,batch_size=16);flat=[]
        partial=StratifiedEpochSampler(rows,batch_size=16);partial.take_batch();partial_stats=partial.statistics()
        self.assertEqual(partial_stats['unique_windows'],16);self.assertEqual(partial_stats['minimum_exposures'],0)
        self.assertEqual(partial_stats['observed_window_exposure_min'],1);self.assertEqual(partial_stats['unexposed_windows'],len(rows)-16)
        for _ in sampler.batches:
            batch=sampler.take_batch();flat.extend(batch)
            self.assertEqual(len(batch),len({rows[i]['family_id']for i in batch}))
        self.assertEqual(sorted(flat),list(range(len(rows))))
        self.assertEqual(sum(rows[i]['category']=='both'for i in flat),1)
        self.assertEqual(sampler.statistics()['maximum_exposures'],1)
        state=sampler.state_dict();restored=StratifiedEpochSampler(rows,batch_size=16);restored.load_state_dict(state)
        for _ in range(5):self.assertEqual(sampler.take_batch(),restored.take_batch())
        self.assertEqual(sampler.state_dict(),restored.state_dict())
        bad=deepcopy(state);bad['total_exposures']+=1;bad.pop('state_sha256');bad['state_sha256']=canonical_sha(bad)
        with self.assertRaisesRegex(ValueError,'counters'):restored.load_state_dict(bad)

    def test_bounded_frame_cache_and_no_perframe_environment_construction(self):
        renderer=RecordedStateRenderer(ROOT,maximum_frame_cache_bytes=3*96*96)
        current=np.zeros(15,dtype=np.float64)
        class Environment:
            def set_render_state(self,state):current[:]=state
            def state(self):return current.copy()
            def render_clean(self):return np.full((3,96,96),int(current[0]),dtype=np.uint8)
        environment=Environment()
        with patch.object(renderer,'_environment',return_value=environment):
            for index in(1,2,2):
                state=np.zeros(15,np.float64);state[0]=index;renderer.render('T',state,cache_key=str(index))
        self.assertEqual(renderer.stats['rendered_frames'],2);self.assertEqual(renderer.stats['cache_hits'],1)
        self.assertEqual(renderer.cache_bytes,3*96*96)
        self.assertEqual(renderer.stats['physics_steps'],0)
        renderer.close()


if __name__=='__main__':unittest.main()
