"""Real official play frames/state, terminal handling, split and family checks."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from tdyssey.collection import derived_seed
from tdyssey.dataset import DatasetWriter, load_dataset
from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.official import load_official_source
from tdyssey.official_collection import (ENVIRONMENT_FAMILY, OfficialCollectionConfig,
    collect_official_episode, freeze_splits, validate_official_dataset)
from tdyssey.training_data import WindowDataset, WindowSpec, read_catalog, write_catalog


@unittest.skipUnless(os.environ.get('TDYSSEY_OFFICIAL_ROOT'),'Enable installed official simulator tests')
class OfficialCollectionRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key,value in {'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1'}.items():
            os.environ.setdefault(key,value)
        import numpy as np
        cls.np=np
        cls.cls=make_official_shape_env_class(load_official_source(Path(os.environ['TDYSSEY_OFFICIAL_ROOT'])))

    def env(self):
        env=self.cls(shape='L',legacy=True)
        self.addCleanup(env.close)
        return env

    def counts(self):
        return dict.fromkeys(('reset_attempts','reset_physics_substeps','control_steps','control_physics_substeps'),0)

    def write(self,root,config,env):
        writer=DatasetWriter(root,config,{'environment_family':ENVIRONMENT_FAMILY})
        writer.manifest['rates']={'simulation_hz':100,'control_hz':10}
        episodes=[]
        for index in range(config.episodes):
            episode=collect_official_episode(env,config,index,writer.manifest['counts'])
            writer.add(episode);episodes.append(episode)
        writer.finish(freeze_splits([e.trajectory for e in episodes],config.seed))
        return episodes

    def test_roundtrip_frames_reset_state_and_all_split_windows(self):
        config=OfficialCollectionConfig(steps=8)
        env=self.env()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'official'
            episodes=self.write(root,config,env)
            manifest,loaded=validate_official_dataset(root)
            self.assertEqual(manifest['counts']['control_steps'],64)
            self.assertEqual({k:len(v) for k,v in manifest['splits'].items()},{'train':6,'validation':1,'model_test':1})
            for saved,(trajectory,record,arrays) in zip(episodes,loaded):
                self.assertEqual(saved.trajectory,trajectory)
                self.assertEqual(arrays['images'].shape,(9,3,96,96))
                self.assertEqual(trajectory.transitions[0].before.geometry_hash,official_shape('L').geometry_hash)
                obs=env.restore_state(trajectory.transitions[0].before)
                self.assertLessEqual(float(abs(arrays['images'][0].astype(float)/255-obs['image']).max()),.5/255+1e-7)
                self.assertTrue(trajectory.transitions[-1].truncated)
                self.assertFalse(record['step_diagnostics'][-1]['upstream_truncated'])
                self.assertNotIn(record['reset_seed'],config.excluded_reset_seeds)
            catalog=Path(temporary)/'catalog.json'
            write_catalog([root],catalog)
            for split in ('train','validation','model_test'):
                dataset=WindowDataset(catalog,split,WindowSpec(2,4))
                self.assertGreater(len(dataset),0)
                self.assertEqual(dataset[0]['targets']['image'].shape,(4,3,96,96))
                self.assertEqual(dataset[0]['actions'].shape,(4,2))
            record=json.loads((root/'episode-0000.json').read_text())
            self.assertIn('upstream_state',record['reset_attempts'][-1])

    def test_excluded_candidates_do_not_simulate_or_break_saved_accounting(self):
        config=OfficialCollectionConfig(steps=4)
        env=self.env()
        def seeds(seed,*parts):
            if 'reset' in parts and parts[-1]==0:
                return 100000
            return derived_seed(seed,*parts)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'data'
            with patch('tdyssey.official_collection.derived_seed',side_effect=seeds):
                self.write(root,config,env)
            manifest,loaded=validate_official_dataset(root)
            self.assertEqual(manifest['counts']['excluded_seed_candidates'],8)
            self.assertEqual(manifest['counts']['reset_attempts'],manifest['counts']['reset_physics_substeps'])
            self.assertEqual(manifest['counts']['reset_attempts'],sum(sum(x['simulated'] for x in record['reset_attempts']) for _,record,_ in loaded))
            self.assertTrue(all(record['reset_attempts'][0]['issues']==['excluded_evaluation_scene'] for _,record,_ in loaded))

    def test_unsafe_reset_rejection_is_counted_and_reproducible(self):
        config=OfficialCollectionConfig(episodes=1,steps=8)
        env=self.env();original=env.reset
        attempted=[]
        def reset():
            obs=original()
            if not attempted:env.agent.position=(10,100)
            attempted.append(True)
            return obs
        counts=self.counts()
        with patch.object(env,'reset',side_effect=reset):
            first=collect_official_episode(env,config,0,counts)
        self.assertGreaterEqual(counts['reset_attempts'],2)
        self.assertIn('pusher_outside_walls',first.resets[0]['issues'])
        attempted.clear()
        with patch.object(env,'reset',side_effect=reset):
            second=collect_official_episode(env,config,0,self.counts())
        self.assertEqual(first.trajectory,second.trajectory)
        self.np.testing.assert_array_equal(first.images,second.images)

    def test_original_done_is_terminal_and_time_budget_has_no_extra_step(self):
        env=self.env();config=OfficialCollectionConfig(episodes=1,steps=8)
        original=env.step
        def terminal(action):
            obs,reward,_,info=original(action)
            return obs,reward,True,info
        counts=self.counts()
        with patch.object(env,'step',side_effect=terminal):
            episode=collect_official_episode(env,config,0,counts)
        self.assertEqual(counts['control_steps'],1)
        self.assertEqual(len(episode.images),2)
        self.assertTrue(episode.trajectory.transitions[-1].terminated)
        self.assertFalse(episode.trajectory.transitions[-1].truncated)
        counts=self.counts()
        with self.assertRaises(TimeoutError):
            collect_official_episode(env,config,0,counts,deadline=time.monotonic()-1)
        self.assertEqual(sum(counts.values()),0)

    def test_official_and_reference_families_cannot_be_combined(self):
        config=OfficialCollectionConfig(steps=4)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'official';other=Path(temporary)/'reference'
            self.write(root,config,self.env());shutil.copytree(root,other)
            manifest=json.loads((other/'manifest.json').read_text())
            manifest['provenance']['environment_family']='gpc-reference-v1'
            (other/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'mix official'):
                write_catalog([root,other],Path(temporary)/'mixed.json')
            with self.assertRaisesRegex(ValueError,'not GPC'):
                validate_official_dataset(other)
            path=Path(temporary)/'catalog.json';write_catalog([root],path)
            catalog=json.loads(path.read_text());catalog['environment_family']='gpc-reference-v1'
            path.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError,'family differs'):
                read_catalog(path)


if __name__=='__main__':
    unittest.main()
