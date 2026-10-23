"""Synthetic supervised dynamics fixtures, CUDA updates and held-out quarantine."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from tdyssey.collection import CollectedEpisode, CollectionConfig
from tdyssey.dataset import DatasetWriter
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.official_collection import ENVIRONMENT_FAMILY, OfficialCollectionConfig
from tdyssey.state import BodyState, Snapshot
from tdyssey.training_data import WindowSpec, write_catalog
from tdyssey.trajectories import Trajectory, Transition
from tdyssey.world_model_data import VisualDynamicsDataset, fit_motion_stats, wrap_angle
from tdyssey.world_model_protocol import WorldModelConfig


@unittest.skipUnless(os.environ.get('TDYSSEY_DATA_TESTS') or os.environ.get('TDYSSEY_GPC_ROOT'),
                     'Enable installed data/model checks')
class WorldModelRuntimeChecks(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        torch.set_num_threads(2)
        self.np,self.torch=np,torch
        self.temporary=tempfile.TemporaryDirectory(prefix='tdyssey-model-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.catalog=self.fixture()

    def fixture(self, shape='L', suffix=''):
        np=self.np;root=self.root/('data'+suffix)
        writer=DatasetWriter(root,OfficialCollectionConfig(shape=shape,episodes=3,steps=8),
                             {'environment_family':ENVIRONMENT_FAMILY,'kind':'synthetic markers, not physical data'})
        writer.manifest['rates']={'simulation_hz':100,'control_hz':10}
        splits={'train':[],'validation':[],'model_test':[]}
        for i,split in enumerate(splits):
            dx=2 if split=='train' else 10
            def state(t):
                return Snapshot(shape,official_shape(shape).geometry_hash,'a'*64,
                    BodyState(Pose((200+dx*t,240+i*5),wrap_angle(math.pi-.1+.2*t))),
                    BodyState(Pose((80+t,100),0)),Pose((300,300),.3),t,None if t==0 else (130+t-1,110))
            transitions=tuple(Transition(state(t),(130+t,110),state(t+1),t/10,False,t==7) for t in range(8))
            trajectory=Trajectory(f'marker-{i}{suffix}','fixture','v1',i,.1,transitions)
            steps=[dict(n_contacts=t%2,iou=0.,coverage=0.,upstream_terminated=False,
                        upstream_truncated=False,collector_truncated=t==7) for t in range(8)]
            images=np.stack([np.full((3,96,96),20+t,np.uint8) for t in range(9)])
            episode=CollectedEpisode(trajectory,images,np.array([state(t).pusher_body.pose.position for t in range(9)]),
                np.array([(*state(t).goal.position,state(t).goal.angle) for t in range(9)]),steps,
                [{'seed':i,'issues':[]}],i,i,'collection_horizon')
            writer.add(episode);splits[split].append(trajectory.trajectory_id)
        writer.manifest['counts']={'control_steps':24,'control_physics_substeps':240,'reset_attempts':3,'reset_physics_substeps':3}
        writer.finish(splits)
        catalog=self.root/('catalog'+suffix+'.json');write_catalog([root],catalog)
        return catalog

    def dataset(self,split='train',future=2):
        return VisualDynamicsDataset(self.catalog,split,WindowSpec(2,future))

    def test_labels_are_correctly_timed_and_normalization_is_train_only(self):
        dataset=self.dataset();sample=dataset[0]
        self.np.testing.assert_allclose(sample['labels']['motion'],[[1,0,2,0,.2]]*2,atol=1e-6)
        self.np.testing.assert_allclose(sample['labels']['object_displacement'],[[2,0,.2],[4,0,.4]],atol=1e-6)
        self.assertEqual(sample['labels']['object_roi'].shape,(2,1,96,96))
        self.assertTrue(sample['labels']['object_roi'].any())
        self.assertNotIn('object_displacement',sample['observations'])
        stats=fit_motion_stats(dataset)
        self.assertEqual(stats['unique_transitions'],8)
        self.assertAlmostEqual(stats['motion_mean'][2],2)
        self.assertEqual(stats['episode_ids'],['marker-0'])
        with self.assertRaisesRegex(ValueError,'training episodes only'):fit_motion_stats(self.dataset('validation'))

    def batch(self):
        return self.torch.utils.data.default_collate([self.dataset()[0],self.dataset()[1]])

    def test_mixed_shape_labels_use_each_episodes_geometry_without_actor_shape_id(self):
        self.fixture('Y', '-Y')
        catalog=self.root/'mixed.json'
        write_catalog([self.root/'data',self.root/'data-Y'],catalog)
        dataset=VisualDynamicsDataset(catalog,'train',WindowSpec(2,2))
        samples={}
        for index in range(len(dataset)):
            sample=dataset[index]
            samples.setdefault(sample['shape'],sample)
        self.assertEqual(set(samples),{'L','Y'})
        self.assertFalse(self.np.array_equal(samples['L']['labels']['object_roi'],
                                           samples['Y']['labels']['object_roi']))
        for sample in samples.values():
            self.assertNotIn('shape',sample['observations'])
            self.np.testing.assert_allclose(sample['labels']['motion'][:,2],[2.,2.])
        self.assertEqual(fit_motion_stats(dataset)['unique_transitions'],16)

    def test_autoregression_is_causal_and_differentiable_through_first_action(self):
        torch=self.torch
        from tdyssey.models.visual_dynamics import VisualDynamics
        model=VisualDynamics(fit_motion_stats(self.dataset()))
        with torch.no_grad():
            model.image_head.weight.normal_(0,.01)
            model.motion_head[-1].weight.normal_(0,.001)
        batch=self.batch();actions=batch['actions'].clone().requires_grad_(True)
        first=model(batch['observations'],actions)
        changed=actions.detach().clone();changed[:,1]+=30
        second=model(batch['observations'],changed)
        torch.testing.assert_close(first['image'][:,0],second['image'][:,0],rtol=0,atol=0)
        gradient=torch.autograd.grad(first['image'][:,1].mean(),actions)[0]
        self.assertGreater(float(gradient[:,0].abs().sum()),0)
        self.assertEqual(first['image'].shape,(2,2,3,96,96))

    def test_persistence_metrics_keep_physical_units_and_foreground_error(self):
        torch=self.torch
        from tdyssey.world_model_training import persistence,metric_values
        batch=self.batch();stats=fit_motion_stats(self.dataset())
        predicted=persistence(batch,stats)
        metrics=metric_values(predicted,batch)
        torch.testing.assert_close(metrics['object_position_error_pixels'],torch.tensor([[2.,4.]]*2))
        torch.testing.assert_close(metrics['agent_position_error_pixels'],torch.tensor([[1.,2.]]*2))
        torch.testing.assert_close(metrics['object_angle_error_degrees'],torch.tensor([[.2,.4]]*2)*180/torch.pi)
        self.assertTrue(torch.isfinite(metrics['object_roi_mse']).all())

    def test_training_updates_on_cuda_and_does_not_read_model_test_images(self):
        torch=self.torch
        if not torch.cuda.is_available():self.skipTest('CUDA required for optimizer check')
        from tdyssey.world_model_training import load_checkpoint,train
        # Corrupt the quarantined model-test payload. Training/validation must
        # succeed without decoding it; catalog metadata still remains intact.
        (self.root/'data/episode-0002.npz').write_bytes(b'unread model-test payload')
        config=WorldModelConfig(epochs=1,max_steps=1,batch_size=2,future=2,seconds=90)
        report=train(self.catalog,self.root/'model',config)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['steps'],1)
        self.assertFalse(report['model_test_used'])
        model,payload=load_checkpoint(self.root/'model/best.pt',torch.device('cuda'))
        self.assertTrue(all(p.device.type=='cuda' for p in model.parameters()))
        self.assertGreater(float(model.image_head.weight.norm()),0)
        self.assertEqual(payload['catalog_sha256'],report['catalog_sha256'])
        with self.assertRaisesRegex(ValueError,'hash/size'):self.dataset('model_test')[0]


if __name__=='__main__':unittest.main()
