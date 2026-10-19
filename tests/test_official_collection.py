"""Source-only official collection limits, evaluation separation and split checks."""

from dataclasses import replace
import unittest

from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.official_collection import DEVELOPMENT_SEEDS, OfficialCollectionConfig, freeze_splits
from tdyssey.state import BodyState, Snapshot
from tdyssey.trajectories import Trajectory, Transition


def trajectory(index):
    state = Snapshot('L', official_shape('L').geometry_hash, 'a'*64,
        BodyState(Pose((200+index,200),0)),BodyState(Pose((100,100),0)),Pose((300,300),0),0)
    after = replace(state, control_step=1, pusher_target=(110,100))
    return Trajectory(f'L-{index}','object-directed','test',index,.1,(Transition(state,(110,100),after,0,False,True),))


class OfficialCollectionChecks(unittest.TestCase):
    def test_scope_and_resource_limits_preserve_development_exclusions(self):
        config=OfficialCollectionConfig()
        self.assertEqual((config.episodes,config.steps),(8,100))
        self.assertTrue(set(DEVELOPMENT_SEEDS).issubset(config.excluded_reset_seeds))
        for changes in ({'shape':'I'},{'seconds':0},{'seconds':601},{'episodes':20,'steps':100},
                        {'excluded_reset_seeds':()}, {'excluded_reset_seeds':(100000,100001,100002,True)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                OfficialCollectionConfig(**changes)

    def test_frozen_whole_episode_splits_are_nonempty_disjoint_and_repeatable(self):
        rows=[trajectory(i) for i in range(8)]
        a=freeze_splits(rows,7)
        self.assertEqual(a,freeze_splits(rows[::-1],7))
        self.assertEqual({k:len(v) for k,v in a.items()},{'train':6,'validation':1,'model_test':1})
        assigned=[value for ids in a.values() for value in ids]
        self.assertEqual(len(set(assigned)),8)
        with self.assertRaisesRegex(ValueError,'nonempty'):
            freeze_splits(rows[:2],7)

    def test_repeated_initial_scene_cannot_cross_frozen_splits(self):
        rows=[trajectory(i) for i in range(8)]
        rows[-1]=replace(rows[0],trajectory_id='different-id')
        with self.assertRaisesRegex(ValueError,'Repeated initial'):
            freeze_splits(rows,7)


if __name__=='__main__':
    unittest.main()
