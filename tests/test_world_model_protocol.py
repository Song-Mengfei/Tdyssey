"""Source-only model resource limits, causal sampling order and metric counts."""

import unittest

from tdyssey.world_model_data import episode_order
from tdyssey.world_model_protocol import MetricTotals, WorldModelConfig


class WorldModelProtocolChecks(unittest.TestCase):
    def test_resource_and_horizon_limits(self):
        for args in ({'epochs':0},{'future':5},{'batch_size':1025},{'max_steps':10000001},
                     {'seconds':604801},{'seed':True},{'learning_rate':float('nan')}):
            with self.subTest(args=args), self.assertRaises(ValueError):WorldModelConfig(**args)
        self.assertEqual(WorldModelConfig().future,4)

    def test_episode_local_order_is_repeatable_complete_and_cache_friendly(self):
        order=list(episode_order([0,4,4,9],7))
        self.assertEqual(order,list(episode_order([0,4,4,9],7)))
        self.assertEqual(sorted(order),list(range(9)))
        groups=[i>=4 for i in order]
        self.assertEqual(sum(a!=b for a,b in zip(groups,groups[1:])),1)

    def test_aggregation_weights_samples_and_keeps_empty_groups_absent(self):
        totals=MetricTotals(1)
        totals.add(1,'all',2,{'error':6.})
        totals.add(1,'all',1,{'error':9.})
        totals.add(1,'contact',0,{'error':0.})
        result=totals.result()['1']
        self.assertEqual(result['all']['error'],5.)
        self.assertIsNone(result['contact']['error'])
        with self.assertRaises(ValueError):totals.add(1,'all',1,{'error':float('inf')})


class CandidateScoreChecks(unittest.TestCase):
    def test_transient_max_can_hide_final_reward_loss(self):
        from tdyssey.world_model_diagnostics import candidate_score_diagnostics
        rows=[{'name':'push','predicted_rewards':[.4,.5],
               'actual':[{'reward':.4,'success':False},{'reward':.1,'success':False}]},
              {'name':'hold','predicted_rewards':[.4,.4],
               'actual':[{'reward':.4,'success':False},{'reward':.4,'success':False}]}]
        scores=candidate_score_diagnostics(rows,2)
        self.assertEqual(scores['max']['physical_range'],0)
        self.assertAlmostEqual(scores['final']['regret'],.3)
        self.assertEqual(scores['final']['best_physical'],'hold')

    def test_sum_scores_pad_only_true_terminal_success(self):
        from tdyssey.world_model_diagnostics import candidate_score_diagnostics
        rows=[{'name':'success','predicted_rewards':[.9,1,1],
               'actual':[{'reward':1,'success':True}]}]
        score=candidate_score_diagnostics(rows,3)['sum']['rows'][0]
        self.assertEqual(score['physical'],3)
        self.assertAlmostEqual(score['predicted'],2.9)


if __name__=='__main__':unittest.main()
