"""Independent physical-control reward/return examples; no simulator or torch."""

import math
import unittest

from tdyssey.policy_returns import (
    ActionBlock,
    BlockBoundary,
    discounted_block_return,
    plan_gae,
    training_reward,
)


class TrainingRewardChecks(unittest.TestCase):
    def test_strict_threshold_raw_metrics_negative_shaping_and_first_bonus(self):
        threshold = training_reward(0.90, 0.95, 1.0)
        self.assertFalse(threshold.success)
        self.assertFalse(threshold.terminated)
        self.assertAlmostEqual(threshold.reward, 0.99 * 0.95 - 0.90)
        self.assertEqual(threshold.next_potential, 0.95)
        self.assertEqual(threshold.upstream_reward, 1.0)
        falling = training_reward(0.80, 0.20, 0.37)
        self.assertAlmostEqual(falling.reward, -0.602)
        self.assertEqual((falling.current_coverage, falling.next_coverage,
                          falling.upstream_reward), (0.80, 0.20, 0.37))
        success = training_reward(0.94, 0.951, 1.0)
        self.assertTrue(success.success and success.first_success and success.terminated)
        self.assertEqual(success.next_coverage, 0.951)
        self.assertEqual(success.next_potential, 0.0)
        self.assertAlmostEqual(success.reward, 0.06)
        repeated = training_reward(0.94, 0.951, 1.0, success_already_seen=True)
        self.assertFalse(repeated.first_success)
        self.assertAlmostEqual(repeated.reward, -0.94)

    def test_third_control_success_telescopes_and_stops_the_block(self):
        # These three physical controls stop an eight-target plan at success.
        # Terminal potential=0 makes the shaped discounted sum telescope.
        records = [training_reward(0.2, 0.4, 0.4 / 0.95),
                   training_reward(0.4, 0.7, 0.7 / 0.95),
                   training_reward(0.7, 0.96, 1.0)]
        rewards = tuple(record.reward for record in records)
        self.assertEqual([record.terminated for record in records], [False, False, True])
        self.assertAlmostEqual(discounted_block_return(rewards), 0.99 ** 2 - 0.2)
        block = ActionBlock(episode_id="T-success", rewards=rewards, value=0.3,
                            final_observation_value=10000.0, boundary=BlockBoundary.SUCCESS)
        result, = plan_gae([block])
        self.assertEqual(block.num_controls, 3)
        self.assertEqual((block.bootstrap_mask, block.continuation_mask), (0, 0))
        self.assertAlmostEqual(result.physical_discount, 0.99 ** 3)
        self.assertAlmostEqual(result.delta, 0.99 ** 2 - 0.2 - 0.3)
        self.assertAlmostEqual(result.value_target, 0.99 ** 2 - 0.2)

    def test_timeout_and_buffer_cutoff_keep_final_observation_potential(self):
        record = training_reward(0.4, 0.7, 0.7 / 0.95, gamma=0.9, beta=2.0)
        self.assertFalse(record.terminated)
        self.assertEqual(record.next_potential, 0.7)
        self.assertAlmostEqual(record.reward, 0.46)
        for boundary in (BlockBoundary.TIMEOUT, BlockBoundary.BUFFER_CUTOFF):
            with self.subTest(boundary=boundary):
                block = ActionBlock(episode_id="L-final", rewards=(record.reward,),
                                    value=0.2, final_observation_value=0.8,
                                    boundary=boundary)
                result, = plan_gae([block], gamma=0.9)
                self.assertEqual((block.bootstrap_mask, block.continuation_mask), (1, 0))
                self.assertAlmostEqual(result.delta, 0.46 + 0.9 * 0.8 - 0.2)

    def test_reward_rejects_invalid_metric_and_configuration(self):
        for current, next_value, upstream, kwargs in (
            (-0.1, 0.5, 0.2, {}), (0.1, 1.1, 0.2, {}),
            (0.1, 0.5, math.nan, {}), (True, 0.5, 0.2, {}),
            (0.1, 0.5, 0.2, {"gamma": 1.1}),
            (0.1, 0.5, 0.2, {"beta": -1}),
            (0.1, 0.5, 0.2, {"success_already_seen": 1}),
        ):
            with self.subTest(current=current, next_value=next_value, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    training_reward(current, next_value, upstream, **kwargs)


class PlanReturnChecks(unittest.TestCase):
    def test_five_control_final_prefix_and_plan_boundary_lambda(self):
        # Independent hand arithmetic, gamma=.5 and lambda=.8. The second
        # plan has only five remaining real controls, rather than eight.
        blocks = [
            ActionBlock(episode_id="L-1", rewards=(1.0, 2.0), value=0.25,
                        final_observation_value=0.75, boundary=BlockBoundary.CONTINUE),
            ActionBlock(episode_id="L-1", rewards=(1.0,) * 5, value=0.75,
                        final_observation_value=4.0, boundary=BlockBoundary.BUFFER_CUTOFF),
        ]
        first, tail = plan_gae(blocks, gamma=0.5, gae_lambda=0.8)
        # R_tail=1+.5+.25+.125+.0625=1.9375; gamma**5=.03125.
        self.assertAlmostEqual(tail.discounted_reward, 1.9375)
        self.assertAlmostEqual(tail.physical_discount, 0.03125)
        self.assertAlmostEqual(tail.delta, 1.3125)
        # delta_first=1+.5*2+.25*.75-.25=1.9375.
        self.assertAlmostEqual(first.delta, 1.9375)
        self.assertAlmostEqual(first.advantage, 2.2)
        self.assertAlmostEqual(first.value_target, 2.45)
        self.assertNotAlmostEqual(first.advantage, first.delta + (0.5 * 0.8) ** 2 * tail.delta)
        self.assertEqual(sum(block.num_controls for block in blocks), 7)

    def test_mixed_T_L_success_timeout_reset_and_buffer_cutoff(self):
        # Numeric values chosen so every boundary has a different bootstrap.
        # The timeout's final value 8 belongs to old T, never L's reset value 3.
        blocks = [
            ActionBlock(episode_id="T-1", rewards=(1.0, 1.0), value=2.0,
                        final_observation_value=8.0, boundary=BlockBoundary.TIMEOUT),
            ActionBlock(episode_id="L-1", rewards=(4.0,), value=3.0,
                        final_observation_value=None, boundary=BlockBoundary.SUCCESS),
            ActionBlock(episode_id="T-2", rewards=(-1.0, 2.0, 4.0), value=1.0,
                        final_observation_value=5.0, boundary=BlockBoundary.CONTINUE),
            ActionBlock(episode_id="T-2", rewards=(2.0,), value=5.0,
                        final_observation_value=6.0, boundary=BlockBoundary.BUFFER_CUTOFF),
        ]
        results = plan_gae(blocks, gamma=0.5, gae_lambda=0.8)
        # deltas: (1+.5)+.25*8-2=1.5; 4-3=1;
        # (-1+.5*2+.25*4)+.125*5-1=.625; 2+.5*6-5=0.
        for result, expected in zip(results, (1.5, 1.0, 0.625, 0.0)):
            self.assertAlmostEqual(result.delta, expected)
            self.assertAlmostEqual(result.advantage, expected)
        self.assertEqual([(block.bootstrap_mask, block.continuation_mask)
                          for block in blocks], [(1, 0), (0, 0), (1, 1), (1, 0)])
        # A completely different next episode cannot alter a previous timeout.
        changed_reset = ActionBlock(episode_id="L-1", rewards=(100.0,), value=-50.0,
                                    final_observation_value=None, boundary=BlockBoundary.SUCCESS)
        changed = plan_gae([blocks[0], changed_reset], gamma=0.5, gae_lambda=0.8)
        self.assertAlmostEqual(changed[0].advantage, results[0].advantage)

    def test_default_discount_lambda_and_per_stream_independence(self):
        first = ActionBlock(episode_id="T", rewards=(0.1,) * 8, value=0.2,
                            final_observation_value=0.3, boundary=BlockBoundary.CONTINUE)
        tail = ActionBlock(episode_id="T", rewards=(0.7,), value=0.3,
                           final_observation_value=None, boundary=BlockBoundary.SUCCESS)
        results = plan_gae([first, tail])
        # Geometric-series formula is independent of discounted_block_return.
        reward = 0.1 * (1.0 - 0.99 ** 8) / (1.0 - 0.99)
        self.assertAlmostEqual(results[0].discounted_reward, reward)
        self.assertAlmostEqual(results[0].advantage,
                               reward + 0.99 ** 8 * 0.3 - 0.2 + 0.99 ** 8 * 0.95 * 0.4)
        alone, = plan_gae([tail])
        self.assertEqual(alone, results[1])

    def test_zero_length_nonfinite_and_missing_final_value_are_rejected(self):
        for rewards in ((), (math.inf,), (True,)):
            with self.subTest(rewards=rewards):
                with self.assertRaises(ValueError):
                    discounted_block_return(rewards)
                with self.assertRaises(ValueError):
                    ActionBlock(episode_id="T", rewards=rewards, value=0.0,
                                final_observation_value=0.0, boundary=BlockBoundary.BUFFER_CUTOFF)
        with self.assertRaisesRegex(ValueError, "pre-reset"):
            ActionBlock(episode_id="L", rewards=(0.0,), value=0.0,
                        final_observation_value=None, boundary=BlockBoundary.TIMEOUT)
        with self.assertRaises(ValueError):
            plan_gae([])
        tail = ActionBlock(episode_id="T", rewards=(0.0,), value=0.0,
                           final_observation_value=0.0, boundary=BlockBoundary.BUFFER_CUTOFF)
        with self.assertRaises(ValueError):
            plan_gae([tail], gae_lambda=1.01)

    def test_episode_join_and_unmarked_tail_are_rejected(self):
        continued = ActionBlock(episode_id="T", rewards=(0.0,), value=0.0,
                                final_observation_value=0.0, boundary=BlockBoundary.CONTINUE)
        cutoff = ActionBlock(episode_id="L", rewards=(0.0,), value=0.0,
                             final_observation_value=0.0, boundary=BlockBoundary.BUFFER_CUTOFF)
        timeout = ActionBlock(episode_id="L", rewards=(0.0,), value=0.0,
                              final_observation_value=0.0, boundary=BlockBoundary.TIMEOUT)
        for blocks, message in (([continued], "buffer tail"),
                                ([continued, cutoff], "episode change"),
                                ([timeout, cutoff], "new episode_id"),
                                ([cutoff, timeout], "final block")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    plan_gae(blocks)
        repeated = ActionBlock(episode_id="T", rewards=(0.0,), value=0.0,
                               final_observation_value=0.0, boundary=BlockBoundary.TIMEOUT)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            plan_gae([repeated, timeout, repeated])


if __name__ == "__main__":
    unittest.main()
