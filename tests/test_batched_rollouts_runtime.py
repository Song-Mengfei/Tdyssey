"""Batched lock-step rollouts: row-wise sampler parity, replayed groups, evaluation parity."""

from collections import defaultdict
from dataclasses import replace
import os
import unittest


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT"), "Enable official CUDA batched-rollout checks")
class BatchedRolloutChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        from tdyssey.envs.official import make_official_shape_env_class
        from tdyssey.forward_policy import ForwardPolicy
        from tdyssey.forward_protocol import ForwardConfig
        from tdyssey.official import OfficialDiffusionPolicy, load_official_source
        source = os.environ["TDYSSEY_OFFICIAL_ROOT"]
        cls.env_class = make_official_shape_env_class(load_official_source(source))
        cls.config = ForwardConfig(group_size=3, score_chunks=2, warmup_controls=8, max_train_controls=4096,
                                   eval_seeds=(100000,), eval_steps=24, batched_rollouts=True)
        cls.agent = ForwardPolicy(OfficialDiffusionPolicy(source, os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"]),
                                  cls.config)

    def history(self, shape, seed):
        env = self.env_class(shape=shape, legacy=True)
        self.addCleanup(env.close)
        env.seed(seed)
        return [env.reset()]*2

    def test_batched_rows_match_sequential_sampler(self):
        import numpy as np
        import torch
        agent = self.agent
        histories = [self.history("T", 100000), self.history("L", 100001), self.history("L", 100002)]
        conditions = agent.encode_batch(histories)
        for row, history in enumerate(histories):
            torch.testing.assert_close(conditions[row:row+1], agent.encode(history), rtol=1e-5, atol=1e-5)
        # B=1 is the original sampler exactly, including the consumed RNG stream.
        generator = torch.Generator(device=agent.device).manual_seed(7)
        agent.generator.manual_seed(7)
        expected, expected_x = agent.sample(conditions[:1])
        actions, x = agent.sample_batch(conditions[:1], [generator])
        np.testing.assert_array_equal(actions[0], expected)
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), agent.generator.get_state(), rtol=0, atol=0)
        # B=3: each row equals its own sequential sample up to batch-size kernel roundoff
        # (cuDNN may pick other TF32 convolution algorithms; observed <0.04 px of 512).
        generators = [torch.Generator(device=agent.device).manual_seed(s) for s in (11, 12, 13)]
        actions, x = agent.sample_batch(conditions, generators, "old")
        for row, seed in enumerate((11, 12, 13)):
            agent.generator.manual_seed(seed)
            expected, expected_x = agent.sample(conditions[row:row+1], "old")
            np.testing.assert_allclose(actions[row], expected, rtol=0, atol=.25)
            torch.testing.assert_close(x[row:row+1], expected_x, rtol=0, atol=1e-3)
        with self.assertRaises(ValueError):
            agent.sample_batch(conditions, generators[:2])

    def test_batched_group_replays_every_candidate_environment(self):
        from tdyssey.policy_branches import BranchCollector
        counts, report = defaultdict(int), {}
        collector = BranchCollector(self.agent, self.env_class, self.config, counts, report, lambda: None)
        self.addCleanup(collector.close)
        self.assertEqual(len(collector.pools["L"]), 3)
        group = collector.collect("L")
        if group.get("warmup_success"):
            self.skipTest("Warmup reached success; no group to check")
        self.assertEqual(len(group["candidates"]), 3)
        self.assertEqual(len(group["records"]), 3)
        prefix = group["prefix_controls"]
        self.assertEqual(prefix, 8)
        # Two extra environments each restore the initial state and replay the prefix.
        self.assertEqual(counts["initial_state_restores"], 3)
        self.assertEqual(counts["replay_controls"], 2*prefix)
        branch = sum(len(c["steps"]) for c in group["candidates"])
        self.assertEqual(counts["branch_controls"], branch)
        self.assertEqual(counts["training_controls"], counts["context_controls"]+counts["replay_controls"]+branch)
        followups = sum(len(c["steps"]) > 8 for c in group["candidates"])
        self.assertEqual(counts["training_plans"], 1+3+followups)
        for candidate in group["candidates"]:
            self.assertLessEqual(len(candidate["steps"]), 16)
        episode = report["training_episodes"][group["episode"]]
        self.assertEqual(episode["steps"][prefix:], group["candidates"][-1]["steps"])
        # A later group on the advanced episode replays the longer prefix in every pool environment.
        if "L" in collector.active:
            again = collector.collect("L")
            self.assertEqual(again["prefix_controls"], len(episode["steps"]) - len(again["candidates"][-1]["steps"]))

    def test_batched_evaluation_matches_sequential(self):
        import numpy as np
        from tdyssey.policy_training import evaluate
        sequential_counts, batched_counts = defaultdict(int), defaultdict(int)
        sequential = evaluate(self.agent, self.env_class, replace(self.config, batched_rollouts=False),
                              sequential_counts, lambda: None)
        batched = evaluate(self.agent, self.env_class, self.config, batched_counts, lambda: None)
        self.assertEqual([(e["shape"], e["requested_seed"]) for e in sequential],
                         [(e["shape"], e["requested_seed"]) for e in batched])
        self.assertEqual(dict(sequential_counts), dict(batched_counts))
        for a, b in zip(sequential, batched):
            self.assertEqual((a["success"], a["controls"]), (b["success"], b["controls"]))
            np.testing.assert_allclose([r["action"] for r in a["steps"]], [r["action"] for r in b["steps"]],
                                       rtol=0, atol=.25)
            np.testing.assert_allclose(a["final_coverage"], b["final_coverage"], rtol=0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
