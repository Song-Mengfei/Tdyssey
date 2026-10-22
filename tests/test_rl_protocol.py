"""Independent telescoping identities and physical-time boundary examples."""

import unittest

from tdyssey.rl_protocol import advantages, block_return, shaped_reward


class RLProtocolChecks(unittest.TestCase):
    def test_shaping_telescopes_with_terminal_potential_and_preserves_success_return(self):
        gamma = .9
        coverage = [.1, .3, .8, .96]
        rewards = [shaped_reward(a, b, i == 2, gamma, 2.)
                   for i, (a, b) in enumerate(zip(coverage, coverage[1:]))]
        self.assertAlmostEqual(block_return(rewards, gamma), gamma**2-2*coverage[0])
        # Timeout is not absorbing: its final potential remains in the identity.
        rewards = [shaped_reward(a, b, False, gamma, 2.)
                   for a, b in zip(coverage, coverage[1:])]
        self.assertAlmostEqual(block_return(rewards, gamma),
                               2*(gamma**3*coverage[-1]-coverage[0]))

    def test_gae_bootstraps_timeout_but_never_crosses_reset(self):
        records = [dict(controls=2, reward=1., value=.5, next_value=2.,
                        terminated=False, episode_end=True),
                   dict(controls=1, reward=100., value=0., next_value=999.,
                        terminated=True, episode_end=True)]
        adv, returns = advantages(records, .9, .8)
        self.assertAlmostEqual(adv[0], 1+.9**2*2-.5)
        self.assertAlmostEqual(returns[0], 1+.9**2*2)
        self.assertEqual(adv[1], 100.)

    def test_discount_uses_actual_short_prefix_and_physical_lambda(self):
        records = [dict(controls=3, reward=block_return([1., 2., 3.], .9),
                        value=0., next_value=4., terminated=False, episode_end=False),
                   dict(controls=1, reward=2., value=4., next_value=0.,
                        terminated=True, episode_end=True)]
        adv, _ = advantages(records, .9, .8)
        self.assertAlmostEqual(adv[0], 1+.9*2+.9**2*3+.9**3*4+(.9*.8)**3*(-2))


if __name__ == "__main__":
    unittest.main()
