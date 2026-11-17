"""Independent loss derivatives, official CUDA updates and exact checkpoint RNG."""

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT"), "Enable official CUDA forward-RL checks")
class ForwardRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.set_num_threads(2)

    def test_branch_loss_derivatives_and_uncertainty_neutral_anchor(self):
        import torch
        from tdyssey.forward_policy import nft_loss, advantage_flow_loss
        old = torch.tensor([[[2., -1.]], [[.5, -.2]]], dtype=torch.float64)
        target = torch.tensor([[[0., 1.]], [[-1., .3]]], dtype=torch.float64)
        value = old.clone().requires_grad_()
        r = torch.tensor([1., 0.], dtype=torch.float64)
        confidence = torch.tensor([.3, 0.], dtype=torch.float64)
        loss = nft_loss(value, old, target, r, .5, confidence)
        derivative = torch.autograd.grad(loss, value, create_graph=True)[0]
        expected = 2*.5*(confidence*(2*r-1))[:, None, None]*(old-target)/value.numel()
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=1e-12)
        # Full Hessian is positive even for negative/neutral feedback.
        hessian = torch.autograd.functional.hessian(lambda x: nft_loss(x, old, target, r, .5), value)
        torch.testing.assert_close(hessian.reshape(4, 4), torch.eye(4, dtype=torch.float64)*.125)
        neutral = nft_loss(value+1, old, target, r, .5, torch.zeros(2))
        self.assertGreater(float(neutral), float(nft_loss(value, old, target, r, .5, torch.zeros(2))))
        adv = torch.tensor([-1., 1.], dtype=torch.float64)
        anchor = torch.tensor([1.01, 1.01], dtype=torch.float64)
        hessian = torch.autograd.functional.hessian(
            lambda x: advantage_flow_loss(x, old, target, adv, anchor), value).reshape(4, 4)
        self.assertGreater(float(torch.linalg.eigvalsh(hessian).min()), 0)

    def test_official_parity_two_objectives_distributed_update_and_reload(self):
        import numpy as np
        import torch
        # Isolate checkpoint/RNG reconstruction from nondeterministic convolution
        # backward reductions. Ordinary training does not force this backend flag.
        previous_determinism = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        self.addCleanup(setattr, torch.backends.cudnn, 'deterministic', previous_determinism)
        from tdyssey.forward_policy import ForwardPolicy, forward_update, forward_noise
        from tdyssey.forward_protocol import ForwardConfig
        from tdyssey.official import OfficialDiffusionPolicy, load_official_source
        from tdyssey.policy_training import frozen_digest
        from tdyssey.forward_training import check_resume
        source = os.environ["TDYSSEY_OFFICIAL_ROOT"]
        official = OfficialDiffusionPolicy(source, os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"])
        env = load_official_source(source)(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        history = [env.reset()]*2
        official.reset_seed(123)
        expected = official.predict_actions(history)
        expected_rng = official.generator.get_state()
        config = ForwardConfig(minibatch_size=1)
        agent = ForwardPolicy(official, config)
        condition = agent.encode(history)
        agent.generator.manual_seed(123)
        actions, first = agent.sample(condition)
        np.testing.assert_array_equal(actions, expected)
        torch.testing.assert_close(agent.generator.get_state(), expected_rng, rtol=0, atol=0)
        # Fresh analytic noise matches the scheduler formula, including t=0/99.
        noise = torch.ones_like(first).repeat(2, 1, 1)
        clean = first.repeat(2, 1, 1)
        t = torch.tensor([0, 99], device=agent.device)
        scheduler = agent.network.noise_scheduler
        original_alpha = scheduler.alphas_cumprod.clone()
        abar = original_alpha.to(agent.device)[t][:, None, None]
        torch.testing.assert_close(forward_noise(scheduler, clean, noise, t),
                                   abar.sqrt()*clean+(1-abar).sqrt()*noise)
        self.assertEqual(scheduler.alphas_cumprod.device.type, 'cpu')
        torch.testing.assert_close(scheduler.alphas_cumprod, original_alpha, rtol=0, atol=0)
        # A forward training query must not change an unchanged sampler's actions.
        agent.generator.manual_seed(123)
        actual, _ = agent.sample(condition)
        np.testing.assert_array_equal(actual, expected)
        _, second = agent.sample(condition)
        records = [{"condition": condition, "plan": plan, "optimality": r,
                    "advantage": 2*r-1, "anchor": 1.01, "reference_plan": first if r else None}
                   for plan, r in ((first, 1.), (second, 0.))]
        frozen = frozen_digest(agent.network)
        from tdyssey.denoising_probe import DenoisingProbe
        diagnostic = DenoisingProbe()
        plan_rng, noise_rng, order_rng = (agent.generator.get_state(),
            agent.noise_generator.get_state(), agent.order_generator.get_state())
        global_rng = torch.get_rng_state()
        diagnostic.capture('T',100000,condition,first)
        diagnostic.capture('T',100001,condition,second)
        self.assertEqual(len(diagnostic.rows),1)
        initial_probe = diagnostic.evaluate(agent)
        self.assertEqual(initial_probe['probe/T/initial_policy_drift_mse'],0)
        self.assertEqual(initial_probe,diagnostic.evaluate(agent))
        for before, after in ((plan_rng,agent.generator.get_state()),
                (noise_rng,agent.noise_generator.get_state()),
                (order_rng,agent.order_generator.get_state()),(global_rng,torch.get_rng_state())):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        initial = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.actor_lr)
        for objective in ("nft", "advantage_flow"):
            agent.config = replace(config, objective=objective)
            metrics = forward_update(agent, records, optimizer)
            self.assertTrue(any(m["gradient_norm"] > 0 for m in metrics))
        self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in agent.actor.state_dict().items()))
        self.assertGreater(diagnostic.evaluate(agent)['probe/T/initial_policy_drift_mse'],0)
        self.assertEqual(frozen, frozen_digest(agent.network))
        for name, value in agent.old.state_dict().items():
            torch.testing.assert_close(value, agent.actor.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in agent.old.parameters()))
        with tempfile.TemporaryDirectory(prefix="tdyssey-forward-") as root:
            path = Path(root)/"last.pt"
            payload = agent.state()
            payload.update(optimizer=optimizer.state_dict(), training_state={"completed_updates": 1,
                "active_contexts": {}, "report": {"source_sha256": {"source": "hash"},
                    "counts": {"training_controls": 32}}})
            torch.save(payload, path)
            self.assertLess(path.stat().st_size, 128*1024**2)
            # Resume must reproduce the very next noise/time/permutation and update.
            expected, _ = agent.sample(condition)
            expected_metrics = forward_update(agent, records, optimizer)
            expected_weights = {k: v.clone() for k, v in agent.actor.state_dict().items()}
            # Match runner resume: CPU loading preserves Adam's CPU step scalar;
            # load_state_dict moves parameter-shaped moments to the CUDA actor.
            restored = torch.load(path, map_location="cpu", weights_only=True)
            agent.load(restored)
            optimizer.load_state_dict(restored["optimizer"])
            self.assertTrue(all(s['step'].device.type == 'cpu' for s in optimizer.state.values()))
            torch.testing.assert_close(agent.noise_generator.get_state(), restored["noise_rng"].cpu(), rtol=0, atol=0)
            torch.testing.assert_close(agent.order_generator.get_state(), restored["order_rng"].cpu(), rtol=0, atol=0)
            actual, _ = agent.sample(condition)
            np.testing.assert_array_equal(actual, expected)
            actual_metrics = forward_update(agent, records, optimizer)
            self.assertEqual([m["noise_timesteps"] for m in actual_metrics],
                             [m["noise_timesteps"] for m in expected_metrics])
            for name, value in agent.actor.state_dict().items():
                # CUDA backward reductions can differ in final-roundoff bits.
                torch.testing.assert_close(value, expected_weights[name], rtol=1e-6, atol=1e-9)
            check_resume(restored, replace(agent.config, updates=2), {"source": "hash"})
            with self.assertRaises(ValueError):
                check_resume(restored, replace(agent.config, gamma=.9), {"source": "hash"})
            # Test the saved policy in a newly constructed sampler, not just an
            # existing scheduler whose mutable state could mask reload bugs.
            agent.load(restored)
            expected, _ = agent.sample(condition)
            fresh = ForwardPolicy(OfficialDiffusionPolicy(source,
                os.environ['TDYSSEY_OFFICIAL_CHECKPOINT']), agent.config)
            fresh.load(restored)
            actual, _ = fresh.sample(fresh.encode(history))
            np.testing.assert_array_equal(actual, expected)

    def test_dense_head_view_restoration_and_ema_rollout(self):
        import torch
        from tdyssey.forward_policy import ForwardPolicy
        from tdyssey.forward_protocol import ForwardConfig
        from tdyssey.official import OfficialDiffusionPolicy
        official = OfficialDiffusionPolicy(os.environ["TDYSSEY_OFFICIAL_ROOT"],
                                            os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"])
        agent = ForwardPolicy(official, ForwardConfig(trainable="head", rollout_ema=.5))
        first = next(agent.actor.parameters())
        old = next(agent.old.parameters()).clone()
        with torch.no_grad():
            first.add_(.1)
        agent.refresh_old()
        torch.testing.assert_close(next(agent.old.parameters()), old+.05)
        with self.assertRaises(RuntimeError):
            with agent.use("reference"):
                raise RuntimeError("restore on error")
        self.assertIs(agent.network.model.final_conv, agent.actor)

    def test_resume_charges_discarded_physical_work_without_advancing_logical_context(self):
        import json
        from tdyssey.forward_training import audit_resume_costs
        from tdyssey.policy import hash_file
        keys=("training_controls", "branch_controls", "replay_controls", "context_controls",
              "training_plans", "reference_plans", "initial_state_restores", "evaluation_controls",
              "evaluation_plans", "reset_physics_substeps", "optimizer_minibatches",
              "checkpoint_verification_plans", "context_attempts")
        report={"source_sha256":{"source":"hash"},"counts":dict.fromkeys(keys,0),
                "sessions":[{"elapsed_seconds":10.}]}
        report['counts'].update(training_controls=32,branch_controls=32,context_attempts=2)
        with tempfile.TemporaryDirectory(prefix='tdyssey-resume-costs-') as root:
            path=Path(root)/'last.pt';path.write_bytes(b'cost-audit fixture, not model weights')
            latest=json.loads(json.dumps(report))
            latest.update(kind='official_forward_rl_pilot',status='failed',checkpoint={'sha256':hash_file(path)})
            latest['counts'].update(training_controls=48,branch_controls=40,replay_controls=8,context_attempts=3)
            latest['sessions'][0]['elapsed_seconds']=14.
            (Path(root)/'report.json').write_text(json.dumps(latest))
            audit_resume_costs(path,report)
            self.assertEqual(report['counts']['training_controls'],48)
            self.assertEqual(report['counts']['context_attempts'],2)
            audit=report['resume_cost_audits'][0]
            self.assertTrue(audit['post_boundary_costs_known'])
            self.assertEqual(audit['additional_elapsed_seconds'],4.)


if __name__ == "__main__":
    unittest.main()
