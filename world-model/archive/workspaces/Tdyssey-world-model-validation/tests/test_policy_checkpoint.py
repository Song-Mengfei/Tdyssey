"""CPU-only delta/RNG/record serialization checks; no assets or environment steps."""

from dataclasses import replace
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tdyssey import policy_checkpoint as checkpoint
    from tdyssey.policy import hash_file
    from tdyssey.policy_rl import SampledChain, SamplerConfig, select_trainable_parameters


def actor_fixture(official=None, *, subset="output", config=SamplerConfig() if torch is not None else None):
    if official is None:
        network = torch.nn.Module()
        network.obs_encoder = torch.nn.Linear(3, 2)
        network.normalizer = torch.nn.Linear(2, 2)
        network.model = torch.nn.Module()
        network.model.stem = torch.nn.Linear(2, 2)
        network.model.final_conv = torch.nn.Linear(2, 2)
    else:
        network = official.network
    return SimpleNamespace(network=network, config=config,
                           selection=select_trainable_parameters(network, subset),
                           scheduler_signature='{"locked_fixture":true}')


def critic_fixture(config=None, device="cpu"):
    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config or {"input_dim": 3, "hidden_dim": 4}
            self.layers = torch.nn.Sequential(torch.nn.Linear(self.config["input_dim"], self.config["hidden_dim"]),
                                               torch.nn.Tanh(), torch.nn.Linear(self.config["hidden_dim"], 1))

    return Critic().to(device)


def optimizers_fixture(actor, critic):
    selected = dict(actor.network.named_parameters())
    parameters = [selected[name] for name in actor.selection.names]
    # Unequal groups with distinct settings detect positional restoration errors.
    actor_optimizer = torch.optim.AdamW([{"params": parameters[:1], "lr": .001},
                                         {"params": parameters[1:], "lr": .002}], weight_decay=0)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=.003)
    for optimizer in (actor_optimizer, critic_optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                # Set a tiny valid state directly: no optimizer.step is run.
                optimizer.state[parameter] = {"step": torch.tensor(2.),
                                              "exp_avg": torch.full_like(parameter, .25),
                                              "exp_avg_sq": torch.full_like(parameter, .5)}
    return actor_optimizer, critic_optimizer


def sample_fixture(mode="training"):
    targets = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
    stds = torch.full((100,), .1)
    if mode == "baseline":
        stds[-1] = 0
    return SampledChain(
        observations={"image": torch.full((1, 2, 3, 96, 96), .5, requires_grad=True),
                      "agent_pos": torch.tensor([[[10., 20.], [11., 21.]]])},
        chain=torch.zeros(1, 101, 16, 2, requires_grad=True), timesteps=torch.arange(99, -1, -1),
        old_means=torch.zeros(1, 100, 16, 2), old_stds=stds,
        old_log_prob=torch.zeros(1, 100) if mode == "training" else None,
        targets=targets, actions=targets[:, 1:9].clone(), sampler_config=SamplerConfig(),
        scheduler_signature='{"fixture":true}', mode=mode, actor_version="fixture-update-1")


@unittest.skipIf(torch is None, "Installed torch/NumPy required for CPU checkpoint checks")
class PolicyCheckpointChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.pt"
        self.base.write_bytes(b"verified tiny fixture; no real model")
        self.identity = {"sha256": hash_file(self.base), "bytes": self.base.stat().st_size}
        # These tests never initialize CUDA or load any research model.
        self.cuda_patch = patch.object(torch.cuda, "is_available", return_value=False)
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)
        torch.set_num_threads(4)

    def save_fixture(self):
        actor, critic = actor_fixture(), critic_fixture()
        optimizers = optimizers_fixture(actor, critic)
        generator = torch.Generator().manual_seed(71)
        path = self.root / "update.pt"
        info = checkpoint.save_update_checkpoint(path, actor, critic, *optimizers,
                                                 base_identity=self.identity, generator=generator,
                                                 metadata={"updates": 1, "purpose": "unit fixture"})
        return path, info, actor, critic, optimizers, generator

    def load_fixture(self, path, restore_rng=False):
        def load_official(source_root, base_checkpoint, device, checkpoint_sha256):
            self.assertEqual(base_checkpoint, self.base)
            self.assertEqual(device, "cpu")
            if hash_file(base_checkpoint) != checkpoint_sha256:
                raise ValueError("Fixture base digest mismatch")
            actor = actor_fixture()
            return SimpleNamespace(network=actor.network, checkpoint_sha256=checkpoint_sha256)

        with (patch.object(checkpoint, "OfficialDiffusionPolicy", side_effect=load_official),
              patch.object(checkpoint, "DenoisingPolicy", side_effect=actor_fixture),
              patch.object(checkpoint, "_construct_critic", side_effect=critic_fixture)):
            return checkpoint.load_update_checkpoint(path, source_root="fixture source", base_checkpoint=self.base,
                                                      device="cpu", restore_rng=restore_rng)

    def test_sample_roundtrip_is_tensor_only_detached_independent_and_preserves_prefix(self):
        for mode in ("training", "baseline"):
            with self.subTest(mode=mode):
                source = sample_fixture(mode)
                payload = checkpoint.sample_to_payload(source)
                path = self.root / f"{mode}-sample.pt"
                torch.save(payload, path)
                safe = torch.load(path, map_location="cpu", weights_only=True)
                restored = checkpoint.sample_from_payload(safe)
                self.assertEqual(restored.sampler_config, source.sampler_config)
                self.assertEqual(restored.mode, mode)
                self.assertEqual(restored.actor_version, source.actor_version)
                for name in ("chain", "timesteps", "old_means", "old_stds", "targets", "actions"):
                    tensor = getattr(restored, name)
                    torch.testing.assert_close(tensor, getattr(source, name), rtol=0, atol=0)
                    self.assertFalse(tensor.requires_grad)
                    self.assertIsNone(tensor.grad_fn)
                    self.assertEqual(tensor.device.type, "cpu")
                if mode == "baseline":
                    self.assertIsNone(restored.old_log_prob)
                else:
                    torch.testing.assert_close(restored.old_log_prob, source.old_log_prob, rtol=0, atol=0)
                for name, tensor in restored.observations.items():
                    torch.testing.assert_close(tensor, source.observations[name], rtol=0, atol=0)
                    self.assertFalse(tensor.requires_grad)
                    self.assertIsNone(tensor.grad_fn)
                with torch.no_grad():
                    source.chain.add_(5)
                self.assertEqual(restored.chain.abs().sum().item(), 0)
                restored.chain.add_(3)
                self.assertEqual(payload["chain"].abs().sum().item(), 0)

    def test_sample_serializer_rejects_inconsistent_training_density_and_action_prefix(self):
        invalid_records = (replace(sample_fixture(), old_log_prob=None),
                           replace(sample_fixture(), old_stds=torch.zeros(100)),
                           replace(sample_fixture(), actions=torch.ones(1, 8, 2)))
        for source in invalid_records:
            with self.subTest(source=source.mode):
                with self.assertRaises(ValueError):
                    checkpoint.sample_to_payload(source)

    def test_delta_reload_matches_selected_weights_critic_optimizer_states_and_rng(self):
        random.seed(19)
        np.random.seed(29)
        torch.manual_seed(39)
        path, info, actor, critic, optimizers, generator = self.save_fixture()
        self.assertEqual(info["bytes"], path.stat().st_size)
        self.assertEqual(info["sha256"], hash_file(path))
        saved = torch.load(path, weights_only=True, map_location="cpu")
        self.assertEqual(set(saved["actor_parameters"]), set(actor.selection.names))
        self.assertFalse(any("obs_encoder" in name or "normalizer" in name or "stem" in name
                             for name in saved["actor_parameters"]))
        self.assertEqual(sum(t.numel() for t in saved["actor_parameters"].values()), 6)
        expected_draws = (torch.rand(4), torch.rand(4, generator=generator), random.random(), np.random.random(4))
        loaded = self.load_fixture(path, restore_rng=True)
        for name, original in saved["actor_parameters"].items():
            torch.testing.assert_close(dict(loaded.actor.network.named_parameters())[name], original, rtol=0, atol=0)
        for name, original in critic.state_dict().items():
            torch.testing.assert_close(loaded.critic.state_dict()[name], original, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4), expected_draws[0], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4, generator=loaded.generator), expected_draws[1], rtol=0, atol=0)
        self.assertEqual(random.random(), expected_draws[2])
        np.testing.assert_array_equal(np.random.random(4), expected_draws[3])
        restored_optimizers = optimizers_fixture(loaded.actor, loaded.critic)
        checkpoint.restore_optimizers(loaded, *restored_optimizers)
        for original, restored in zip(optimizers, restored_optimizers):
            original_state, restored_state = original.state_dict(), restored.state_dict()
            self.assertEqual(original_state["param_groups"], restored_state["param_groups"])
            for parameter_id, state in original_state["state"].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(restored_state["state"][parameter_id][name], tensor, rtol=0, atol=0)
        # CPU artifacts do not contain aliased live tensors.
        with torch.no_grad():
            for parameter in actor.network.parameters():
                parameter.add_(10)
        for name, tensor in loaded.payload["actor_parameters"].items():
            torch.testing.assert_close(tensor, saved["actor_parameters"][name], rtol=0, atol=0)

    def test_refuses_overwrite_and_rejects_wrong_optimizer_parameter_selection(self):
        path, info, actor, critic, optimizers, generator = self.save_fixture()
        with self.assertRaises(FileExistsError):
            checkpoint.save_update_checkpoint(path, actor, critic, *optimizers,
                                               base_identity=self.identity, generator=generator, metadata={})
        self.assertEqual(hash_file(path), info["sha256"])
        wrong_optimizer = torch.optim.SGD(actor.network.parameters(), lr=.1)
        invalid_path = self.root / "invalid.pt"
        with self.assertRaisesRegex(ValueError, "outside"):
            checkpoint.save_update_checkpoint(invalid_path, actor, critic, wrong_optimizer, optimizers[1],
                                               base_identity=self.identity, generator=generator, metadata={})
        self.assertFalse(invalid_path.exists())

    def test_load_rejects_changed_base_schema_selection_and_tensor_shapes(self):
        path, _, _, _, _, _ = self.save_fixture()
        for case in ("schema", "base", "selection", "shape", "official"):
            with self.subTest(case=case):
                payload = torch.load(path, weights_only=True)
                if case == "schema":
                    payload["schema_version"] = 999
                elif case == "base":
                    payload["base_identity"]["sha256"] = "0" * 64
                elif case == "selection":
                    payload["selection"]["counts"] = (1, 1)
                elif case == "shape":
                    payload["actor_parameters"][payload["selection"]["names"][0]] = torch.zeros(1)
                else:
                    payload["official"]["source_commit"] = "0" * 40
                corrupted = self.root / f"{case}.pt"
                torch.save(payload, corrupted)
                with self.assertRaises(ValueError):
                    self.load_fixture(corrupted)

    def test_optimizer_restore_rejects_changed_group_order_before_loading_any_state(self):
        path, _, _, _, _, _ = self.save_fixture()
        loaded = self.load_fixture(path)
        parameters = dict(loaded.actor.network.named_parameters())
        selected = [parameters[name] for name in reversed(loaded.actor.selection.names)]
        actor_optimizer = torch.optim.AdamW([{"params": selected[:1]}, {"params": selected[1:]}], weight_decay=0)
        critic_optimizer = torch.optim.Adam(loaded.critic.parameters(), lr=.003)
        with self.assertRaisesRegex(ValueError, "group order"):
            checkpoint.restore_optimizers(loaded, actor_optimizer, critic_optimizer)
        self.assertFalse(actor_optimizer.state)
        self.assertFalse(critic_optimizer.state)


if __name__ == "__main__":
    unittest.main()
