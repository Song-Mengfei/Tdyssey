"""C loop/checkpoint integration with fake physics and bounded tiny networks only."""

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tdyssey import policy_checkpoint as checkpoint
    from tdyssey import policy_experiment as experiment
    from tdyssey.policy_protocol import BudgetExceeded, CConfig
    from tdyssey.policy_rl import SamplerConfig, select_trainable_parameters
    from tests.test_policy_stream import collector_fixture, FakeBudget, EXCLUDED, SCENES


_BudgetBase = FakeBudget if torch is not None else object


class ExperimentFakeBudget(_BudgetBase):
    def __init__(self, *, controls=1000, plans=100, resets=100, actor_steps=100,
                 critic_steps=100, batches=100, stop_after_controls=None):
        super().__init__(controls=controls, plans=plans, resets=resets, stop_after_controls=stop_after_controls)
        self.global_limits = {"actor_step": actor_steps, "critic_step": critic_steps, "batch": batches}
        self.reserved["global"] = {kind: 0 for kind in self.global_limits}
        self.completed["global"] = deepcopy(self.reserved["global"])
        for phase in ("development", "diagnostic"):
            self.reserved[phase] = {kind: 0 for kind in self.limits}
            self.completed[phase] = {kind: 0 for kind in self.limits}
        self.experiment_id = "fake-C-permanent"
        self.storage = {}
        self.storage_events = {}
        self.markers = {}
        self.output_dirs = set()
        self.storage_limit = 512 * 1024**2

    def remaining(self, kind, phase="train"):
        if kind in self.global_limits:
            return self.global_limits[kind] - self.reserved["global"][kind]
        return self.limits[kind] - self.reserved[phase][kind]

    def reserve(self, kind, phase="train", shape=None, amount=1):
        self.check_time()
        if self.remaining(kind, phase) < amount:
            raise BudgetExceeded("fake cap")
        phase = "global" if kind in self.global_limits else phase
        self.reserved[phase][kind] += amount
        self.events.append({"kind": kind, "phase": phase, "shape": shape,
                            "amount": amount, "completed": False})
        return len(self.events)

    def snapshot(self):
        return {**super().snapshot(), "experiment_id": self.experiment_id}

    def reserve_storage(self, amount, path=None):
        pending = sum(event["amount"] for event in self.storage_events.values() if not event["written"])
        if pending + sum(self.storage.values()) + amount > self.storage_limit:
            raise BudgetExceeded("fake storage cap")
        event_id = len(self.storage_events) + 1
        self.storage_events[event_id] = {"path": str(Path(path).resolve()), "amount": amount, "written": False}
        return event_id

    def mark_storage_written(self, event_id, actual_bytes):
        event = self.storage_events[event_id]
        assert actual_bytes <= event["amount"]
        event["written"] = True
        self.storage[event["path"]] = actual_bytes
        self.output_dirs.add(Path(event["path"]).parent)
        return {"retire_paths": list(self.storage)[:-2]}

    def record_boundary(self, path, metadata=None):
        path = str(Path(path).resolve())
        self.markers[path] = {"counts": experiment._training_counts(experiment._counts(self)),
                              "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        return {"retire_paths": list(self.storage)[:-2]}

    def validate_boundary(self, path=None):
        path = str(Path(path).resolve())
        marker = self.markers[path]
        if (marker["counts"] != experiment._training_counts(experiment._counts(self))
                or marker["sha256"] != hashlib.sha256(Path(path).read_bytes()).hexdigest()):
            raise ValueError("Fake boundary operations or checkpoint bytes advanced")
        return marker

    def retire_paths(self, paths):
        for path in paths:
            path = str(Path(path).resolve())
            if path not in self.storage or Path(path).parent not in self.output_dirs or not Path(path).match("checkpoint-*.pt"):
                raise ValueError("Cannot retire an unowned path")
            Path(path).unlink()
            del self.storage[path]

    def check_storage(self):
        assert len(self.storage) <= 2
        return {"current_bytes": sum(self.storage.values()), "limit_bytes": self.storage_limit}


class FakeUpdater:
    def __init__(self, collector, budget, *, max_passes=3):
        self.actor, self.critic = collector.actor, collector.critic
        self.budget = budget
        self.run_id = budget.experiment_id
        self.config = SimpleNamespace(max_passes=max_passes, target_kl=.02, cache_features=True,
            gamma=.99, gae_lambda=.95, update=SimpleNamespace(clip=.2, max_grad_norm=1.0,
                plan_microbatch=1, step_microbatch=1, denoising_discount=1.0, initial_logprob_tolerance=1e-4))
        self.actor_optimizer = self.critic_optimizer = None
        self.batches = self.actor_steps = self.critic_steps = 0
        self._active = None
        self.last_buffer = None

    @property
    def actor_version(self):
        return f"{self.budget.experiment_id}:actor:{self.actor_steps}"

    def prepare(self, fragments):
        assert self._active is None
        assert all(plan.sample.actor_version == self.actor_version for fragment in fragments for plan in fragment)
        event = self.budget.reserve("batch")
        self.budget.mark_completed(event)
        buffer = type("FakeBuffer", (), {})()
        buffer.fragments = fragments
        buffer.consumed = False
        self._active = id(buffer)
        self.last_buffer = weakref.ref(buffer)
        self.batches += 1
        return buffer

    def update(self, buffer, *, passes):
        assert self._active == id(buffer) and not buffer.consumed
        report = {"passes": [], "entry": {"max_logprob_error": 0.0}}
        try:
            for index in range(passes):
                actor = self.budget.reserve("actor_step")
                critic = self.budget.reserve("critic_step")
                self.actor_steps += 1
                self.budget.mark_completed(actor)
                self.critic_steps += 1
                self.budget.mark_completed(critic)
                report["passes"].append({"index": index, "actor_grad_norm": 0.2, "critic_grad_norm": 0.1,
                    "critic_mse": 0.1, "critic_explained_variance": None,
                    "after": {"sample_approx_kl": 0.0, "clip_fraction": 0.0}})
            report.update(completed_passes=passes, actor_steps=self.actor_steps, critic_steps=self.critic_steps)
            return report
        finally:
            buffer.consumed = True
            self._active = None

    def state_dict(self):
        assert self._active is None
        return {"actor_steps": self.actor_steps, "critic_steps": self.critic_steps, "batches": self.batches}

    def load_state_dict(self, state):
        assert not self.batches and self._active is None
        self.actor_steps, self.critic_steps, self.batches = (state[key] for key in ("actor_steps", "critic_steps", "batches"))


def loop_fixture(*, budget=None, episode_limit=300):
    budget = budget or ExperimentFakeBudget()
    collector, envs, _, _ = collector_fixture(budget=budget, episode_limit=episode_limit)
    return collector, FakeUpdater(collector, budget), budget, envs


def config_fixture(base=None, *, episode_controls=2):
    data = json.loads((Path(__file__).resolve().parents[1] / "configs/policy_c.example.json").read_text())
    data["scenes"] = {"training": deepcopy(SCENES), "excluded_evaluation": sorted(EXCLUDED)}
    data["evaluation"]["scenes"] = sorted(EXCLUDED)
    data["evaluation"]["new_development_scenes"] = [900001, 900002]
    data["training"]["episode_controls"] = episode_controls
    data["runtime"]["device"] = "cpu"
    if base is not None:
        data["base"] = {"sha256": hashlib.sha256(base.read_bytes()).hexdigest(), "bytes": base.stat().st_size}
    return CConfig(data)


def checkpoint_owners(collector, updater, config):
    """Supply actual tiny optimizer ownership/hyperparameters for fake saves."""
    network = torch.nn.Module()
    network.model = torch.nn.Module()
    network.model.stem = torch.nn.Linear(2, 2)
    network.model.final_conv = torch.nn.Linear(2, 2)
    collector.actor.network = network
    collector.actor.selection = select_trainable_parameters(network, "output")
    collector.actor.config = SamplerConfig(std_floor=config.data["optimization"]["std_floor"])
    collector.critic.fixture_parameter = torch.nn.Parameter(torch.tensor(.1))
    collector.official.network = network
    collector.official.checkpoint_sha256 = config.data["base"]["sha256"]
    updater.actor_optimizer = torch.optim.Adam([p for p in network.parameters() if p.requires_grad],
                                              lr=config.data["optimization"]["actor_lr"])
    updater.critic_optimizer = torch.optim.Adam(collector.critic.parameters(),
                                               lr=config.data["optimization"]["critic_lr"])
    return collector.official


@unittest.skipIf(torch is None, "Optional numpy/torch runtime is unavailable")
class ExperimentLoopChecks(unittest.TestCase):
    def test_fresh_buffers_first_pass_then_multiple_passes_and_no_buffer_retention(self):
        collector, updater, budget, envs = loop_fixture()
        replays, rows = [], []

        def replay(*, fragments, loop_state):
            replays.append([weakref.ref(plan.sample.chain) for fragment in fragments for plan in fragment])
            return {"frames_saved": 2}

        result = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=16,
            max_batches=2, control_target=32, on_replay=replay, log=rows.append)
        self.assertEqual((result["batches"], result["actor_steps"], result["critic_steps"]), (2, 4, 4))
        self.assertEqual(result["controls"], 32)
        self.assertEqual(len(replays), 1)
        self.assertEqual([len(env.reset_seeds) for env in envs.values()], [1, 1])
        self.assertEqual(len([row for row in rows if row["event"] == "ppo_pass"]), 4)
        self.assertIsNone(result["last_update"]["passes"][0]["critic_explained_variance"])
        gc.collect()
        self.assertTrue(all(reference() is None for reference in replays[0]))
        self.assertIsNone(updater.last_buffer())
        self.assertFalse(result["collector"]["resumable"])

    def test_joint_boundary_drain_checkpoint_and_nonresumable_terminal_tail(self):
        collector, updater, budget, envs = loop_fixture(episode_limit=5)
        checkpoints = []

        def checkpoint(**kwargs):
            self.assertIsNone(updater.last_buffer())
            checkpoints.append(deepcopy(kwargs))
            return {"resumable": kwargs["resumable"], "controls": kwargs["loop_state"]["controls"]}

        result = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=6,
            first_batch_passes=1, passes=1, max_batches=8, control_target=12,
            checkpoint_every=1, on_checkpoint=checkpoint)
        self.assertEqual(result["controls"], 12)
        self.assertEqual([(row["resumable"], row["terminal"]) for row in checkpoints], [(True, False), (False, True)])
        self.assertEqual(checkpoints[0]["loop_state"]["controls"], 10)
        self.assertFalse(checkpoints[0]["loop_state"]["checkpoint_pending"])
        self.assertEqual(envs["T"].reset_seeds, [700001, 700002])
        self.assertEqual(envs["L"].reset_seeds, [800001])

    def test_optimizer_budget_reduces_passes_then_stops_without_extra_collection(self):
        collector, updater, budget, envs = loop_fixture(
            budget=ExperimentFakeBudget(actor_steps=2, critic_steps=2))
        rows = []
        result = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            max_batches=5, control_target=100, log=rows.append)
        self.assertEqual((updater.batches, updater.actor_steps), (2, 2))
        self.assertEqual(result["controls"], 4)
        self.assertEqual(result["stop_reason"], "optimizer_budget_exhausted")
        reductions = [row for row in rows if row["event"] == "pass_budget_reduction"]
        self.assertEqual([(row["requested_passes"], row["actual_passes"]) for row in reductions], [(3, 1)])

    def test_resume_diagnostics_and_absolute_job_limits_do_not_restart_first_pass(self):
        collector, updater, budget, _ = loop_fixture(episode_limit=1)
        checkpoints = []
        first = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            max_batches=1, control_target=2, checkpoint_every=1,
            on_checkpoint=lambda **row: checkpoints.append(deepcopy(row)) or {"saved": True})
        state = checkpoints[0]["loop_state"]
        second = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            max_batches=2, control_target=4, checkpoint_every=1, resume_state=state)
        self.assertEqual(second["actor_steps"], 4)
        self.assertEqual(second["controls"], 4)
        self.assertEqual(second["diagnostics_by_shape"]["T"]["controls"], 2)
        self.assertEqual(second["diagnostic_scope"], "whole_job")
        self.assertEqual(second["last_update"]["completed_passes"], 3)

    def test_optional_empty_stop_requires_known_zero_contacts_and_raw_rewards(self):
        collector, updater, budget, _ = loop_fixture()
        original = collector.collect_batch

        def empty_batch(*args):
            fragments = original(*args)
            for fragment in fragments:
                for plan in fragment:
                    for record in plan.control_records:
                        record.update(raw_reward=0.0, n_contacts=0)
            return fragments

        collector.collect_batch = empty_batch
        result = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            max_batches=8, control_target=100, stop_after_empty_batches=2)
        self.assertEqual((result["batches"], result["stop_reason"]), (2, "optional_empty_batch_streak"))
        collector, updater, budget, _ = loop_fixture()
        original = collector.collect_batch

        def unknown_batch(*args):
            fragments = original(*args)
            for fragment in fragments:
                for plan in fragment:
                    for record in plan.control_records:
                        record.update(raw_reward=0.0, n_contacts=None)
            return fragments

        collector.collect_batch = unknown_batch
        result = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            max_batches=3, control_target=6, stop_after_empty_batches=1)
        self.assertEqual(result["batches"], 3)
        self.assertEqual(result["empty_streak"], 0)


@unittest.skipIf(torch is None, "Optional numpy/torch runtime is unavailable")
class ExperimentCheckpointChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = config_fixture()
        self.collector, self.updater, self.budget, _ = loop_fixture(episode_limit=2)
        experiment.train_batches(self.collector, self.updater, self.budget,
            batch_plans=2, batch_controls=4, passes=1, max_batches=1, control_target=4)
        self.official = checkpoint_owners(self.collector, self.updater, self.config)

    def fake_save(self, path, *args, metadata, **kwargs):
        assert self.budget.storage_events
        with Path(path).open("xb") as output:
            torch.save({"metadata": metadata, "rng": {"fake": True}}, output)
        return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    def save(self, path, terminal=False):
        with patch.object(experiment, "save_update_checkpoint", side_effect=self.fake_save):
            return experiment.save_c_checkpoint(path, self.official, self.collector.actor, self.collector.critic,
                self.updater.actor_optimizer, self.updater.critic_optimizer,
                self.collector, self.updater, self.budget, self.config,
                terminal=terminal, metadata={"loop_state": {"test": True}})

    def test_boundary_metadata_and_cross_directory_audit_retention(self):
        first = self.root / "one" / "checkpoint-0001.pt"
        first.parent.mkdir()
        result = self.save(first)
        self.assertTrue(result["resumable"] and result["boundary_recorded"])
        second = self.root / "two" / "checkpoint-0002.pt"
        second.parent.mkdir()
        self.save(second)
        self.collector.collect_batch(1, 1, self.updater.actor_version)
        third = self.root / "three" / "checkpoint-final.pt"
        third.parent.mkdir()
        result = self.save(third, terminal=True)
        self.assertFalse(result["resumable"] or result["boundary_recorded"])
        self.assertFalse(first.exists())
        self.assertTrue(second.exists() and third.exists())
        self.assertEqual(result["retired"], [str(first.resolve())])

    def test_failed_save_keeps_storage_reservation_and_never_marks_boundary(self):
        path = self.root / "checkpoint-failed.pt"
        with patch.object(experiment, "save_update_checkpoint", side_effect=RuntimeError("fake disk failure")):
            with self.assertRaisesRegex(RuntimeError, "disk failure"):
                experiment.save_c_checkpoint(path, self.official, self.collector.actor, self.collector.critic,
                    self.updater.actor_optimizer, self.updater.critic_optimizer,
                    self.collector, self.updater, self.budget, self.config)
        self.assertEqual(len(self.budget.storage_events), 1)
        self.assertFalse(self.budget.storage_events[1]["written"])
        self.assertEqual(self.budget.markers, {})

    def test_live_config_owner_and_optimizer_drift_fail_before_write_or_resume(self):
        from dataclasses import replace
        cases = [
            (self.updater.config, "target_kl", 100.0),
            (self.updater.config, "max_passes", 2),
            (self.updater.config, "gamma", .9),
            (self.updater.config, "cache_features", False),
            (self.updater.config.update, "clip", .3),
            (self.updater.config.update, "max_grad_norm", 2.0),
            (self.updater.config.update, "plan_microbatch", 2),
            (self.updater.config.update, "step_microbatch", 4),
            (self.updater.config.update, "denoising_discount", .9),
            (self.collector.actor, "config", SamplerConfig(std_floor=.02)),
            (self.collector.actor, "config", SimpleNamespace(std_floor=.01, version="unknown")),
            (self.collector.actor, "selection", replace(self.collector.actor.selection, subset="last_up_and_output")),
        ]
        for index, (owner, name, value) in enumerate(cases):
            with self.subTest(name=name, value=value), patch.object(owner, name, value):
                with self.assertRaises(ValueError):
                    self.save(self.root / f"checkpoint-invalid-{index}.pt")
        for field, value in (("lr", .001), ("weight_decay", .1)):
            with self.subTest(field=field), patch.dict(self.updater.actor_optimizer.param_groups[0], {field: value}):
                with self.assertRaisesRegex(ValueError, "learning rate or weight decay"):
                    self.save(self.root / "checkpoint-invalid-opt.pt")
        wrong_owner = torch.optim.Adam(self.collector.actor.network.parameters(), lr=1e-5)
        with self.assertRaisesRegex(ValueError, "owner identities"):
            experiment.save_c_checkpoint(self.root / "checkpoint-foreign-owner.pt", self.official,
                self.collector.actor, self.collector.critic, wrong_owner, self.updater.critic_optimizer,
                self.collector, self.updater, self.budget, self.config)
        self.assertEqual(self.budget.storage_events, {})
        path = self.root / "checkpoint-valid.pt"
        self.save(path)
        fresh, _, _, _ = collector_fixture(budget=self.budget, episode_limit=2)
        updater = FakeUpdater(fresh, self.budget)
        checkpoint_owners(fresh, updater, self.config)
        loaded = SimpleNamespace(actor=fresh.actor, critic=fresh.critic,
                                 payload=torch.load(path, map_location="cpu", weights_only=True))
        with patch.object(updater.config, "target_kl", 100.0), patch.object(experiment, "_restore_rng") as restore:
            with self.assertRaisesRegex(ValueError, "PPO passes/KL"):
                experiment.restore_c_checkpoint(loaded, fresh, updater, self.budget, self.config, path)
            restore.assert_not_called()

    def test_resume_allows_independent_development_but_rejects_train_advance_or_partial_reservation(self):
        path = self.root / "checkpoint-0001.pt"
        self.save(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        budget = deepcopy(self.budget)
        event = budget.reserve("control", phase="development", shape="T")
        budget.mark_completed(event)
        fresh, _, _, _ = collector_fixture(budget=budget, episode_limit=2)
        updater = FakeUpdater(fresh, budget)
        checkpoint_owners(fresh, updater, self.config)
        loaded = SimpleNamespace(actor=fresh.actor, critic=fresh.critic, payload=payload)
        with patch.object(experiment, "_restore_rng") as restore_rng:
            result = experiment.restore_c_checkpoint(loaded, fresh, updater, budget, self.config, path)
            restore_rng.assert_called_once_with(payload["rng"])
        self.assertEqual(updater.batches, 1)
        self.assertEqual(result["loop_state"], {"test": True})
        budget.reserve("control", phase="train", shape="T")
        with self.assertRaisesRegex(ValueError, "boundary operations"):
            experiment.restore_c_checkpoint(loaded, fresh, updater, budget, self.config, path)
        bad_payload = deepcopy(payload)
        bad_payload["metadata"]["counts"]["reserved"]["global"]["actor_step"] += 1
        loaded.payload = bad_payload
        with self.assertRaises(ValueError):
            experiment.restore_c_checkpoint(loaded, fresh, updater, self.budget, self.config, path)

    def test_real_registry_interface_with_fake_collection_and_checkpoint_marker(self):
        from tdyssey.policy_protocol import ExperimentBudget

        class LedgerView:
            def __init__(self, ledger):
                self.ledger = ledger

            @property
            def events(self):
                return self.ledger.snapshot()["recent_events"]

            def __getattr__(self, name):
                return getattr(self.ledger, name)

        run_dir = self.root / "registered-run"
        with ExperimentBudget(self.root / "registry", "fake-registry-integration", self.config,
                              run_dir, create=True) as ledger:
            budget = LedgerView(ledger)
            collector, _, _, _ = collector_fixture(budget=budget, episode_limit=2)
            updater = FakeUpdater(collector, budget)
            official = checkpoint_owners(collector, updater, self.config)
            experiment.train_batches(collector, updater, budget, batch_plans=2,
                batch_controls=4, passes=1, max_batches=1, control_target=4)
            path = run_dir / "checkpoint-0001.pt"

            def write(path, *args, metadata, **kwargs):
                with Path(path).open("xb") as output:
                    torch.save({"metadata": metadata, "rng": {}}, output)
                return {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            with patch.object(experiment, "save_update_checkpoint", side_effect=write):
                saved = experiment.save_c_checkpoint(path, official, collector.actor, collector.critic,
                    updater.actor_optimizer, updater.critic_optimizer, collector, updater, budget, self.config)
            self.assertTrue(saved["resumable"])
            self.assertEqual(ledger.validate_boundary(path)["completed"]["train"]["control"], 4)
            event = ledger.reserve("control", phase="development", shape="L")
            ledger.mark_completed(event)
            fresh, _, _, _ = collector_fixture(budget=budget, episode_limit=2)
            restored = FakeUpdater(fresh, budget)
            checkpoint_owners(fresh, restored, self.config)
            loaded = SimpleNamespace(actor=fresh.actor, critic=fresh.critic,
                payload=torch.load(path, map_location="cpu", weights_only=True))
            with patch.object(experiment, "_restore_rng"):
                experiment.restore_c_checkpoint(loaded, fresh, restored, budget, self.config, path)
            self.assertEqual(restored.batches, 1)

    def test_tiny_checkpoint_optimizer_rng_and_boundary_resume_match_uninterrupted_update(self):
        from tests.test_policy_rl_runtime import actor_fixture as tiny_actor, torch as tiny_torch
        if tiny_torch is None:
            self.skipTest("Tiny diffusion fixture requires installed diffusers")
        from tests.test_policy_stream import FakeEnv, FakeOfficial
        from tdyssey.policy_stream import ContinuousCollector, SceneCursor
        from tdyssey.policy_ppo import MultiPassConfig, PPOUpdater
        from tdyssey.policy_training import UpdateConfig, VisualCritic, make_optimizers

        torch.set_num_threads(4)
        base = self.root / "tiny-base.pt"
        base.write_bytes(b"tiny deterministic frozen base; no research checkpoint")
        config_data = config_fixture(base, episode_controls=1).to_dict()
        config_data["optimization"]["max_passes"] = 1
        config_data["optimization"]["plan_microbatch"] = 2
        config_data["optimization"]["step_microbatch"] = 4
        config = CConfig(config_data)

        def build(budget, actor=None, critic=None):
            actor = actor if actor is not None else tiny_actor()
            critic = critic if critic is not None else VisualCritic(10, hidden_dim=4)
            official = FakeOfficial()
            # This tiny normalizer omits the real upstream dtype conversion;
            # present FP32 toy inputs rather than change production behavior.
            tensorize = official.observation_tensors
            official.observation_tensors = lambda history: {
                key: value.float() for key, value in tensorize(history).items()}
            official.network = actor.network
            official.checkpoint_sha256 = config.data["base"]["sha256"]
            envs = {shape: FakeEnv(shape, budget) for shape in ("T", "L")}
            collector = ContinuousCollector(official, actor, critic, envs,
                {shape: torch.Generator().manual_seed(201 + index) for index, shape in enumerate(("T", "L"))},
                SceneCursor(SCENES, EXCLUDED), budget, episode_limit=1,
                metrics_fn=lambda env: {"coverage": env.coverage, "iou": env.coverage / 2})
            optimizers = make_optimizers(actor, critic)
            updater = PPOUpdater(actor, critic, *optimizers, budget=budget, run_id=budget.experiment_id,
                config=MultiPassConfig(max_passes=1, target_kl=.02,
                    update=UpdateConfig(plan_microbatch=2, step_microbatch=4)))
            return official, collector, updater, optimizers

        budget = ExperimentFakeBudget(actor_steps=2, critic_steps=2, batches=2)
        official, collector, updater, optimizers = build(budget)
        path = self.root / "checkpoint-tiny-0001.pt"

        def save(**row):
            return experiment.save_c_checkpoint(path, official, updater.actor, updater.critic, *optimizers,
                collector, updater, budget, config, terminal=row["terminal"], metadata={"loop_state": row["loop_state"]})

        with patch.object(torch.cuda, "is_available", return_value=False):
            first = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
                first_batch_passes=1, passes=1, max_batches=1, control_target=2,
                checkpoint_every=1, on_checkpoint=save)
        self.assertTrue(first["last_checkpoint"]["resumable"])
        resumed_budget = deepcopy(budget)
        expected_rng = (torch.rand(3), random.random(), np.random.random(3))
        expected = experiment.train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            first_batch_passes=1, passes=1, max_batches=2, control_target=4,
            checkpoint_every=1, resume_state=first["loop_state"])
        expected_actor = {name: value.detach().clone() for name, value in updater.actor.network.state_dict().items()}
        expected_critic = {name: value.detach().clone() for name, value in updater.critic.state_dict().items()}

        def load_official(source_root, checkpoint_path, device, checkpoint_sha256):
            self.assertEqual(checkpoint_path, base)
            self.assertEqual(checkpoint_sha256, config.data["base"]["sha256"])
            return SimpleNamespace(network=tiny_actor().network, checkpoint_sha256=checkpoint_sha256)

        with (patch.object(checkpoint, "OfficialDiffusionPolicy", side_effect=load_official),
              patch.object(torch.cuda, "is_available", return_value=False)):
            loaded = checkpoint.load_update_checkpoint(path, source_root="tiny-fixture", base_checkpoint=base,
                                                       device="cpu", restore_rng=True)
        _, fresh, restored, restored_optimizers = build(resumed_budget, loaded.actor, loaded.critic)
        checkpoint.restore_optimizers(loaded, *restored_optimizers)
        # Simulate construction consuming globals after the initial load: the C
        # helper must restore them last, while its two local generators restore
        # from the collector state rather than from a simulator/contact cache.
        torch.rand(19)
        random.random()
        np.random.random(7)
        metadata = experiment.restore_c_checkpoint(loaded, fresh, restored, resumed_budget, config, path)
        torch.testing.assert_close(torch.rand(3), expected_rng[0], rtol=0, atol=0)
        self.assertEqual(random.random(), expected_rng[1])
        np.testing.assert_array_equal(np.random.random(3), expected_rng[2])
        actual = experiment.train_batches(fresh, restored, resumed_budget, batch_plans=2, batch_controls=2,
            first_batch_passes=1, passes=1, max_batches=2, control_target=4,
            checkpoint_every=1, resume_state=metadata["loop_state"])
        self.assertEqual((actual["controls"], actual["actor_steps"], actual["critic_steps"]), (4, 2, 2))
        self.assertEqual(actual["diagnostics_by_shape"], expected["diagnostics_by_shape"])
        for name, tensor in restored.actor.network.state_dict().items():
            torch.testing.assert_close(tensor, expected_actor[name], rtol=0, atol=0)
        for name, tensor in restored.critic.state_dict().items():
            torch.testing.assert_close(tensor, expected_critic[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
