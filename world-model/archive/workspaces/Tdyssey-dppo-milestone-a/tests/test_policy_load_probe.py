"""Source/trace/fake-counter checks only; never import torch or open weights."""

import ast
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from scripts import probe_c_policy_load as probe


ROOT = Path(__file__).resolve().parents[1]


def fake_constructor(source=None):
    namespace = {}
    source = source or "def Constructor():\n    first = 1\n    second = 2\n    third = 3\n    return third\n"
    exec(compile(source, "fake-loader.py", "exec"), namespace)
    return namespace["Constructor"], {3: "first_done", 4: "second_done", 5: "third_done"}


class FakeRecorder:
    def __init__(self):
        self.stages = []

    def emit(self, stage):
        self.stages.append(stage)


class SourceAndTraceChecks(unittest.TestCase):
    def test_exact_official_source_stages_are_post_operations_in_order(self):
        path = ROOT / "tdyssey/official.py"
        constructor_line, stages = probe._loader_stage_lines(path)
        self.assertGreater(constructor_line, 0)
        self.assertEqual(list(stages.values()), ["mmap_done", "model_constructed",
                                                "strict_load_done", "cuda_transfer_done"])
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertIn('payload["source_checkpoint_sha256"]', lines[next(iter(stages)) - 1])
        expected_next = ["self.network.load_state_dict", "self.network.eval", "self.parameter_count"]
        for line, expected in zip(list(stages)[1:], expected_next):
            self.assertIn(expected, lines[line - 1])

    def test_loader_semantic_changes_fail_before_runtime(self):
        original = (ROOT / "tdyssey/official.py").read_text(encoding="utf-8")
        for before, after in (("mmap=True", "mmap=False"),
                              ("strict=True", "strict=False"),
                              (".to(self.device)", ".to('cpu')")):
            with self.subTest(change=after), tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
                path = Path(directory) / "official.py"
                path.write_text(original.replace(before, after), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "layout changed"):
                    probe._loader_stage_lines(path)

    def test_scoped_trace_observes_one_fake_constructor_and_releases_locals(self):
        constructor, stages = fake_constructor()
        recorder = FakeRecorder()
        with probe._observe_loader(constructor.__code__, stages, recorder) as state:
            self.assertEqual(constructor(), 3)
        self.assertEqual(state["constructor_calls"], 1)
        self.assertEqual(recorder.stages, list(stages.values()))
        self.assertIsNone(sys.gettrace())
        namespace = {}
        exec(compile("def Constructor(value):\n    first = 1\n    second = 2\n    third = 3\n    return None\n",
                     "fake-loader.py", "exec"), namespace)
        value = SimpleNamespace()
        # SimpleNamespace has no weakref support; use a plain Python record.
        class Value:
            pass
        value = Value()
        ref = weakref.ref(value)
        with probe._observe_loader(namespace["Constructor"].__code__, stages, FakeRecorder()):
            namespace["Constructor"](value)
        del value
        gc.collect()
        self.assertIsNone(ref())

    def test_trace_refuses_missing_stages_repeated_constructor_and_existing_tracer(self):
        constructor, stages = fake_constructor()
        with self.assertRaisesRegex(probe.LoadProbeStopped, "complete every"):
            with probe._observe_loader(constructor.__code__, {**stages, 99: "missing"}, FakeRecorder()):
                constructor()
        self.assertIsNone(sys.gettrace())
        with self.assertRaisesRegex(probe.LoadProbeStopped, "Only one"):
            with probe._observe_loader(constructor.__code__, stages, FakeRecorder()):
                constructor()
                constructor()
        self.assertIsNone(sys.gettrace())
        with patch.object(sys, "gettrace", return_value=object()):
            with self.assertRaisesRegex(probe.LoadProbeStopped, "existing Python tracer"):
                with probe._observe_loader(constructor.__code__, stages, FakeRecorder()):
                    self.fail("An existing tracer must prevent entry")

    def test_forward_optimizer_simulator_and_sampler_are_blocked_without_execution(self):
        constructor, stages = fake_constructor()
        for name, path in (("forward", "fake-module.py"),
                           ("step", "C:/site-packages/torch/optim/adam.py"),
                           ("reset", "C:/source/diffusion_policy/env/pusht/pusht_env.py"),
                           ("sample", "C:/source/tdyssey/policy_rl.py")):
            with self.subTest(name=name):
                namespace = {"calls": []}
                exec(compile(f"def {name}():\n    calls.append('forbidden')\n", path, "exec"), namespace)
                with self.assertRaises(probe.LoadProbeStopped):
                    with probe._observe_loader(constructor.__code__, stages, FakeRecorder()):
                        namespace[name]()
                self.assertEqual(namespace["calls"], [])
                self.assertIsNone(sys.gettrace())

    def test_scheduler_step_is_allowed_without_optimizer_step(self):
        constructor, stages = fake_constructor()
        namespace = {"calls": []}
        exec(compile("def step():\n    calls.append('scheduler')\n",
                     "C:/site-packages/torch/optim/lr_scheduler.py", "exec"), namespace)
        with probe._observe_loader(constructor.__code__, stages, FakeRecorder()):
            namespace["step"]()
            constructor()
        self.assertEqual(namespace["calls"], ["scheduler"])

    def test_recorder_fsyncs_primitive_snapshots_before_guard_failure(self):
        calls = []
        memory = SimpleNamespace(
            system_memory=lambda: {"ok": True, "physical": {"available_bytes": 1}},
            process_memory=lambda pid: {"ok": True, "pid": pid, "private_bytes": 7},
            gpu_memory=lambda **kw: calls.append(kw) or {"ok": True, "devices": []},
            guard_reason=lambda system, **kw: calls.append(kw) or "low memory")
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            recorder = probe._Recorder(Path(directory), memory, config_hash="a", loader_sha256="b")
            with patch.object(probe.os, "fsync") as fsync:
                with self.assertRaisesRegex(probe.LoadProbeStopped, "low memory"):
                    recorder.emit("before_import")
                fsync.assert_called_once()
            recorder.close()
            rows = [json.loads(line) for line in (Path(directory) / "stages.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["stage"], "before_import")
            self.assertEqual(rows[0]["guard_reason"], "low memory")
            self.assertEqual(recorder.stages, ["before_import"])
            self.assertEqual(calls, [{"minimum_available_bytes": 1024**3,
                                     "minimum_commit_bytes": 2 * 1024**3}, {"timeout": 3.0}])

    def test_cli_help_and_import_are_source_only(self):
        code = ("import sys; import scripts.probe_c_policy_load; "
                "assert not any(n == 'torch' or n == 'numpy' or n == 'pymunk' "
                "or n == 'tdyssey.official' or n == 'tdyssey.policy_memory' for n in sys.modules)")
        subprocess.run([sys.executable, "-B", "-S", "-c", code], cwd=ROOT,
                       check=True, capture_output=True, text=True)
        result = subprocess.run([sys.executable, "-B", "-S", "scripts/probe_c_policy_load.py", "--help"],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("--source-root", result.stdout)
        tree = ast.parse((ROOT / "scripts/probe_c_policy_load.py").read_text())
        forbidden = {"forward", "sample", "step", "reset", "predict_action", "train_on_batch", "ExperimentBudget"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
                self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()
