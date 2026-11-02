"""Tiny CPU/synthetic end-to-end checks; never a real GPC or simulator."""
from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation as adapter
from tdyssey.gpc_adaptation_runtime import AdaptationBudget, CheckpointStore

EVIDENCE = adapter.ROOT/"artifacts/wm-l-adaptation-preparation-20261025/integration"


class PlanChecks(unittest.TestCase):
    def test_plan_and_missing_data_execute_work_without_tensor_runtime_or_writes(self):
        script = adapter.ROOT/"scripts/adapt_gpc_world_model.py"
        candidate = adapter.repository_path(adapter.read_config()[0]["output_dir"])
        self.assertFalse(candidate.exists())
        for flags, expected in (([], 0), (["--execute"], 2)):
            result = subprocess.run([sys.executable, "-S", "-B", str(script), *flags],
                cwd=adapter.ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, expected, result.stderr)
            payload = json.loads(result.stdout)
            if not flags:
                self.assertFalse(payload["ready"])
                self.assertEqual(payload["native_resource_queries"], 0)
            else:
                self.assertIn("Real startup blocked", payload["reason"])
        self.assertFalse(candidate.exists())

    def test_configuration_rejects_relaxed_selection_caps_or_thresholds(self):
        cfg, _ = adapter.read_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            for section, field in (("train", "selected_update"), ("limits", "forecasts"), ("resources", "native_job_gib")):
                changed = deepcopy(cfg); changed[section][field] += 1
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    adapter.read_config(path)

    def test_plan_blocks_too_few_independent_windows_before_real_entry(self):
        cfg, identity=adapter.read_config()
        data=dict(path="unused",sha256="a"*64,episodes=[],readiness=dict(ready=True,gaps=[],shapes={
            "T":dict(independent_validation_windows=7),"L":dict(independent_validation_windows=8)}))
        with patch("tdyssey.gpc_adaptation_data.read_adaptation_manifest",return_value=data), \
             patch.object(adapter,"read_benchmark",return_value=dict(ready=True,gaps=[])):
            prepared=adapter.plan()
        self.assertFalse(prepared["ready"])
        self.assertTrue(any("T: fewer than eight" in gap for gap in prepared["gaps"]))

    def test_benchmark_metadata_never_opens_payloads_and_rejects_held_file_alias(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root=Path(directory);data_path=root/"data.json"
            data_path.write_text(json.dumps(dict(reviewed_scene_sha256=[])),encoding="utf-8")
            held=root/"held.npz";held.write_bytes(b"synthetic held payload; never read")
            data=dict(path=str(data_path),episodes=[dict(root=str(root),initial_state_sha256="f"*64,
                role="held_model_test",entry=dict(files=[dict(name="held.npz")]))])
            value=dict(kind="tdyssey.gpc_adaptation_benchmark.v1",regression=dict(
                path="artifacts/frozen-wm-validation-v1-20261024",report_sha256=adapter.REPORT_SHA),cases=[])
            truth=dict(pose=dict(position=[30,40],angle=.2),pusher=[10,20],coverage=.2)
            for shape in ("T","L"):
                for n in range(6):
                    anchor=f"I{shape}{n:02d}"
                    for candidate in ("hold","toward","side","away"):
                        content=(anchor+candidate).encode();name=anchor+candidate+".npz"
                        (root/name).write_bytes(content)
                        value["cases"].append(dict(anchor=anchor,candidate=candidate,shape=shape,role="independent",prior_uses=[],
                            initial_state_sha256=hashlib.sha256(anchor.encode()).hexdigest(),goal=dict(position=[30,40],angle=0),
                            latest=truth,truth=[truth]*4,contact_by_h4=False,
                            payload=dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest())))
            manifest=root/"benchmark.json";manifest.write_text(json.dumps(value),encoding="utf-8")
            original=Path.open
            def guarded(path,*args,**kwargs):
                if path.suffix==".npz":raise AssertionError("Payload opened during metadata readiness")
                return original(path,*args,**kwargs)
            with patch.object(Path,"open",guarded):
                self.assertTrue(adapter.read_benchmark(manifest,data)["ready"])
                value["cases"][0]["payload"]["path"]="held.npz"
                manifest.write_text(json.dumps(value),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"including held test"):
                    adapter.read_benchmark(manifest,data)


class Clock:
    wall = 100.
    mono = 50.
    def time(self): return self.wall
    def monotonic(self): return self.mono


class TinyDataset:
    def __init__(self, *args, **kwargs): pass
    def shape_indices(self, shape): return (0, 2, 4, 6) if shape == "T" else (1, 3, 5, 7)
    def __getitem__(self, index):
        import numpy as np
        image = np.full((5, 3, 96, 96), .1+index*.001, dtype=np.float32)
        image[4] += .05
        return dict(image=image, action=np.full((4, 2), index*.01, dtype=np.float32))


class FakeSampler:
    def __init__(self, model=None): self.counter=0; self.seen=[]
    def capture(self): return self.counter.to_bytes(8, "little")
    def restore(self, value): self.counter=int.from_bytes(value, "little")
    def seed(self, value): self.counter=value
    def sample(self, context, actions):
        import numpy as np
        self.seen.append((context.copy(), actions.copy())); self.counter += 1
        return np.full((3,96,96), 1.25, dtype=np.float32)


def tiny_case():
    import numpy as np
    return dict(anchor="IT00", candidate="hold", shape="T", role="independent", goal={}, truth=[{}]*4,
        history=np.full((4,3,96,96), .3, dtype=np.float32), actual=np.full((4,3,96,96), .4, dtype=np.float32),
        past_actions=np.array([[0,1],[2,3],[4,5]], dtype=np.float32),
        future_actions=np.array([[6,7],[8,9],[10,11],[12,13]], dtype=np.float32))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Installed Torch required for tiny CPU substitutes")
class TinyIntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from test_gpc_supervised_adaptation import TinyDenoiser
        cls.torch, cls.TinyDenoiser = torch, TinyDenoiser
        torch.set_num_threads(2)

    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=EVIDENCE, prefix="tiny-")
        self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name); self.clock=Clock()
        self.cfg = deepcopy(adapter.read_config()[0])
        self.cfg["train"].update(batch_size=4, max_updates=2, max_examples=5, checkpoint_every=1,
            selected_update=2, phase_seconds=9)
        self.cfg["evaluation"].update(loss_examples_per_role=2, seeds=[5478])
        self.cfg["limits"].update(eval_batches=4, eval_examples=4, forecasts=12, euler_steps=36,
            checkpoint_bytes=2*1024**2, output_bytes=16*1024**2, metadata_bytes=1024**2)

    def bundle(self):
        from tdyssey.gpc_supervised_adaptation import make_generator, make_optimizer, select_trainable_parameters
        model=self.TinyDenoiser(); select_trainable_parameters(model)
        optimizer=make_optimizer(model); generator=make_generator(5478)
        budget=AdaptationBudget.create(self.root/"ledger", dict(candidate="tiny", config="fixed"),
            max_updates=2, max_eval_batches=4, max_batch_size=4, max_train_examples=5, max_eval_examples=4,
            max_forecasts=12, max_euler_steps=36, deadline_seconds=10,
            clock=self.clock.time, monotonic=self.clock.monotonic)
        op=budget.reserve_model_load("train_init"); budget.complete_model_load(op)
        store=CheckpointStore(self.root/"latest.pt", budget, dict(config="tiny"), max_bytes=2*1024**2,
            generators={"training":generator})
        return model, optimizer, generator, budget, store

    def train(self, bundle, **kwargs):
        model, optimizer, generator, budget, store=bundle
        return adapter.train_candidate(model, optimizer, TinyDataset(), cfg=self.cfg, budget=budget,
            store=store, generator=generator, monotonic=self.clock.monotonic, **kwargs)

    def test_short_final_batch_updates_then_unique_final_checkpoint_selection(self):
        bundle=self.bundle(); before=bundle[0].inner_model.output_scale.detach().clone(); events=[]
        result=self.train(bundle, metrics=lambda step, values:events.append(values))
        state=bundle[3].snapshot()
        self.assertEqual([row["batch_size"] for row in events], [4,1])
        self.assertEqual([row["indices"] for row in events], [[0,1,2,3],[4]])
        self.assertEqual((state["spent_train_examples"], state["completed_updates"],state["selection_count"]),(5,2,1))
        self.assertEqual(result["selected_checkpoint"]["sha256"],state["selected_checkpoint_sha256"])
        self.assertFalse(self.torch.equal(before,bundle[0].inner_model.output_scale))
        with self.assertRaises(Exception): bundle[3].reserve_update(1)

    def test_cap_smaller_than_one_requested_batch_uses_only_actual_examples(self):
        bundle=self.bundle(); self.cfg["train"].update(max_examples=1,max_updates=1,selected_update=1)
        result=self.train(bundle)
        self.assertEqual(result["completed_updates"],1)
        self.assertEqual(bundle[3].snapshot()["spent_train_examples"],1)

    def test_slow_selection_publication_cannot_extend_training_deadline(self):
        bundle=self.bundle(); original=bundle[3].select_checkpoint
        def slow(digest):
            result=original(digest);self.clock.mono=59.;return result
        with patch.object(bundle[3],"select_checkpoint",slow):
            with self.assertRaisesRegex(TimeoutError,"Training phase deadline"):self.train(bundle)
        self.assertEqual(bundle[3].snapshot()["status"],"stopped")
        self.assertEqual(bundle[3].snapshot()["completed_forecasts"],0)

    def test_forward_exception_spends_first_update_and_never_retries(self):
        bundle=self.bundle()
        with patch.object(bundle[0],"compute_model_output",side_effect=RuntimeError("injected forward")) as call:
            with self.assertRaisesRegex(RuntimeError,"injected forward"): self.train(bundle)
        self.assertEqual(call.call_count,1)
        state=bundle[3].snapshot()
        self.assertEqual((state["spent_reserved_updates"],state["completed_updates"],state["selection_count"]),(1,0,0))
        self.assertEqual(state["status"],"stopped")

    def test_training_deadline_during_first_step_does_not_complete_or_select(self):
        bundle=self.bundle()
        def expire(*args): self.clock.mono=59.
        with self.assertRaisesRegex(TimeoutError,"Training phase deadline"): self.train(bundle,progress=expire)
        state=bundle[3].snapshot()
        self.assertEqual((state["spent_reserved_updates"],state["completed_updates"],state["selection_count"]),(1,0,0))

    def test_insufficient_examples_cannot_select_an_earlier_saved_prefix(self):
        bundle=self.bundle(); self.cfg["train"].update(max_examples=4, selected_update=2)
        with self.assertRaisesRegex(RuntimeError,"incomplete"): self.train(bundle)
        state=bundle[3].snapshot()
        self.assertEqual((state["checkpoint_committed_updates"],state["selection_count"]),(1,0))

    def test_matched_validation_rng_and_raw_ar_tf_action_windows(self):
        import numpy as np
        model, optimizer, generator, budget, store=self.bundle(); self.train((model,optimizer,generator,budget,store))
        train_rng=generator.get_state().clone(); global_rng=self.torch.get_rng_state().clone()
        matched={}; losses={}; all_rows={}; samplers={}
        for role in ("frozen","adapted"):
            op=budget.reserve_model_load(role)
            if role=="adapted":store.load_selected_for_evaluation(model)
            budget.complete_model_load(op)
            losses[role]=adapter.evaluate_loss(model,TinyDataset(),cfg=self.cfg,budget=budget,role=role)
            sampler=FakeSampler(); samplers[role]=sampler
            all_rows[role]=adapter.rollout_case(tiny_case(),5478,role=role,sampler=sampler,rng=sampler,budget=budget,matched=matched)
            self.assertEqual(sampler.counter,0)
        self.assertEqual([r["noise"] for r in losses["frozen"]],[r["noise"] for r in losses["adapted"]])
        self.assertTrue(self.torch.equal(train_rng,generator.get_state()))
        self.assertTrue(self.torch.equal(global_rng,self.torch.get_rng_state()))
        self.assertTrue(np.all(samplers["frozen"].seen[1][0][-1]==np.float32(1.25)))
        self.assertTrue(np.all(samplers["frozen"].seen[4][0][-1]==np.float32(.4)))
        expected=np.array([[2,3],[4,5],[6,7],[8,9]],dtype=np.float32)/np.float32(511)*2-1
        np.testing.assert_array_equal(samplers["frozen"].seen[1][1],expected)
        self.assertEqual([r["rng_entry"] for r in all_rows["frozen"]],[r["rng_entry"] for r in all_rows["adapted"]])
        self.assertEqual((budget.snapshot()["completed_forecasts"],budget.snapshot()["completed_euler_steps"]),(12,36))

    def test_complete_injected_cpu_pipeline_uses_three_serial_models(self):
        import gc
        import weakref
        cfg=self.cfg; cfg["train"].update(batch_size=1,max_examples=2)
        prepared=dict(config=cfg,data=dict(path="synthetic-only"))
        loaded=[]; alive=[]
        def factory(role):
            gc.collect(); self.assertFalse(any(ref() is not None for ref in alive))
            model=self.TinyDenoiser(); alive.append(weakref.ref(model)); loaded.append(role); return model
        empty=lambda *args,**kwargs:{}
        with patch("tdyssey.gpc_adaptation_data.GPCAdaptationDataset",TinyDataset), \
             patch("tdyssey.gpc_adaptation_data.fit_train_stats",return_value=dict(synthetic=True)), \
             patch.object(adapter,"load_cases",return_value=[tiny_case()]), \
             patch("tdyssey.wm_image_measurement_v2.decode",empty):
            report=adapter.run_candidate(prepared,dict(config="tiny"),factory=factory,close_model=gc.collect,
                make_sampler=FakeSampler,writer=adapter.OutputWriter(self.root,cfg["limits"]),
                clock=self.clock.time,monotonic=self.clock.monotonic)
        self.assertEqual(loaded,["train_init","frozen","adapted"])
        self.assertEqual((report["status"],report["forecast_records"]),("complete",12))
        self.assertEqual(report["budget"]["completed_updates"],2)
        self.assertEqual(report["physical_operations"],0)
        self.assertTrue((self.root/"completion.json").is_file())

    def test_worker_heartbeat_contract_with_fake_ownership_and_cpu_pipeline(self):
        spec=importlib.util.spec_from_file_location("adaptation_entry_fixture",adapter.ROOT/"scripts/adapt_gpc_world_model.py")
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        cfg=self.cfg;cfg["train"].update(batch_size=1,max_examples=2)
        prepared=dict(config=cfg,config_sha256="a"*64,data=dict(path="synthetic-only",sha256="b"*64),
            benchmark=dict(sha256="c"*64),ready=True)
        handoff=dict(config_path="synthetic-config",binding=dict(config_sha256="a"*64,data_sha256="b"*64,
            benchmark_sha256="c"*64),nonce="fake-owned",gpc_root="never-import",checkpoint="never-load")
        published=[]
        class Heartbeat:
            def __init__(self,path,payload,refresh):self.payload,self.refresh=payload,refresh
            def __enter__(self):self.update();return self
            def __exit__(self,*args):return False
            def update(self,**values):
                self.payload.update(values);self.payload.update(self.refresh());published.append(deepcopy(self.payload))
            def check(self):pass
        cuda=self.torch.cuda
        with ExitStack() as stack:
            stack.enter_context(patch("tdyssey.gpc_adaptation_guard.verify_worker",return_value=handoff))
            stack.enter_context(patch("tdyssey.gpc_adaptation_guard.Heartbeat",Heartbeat))
            stack.enter_context(patch.object(entry,"plan",return_value=prepared))
            stack.enter_context(patch.object(entry,"repository_path",return_value=self.root))
            stack.enter_context(patch.object(entry,"NativeSampler",lambda model,check:FakeSampler(model)))
            stack.enter_context(patch("tdyssey.gpc_supervised_adaptation.make_real_denoiser",side_effect=lambda *a,**k:self.TinyDenoiser()))
            stack.enter_context(patch("tdyssey.gpc_adaptation_data.GPCAdaptationDataset",TinyDataset))
            stack.enter_context(patch("tdyssey.gpc_adaptation_data.fit_train_stats",return_value=dict(synthetic=True)))
            stack.enter_context(patch.object(adapter,"load_cases",return_value=[tiny_case()]))
            stack.enter_context(patch("tdyssey.wm_image_measurement_v2.decode",return_value={}))
            stack.enter_context(patch.object(cuda,"is_available",return_value=True))
            for name in ("reset_peak_memory_stats","synchronize","empty_cache","memory_allocated","memory_reserved",
                    "max_memory_allocated","max_memory_reserved"):
                stack.enter_context(patch.object(cuda,name,return_value=0))
            with redirect_stdout(io.StringIO()):self.assertEqual(entry.worker("synthetic-handoff"),0)
        self.assertEqual(published[0]["phase"],"setup")
        complete=next(row for row in reversed(published) if row["phase"]=="complete")
        self.assertTrue(complete["training_complete"])
        self.assertEqual((complete["optimizer_updates"],complete["model_loads"],complete["forecasts"],complete["eval_batches"]),(2,3,12,4))
        self.assertEqual(published[-1]["model_instances"],0)
        self.assertFalse(cuda.is_initialized())
