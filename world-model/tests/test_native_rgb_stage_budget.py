"""CPU/mock stage3 accounting/callback/checkpoint tests; no optimizer or GPU."""

import ast
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey import native_rgb_training as training
from tdyssey.native_rgb_stage_budget import ModelWorkBudget,StorageBudget,UpdateLimitReached,storage_footprint


def temporary_directory():
    parent=Path(__file__).resolve().parents[1]/"artifacts"
    parent.mkdir(exist_ok=True)
    temporary=tempfile.TemporaryDirectory(prefix="native-rgb-stage3-cpu-",dir=parent)
    if not Path(temporary.name).resolve().is_relative_to(parent.resolve()):raise RuntimeError("Test cleanup target escaped workspace")
    return temporary


class Clock:
    def __init__(self):self.now=1000.
    def __call__(self):return self.now
    def advance(self,value):self.now+=value


class FakeParameter:
    def __init__(self):self.requires_grad=True
    def requires_grad_(self,value):self.requires_grad=value;return self


class FakeModel:
    def __init__(self):self.training=True;self.parameter=FakeParameter();self.weight="original"
    def parameters(self):return iter([self.parameter])
    def eval(self):self.training=False
    def train(self,value=True):self.training=value
    def state_dict(self):return dict(weight=[1.,2.])
    def load_state_dict(self,value,strict=True):self.restored=value


class StageBudgetTests(unittest.TestCase):
    def test_durable_stage_attempt_cap_200_never_changes_global3000(self):
        clock=Clock()
        with temporary_directory() as directory,patch.object(training.time,"monotonic",clock):
            ledger=training.CostLedger(Path(directory)/"ledger.json",3000,5400,legacy_cost=dict(old_native_attempts=1899,old_native_seconds=5352.516))
            stage=ledger.start("stage3");budget=ModelWorkBudget(stage)
            for i in range(200):
                budget.reserve_update()
                if i<199:budget.complete_update()
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            clock.advance(37);budget.finish("failed_amp_attempt_kept")
            value=ledger.snapshot()
            self.assertEqual((value["charged_updates"],value["completed_updates"]),(200,199))
            self.assertEqual((value["max_updates"],value["max_seconds"]),(3000,5400))
            self.assertEqual(value["charged_seconds"],37)
            self.assertEqual(value["historical_cancelled_experiment"]["old_native_attempts"],1899)
            second=ModelWorkBudget(ledger.start("later_mock_phase"),max_attempts=1)
            second.reserve_update();second.finish()
            self.assertEqual(ledger.snapshot()["charged_updates"],201)

    def test_baseline_reserves_tail_stops_gradient_earlier_and_all_work_is_charged(self):
        clock=Clock()
        with temporary_directory() as directory,patch.object(training.time,"monotonic",clock):
            ledger=training.CostLedger(Path(directory)/"ledger.json",3000,5400);budget=ModelWorkBudget(ledger.start("stage3"))
            clock.advance(200);self.assertEqual(budget.record_baseline(200),240)
            clock.advance(39);budget.reserve_update();budget.complete_update()
            clock.advance(1)
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            budget.require_operation(300,"fitted54+save")
            clock.advance(300);budget.heartbeat() # Includes native inference and saving, not just training.
            clock.advance(59);budget.before_checkpoint(Path(directory)/"last.pt",estimated_seconds=.5)
            clock.advance(1)
            with self.assertRaises(training.BudgetExceeded):budget.heartbeat()
            budget.finish("hard_cap")
            self.assertEqual(ledger.snapshot()["charged_seconds"],600)
            self.assertEqual(budget.report()["elapsed_model_work_seconds"],600)
            self.assertEqual(ledger.snapshot()["charged_updates"],1)

    def test_no_update_at420_and_no_unreserved_checkpoint_after_deadline(self):
        clock=Clock()
        with temporary_directory() as directory,patch.object(training.time,"monotonic",clock):
            ledger=training.CostLedger(Path(directory)/"ledger.json",3000,5400);budget=ModelWorkBudget(ledger.start("stage3"))
            clock.advance(420)
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            clock.advance(160)
            with self.assertRaises(training.BudgetExceeded):budget.before_checkpoint(Path(directory)/"last.pt")
            budget.finish("retained_previous_complete_last")
            self.assertEqual(ledger.snapshot()["charged_updates"],0)
        with temporary_directory() as directory:
            ledger=training.CostLedger(Path(directory)/"wrong.json",200,600);stage=ledger.start("wrong")
            with self.assertRaises(ValueError):ModelWorkBudget(stage)
            stage.finish()

    def test_remaining_aggregate_time_reduces_stage_tail_and_never_resets(self):
        clock=Clock()
        with temporary_directory() as directory,patch.object(training.time,"monotonic",clock):
            ledger=training.CostLedger(Path(directory)/"ledger.json",3000,5400);prior=ledger.start("prior_mock_cost")
            clock.advance(5300);prior.finish()
            budget=ModelWorkBudget(ledger.start("stage3"))
            self.assertEqual(budget.effective_time_cap,100);self.assertEqual(budget.update_cutoff,0)
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            with self.assertRaises(training.BudgetExceeded):budget.require_operation(100,"native tail")
            budget.finish("aggregate_tail_unavailable")
            self.assertEqual(ledger.snapshot()["charged_seconds"],5300)

    def test_combined_storage_deduplicates_overlaps_and_counts_atomic_temp(self):
        with temporary_directory() as directory:
            root=Path(directory);old=root/"old";new=root/"new";old.mkdir();new.mkdir()
            (old/"old.pt").write_bytes(b"a"*100);(new/"last.pt").write_bytes(b"b"*100);(new/"last.pt.tmp").write_bytes(b"c"*100)
            report=storage_footprint([root,old,new/"last.pt"])
            self.assertEqual((report["bytes"],report["files"]),(300,3))
            with patch("tdyssey.native_rgb_stage_budget.shutil.disk_usage",return_value=SimpleNamespace(free=10**9)):
                safe=StorageBudget([old,new],max_bytes=500,atomic_reserve_bytes=150,minimum_free_bytes=0)
                self.assertEqual(safe.check(force=True)["projected_bytes"],450)
                with self.assertRaises(training.BudgetExceeded):safe.check(force=True,write_reserve_bytes=250)
            self.assertTrue((old/"old.pt").exists())


class CallbackAndCheckpointTests(unittest.TestCase):
    def test_mock_train_hooks_t0_final_only_and_external_clock_not_finished_twice(self):
        class Tensor:pass
        model=FakeModel();model.parameter.device=SimpleNamespace(type="cuda",index=0);model.parameter.numel=lambda:1
        saved=[];events=[]
        component=lambda:SimpleNamespace(state_dict=lambda:dict(mock=True),zero_grad=lambda **kwargs:None)
        cuda=SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda seed:None,reset_peak_memory_stats=lambda device:None,
            max_memory_allocated=lambda device:0,max_memory_reserved=lambda device:0,is_initialized=lambda:False)
        torch=SimpleNamespace(Tensor=Tensor,cuda=cuda,set_num_threads=lambda count:None,manual_seed=lambda seed:None,
            inference_mode=nullcontext,optim=SimpleNamespace(AdamW=lambda *args,**kwargs:component(),
                lr_scheduler=SimpleNamespace(CosineAnnealingLR=lambda *args,**kwargs:component())),
            amp=SimpleNamespace(GradScaler=lambda *args,**kwargs:component()),
            save=lambda payload,path:(saved.append(deepcopy(payload)),Path(path).write_bytes(b"mock-checkpoint")))
        class Dataset:
            def __init__(self,*args,**kwargs):pass
            def __len__(self):return 64
        order=SimpleNamespace(state_dict=lambda:dict(cursor=0))
        with temporary_directory() as directory:
            root=Path(directory);ledger=training.CostLedger(root/"cost.json",3000,5400)
            storage=StorageBudget([root]);budget=ModelWorkBudget(ledger.start("stage3"),storage=storage)
            def callback(name):
                def invoke(context):events.append((name,len(saved),(context["output"]/"last.pt").is_file()));return dict(panel=name)
                return invoke
            from contextlib import ExitStack
            with ExitStack() as patches:
                patches.enter_context(patch.dict("sys.modules",{"torch":torch}))
                for name,value in dict(data_identity=lambda path:dict(mock="data"),source_identity=lambda:dict(files={}),
                    world_model_spec=lambda:dict(checkpoint=dict(mock="published")),construct_model=lambda *args,**kwargs:(model,"source"),
                    WindowOrder=lambda *args,**kwargs:order,make_generator=lambda *args:"rng",capture_rng=lambda generator:dict(mock="rng"),
                    restore_rng=lambda *args:None,tensor_hashes=lambda m:dict(w=m.weight),
                    BatchRenderer=lambda *args:SimpleNamespace(close=lambda:None),measure_denoising=lambda *args,**kwargs:1.,
                    parameter_update_report=lambda *args:dict(mock=True)).items():patches.enter_context(patch.object(training,name,value))
                patches.enter_context(patch("tdyssey.native_rgb_data.NativeRGBDataset",Dataset))
                config=dict(max_updates=3000,max_train_seconds=5400,artifact_root=str(root),ledger_path=str(ledger.path),microbatch=4,
                    save_every=50,validation_every=100000)
                result=training.train(config,root/"training","data","gpc","published",execute=True,phase="stage3",updates=0,
                    indices=list(range(18)),validation_indices=list(range(36)),checkpoint_policy="last_only",work_budget=budget,
                    physics_source_root="physics",before_updates_callback=callback("before"),after_updates_callback=callback("after"))
            self.assertEqual(events,[("before",1,True),("after",2,True)])
            self.assertEqual(len(saved),3);self.assertIsNone(result["best"])
            self.assertFalse((root/"training/best.pt").exists());self.assertFalse(budget.closed)
            self.assertIsNotNone(ledger.snapshot()["active_stage"])
            self.assertEqual(saved[-1]["ledger"]["max_updates"],3000)
            budget.finish();self.assertIsNone(ledger.snapshot()["active_stage"])

    def invoke(self,model,callback):
        rng={"value":5478}
        def capture(generator):return dict(rng)
        def restore(value,generator):rng.clear();rng.update(value)
        context=dict(model=model,generator="mock-generator",rng=rng)
        with patch.dict("sys.modules",{"torch":SimpleNamespace(inference_mode=nullcontext)}),\
             patch.object(training,"capture_rng",capture),patch.object(training,"restore_rng",restore),\
             patch.object(training,"tensor_hashes",lambda m:dict(w=m.weight)):
            result=training.invoke_native_prediction_callback(callback,context)
        return result,rng

    def test_native_callback_preserves_mode_rng_and_grad_flags(self):
        model=FakeModel()
        def callback(context):
            self.assertFalse(context["model"].training)
            context["rng"]["value"]=9999
            return dict(raw_output="same-model-direct-rgb")
        result,rng=self.invoke(model,callback)
        self.assertEqual(rng["value"],5478);self.assertTrue(model.training);self.assertTrue(model.parameter.requires_grad)
        self.assertEqual(result["raw_output"],"same-model-direct-rgb")
        def bad(context):context["model"].parameter.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError,"requires_grad"):self.invoke(model,bad)
        self.assertTrue(model.parameter.requires_grad);self.assertTrue(model.training)
        def failure(context):context["rng"]["value"]=5;raise RuntimeError("inference failure")
        with self.assertRaisesRegex(RuntimeError,"inference failure"):self.invoke(model,failure)
        self.assertTrue(model.training)

    def test_mock_complete_checkpoint_contains_all_math_order_and_ledger_state(self):
        captured=[]
        class FakeTensor:pass
        fake_torch=SimpleNamespace(Tensor=FakeTensor,save=lambda payload,path:(captured.append(deepcopy(payload)),Path(path).write_bytes(b"mock-complete-state")))
        def component(value):return SimpleNamespace(state_dict=lambda:deepcopy(value),load_state_dict=lambda state:None)
        model=FakeModel();optimizer=component(dict(state={"w":dict(step=5,exp_avg=[.1],exp_avg_sq=[.2])}))
        scaler=component(dict(scale=128));scheduler=component(dict(T_max=3000,eta_min=1e-5,last_epoch=5));order=component(dict(cursor=80,window_positions={"A":5}))
        progress=dict(updates=5,samples=80);config=dict(max_updates=3000,max_train_seconds=5400);identity=dict(source="fixed")
        ledger=dict(max_updates=3000,charged_updates=6,completed_updates=5,charged_seconds=220)
        with temporary_directory() as directory,patch.dict("sys.modules",{"torch":fake_torch}),\
             patch.object(training,"capture_rng",return_value=dict(python="p",numpy="n",torch="t",cuda=None,noise_generator="g")):
            saved=training.save_training_checkpoint(Path(directory)/"last.pt",model,optimizer,scaler,scheduler,"g",order,progress,config,identity,ledger_state=ledger)
            self.assertEqual(saved["updates"],5)
            self.assertEqual(set(("model","optimizer","scaler","scheduler","rng","order","progress","config","identity","ledger"))-set(captured[0]),set())
            self.assertEqual(captured[0]["ledger"]["charged_updates"],6)
            self.assertEqual(captured[0]["scheduler"]["T_max"],3000)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))),1)

    def test_explicit_indices_no_legacy_linspace_and_source_clean_boundary_guards(self):
        dataset=list(range(100))
        self.assertEqual(training.explicit_window_indices(dataset,[3,9],"dev"),[3,9])
        for bad in (None,[],[3,3],[-1],[101],[True]):
            with self.assertRaises(ValueError):training.explicit_window_indices(dataset,bad,"dev")
        with self.assertRaisesRegex(ValueError,"explicit TRAIN/dev"):
            training.train({},"not-created","not-opened","not-loaded","not-loaded",execute=True,checkpoint_policy="last_only")
        source=Path(training.__file__).read_text(encoding="utf-8");ast.parse(source)
        self.assertIn("if work_budget is None:\n                stage.finish",source)
        self.assertIn('order.load_state_dict(boundary_order);restore_rng(boundary_rng,generator)',source)
        self.assertLess(source.index('if checkpoint_policy=="last_only" and resume is None:last_info=save("last.pt")'),source.index('run_hook(before_updates_callback,"before_updates")'))
        self.assertLess(source.index('run_hook(after_updates_callback,"after_updates")'),source.index('if after_updates_callback is not None:last_info=save("last.pt")'))


if __name__=="__main__":unittest.main()
