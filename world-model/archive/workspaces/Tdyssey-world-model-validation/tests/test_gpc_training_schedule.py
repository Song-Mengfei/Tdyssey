"""JSON/marker checks; no Torch, neural updates, image arrays or simulation."""
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey import gpc_training_schedule as schedule
from tdyssey import gpc_adaptation as adapter
from tdyssey import gpc_adaptation_runtime as runtime


def fixture(controls=35):
    rows=[];payloads={}
    for shape in ("T","L"):
        for scene in range(12):
            identifier=f"fake-{shape}-{scene:02d}";transitions=[];diagnostics=[];angle=3.1
            for index in range(controls):
                delta=.2 if index==3 else -.2 if index==7 else .01
                after=math.atan2(math.sin(angle+delta),math.cos(angle+delta))
                before_state=dict(shape=shape,control_step=index,object_body=dict(pose=dict(angle=angle)))
                after_state=dict(shape=shape,control_step=index+1,object_body=dict(pose=dict(angle=after)))
                transitions.append(dict(before=before_state,after=after_state));angle=after
                diagnostics.append(dict(n_contacts=int(index==3 or index!=7 and index%3==0)))
            record=dict(trajectory=dict(trajectory_id=identifier,collector_version="official-waypoint-play-v1",
                control_dt=.1,transitions=transitions),step_diagnostics=diagnostics)
            payload=schedule.schedule_bytes(record);payloads[identifier]=payload
            rows.append(dict(trajectory_id=identifier,shape=shape,role="train",split="train",control_steps=controls,
                windows=controls-3,contact_steps=sum(d["n_contacts"]>0 for d in diagnostics),payloads_present=True,
                initial_state_sha256=hashlib.sha256(identifier.encode()).hexdigest(),catalog_sha256="c"*64,
                source_manifest_sha256="e"*64,root="never-opened-by-reader",
                entry=dict(files=[dict(name="episode.json",bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),
                                  dict(name="never-open.npz",bytes=1,sha256="d"*64)])))
    for role in ("development","independent_validation","held_model_test"):
        rows.append(dict(role=role,split="model_test" if role=="held_model_test" else "validation",trajectory_id="forbidden-"+role,
            root="must-never-open",entry=dict(files=[dict(name="forbidden.json"),dict(name="forbidden.npz")])))
    data=dict(sha256="a"*64,episodes=rows);opened=[]
    def reader(row):
        if row["role"]!="train" or row["split"]!="train":raise AssertionError("Nontrain payload opened")
        opened.append(row["trajectory_id"]);return payloads[row["trajectory_id"]]
    return data,payloads,opened,reader


class MarkerDataset:
    def __init__(self,data):
        self.manifest=data;self.split=self.role="train"
        self.episodes=[row for row in data["episodes"] if row["role"]=="train"];self.offsets=[0]
        for row in self.episodes:self.offsets.append(self.offsets[-1]+row["windows"])


class MarkerModel:
    def __init__(self):self.parameter=SimpleNamespace(requires_grad=True,device="marker-cpu",grad=None);self.marker=0
    def parameters(self):return iter([self.parameter])
    def named_parameters(self):return iter([("marker",self.parameter)])
    def train(self):return self
    def state_dict(self):return dict(marker=self.marker)
    def load_state_dict(self,value,strict=True):self.marker=value["marker"]


class MarkerOptimizer:
    def __init__(self,model):self.model=model;self.steps=0;self.param_groups=[dict(params=[model.parameter])]
    def zero_grad(self,**kwargs):self.model.parameter.grad=None
    def step(self):self.steps+=1;self.model.marker+=1
    def state_dict(self):return dict(steps=self.steps)
    def load_state_dict(self,value):self.steps=value["steps"]


class ScheduleChecks(unittest.TestCase):
    def setUp(self):
        self.data,self.payloads,self.opened,self.reader=fixture()
        self.value=schedule.generate_schedule(self.data,record_reader=self.reader)
        self.sha=hashlib.sha256(schedule.schedule_bytes(self.value)).hexdigest()
        self.dataset=MarkerDataset(self.data);self.consumer=schedule.ScheduleConsumer(self.value,self.sha,self.dataset)

    def test_fixed_500_order_scene_quotas_rotation_first_and_no_nontrain_opens(self):
        entries=self.value["entries"]
        self.assertEqual([r["shape"] for r in entries],["T","L"]*250)
        self.assertEqual(len(self.opened),24);self.assertTrue(all(x.startswith("fake-") for x in self.opened))
        self.assertEqual(Counter(r["shape"] for r in entries),dict(T=250,L=250))
        for shape in ("T","L"):
            coverage=self.value["coverage"][shape]
            self.assertEqual((coverage["distinct_scenes"],coverage["unique_windows"],coverage["repeated_exposures"]),(12,250,0))
            self.assertEqual(coverage["sampled_unique_rotation_windows"],coverage["source_rotation_windows"])
            self.assertEqual((coverage["clockwise_examples"],coverage["counterclockwise_examples"]),(12,12))
            for scene in range(12):
                rows=[r for r in entries if r["trajectory_id"]==f"fake-{shape}-{scene:02d}"]
                self.assertEqual(len(rows),21 if scene<10 else 20)
                self.assertEqual({r["anchor"] for r in rows[:2]},{3,7})
                fill=[r["contact"] for r in rows[2:]]
                available=coverage["per_scene"][scene]["source_strata"]
                paired=min(available.get("contact_only",0),available.get("other",0),len(fill)//2)
                self.assertEqual(fill[:2*paired],[True,False]*paired)
        again=schedule.generate_schedule(self.data,record_reader=self.reader)
        self.assertEqual(schedule.schedule_bytes(self.value),schedule.schedule_bytes(again))

    def test_finite_consumer_preserves_source_window_indices(self):
        for cursor in (0,1,23,24,249,499):
            row=self.value["entries"][cursor];index=self.consumer.indices(cursor,1)[0]
            episode=next(i for i in range(24) if self.dataset.offsets[i]<=index<self.dataset.offsets[i+1])
            self.assertEqual(self.dataset.episodes[episode]["trajectory_id"],row["trajectory_id"])
            self.assertEqual(3+index-self.dataset.offsets[episode],row["anchor"])
        for cursor,count in ((-1,1),(500,1),(499,2),(0,0)):
            with self.assertRaises(ValueError):self.consumer.indices(cursor,count)
        self.assertEqual(self.consumer.checkpoint_state(500)["cursor"],500)

    def test_public_report_mutation_cannot_change_the_frozen_consumed_order(self):
        expected=self.consumer.indices(0,4)
        self.consumer.value["entries"][0]["dataset_index"]+=1000
        self.value["entries"][1]["dataset_index"]+=1000
        self.assertEqual(self.consumer.indices(0,4),expected)

    def test_short_scene_repeats_are_disclosed_and_never_pull_another_role(self):
        data,_,opened,reader=fixture(controls=6);value=schedule.generate_schedule(data,record_reader=reader)
        for shape in ("T","L"):
            coverage=value["coverage"][shape]
            self.assertEqual((coverage["unique_windows"],coverage["repeated_exposures"]),(36,214))
        self.assertEqual(len(opened),24)

    def test_hash_order_source_and_scene_reassignment_are_rejected(self):
        changed=deepcopy(self.value);changed["entries"][0]["dataset_index"]+=1
        with self.assertRaisesRegex(ValueError,"ordered"):schedule.validate_schedule(changed,data=self.data)
        changed=deepcopy(self.data);changed["episodes"]=list(reversed(changed["episodes"]))
        with self.assertRaises(ValueError):schedule.ScheduleConsumer(self.value,self.sha,MarkerDataset(changed))
        changed=deepcopy(self.data);changed["sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"another source"):schedule.validate_schedule(self.value,data=changed)
        self.payloads["fake-T-00"]+=b" "
        with self.assertRaisesRegex(ValueError,"JSON changed"):schedule.generate_schedule(self.data,record_reader=self.reader)

    def test_canonical_loader_opens_no_episode_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"schedule.json";path.write_bytes(schedule.schedule_bytes(self.value));original=Path.read_bytes
            with patch.object(Path,"read_bytes",lambda p:original(p) if p==path else (_ for _ in ()).throw(AssertionError("payload open"))):
                self.assertEqual(schedule.load_schedule(path,self.sha,data=self.data),self.value)
            with self.assertRaisesRegex(ValueError,"SHA256"):schedule.load_schedule(path,"0"*64)
        self.dataset.split="independent_validation"
        with self.assertRaisesRegex(ValueError,"train dataset"):schedule.ScheduleConsumer(self.value,self.sha,self.dataset)

    def make_budget(self,path):
        budget=runtime.AdaptationBudget.create(path,dict(candidate="marker-only",**self.consumer.binding),
            max_updates=500,max_eval_batches=32,max_batch_size=1,max_train_examples=500,max_eval_examples=32,
            deadline_seconds=3600,clock=lambda:100.,monotonic=lambda:50.)
        op=budget.reserve_model_load("train_init");budget.complete_model_load(op)
        return budget

    def reserve(self,budget,cursor):
        return budget.reserve_update(1,metadata=dict(training_schedule=self.consumer.checkpoint_state(cursor),
            next_cursor=cursor+1,dataset_indices=self.consumer.indices(cursor,1)))

    def test_500_marker_operations_consume_manifest_order_and_checkpoint_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);budget=self.make_budget(root/"ledger");model=MarkerModel();optimizer=MarkerOptimizer(model)
            seen=[];extras=[];loss_calls=[]
            class MarkerStore:
                def save(inner,unused_model,unused_optimizer,extra_state):
                    runtime._checkpoint_schedule_extra(budget,extra_state);extras.append(deepcopy(extra_state))
                    path=root/"marker-boundary.bin";payload=str(budget.snapshot()["completed_updates"]).encode();path.write_bytes(payload)
                    budget._operation("checkpoint_committed",dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),boundary=budget._boundary()))
            def marker_loss(unused_model,inputs,**kwargs):
                loss_calls.append(inputs["marker_indices"])
                return SimpleNamespace(backward=lambda:setattr(model.parameter,"grad","marker-gradient")),dict(loss_denoising=1.)
            fake_torch=SimpleNamespace(_marker_only=True,isfinite=lambda value:SimpleNamespace(all=lambda:True),
                nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *args,**kwargs:0.)))
            cfg=dict(training_schedule=dict(path="synthetic-only",sha256=self.sha),train=dict(order=schedule.ORDER,
                batch_size=1,max_updates=500,max_examples=500,phase_seconds=900,checkpoint_every=50,selected_update=500,grad_clip_norm=1.))
            with patch.dict("sys.modules",torch=fake_torch),patch.object(adapter,"balanced_indices",side_effect=AssertionError("legacy prefix")), \
                 patch.object(adapter,"batch",side_effect=lambda ds,indices,device:dict(marker_indices=indices)), \
                 patch("tdyssey.gpc_supervised_adaptation.supervised_denoising_loss",side_effect=marker_loss):
                result=adapter.train_candidate(model,optimizer,self.dataset,cfg=cfg,budget=budget,store=MarkerStore(),generator="marker",
                    schedule=self.consumer,monotonic=lambda:50.,metrics=lambda step,values:seen.extend(values["indices"]))
            self.assertEqual(seen,[row["dataset_index"] for row in self.value["entries"]])
            self.assertEqual((len(loss_calls),optimizer.steps,budget.snapshot()["completed_updates"]),(500,500,500))
            self.assertEqual([e["training_schedule"]["cursor"] for e in extras],list(range(50,501,50)))
            self.assertEqual(result["training_schedule"],self.consumer.checkpoint_state(500))
            self.assertEqual(budget.snapshot()["training_schedule"]["cursor"],500)
            self.assertEqual(budget._boundary()["training_schedule"],self.consumer.checkpoint_state(500))

    def test_resume_binds_order_and_cumulative_cursor_without_replaying_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);budget=self.make_budget(root/"ledger")
            for cursor in range(5):budget.complete_update(self.reserve(budget,cursor))
            extra=dict(next_example=5,training_schedule=self.consumer.checkpoint_state(5))
            runtime._checkpoint_schedule_extra(budget,extra)
            path=root/"marker.bin";payload=b"marker weights only";path.write_bytes(payload)
            budget._operation("checkpoint_committed",dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),boundary=budget._boundary()))
            recovered=runtime.AdaptationBudget.resume(root/"ledger",budget.attempt["binding"],clock=lambda:100.,monotonic=lambda:50.)
            self.assertEqual(recovered.snapshot()["training_schedule"],self.consumer.checkpoint_state(5))
            self.assertTrue(recovered.snapshot()["requires_checkpoint_restore"])
            self.assertEqual(recovered.attempt["deadline_unix"],budget.attempt["deadline_unix"])
            wrong=dict(budget.attempt["binding"],training_schedule_order_sha256="b"*64)
            with self.assertRaisesRegex(runtime.AdaptationRuntimeError,"binding differs"):
                runtime.AdaptationBudget.resume(root/"ledger",wrong,clock=lambda:100.,monotonic=lambda:50.)
            wrong_extra=dict(next_example=4,training_schedule=self.consumer.checkpoint_state(4))
            with self.assertRaisesRegex(runtime.AdaptationRuntimeError,"ordered cursor"):runtime._checkpoint_schedule_extra(budget,wrong_extra)
            with self.assertRaisesRegex(runtime.AdaptationRuntimeError,"cursor/order"):self.reserve(budget,4)
            self.assertEqual(budget.snapshot()["training_schedule"]["cursor"],5)

    def test_pending_failure_spends_cursor_and_missing_schedule_cannot_use_legacy_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);budget=self.make_budget(root/"ledger");self.reserve(budget,0)
            self.assertEqual((budget.snapshot()["spent_train_examples"],budget.snapshot()["training_schedule"]["cursor"]),(1,1))
            with self.assertRaisesRegex(runtime.AdaptationRuntimeError,"not complete"):
                runtime.AdaptationBudget.resume(root/"ledger",budget.attempt["binding"],clock=lambda:100.,monotonic=lambda:50.)
        with tempfile.TemporaryDirectory() as temporary:
            budget=self.make_budget(Path(temporary)/"ledger")
            with self.assertRaisesRegex(runtime.AdaptationRuntimeError,"cursor/order"):budget.reserve_update(1)

    def test_checkpoint_restore_checks_schedule_cursor_before_marker_state_mutation(self):
        for wrong_cursor in (True,False):
            with self.subTest(wrong_cursor=wrong_cursor),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);budget=self.make_budget(root/"ledger")
                for cursor in range(5):budget.complete_update(self.reserve(budget,cursor))
                path=root/"marker.bin";raw=b"synthetic marker checkpoint";path.write_bytes(raw)
                budget._operation("checkpoint_committed",dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),boundary=budget._boundary()))
                recovered=runtime.AdaptationBudget.resume(root/"ledger",budget.attempt["binding"],clock=lambda:100.,monotonic=lambda:50.)
                model=MarkerModel();optimizer=MarkerOptimizer(model);config=dict(marker="fixture")
                cursor=4 if wrong_cursor else 5
                payload=dict(kind="gpc_adaptation_checkpoint.v1",budget=recovered._boundary(),attempt=recovered.attempt,
                    config_identity=config,logical_steps=5,optimizer_signature=runtime._optimizer_signature(model,optimizer),
                    extra_state=dict(next_example=cursor,training_schedule=self.consumer.checkpoint_state(cursor)),
                    model=dict(marker=17),optimizer=dict(steps=5),rng=dict(marker="fixed"))
                fake_torch=SimpleNamespace(_marker_only=True,load=lambda *args,**kwargs:deepcopy(payload))
                store=runtime.CheckpointStore(path,recovered,config,max_bytes=1024)
                with patch.dict("sys.modules",torch=fake_torch),patch.object(runtime,"_tensor_copy",side_effect=lambda value,torch:deepcopy(value)) as copy, \
                     patch.object(runtime,"_capture_rng",return_value={}),patch.object(runtime,"_restore_rng") as rng:
                    if wrong_cursor:
                        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,"ordered cursor"):store.restore(model,optimizer)
                        copy.assert_not_called();rng.assert_not_called()
                        self.assertEqual((model.marker,optimizer.steps),(0,0))
                    else:
                        result=store.restore(model,optimizer)
                        self.assertEqual((model.marker,optimizer.steps),(17,5));rng.assert_called_once()
                        self.assertEqual(result["extra_state"]["training_schedule"]["cursor"],5)
                        self.assertFalse(recovered.snapshot()["requires_checkpoint_restore"])


if __name__=="__main__":unittest.main()
