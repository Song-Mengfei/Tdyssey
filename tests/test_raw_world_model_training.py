"""Metadata, tiny RGB and arithmetic markers; no actual Torch or physics."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tdyssey import raw_world_model_training as training


def manifest(root,shape,controls=7):
    root.mkdir();entries=[]
    for index,split in enumerate(("train","validation","model_test")):
        identifier=shape+"-"+split
        entries.append(dict(trajectory_id=identifier,control_steps=controls,observation_count=controls+1,contact_steps=0,
            files=[dict(name=f"e{index}.npz",bytes=1,sha256=hashlib.sha256((identifier+"npz").encode()).hexdigest()),
                   dict(name=f"e{index}.json",bytes=1,sha256=hashlib.sha256((identifier+"json").encode()).hexdigest())]))
    value=dict(schema_version=1,status="complete",config=dict(episodes=3,shape=shape),
        provenance=dict(environment_family=training.ENVIRONMENT_FAMILY),episodes=entries,
        splits={split:[shape+"-"+split] for split in ("train","validation","model_test")},
        counts=dict(control_steps=3*controls,control_physics_substeps=30*controls,reset_attempts=3,reset_physics_substeps=3),
        rates=dict(simulation_hz=100,control_hz=10))
    (root/"manifest.json").write_text(json.dumps(value));return value


class RawTrainingChecks(unittest.TestCase):
    def test_balanced_finite_metadata_index_and_cross_split_image_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);first=manifest(root/"T","T");second=manifest(root/"L","L",8)
            original=Path.read_bytes
            def read(path):
                if path.name!="manifest.json":raise AssertionError("Unexpected payload read")
                return original(path)
            with patch.object(Path,"read_bytes",read):plan=training.plan_training([root/"T",root/"L"])
            self.assertEqual(plan["steps"],8);self.assertEqual(plan["unused_windows"],dict(T=0,L=1))
            self.assertEqual([row["shape"] for row in plan["entries"]],["T","L"]*4)
            self.assertEqual(len({(row["trajectory_id"],row["anchor"]) for row in plan["entries"]}),8)
            for steps in (3,10):
                with self.assertRaises(ValueError):training.plan_training([root/"T",root/"L"],steps)
            second["episodes"][2]["files"][0]["sha256"]=first["episodes"][0]["files"][0]["sha256"]
            (root/"L/manifest.json").write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError,"Image payload crosses"):training.plan_training([root/"T",root/"L"])

    def test_read_only_help_and_explicit_pre_torch_gate(self):
        from scripts import train_raw_world_model as script
        original=__import__
        def blocked(name,*args,**kwargs):
            if name=="torch" or name.startswith("torch."):raise AssertionError("Actual Torch import")
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary,patch("builtins.__import__",side_effect=blocked):
            root=Path(temporary);manifest(root/"T","T");manifest(root/"L","L")
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:script.main(["--help"])
                value=script.main(["--datasets",str(root/"T"),str(root/"L"),"--output",str(root/"unused")])
            self.assertEqual(stopped.exception.code,0);self.assertFalse(value["training_executed"])
            self.assertFalse((root/"unused").exists())
            with self.assertRaises(PermissionError):training.train({},None,{},root/"unused")
            with self.assertRaises(PermissionError):training.train({},None,{},root/"unused",execute=True,check=lambda *_:None)

    def test_guard_stopped_result_has_nonzero_cli_exit(self):
        from scripts import train_raw_world_model as script
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest(root/"T","T");manifest(root/"L","L")
            argv=[script.__file__,"--datasets",str(root/"T"),str(root/"L"),"--output",str(root/"unused"),
                  "--execute","--initial","--checkpoint","unused.pt","--gpc-root","unused-source"]
            with patch.object(sys,"argv",argv),patch("tdyssey.raw_training_guard.run_guarded",return_value=dict(status="stopped")):
                with self.assertRaises(SystemExit) as stopped:runpy.run_path(script.__file__,run_name="__main__")
            self.assertEqual(stopped.exception.code,1)

    def test_frozen_scientific_loss_source_parity_without_external_files(self):
        local=ast.parse(Path(training.__file__).read_text())
        names=("NoisePlan","make_generator","_state_hash","_batch","make_noise_plan","_validate_noise","supervised_denoising_loss")
        nodes=[next(node for node in local.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name) for name in names]
        digest=hashlib.sha256('\n'.join(ast.dump(node) for node in nodes).encode()).hexdigest()
        self.assertEqual(digest,training.OBJECTIVE_AST_SHA256)

    def test_model_only_export_reloads_asset_metadata_without_optimizer_restore(self):
        from tdyssey.official_world_model import inspect_profile
        class Tensor:
            def detach(self):return self
            def cpu(self):return self
            def numel(self):return 1
            def element_size(self):return 4
        saved=[]
        def save(value,stream):saved.append(set(value));stream.write(b'model-only-marker')
        sdk=SimpleNamespace(save=save,isfinite=lambda value:SimpleNamespace(all=lambda:True))
        model=SimpleNamespace(state_dict=lambda:{'inner_model.weight':Tensor()})
        identity=dict(init_sha256='a'*64,data_identity=[],list_identity=dict(sha256='b'*64,examples=8),
            source_identity=dict(upstream_commit='c'*40))
        with tempfile.TemporaryDirectory() as temporary:
            result=training.export_model(model,temporary,8,identity,sdk)
            side=json.loads(Path(result['sidecar']).read_bytes())
            self.assertEqual(saved,[{'inner_model.weight'}])
            self.assertFalse(side['optimizer_state']);self.assertFalse(side['rng_state'])
            self.assertEqual(side['updates'],8);self.assertTrue(inspect_profile(result['profile'])['ready'])
            with self.assertRaises(FileExistsError):training.export_model(model,temporary,8,identity,sdk)

    def test_raw_cache_and_exact_frame_action_alignment(self):
        try:import numpy as np
        except ImportError:self.skipTest("NumPy needed for tiny RGB fixture")
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest(root/"T","T");manifest(root/"L","L")
            dataset=training.RawWindowDataset(training.plan_training([root/"T",root/"L"]))
            frames=np.full((8,3,96,96),119,np.uint8)
            for index in range(8):frames[index,:,0,0]=index
            def load(_root,entry):
                shape=entry["trajectory_id"][0]
                steps=[SimpleNamespace(action=(100.1+index,200.2+index),before=SimpleNamespace(shape=shape),after=SimpleNamespace(shape=shape)) for index in range(7)]
                return SimpleNamespace(trajectory_id=entry["trajectory_id"],control_dt=.1,transitions=steps),{},dict(images=frames)
            with patch.object(training,"load_episode",side_effect=load) as loader:
                first=dataset[0];later=dataset[2];self.assertEqual(loader.call_count,1)
                dataset[1];self.assertEqual(loader.call_count,2)
            np.testing.assert_array_equal(first["image"],frames[:5].astype(np.float32)/np.float32(255))
            np.testing.assert_array_equal(later["image"],frames[1:6].astype(np.float32)/np.float32(255))
            action=np.asarray([(100.1+index,200.2+index) for index in range(4)],np.float32)
            np.testing.assert_array_equal(first["action"],action/np.float32(511)*np.float32(2)-np.float32(1))

    def test_loop_reserves_before_forward_and_stops_without_replacement(self):
        class Marker:
            def __getitem__(self,_):return self
            def to(self,_):return self
        parameter=SimpleNamespace(requires_grad=True,grad=None);model=SimpleNamespace(parameters=lambda:iter([parameter]));calls=[]
        torch=SimpleNamespace(from_numpy=lambda _:Marker(),isfinite=lambda _:SimpleNamespace(all=lambda:True),
                              nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *_a,**_k:.5)))
        optimizer=SimpleNamespace(zero_grad=lambda **_:setattr(parameter,"grad",None),step=lambda:calls.append("step"))
        samples=[dict(image=Marker(),action=Marker(),shape=shape,trajectory_id=shape,anchor=3) for shape in ("T","L","T","L")]
        counts=dict(model_instances=1,model_loads=1,reserved_updates=0,completed_updates=0,exported_steps=[])
        def check(_phase,value):
            if value["reserved_updates"]==3:raise RuntimeError("synthetic stop")
        def loss(*_,**__):
            self.assertEqual(counts["reserved_updates"],counts["completed_updates"]+1);calls.append("forward")
            return SimpleNamespace(backward=lambda:setattr(parameter,"grad",1.)),dict(loss_denoising=.1)
        with self.assertRaisesRegex(RuntimeError,"synthetic stop"):
            training.run_updates(model,optimizer,samples,"cpu",None,counts,check,lambda _:None,[],torch,loss_function=loss)
        self.assertEqual(calls,["forward","step","forward","step"])
        self.assertEqual((counts["reserved_updates"],counts["completed_updates"]),(3,2))
