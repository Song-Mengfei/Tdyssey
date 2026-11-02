"""Root-run only: ordinary files, fake Torch, and optional synthetic RGB/Pillow."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tdyssey import gpc_raw_fulltrain_outputs as outputs


class MarkerTensor:
    def __init__(self,value):self.value=value
    def detach(self):return self
    def cpu(self):return self
    def numel(self):return 2
    def element_size(self):return 4


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo=Path(self.temp.name)
        self.root=self.repo/"artifacts"/outputs.OUTPUT_NAME
        self.provenance={key:{"path":key,"sha256":"a"*64} for key in
                         ("data_identity","list_identity","source_identity")}
        self.saved=[]
        self.torch=ModuleType("torch")
        self.torch.Tensor=MarkerTensor
        self.torch.isfinite=lambda value:SimpleNamespace(all=lambda:True)
        def save(state,stream):
            self.saved.append(deepcopy(state))
            stream.write(("model-only:"+str(state["inner_model.weight"].value)).encode())
        self.torch.save=save
        self.fake=patch.dict(sys.modules,{"torch":self.torch})
        self.fake.start();self.addCleanup(self.fake.stop)

    def model(self,value):
        return SimpleNamespace(state_dict=lambda:{"inner_model.weight":MarkerTensor(value)},
                               optimizer=object(),rng_state=object())

    def export_pair(self):
        first=outputs.export_checkpoint(self.model(500),self.root,500,self.provenance)
        final=outputs.export_checkpoint(self.model(2328),self.root,2328,self.provenance)
        return first,final

    def test_plain_model_only_fixed_exports_and_truthful_final_identities(self):
        first,final=self.export_pair()
        for info,step in ((first,500),(final,2328)):
            path=Path(info["checkpoint"]["path"])
            self.assertEqual(info["checkpoint"]["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            side=json.loads(Path(info["sidecar"]["path"]).read_bytes())
            self.assertEqual(side["updates"],step)
            self.assertEqual(side["init_sha256"],outputs.INITIAL_SHA256)
            self.assertEqual(side["train_shapes"],["T","L"])
            self.assertFalse(side["t1"])
            self.assertFalse(side["optimizer_state"])
            self.assertFalse(side["rng_state"])
            self.assertEqual(set(self.saved[0]),{"inner_model.weight"})
            self.assertEqual(info["profile"]["expected_sidecar_sha256"],info["sidecar"]["sha256"])
        manifest=json.loads((self.root/"raw-tl-final.json").read_bytes())
        self.assertEqual(manifest["default_updates"],2328)
        self.assertEqual(set(manifest["exports"]),{"500","2328"})
        self.assertIn("unknown",manifest["diagnostic_status"])
        self.assertIn("not validation",manifest["selection"])
        before=Path(first["checkpoint"]["path"]).read_bytes()
        with self.assertRaises(FileExistsError):
            outputs.export_checkpoint(self.model(999),self.root,500,self.provenance)
        self.assertEqual(Path(first["checkpoint"]["path"]).read_bytes(),before)
        self.assertEqual(len(self.saved),2)

    def test_failed_serialization_leaves_spent_partial_slot_without_retry(self):
        def fail(state,stream):
            stream.write(b"partial")
            raise OSError("injected serialization failure")
        self.torch.save=fail
        with self.assertRaisesRegex(OSError,"injected"):
            outputs.export_checkpoint(self.model(500),self.root,500,self.provenance)
        path=self.root/"raw-tl-step-0500.pt"
        self.assertEqual(path.read_bytes(),b"partial")
        self.assertFalse(path.with_suffix(".json").exists())
        with self.assertRaises(FileExistsError):
            outputs.export_checkpoint(self.model(500),self.root,500,self.provenance)
        self.assertEqual(path.read_bytes(),b"partial")

    def test_bounds_and_provenance_reject_before_publication(self):
        stream=io.BytesIO();bounded=outputs._BoundedFile(stream,4)
        bounded.write(b"1234")
        with self.assertRaises(OSError):bounded.write(b"5")
        self.assertEqual(stream.getvalue(),b"1234")
        for step,provenance in ((499,self.provenance),(500,{}),
                (500,dict(self.provenance,init_sha256="b"*64)),
                (500,dict(self.provenance,note="x"*33000))):
            with self.subTest(step=step),self.assertRaises(ValueError):
                outputs.export_checkpoint(self.model(step),self.root,step,provenance)
        self.assertFalse(self.root.exists())
        self.assertLessEqual(outputs.MAX_MODULE_BYTES,outputs.OUTPUT_MAX_BYTES)

    def source_directories(self):
        initial=self.repo/"artifacts/frozen-wm-assets/denoiser.pth"
        initial.parent.mkdir(parents=True,exist_ok=True);initial.write_bytes(b"initial-marker")
        gpc=self.repo/"third_party/gpc_code";official=self.repo/"third_party/diffusion_policy"
        gpc.mkdir(parents=True);official.mkdir(parents=True)
        return gpc,official

    def test_launcher_defaults_final2328_and_explicit500_uses_actual_hash_flags(self):
        first,final=self.export_pair()
        gpc,official=self.source_directories()
        runner=Mock(return_value=SimpleNamespace(returncode=0))
        with patch.dict(os.environ,{"TDYSSEY_GPC_ROOT":str(gpc),"TDYSSEY_OFFICIAL_SOURCE":str(official)}):
            self.assertEqual(outputs.launch_demo(self.repo,runner=runner),0)
            command=runner.call_args.args[0]
            self.assertEqual(command[command.index("--checkpoint")+1],final["checkpoint"]["path"])
            self.assertEqual(command[command.index("--checkpoint-sha256")+1],final["checkpoint"]["sha256"])
            self.assertEqual(command[command.index("--sidecar-sha256")+1],final["sidecar"]["sha256"])
            self.assertEqual(command[command.index("--model-label")+1],"Raw T+L")
            self.assertNotIn("--initial-checkpoint",command)
            self.assertEqual(outputs.launch_demo(self.repo,500,runner=runner),0)
            command=runner.call_args.args[0]
            self.assertEqual(command[command.index("--checkpoint")+1],first["checkpoint"]["path"])
        self.assertEqual(runner.call_count,2)

    def test_launcher_missing_or_changed_final_never_runs_initial_as_fallback(self):
        runner=Mock()
        with self.assertRaises(FileNotFoundError):outputs.launch_demo(self.repo,runner=runner)
        _,final=self.export_pair()
        Path(final["checkpoint"]["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):outputs.launch_demo(self.repo,runner=runner)
        runner.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"),"Synthetic RGB dependencies unavailable")
    def test_sixteen_panels_preserve_raw_arrays_and_supplied_metrics_without_decoder(self):
        import builtins
        import numpy as np
        from PIL import Image
        cases=[dict(case_id=f"case-{i:02d}",target=np.full((3,96,96),.5,np.float32)) for i in range(16)]
        results={case["case_id"]:dict(initial=np.zeros((3,96,96),np.float32),
            raw500=np.full((3,96,96),-.2,np.float32),raw2328=np.full((3,96,96),1.2,np.float32),
            metrics=dict(root_supplied_raw_mae=.123)) for case in cases}
        original=deepcopy(results)
        imported=builtins.__import__
        def no_decoder(name,*args,**kwargs):
            if "wm_image_measurement" in name:raise AssertionError("No legacy decoder in panels")
            return imported(name,*args,**kwargs)
        with patch.object(builtins,"__import__",side_effect=no_decoder):
            report=outputs.write_panels(self.root,cases,results)
        self.assertEqual(len(report["panels"]),16)
        with Image.open(report["panels"][0]["panel"]["path"]) as image:
            self.assertEqual(image.size,(384,120))
        overview=json.loads(Path(report["overview"]["path"]).read_bytes())
        self.assertEqual(overview["columns"],["target","initial","raw500","raw2328"])
        self.assertTrue(overview["display_clip_only"])
        self.assertEqual(overview["rows"][0]["metrics"],dict(root_supplied_raw_mae=.123))
        for key in results:
            for role in ("initial","raw500","raw2328"):
                np.testing.assert_array_equal(results[key][role],original[key][role])
        with self.assertRaises(FileExistsError):outputs.write_panels(self.root,cases,results)

    def test_viewer_custom_button_label_is_wired_and_launcher_does_not_start_old_demo(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/"scripts/visualize_world_model.py").read_text(encoding="utf-8"))
        declared=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and node.func.attr=="add_argument"
            and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=="--model-label"]
        self.assertEqual(len(declared),1)
        app=next(node for node in ast.walk(tree) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Name) and node.func.id=="App")
        labels=next(keyword.value for keyword in app.keywords if keyword.arg=="role_labels")
        value=labels.values[next(i for i,key in enumerate(labels.keys) if key.value=="adapted")]
        self.assertIsInstance(value,ast.Attribute)
        self.assertEqual(value.attr,"model_label")
        launcher=(root/"start_raw_tl_demo.cmd").read_text(encoding="utf-8")
        self.assertIn("tdyssey.gpc_raw_fulltrain_outputs",launcher)
        self.assertIn("2328",launcher)
        self.assertNotIn("scripts\\manual_world_model.py",launcher)


if __name__=="__main__":unittest.main()
