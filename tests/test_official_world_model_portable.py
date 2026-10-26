"""Raw model boundary checks with marker files/arrays; no real Torch or GPU."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tdyssey import official_world_model as wm


class MarkerTensor:
    def __init__(self,value):self.value=np.asarray(value)
    def __getitem__(self,index):return MarkerTensor(self.value[index])
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


class RawModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.weight=root/'model.pt';self.side=root/'model.json'
        self.weight.write_bytes(b'model-state-marker')
        self.sha=hashlib.sha256(self.weight.read_bytes()).hexdigest()
        self.metadata=dict(updates=12,init_sha256='a'*64,input_contract=wm.RAW_CONTRACT['version'],
            t1=False,optimizer_state=False,train_shapes=['T','L'],
            checkpoint=dict(path='/previous-machine/unused.pt',bytes=self.weight.stat().st_size,sha256=self.sha))
        self.save_sidecar()
        self.profile=wm.model_profile(self.weight,self.sha,sidecar=self.side)

    def save_sidecar(self):self.side.write_text(json.dumps(self.metadata),encoding='utf-8')

    def test_marker_assets_are_portable_and_require_matching_identity(self):
        value=wm.inspect_profile(self.profile)
        self.assertTrue(value['ready']);self.assertEqual(value['sidecar']['metadata']['updates'],12)
        self.weight.write_bytes(b'changed')
        self.assertEqual(wm.inspect_profile(self.profile)['status'],'invalid')
        self.weight.unlink();self.assertEqual(wm.inspect_profile(self.profile)['status'],'missing')

    def test_missing_or_wrong_sidecar_never_loads_runtime_or_falls_back(self):
        self.side.unlink()
        with patch.dict(sys.modules,{'torch':None}):
            self.assertFalse(wm.inspect_profile(self.profile)['ready'])
            with self.assertRaises(ValueError):wm.load_denoiser(self.profile,'no-source',execute=True)
        self.save_sidecar();self.metadata['t1']=True;self.save_sidecar()
        self.assertFalse(wm.inspect_profile(self.profile)['ready'])
        self.metadata['t1']=False;self.metadata['optimizer_state']=True;self.save_sidecar()
        self.assertFalse(wm.inspect_profile(self.profile)['ready'])
        with self.assertRaises(PermissionError):wm.load_denoiser(self.profile,'no-source')
        with self.assertRaises(ValueError):wm.OfficialWorldModel(wm.initial_profile(self.weight),'no-source')

    def test_single_model_boundary_rejects_second_load(self):
        with patch.object(wm,'_ACTIVE_MODEL',lambda:object()),patch.object(wm,'_construct_denoiser') as construct:
            with self.assertRaises(RuntimeError):wm.load_denoiser(self.profile,'no-source',execute=True)
            construct.assert_not_called()

    def test_sampler_constructor_failure_releases_partial_network(self):
        class Network:
            asset_identity={}
            def cpu(self):self.on_cpu=True;return self
        network=Network();network.on_cpu=False
        source=SimpleNamespace(DiffusionSamplerConfig=lambda **kw:kw,
            DiffusionSampler=lambda *args:(_ for _ in ()).throw(RuntimeError('sampler failed')))
        fake=ModuleType('torch');fake.cuda=SimpleNamespace(is_initialized=lambda:True,empty_cache=lambda:None)
        model=wm.OfficialWorldModel.__new__(wm.OfficialWorldModel)
        with patch.dict(sys.modules,{'torch':fake}),patch.object(wm,'load_denoiser',return_value=(network,object(),source)):
            with self.assertRaisesRegex(RuntimeError,'sampler failed'):
                model.__init__(self.profile,'no-source',execute=True)
        self.assertTrue(network.on_cpu);self.assertIsNone(model.sampler);self.assertIsNone(model.device)

    def test_raw_h1_delegates_once_and_preserves_original_inputs(self):
        image=np.full((4,3,96,96),.5,np.float32);actions=np.array([[0,511],[200.25,300],[12,36],[511,0]],float)
        before=image.copy();calls=[];seed=[]
        def sample(images,targets):
            calls.append((images.value.copy(),targets.value.copy()))
            return MarkerTensor(np.full((1,3,96,96),1.1,np.float32)),None
        fake=ModuleType('torch')
        fake.cuda=SimpleNamespace(device=lambda i:nullcontext(),current_device=lambda:0,manual_seed=seed.append)
        fake.random=SimpleNamespace(fork_rng=lambda devices:nullcontext())
        fake.inference_mode=lambda:nullcontext();fake.as_tensor=lambda x,device:MarkerTensor(x)
        model=wm.OfficialWorldModel.__new__(wm.OfficialWorldModel)
        model.sampler=SimpleNamespace(sample=sample);model.device=SimpleNamespace(index=0);model.check_limits=lambda:None
        with patch.dict(sys.modules,{'torch':fake}):result=model.predict_next(image,actions,seed=5478)
        self.assertEqual(len(calls),1);self.assertEqual(seed,[5478])
        np.testing.assert_array_equal(image,before)
        np.testing.assert_array_equal(calls[0][0][0],image)
        np.testing.assert_array_equal(calls[0][1][0],wm.normalize_actions(actions))
        np.testing.assert_array_equal(result,np.ones((3,96,96),np.float32))
        self.assertEqual(wm.RAW_CONTRACT['horizon'],1);self.assertEqual(wm.RAW_CONTRACT['euler_steps'],3)
        with self.assertRaises(ValueError):model.predict_next(image[:3],actions)
        with self.assertRaises(ValueError):model.predict_next(image,actions+1000)


if __name__=='__main__':unittest.main()
