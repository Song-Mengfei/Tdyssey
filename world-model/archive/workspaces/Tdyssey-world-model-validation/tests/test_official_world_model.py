"""CPU arrays/ordinary spies only; no Torch, weights, SDK, physics or process.

Root owns execution. Optional df99f8 parity compiles only small viewer helper
functions, never their module imports or a learned model.
"""
import ast
import builtins
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tdyssey import official_world_model as official

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'artifacts/official-raw-world-model-engineering-20261026/fixtures'
REMOTE = Path(os.environ.get('TDYSSEY_REMOTE_REVIEW_ROOT', ROOT.parent/'Tdyssey-remote-review'))
REMOTE_HELPERS = REMOTE/'tdyssey/gpc_denoiser_training.py'
REMOTE_CLICK = REMOTE/'tdyssey/world_model_click.py'


class ArrayMarker:
    def __init__(self, value):self.value = np.asarray(value).copy()
    @property
    def shape(self):return self.value.shape
    def __getitem__(self, key):return ArrayMarker(self.value[key])
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value.copy()


def fake_torch():
    torch = ModuleType('torch')
    torch.as_tensor = lambda value, *, device: ArrayMarker(value)
    torch.cuda = SimpleNamespace(current_device=lambda:3, device=lambda gpu:nullcontext(), manual_seed=Mock())
    torch.random = SimpleNamespace(fork_rng=lambda **kwargs:nullcontext())
    torch.inference_mode = nullcontext
    return torch


def remote_functions(path, blob, names):
    raw = path.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob:
        raise AssertionError('Cached source differs from df99f8; no source execution permitted')
    tree = ast.parse(raw.decode('utf-8'))
    nodes = [deepcopy(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)) for name in names]
    context = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),context)
    return context


class OfficialWorldModelTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='asset-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root = Path(self.temp.name)
        self.checkpoint = self.root/'denoiser.pt';self.checkpoint.write_bytes(b'ordinary marker, not tensor state')
        self.sidecar = self.root/'denoiser.json'
        self.sidecar.write_text(json.dumps(dict(updates=4,init_sha256=official.INITIAL_SHA256)),encoding='utf-8')
        self.sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def profile(self):
        return official.model_profile('official',self.checkpoint,expected_sha256=self.sha,
            expected_sidecar_sha256=hashlib.sha256(self.sidecar.read_bytes()).hexdigest())

    def test_missing_official_asset_never_falls_back_or_reads_initial_weights(self):
        with patch.object(official,'_file_identity',side_effect=AssertionError('No other asset may be read')):
            absent = official.inspect_profile(official.model_profile('official'))
            self.assertEqual((absent['status'],absent['ready']),('missing',False))
            missing = official.model_profile('official',self.root/'absent.pt',expected_sha256='a'*64)
            self.assertEqual(official.inspect_profile(missing)['status'],'missing')
            with self.assertRaises(FileNotFoundError):
                official.OfficialWorldModel(missing,'marker-source',execute=True)

    def test_known_t1_or_released_initial_cannot_be_labeled_official(self):
        with patch.object(official,'_file_identity',side_effect=AssertionError('Rejected before weight reads')):
            for digest in (official.LEGACY_T1_SHA256,official.INITIAL_SHA256):
                value = official.model_profile('official',self.checkpoint,expected_sha256=digest)
                self.assertEqual(official.inspect_profile(value)['status'],'invalid')
                with self.assertRaises(ValueError):official.OfficialWorldModel(value,'marker-source',execute=True)
        changed = official.model_profile('official');changed['contract']['t1'] = True
        self.assertEqual(official.inspect_profile(changed)['status'],'invalid')

    def test_explicit_sidecar_hash_and_initial_kind_are_json_safe(self):
        inspected = official.inspect_profile(self.profile())
        self.assertTrue(inspected['ready']);self.assertEqual(inspected['checkpoint']['sha256'],self.sha)
        self.assertEqual(inspected['sidecar']['metadata']['updates'],4)
        json.dumps(inspected,allow_nan=False)
        wrong = self.profile();wrong['expected_sidecar_sha256'] = '0'*64
        self.assertEqual(official.inspect_profile(wrong)['status'],'invalid')
        spec = dict(checkpoint=dict(sha256=self.sha,bytes=self.checkpoint.stat().st_size))
        with patch.object(official,'world_model_spec',return_value=spec):
            initial = official.model_profile('initial',self.checkpoint)
            available = official.inspect_profile(initial)
            labeled = official.model_profile('initial',self.checkpoint,label='explicit comparison')
        self.assertEqual(initial['kind'],'initial');self.assertEqual(initial['origin'],'released_pretrained')
        self.assertIsNone(available['sidecar']);self.assertTrue(available['ready'])
        self.assertIn('not official-trained',initial['label'])
        self.assertTrue(labeled['label'].startswith('Released initial GPC (not official-trained)'))
        self.sidecar.write_text('{"updates":4,"init_sha256":"'+official.INITIAL_SHA256+'","bad":NaN}')
        value = official.model_profile('official',self.checkpoint,expected_sha256=self.sha)
        self.assertEqual(official.inspect_profile(value)['status'],'invalid')

    def test_normalization_preserves_remote_clipping_and_source_targets(self):
        actions = np.asarray([[-2,0],[100.25,255.5],[510,511],[512,900]],dtype=np.float64)
        before = actions.tobytes()
        actual = official.normalize_actions(actions)
        self.assertEqual(actual.dtype,np.dtype('float32'))
        np.testing.assert_array_equal(actual[[0,3]],np.asarray([[-1,-1],[1,1]],np.float32))
        self.assertEqual(actions.tobytes(),before)

    def test_invalid_raw_history_or_actions_refuse_before_torch_import(self):
        images = np.full((4,3,96,96),.5,np.float32);actions = np.zeros((4,2),np.float32)
        original_import = builtins.__import__
        def imports(name,*args,**kwargs):
            if name=='torch':raise AssertionError('Invalid inputs must not reach Torch')
            return original_import(name,*args,**kwargs)
        outside = images.copy();outside[0,0,0,0] = -.01
        large = actions.copy();large[3,0] = 512
        with patch('builtins.__import__',side_effect=imports):
            for image,action in ((images[:3],actions),(images.astype('float64'),actions),
                                 (outside,actions),(images,large),(images,actions[:3])):
                with self.assertRaises(ValueError):
                    official.predict_next_frame(object(),image,action,SimpleNamespace(index=0))

    def test_loader_requires_matching_explicit_profile_before_runtime_or_source(self):
        original_import = builtins.__import__
        def imports(name,*args,**kwargs):
            if name=='torch':raise AssertionError('No real Torch import')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=imports), \
                patch.object(official,'load_world_model_source',side_effect=AssertionError('No source runtime')):
            with self.assertRaises(PermissionError):official.load_denoiser_sampler('source',self.checkpoint)
            with self.assertRaises(ValueError):
                official.load_denoiser_sampler('source',self.root/'other.pt',profile=self.profile(),execute=True)

    def test_actual_native_bridge_checks_profile_once_and_never_calls_t1(self):
        from tdyssey import wm_compat_transform as palette
        images = np.full((4,3,96,96),.45,np.float32)
        actions = np.asarray([[0,511],[100.25,200.5],[233.5,300.75],[410.25,125.5]],np.float64)
        normalized = official.normalize_actions(actions)
        raw = np.full((1,3,96,96),-.2,np.float32);raw[0,1,0,0] = 1.2
        calls = []
        class Network:
            def __init__(self,cfg):self.cfg = cfg;self.evaluating = False
            def float(self):return self
            def load_state_dict(self,payload,strict):
                if strict is not True or set(payload)!={'weight'}:raise AssertionError('Strict state load required')
            def to(self,device):return self
            def eval(self):self.evaluating = True;return self
        class Sampler:
            def __init__(self,network,cfg):self.denoiser,self.cfg = network,cfg
            def sample(self,image,action):
                np.testing.assert_array_equal(image.value,images[None])
                np.testing.assert_array_equal(action.value,normalized[None])
                calls.append('native_sample')
                return ArrayMarker(raw),[]
        spec = dict(model=dict(img_channels=3,num_steps_conditioning=4,sigma_data=.5,
            sigma_offset_noise=.1,noise_previous_obs=True),sampler=dict(official._NATIVE_SAMPLER_SPEC))
        source = SimpleNamespace(Denoiser=Network,DenoiserConfig=lambda **kw:SimpleNamespace(**kw),
            InnerModelConfig=lambda **kw:SimpleNamespace(**kw),DiffusionSampler=Sampler,
            DiffusionSamplerConfig=lambda **kw:SimpleNamespace(**kw))
        torch = fake_torch();torch.cuda.is_available = lambda:True
        torch.Tensor = ArrayMarker;torch.isfinite = lambda value:np.isfinite(value.value)
        torch.device = lambda value:SimpleNamespace(index=0)
        torch.load = Mock(return_value={'weight':ArrayMarker(np.asarray([1.],np.float32))})
        profile = self.profile()
        with patch.dict(sys.modules,{'torch':torch}), \
                patch.object(official,'world_model_spec',return_value=spec), \
                patch.object(official,'load_world_model_source',return_value=source) as loader, \
                patch.object(official,'inspect_profile',wraps=official.inspect_profile) as inspect, \
                patch.object(palette,'transform_t1',side_effect=AssertionError('No T1 in official path')) as t1:
            model = official.OfficialWorldModel(profile,'marker-source',execute=True)
            self.assertEqual(inspect.call_count,1)
            actual = model.predict_next(images,actions,seed=9)
            np.testing.assert_array_equal(actual,np.clip(raw[0],0,1))
            self.assertEqual(inspect.call_count,1)  # Prediction does not rehash assets.
            self.assertTrue(model.sampler.denoiser.evaluating)
            self.assertEqual(model.asset_identity['kind'],'official')
            json.dumps(model.asset_identity,allow_nan=False)
            model.close()
            with self.assertRaises(RuntimeError):model.predict_next(images,actions)
        self.assertEqual(calls,['native_sample']);t1.assert_not_called()
        loader.assert_called_once_with('marker-source',spec)
        torch.load.assert_called_once_with(profile['checkpoint'],map_location='cpu',weights_only=True,mmap=True)

    @unittest.skipUnless(REMOTE_HELPERS.is_file() and REMOTE_CLICK.is_file(),'Optional pinned df99f8 source cache unavailable')
    def test_remote_actual_helpers_match_window_scaling_and_clipped_h1(self):
        native = remote_functions(REMOTE_HELPERS,'597c13510969c24f0dcfbe0cac2230470dccd5e8',
            ('model_action','normalize_actions','predict_next_frame'))
        click = remote_functions(REMOTE_CLICK,'86f1cc66c264509b4ca06053e03b45e06ef882f3',('conditioning_window',))
        frames = [np.full((3,96,96),value,np.float32) for value in (.2,.3,.4,.5)]
        past = [(0,511),(100.25,200.5),(233.5,300.75)];target = (410.25,125.5)
        selected,absolute = click['conditioning_window'](frames,past,target,(256,405))
        self.assertEqual([id(value) for value in selected],[id(value) for value in frames])
        self.assertEqual(absolute,past+[target])
        actions = np.asarray(absolute,np.float64);images = np.stack(selected)
        np.testing.assert_array_equal(official.normalize_actions(actions),native['normalize_actions'](actions))
        raw = np.full((1,3,96,96),-.25,np.float32);raw[0,2,0,0] = 1.25
        observed = []
        class Sampler:
            def sample(self,image,action):
                observed.append((image.value.copy(),action.value.copy()))
                return ArrayMarker(raw),[]
        torch = fake_torch();device = SimpleNamespace(index=0)
        with patch.dict(sys.modules,{'torch':torch}):
            expected = native['predict_next_frame'](Sampler(),images,actions,device,seed=5478)
            actual = official.predict_next_frame(Sampler(),images,actions,device,seed=5478)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual,np.clip(raw[0],0,1))
        for left,right in zip(observed[0],observed[1]):np.testing.assert_array_equal(left,right)
        self.assertEqual(torch.cuda.manual_seed.call_count,2)


if __name__ == '__main__':unittest.main()
