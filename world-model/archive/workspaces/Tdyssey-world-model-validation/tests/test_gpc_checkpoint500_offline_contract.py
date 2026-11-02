"""NumPy/source-AST checks: no real Torch, GPC, weights, data files or autograd.

Only synthetic arrays and ordinary objects execute the inspected formulas.
Root owns execution in the separately frozen numerical command budget.
"""
import ast
from collections import namedtuple
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import numpy as np
from tdyssey import gpc_adaptation as adaptation
from tdyssey import gpc_supervised_adaptation as objective
from scripts.gpc_checkpoint500_offline_diagnosis import choose_branches, choose_train

ROOT = Path(__file__).resolve().parents[1]
GPC_ROOT = Path(os.environ['TDYSSEY_GPC_ROOT']) if os.environ.get('TDYSSEY_GPC_ROOT') else ROOT.parent/'Tdyssey/third_party/gpc_code'
UPSTREAM = GPC_ROOT/'world_model_train_phase_two/diffusion/denoiser.py'


class ArrayTensor(np.ndarray):
    def __new__(cls, value):return np.asarray(value, dtype=np.float32).view(cls)
    @property
    def device(self):return 'cpu'
    @property
    def requires_grad(self):return False
    def to(self, *args, **kwargs):return self
    def detach(self):return self
    def clone(self):return ArrayTensor(np.array(self, copy=True))
    def size(self):return self.shape
    def exp(self):return ArrayTensor(np.exp(np.asarray(self)))
    def log(self):return ArrayTensor(np.log(np.asarray(self)))
    def sqrt(self):return ArrayTensor(np.sqrt(np.asarray(self)))
    def clamp(self, low, high):return ArrayTensor(np.clip(np.asarray(self), low, high))
    clip = clamp
    def add(self, value):return ArrayTensor(np.asarray(self)+value)
    def sub(self, value):return ArrayTensor(np.asarray(self)-value)
    def div(self, value):return ArrayTensor(np.asarray(self)/value)
    def mul(self, value):return ArrayTensor(np.asarray(self)*value)
    def byte(self):return np.asarray(self, dtype=np.uint8).view(ArrayTensor)


def fake_torch(script=None):
    """Scripted normal draws; named fake modules never load real Torch."""
    torch=ModuleType('torch');nn=ModuleType('torch.nn');functional=ModuleType('torch.nn.functional')
    state=SimpleNamespace(draws=[],cursor=0,mse_calls=0)
    def draw(*shape, **kwargs):
        shape=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],tuple) else tuple(shape)
        state.draws.append(shape);value=script[state.cursor] if script is not None else 0.;state.cursor+=1
        return ArrayTensor(np.broadcast_to(np.asarray(value,dtype=np.float32),shape).copy())
    def mse(x,y):
        state.mse_calls+=1
        return ArrayTensor(np.mean((np.asarray(x)-np.asarray(y))**2,dtype=np.float32))
    torch.Tensor=ArrayTensor;torch.device=str;torch.float32=np.dtype('float32');torch.randn=draw
    torch.randn_like=lambda value:draw(*value.shape);torch.isfinite=lambda value:np.isfinite(np.asarray(value))
    torch.zeros_like=lambda value:ArrayTensor(np.zeros_like(np.asarray(value)))
    functional.mse_loss=mse;nn.functional=functional;torch.nn=nn
    generator=SimpleNamespace(device='cpu',get_state=lambda:SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:[state.cursor])))
    return {'torch':torch,'torch.nn':nn,'torch.nn.functional':functional},state,generator


def native_functions(torch):
    """Compile selected pinned methods, without imports or constructors."""
    source=UPSTREAM.read_bytes();manifest=json.loads((ROOT/'manifests/gpc_world_model.json').read_bytes())
    expected=manifest['source_sha256'].get('world_model_train_phase_two/diffusion/denoiser.py',
        manifest['source_sha256']['gpc_rank_evaluation/diffusion/denoiser.py'])
    if manifest['source_commit']!='a96146e568826ef46134f73c22fdf59a675e7264' or hashlib.sha256(source).hexdigest()!=expected:
        raise AssertionError('Pinned native source differs; no AST method execution permitted')
    tree=ast.parse(source.decode('utf-8'))
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Denoiser')
    names=('setup_training','apply_noise','compute_conditioners','compute_model_output','wrap_model_output','forward')
    nodes=[deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add_dims'))]
    nodes += [deepcopy(next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names]
    for node in nodes:node.decorator_list=[]
    context=dict(torch=torch,Tensor=ArrayTensor,Optional=Optional,Batch=dict,SigmaDistributionConfig=SimpleNamespace,
        F=torch.nn.functional,Conditioners=namedtuple('Conditioners','c_in c_out c_skip c_noise c_noise_cond'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(UPSTREAM),'exec'),context)
    return {name:context[name] for name in names}


class OrdinaryDenoiser:
    """Arithmetic inner-output substitute; never a neural constructor."""
    def __init__(self,torch):
        self.cfg=SimpleNamespace(sigma_data=.5,sigma_offset_noise=.1,noise_previous_obs=True,
            inner_model=SimpleNamespace(num_steps_conditioning=4),upsampling_factor=None)
        self.is_upsampler=False;self.device='cpu';self.sample_sigma_training=None;self.calls=[];self.predicted=[]
        for name,function in native_functions(torch).items():setattr(self,name,MethodType(function,self))
        def inner(noisy,noise,noise_cond,history,action):
            self.calls.append(dict(noisy=np.array(noisy),noise=np.array(noise),noise_cond=np.array(noise_cond),history=np.array(history),action=np.array(action)))
            action_term=np.asarray(action).mean(axis=(1,2))[:,None,None,None]
            return ArrayTensor(.2*np.asarray(noisy)+.15*np.asarray(history[:,:3])+.01*action_term
                +.03*np.asarray(noise)[:,None,None,None]+.02*np.asarray(noise_cond)[:,None,None,None])
        self.inner_model=inner;original=self.wrap_model_output
        def wrap(*args):
            result=original(*args);self.predicted.append(np.array(result));return result
        self.wrap_model_output=wrap
        self.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
    def parameters(self):return iter([ArrayTensor([0.])])


def images(count):
    return ArrayTensor(np.broadcast_to(np.arange(count,dtype=np.float32)[None,:,None,None,None]/20,(1,count,3,96,96)).copy())


class OfflineContractTests(unittest.TestCase):
    @unittest.skipUnless(UPSTREAM.is_file(),'Pinned native source cache missing; set TDYSSEY_GPC_ROOT')
    def test_actual_local_first_target_matches_pinned_native_forward_and_draw_order(self):
        draws=[np.array([.25]),np.linspace(-.3,.3,12).reshape(1,12,1,1),.125,
            np.array([-.5]),np.array([.2,-.1,.4]).reshape(1,3,1,1),-.125]
        batch=dict(image=images(5),action=ArrayTensor(np.linspace(-.8,.8,8).reshape(1,4,2)))
        modules,state,generator=fake_torch(draws)
        with patch.dict('sys.modules',modules):
            native=OrdinaryDenoiser(modules['torch']);native_loss,_=native.forward(batch,'cpu')
        native_calls=state.draws.copy();self.assertEqual(state.mse_calls,1)
        modules,state,generator=fake_torch(draws)
        with patch.dict('sys.modules',modules):
            local=OrdinaryDenoiser(modules['torch'])
            local_loss,metrics=objective.supervised_denoising_loss(local,batch,generator=generator)
            none_condition=local.compute_conditioners(ArrayTensor([.7]),None)
        self.assertEqual(state.draws,native_calls)
        self.assertEqual(native_calls,[(1,),(1,12,1,1),(1,12,96,96),(1,),(1,3,1,1),(1,3,96,96)])
        np.testing.assert_allclose(np.asarray(local_loss),np.asarray(native_loss),rtol=2e-6,atol=1e-7)
        for name in ('noisy','history','action','noise','noise_cond'):
            np.testing.assert_allclose(local.calls[0][name],native.calls[0][name],rtol=2e-6,atol=1e-7)
        self.assertEqual((metrics['target_frames'],metrics['model_output_calls'],metrics['inference_sampler_calls']),(1,1,0))
        self.assertFalse(metrics['actions_renormalized']);self.assertEqual(state.mse_calls,1)
        np.testing.assert_array_equal(np.asarray(none_condition.c_noise_cond),[0.])
        self.assertTrue(np.any(local.calls[0]['noise_cond']!=0))

    @unittest.skipUnless(UPSTREAM.is_file(),'Pinned native source cache missing; set TDYSSEY_GPC_ROOT')
    def test_native_sixteen_frames_have_twelve_terms_and_generated_history(self):
        modules,state,_=fake_torch();batch=dict(image=images(16),action=ArrayTensor(np.linspace(-1,1,32).reshape(1,16,2)))
        original=np.array(batch['image'])
        with patch.dict('sys.modules',modules):
            native=OrdinaryDenoiser(modules['torch']);native.forward(batch,'cpu')
        self.assertEqual((len(native.calls),len(native.predicted),state.mse_calls),(12,12,12))
        np.testing.assert_array_equal(np.asarray(batch['image']),original)
        np.testing.assert_allclose(native.calls[0]['history'],original[:,:4].reshape(1,12,96,96)/.5)
        expected=np.concatenate((original[:,1:4],native.predicted[0][:,None]),axis=1).reshape(1,12,96,96)/.5
        np.testing.assert_allclose(native.calls[1]['history'],expected,rtol=2e-6,atol=1e-7)
        self.assertFalse(np.allclose(native.predicted[0],original[:,4]))
        conditioners=native.compute_conditioners(ArrayTensor([np.exp(-1.2)]),ArrayTensor([np.exp(-1.2)]))
        for index,call in enumerate(native.calls):
            np.testing.assert_array_equal(call['action'],np.asarray(batch['action'])[:,index:index+4])
            np.testing.assert_allclose(np.asarray(call['noisy']/conditioners.c_in),original[:,index+4],rtol=2e-6,atol=1e-7)

    def test_choose_branches_lexfirst_is_physical_not_validity_or_noise_and_keeps_empty(self):
        def row(anchor,candidate,seed,valid):
            return dict(stratum='regression',shape='T',contact_by_h4=False,anchor=anchor,candidate=candidate,
                seed=seed,role='adapted',measurement=dict(shape=dict(valid=valid)))
        rows=[row('T01','hold',5478,True),row('T00','hold',5480,True),row('T00','away',5479,False),row('T00','away',5478,True)]
        before=deepcopy(rows);first=choose_branches(rows);second=choose_branches(list(reversed(rows)))
        self.assertEqual(first,second);self.assertEqual(rows,before);self.assertEqual(len(first),8)
        cell=next(c for c in first if c['shape']=='T' and c['stratum']=='regression' and not c['contact_by_h4'])
        self.assertEqual(cell['selected'],('regression','T','T00','away'));self.assertEqual(cell['available_branches'],3)
        self.assertEqual(sum(c['selected'] is None for c in first),7)
        conflict=deepcopy(rows[-1]);conflict['contact_by_h4']=True
        with self.assertRaisesRegex(ValueError,'Physical label'):choose_branches(rows+[conflict])

    def test_choose_train_first_two_unique_keep_original_within_cell_order_and_cursor(self):
        specs=[('L',True,'x',10),('T',False,'a',3),('T',True,'b',5),('T',False,'a',3),
            ('L',False,'c',7),('T',False,'d',9),('L',True,'e',11),('L',False,'f',12),
            ('T',True,'g',13),('T',False,'third',14)]
        entries=[dict(shape=s,contact=c,trajectory_id=t,anchor=a,original_cursor=i) for i,(s,c,t,a) in enumerate(specs)]
        before=deepcopy(entries);selected=choose_train(entries)
        self.assertEqual(entries,before);self.assertEqual([r['original_cursor'] for r in selected],[1,5,2,8,4,7,0,6])
        for shape in ('T','L'):
            for contact in (False,True):
                rows=[r for r in selected if r['shape']==shape and r['contact'] is contact]
                self.assertEqual(len(rows),2);self.assertLess(rows[0]['original_cursor'],rows[1]['original_cursor'])
                self.assertNotEqual((rows[0]['trajectory_id'],rows[0]['anchor']),(rows[1]['trajectory_id'],rows[1]['anchor']))
                for row in rows:self.assertIs(row,entries[row['original_cursor']])


if __name__=='__main__':unittest.main()
