"""Ordinary32-call/state-machine fixtures; no real Torch, assets or physics."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from tdyssey.gpc_conditioning_diagnostic import run_diagnostic, ConditioningBudget
from tdyssey import gpc_adaptation_runtime as runtime

BASE=Path(__file__).resolve().parents[1]/'artifacts/gpc-conditioning-contract-preparation-20261026/fixtures'


class Model:
    def __init__(self, role): self.role=role
    def requires_grad_(self, value):
        if value is not False: raise AssertionError('Gradient enabling forbidden')
        return self
    def eval(self): return self


class MemoryWriter:
    def __init__(self, directory):
        self.directory=directory;directory.mkdir()
        self.documents={}
    def json(self,name,value): self.documents[name]=deepcopy(value);return name
    def raw(self,name,value): self.documents[name]=value.copy();return name
    def write(self,name,value,**kwargs): self.documents[name]=value;return name


class DiagnosticTests(unittest.TestCase):
    def run_fixture(self, mismatch=False):
        BASE.mkdir(parents=True,exist_ok=True)
        self.folder=tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.folder.cleanup)
        writer=MemoryWriter(Path(self.folder.name)/'run')
        specs=[dict(window_id='train-cursor-%03d'%cursor,shape='T' if index<4 else 'L')
            for index,cursor in enumerate((24,26,0,2,31,33,1,3))]
        prepared=dict(ready=True,assets={},inputs=dict(windows=specs,denominators={'scenes':6}))
        binding=dict(candidate_id=ConditioningBudget.CANDIDATE_ID,
            authorization_id=ConditioningBudget.AUTHORIZATION_ID,deadline_unix=400.,deadline_monotonic=310.)
        calls,loads,refs,closed,reads=[],[],[],[],[]
        def factory(role,*,state_dict):
            self.assertEqual(state_dict,{'role':role})
            model=Model(role);loads.append(role);refs.append(weakref.ref(model));return model
        def close():
            gc.collect();self.assertIsNone(refs[-1]());closed.append(loads[-1])
        def loader(spec):
            reads.append(spec['window_id'])
            return dict(images=np.ones((5,3,96,96),np.float32),actions=np.zeros((4,2),np.float32),read_receipt=spec)
        class Pair:
            def __init__(self,model,check):self.model=model
            def sample(self,images,actions,arm):
                calls.append((self.model.role,arm))
                return dict(raw=np.zeros((3,96,96),np.float32),rng_entry=b'bad' if mismatch and arm=='B' else b'entry',
                    rng_exit=b'exit',conditioners=[{'marker':index} for index in range(3)])
        arguments=dict(factory=factory,close_model=close,writer=writer,make_pair=Pair,loader=loader,
            state_loader=lambda role,assets:{'role':role},measure=lambda *args:{'state':{'valid':False,'pose':None}},
            panel=lambda *args:b'png-marker',clock=lambda:100.,monotonic=lambda:10.)
        with patch.object(runtime.os,'fsync'):
            if mismatch:
                with self.assertRaisesRegex(ValueError,'pairing differs'):run_diagnostic(prepared,binding,**arguments)
                return writer,calls,loads,closed,reads,None
            report=run_diagnostic(prepared,binding,**arguments)
        return writer,calls,loads,closed,reads,report

    def test_full_fixed_order_uses_eight_inputs_two_serial_models_and32_calls(self):
        writer,calls,loads,closed,reads,report=self.run_fixture()
        self.assertEqual(loads,closed)
        self.assertEqual(loads,['frozen','adapted'])
        self.assertEqual(calls,[(role,arm) for role in loads for _ in range(8) for arm in ('A','B')])
        self.assertEqual(len(reads),8)
        self.assertEqual(report['forecast_records'],32)
        self.assertEqual(report['budget']['completed_euler_steps'],96)
        self.assertEqual(report['budget']['spent_reserved_updates'],0)
        self.assertEqual(report['budget']['spent_reserved_eval_batches'],0)
        self.assertTrue(all(row['diagnostics']['state']['pose'] is None for row in report['rows']))
        self.assertEqual(sum(name.endswith('.npy') for name in writer.documents),32)

    def test_rng_mismatch_stops_with_spent_second_call_and_releases_model(self):
        writer,calls,loads,closed,reads,_=self.run_fixture(mismatch=True)
        self.assertEqual(calls,[('frozen','A'),('frozen','B')])
        self.assertEqual(loads,closed)
        failure=json.loads((writer.directory/'ledger/first-failure.json').read_bytes())
        self.assertEqual(failure['state']['spent_reserved_forecasts'],2)
        self.assertEqual(failure['state']['spent_reserved_euler_steps'],6)
        self.assertEqual(failure['state']['completed_forecasts'],1)


if __name__=='__main__': unittest.main()
