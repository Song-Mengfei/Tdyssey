"""Fake stage integration and a model-free/default-CLI boundary."""

from contextlib import redirect_stdout
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
NUMPY=importlib.util.find_spec("numpy") is not None


class CompatibilityEntryChecks(unittest.TestCase):
    def test_worker_budget_receives_the_verified_shared_storage_context(self):
        spec=importlib.util.spec_from_file_location('compat_entry_storage',ROOT/'scripts/validate_wm_compat.py')
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        shared=object()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(output_dir=root/'output',ledger_dir=root/'ledger')
            with patch('tdyssey.wm_compat_guard.WorkerHeartbeat'), \
                    patch('tdyssey.wm_compat_guard.from_handoff',return_value=shared) as handoff_factory, \
                    patch.object(entry,'build_binding',return_value={'controlled':'binding'}), \
                    patch.object(entry,'binding_checker'), \
                    patch('tdyssey.wm_compat_budget.CompatBudget.create',side_effect=RuntimeError('controlled pre-claim stop')) as create:
                handoff=dict(deadline_monotonic=time.monotonic()+80)
                with self.assertRaisesRegex(RuntimeError,'controlled pre-claim stop'):
                    entry.execute(args,handoff)
                handoff_factory.assert_called_once_with(handoff)
                self.assertIs(create.call_args.kwargs['storage'],shared)
            self.assertFalse(args.output_dir.exists())
            self.assertFalse(args.ledger_dir.exists())

    def test_default_plan_is_source_only_and_never_claims_an_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'never-created'
            result=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/validate_wm_compat.py'),
                '--output-dir',str(output),'--checkpoint',str(output/'missing.pth')],
                cwd=ROOT,capture_output=True,text=True,check=True)
            plan=json.loads(result.stdout)
            self.assertFalse(plan['execution_authorized'])
            self.assertEqual(plan['available_forecast_rows'],2268)
            self.assertEqual(plan['planned_forecast_rows'],2304)
            self.assertFalse(output.exists())

    def test_execute_without_separate_authority_fails_before_model_imports(self):
        result=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/validate_wm_compat.py'),'--execute'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('requires separate authorization',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_abbreviated_paths_cannot_override_guardian_roots(self):
        result=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/validate_wm_compat.py'),
            '--output','ignored'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('unrecognized arguments: --output',result.stderr)

    def test_worker_paths_must_match_both_guardian_roots(self):
        spec=importlib.util.spec_from_file_location('compat_entry',ROOT/'scripts/validate_wm_compat.py')
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(output_dir=root/'output',ledger_dir=root/'ledger')
            good=dict(output_dir=str(args.output_dir.resolve()),ledger_dir=str(args.ledger_dir.resolve()))
            entry.validate_handoff_paths(args,good)
            for name in ('output_dir','ledger_dir'):
                bad=dict(good,**{name:str(root/'unmonitored')})
                with self.assertRaisesRegex(ValueError,name):
                    entry.validate_handoff_paths(args,bad)
            with self.assertRaises(ValueError):
                entry.validate_handoff_paths(args,dict(output_dir=good['output_dir'],ledger_dir=None))
            self.assertFalse(args.output_dir.exists())
            self.assertFalse(args.ledger_dir.exists())

    def test_changed_saved_array_cannot_be_rebound_as_an_original_input(self):
        spec=importlib.util.spec_from_file_location('compat_entry_inventory',ROOT/'scripts/validate_wm_compat.py')
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=root/'copied-inputs'; run.mkdir()
            array=run/'L00-hold-physical.npz'; array.write_bytes(b'controlled original bytes')
            inventory=root/'sealed.json'
            inventory.write_text(json.dumps(dict(files=[dict(path='X:/old/'+entry.FROZEN_RUN_NAME+'/'+array.name,
                bytes=array.stat().st_size,sha256=entry.digest(array))])))
            sealed=hashlib.sha256(inventory.read_bytes()).hexdigest()
            with patch.object(entry,'FROZEN_INVENTORY_SHA256',sealed):
                identity=entry.verify_frozen_assets([array],run,root/'unused-checkpoint',inventory)
                self.assertEqual(identity[str(array.resolve())],entry.digest(array))
                array.write_bytes(b'controlled modified bytes')
                with self.assertRaisesRegex(ValueError,'sealed inventory'):
                    entry.verify_frozen_assets([array],run,root/'unused-checkpoint',inventory)
                inventory.write_bytes(b'changed manifest')
                with self.assertRaisesRegex(ValueError,'provenance inventory'):
                    entry.verify_frozen_assets([],run,root/'unused-checkpoint',inventory)

    def test_real_entry_requires_fixed_launch_seal_and_matching_review_authority(self):
        from tdyssey.wm_compat_identity import ATTEMPT_ID, AUTHORIZATION_ID, LEDGER, OUTPUT, REGISTRY
        spec=importlib.util.spec_from_file_location('compat_entry_seal',ROOT/'scripts/validate_wm_compat.py')
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(output_dir=root/'artifacts'/OUTPUT,ledger_dir=root/'artifacts'/LEDGER,
                                 authorization_id=AUTHORIZATION_ID,reviewed_commit='a'*40)
            seal=dict(path=str(root/'artifacts'/REGISTRY/'launcher-attempt.json'),attempt_id=ATTEMPT_ID,
                      authorization_id=args.authorization_id,reviewed_commit=args.reviewed_commit)
            with patch.object(entry,'ROOT',root):
                entry.validate_execution_authority(args,dict(launch_seal=seal))
                for changed in (None,dict(seal,authorization_id='different'),dict(seal,reviewed_commit='b'*40),
                                dict(seal,path=str(root/'artifacts/wm-input-compat-alternate/launcher-attempt.json')),
                                dict(seal,attempt_id='wm-input-compat-eval-v1')):
                    with self.assertRaisesRegex(ValueError,'permanent launcher seal'):
                        entry.validate_execution_authority(args,dict(launch_seal=changed))
            self.assertFalse((root/'artifacts').exists())

    def test_old_v1_or_wrong_fixed_identity_is_rejected_before_execution(self):
        from tdyssey.wm_compat_identity import ATTEMPT_ID, AUTHORIZATION_ID, LEDGER, OUTPUT, REGISTRY
        spec=importlib.util.spec_from_file_location('compat_entry_fixed_identity',ROOT/'scripts/validate_wm_compat.py')
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixed_output,fixed_ledger=root/'artifacts'/OUTPUT,root/'artifacts'/LEDGER
            fixed_seal=dict(path=str(root/'artifacts'/REGISTRY/'launcher-attempt.json'),attempt_id=ATTEMPT_ID,
                            authorization_id=AUTHORIZATION_ID,reviewed_commit='a'*40)
            cases=(
                dict(name='old_v1_seal',seal=dict(fixed_seal,path=str(root/'artifacts/wm-input-compat-eval-v1-registry/launcher-attempt.json'))),
                dict(name='old_v1_attempt',seal=dict(fixed_seal,attempt_id='wm-input-compat-eval-v1')),
                dict(name='missing_attempt',seal={key:value for key,value in fixed_seal.items() if key!='attempt_id'}),
                dict(name='wrong_output',output=root/'artifacts'/(OUTPUT+'-other')),
                dict(name='wrong_ledger',ledger=root/'artifacts'/(LEDGER+'-other')),
                dict(name='wrong_authorization',authorization=AUTHORIZATION_ID+'-other'),
                dict(name='old_v1_paths',output=root/'artifacts/wm-input-compat-eval-v1-20261025',
                     ledger=root/'artifacts/wm-input-compat-eval-v1-ledger-20261025'),
            )
            for case in cases:
                with self.subTest(case=case['name']):
                    output,ledger=case.get('output',fixed_output),case.get('ledger',fixed_ledger)
                    authority=case.get('authorization',AUTHORIZATION_ID)
                    seal=case.get('seal',dict(fixed_seal,authorization_id=authority))
                    handoff=dict(output_dir=str(output),ledger_dir=str(ledger),launch_seal=seal)
                    argv=['--execute','--authorization-id',authority,'--reviewed-commit','a'*40,
                          '--output-dir',str(output),'--ledger-dir',str(ledger),
                          '--gpc-root',str(root/'unused-source'),'--checkpoint',str(root/'unused.pth')]
                    with patch.object(entry,'ROOT',root), \
                            patch('tdyssey.wm_compat_guard.verify_guardian',return_value=handoff), \
                            patch.object(entry,'execute',return_value=0) as execute:
                        with self.assertRaises(ValueError):
                            entry.main(argv)
                        execute.assert_not_called()
            self.assertFalse((root/'artifacts').exists())


@unittest.skipUnless(NUMPY,'numpy required for controlled array fixtures')
class CompatibilityStageChecks(unittest.TestCase):
    def budget(self,directory):
        from tdyssey.wm_compat_budget import CompatBudget,EXACT_CHECKPOINT
        root=Path(directory)
        binding=dict(protocol_sha256='a'*64,code_sha256={'fake.py':'b'*64},
            asset_sha256={'fake.npz':'c'*64},checkpoint_sha256=EXACT_CHECKPOINT)
        return CompatBudget.create(root,root/'artifacts/wm-input-compat-test-ledger',
            root/'artifacts/wm-input-compat-test-output/results',binding)

    def test_M_failure_keeps_all_forecast_denominators_and_never_calls_factory(self):
        from tdyssey.wm_compat_evaluation import run_evaluation
        with tempfile.TemporaryDirectory() as directory:
            budget=self.budget(directory)
            def forbidden(): raise AssertionError('backend must not load before M passes')
            result=run_evaluation(dict(branches=[],real_items=[]),budget=budget,
                backend_factory=forbidden,measure=forbidden)
            self.assertEqual(result['budget']['counts']['model_loads'],0)
            self.assertEqual(result['budget']['counts']['images'],0)
            self.assertEqual(result['budget']['stage_status']['L'],'not_attempted')
            self.assertEqual(len(result['rows']),2304)
            self.assertEqual(sum(r['status']=='missing' for r in result['rows']),36)
            self.assertEqual(sum(r['status']=='not_attempted' for r in result['rows']),2268)

    def test_real_transform_audit_retains_counts_residuals_and_range_checks(self):
        import numpy as np
        from tdyssey.wm_compat_evaluation import measure_real_items
        class Budget:
            def check(self): pass
        image=np.ones((3,96,96),dtype=np.float32)
        aggregate={}
        rows=measure_real_items([dict(source_key='controlled-real',rgb=image,shape='T',goal=None)],
            measure=lambda *args:{},budget=Budget(),transform_diagnostics=aggregate)
        self.assertEqual(len(rows),2)
        for arm in ('A','B'):
            self.assertEqual(aggregate[arm]['pixels'],96*96)
            self.assertTrue(aggregate[arm]['all_output_in_range'])
            self.assertTrue(aggregate[arm]['all_output_finite'])
            self.assertFalse(aggregate[arm]['any_input_mutated'])
        self.assertEqual(sum(aggregate['B']['winning_segment_counts'].values()),96*96)
        self.assertEqual(aggregate['B']['residual_rms_rgb_255']['max'],0.)
        self.assertIn('not pooled',aggregate['B']['residual_rms_rgb_255']['quantile_interpretation'])

    def test_backend_cleanup_failure_keeps_primary_stage_error(self):
        from tdyssey.wm_compat_evaluation import run_evaluation
        from tdyssey.wm_compat_budget import BudgetError
        class Backend:
            def close(self): raise OSError('controlled secondary close failure')
        with tempfile.TemporaryDirectory() as directory:
            budget=self.budget(directory)
            with self.assertRaisesRegex(BudgetError,'every fixed T call') as caught:
                run_evaluation(dict(branches=[],real_items=[]),budget=budget,
                    backend_factory=Backend,measure=lambda *args:{},m_gate=lambda rows:dict(passed=True))
            self.assertTrue(any('secondary close failure' in note for note in caught.exception.__notes__))
            self.assertEqual(budget.snapshot()['counts']['model_loads'],1)

    def test_T_failure_is_not_L_failure_and_TF_never_enters_scores(self):
        import numpy as np
        from tdyssey.wm_compat_budget import canonical_plan
        from tdyssey.wm_compat_engine import SavedBranch
        from tdyssey.wm_compat_evaluation import run_evaluation
        from tdyssey.wm_measurement_contract import measurement_frame,coverage_measurement
        frames=np.full((4,3,96,96),.4,dtype=np.float32)
        actions=np.full((4,2),200,dtype=np.float32)
        truth=dict(pose=dict(position=[240,245],angle=0.),pusher=[60,60],coverage=.4)
        branches=[SavedBranch(p['anchor'],p['candidate'],p['shape'],frames,actions[:3],actions,frames,
            truth=[truth]*4,latest=truth,contact_by_h4=False) for p in canonical_plan()['branches'] if p['status']=='available']
        class Backend:
            value=1
            calls=0
            closed=False
            def capture(self): return struct.pack('<I',self.value)
            def restore(self,state): self.value=struct.unpack('<I',state)[0]
            def seed(self,seed): self.value=seed
            def sample(self,images,actions):
                self.calls+=1
                self.value=(self.value*1664525+1013904223)%2**32
                return np.full((3,96,96),-.1,dtype=np.float32)
            def close(self): self.closed=True
        backend=Backend()
        def measure(*args):
            return measurement_frame(state=dict(valid=False,pose=None),shape=dict(valid=False),
                pusher=dict(valid=True,position=[60,60]),coverage=coverage_measurement(None,valid=False))
        with tempfile.TemporaryDirectory() as directory:
            budget=self.budget(directory)
            result=run_evaluation(dict(branches=branches,real_items=[]),budget=budget,
                backend_factory=lambda:backend,measure=measure,m_gate=lambda rows:dict(passed=True))
            self.assertEqual(backend.calls,576)
            self.assertTrue(backend.closed)
            self.assertEqual(result['budget']['counts']['images'],576)
            self.assertEqual(result['budget']['counts']['euler_steps'],1728)
            self.assertEqual(result['budget']['counts']['model_loads'],1)
            self.assertEqual(result['budget']['stage_status']['T'],'failed')
            self.assertEqual(result['budget']['stage_status']['L'],'not_attempted')
            self.assertEqual(sum(r['shape']=='L' and r['status']=='not_attempted' for r in result['rows']),1692)
            self.assertTrue(all(r['truth'] is not None for r in result['rows'] if r['status']=='not_attempted'))
            self.assertFalse(any(r.get('mode')=='TF' for r in result['ranking']['anchor_rows']))

    def test_final_export_failure_cannot_mark_the_ledger_finished(self):
        from tdyssey.wm_compat_evaluation import save_summary
        class Budget:
            finished=False
            def finish(self,*args): self.finished=True
        class Sink:
            count=0
            def record(self,*args,**kwargs):
                self.count+=1
                if self.count==2: raise OSError('controlled final export failure')
        result=dict(rows=[],real_rows=[],M=None,budget=dict(status='active',stop_reason=None))
        budget=Budget()
        with self.assertRaises(OSError):
            save_summary(result,Sink(),budget)
        self.assertFalse(budget.finished)


if __name__=='__main__':
    unittest.main()
