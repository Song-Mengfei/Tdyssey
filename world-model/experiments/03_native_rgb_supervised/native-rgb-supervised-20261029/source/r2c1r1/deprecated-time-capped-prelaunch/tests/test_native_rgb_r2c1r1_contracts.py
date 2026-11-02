"""Finite CPU integration contracts for the single R2C1R1 correction.

No actual model, Torch, optimizer, CUDA, environment or physics operation.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class RetryContracts(unittest.TestCase):
    def temporary_prior(self,folder):
        config=json.loads((ROOT/'configs/native_rgb_r2c1r1.json').read_text())
        data=Path(config['prior_ledger_copy']).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'517e9a6e46c3d56d33ea3899037f910090ff749020da892cb2ea4fff9200422e')
        config['ledger_path']=str(Path(folder)/'mock-live-ledger.json');config['prior_ledger_copy']=str(Path(folder)/'mock-prior-ledger.json')
        for key in ('ledger_path','prior_ledger_copy'):Path(config[key]).write_bytes(data)
        return config,data

    def test_stub_checkpoint_writer_to_actual_success_logger_all_labels(self):
        from tdyssey.native_rgb_r2c1r1_training import event
        from tdyssey.native_rgb_r2c1_training import event as failed_event
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);log=base/'events.jsonl';writes=[]
            def stub_checkpoint_writer(path,updates):
                data=('FAKE_CPU_CHECKPOINT_BOUNDARY_'+str(updates)).encode();Path(path).write_bytes(data);writes.append(updates)
                return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),updates=updates)
            for label,update in (('initial_t0',0),('final_complete_boundary',300),('last_complete_after_error',17)):
                info=stub_checkpoint_writer(base/'last.pt',update)
                # Same real expanded-info call as train.save_checkpoint().
                event(log,'checkpoint',label=label,**info,model_seconds=2.)
                row=json.loads(log.read_text().splitlines()[-1]);self.assertEqual(row,dict(kind='checkpoint',label=label,**info,model_seconds=2.))
            self.assertEqual(writes,[0,300,17]);self.assertEqual(len(log.read_text().splitlines()),3)
            with self.assertRaisesRegex(TypeError,"multiple values for argument 'path'"):
                failed_event(base/'should-not-exist.jsonl','checkpoint',label='initial_t0',**info,model_seconds=2.)
            self.assertFalse((base/'should-not-exist.jsonl').exists());self.assertNotIn('torch',sys.modules)

    def test_all_existing_expanded_event_payloads_write_real_JSON(self):
        from tdyssey.native_rgb_r2c1r1_training import event
        payloads={
            'checkpoint':dict(label='initial_t0',path='last.pt',bytes=37,sha256='a'*64,updates=0,model_seconds=1.),
            'native_call_start':dict(method='baseline',cohort='train_visual',step=0,window_ids=['w'],model_seconds=2.),
            'native_cohort':dict(method='final',cohort='val_motion',hard_stop=False,statuses_h1=['ok'],statuses_h4=['finite_out_of_range'],calls=4,model_seconds=3.),
            'attempt_reserved_before_draw':dict(attempt=1,indices=list(range(16)),windows=[dict(path='source-sidecar',letter='T')],model_seconds=4.),
            'microbatch_backward_complete':dict(attempt=1,micro_index=0,indices=[0,1,2,3],loss=.2,model_seconds=5.),
            'update_complete':dict(update=1,attempt=1,loss=.2,gradient_norm=.1,lr=.0001,scaler=128.,cuda_reserved=0,model_seconds=6.),
            'checkpoint_reload_verified':dict(update=1,model_seconds=7.),
            'fatal_stop':dict(type='FakeCPUError',reason='fixture',during_optimizer_step=False,clean_boundary=True,model_seconds=8.)}
        tree=ast.parse((ROOT/'tdyssey/native_rgb_r2c1r1_training.py').read_text())
        kinds={c.args[1].value for c in ast.walk(tree) if isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='event'}
        self.assertEqual(kinds,set(payloads))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'all-events.jsonl'
            for kind,value in payloads.items():event(path,kind,**value)
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows,[dict(kind=kind,**value) for kind,value in payloads.items()])

    def test_training_AST_only_logger_and_exact_residual_guard_changed(self):
        old=ast.parse((ROOT/'tdyssey/native_rgb_r2c1_training.py').read_text());new=ast.parse((ROOT/'tdyssey/native_rgb_r2c1r1_training.py').read_text())
        old_event=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='event')
        new_event=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='event')
        self.assertEqual(new_event.args.args[0].arg,'log_path')
        new_event.args.args[0].arg='path'
        for node in ast.walk(new_event):
            if isinstance(node,ast.Name) and node.id=='log_path':node.id='path'
        self.assertEqual(ast.dump(new_event),ast.dump(old_event))
        a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='train');b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='train')
        guard=b.body[2];guardtext=ast.unparse(guard)
        self.assertIn('1172.641',guardtext);self.assertIn('872.641',guardtext);self.assertIn('config[',guardtext)
        b.body[2]=deepcopy(a.body[2]);self.assertEqual(ast.dump(old),ast.dump(new))

    def test_actual_bad_preflight_never_reaches_model_entry(self):
        from scripts import native_rgb_r2c1r1_run as cli
        config=json.loads((ROOT/'configs/native_rgb_r2c1r1.json').read_text());config['budget']['max_model_seconds']=1200.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad-config.json';path.write_text(json.dumps(config))
            with patch.object(cli,'train',side_effect=AssertionError('model entry forbidden')) as model_entry:
                with self.assertRaises(ValueError):cli.run(path,execute=True)
                model_entry.assert_not_called()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),['bad-config.json'])
        self.assertNotIn('torch',sys.modules)

    def test_real_temporary_prior_SHA_authority_and_history_guards(self):
        from tdyssey.native_rgb_r2c1r1_budget import R2C1R1ModelWorkBudget,validate_authorization,verify_closed_prior
        from tdyssey.native_rgb_training import CostLedger
        with tempfile.TemporaryDirectory() as tmp:
            config,raw=self.temporary_prior(tmp);prior=verify_closed_prior(config);validate_authorization(config)
            bad=deepcopy(config);bad['budget']['max_model_seconds']=1200.
            with self.assertRaises(ValueError):validate_authorization(bad)
            authorization=json.loads(Path(config['parent_authorization']).read_text());authorization['parent_authorized']=False
            auth=Path(tmp)/'bad-parent.json';auth.write_text(json.dumps(authorization));bad=deepcopy(config);bad['parent_authorization']=str(auth);bad['parent_authorization_sha256']=hashlib.sha256(auth.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):validate_authorization(bad)
            Path(config['prior_ledger_copy']).write_bytes(raw+b'\n')
            with self.assertRaises(ValueError):verify_closed_prior(config)
            Path(config['prior_ledger_copy']).write_bytes(raw)
            clock=[100.]
            with patch('tdyssey.native_rgb_stage_budget.time.monotonic',side_effect=lambda:clock[0]):
                ledger=CostLedger(config['ledger_path'],3000,5400.);stage=ledger.start('R2C1R1-test-only-no-model')
                budget=R2C1R1ModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:{}))
                self.assertEqual(ledger.snapshot()['stages'],prior['stages']);self.assertEqual(budget.aggregate_seconds_before,895.8430000000008)
                changed=ledger.snapshot();changed['stages'][0]['status']='tampered-history';Path(config['ledger_path']).write_text(json.dumps(changed))
                with self.assertRaises(ValueError):R2C1R1ModelWorkBudget(stage,config=config,storage=object())
                stage.finish('mock-closed')
        self.assertNotIn('torch',sys.modules)

    def test300_temporary_attempts_remain_charged_without_optimizer_or_refund(self):
        from tdyssey.native_rgb_r2c1r1_budget import R2C1R1ModelWorkBudget
        from tdyssey.native_rgb_stage_budget import UpdateLimitReached
        from tdyssey.native_rgb_training import CostLedger
        with tempfile.TemporaryDirectory() as tmp:
            config,_=self.temporary_prior(tmp);clock=[100.]
            with patch('tdyssey.native_rgb_stage_budget.time.monotonic',side_effect=lambda:clock[0]):
                ledger=CostLedger(config['ledger_path'],3000,5400.);stage=ledger.start('R2C1R1-test-only-no-model')
                budget=R2C1R1ModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:{}))
                for _ in range(300):budget.reserve_update()
                with self.assertRaises(UpdateLimitReached):budget.reserve_update()
                self.assertEqual(ledger.snapshot()['charged_updates'],550);self.assertEqual(ledger.snapshot()['completed_updates'],250)
                budget.finish('all-mock-attempts-still-charged')
                self.assertIsNone(ledger.snapshot()['active_stage']);self.assertEqual(ledger.snapshot()['stages'][-1]['reserved_updates'],300)

    def test_remaining_seconds_cutoff_earlier_tail_and_combined_round_cap(self):
        from tdyssey.native_rgb_r2c1r1_budget import FAILED_MODEL_SECONDS,R2C1R1ModelWorkBudget
        from tdyssey.native_rgb_stage_budget import UpdateLimitReached
        from tdyssey.native_rgb_training import BudgetExceeded,CostLedger
        with tempfile.TemporaryDirectory() as tmp:
            config,_=self.temporary_prior(tmp);clock=[100.]
            with patch('tdyssey.native_rgb_stage_budget.time.monotonic',side_effect=lambda:clock[0]):
                ledger=CostLedger(config['ledger_path'],3000,5400.);stage=ledger.start('R2C1R1-test-only-no-model')
                budget=R2C1R1ModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:{}))
                self.assertAlmostEqual(budget.effective_time_cap,1172.641);self.assertAlmostEqual(budget.update_cutoff,872.641)
                self.assertAlmostEqual(budget.record_baseline(500.),317.641)
                clock[0]=100.+318.
                with self.assertRaises(UpdateLimitReached):budget.reserve_update()
                self.assertEqual(ledger.snapshot()['charged_updates'],250)
                clock[0]=100.+1172.641
                with self.assertRaises(BudgetExceeded):budget.require_operation(0.,'must not start at time cap')
                budget.finish('mock-time-cap')
                self.assertLessEqual(FAILED_MODEL_SECONDS+budget.elapsed,1200.+1e-8)
                self.assertAlmostEqual(ledger.snapshot()['charged_seconds'],895.8430000000008+budget.elapsed)


if __name__=='__main__':unittest.main()
