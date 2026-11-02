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
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
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
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=Path(tmp)/'all-events.jsonl'
            for kind,value in payloads.items():event(path,kind,**value)
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows,[dict(kind=kind,**value) for kind,value in payloads.items()])

    def test_training_AST_only_logger_and_explicit_time_policy_changed(self):
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
        self.assertIn("work_budget.time_policy != 'record_only_v2'",guardtext)
        self.assertIn('work_budget.max_seconds is not None',guardtext)
        self.assertIn('work_budget.update_cutoff is not None',guardtext)
        self.assertIn("config.get('cap_time', False) is not None",guardtext)
        b.body[2]=deepcopy(a.body[2])
        def assigned(tree,key):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and
                isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].value,ast.Name) and
                n.targets[0].value.id=='result' and isinstance(n.targets[0].slice,ast.Constant) and n.targets[0].slice.value==key)
        prior=assigned(a,'tail_plan');current=assigned(b,'operation_time_estimates')
        self.assertIn('statistics_only=True',ast.unparse(current));self.assertIn('enforced_time_cap=None',ast.unparse(current))
        for parent in ast.walk(b):
            for field,value in ast.iter_fields(parent):
                if isinstance(value,list):
                    for i,node in enumerate(value):
                        if node is current:value[i]=deepcopy(prior)
        oldwhile=next(n for n in ast.walk(a) if isinstance(n,ast.While))
        newwhile=next(n for n in ast.walk(b) if isinstance(n,ast.While))
        self.assertEqual(ast.unparse(newwhile.body[0]),'check()')
        self.assertIn('work_budget.update_cutoff',ast.unparse(oldwhile.body[2]))
        newwhile.body[:1]=deepcopy(oldwhile.body[:3])
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_actual_bad_preflight_never_reaches_model_entry(self):
        from scripts import native_rgb_r2c1r1_run as cli
        config=json.loads((ROOT/'configs/native_rgb_r2c1r1.json').read_text());config['budget']['max_attempts']=301
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=Path(tmp)/'bad-config.json';path.write_text(json.dumps(config))
            with patch.object(cli,'train',side_effect=AssertionError('model entry forbidden')) as model_entry:
                with self.assertRaises(ValueError):cli.run(path,execute=True)
                model_entry.assert_not_called()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),['bad-config.json'])
        self.assertNotIn('torch',sys.modules)

    def test_real_temporary_prior_SHA_authority_and_history_guards(self):
        from tdyssey.native_rgb_r2c1r1_budget import R2C1R1ModelWorkBudget,create_ledger,validate_authorization,verify_closed_prior
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            config,raw=self.temporary_prior(tmp);prior=verify_closed_prior(config);validate_authorization(config)
            bad=deepcopy(config);bad['budget']['max_attempts']=301
            with self.assertRaises(ValueError):validate_authorization(bad)
            authorization=json.loads(Path(config['parent_authorization']).read_text());authorization['parent_authorized']=False
            auth=Path(tmp)/'bad-parent.json';auth.write_text(json.dumps(authorization));bad=deepcopy(config);bad['parent_authorization']=str(auth);bad['parent_authorization_sha256']=hashlib.sha256(auth.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):validate_authorization(bad)
            Path(config['prior_ledger_copy']).write_bytes(raw+b'\n')
            with self.assertRaises(ValueError):verify_closed_prior(config)
            Path(config['prior_ledger_copy']).write_bytes(raw)
            clock=[100.];ledger=create_ledger(config,clock=lambda:clock[0]);migrated=ledger.snapshot()
            self.assertEqual(migrated['stages'],prior['stages']);self.assertEqual(migrated['charged_seconds'],prior['charged_seconds'])
            self.assertEqual(migrated['charged_updates'],250);self.assertEqual(migrated['completed_updates'],250)
            self.assertEqual(migrated['legacy_budget_metadata']['previous_max_seconds'],5400.)
            self.assertEqual(migrated['kind'],'native_gpc_rgb_cost.v2-record-only-time');self.assertIsNone(migrated['max_seconds'])
            self.assertFalse(migrated['time_policy']['hard_time_limits_enabled'])
            self.assertEqual(Path(config['prior_ledger_copy']).read_bytes(),raw)
            stage=ledger.start('R2C1R1-test-only-no-model')
            R2C1R1ModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:{}))
            changed=ledger.snapshot();changed['stages'][0]['status']='tampered-history';Path(config['ledger_path']).write_text(json.dumps(changed))
            with self.assertRaises(ValueError):R2C1R1ModelWorkBudget(stage,config=config,storage=object())
            changed['stages']=prior['stages'];Path(config['ledger_path']).write_text(json.dumps(changed));stage.finish('mock-closed')
        self.assertNotIn('torch',sys.modules)

    def test_300th_attempt_charge_301st_rejection_and_storage_remain_active(self):
        from tdyssey.native_rgb_r2c1r1_budget import R2C1R1ModelWorkBudget,create_ledger
        from tdyssey.native_rgb_stage_budget import UpdateLimitReached
        from tdyssey.native_rgb_training import BudgetExceeded
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            config,_=self.temporary_prior(tmp);clock=[100.]
            ledger=create_ledger(config,clock=lambda:clock[0]);stage=ledger.start('R2C1R1-test-only-no-model');calls=[];fail=[False]
            def storage_check(**values):
                calls.append(values)
                if fail[0]:raise BudgetExceeded('synthetic storage guard')
                return {}
            budget=R2C1R1ModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=storage_check))
            fail[0]=True
            with self.assertRaisesRegex(BudgetExceeded,'storage'):budget.require_operation(0.,'not time constrained')
            self.assertEqual(ledger.snapshot()['charged_updates'],250);fail[0]=False
            # A test-only near-boundary fixture avoids 299 redundant fsyncs.
            value=ledger.snapshot();value['active_stage']['reserved_updates']=299;value['charged_updates']=549
            Path(config['ledger_path']).write_text(json.dumps(value));clock[0]=6100.
            budget.reserve_update()
            self.assertEqual(ledger.snapshot()['charged_updates'],550)
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            self.assertEqual(ledger.snapshot()['charged_updates'],550);self.assertEqual(ledger.snapshot()['completed_updates'],250)
            budget.before_checkpoint(Path(tmp)/'fake-boundary.pt')
            self.assertTrue(any(x.get('force') and x.get('write_reserve_bytes')==256*1024**2 for x in calls))
            budget.finish('all-mock-attempts-still-charged')
            self.assertIsNone(ledger.snapshot()['active_stage']);self.assertEqual(ledger.snapshot()['stages'][-1]['reserved_updates'],300)

    def test_record_only_whole_public_chain_crosses_all_revoked_time_limits(self):
        from tdyssey.native_rgb_r2c1r1_budget import R2C1R1ModelWorkBudget,create_ledger
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            config,_=self.temporary_prior(tmp);clock=[100.]
            ledger=create_ledger(config,clock=lambda:clock[0]);stage=ledger.start('R2C1R1-test-only-no-model');calls=[]
            budget=R2C1R1ModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:calls.append(kw)))
            for seconds in (900.,1200.,5400.,6000.):
                clock[0]=100.+seconds
                stage.heartbeat();budget.require_operation(1000000.,'estimate is metadata, never a time ceiling')
                budget.reserve_update();budget.complete_update();budget.before_checkpoint(Path(tmp)/'not-written.pt')
                self.assertAlmostEqual(ledger.snapshot()['charged_seconds'],895.8430000000008+seconds)
            self.assertIsNone(budget.max_seconds);self.assertIsNone(budget.effective_time_cap);self.assertIsNone(budget.update_cutoff)
            self.assertIsNone(budget.record_baseline(5000.));self.assertEqual(budget.baseline_seconds,5000.)
            clock[0]=6300.;budget.finish('mock-time-recorded-only');report=budget.report()
            self.assertEqual(report['elapsed_model_work_seconds'],6200.);self.assertIsNone(report['effective_time_cap_seconds'])
            self.assertIsNone(report['reserved_tail_seconds']);self.assertFalse(report['hard_time_limits_enabled'])
            self.assertEqual(ledger.snapshot()['charged_updates'],254);self.assertEqual(ledger.snapshot()['completed_updates'],254)
            self.assertEqual(len(calls),16)
            # Even ledger.start after historical+new time exceeds5400 is valid.
            next_stage=ledger.start('test-no-model-after-old-global-time');clock[0]=6400.;next_stage.heartbeat();next_stage.finish('mock-closed')
            self.assertAlmostEqual(ledger.snapshot()['charged_seconds'],895.8430000000008+6300.)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
