"""Finite bookkeeping checks; no tensors, SDK, data payloads or native calls."""
import ast
from pathlib import Path
import tempfile
import unittest

from tdyssey.gpc_raw_fulltrain_runtime import RawTrainingBudget, ROLES
from tdyssey.gpc_raw_fulltrain_entry import CANDIDATE,AUTHORIZATION,configuration_template,validate_config


class RawRuntimeTests(unittest.TestCase):
    def account(self,root):
        entries = [dict(trajectory_id='T' if i%2 == 0 else 'L',anchor=i//2+3) for i in range(2328)]
        binding = dict(candidate_id=CANDIDATE,authorization_id=AUTHORIZATION,
            deadline_unix=1900.,deadline_monotonic=1900.)
        return RawTrainingBudget.create(Path(root)/'ledger',binding,entries=entries,
            case_keys=['case'+str(i) for i in range(16)],clock=lambda:100.,monotonic=lambda:100.)

    def train(self,budget,limit=2328):
        token=budget.reserve_model_load('train_init');budget.complete_model_load(token)
        for i in range(limit):
            token=budget.reserve_update(1,dict(cursor=i,entry=budget.entries[i]));budget.complete_update(token)
            if i+1 in (500,2328):budget.exported(i+1)

    def test_full_finite_order(self):
        with tempfile.TemporaryDirectory() as root:
            b=self.account(root);self.train(b);b.training_finished();b.release_model('train_init')
            for role in ROLES[1:]:
                token=b.reserve_model_load(role);b.complete_model_load(token)
                for i in range(16):
                    token=b.reserve_eval(1,dict(role=role,example=i,case_id='case'+str(i)));b.complete_eval(token)
                for i in range(16):
                    token=b.reserve_forecast(1,3,dict(role=role,example=i,case_id='case'+str(i)));b.complete_forecast(token)
                b.release_model(role)
            b.finish();s=b.snapshot()
            self.assertEqual((s['completed_updates'],s['completed_model_loads'],s['completed_forecasts'],s['completed_eval_batches']),(2328,4,48,48))
            self.assertEqual(s['completed_euler_steps'],144);self.assertEqual(s['active_models'],0)
            self.assertLess(b._journal_bytes,16*1024**2)
            with self.assertRaises(RuntimeError):b.reserve_model_load('raw2328')

    def test_500_cannot_be_final(self):
        with tempfile.TemporaryDirectory() as root:
            b=self.account(root);self.train(b,500)
            with self.assertRaises(RuntimeError):b.training_finished()
            self.assertEqual(b.snapshot()['completed_updates'],500)
            self.assertIsNotNone(b.snapshot()['first_reason'])

    def test_wrong_order_spends_failure_without_forward(self):
        with tempfile.TemporaryDirectory() as root:
            b=self.account(root);token=b.reserve_model_load('train_init');b.complete_model_load(token)
            with self.assertRaises(RuntimeError):b.reserve_update(1,dict(cursor=0,entry=b.entries[1]))
            self.assertEqual(b.snapshot()['spent_train_examples'],0)
            with self.assertRaises(RuntimeError):b.reserve_update(1,dict(cursor=0,entry=b.entries[0]))
            with self.assertRaises(RuntimeError):RawTrainingBudget.resume(root,{})

    def test_training_math_remains_pilot03(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'tdyssey/gpc_raw_fulltrain_worker.py').read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='supervised_denoising_loss' for n in calls),1)
        text=ast.unparse(tree)
        self.assertIn('del grads',text);self.assertIn('optimizer.zero_grad(set_to_none=True)',text)
        self.assertIn('clip_grad_norm_(selected, 1.0, error_if_nonfinite=True)',text)
        self.assertNotIn('weighted_denoiser_loss',text);self.assertNotIn('transform_t1',text)
        cfg=configuration_template();validate_config(cfg)
        self.assertEqual(cfg['train']['batch_size'],1);self.assertEqual(cfg['evaluation']['sigma_cond'],None)
        cfg['train']['optimizer']['lr']=.01
        with self.assertRaises(ValueError):validate_config(cfg)


if __name__=='__main__':unittest.main()
