import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tdyssey.native_rgb_training import make_window_order, formal_validation_rank,require_complete_native_work
from tdyssey.native_rgb_formal_checkpoints import FormalCheckpointFiles
from tdyssey.native_rgb_stage_budget import ModelWorkBudget, FormalModelWorkBudget, UpdateLimitReached

TEST_TEMP_ROOT=Path(__file__).resolve().parents[1]/'artifacts/native-rgb-supervised-20261029/stage4a'


class Ledger:
    max_updates=3000;max_seconds=5400.
    def __init__(self):
        self.path=Path('fixture-ledger-only.json').resolve()
        self.value=dict(charged_updates=250,completed_updates=250,charged_seconds=868.484,
            active_stage=dict(token='fixture',seconds_before=868.484,reserved_updates=0,completed_updates=0))
    def snapshot(self):return self.value


class Stage:
    def __init__(self):self.ledger=Ledger();self.token='fixture';self.started=0.
    def heartbeat(self):return self.ledger.snapshot()
    def reserve_update(self):
        v=self.ledger.value;v['charged_updates']+=1;v['active_stage']['reserved_updates']+=1;return v
    def complete_update(self):
        v=self.ledger.value;v['completed_updates']+=1;v['active_stage']['completed_updates']+=1;return v


def write_json(path,value):path.write_text(json.dumps(value),encoding='utf-8')
def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def writer(update):
    def save(path):
        payload=('fixture complete boundary '+str(update)).encode();path.write_bytes(payload)
        return dict(path=str(path),updates=update,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    return save


class FormalTrainingInterfaceTests(unittest.TestCase):
    def test_finite_range_failures_continue_but_missing_or_fatal_native_work_stops(self):
        require_complete_native_work(dict(evaluation_complete=True,fatal_error=None,hard_generation_failures=13))
        for record in [dict(evaluation_complete=False,fatal_error=None),
                       dict(evaluation_complete=False,fatal_error={'whole_work_stop':True,'error':'nonfinite'}),
                       dict(evaluation_complete=True,stop_model_work=True)]:
            with self.assertRaises(RuntimeError):require_complete_native_work(record)

    def test_explicit_sampler_identity_and_legacy_default_do_not_take_draws(self):
        dataset=object();expected={'policy':'formal','pools_sha256':'abc'};order=Mock()
        order.identity.return_value=expected
        factory=Mock(return_value=order)
        self.assertIs(make_window_order(dataset,5478,None,sampler_factory=factory,sampling_identity=expected),order)
        order.take.assert_not_called();factory.assert_called_once_with(dataset,5478,None)
        with self.assertRaises(ValueError):make_window_order(dataset,5478,None,sampler_factory=factory,sampling_identity={'policy':'wrong'})
        with patch('tdyssey.native_rgb_training.WindowOrder',return_value='legacy') as legacy:
            self.assertEqual(make_window_order(dataset,5478,[1,2]),'legacy')
            legacy.assert_called_once_with(dataset,5478,[1,2])

    def test_formal_rank_has_no_denoising_fallback_and_rejects_wrong_point(self):
        row={'update':250,'denoising_loss':0.}
        self.assertIsNone(formal_validation_rank(row,lambda r:None))
        self.assertEqual(formal_validation_rank(row,lambda r:[2,.4,250]),(2,.4,250))
        for bad in ([0,.1,129],[0,float('nan'),250],[73,.1,250],[False,.1,250]):
            with self.assertRaises(ValueError):formal_validation_rank(row,lambda r:bad)

    def test_independent500_budget_cap_cutoff_and_measured_tail_keep_old_limits(self):
        storage=Mock();storage.check.return_value={}
        with patch('tdyssey.native_rgb_stage_budget.time.monotonic',return_value=1.):
            stage=Stage();budget=FormalModelWorkBudget(stage,storage=storage)
            for _ in range(500):budget.reserve_update();budget.complete_update()
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            self.assertEqual(stage.ledger.value['charged_updates'],750)
            self.assertEqual(stage.ledger.value['completed_updates'],750)
            budget=FormalModelWorkBudget(Stage(),storage=storage)
            budget.record_baseline(300.);self.assertGreaterEqual(budget.tail_reserve_seconds,690.)
            budget.record_validation(500.);self.assertEqual(budget.update_cutoff,960.)
            self.assertEqual(budget.identity()['kind'],'native-rgb-formal-first500-model-work-budget.v1')
            with self.assertRaises(ValueError):ModelWorkBudget(Stage(),max_attempts=500,max_seconds=1800.,storage=storage)
            with self.assertRaises(ValueError):FormalModelWorkBudget(Stage(),max_attempts=501,storage=storage)
        with patch('tdyssey.native_rgb_stage_budget.time.monotonic',return_value=1620.):
            with self.assertRaises(UpdateLimitReached):FormalModelWorkBudget(Stage(),storage=storage).reserve_update()

    def test_best_promotes_old_last_and_final_best_equals_last_without_copy(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as name:
            out=Path(name);files=FormalCheckpointFiles(out,write_json=write_json,read_json=read_json)
            self.assertTrue(files.consider((1,.2,250),250));files.save_last(250,writer(250))
            self.assertTrue(read_json(out/'best.json')['best_is_last']);self.assertFalse((out/'best.pt').exists())
            old=(out/'last.pt').read_bytes();files.save_last(300,writer(300))
            self.assertEqual((out/'best.pt').read_bytes(),old)
            self.assertFalse(files.consider((1,.2,500),500))
            self.assertTrue(files.consider((0,.3,500),500));files.save_last(500,writer(500))
            self.assertEqual([p.name for p in out.glob('*.pt')],['last.pt'])
            self.assertEqual(read_json(out/'best.json')['updates'],500)
            self.assertTrue(read_json(out/'best.json')['best_is_last'])

    def test_failed_new_last_keeps_previous_complete_best_reference_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as name:
            out=Path(name);files=FormalCheckpointFiles(out,write_json=write_json,read_json=read_json)
            files.consider((1,.2,250),250);files.save_last(250,writer(250))
            def fail(path):raise OSError('fixture failed new atomic save')
            with self.assertRaises(OSError):files.save_last(300,fail)
            saved=read_json(out/'best.json');self.assertEqual(Path(saved['checkpoint']['path']).name,'best.pt')
            self.assertEqual(saved['updates'],250)
            resumed=FormalCheckpointFiles(out,write_json=write_json,read_json=read_json)
            resumed.verify_restored_selection(dict(rank=[1,.2,250],best_update=250))
            with self.assertRaises(ValueError):resumed.verify_restored_selection(None)
            with self.assertRaises(ValueError):resumed.verify_restored_selection(dict(rank=[0,.1,500],best_update=500))
            resumed.save_last(300,writer(300))
            (out/'best.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):FormalCheckpointFiles(out,write_json=write_json,read_json=read_json)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
