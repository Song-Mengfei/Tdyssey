"""CPU contracts for the released first epoch; no Torch/models/checkpoints loaded."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_r3b_training as training
FIXTURE_ROOT=Path(__file__).resolve().parents[1]/'artifacts/native-rgb-supervised-20261029/r3b'


def fixture_order():
    rows=[dict(window_id=f'fixture:{i}:H1',family_id=f'family{i}')for i in range(23984)]
    batches=[list(range(i,i+16))for i in range(0,23984,16)]
    with patch.object(training,'ORDER_SHA',training.identity([r['window_id']for r in rows])):
        return training.FrozenFirstEpochOrder(rows,batches,index_plan_sha256='fixture')


class FrozenEpochContracts(unittest.TestCase):
    def test_consumed_cursor_round_trip_and_no_epoch2(self):
        order=fixture_order();unconsumed=order.remaining_batches()
        self.assertEqual((order.batch_cursor,order.total_exposures),(0,0))
        order.consume(unconsumed[0]);state=order.state_dict()
        restored=fixture_order();restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(),state)
        with self.assertRaisesRegex(ValueError,'rendered batch'):
            restored.consume(restored.batches[2])
        self.assertEqual(restored.batch_cursor,1)
        for batch in restored.remaining_batches():restored.consume(batch)
        self.assertEqual((restored.batch_cursor,restored.total_exposures),(1499,23984))
        self.assertEqual(restored.remaining_batches(),[])
        with self.assertRaises(StopIteration):restored.consume(restored.batches[0])

    def test_cursor_checksum_and_exposure_accounting_are_both_required(self):
        order=fixture_order();state=order.state_dict();state['batch_cursor']=1
        with self.assertRaisesRegex(ValueError,'identity'):order.load_state_dict(state)
        state.pop('state_sha256');state['state_sha256']=training.identity(state)
        with self.assertRaisesRegex(ValueError,'exposure'):order.load_state_dict(state)
        self.assertEqual(order.batch_cursor,0)

    def test_skipped_and_abandoned_batches_are_consumed_but_not_lineage_updates(self):
        progress=training.fresh_progress()
        progress.update(stage_attempts=3,stage_successes=1,AMP_skips=1,abandoned_attempts=1,
            formal_draws=48,lineage_updates=6001)
        order=dict(batch_cursor=3,total_exposures=48)
        training.validate_formal_progress(progress,order)
        for change in(dict(lineage_updates=6003),dict(formal_draws=32),dict(stage_successes=2),
                dict(complete_optimizer_boundary=False),dict(epoch_complete=True),dict(stage_attempts=True)):
            with self.assertRaises(ValueError):training.validate_formal_progress(dict(progress,**change),order)
        with self.assertRaisesRegex(RuntimeError,'refusing to replay'):
            training.check_durable_cursor({'order_batch_cursor':3},2)
        training.check_durable_cursor({'order_batch_cursor':3},3)

    def test_original_compact_catalogue_and_frozen_order_without_rendering(self):
        config=training.normalize_config({})
        if not Path(config['data_manifest']).exists():self.skipTest('Accepted local R3A archive absent')
        kwargs,order=training.training_inputs(config)
        self.assertEqual(len(kwargs['catalogue']['h1']),41040)
        self.assertEqual(len(order.rows),23984)
        self.assertEqual(order.identity['window_order_sha256'],training.ORDER_SHA)
        self.assertEqual(order.identity['index_plan_sha256'],training.INDEX_PLAN_SHA)
        self.assertEqual({row['split']for row in order.rows},{'train'})
        self.assertEqual(len({row['window_id']for row in order.rows}),23984)

    def test_failed_engineering_branch_cannot_be_saved_as_formal_checkpoint(self):
        session=training.TrainingSession.__new__(training.TrainingSession)
        session.phase='engineering_probe';session.clean=True
        self.assertEqual(session.checkpoint_if_possible('worker_exception'),dict(saved=False,
            reason='Engineering branch is not a formal checkpoint source'))


class ModelCostContracts(unittest.TestCase):
    def test_prior_cost_preserved_attempt_outcomes_split_and_time_charged_once(self):
        FIXTURE_ROOT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=FIXTURE_ROOT)as temp:
            root=Path(temp);path=root/'ledger.json';prior=root/'prior.json'
            original=dict(kind='native_gpc_rgb_cost.v2-record-only-time',max_seconds=None,max_updates=None,
                active_stage=None,charged_seconds=8346.11,charged_updates=6250,completed_updates=6250,
                stages=[{'immutable_prior_stage':True}],historical_cancelled_experiment={'kept':1899})
            path.write_text(json.dumps(original),encoding='utf-8');prior.write_bytes(path.read_bytes())
            config=dict(ledger_path=str(path),prior_ledger_copy=str(prior),prior_ledger_sha256=training.sha(prior))
            elapsed=[100.]
            with patch.object(training.time,'monotonic',lambda:elapsed[0]):
                ledger=training.R3BModelLedger(config).begin()
                with ledger.model_section('fixture_inference'):
                    elapsed[0]=105.
                ledger.reserve_attempt('engineering');ledger.complete_attempt('engineering')
                ledger.reserve_attempt('formal');ledger.skip_attempt('formal',{'AMP_scale_halved':True})
                ledger.reserve_attempt('formal');ledger.complete_attempt('formal')
                ledger.reserve_attempt('formal');ledger.abandon_attempt('formal','finite-input failure')
                # Heartbeat is called by a separate worker thread during mutations.
                with ThreadPoolExecutor(max_workers=3)as executor:
                    for result in executor.map(lambda _:ledger.heartbeat(),range(12)):self.assertIsNotNone(result)
                snapshot=ledger.snapshot();active=snapshot['active_stage']
                self.assertEqual(active['engineering'],dict(attempts=1,successes=1,skips=0,abandoned=0))
                self.assertEqual(active['formal'],dict(attempts=3,successes=1,skips=1,abandoned=1))
                self.assertEqual(snapshot['charged_seconds'],8346.11+5.)
                self.assertEqual(active['sections'],[dict(label='fixture_inference',seconds=5.)])
                elapsed[0]=107.;closed=ledger.finish('fixture_complete')
            self.assertEqual(closed['stages'][:-1],original['stages'])
            self.assertEqual(closed['historical_cancelled_experiment'],original['historical_cancelled_experiment'])
            self.assertEqual(closed['charged_updates'],6254);self.assertEqual(closed['completed_updates'],6252)
            self.assertEqual(closed['charged_seconds'],8346.11+7.)
            self.assertIsNone(closed['active_stage']);self.assertIsNone(closed['max_seconds'])
            self.assertIsNone(closed['max_updates']);self.assertEqual(training.read(prior),original)

    def test_pending_attempt_cannot_be_reclassified_or_reserved_twice(self):
        FIXTURE_ROOT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=FIXTURE_ROOT)as temp:
            path=Path(temp)/'ledger.json'
            training.atomic_json(path,dict(kind='native_gpc_rgb_cost.v2-record-only-time',max_seconds=None,
                max_updates=None,active_stage=None,charged_seconds=0.,charged_updates=0,completed_updates=0,stages=[]))
            ledger=training.R3BModelLedger(dict(ledger_path=str(path))).begin()
            ledger.reserve_attempt('formal')
            with self.assertRaises(RuntimeError):ledger.reserve_attempt('formal')
            with self.assertRaises(RuntimeError):ledger.complete_attempt('engineering')
            ledger.skip_attempt('formal');value=ledger.finish('fixture_complete')
            self.assertEqual(value['charged_updates'],1);self.assertEqual(value['completed_updates'],0)


if __name__=='__main__':unittest.main()
