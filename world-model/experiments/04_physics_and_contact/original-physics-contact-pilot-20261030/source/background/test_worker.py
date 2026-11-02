"""Synthetic supervisor checks. No subprocess, physics, numpy, torch or GPU."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from common import PROTOCOL, atomic_json, sha, storage_guard, validate_calibration_partition, validate_partition, bound
from worker import merge_ledgers, selected_ids, truth, weak_motion


def plan_config():
    slots = []
    for letter in 'TLOCXB':
        for phase, count, split in [('formal', 60, 'train_candidate'), ('supplement', 12, 'supplement')]:
            for index in range(count):
                slots.append(dict(slot_id=f'{phase}-{letter}-{index}', phase=phase, letter=letter, split=split))
    config = dict(formal_slot_ids=[s['slot_id'] for s in slots if s['phase'] == 'formal'],
        supplement_slot_ids={letter:[s['slot_id'] for s in slots if s['phase'] == 'supplement' and s['letter'] == letter] for letter in 'TLOCXB'},
        shards=[dict(formal_label='formal-TLO',supplement_label='supplement-TLO',letters='TLO'),
                dict(formal_label='formal-CXB',supplement_label='supplement-CXB',letters='CXB')])
    return config, dict(slots=slots)


class SupervisorChecks(unittest.TestCase):
    def calibration_fixture(self):
        slots=[dict(slot_id=f'cal2-{letter}-{i}',phase='calibration-v2',letter=letter,split='calibration')
               for letter in 'TLOCXB' for i in range(3)]
        return (dict(run_label='calibration-v2',calibration_slot_ids=[row['slot_id'] for row in slots]),
                dict(slots=slots),dict(attempts=[dict(phase='calibration-v1')]))

    def test_calibration_exact18_once(self):
        config,plan,base=self.calibration_fixture()
        self.assertEqual(18,len(validate_calibration_partition(config,plan,base)))
        config['calibration_slot_ids'].pop()
        with self.assertRaisesRegex(ValueError,'18 original'):
            validate_calibration_partition(config,plan,base)

    def test_calibration_alternate_label_not_a_retry(self):
        config,plan,base=self.calibration_fixture();config['run_label']='calibration-v2-retry'
        with self.assertRaisesRegex(ValueError,'run label is fixed'):
            validate_calibration_partition(config,plan,base)

    def test_prior_cal2_attempt_never_resumed(self):
        config,plan,base=self.calibration_fixture();base['attempts'].append(dict(phase='calibration-v2'))
        with self.assertRaisesRegex(ValueError,'calibration-v1 only'):
            validate_calibration_partition(config,plan,base)

    def test_exact_disjoint_release(self):
        config, plan = plan_config()
        by_id = validate_partition(config, plan)
        self.assertEqual(180, len(selected_ids(config, by_id, 'formal', 'TLO')))
        self.assertEqual(24, len(selected_ids(config, by_id, 'supplement', 'OC')))

    def test_overlapping_shard_refused(self):
        config, plan = plan_config()
        config['shards'][1]['letters'] = 'CXO'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_partition(config, plan)

    def test_missing_formal_slot_refused(self):
        config, plan = plan_config()
        config['formal_slot_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'all360'):
            validate_partition(config, plan)

    def test_unbounded_supplement_refused(self):
        config, plan = plan_config()
        config['supplement_slot_ids']['T'].append('invented-outcome-selected-slot')
        with self.assertRaisesRegex(ValueError, '12 original'):
            validate_partition(config, plan)

    def test_source_sha_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'source.txt'
            path.write_text('source', encoding='utf-8')
            self.assertEqual(path.resolve(), bound(dict(path=str(path),sha256=sha(path))))
            with self.assertRaisesRegex(ValueError, 'Identity mismatch'):
                bound(dict(path=str(path),sha256='0'*64))

    def test_formal_all_valid_controls_trigger(self):
        config, plan = plan_config()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'control-metrics.csv'
            fields = ['phase_group','trajectory_id','letter','control','valid','translation_image_px','rotation_deg']
            rows = [dict(phase_group='formal',trajectory_id='formal-T-0',letter='T',control=i,valid=True,
                         translation_image_px=.1 if i < 2 else 2,rotation_deg=5 if i == 0 else 0) for i in range(3)]
            # Invalid data and cal/supp rows cannot dilute or boost the formal denominator.
            rows += [dict(phase_group='formal',trajectory_id='formal-T-0',letter='T',control=3,valid=False,translation_image_px=99,rotation_deg=99),
                     dict(phase_group='calibration-v1',trajectory_id='cal-T',letter='T',control=0,valid=True,translation_image_px=99,rotation_deg=99),
                     dict(phase_group='supplement',trajectory_id='supp-T',letter='T',control=0,valid=True,translation_image_px=99,rotation_deg=99)]
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            result = weak_motion(path, config)['decisions'][0]
            self.assertEqual(3, result['H1_valid_actual_controls'])
            self.assertTrue(result['translation_trigger'])
            self.assertFalse(result['rotation_trigger'])
            self.assertEqual(['formal-T-0'],result['H1_rotation_at_least5deg_trajectory_ids'])

    def test_exact_half_is_not_translation_trigger(self):
        config, plan = plan_config()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'control-metrics.csv'
            path.write_text('phase_group,trajectory_id,letter,control,valid,translation_image_px,rotation_deg\n'
                'formal,formal-T-0,T,0,True,0.49,5\nformal,formal-T-0,T,1,True,0.5,0\n',encoding='utf-8')
            decision=weak_motion(path,config)['decisions'][0]
            self.assertFalse(decision['triggered'])

    def test_wrong_report_letter_refused(self):
        config, plan = plan_config()
        by_id={s['slot_id']:s for s in plan['slots']}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'control-metrics.csv'
            path.write_text('phase_group,trajectory_id,letter,control,valid,translation_image_px,rotation_deg\n'
                            'formal,formal-T-0,O,0,True,1,5\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'letter differs'):
                weak_motion(path,config,by_id)

    def make_ledger(self, directory, *, controls=64, valid=64):
        collection = Path(directory)/'collection'
        slot = dict(slot_id='formal-T-0',phase='formal',letter='T',split='train_candidate')
        row = dict(slot,actual_controls=controls,status='completed' if controls==valid==64 else 'rejected_or_early_terminated',
                   plan_sha256='a'*64,physics_protocol_sha256=PROTOCOL)
        attempt=collection/'trajectories'/slot['slot_id'];attempt.mkdir(parents=True)
        raw=attempt/'raw.npz';raw.write_bytes(b'SYNTHETIC ENGINEERING FIXTURE, NOT PHYSICS DATA')
        atomic_json(attempt/'metadata.json',dict(slot=slot,reservation=row,complete=controls==valid==64,
            actual_valid_controls=valid,protocol_sha256=PROTOCOL,models=0,optimizer_updates=0,
            contiguous_frames=True,hidden_replay_or_filled_frames=False,future_training_authorized=False,failure=None,
            raw=dict(file='raw.npz',bytes=raw.stat().st_size,sha256=sha(raw))))
        ledger=collection/'shard-ledger.json'
        atomic_json(ledger,dict(attempts=[row],actual_controls=controls,actual_substeps=controls*200,
            actual_physics_seconds=controls*.1,model_calls=0,optimizer_updates=0))
        config=dict(collection=str(collection),plan=dict(sha256='a'*64),formal_slot_ids=[slot['slot_id']])
        return config,{slot['slot_id']:slot},ledger

    def test_merge_deduplicates_by_rejecting_repeated_reservation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            config,by_id,ledger=self.make_ledger(directory)
            with self.assertRaisesRegex(ValueError,'Duplicate slot'):
                merge_ledgers(config,by_id,[ledger,ledger])

    def test_partial_actual_cost_is_retained_but_not_formal_success(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            config,by_id,ledger=self.make_ledger(directory,controls=7,valid=6)
            merged=merge_ledgers(config,by_id,[ledger])
            self.assertEqual(7,merged['actual_controls']);self.assertEqual(6,merged['actual_valid_controls'])
            self.assertEqual(1400,merged['actual_substeps'])
            with self.assertRaisesRegex(ValueError,'Formal360'):
                merge_ledgers(config,by_id,[ledger],require_formal=True)

    def test_corrupt_raw_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            config,by_id,ledger=self.make_ledger(directory)
            (Path(config['collection'])/'trajectories/formal-T-0/raw.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Durable raw'):
                merge_ledgers(config,by_id,[ledger])

    def test_no_physics_or_model_import(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)
        self.assertFalse(any(name.startswith('tdyssey') for name in sys.modules))
        self.assertFalse(truth('False'))
        with self.assertRaises(ValueError):truth('uncertain')

    def test_storage_guard_deduplicates_roots_and_refuses_without_headroom(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);collection=root/'collection';collection.mkdir()
            (collection/'saved.bin').write_bytes(b'a'*100)
            roots=root/'roots.json';policy=root/'policy.json'
            atomic_json(roots,dict(roots=[str(root),str(collection),str(collection/'saved.bin')]))
            settings=dict(shared_cap_bytes=100000,atomic_reserve_bytes=1000,
                          collection_cap_bytes=10000,per_attempt_atomic_reserve_bytes=100)
            atomic_json(policy,settings)
            config=dict(collection=str(collection),artifact_root=str(root),
                resource_policy=dict(path=str(policy)),storage_roots=dict(path=str(roots)))
            result=storage_guard(config,extra_bytes=200)
            self.assertEqual(100,result['collection_bytes'])
            self.assertEqual(sum(p.stat().st_size for p in (collection/'saved.bin',roots,policy)),result['shared_bytes'])
            self.assertEqual(1400,result['reserved_bytes'])
            settings['shared_cap_bytes']=1;atomic_json(policy,settings)
            with self.assertRaisesRegex(RuntimeError,'Shared3GiB'):
                storage_guard(config)


if __name__=='__main__':
    unittest.main(verbosity=2)
