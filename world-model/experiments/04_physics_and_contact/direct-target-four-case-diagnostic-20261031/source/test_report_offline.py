"""Saved/mock report checks only. No simulator construction or execution."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('four_case_saved_report',HERE/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def sample(depth, contact, gap, time):
    return dict(AO_depth=depth,AO_native_contact=contact,AO_native_distance=-depth if contact else None,
        AO_native_depth=depth if contact else None,AO_positive_impulse=contact,AO_impulse_norm=1. if contact else 0.,
        primary_signed_geometric_gap=gap,time_seconds=time,dt=.01)


class SavedOnlyReportChecks(unittest.TestCase):
    def test_during_contact_return_and_later_reexceed_are_separate(self):
        rows=[sample(1.1,True,-1.1,.01),sample(.09,True,-.09,.02),sample(1.2,True,-1.2,.03)]
        value=report.episode_metrics(rows,'AO',0,.1)
        self.assertEqual(value['return_kind'],'return_to_slop_during_continuous_native_contact')
        self.assertTrue(value['continuous_native_contact_until_return'])
        self.assertFalse(value['any_geometric_separation_before_return'])
        self.assertEqual(value['later_reexceed_1_microsteps'],1)
        self.assertEqual(value['later_reexceed_1_peak'],1.2)
        self.assertEqual(len(value['above_1_episodes']),2)
        self.assertEqual(value['strictly_followup_above_1_microsteps'],1)
        self.assertAlmostEqual(value['positive_impulse_seconds'],.03)
        self.assertAlmostEqual(value['strictly_followup_positive_impulse_seconds'],.02)

    def test_return_with_a_manifold_gap_is_not_claimed_continuous(self):
        rows=[sample(1.1,True,-1.1,.01),sample(.3,False,-.3,.02),sample(.09,True,-.09,.03)]
        value=report.episode_metrics(rows,'AO',0,.1)
        self.assertEqual(value['return_kind'],'return_during_contact_after_manifold_gap_or_unknown')
        self.assertFalse(value['continuous_native_contact_until_return'])
        self.assertFalse(value['any_geometric_separation_before_return'])
        self.assertIsNone(value['first_geometric_separation_time_seconds'])

    def test_zero_after_separation_is_not_persistent_contact_recovery(self):
        value=report.episode_metrics([sample(1.1,True,-1.1,.01),sample(0.,False,.8,.02)],'AO',0,.1)
        self.assertEqual(value['return_kind'],'return_after_geometric_separation')
        self.assertFalse(value['first_return_native_contact'])
        self.assertGreater(value['first_return_signed_gap'],0)
        self.assertEqual(value['first_geometric_separation_time_seconds'],.02)
        self.assertFalse(value['any_geometric_separation_before_return'])
        self.assertTrue(value['geometric_separation_at_or_before_return'])

    def test_no_return_remains_finite_budget_unknown(self):
        value=report.episode_metrics([sample(1.1,True,-1.1,.01),sample(.2,True,-.2,.02)],'AO',0,.1)
        self.assertEqual(value['return_kind'],'no_return_to_slop_observed_in_budget')
        self.assertIsNone(value['first_return_time_seconds'])
        missing=report.episode_metrics([], 'AO',3,.1)
        self.assertFalse(missing['observed_after_old_stop'])

    def test_native_missing_is_unknown_and_aw_pre_is_preserved(self):
        events=[dict(kind='agent_wall',phase='pre_solve',points=[dict(distance=-1.3)])]
        self.assertEqual(report.native_info(events,'AW')['minimum_distance'],-1.3)
        self.assertIsNone(report.native_info(events,'AW','post_solve')['minimum_distance'])
        self.assertFalse(report.native_info(events,'AW')['positive_impulse'])

    def test_raw_exact_input_and_real_partial_time(self):
        payload=json.dumps(dict(telemetry=[dict(substeps=[dict(dt=.01,executed=True)]*3)],decisions=[])).encode()
        arrays=dict(states=np.zeros((2,15)),rgb=np.zeros((2,3,96,96),np.uint8),
            requested_actions=np.asarray([[512.,0.]]),passed_actions=np.asarray([[512.,0.]]),
            control_time_seconds=np.asarray([0.,.03]),control_valid=np.asarray([False]),
            telemetry_zlib=np.frombuffer(zlib.compress(payload),np.uint8),
            telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(),np.uint8))
        with TemporaryDirectory(dir=HERE) as directory:
            path=Path(directory)/'saved.npz';np.savez_compressed(path,**arrays)
            value=report.load_raw(path)
            self.assertEqual(value['control_time_seconds'][-1],.03)
            arrays['passed_actions']=np.asarray([[511.,0.]])
            np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'actual environment input'):report.load_raw(path)

    def test_report_writer_is_bounded_to_new_directory(self):
        with TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory)/'new';root.mkdir()
            report.bounded_write(root,root/'a.txt',b'fixture')
            with self.assertRaises(ValueError):report.bounded_write(root,root.parent/'escaped.txt',b'x')
            old=report.CAP
            try:
                report.CAP=7
                with self.assertRaises(RuntimeError):report.bounded_write(root,root/'b.txt',b'xx')
            finally:report.CAP=old

    def test_end_to_end_saved_censored_report_and_decoded_truth_media(self):
        source=HERE.parent/'direct-target-acceptance-20261031'
        approval=json.loads((HERE/'APPROVAL.json').read_text(encoding='utf-8'))
        with TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);(root/'results').mkdir()
            (root/'APPROVAL.json').write_text(json.dumps(approval),encoding='utf-8')
            (root/'runtime-config.json').write_text(json.dumps(dict(run_id='SAVED_FIXTURE_ONLY_NOT_AN_EXPERIMENT',old_root=str(source))),encoding='utf-8')
            for case in approval['cases']:
                ident=case['id'];old=json.loads((source/'collection/attempts'/f'{ident}.json').read_text(encoding='utf-8'))
                shutil.copyfile(source/'collection/raw'/f'{ident}.npz',root/'results'/f'{ident}.npz')
                metadata=dict(case=old['attempt'],initial_identity=old['initial_identity'],prefix_match=dict(matched=True,fixture_copy_only=True),
                    cost=old['cost'],status='saved_fixture_only',hard_stop=None,raw=old['raw'])
                (root/'results'/f'{ident}.json').write_text(json.dumps(metadata),encoding='utf-8')
            with patch.object(report,'storage_preflight',return_value=dict(fixture_only=True)):
                value=report.generate_report(root)
            self.assertEqual(value['actual_saved_cases'],4)
            self.assertEqual(value['report_physics_calls'],0)
            self.assertEqual(value['model_calls'],0)
            self.assertTrue((root/'REPORT.md').exists())
            for case in value['cases']:
                self.assertFalse(case['metrics']['observed_after_old_stop'])
                self.assertEqual(case['video']['fps'],10.)
                self.assertGreater(case['video']['decoded_frames'],0)
            with self.assertRaises(FileExistsError):report.generate_report(root)


if __name__=='__main__':unittest.main()
