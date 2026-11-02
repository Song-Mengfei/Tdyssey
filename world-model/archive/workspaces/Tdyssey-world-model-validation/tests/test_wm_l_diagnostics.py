"""Synthetic numeric fixtures only; no RGB decoding, model, or environment."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from tdyssey.wm_l_diagnostics import analyze_records, save_artifacts


def fixtures():
    branches=[]
    for i in range(12):
        for j,c in enumerate(('hold','toward','side','away')):
            value=.1*(j+1)+(.4 if i==2 else 0)
            truth=dict(pose=dict(position=[0.,0.],angle=0.),pusher=[0.,0.],coverage=value)
            branches.append(dict(anchor=f'L{i:02d}',candidate=c,latest=deepcopy(truth),
                truth=[deepcopy(truth) for _ in range(4)],contact_by_h4=c=='toward'))
    rows=[]
    for a in ('L00','L02'):
        for c in ('hold','toward','side','away'):
            truth=next(b for b in branches if (b['anchor'],b['candidate'])==(a,c))['truth']
            for arm in ('A','B'):
                for seed in (5478,5479,5480):
                    for mode,hs in (('AR',(1,2,3,4)),('TF',(2,4))):
                        for h in hs:
                            value=truth[h-1]['coverage']
                            measurement=dict(state=dict(valid=True,pose=dict(position=[0.,0.],angle=0.)),
                                shape=dict(valid=True,reason='ok'),pusher=dict(valid=True,position=[0.,0.]),
                                coverage=dict(valid=True,estimate=value,interval=[max(0,value-.03),min(1,value+.03)],
                                    uncertainty_kind='development_absolute_error_bound',uncertainty_valid=True))
                            rows.append(dict(key=f'{a}/{c}/{seed}/{mode}/{arm}/H{h}',anchor=a,candidate=c,
                                shape='L',arm=arm,seed=seed,mode=mode,horizon=h,status='completed',
                                measurement=measurement,truth=deepcopy(truth[h-1]),contact_by_h4=c=='toward'))
    return rows,branches


def cell(result,arm='B',mode='AR',horizon=1,stratum='all'):
    item=next(c for c in result['numeric_statistics']['cells'] if
        (c['mode'],c['horizon'],c['stratum'])==(mode,horizon,stratum))
    return item,item['arms'][arm]


class LDiagnosticsTests(unittest.TestCase):
    def test_first_failure_is_earliest_and_independent_parts_survive(self):
        rows,branches=fixtures()
        for row in rows:
            if (row['anchor'],row['candidate'],row['arm'],row['seed'],row['mode'])!=('L00','hold','B',5478,'AR'):
                continue
            if row['horizon'] in (2,4):
                row['measurement']['shape'].update(valid=False,reason='rigid_shape_mismatch')
            if row['horizon']==3:
                row['measurement']['state']['pose']['position']=[6.,0.]
        result=analyze_records(rows,branches)
        stream=next(r for r in result['first_failures']['streams'] if
            (r['anchor'],r['candidate'],r['arm'],r['seed'],r['mode'])==('L00','hold','B',5478,'AR'))
        self.assertEqual(stream['first_failures']['validity_shape']['horizon'],2)
        self.assertIsNone(stream['first_failures']['validity_object'])
        self.assertIsNone(stream['first_failures']['validity_coverage_independent'])
        self.assertEqual(stream['first_failures']['physical_position']['horizon'],3)
        self.assertEqual(stream['first_failures']['physical_coverage']['kind'],'unavailable')
        self.assertEqual(cell(result,horizon=3)[1]['metrics']['position']['error']['row_statistics']['max'],6.)
        self.assertTrue(cell(result,mode='TF')[0]['H1_reused_from_AR'])
        self.assertFalse(any(c['mode']=='TF' and c['horizon']==3 for c in result['numeric_statistics']['cells']))

    def test_seed_repeats_are_averaged_inside_branch_then_equal_anchors(self):
        rows,branches=fixtures()
        for row in rows:
            if row['arm']=='B' and row['anchor']=='L00':
                if row['candidate']=='hold':row['measurement']['state']['pose']['position']=[9.,0.]
                else:row['measurement']['state'].update(valid=False,pose=None)
        whole,b=cell(analyze_records(rows,branches))
        error=b['metrics']['position']['error']
        self.assertEqual((error['rows'],error['branches'],error['anchors']),(15,5,2))
        self.assertAlmostEqual(error['row_statistics']['mean'],1.8)
        self.assertAlmostEqual(error['equal_anchor_statistics']['mean'],4.5)
        self.assertEqual(whole['denominators']['available_rows_per_arm'],141)
        self.assertEqual(whole['denominators']['planned_rows_per_arm'],144)
        for seed in b['by_seed'].values():self.assertEqual(seed['errors']['position']['count'],5)

    def test_own_B_and_common_rank_subsets_use_identical_baseline_anchors(self):
        rows,branches=fixtures()
        for row in rows:
            if (row['arm']=='A' and row['anchor']=='L00') or (row['arm']=='B' and row['anchor']=='L02'):
                row['measurement']['shape']['valid']=False
        ranking=analyze_records(rows,branches)['ranking_analysis']
        own=next(r for r in ranking['subsets'] if (r['subset'],r['horizon'],r['scoring_mode'])==('B_own',1,'raw_endpoint'))
        common=next(r for r in ranking['subsets'] if (r['subset'],r['horizon'],r['scoring_mode'])==('A_B_common',1,'raw_endpoint'))
        self.assertEqual(own['included_anchors'],['L00'])
        self.assertAlmostEqual(own['physical_baselines_same_anchors']['hold_reward'],.1)
        self.assertAlmostEqual(own['physical_baselines_same_anchors']['uniform_expected_reward'],.25)
        self.assertEqual(own['arms']['B']['mean_regret'],0.)
        self.assertEqual(common['included_anchors'],[])
        self.assertIsNone(common['physical_baselines_same_anchors']['hold_reward'])
        self.assertFalse(ranking['TF_used'])

    def test_abstentions_and_prediction_ties_keep_eligible_pair_denominators(self):
        rows,branches=fixtures()
        for row in rows:
            if row['arm']=='B' and row['anchor']=='L00':
                if row['candidate']=='away' and row['seed']==5480:row['measurement']['shape']['valid']=False
                else:row['measurement']['coverage'].update(estimate=.2,interval=[.17,.23])
        ranking=analyze_records(rows,branches)['ranking_analysis']
        anchor=next(r for r in ranking['anchor_rows'] if (r['arm'],r['anchor'],r['horizon'],r['scoring_mode'])==('B','L00',1,'raw_endpoint'))
        self.assertEqual(anchor['pairs']['eligible_pairs'],6)
        self.assertEqual(anchor['pairs']['abstained_pairs'],3)
        self.assertEqual(anchor['pairs']['scored_pairs'],3)
        self.assertEqual(anchor['pairs']['predicted_ties_among_scored_separable_pairs'],3)
        self.assertEqual(anchor['pairs']['incorrect_scored_pairs'],3)
        self.assertFalse(anchor['selection_eligible'])

    def test_rotation_strict_threshold_and_missing_branch_are_explicit(self):
        rows,branches=fixtures()
        branch=next(b for b in branches if (b['anchor'],b['candidate'])==('L00','hold'))
        for h,angle in enumerate((math.radians(5),math.radians(6),2*math.pi-.01,0)):
            branch['truth'][h]['pose']['angle']=angle
        for row in rows:
            if (row['anchor'],row['candidate'])==('L00','hold'):row['truth']=deepcopy(branch['truth'][row['horizon']-1])
        result=analyze_records(rows,branches)
        self.assertEqual(cell(result,stratum='rotation_gt_5')[0]['denominators']['planned_rows_per_arm'],0)
        self.assertEqual(cell(result,horizon=2,stratum='rotation_gt_5')[0]['denominators']['planned_rows_per_arm'],3)
        missing=next(b for b in result['numeric_statistics']['branch_table'] if (b['anchor'],b['candidate'])==('L01','hold'))
        self.assertEqual(missing['status'],'original_missing')
        stream=next(r for r in result['first_failures']['streams'] if (r['anchor'],r['candidate'])==('L01','hold'))
        self.assertFalse(stream['physical_available'])
        self.assertTrue(all(v is None for v in stream['first_failures'].values()))

    def test_exclusive_output_duplicate_seed_and_conflicting_truth(self):
        rows,branches=fixtures()
        result=analyze_records(rows,branches)
        with tempfile.TemporaryDirectory() as tmp:
            files=save_artifacts(result,tmp)
            for record in files.values():json.loads(Path(record['path']).read_bytes())
            with self.assertRaises(FileExistsError):save_artifacts(result,tmp)
        with self.assertRaises(ValueError):analyze_records(rows+[deepcopy(rows[0])],branches)
        rows[0]['truth']['coverage']=.9
        with self.assertRaisesRegex(ValueError,'Conflicting'):analyze_records(rows,branches)

    def test_missing_truth_or_numeric_chunk_is_unavailable_never_zero_error(self):
        rows,branches=fixtures()
        next(b for b in branches if (b['anchor'],b['candidate'])==('L00','hold'))['truth'][0]['coverage']=None
        for row in rows:
            if (row['anchor'],row['candidate'],row['horizon'])==('L00','hold',1):row['truth']['coverage']=None
        rows=[r for r in rows if (r['anchor'],r['candidate'],r['arm'],r['seed'],r['mode'],r['horizon'])!=('L02','away','B',5480,'AR',1)]
        result=analyze_records(rows,branches)
        whole,b=cell(result)
        metric=b['metrics']['coverage']
        self.assertEqual(metric['truth_missing_rows'],3)
        self.assertEqual(metric['error']['rows']+metric['truth_missing_rows']+metric['measurement_rejected_or_unavailable_rows'],141)
        self.assertEqual(b['metrics']['position']['error']['rows'],23)
        self.assertEqual(whole['denominators']['planned_rows_per_arm'],144)
        stream=next(r for r in result['first_failures']['streams'] if
            (r['anchor'],r['candidate'],r['arm'],r['seed'],r['mode'])==('L00','hold','B',5478,'AR'))
        self.assertEqual(stream['first_failures']['physical_coverage']['kind'],'unavailable')

    def test_contact_onset_is_horizon_specific_and_unknown_is_not_zero(self):
        rows,branches=fixtures()
        next(b for b in branches if (b['anchor'],b['candidate'])==('L00','hold'))['contacts_by_horizon']=[0,0,1,1]
        next(b for b in branches if (b['anchor'],b['candidate'])==('L02','hold'))['contacts_by_horizon']=[0,None,1,0]
        for row in rows:
            if row['arm']=='B' and row['mode']=='AR':
                fail=2 if row['seed']==5478 else 4 if row['seed']==5479 else 3
                if row['candidate']=='hold' and row['anchor'] in ('L00','L02') and row['horizon']==fail:
                    row['measurement']['shape']['valid']=False
        timing=analyze_records(rows,branches)['contact_onset_timing']
        selected={r['seed']:r for r in timing['streams'] if (r['anchor'],r['candidate'])==('L00','hold')}
        self.assertEqual(selected[5478]['first_contact_h'],3)
        self.assertEqual(selected[5478]['relation'],'before')
        self.assertEqual(selected[5479]['relation'],'after')
        self.assertEqual(selected[5480]['relation'],'same')
        unknown=[r for r in timing['streams'] if (r['anchor'],r['candidate'])==('L02','hold')]
        self.assertTrue(all(r['relation']=='unknown' and r['first_contact_h'] is None for r in unknown))
        self.assertTrue(all(r['contacts_by_horizon'][1] is None for r in unknown))
        self.assertEqual(timing['denominators']['planned_seed_streams'],144)
        self.assertEqual(timing['denominators']['available_branches'],47)
        self.assertEqual(timing['branch_unanimous_relation_counts']['mixed'],1)


if __name__=='__main__':unittest.main()
