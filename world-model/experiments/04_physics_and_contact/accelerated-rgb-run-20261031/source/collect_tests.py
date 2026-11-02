"""Focused pure/saved-data checks; no environment construction or integration."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import collect


def row(depth, *, contact=True, index=0):
    return dict(penetrations=[0.,depth,0.],signed_wall_gaps=dict(left=30.,right=30.,top=30.,bottom=30.),
        index=index,global_microstep=index,contacts=[dict(kind='agent_object',phase='post_solve',
            total_impulse=[1.,0.],points=[])] if contact else [])


class EventChecks(unittest.TestCase):
    def test_continuous_recovery(self):
        t=collect.WatchTracker()
        self.assertIsNone(t.update(row(1.1),3,.1,19,lambda *x:False))
        self.assertIsNone(t.update(row(.036,index=1),3,.1,19,lambda *x:False))
        self.assertEqual(t.events[0]['status'],'RECOVERED_CONTACT_CANDIDATE')
        self.assertEqual(t.first_warning_control,3)
        self.assertFalse(t.active)
        # A second excess creates a separate event, not permanent exemption.
        t.update(row(1.2,index=2),4,.1,19,lambda *x:False)
        self.assertEqual(len(t.events),2)

    def test_separation_is_not_recovery(self):
        t=collect.WatchTracker();t.update(row(4.69),3,.1,19,lambda *x:False)
        result=t.update(row(0.,contact=False,index=1),3,.1,19,lambda *x:True)
        self.assertEqual(result,'SEPARATED_UNRESOLVED')
        self.assertEqual(t.events[0]['status'],'SEPARATED_UNRESOLVED')

    def test_budget_end_is_censored(self):
        t=collect.WatchTracker();t.update(row(1.69),18,.1,19,lambda *x:False)
        t.update(row(.138,index=1),18,.1,19,lambda *x:False)
        self.assertTrue(t.end_block(19))
        self.assertEqual(t.events[0]['status'],'UNRESOLVED_AT_BLOCK_END')

    def test_contact_gap_blocks_recovery(self):
        t=collect.WatchTracker();t.update(row(1.1),3,.1,19,lambda *x:False)
        self.assertEqual(t.update(row(.01,contact=False,index=1),3,.1,19,lambda *x:False),
                         'SEPARATED_UNRESOLVED')

    def test_saved_raw_alignment_and_unmodified_requests(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            p=Path(d)/'raw.npz';a=np.asarray([[512.,0.],[200.,300.]])
            states=np.zeros((3,15));rgb=np.zeros((3,3,96,96),np.uint8)
            telemetry=[dict(substeps=[dict(executed=True,dt=.01,contacts=[]) for _ in range(10)]) for _ in range(2)]
            collect._save_raw(p,states,rgb,a,a,[True,False],telemetry,[])
            with np.load(p,allow_pickle=False) as r:
                self.assertTrue(np.array_equal(r['requested_actions'],a))
                self.assertTrue(np.array_equal(r['passed_actions'],a))
                self.assertEqual(r['rgb'].shape,(3,3,96,96))
                np.testing.assert_allclose(r['control_time_seconds'],[0.,.1,.2],atol=1e-12)
                self.assertEqual(r['control_valid'].tolist(),[True,False])

    def test_planning_pure_and_commitment_length(self):
        import json
        old=json.loads((ROOT/'artifacts/direct-target-acceptance-20261031/plan.json').read_text())
        original=next(x for x in old['attempts'] if x['letter']=='O' and x['scenario']=='translation')
        state=np.asarray(original['initial_state15'],float);before=state.copy()
        slot=dict(original,task='translation',replicate=1,distance_bin=0,
            initial_drive_direction_local=(collect._rotation(-state[4])@np.asarray(original['selected_contact']['drive_direction_world'])).tolist())
        a,detail=collect.block_targets(slot,0,state)
        self.assertEqual(a.shape,(16,2));self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a>=0).all() and (a<=512).all())
        self.assertTrue(np.array_equal(state,before))
        slot['task']='stop_response';a,_=collect.block_targets(slot,1,state)
        self.assertTrue(np.array_equal(a,np.repeat(state[:2][None,:],16,axis=0)))

    def test_four_saved_cases_keep_distinct_outcomes(self):
        import json,zlib
        from shapely.geometry import Point
        folder=ROOT/'artifacts/direct-target-four-case-diagnostic-20261031/results'
        expected={'T':'RECOVERED_CONTACT_CANDIDATE','O':'UNRESOLVED_AT_BLOCK_END',
                  'L':'SEPARATED_UNRESOLVED'}
        for letter,aid in [('T','DT-T-00-0-b0'),('O','DT-O-01-0-b0'),('L','DT-L-05-0-b0')]:
            with np.load(folder/(aid+'.npz'),allow_pickle=False) as raw:
                controls=json.loads(zlib.decompress(raw['telemetry_zlib'].tobytes()))['telemetry']
            tracker=collect.WatchTracker()
            for control_index,info in enumerate(controls):
                for original in info['substeps']:
                    if not original.get('executed'):continue
                    r=dict(original);r.setdefault('global_microstep',control_index*10+r['index'])
                    state=np.asarray(r['post_state'],float)
                    def separated(key,side):
                        if key=='AO':return collect.geometry_at(letter,state).distance(Point(*state[:2]))-15.>collect.EPS
                        return r['signed_wall_gaps'][side]>collect.EPS
                    stopped=tracker.update(r,control_index,.10000000149011612,19,separated)
                    if stopped:break
                if stopped:break
            if tracker.active:tracker.end_block(len(controls))
            self.assertEqual(tracker.events[0]['status'],expected[letter],letter)
        with np.load(folder/'DT-C-06-1-b0.npz',allow_pickle=False) as raw:
            controls=json.loads(zlib.decompress(raw['telemetry_zlib'].tobytes()))['telemetry']
        last=controls[-1]['substeps'][-1]
        self.assertGreater(last['object_collision_vertex_bounds'][2],508.)

    def test_wall_side_is_not_conflated_at_corner(self):
        r=dict(contacts=[dict(kind='object_wall',phase='post_solve',wall_side='right',
            shapes_order=['object','wall'],points=[dict(point_a=[504.,5.],point_b=[504.,5.])])])
        self.assertTrue(collect._native_contact(r,'object_wall','right'))
        self.assertFalse(collect._native_contact(r,'object_wall','top'))


if __name__=='__main__':unittest.main()
