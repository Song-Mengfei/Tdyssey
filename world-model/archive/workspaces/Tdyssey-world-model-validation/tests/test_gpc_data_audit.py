"""Minimal stdlib prefix, held-payload and physical-support checks."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tdyssey.gpc_data_audit import training_prefix,audit_benchmark_support,audit_training_schedule


class AuditChecks(unittest.TestCase):
    def schedule_fixture(self, root):
        rows=[dict(trajectory_id=s,shape=s,role='train',split='train',initial_state_sha256=hashlib.sha256(s.encode()).hexdigest()) for s in ('T','L')]
        data=dict(path=str(root/'data.json'),sha256='a'*64,episodes=rows)
        dataset=SimpleNamespace(split='train',episodes=rows,offsets=[0,2,4],shape_indices=lambda s:(0,1) if s=='T' else (2,3))
        controls={}
        for row in rows:
            sign=1 if row['shape']=='T' else -1
            controls[row['shape']]=[dict(episode_id=row['shape'],scene=row['initial_state_sha256'],shape=row['shape'],
                control=i,contact=i%2==1,translation=1.,rotation=6. if i==3 else 0.,
                signed_rotation=sign*6. if i==3 else 0.,coverage=.2) for i in range(5)]
        entries=[]
        for cursor in range(500):
            s=('T','L')[cursor%2]; local=(cursor//2)%2; anchor=3+local; row=rows[cursor%2]; target=controls[s][anchor]
            entries.append(dict(cursor=cursor,step=cursor+1,shape=s,trajectory_id=s,initial_state_sha256=row['initial_state_sha256'],
                anchor=anchor,dataset_index=(0 if s=='T' else 2)+local,contact=target['contact'],rotation_degrees=target['signed_rotation'],
                stratum='contact_rotation' if local==0 else 'other'))
        value=dict(kind='tdyssey.gpc_training_schedule.v1',source_manifest_sha256='a'*64,rule_sha256='b'*64,entries=entries)
        value['order_sha256']=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        path=root/'schedule.json'; path.write_text(json.dumps(value))
        baseline=dict(actual_500_prefix=dict(marker='old'),full_corpus_by_role=dict(train=dict(marker='full')),
            independent_validation_support={},held_model_test=dict(payloads_opened=0),operations=dict(model_operations=0))
        return data,dataset,controls,baseline,path,value

    def test_schedule_audit_reads_frozen_entries_and_reports_both_turn_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,dataset,controls,baseline,path,_=self.schedule_fixture(Path(tmp))
            with patch('tdyssey.gpc_data_audit.read_adaptation_manifest',return_value=data), \
                    patch('tdyssey.gpc_data_audit.GPCAdaptationDataset',return_value=dataset), \
                    patch('tdyssey.gpc_data_audit._episode',side_effect=lambda row,guard:(controls[row['shape']],dict(path=row['shape']))), \
                    patch('tdyssey.gpc_data_audit.audit_data_manifest',side_effect=AssertionError('Nontraining corpus read')), \
                    patch('tdyssey.gpc_training_schedule.generate_schedule',side_effect=AssertionError('Schedule regenerated')):
                result=audit_training_schedule('unused',path)
            self.assertFalse(result['schedule_regenerated'])
            self.assertEqual(result['repeated_window_occurrences'],496)
            self.assertEqual(result['by_shape']['T']['windows'],250)
            self.assertEqual(result['weighted_target_statistics']['T']['clockwise_over_5_degrees'],125)
            self.assertEqual(result['weighted_target_statistics']['L']['counterclockwise_over_5_degrees'],125)
            self.assertEqual(result['old500_prefix']['unique_windows'],4)
            self.assertEqual(result['full_training_corpus']['episodes'],2)
            overlap=result['overlap']['new_schedule']['T']
            self.assertEqual((overlap['five_frame_occurrences'],overlap['unique_five_frames']),(1250,6))
            self.assertEqual((overlap['action_occurrences'],overlap['unique_actions']),(1000,5))
            self.assertEqual(overlap['same_scene_unordered_pairs'],31125)
            self.assertEqual(overlap['context_or_action_overlapping_pairs'],31125)

    def test_schedule_order_rehash_cannot_reassign_a_window_to_another_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,dataset,_,_,path,value=self.schedule_fixture(Path(tmp))
            value['entries'][0]['dataset_index']=2
            value['order_sha256']=hashlib.sha256(json.dumps(value['entries'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            path.write_text(json.dumps(value))
            with patch('tdyssey.gpc_data_audit.read_adaptation_manifest',return_value=data), \
                    patch('tdyssey.gpc_data_audit.GPCAdaptationDataset',return_value=dataset), \
                    patch('tdyssey.gpc_data_audit._episode',side_effect=AssertionError('Invalid source read')):
                with self.assertRaisesRegex(ValueError,'source-train window'):
                    audit_training_schedule('unused',path)

    def test_prefix_exact500_alternation_and_repeats(self):
        data=SimpleNamespace(split='train',shape_indices=lambda s:(0,1,2) if s=='T' else (3,4))
        indices=training_prefix(data)
        self.assertEqual(indices[:12],[0,3,1,4,2,3,0,4,1,3,2,4])
        self.assertEqual(len(indices),500)
        self.assertEqual(sum(i<3 for i in indices),250)
        self.assertEqual(500-len(set(indices)),495)
        data.shape_indices=lambda s:(0,) if s=='T' else ()
        with self.assertRaisesRegex(ValueError,'Both T and L'): training_prefix(data)

    def test_held_record_refused_before_any_payload_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'held.json'; path.write_bytes(b'not JSON; never open')
            data=dict(episodes=[dict(role='held_model_test',root=str(root),entry=dict(files=[dict(name=path.name)]))])
            with patch.object(Path,'open',side_effect=AssertionError('Held payload opened')):
                with self.assertRaisesRegex(ValueError,'Held model_test'):
                    audit_benchmark_support(path,data=data)

    def test_pair_boundary_and_shape_specific_termination_without_image_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'data.json'; source.write_text('{}')
            data=dict(path=str(source),episodes=[]); cases=[]
            for shape in ('T','L'):
                for n in range(6):
                    anchor=f'I{shape}{n:02d}'
                    for candidate,value in (('hold',.1),('toward',.15),('side',.95),('away',.1)):
                        image=root/(anchor+'-'+candidate+'.npz'); image.write_bytes(b'never open')
                        if shape=='T' and candidate=='toward': value=.96
                        cases.append(dict(shape=shape,anchor=anchor,candidate=candidate,role='independent',prior_uses=[],
                            initial_state_sha256=hashlib.sha256(anchor.encode()).hexdigest(),truth=[dict(coverage=value)]*4,
                            payload=dict(path=image.name,bytes=image.stat().st_size)))
            path=root/'benchmark.json'; path.write_text(json.dumps(dict(cases=cases)))
            original=Path.open
            def opened(p,*args,**kwargs):
                if p.suffix=='.npz': raise AssertionError('Image archive opened')
                return original(p,*args,**kwargs)
            with patch.object(Path,'open',opened): result=audit_benchmark_support(path,data=data)
            self.assertTrue(result['metadata_and_stat_ready'])
            self.assertEqual(result['pair_rule'],'abs(diff)+1e-12 >= .05')
            # L: exactly .15-.1 is included, as well as the three .95 pairs.
            self.assertEqual(result['physical_endpoint_separation']['L']['1']['separable_pairs'],30)
            self.assertTrue(result['termination']['by_shape']['T']['sufficient'])
            self.assertEqual(result['termination']['by_shape']['L']['positive_cases'],0)
            self.assertFalse(result['termination']['sufficient'])
            self.assertFalse(result['termination']['prediction_certified'])


if __name__=='__main__': unittest.main()
