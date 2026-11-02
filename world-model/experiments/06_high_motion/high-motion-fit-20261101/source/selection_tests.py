"""Saved-only checks of quota matching, frozen exposure and family isolation."""
from collections import Counter
from pathlib import Path
import copy,unittest
import selection
from data import FrozenSampler

ROOT=Path(__file__).resolve().parent


class SelectionTests(unittest.TestCase):
    def test_distinct_quota_matching_mixed_translation_and_shortage(self):
        def row(family,angle,distance):
            return dict(letter='O',family_id=family,anchor=3,pose_bin=0,
                motion={'16':dict(signed_rotation_deg=angle,net_translation_imagepx=distance)},
                contact=dict(actual_wall_contact=False,concave_facing_contact_candidate=False))
        # Only four mixed CW families can fill translation. A greedy CW choice
        # must not consume them and falsely reject this feasible16family set.
        rows=[row(f'cw{x}',30,1) for x in range(6)]+[row(f'ccw{x}',-30,1) for x in range(6)]+[row(f'mixed{x}',60,10) for x in range(4)]
        chosen,_=selection.choose_train(rows,'O')
        self.assertEqual(len({r['family_id'] for r in chosen}),16)
        self.assertEqual(Counter(r['quota'] for r in chosen),dict(cw=6,ccw=6,translation=4))
        self.assertTrue(all(r['mixed_translation_and_rotation'] for r in chosen if r['quota']=='translation'))
        with self.assertRaisesRegex(ValueError,'Insufficient distinct O families'):
            selection.choose_train(rows[:-1],'O')

    def test_frozen_order_coverage_balance_and_exact_resume(self):
        index=selection.read(ROOT/'selection/index.json');targets=index['targets'];order=selection.read(ROOT/'selection/order.json')
        batches=order['batches'];self.assertEqual((len(batches),len(targets)),(3000,768))
        seen=set();first=None
        for n,batch in enumerate(batches,1):
            self.assertEqual(len(batch),16)
            self.assertTrue(all(targets[i]['strong'] for i in batch[:12]))
            self.assertEqual(Counter(targets[i]['letter'] for i in batch[:12]),dict(O=4,C=4,B=4))
            seen.update(batch)
            if first is None and len(seen)==768:first=n
        self.assertEqual(first,order['first_all_unique_coverage_batch'])
        for start in range(0,3000,3):
            self.assertEqual(Counter(targets[i]['letter'] for b in batches[start:start+3] for i in b),dict(O=16,C=16,B=16))
        actual=Counter(str(i) for batch in batches for i in batch)
        self.assertEqual(dict(actual),order['planned_target_exposures']);self.assertEqual(sum(actual.values()),48000)
        sampler=FrozenSampler(ROOT)
        for _ in range(first):sampler.next_batch()
        state=sampler.state_dict();expected=sampler.next_batch();restored=FrozenSampler(ROOT);restored.load_state_dict(state)
        self.assertEqual(restored.next_batch(),expected)
        self.assertEqual(set(restored.planned_exposure_prefix()),set(map(str,range(768))))
        bad=copy.deepcopy(state);bad['selection_index_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'identity differs'):restored.load_state_dict(bad)
        state['cursor']=3000;restored.load_state_dict(state)
        with self.assertRaises(StopIteration):restored.next_batch()

    def test_frozen_family_motion_and_original_admission(self):
        index=selection.read(ROOT/'selection/index.json')
        train=index['train48'];val=index['seen_val24'];fail=index['failures3']
        self.assertEqual(len({r['family_id'] for r in train+val+fail}),75)
        for letter in selection.LETTERS:
            rows=[r for r in train if r['letter']==letter]
            self.assertEqual(Counter(r['quota'] for r in rows),dict(cw=6,ccw=6,translation=4))
            self.assertEqual(len({r['family_id'] for r in rows}),16)
        source_ids={r['sample_id'] for r in selection.read(selection.AR1/'selection/train-index.json')}
        for row in train:
            self.assertEqual(row['split'],'train');self.assertTrue(row['committed_future'])
            self.assertGreaterEqual(row['strong_H1_count'],4)
            self.assertEqual(len(row['source_H1_ids']),16);self.assertTrue(set(row['source_H1_ids'])<=source_ids)
        for row in train+val:
            net=row['motion']['16']
            self.assertTrue(net['net_translation_imagepx']>=8 or abs(net['signed_rotation_deg'])>=20)
        self.assertEqual(len(index['fixed_train_panel12']),12)
        for letter in selection.LETTERS:
            self.assertEqual([r['sample_id'] for r in index['fixed_train_panel12'] if r['letter']==letter],
                [r['sample_id'] for r in train if r['letter']==letter][:4])


if __name__=='__main__':unittest.main()
