from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tdyssey.native_rgb_r2c0_data import (LETTERS,R2C0Dataset,R2C0WindowOrder,build_pool_manifest,canonical_hash,choose_split)


def synthetic_pools():
    rows=[]
    for letter in LETTERS:
        for family in range(2):
            for cat in ('translation_only','rotation_only','both','other'):
                for n in range(3):
                    rows.append(dict(dataset_index=len(rows),split='train',horizon=1,letter=letter,family_id=f'{letter}-{family}',
                        window_id=f'{letter}-{family}-{cat}-{n}',anchor=3+n,moving=cat!='other',translation=cat in ('translation_only','both'),rotation=cat in ('rotation_only','both'),category=cat))
                    rows[-1].update(translation_image96_px=2.2 if rows[-1]['translation'] else .3,abs_rotation_deg=5.5 if rows[-1]['rotation'] else .2)
    return build_pool_manifest(rows,{'manifest_sha256':'a'*64,'split':'train'})


class R2C0Tests(unittest.TestCase):
    def test480_quotas_family_balance_and_overlap_once(self):
        pools=synthetic_pools();order=R2C0WindowOrder(pools);indices=order.take(480)
        rows=[pools['windows'][i] for i in indices];quota=Counter((r['letter'],r['moving']) for r in rows)
        for l in LETTERS:
            self.assertEqual(quota[l,True],64);self.assertEqual(quota[l,False],16)
            for moving in (True,False):
                c=Counter(r['family_id'] for r in rows if r['letter']==l and r['moving']==moving)
                self.assertLessEqual(max(c.values())-min(c.values()),1)
        self.assertEqual(len(indices),384+96);self.assertTrue(any(r['category']=='both' for r in rows))
        self.assertEqual(len({i for families in pools['groups'].values() for group in families.values() for cats in group.values() for values in cats.values() for i in values}),len(pools['windows']))

    def test_cross_cycle_restore_exact_and_not_mechanical_repeat(self):
        pools=synthetic_pools();a=R2C0WindowOrder(pools);a.take(479);state=a.state_dict();expected=a.take(33)
        b=R2C0WindowOrder(pools);b.load_state_dict(json.loads(json.dumps(state)));self.assertEqual(expected,b.take(33));self.assertEqual(a.state_dict(),b.state_dict())
        c=R2C0WindowOrder(pools);first=c.take(480);self.assertNotEqual(first,c.take(480))

    def test_identity_and_resigned_queue_tampering_rejected(self):
        pools=synthetic_pools();a=R2C0WindowOrder(pools);a.take(17);state=a.state_dict()
        for mutation in ('identity','cursor','queue'):
            bad=deepcopy(state)
            if mutation=='identity':bad['identity']['data_sha256']='b'*64
            elif mutation=='cursor':bad['cursor']+=1
            else:
                key=next(iter(bad['queues']));bad['queues'][key]['position']=0
            bad.pop('state_sha256');bad['state_sha256']=canonical_hash(bad)
            with self.assertRaises(ValueError):R2C0WindowOrder(pools).load_state_dict(bad)

    def test_empty_group_and_val_leak_fail(self):
        pools=synthetic_pools();rows=deepcopy(pools['windows']);rows[0]['split']='seen_val'
        with self.assertRaises(ValueError):build_pool_manifest(rows,pools['data_identity'])
        rows=deepcopy(pools['windows']);rows[0]['translation_image96_px']=2/512*96
        with self.assertRaises(ValueError):build_pool_manifest(rows,pools['data_identity'])
        rows=[r for r in pools['windows'] if not (r['letter']=='T' and not r['moving'])]
        for i,row in enumerate(rows):row['dataset_index']=i
        with self.assertRaises(ValueError):build_pool_manifest(rows,pools['data_identity'])

    def test_family_split_deterministic_motion_and_training_sign_support(self):
        entries=[];rows=[];signatures={}
        for letter in LETTERS:
            for i in range(6):
                tid=f'{letter}-{i}';entries.append(dict(trajectory_id=tid,letter=letter,initial_area_centroid=[i*100,10],initial_state15=[0,0,0,0,0]+[0]*10,initial_pusher_local=[i*40,0],symmetry=1))
                signatures[tid]={k:f'{tid}-{k}' for k in ('states','actions','initial_state','initial_RGB')}
                rows.append(dict(trajectory_id=tid,anchor=4,translation=i<2,rotation=i>=2,signed_rotation_deg=6 if 2<=i<4 else -6 if i>=4 else 0))
        catalogue=dict(entries=entries,h1=rows,signatures=signatures)
        a=choose_split(catalogue);self.assertEqual(a,choose_split(catalogue));self.assertEqual(a['counts'],{'train':24,'seen_val':12})
        for row in a['selections']:
            self.assertNotEqual(row['translation_family'],row['rotation_family']);self.assertTrue(all(row['train_kept_support'].values()))

    def test_dataset_4history_1target_actions_exact_raw_no_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rgb=np.empty((25,3,96,96),dtype=np.uint8)
            for t in range(25):rgb[t]=t
            actions=np.linspace(0,511,48).reshape(24,2);np.savez(root/'a.npz',rgb=rgb,actions=actions,states=np.zeros((25,15)))
            from tdyssey.native_rgb_r2c0_data import file_sha
            entry=dict(trajectory_id='fixture',family_id='family',letter='T',split='train',file='a.npz',sha256=file_sha(root/'a.npz'))
            rows=[dict(trajectory_id='fixture',anchor=3,split='train'),dict(trajectory_id='fixture',anchor=20,split='train')]
            (root/'h1-windows.json').write_text(json.dumps(rows));(root/'h4-windows.json').write_text(json.dumps(rows))
            with patch('tdyssey.native_rgb_r2c0_data.verify_reference',return_value=({'entries':[entry]},root)):
                data=R2C0Dataset(root/'manifest.json',cache_bytes=1)
            item=data[0];self.assertEqual(item['image'].shape,(5,3,96,96));self.assertEqual(item['action'].shape,(4,2))
            np.testing.assert_array_equal(item['image'],rgb[:5].astype(np.float32)/255)
            np.testing.assert_array_equal(item['action'],actions[:4].astype(np.float32)/511*2-1)
            self.assertEqual(data.cache_nbytes,0);data.close()

    def test_reference_hash_wrong_source_and_path_escape_rejected(self):
        from tdyssey.native_rgb_r2c0_data import PROTOCOL_SHA256,_safe,verify_reference
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'r2b';source.mkdir();(source/'index.json').write_text(json.dumps({'collection_closed':True}))
            manifest=dict(kind='R2C0-R2B-saved-RGB-reference.v1',source_root='r2b',protocol_sha256=PROTOCOL_SHA256,source_identity={'index_sha256':'0'*64})
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_reference(path)
            manifest['source_root']='r2a';path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_reference(path)
            with self.assertRaises(ValueError):_safe(source,'../old-D0.npz')


if __name__=='__main__':unittest.main()
