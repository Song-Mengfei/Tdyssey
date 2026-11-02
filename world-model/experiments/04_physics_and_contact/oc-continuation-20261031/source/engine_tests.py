"""Pure continuation/order contracts; no Torch/model/GPU or simulator call."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
spec=importlib.util.spec_from_file_location('oc_engine_contract',HERE/'engine.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Until the owner copies the identical small common governor into this new
# root, use its existing pure implementation solely for temporary-file tests.
sys.path.append(str(m.AR1_ROOT))

def rows(): return [dict(sample_id=f'w{i}',family_id=f'f{i%1339}',event='moving') for i in range(30935)]
def payload(train):
    return dict(kind='accelerated_native_rgb_complete.v1',model={},optimizer={'param_groups':[{'lr':1e-5}]},scaler={},global_rng={},conditioning_rngs={},
        identity={},legacy_scheduler_provenance={},order={'cursor':1934,'sample_ids':[r['sample_id'] for r in train]},
        progress=dict(lineage_updates=9433,epoch_complete=True,complete_optimizer_boundary=True,pending_attempt=None,unique_samples_exposed=30935,batches_consumed=1934))

class Contracts(unittest.TestCase):
    def test_exact9433boundary_and_pool(self):
        train=rows();p=payload(train);self.assertTrue(m.validate_parent(p,train))
        for key,value in [('lineage_updates',7499),('epoch_complete',False),('pending_attempt',{'unsafe':True})]:
            bad=deepcopy(p);bad['progress'][key]=value
            with self.assertRaises(ValueError):m.validate_parent(bad,train)
        p['order']['sample_ids'][0]='wrong'
        with self.assertRaises(ValueError):m.validate_parent(p,train)

    def test_two_reordered_complete_epochs_and_stable_freeze(self):
        # The exact original pure order helper is reused; no model is imported.
        ar1=m.load_ar1();train=rows()
        with tempfile.TemporaryDirectory(dir=HERE,prefix='offline-order-test-') as temp:
            root=Path(temp);o,r=m.freeze_orders(root,{},train,ar1)
            self.assertEqual(len(o[2]),1934);self.assertEqual(len(o[2][-1]),7)
            self.assertEqual(sorted(i for b in o[3] for i in b),list(range(30935)))
            self.assertNotEqual(o[2],o[3]);self.assertEqual((o,r),m.freeze_orders(root,{},train,ar1))
            with self.assertRaises(ValueError):m.freeze_orders(root,{'epoch_order_seeds':{'2':1,'3':2}},train,ar1)

    def test_epoch_progress_retains_lineage_and_stage(self):
        stage=dict(unique_samples_exposed=30935,batches_consumed=1934,successful_updates=1934,AMP_skips=0)
        p=m.new_progress(3,11367,stage)
        self.assertEqual(p['lineage_updates'],11367);self.assertEqual(p['unique_samples_exposed'],0)
        stage['successful_updates']=0
        self.assertEqual(p['additional_stage']['successful_updates'],1934)

if __name__=='__main__':unittest.main()
