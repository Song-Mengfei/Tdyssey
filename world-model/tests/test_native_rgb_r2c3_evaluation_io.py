import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from tdyssey.native_rgb_r2c3_evaluation_io import load_referenced_predictions
from scripts.native_rgb_r2c1_report import metrics_input_identity

class RawEndpointReferenceTest(unittest.TestCase):
    def test_exact_values_and_metric_identity_without_duplicate_pixels(self):
        root=Path(__file__).resolve().parents[1]/'artifacts'
        with tempfile.TemporaryDirectory(dir=root,prefix='r2c3-ref-test-') as d:
            folder=Path(d);ids=['x','y'];steps=[[None]*4 for _ in ids];records=[];refs={k:{}for k in ids}
            for step in range(4):
                raw=np.random.default_rng(10+step).normal(.5,.4,(2,3,96,96)).astype(np.float32)
                path=folder/f'step{step}.npz';np.savez_compressed(path,raw=raw,window_ids=np.asarray(ids));digest=hashlib.sha256(path.read_bytes()).hexdigest()
                records.append(dict(file=path.name,sha256=digest,bytes=path.stat().st_size,window_ids=ids,step=step))
                for i,key in enumerate(ids):
                    steps[i][step]=raw[i].copy()
                    if step in (0,3):refs[key]['H1'if step==0 else'H4']=dict(file=path.name,row=i,step=step,sha256=digest)
            s=dict(window_ids=ids,cohort_identity={'a':'b'},raw_step_files=records,raw_endpoint_refs=refs,status_h1=['finite_out_of_range']*2,status_h4=['finite_out_of_range']*2,hard_stop=False,step_records=[])
            (folder/'cohort-test.json').write_text(json.dumps(s),encoding='utf-8')
            prepared=dict(window_ids=ids,identity={'a':'b'},truth_h1=np.zeros((2,3,96,96),np.uint8),truth_h4=np.zeros((2,3,96,96),np.uint8),histories=np.zeros((2,4,3,96,96),np.float32))
            result,loaded,source=load_referenced_predictions(folder,'test',prepared)
            reference=dict(result,h1=np.stack([x[0]for x in steps]).astype(np.float64),h4=np.stack([x[3]for x in steps]).astype(np.float64))
            self.assertEqual(metrics_input_identity(prepared,result,loaded),metrics_input_identity(prepared,reference,steps))
            self.assertEqual(source['raw_steps_present'],8);self.assertEqual(result['h1'].dtype,np.float64)
            s['raw_endpoint_refs']['x']['H4']['row']=1;(folder/'cohort-test.json').write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'row/window'):load_referenced_predictions(folder,'test',prepared)
            s['raw_endpoint_refs']['x']['H4']['row']=0
            s['raw_endpoint_refs']['x']['H1']=dict(s['raw_endpoint_refs']['x']['H4'],step=0)
            (folder/'cohort-test.json').write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'step/hash'):load_referenced_predictions(folder,'test',prepared)

if __name__=='__main__':unittest.main()
