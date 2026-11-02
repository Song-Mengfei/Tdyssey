"""CPU callbacks/saved arrays only; no native sampler, model, Torch or render."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

from tdyssey import native_rgb_formal_validation as validation
from tdyssey.native_rgb_evaluation import NativePredictor,validate_native_batch_inputs


class ArrayDataset:
    def __init__(self):
        self.windows=[];self.entries=[];self.records=[]
        for i in range(36):
            anchor=3 if i%2==0 else 60;self.windows.append((i,anchor))
            self.entries.append(dict(trajectory_id=f"mock-{i:02d}",shape=validation.LETTERS[i%18],split="seen_val",sha256="a"*64,metadata_sha256="b"*64))
            self.records.append(dict(states=np.arange(65*15,dtype=float).reshape(65,15),
                actions=np.column_stack((np.arange(64),np.arange(64)+100)).astype(np.float32),valid_transition=np.ones(64,bool)))
    def trajectory(self,index):return self.entries[index],self.records[index]
    def render_state(self,*args):raise AssertionError("CPU preparation must not render")


def fixture():
    dataset=ArrayDataset();windows=[]
    for i,(entry,(_,anchor)) in enumerate(zip(dataset.entries,dataset.windows)):
        windows.append(dict(dataset_index=i,anchor=anchor,window_id=f"{entry['trajectory_id']}:{anchor}",trajectory_id=entry['trajectory_id'],
            letter=entry['shape'],split="seen_val",npz_sha256="a"*64,metadata_sha256="b"*64,h1=dict(motion_class="moving"),h4=dict(endpoint_moving=True)))
    histories=np.zeros((36,4,3,96,96),dtype=np.float32)
    for i in range(36):
        for frame in range(4):histories[i,frame].fill((i*4+frame)/200)
    batches=[dict(batch_index=b,batch_size=4,batch_seed=(2**32-2+b)%2**32,positions=list(range(b*4,b*4+4)),
                  window_ids=[w['window_id'] for w in windows[b*4:b*4+4]]) for b in range(9)]
    plan=validation.build_validation_plan(dataset,windows,batches,real_histories=histories)
    return dataset,windows,plan


class FormalValidationTests(unittest.TestCase):
    def run_plan(self,plan,predict,**kwargs):
        return validation.autoregressive_batches(predict,plan['real_histories'],plan['absolute_actions'],plan['batches'],window_ids=plan['window_ids'],**kwargs)

    def test_anchor3_and60_align_all_actions_targets_without_rendering(self):
        dataset,windows,plan=fixture()
        self.assertEqual(plan['window_metadata'][0]['history_state_indices'],[0,1,2,3])
        self.assertEqual(plan['window_metadata'][0]['h1_target_state_index'],4)
        self.assertEqual(plan['window_metadata'][0]['h4_target_state_index'],7)
        self.assertEqual(plan['window_metadata'][1]['action_indices'],list(range(57,64)))
        self.assertEqual(plan['window_metadata'][1]['h4_target_state_index'],64)
        np.testing.assert_array_equal(plan['absolute_actions'][0],dataset.records[0]['actions'][:7])
        np.testing.assert_array_equal(plan['absolute_actions'][1,-4:],dataset.records[1]['actions'][60:64])

    def test_future_truth_state_changes_cannot_change_prepared_feedback(self):
        dataset,windows,first=fixture()
        for i,(_,a) in enumerate(dataset.windows):dataset.records[i]['states'][a+1:]+=123456
        second=validation.build_validation_plan(dataset,windows,first['batches'],real_histories=first['real_histories'])
        np.testing.assert_array_equal(first['real_histories'],second['real_histories'])
        np.testing.assert_array_equal(first['absolute_actions'],second['absolute_actions'])
        self.assertFalse(np.array_equal(first['window_metadata'][0]['h4_target_state'],second['window_metadata'][0]['h4_target_state']))
        def predict(h,a,**kw):return h[:,-1].copy()
        np.testing.assert_array_equal(self.run_plan(first,predict)['prediction'],self.run_plan(second,predict)['prediction'])

    def test_every_history_action_seed_and_negative_prediction_is_exact(self):
        _,_,plan=fixture();seen=[];current_batch=-1;history=None
        def predict(h,a,*,batch_seed,allow_generated_history):
            nonlocal current_batch,history
            index=len(seen);b,k=divmod(index,4);k+=1;positions=plan['batches'][b]['positions']
            if b!=current_batch:current_batch=b;history=plan['real_histories'][positions].copy()
            np.testing.assert_array_equal(h,history)
            np.testing.assert_array_equal(a,plan['absolute_actions'][positions,k-1:k+3])
            self.assertEqual(batch_seed,(plan['batches'][b]['batch_seed']+k-1)%2**32)
            self.assertEqual(allow_generated_history,k>1)
            raw=np.full((4,3,96,96),k/10,dtype=np.float32);raw[:,0,0,0]=np.float32(-.019607782)
            history=np.concatenate((history[:,1:],raw[:,None]),axis=1);seen.append((b,k));return raw
        result=self.run_plan(plan,predict)
        self.assertTrue(result['evaluation_complete']);self.assertEqual(result['predictor_calls'],36)
        self.assertTrue(np.all(result['status']=='out_of_range_native_RGB'))
        self.assertEqual(result['prediction'][0,0,0,0,0],np.float32(-.019607782))

    def test_generated_input_guard_and_native_interface_forwarding_without_init(self):
        h=np.zeros((4,4,3,96,96),dtype=np.float32);h[0,3,0,0,0]=-.25;a=np.zeros((4,4,2))
        with self.assertRaises(ValueError):validate_native_batch_inputs(h,a)
        checked,_=validate_native_batch_inputs(h,a,allow_generated_history=True)
        np.testing.assert_array_equal(checked,h)
        h[0,3,0,0,0]=np.nan
        with self.assertRaises(ValueError):validate_native_batch_inputs(h,a,allow_generated_history=True)
        obj=object.__new__(NativePredictor);calls=[]
        obj._batch_predict=lambda histories,actions,**kwargs:calls.append(kwargs) or histories[:,-1]
        obj.batch_predict(np.zeros((4,4,3,96,96)),a,batch_seed=7,allow_generated_history=True)
        self.assertEqual(calls,[dict(batch_seed=7,allow_generated_history=True)])

    def test_cached_H1_does_not_repeat_first_step_and_continues_raw(self):
        _,_,plan=fixture();cache=np.full((36,3,96,96),-.125,dtype=np.float32);seeds=[]
        cached=dict(prediction=cache,status=np.asarray(['out_of_range_native_RGB']*36),window_ids=plan['window_ids'],batches=plan['batches'])
        def predict(h,a,*,batch_seed,allow_generated_history):
            self.assertTrue(allow_generated_history);self.assertTrue(np.all(h[:,-1]==-.125));seeds.append(batch_seed);return h[:,-1].copy()
        result=self.run_plan(plan,predict,cached_h1=cached)
        self.assertEqual(result['predictor_calls'],27);self.assertTrue(result['cached_h1_reused'])
        self.assertEqual(seeds[0],(plan['batches'][0]['batch_seed']+1)%2**32)
        np.testing.assert_array_equal(result['prediction'][:,0],cache)

    def test_finite_double_prediction_not_clipped_cast_or_mapped(self):
        _,_,plan=fixture();value=np.float64(-.123456789012345);count=0
        def predict(h,a,**kwargs):
            nonlocal count
            if count%4:self.assertEqual(h[0,-1,0,0,0],value)
            count+=1;return np.full((4,3,96,96),value,dtype=np.float64)
        result=self.run_plan(plan,predict)
        self.assertEqual(result['prediction'].dtype,np.float64);self.assertEqual(result['prediction'][0,3,0,0,0],value)

    def test_nonfinite_shape_and_oom_stop_whole_work_and_keep_denominator(self):
        _,_,plan=fixture()
        for mode in ('nonfinite','shape','oom'):
            count=0
            def predict(h,a,**kwargs):
                nonlocal count
                count+=1
                if mode=='oom':raise RuntimeError('mock CUDA out of memory')
                if mode=='shape':return np.zeros((3,3,96,96),dtype=np.float32)
                x=np.zeros((4,3,96,96),dtype=np.float32);x[0,0,0,0]=np.nan;return x
            result=self.run_plan(plan,predict)
            self.assertFalse(result['evaluation_complete']);self.assertEqual(count,1)
            self.assertEqual(result['predictor_calls'],1)
            self.assertEqual(result['prediction'].shape,(36,4,3,96,96));self.assertIsNotNone(result['fatal_error'])

    def test_budget_guard_failure_prevents_call_and_marks_incomplete(self):
        _,_,plan=fixture();called=[]
        def before(c):raise RuntimeError('mock budget exhausted')
        result=self.run_plan(plan,lambda *a,**k:called.append(1),before_call=before)
        self.assertFalse(result['evaluation_complete']);self.assertEqual(called,[]);self.assertEqual(result['predictor_calls'],0)

    def test_scoring_source_migration_and_original_negative_penalty(self):
        snapshot=validation.ROOT/'artifacts/native-rgb-supervised-20261029/stage4a/evaluation-before/native_rgb_evaluation.py'
        proof=validation.verify_scoring_migration(snapshot)
        self.assertTrue(proof['scoring_math_AST_identical']);self.assertTrue(proof['sampler_math_AST_identical'])
        scorer,identity=validation.make_original_scorer(snapshot_path=snapshot)
        image=np.ones((3,96,96),np.float32);image[:,30:40,30:40]=np.asarray([119,136,153])[:,None,None]/255
        image[:,70:74,70:74]=np.asarray([65,105,225])[:,None,None]/255
        raw=image.copy();raw[0,0,0]=-.25
        result=scorer(raw,image,image,letter='T',status='out_of_range_native_RGB')
        self.assertEqual(result['raw_object_error'],1);self.assertTrue(result['hard_generation_failure'])
        self.assertEqual(identity['protocol_sha256'],validation.OLD_PROTOCOL_SHA)

    def test_factory_authorization_guard_runs_before_predictor_factory(self):
        _,_,plan=fixture();called=[]
        def guard(c):raise RuntimeError('4A is CPU only, no model execution authorized')
        callback=validation.make_validation_callback(plan,predictor_factory=lambda c:called.append(1),truth_provider=lambda *a:None,
            row_scorer=lambda *a,**k:None,execution_guard=guard,raw_writer=lambda *a:None,formal_experiment_id='formal-test')
        with self.assertRaises(RuntimeError):callback(dict(progress=dict(updates=250)))
        self.assertEqual(called,[])
        guarded_native=validation.make_validation_callback(plan,truth_provider=lambda *a,**k:None,row_scorer=lambda *a,**k:None,
            execution_guard=guard,raw_writer=lambda *a:None,formal_experiment_id='formal-test')
        with self.assertRaises(RuntimeError):guarded_native(dict(progress=dict(updates=250)))
        self.assertNotIn('torch',sys.modules)

    def test_callback_returns72_records_and_scores_only_after_rollout(self):
        _,_,plan=fixture();calls=[];truth_calls=[];saved=[];score_calls=[]
        def predictor(h,a,**kwargs):calls.append(kwargs);return np.full((4,3,96,96),.7,dtype=np.float32)
        def truth(h,positions,ids,*,context,rollout_token):
            self.assertEqual(len(calls),36);self.assertTrue(rollout_token['authorized']);self.assertTrue(rollout_token['rollout_finished'])
            truth_calls.append(h);return dict(rgb=np.ones((36,3,96,96),np.float32),horizon=h,window_ids=ids)
        def scorer(pred,truth,last,**kwargs):
            i=(len(score_calls)//2)%36;np.testing.assert_array_equal(last,plan['real_histories'][i,-1])
            if len(score_calls)%2:np.testing.assert_array_equal(pred,plan['real_histories'][i,-1])
            score_calls.append(1);return dict(raw_object_error=0.25,hard_generation_failure=kwargs['status']!='ok')
        callback=validation.make_validation_callback(plan,predictor_factory=lambda c:predictor,truth_provider=truth,row_scorer=scorer,
            execution_guard=lambda c:None,raw_writer=lambda r,c:saved.append(r['prediction'].shape) or dict(path='stub-only'),
            formal_experiment_id='formal-test',assess_candidate=lambda candidate,catalog,**k:dict(eligible=candidate['evaluation_complete']))
        result=callback(dict(progress=dict(updates=250),validation_role='scheduled',check=lambda:None))
        self.assertTrue(result['evaluation_complete']);self.assertEqual(len(result['h1']),36);self.assertEqual(len(result['h4']),36)
        self.assertEqual(truth_calls,[1,4]);self.assertEqual(saved,[(36,4,3,96,96)])
        self.assertEqual(result['candidate']['candidate_role'],'scheduled');self.assertEqual(result['candidate']['update'],250)
        self.assertTrue(result['selection']['eligible']);self.assertNotIn('prediction',result)
        self.assertEqual(len(score_calls),144);self.assertEqual(result['h4'][0]['hold_metrics']['raw_object_error'],.25)

    def test_real_history_range_and_input_hash_mismatch_rejected(self):
        dataset,windows,plan=fixture();bad=plan['real_histories'].copy();bad[0,0,0,0,0]=-.1
        with self.assertRaises(ValueError):validation.build_validation_plan(dataset,windows,plan['batches'],real_histories=bad)
        windows[0]['npz_sha256']='changed'
        with self.assertRaises(ValueError):validation.build_validation_plan(dataset,windows,plan['batches'],real_histories=plan['real_histories'])


if __name__=='__main__':unittest.main()
