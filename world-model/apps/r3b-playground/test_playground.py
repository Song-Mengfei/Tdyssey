"""Focused CPU interface/causality checks; actual GPU/UI checks are separate."""
import unittest
import numpy as np
from server import Playground,MODEL_TRAINING_PHYSICS_SHA
from tdyssey.native_rgb_original_physics import PROTOCOL_SHA256 as PHYSICS_SHA,PROTOCOL

class Spy:
    def __init__(self):self.inputs=[]
    def __call__(self,images,actions,seed):
        self.inputs.append((images.copy(),actions.copy(),seed))
        return np.full((3,96,96),len(self.inputs)/10,dtype=np.float32)

class Tests(unittest.TestCase):
    def setUp(self):self.spy=Spy();self.app=Playground(self.spy)
    def tearDown(self):self.app.env.close()
    def test_strict_original_physics_and_real_history(self):
        i=self.app.env.initial_identity();self.assertEqual(i['protocol_sha256'],PHYSICS_SHA)
        self.assertEqual(i['physics']['agent']['body_type'],1)
        self.assertEqual(i['physics']['space']['damping'],0)
        self.assertEqual(i['physics']['space']['iterations'],PROTOCOL['solver_iterations'])
        self.assertEqual(i['physics']['substep_dt'],.01)
        self.assertEqual(i['physics']['substeps_per_control'],10)
        self.assertEqual(self.app.env.capture_state().physics_hash,self.app.env._physics_hash())
        self.assertEqual(PHYSICS_SHA,MODEL_TRAINING_PHYSICS_SHA)
        self.assertEqual(self.app.env.control_step,3);self.assertEqual(len(self.app.history),4)
    def test_open_loop_never_uses_future_truth(self):
        a=np.array(self.app.target)+[12,0];initial=np.stack(self.app.history);past=np.stack(self.app.past)
        out=self.app.step(self.app.session,self.app.revision,a,4)
        self.assertEqual(out['steps'],4);self.assertEqual(len(self.spy.inputs),4)
        np.testing.assert_array_equal(self.spy.inputs[0][0],initial)
        np.testing.assert_array_equal(self.spy.inputs[0][1][:3],past)
        for k in range(4):
            np.testing.assert_array_equal(self.spy.inputs[k][1][-1],a)
            if k:np.testing.assert_array_equal(self.spy.inputs[k][0][-1],np.full((3,96,96),k/10,np.float32))
        self.assertFalse(np.array_equal(np.stack(self.app.history)[-1],self.spy.inputs[-1][0][-1]))
        # Next ordinary H1 is reconditioned on actual physical images, not stale rollout.
        actual=np.stack(self.app.history).copy();self.app.step(self.app.session,self.app.revision,a,1)
        np.testing.assert_array_equal(self.spy.inputs[4][0],actual)
    def test_raw_capture_and_complete_session_export(self):
        import base64
        from capture import unpack
        target=np.array(self.app.target)+[20,0]
        self.app.step(self.app.session,self.app.revision,target,4)
        row=self.app.capture.records[-1];arrays=row['arrays'];meta=row['metadata']
        self.assertEqual(arrays['input_histories_float32'].shape,(4,4,3,96,96))
        self.assertEqual(arrays['raw_predictions_float32'].dtype,np.float32)
        self.assertEqual(meta['seeds'],[5478,5479,5480,5481])
        np.testing.assert_array_equal(arrays['input_histories_float32'][1,-1],arrays['raw_predictions_float32'][0])
        self.assertEqual(len(meta['physical_states']),5)
        self.assertIn('velocity',meta['physical_states'][-1]['snapshot']['pusher_body'])
        self.assertIn('angular_velocity',meta['physical_states'][-1]['snapshot']['object_body'])
        exported=self.app.export_session();saved=self.app.env.state().copy();history=np.stack(self.app.history).copy();saved_controller=self.app.session_record()['physical_state']['controller']
        decoded,_=unpack(base64.b64decode(exported['archive_base64']))
        self.assertEqual(decoded['environment_physics_sha256'],PHYSICS_SHA)
        self.app.step(self.app.session,self.app.revision,target+[4,0],1)
        self.app.restore_session(exported)
        np.testing.assert_array_equal(self.app.env.state(),saved)
        self.assertEqual(self.app.session_record()['physical_state']['controller'],saved_controller)
        np.testing.assert_array_equal(np.stack(self.app.history),history)
        self.assertFalse(exported['solver_cache_serialized'])
    def test_recording_linkage_ring_bounds_and_pause_semantics(self):
        from unittest.mock import patch
        from capture import ClipRing
        with patch('server.RecordingWriter')as factory,patch('server.persist_clip',side_effect=lambda repo,data,metadata:dict(saved=True,recording_id=metadata['recording_id'],batches=len(metadata['records']),download_url='/captures/test.npz')):
            writer=factory.return_value;writer.can_accept.return_value=True;writer.finish.return_value=dict(raw_complete=True)
            before=self.app.env.control_step;recording=self.app.start_recording();identity=recording['recording_id']
            self.assertEqual(before,self.app.env.control_step)
            writer.can_accept.return_value=False
            self.app.step(self.app.session,self.app.revision,self.app.target,4)
            self.assertEqual(before,self.app.env.control_step);self.assertFalse(self.spy.inputs)
            writer.can_accept.return_value=True
            self.app.step(self.app.session,self.app.revision,np.array(self.app.target)+[8,0],1)
            self.assertEqual(self.app.capture.records[-1]['metadata']['recording_id'],identity)
            self.app.reset('C','center');self.assertEqual(self.app.recording_id,identity)
            saved=self.app.stop_recording(identity)
        self.assertEqual(saved['clip']['recording_id'],identity);self.assertIsNone(self.app.recording_id)
        ring=ClipRing(max_bytes=3000,max_batches=2)
        for index in range(4):ring.append(dict(index=index),dict(raw=np.ones(200,dtype=np.float32)))
        self.assertEqual(len(ring.records),2);self.assertEqual(ring.dropped,2);self.assertLessEqual(ring.bytes,3000)
    def test_preview_does_not_change_batch_target_or_physics(self):
        before=self.app.env.state().copy();step=self.app.env.control_step
        self.app.set_preview(self.app.session,[100,100],123)
        self.assertEqual(self.app.preview_snapshot()['target'],[100,100])
        np.testing.assert_array_equal(self.app.env.state(),before);self.assertEqual(self.app.env.control_step,step)
    def test_out_of_range_raw_feedback_and_empty_clip_history(self):
        calls=[]
        def raw(images,actions,seed):
            if calls:np.testing.assert_array_equal(images[-1],calls[-1])
            result=np.full((3,96,96),1.25+len(calls)*.1,dtype=np.float32);calls.append(result);return result
        data,metadata=self.app.capture.pack(self.app.session_record(),session_arrays=dict(history_float32=np.stack(self.app.history)))
        from capture import unpack
        meta,arrays=unpack(data);self.assertEqual(arrays[meta['session_arrays']['history_float32']].shape,(4,3,96,96))
        self.app.predict=raw;self.app.step(self.app.session,self.app.revision,self.app.target,4)
        saved=self.app.capture.records[-1]['arrays']['raw_predictions_float32']
        self.assertGreater(float(saved.min()),1);self.assertTrue(self.app.metrics['raw_out_of_range'])
    def test_recording_raw_stream_outlives_the_ring_and_empty_is_not_truncated(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from capture import ClipRing,RecordingWriter,session_export
        ring=ClipRing(max_bytes=3000,max_batches=2)
        _,metadata=ring.pack(dict(session='fixture'),recording_id='fixture',first_sequence=1)
        self.assertTrue(metadata['empty_capture']);self.assertFalse(metadata['history_truncated'])
        with tempfile.TemporaryDirectory(dir=__import__('server').REPO/'artifacts')as temp,patch('capture.storage_check',return_value=dict(shared_bytes=0)):
            initial=session_export(dict(session='fixture'),dict(history=np.zeros((4,3,96,96),np.float32)))
            writer=RecordingWriter(Path(temp),'recording-test',initial)
            for index in range(6):
                ring.append(dict(recording_id='recording-test',session='fixture',letter='O',requested_horizon=1,completed_controls=1,ui_step_before=index),dict(raw=np.ones(200,np.float32)))
                writer.append(ring.records[-1])
            manifest=writer.finish()
            self.assertEqual(len(ring.records),2);self.assertEqual(manifest['raw_batches'],6);self.assertTrue(manifest['raw_complete'])
    def test_stale_and_invalid_actions_do_not_advance_physics(self):
        before=self.app.env.state().copy();step=self.app.env.control_step
        for revision,target in [(self.app.revision-1,[200,200]),(self.app.revision,[512,100]),(self.app.revision,[float('nan'),0])]:
            with self.assertRaises(ValueError):self.app.step(self.app.session,revision,target,1)
        np.testing.assert_array_equal(before,self.app.env.state());self.assertEqual(step,self.app.env.control_step);self.assertFalse(self.spy.inputs)
    def test_sealed_letters_rejected_and_scenes_reset_cleanly(self):
        with self.assertRaises(ValueError):self.app.reset('D','center')
        for letter in ('T','L','O','C','X','B'):
            for scene in ('center','left','right','rotated'):
                s=self.app.reset(letter,scene);self.assertEqual(s['steps'],0);self.assertFalse(s['terminal']);self.assertEqual(len(self.app.history),4)
                self.assertLessEqual(float(self.app.env.penetrations().max()),2)

    def test_wrong_original_controller_rejected_before_reset(self):
        import base64
        from capture import unpack,session_export
        exported=self.app.export_session();metadata,arrays=unpack(base64.b64decode(exported['archive_base64']))
        metadata['physical_state']['controller']['physics_hz']=2000
        bad=session_export(metadata,arrays);before=self.app.env.state().copy();session=self.app.session
        with self.assertRaises(ValueError):self.app.restore_session(bad)
        self.assertEqual(self.app.session,session);np.testing.assert_array_equal(self.app.env.state(),before)
    def test_real_ten_substeps_and_contact_seconds(self):
        self.app.step(self.app.session,self.app.revision,np.asarray(self.app.target)+[8,0],1)
        info=self.app.info
        self.assertEqual(info['physics_substeps'],10)
        self.assertEqual([r['dt']for r in info['substeps']],[.01]*10)
        self.assertEqual(self.app.metrics['contact_seconds'],info['agent_object_contact_substeps']/100)
        self.assertNotIn('interventions',info)

class ConcurrentRequestTests(unittest.TestCase):
    """Exercise the actual HTTP handler methods concurrently, without a socket/GPU."""
    def setUp(self):
        import ast,threading
        from pathlib import Path
        import server
        self.spy=Spy();self.app=Playground(self.spy);self.addCleanup(self.app.env.close)
        self.lock=threading.Lock()
        tree=ast.parse(Path(server.__file__).read_text(encoding='utf-8'))
        serve=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='serve')
        handler=next(node for node in serve.body if isinstance(node,ast.ClassDef)and node.name=='Handler')
        namespace=dict(vars(server));namespace.update(app=self.app,predictor=self.spy,lock=self.lock)
        exec(compile(ast.Module(body=[handler],type_ignores=[]),server.__file__,'exec'),namespace)
        self.Handler=namespace['Handler']
    def request(self,path,body=None,respond=None):
        import io,json
        from types import SimpleNamespace
        request=object.__new__(self.Handler);request.path=path;request.server=SimpleNamespace(server_port=8767)
        data=json.dumps({}if body is None else body).encode()
        request.headers={'Content-Type':'application/json','Content-Length':str(len(data))};request.rfile=io.BytesIO(data);request.results=[]
        request.respond=respond or(lambda value,status=200:request.results.append((status,value)))
        return request
    def step_request(self):
        return self.request('/api/step',dict(session=self.app.session,revision=self.app.revision,target=(np.asarray(self.app.target)+[2,0]).tolist(),horizon=1))
    def test_event_and_step_have_independent_locks(self):
        import threading
        before=self.app.env.control_step
        # Logging succeeds even while the physical scene lock is occupied.
        event=self.request('/api/event',dict(kind='pointer_down',detail={}))
        with self.lock:event.do_POST()
        self.assertEqual(event.results[0][0],200);self.assertEqual(before,self.app.env.control_step)
        entered=threading.Event();original=self.app.event
        def blocked_event(kind,**value):entered.set();return original(kind,**value)
        self.app.event=blocked_event
        event=self.request('/api/event',dict(kind='batch_requested',detail={}))
        step=self.step_request();event_thread=threading.Thread(target=event.do_POST);step_thread=threading.Thread(target=step.do_POST)
        self.app.event_lock.acquire()
        try:
            event_thread.start();self.assertTrue(entered.wait(2));step_thread.start();step_thread.join(3)
            self.assertFalse(step_thread.is_alive(),'A scene step waited on an unrelated UI event lock')
        finally:
            self.app.event_lock.release();event_thread.join(3);step_thread.join(3)
        self.assertEqual(event.results[0][0],200);self.assertEqual(step.results[0][0],200)
        self.assertEqual(self.app.env.control_step,before+1);self.assertEqual(len(self.spy.inputs),1)
    def test_get_serialization_does_not_hold_the_scene_lock(self):
        import threading
        for path in('/api/state','/api/session/export'):
            entered=threading.Event();release=threading.Event()
            def slow_response(value,status=200):entered.set();release.wait(5)
            read=self.request(path,respond=slow_response);read_thread=threading.Thread(target=read.do_GET)
            step=self.step_request();step_thread=threading.Thread(target=step.do_POST)
            try:
                read_thread.start();self.assertTrue(entered.wait(2));step_thread.start();step_thread.join(3)
                self.assertFalse(step_thread.is_alive(),'A scene step waited for JSON/socket serialization')
                self.assertEqual(step.results[0][0],200)
            finally:release.set();read_thread.join(3);step_thread.join(3)

if __name__=='__main__':unittest.main()
