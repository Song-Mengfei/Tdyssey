"""Only synthetic arrays, ordinary fake environments/models, and UI method spies."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy unavailable for portable synthetic checks")
import numpy as np

from tdyssey.geometry import Pose
from tdyssey.state import BodyState, Snapshot
from tdyssey.manual_world_model import ManualSession,ModelPredictor
from scripts.manual_world_model import App,LEFT,target_from_click


class FakeEnvironment:
    def __init__(self,shape):
        self.shape,self.steps,self.closed,self.target=shape,0,0,None
    def capture_state(self):
        return Snapshot(self.shape,"a"*64,"b"*64,BodyState(Pose((200,200),0)),
            BodyState(Pose((10+self.steps,20+self.steps),0)),Pose((256,256),.785),self.steps,self.target)
    def observation(self):
        return dict(image=np.full((3,96,96),self.steps/100,np.float32),
                    agent_pos=np.array([10+self.steps,20+self.steps],np.float64))
    def step(self,action):
        self.steps+=1;self.target=tuple(float(v) for v in action)
        return self.observation(),.1,False,{"n_contacts":1}
    def close(self):self.closed+=1


class FakeModel:
    def __init__(self):
        self.asset_identity={"label":"Raw T+L","sha256":"c"*64}
        self.calls=[];self.closed=0
    def predict_next(self,images,actions,seed):
        self.calls.append((images.copy(),actions.copy(),seed))
        value=np.full((3,96,96),-.2,np.float32);value[1,0,0]=1.2
        return value
    def close(self):self.closed+=1


class PortableManualTests(unittest.TestCase):
    def setUp(self):
        self.envs=[]
        def factory(shape):
            env=FakeEnvironment(shape);self.envs.append(env)
            return env,env.observation()
        self.session=ManualSession(factory)
        self.addCleanup(self.session.close)

    def accepted(self):
        self.session.select((100,200))
        request=self.session.request();model=FakeModel();predictor=ModelPredictor(model)
        raw=predictor.predict(request)
        self.session.accept(request,raw)
        return request,raw,model,predictor

    def ui(self):
        app=object.__new__(App)
        app.session=self.session;app.predictor=ModelPredictor(FakeModel())
        app.busy=app.failed=app.closed=False
        app.requests,app.results=queue.Queue(),queue.Queue()
        app.buttons=[];app.log=io.StringIO()
        app.started=time.monotonic();app.seconds=1800
        app.pg=SimpleNamespace(QUIT=1,KEYDOWN=2,MOUSEBUTTONDOWN=3,K_ESCAPE=27,
            K_t=116,K_l=108,K_r=114,K_s=115,K_v=118,event=SimpleNamespace(get=lambda:[]))
        return app

    def test_three_recorded_holds_create_four_real_aligned_frames(self):
        self.assertEqual(self.session.controls,3)
        self.assertEqual(self.envs[0].steps,3)
        self.session.select((100,200));request=self.session.request()
        np.testing.assert_array_equal(request["images"][:,0,0,0],np.array([0,.01,.02,.03],np.float32))
        np.testing.assert_array_equal(request["past_actions"],[[10,20]]*3)
        self.assertEqual([state["control_step"] for state in request["history_states"]],[0,1,2,3])
        self.assertEqual(request["horizon"],1)
        self.assertEqual(self.session.controls,3)

    def test_predictor_preserves_original_rgb_absolute_actions_and_returned_pixels(self):
        request,raw,model,predictor=self.accepted()
        images,actions,seed=model.calls[0]
        np.testing.assert_array_equal(images,request["images"])
        np.testing.assert_array_equal(actions,[[10,20],[10,20],[10,20],[100,200]])
        self.assertEqual(seed,5478)
        self.assertLess(float(raw.min()),0);self.assertGreater(float(raw.max()),1)
        self.assertEqual(request["asset_identity"],model.asset_identity)
        result=self.session.execute()
        self.assertEqual(result["executed_controls"],1)
        self.assertEqual(result["step"]["before_state"]["control_step"],3)
        self.assertEqual(result["step"]["after_state"]["control_step"],4)
        np.testing.assert_array_equal(self.session.prediction,raw)
        with self.assertRaises(ValueError):self.session.execute()
        predictor.close();predictor.close();self.assertEqual(model.closed,1)

    def test_changed_or_obsolete_request_rejected_without_a_push(self):
        self.session.select((100,200));request=self.session.request()
        request["images"][0,0,0,0]=.5
        with self.assertRaises(ValueError):self.session.accept(request,np.zeros((1,3,96,96),np.float32))
        self.assertEqual(self.session.controls,3)
        self.session.select((100,200));request=self.session.request()
        self.session.reset("L")
        with self.assertRaises(ValueError):self.session.accept(request,np.zeros((1,3,96,96),np.float32))
        self.assertEqual(self.envs[0].closed,1)
        self.assertEqual(self.envs[-1].steps,3)
        self.assertEqual(self.session.shape,"L")

    def test_saved_case_has_full_states_exact_arrays_and_exclusive_new_files(self):
        request,raw,_,predictor=self.accepted();self.session.execute()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"case"
            self.session.save_case(path)
            meta=json.loads((path/"case.json").read_bytes())
            payload=(path/"arrays.npz").read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),meta["payload"]["sha256"])
            with np.load(path/"arrays.npz",allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["history_raw"],request["images"])
                np.testing.assert_array_equal(arrays["prediction"],raw)
                self.assertEqual(arrays["actual_raw"].shape,(1,3,96,96))
                for key in arrays:
                    self.assertEqual(hashlib.sha256(arrays[key].tobytes()).hexdigest(),meta["array_identities"][key]["sha256"])
            self.assertEqual(meta["request"]["initial_state"],meta["execution"]["before_state"])
            self.assertEqual(meta["execution"]["after_state"]["control_step"],4)
            self.assertFalse(meta["t1"]);self.assertFalse(meta["training_data"])
            with self.assertRaises(FileExistsError):self.session.save_case(path)
        predictor.close()

    def test_busy_double_click_and_reset_do_not_add_calls_or_controls(self):
        app=self.ui()
        click=SimpleNamespace(type=app.pg.MOUSEBUTTONDOWN,button=1,pos=(LEFT[0]+20,LEFT[1]+30))
        app.handle(click);app.handle(click)
        app.action("L");app.action("reset")
        self.assertTrue(app.busy)
        self.assertEqual(app.requests.qsize(),1)
        self.assertEqual(len(self.envs),1)
        self.assertEqual(self.session.controls,3)
        request=app.requests.get_nowait()
        app.results.put((request,app.predictor.predict(request),None))
        app.poll();app.poll()
        self.assertFalse(app.busy)
        self.assertEqual(self.session.controls,4)
        self.assertEqual(len(app.predictor.model.calls),1)
        self.assertEqual(self.session.case["execution"]["after_state"]["control_step"],4)
        app.predictor.close()

    def test_close_discards_late_prediction_and_worker_serially_closes_one_model(self):
        app=self.ui()
        self.session.select((100,200));request=self.session.request()
        model=app.predictor.model
        app.requests.put(request);app.requests.put(None)
        app._work()  # Ordinary synchronous fake worker; no thread/process/model runtime.
        self.assertEqual(len(model.calls),1)
        self.assertEqual(model.closed,1)
        self.assertEqual(app.results.qsize(),1)
        app.closed=True;app.poll()
        self.assertEqual(self.session.controls,3)
        self.assertIsNone(self.session.prediction)

    def test_s_v_shortcuts_and_click_coordinates_have_no_comparison_or_h4_actions(self):
        app=self.ui();app.save=Mock();app.record=Mock()
        app.handle(SimpleNamespace(type=app.pg.KEYDOWN,key=app.pg.K_s))
        app.handle(SimpleNamespace(type=app.pg.KEYDOWN,key=app.pg.K_v))
        app.save.assert_called_once_with();app.record.assert_called_once_with()
        app.busy=True
        app.handle(SimpleNamespace(type=app.pg.KEYDOWN,key=app.pg.K_s))
        self.assertEqual(app.save.call_count,1)
        self.assertIsNone(target_from_click((LEFT[0]+480,LEFT[1])))
        self.assertEqual(target_from_click((LEFT[0],LEFT[1])),(0.,0.))
        self.assertEqual(self.session.controls,3)
        app.predictor.close()


if __name__=="__main__":unittest.main()
