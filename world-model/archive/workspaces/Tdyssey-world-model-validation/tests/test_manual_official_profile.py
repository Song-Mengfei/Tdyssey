"""Raw-domain session/UI integration using ordinary states and fake predictions."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pygame
from scripts.manual_world_model import App,PANELS,SIDE,WIDTH,HEIGHT
from tdyssey import manual_world_model as manual
from tdyssey.manual_official_predictor import OfficialPredictor
from tests.test_manual_world_model_ui import fake_environment

BASE=Path(__file__).resolve().parents[1]/'artifacts/official-raw-alignment-20261026/fixtures'


class OfficialManualTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True,exist_ok=True)
        self.folder=tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.folder.cleanup)
        app=self.app=App.__new__(App)
        app.output=Path(self.folder.name);app.log=io.StringIO()
        app.session=manual.ManualSession(fake_environment,image_domain=manual.OFFICIAL_DOMAIN)
        self.addCleanup(app.session.close)
        app.requests,app.results=queue.Queue(),queue.Queue()
        app.predictor=SimpleNamespace(loads=0,assets={'weights':{'frozen':{},'adapted':{}}},
            usage=lambda:{},last_partial_predictions=[])
        app.busy=app.failed=app.closed=False;app.buttons=[];app.result=None
        app.started,app.seconds=time.monotonic(),1800;app.last_memory=float('inf')
        app.monitor_host=False
        pygame.font.init();app.window=pygame.Surface((WIDTH,HEIGHT))
        app.font=pygame.font.Font(None,19);app.small=pygame.font.Font(None,16)
        app.status,app.video='Click left to push',None

    def return_fake_result(self):
        request=self.app.requests.get_nowait()
        self.app.results.put((request,(request['images'][-1][None].copy(),{},0.),None))
        with patch('pygame.event.get',return_value=[]):self.app.poll()
        return request

    def test_raw_click_compare_save_and_draw_never_call_t1(self):
        app=self.app
        with patch.object(manual,'transform_t1',side_effect=AssertionError('T1 trap')):
            app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,
                pos=(PANELS[0].x+SIDE//2,PANELS[0].y+SIDE//2)))
            original=self.return_fake_result()
            self.assertEqual(app.session.controls,4)
            app.action('compare')
            comparison=self.return_fake_result()
            self.assertEqual(app.session.controls,4)
            self.assertEqual(original['request_key'],comparison['request_key'])
            np.testing.assert_array_equal(original['images'],comparison['images'])
            with patch('pygame.display.flip'),patch.object(app,'panel') as panel:app.draw()
            np.testing.assert_array_equal(panel.call_args_list[0].args[1],app.session.observation['image'])
            app.save()
            folder=next(app.output.glob('view-*-case'))
            bundle=manual.load_case_bundle(folder)
            self.assertEqual(bundle['metadata']['kind'],'manual_world_model_case.v2')
            self.assertIsNone(bundle['metadata']['t1_sha256'])
            self.assertNotIn('history_t1',bundle['arrays'])
            np.testing.assert_array_equal(bundle['arrays']['history_model'],original['images'])
            np.testing.assert_array_equal(bundle['arrays']['actual_model'],bundle['arrays']['actual_raw'])
            self.assertEqual(bundle['metadata']['execution']['result']['executed_controls'],1)
            self.assertEqual(len(bundle['metadata']['predictions']),2)

    def test_h1_and_domain_boundaries_are_enforced_before_model_import(self):
        app=self.app
        app.action('horizon')
        self.assertEqual(app.session.horizon,1)
        with self.assertRaisesRegex(ValueError,'H1'):app.session.configure(horizon=4)
        app.session.select([250,330]);request=app.session.request()
        with self.assertRaisesRegex(ValueError,'Legacy predictor'):manual.PinnedPredictor().predict(request)
        request['image_domain']=manual.LEGACY_DOMAIN
        with self.assertRaisesRegex(ValueError,'separate original'):OfficialPredictor.__new__(OfficialPredictor).predict(request)

    def test_manual_predictor_calls_the_shared_model_with_absolute_four_actions(self):
        self.app.session.select([210.5,330.25]);request=self.app.session.request()
        predictor=OfficialPredictor.__new__(OfficialPredictor)
        manual.PinnedPredictor.__init__(predictor)
        predictor.assets={'weights':{'frozen':{'kind':'initial'}}}
        predictor.load=Mock();predictor.check_cuda=Mock()
        frame=np.full((3,96,96),.75,np.float32)
        predictor.model=SimpleNamespace(predict_next=Mock(return_value=frame))
        with patch.object(manual,'transform_t1',side_effect=AssertionError('T1 trap')), \
                patch('tdyssey.wm_image_measurement_v2.decode',side_effect=AssertionError('legacy decoder trap')):
            predicted,_,_=predictor.predict(request)
        given=predictor.model.predict_next.call_args
        np.testing.assert_array_equal(given.args[0],request['images'])
        np.testing.assert_array_equal(given.args[1],list(request['past_actions'])+[request['target']])
        np.testing.assert_array_equal(predicted[0],frame)
        self.assertEqual(predictor.forecast_reservations,1)
        self.assertEqual(predictor.euler_reservations,3)
        self.assertFalse(request['prediction_trace'][0]['t1_applied'])

    def test_official_entry_reports_missing_assets_without_ui_or_initial_fallback(self):
        from scripts import visualize_world_model as entry
        original_import=__import__
        def forbid(name,*args,**kwargs):
            if name=='torch' or name.startswith('torch.'):
                raise AssertionError('Asset/help path must not import a tensor runtime')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=forbid),redirect_stdout(io.StringIO()) as output:
            code=entry.main(['--check-assets'])
        value=json.loads(output.getvalue())
        self.assertEqual(code,2)
        self.assertFalse(value['ready'])
        self.assertEqual(set(value['assets']),{'adapted'})
        self.assertEqual(value['assets']['adapted']['status'],'missing')
        self.assertEqual(value['model_loads'],0)
        with patch('builtins.__import__',side_effect=forbid),redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:entry.main(['--help'])
        self.assertEqual(stopped.exception.code,0)


if __name__=='__main__':unittest.main()
