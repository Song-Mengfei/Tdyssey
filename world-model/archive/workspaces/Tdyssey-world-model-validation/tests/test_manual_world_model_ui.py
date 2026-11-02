"""Click-to-step UI checks without loading models or running real physics."""
import io
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pygame

from scripts.manual_world_model import App, PANELS, SIDE, WIDTH, HEIGHT
from tdyssey.manual_world_model import ManualSession
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.state import Snapshot, BodyState


class UIEnvironment:
    """Serializable deterministic fixture; no simulator backend."""
    def __init__(self, shape):
        self.shape, self.steps, self.target = shape, 0, None

    def observation(self):
        return dict(image=np.full((3, 96, 96), self.steps / 255, dtype=np.float32),
                    agent_pos=np.array([256, 405]))

    def step(self, action):
        self.steps += 1
        self.target = tuple(float(x) for x in action)
        return self.observation(), 0., False, dict(n_contacts=0)

    def capture_state(self):
        return Snapshot(self.shape, official_shape(self.shape).geometry_hash, "0" * 64,
                        BodyState(Pose((256, 240), 0)), BodyState(Pose((256, 405), 0)),
                        Pose((256, 256), 0), self.steps, self.target)

    def close(self):
        pass


def fake_environment(shape):
    env = UIEnvironment(shape)
    return env, env.observation()


class ClickToStepTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = app = App.__new__(App)
        app.output = Path(self.folder.name)
        app.log = io.StringIO()
        app.session = ManualSession(fake_environment)
        app.requests, app.results = queue.Queue(), queue.Queue()
        app.predictor = SimpleNamespace(loads=0, assets={"weights": {"frozen": {}, "adapted": {}}},
                                        usage=lambda: {}, last_partial_predictions=[])
        app.busy = app.failed = app.closed = False
        app.buttons = []
        app.result = None
        app.started, app.seconds = time.monotonic(), 1800
        app.last_memory = float("inf")

    def tearDown(self):
        self.app.session.close()
        self.folder.cleanup()

    def click(self):
        self.app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
            pos=(PANELS[0].x + SIDE // 2, PANELS[0].y + SIDE // 2)))

    def result(self):
        request = self.app.requests.get_nowait()
        raw = np.stack([request["images"][-1]] * request["horizon"])
        self.app.results.put((request, (raw, {}, 0.), None))

    def poll(self):
        with patch("pygame.event.get", return_value=[]):
            self.app.poll()

    def test_click_executes_once_after_prediction_without_space(self):
        self.app.session.configure(horizon=4)
        before = self.app.session.controls
        self.click()
        self.assertTrue(self.app.busy)
        self.assertEqual(self.app.session.controls, before)
        self.click()  # A second click while busy must not enqueue a second action.
        self.assertEqual(self.app.requests.qsize(), 1)
        self.result()
        self.poll()
        self.assertEqual(self.app.session.controls, before + 4)
        self.assertIsNotNone(self.app.session.actual)
        self.assertEqual(self.app.result["matched_horizon"], 4)
        self.poll()
        self.app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        self.assertEqual(self.app.session.controls, before + 4)

    def test_failed_prediction_never_executes(self):
        self.click()
        request = self.app.requests.get_nowait()
        self.app.results.put((request, None, "synthetic failure"))
        self.poll()
        self.assertTrue(self.app.failed)
        self.assertEqual(self.app.session.controls, 3)

    def test_rejected_return_saves_exact_input_output_and_stops(self):
        self.click()
        request = self.app.requests.queue[0]
        self.result()
        self.app.session.configure(role="adapted")
        self.poll()
        self.assertTrue(self.app.failed)
        self.assertEqual(self.app.session.controls, 3)
        failed = list(self.app.output.glob("failed-*"))
        self.assertEqual(len(failed), 1)
        with np.load(failed[0] / "arrays.npz", allow_pickle=False) as evidence:
            np.testing.assert_array_equal(evidence["history_raw"], request["images"])
            np.testing.assert_array_equal(evidence["partial_prediction"][0], request["images"][-1])

    def test_exit_or_deadline_cancels_unconsumed_result(self):
        self.click()
        self.result()
        self.app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.poll()
        self.assertEqual(self.app.session.controls, 3)
        self.app.closed = False
        self.app.started -= 1801
        self.poll()
        self.assertTrue(self.app.closed)
        self.assertEqual(self.app.session.controls, 3)

    def test_compare_reuses_saved_input_and_never_executes_again(self):
        self.app.session.configure(horizon=4)
        self.click()
        original = self.app.requests.queue[0]
        self.result()
        self.poll()
        before = self.app.session.controls
        self.app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        comparison = self.app.requests.queue[0]
        self.assertEqual(comparison["request_key"], original["request_key"])
        self.assertEqual(comparison["role"], "adapted")
        np.testing.assert_array_equal(comparison["images"], original["images"])
        self.result()
        self.poll()
        self.assertEqual(self.app.session.controls, before)
        self.assertIsNone(self.app.session.pending)
        self.assertEqual(set(self.app.session.comparison_predictions()), {"frozen", "adapted"})

    def test_two_panel_layout_and_draw(self):
        self.assertEqual(len(PANELS), 2)
        self.assertTrue(all(pygame.Rect(0, 0, WIDTH, HEIGHT).contains(r) for r in PANELS))
        self.assertFalse(PANELS[0].colliderect(PANELS[1]))
        pygame.font.init()
        app = self.app
        app.window = pygame.Surface((WIDTH, HEIGHT))
        app.font = pygame.font.SysFont("microsoftyahei", 19)
        app.small = pygame.font.SysFont("microsoftyahei", 16)
        app.title_font = pygame.font.SysFont("microsoftyahei", 25)
        app.status, app.video = "Click left to push", None
        with patch("pygame.display.flip"):
            app.draw()
        app.session.configure(horizon=4)
        app.session.prediction = np.stack([np.full((3, 96, 96), n/10, np.float32) for n in range(1,5)])
        with patch("pygame.display.flip"), patch.object(app, "panel") as panel:
            app.draw()
        np.testing.assert_array_equal(panel.call_args_list[1].args[1], app.session.prediction[3])
        # A real episode that stops at H2 must display the corresponding H2 image.
        app.session.actual = np.stack([np.zeros((3,96,96),np.float32)]*2)
        with patch("pygame.display.flip"), patch.object(app, "panel") as panel:
            app.draw()
        np.testing.assert_array_equal(panel.call_args_list[1].args[1], app.session.prediction[1])


if __name__ == "__main__":
    unittest.main()
