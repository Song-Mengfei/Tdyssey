"""Opt-in real D search/contact checks; no model or checkpoint is loaded."""

from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.geometry import Pose
from tdyssey.official import load_official_source
from tdyssey.simulator_planning import state_error
from tdyssey.state import BodyState, Snapshot


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set official source path for actual D action search")
class SimulatorSearchRuntimeChecks(unittest.TestCase):
    def run_cli(self, output, *extra):
        root = Path(__file__).resolve().parents[1]
        main = runpy.run_path(str(root / "scripts/evaluate_simulator_search.py"))["main"]
        arguments = ["evaluate_simulator_search.py", "--source-root", os.environ["TDYSSEY_OFFICIAL_ROOT"],
                     "--output", str(output), "--steps", "4", "--seconds", "30", "--horizon", "16",
                     "--population", "4", "--generations", "1", "--elite-count", "2",
                     "--max-evaluations", "4", *extra]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            main()

    def test_cli_video_draws_live_images_and_decodes(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "video-run"
            self.run_cli(output, "--video")
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["counts"]["control_steps"], 4)
            capture = cv2.VideoCapture(str(output / "D-search-wall-time.mp4"))
            decoded = 0
            while capture.read()[0]:
                decoded += 1
            capture.release()
            self.assertGreater(decoded, 0)
            self.assertEqual(decoded, report["video_frames"])

    def test_cli_bad_resume_saves_original_failure_before_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad-resume.json"
            source.write_text('{"episodes": []}', encoding="utf-8")
            output = Path(directory) / "failed-run"
            with self.assertRaisesRegex(ValueError, "Resume report"):
                self.run_cli(output, "--resume-report", str(source), "--resume-step", "1")
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["control_steps"], 0)
            self.assertIn("Resume report", report["error"])

    def test_cli_counts_executed_control_when_prediction_identity_fails(self):
        from tdyssey.simulator_search import SimulatorActionSearch

        choose = SimulatorActionSearch.choose

        def wrong_identity(search, *args, **kwargs):
            decision = choose(search, *args, **kwargs)
            decision.report["selected_predictions"][0]["control_step"] += 1
            return decision

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-prediction"
            with patch.object(SimulatorActionSearch, "choose", wrong_identity):
                with self.assertRaises(ValueError):
                    self.run_cli(output)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["control_steps"], 1)
            self.assertEqual(report["counts"]["control_physics_substeps"], 10)
            self.assertEqual(len(report["steps"]), 1)
            self.assertIn("state", report["steps"][0])

    def test_search_reconstructs_contact_and_selected_controls_match_live_without_policy(self):
        import numpy as np
        from tdyssey.simulator_search import SearchConfig, SimulatorActionSearch

        for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                           "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
            os.environ.setdefault(key, value)
        original = make_official_shape_env_class(load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"])))

        class PhysicsOnly(original):
            def _get_obs(self):
                return {}

            def step(self, action):
                return super().step(np.asarray(action, dtype=np.float64))

        def environment():
            env = PhysicsOnly(shape="D", legacy=True)
            env.seed(100000)
            env.reset()
            return env

        live = environment()
        self.addCleanup(live.close)
        rect = min(official_shape("D").components, key=lambda r: (r.x0, r.y0))
        start = (256 + rect.x0 - 20, 256 + rect.center[1])
        initial = replace(live.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
                          pusher_body=BodyState(Pose(start, 0)), goal=Pose((400, 120), .5),
                          control_step=0, pusher_target=None)
        live.restore_state(initial)
        search = SimulatorActionSearch(environment, initial,
                                       SearchConfig(horizon=32, population=4, generations=1,
                                                    elite_count=2, max_evaluations=8,
                                                    execute_steps=4, settle_steps=4, seed=7))
        self.addCleanup(search.close)
        contacts = 0
        for _ in range(8):
            action = (start[0] + 65, start[1])
            _, _, done, info = live.step(action)
            self.assertFalse(done)
            contacts += info["n_contacts"]
            search.record_action(action)
        self.assertGreater(contacts, 0)
        before = live.capture_state()
        decision = search.choose(before, remaining_steps=40)
        self.assertEqual(state_error(live.capture_state(), before), 0)
        self.assertGreater(search.counts["replay_steps"], 0)
        self.assertGreater(search.counts["candidate_steps"], 0)
        self.assertLessEqual(len(decision.actions), 40)
        for index, action in enumerate(decision.actions[:4]):
            _, _, done, _ = live.step(action)
            expected = Snapshot.from_dict(decision.report["selected_predictions"][index])
            self.assertLessEqual(state_error(live.capture_state(), expected), 1e-7)
            search.record_action(action)
            if done:
                break


if __name__ == "__main__":
    unittest.main()
