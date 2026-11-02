"""Opt-in real contact and CLI replay checks; no performance claims."""
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
from tdyssey.simulator_sequence import ContactSequenceSearch, ContactStage, SequenceConfig
from tdyssey.contact_geometry import rectangle_exterior_edges
from tdyssey.state import BodyState


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set official source for real D checks")
class SequenceRuntimeChecks(unittest.TestCase):
    def test_generated_precision_press_matches_real_contact_and_live_state(self):
        import numpy as np
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        original = make_official_shape_env_class(load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"])))
        class PhysicsOnly(original):
            def _get_obs(self):
                return {}
            def step(self, action):
                return super().step(np.asarray(action, dtype=np.float64))
        def env():
            instance=PhysicsOnly(shape="D",legacy=True)
            instance.seed(100000)
            instance.reset()
            return instance
        live=env()
        self.addCleanup(live.close)
        edges=rectangle_exterior_edges(official_shape("D").components)
        edge=max(range(len(edges)),key=lambda i:edges[i].point(.5)[0])
        q,normal=edges[edge].point(.5),edges[edge].normal
        start=tuple(256+q[k]+15.5*normal[k] for k in (0,1))
        initial=replace(live.capture_state(),object_body=BodyState(Pose((256,256),0)),
            pusher_body=BodyState(Pose(start,0)),goal=Pose((250,256),.05),control_step=0,pusher_target=None)
        live.restore_state(initial)
        search=ContactSequenceSearch(env,initial,SequenceConfig(horizon=12,roots=1,beam_width=1,
            children=1,refinements=0,max_evaluations=3))
        self.addCleanup(search.close)
        search._proposals=lambda state,n:[ContactStage(edge,.5,.5,3.,speed=20.,
            approach_speed=220.,sustain=3,precision=True)]*n
        # Fixture selection verifies generated slow targets and dwell physics;
        # it is not a claim about optimizing a task or completing the goal.
        with patch("tdyssey.simulator_sequence.path_key",side_effect=lambda r:(r["unsafe"],not bool(r["stages"]),r["pose_rms"])):
            decision=search.choose(initial,12)
        self.assertEqual(state_error(initial,live.capture_state()),0)
        detail=decision.report["selected"]["stage_details"][0]
        self.assertGreaterEqual(detail["push_controls"],4)
        contacts=0
        for index,action in enumerate(decision.actions):
            _,_,done,info=live.step(action)
            contacts+=info["n_contacts"]
            expected=type(initial).from_dict(decision.report["selected_predictions"][index])
            self.assertEqual(state_error(live.capture_state(),expected),0)
            search.record_action(action)
            self.assertFalse(done)
        self.assertGreater(contacts,0)

    def test_complete_two_contact_fixture_matches_live_with_contact_history(self):
        self.contact_fixture(2)

    def test_three_contact_fixture_replays_both_prefixes_with_real_contact_history(self):
        self.contact_fixture(3)

    def contact_fixture(self, stage_count):
        import numpy as np
        for key, value in (("SDL_VIDEODRIVER", "dummy"), ("SDL_AUDIODRIVER", "dummy"),
                           ("PYGAME_HIDE_SUPPORT_PROMPT", "1")):
            os.environ.setdefault(key, value)
        original = make_official_shape_env_class(load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"])))
        class PhysicsOnly(original):
            def _get_obs(self):
                return {}
            def step(self, action):
                return super().step(np.asarray(action, dtype=np.float64))
        def env():
            instance = PhysicsOnly(shape="D", legacy=True)
            instance.seed(100000)
            instance.reset()
            return instance
        live = env()
        self.addCleanup(live.close)
        rect = min(official_shape("D").components, key=lambda r: (r.x0, r.y0))
        start = (256+rect.x0-20, 256+rect.center[1])
        initial = replace(live.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
            pusher_body=BodyState(Pose(start, 0)), goal=Pose((400, 120), .5), control_step=0, pusher_target=None)
        live.restore_state(initial)
        search = ContactSequenceSearch(env, initial,
            SequenceConfig(horizon=16, roots=1 if stage_count==3 else 2,
                beam_width=1 if stage_count==3 else 2, children=1 if stage_count==3 else 2,
                refinements=0, max_evaluations=8))
        self.addCleanup(search.close)
        contacts = 0
        for _ in range(8):
            action = (start[0]+65, start[1])
            _, _, done, info = live.step(action)
            self.assertFalse(done)
            contacts += info["n_contacts"]
            search.record_action(action)
        self.assertGreater(contacts, 0)
        before = live.capture_state()
        search._proposals = lambda state, n: [ContactStage(0)]*n
        def builder(shape, state, stage, budget, maximum):
            p = state.pusher_target or state.pusher_body.pose.position
            actions = tuple((p[0]+2*(i+1), p[1]) for i in range(min(4, budget)))
            return actions, ["push"]*len(actions), {"approach_controls": 0, "push_controls": len(actions)}
        # Force a generated two-contact leaf to test the entire replay protocol,
        # not to claim the optimizer solves this deliberately staged fixture.
        def prefer_pair(row):
            return (row["unsafe"], len(row["stage_objects"]) != stage_count, row["pose_rms"])
        # Wide planning tolerances isolate triple history reconstruction. They
        # do not alter collision dynamics, the actual goal or success threshold.
        limits = patch("tdyssey.simulator_sequence.joint_limits",return_value=dict(
            center_limit=512.,angle_limit=6.,path_center_limit=512.,path_angle_limit=6.)) if stage_count==3 else contextlib.nullcontext()
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder), \
                patch("tdyssey.simulator_sequence.path_key", side_effect=prefer_pair), limits:
            decision = search.choose(before, 30 if stage_count==3 else 100)
        self.assertEqual(decision.report["selected"]["kind"], "third_contact" if stage_count==3 else "second_contact")
        self.assertEqual(len(decision.actions), 16)
        self.assertEqual(state_error(live.capture_state(), before), 0)
        for index, action in enumerate(decision.actions):
            _, reward, done, info = live.step(action)
            expected = type(initial).from_dict(decision.report["selected_predictions"][index])
            self.assertEqual(state_error(live.capture_state(), expected), 0)
            self.assertFalse(done)
            search.record_action(action)
        self.assertGreater(search.counts["replay_steps"], 8)

    def test_cli_resumes_saved_search_report_from_initial_history(self):
        root = Path(__file__).resolve().parents[1]
        main = runpy.run_path(str(root/"scripts/evaluate_contact_sequence.py"))["main"]
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/"first", Path(directory)/"second"
            for output, extra in ((first, []), (second, ["--resume-report", str(first/"report.json"), "--resume-step", "2"])):
                args = ["evaluate_contact_sequence.py", "--source-root", os.environ["TDYSSEY_OFFICIAL_ROOT"],
                        "--output", str(output), "--steps", "4", "--seconds", "30",
                        "--horizon", "4", "--max-evaluations", "2", *extra]
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    main()
            report = json.loads((second/"report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["counts"]["control_steps"], 2)
            self.assertEqual(report["counts"]["resume_replay_controls"], 2)
            self.assertTrue(report["resume_sources_unchanged"])
            self.assertEqual(max(row["prediction_state_error"] for row in report["steps"]), 0)
            self.assertFalse(report["torch_loaded"])


if __name__ == "__main__":
    unittest.main()
