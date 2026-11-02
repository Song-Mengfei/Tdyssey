"""R2B reporter analytic fixtures only, without any environment construction."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import native_rgb_r2b_report as report

POLICY = "a"*64
RULES = "b"*64


def fixture(n=24):
    states = np.zeros((n+1, 15), dtype=np.float64)
    states[:, :4] = (100., 100., 256., 256.)
    states[:, 2] += np.arange(n+1)*2.
    actions = np.full((n, 2), 120., dtype=np.float64)
    arrays = dict(states=states, actions=actions, attempted_actions=actions.copy(),
        rgb=np.zeros((n+1, 3, 96, 96), dtype=np.uint8))
    request = dict(request_id="mock", family_id="mock-family", letter="T", initial_space="center",
        role="rotation-positive-intent", process_goal="interior", max_controls=24,
        protocol=report.SELECTED_ARM, protocol_sha256=report.SELECTED_PROTOCOL_SHA256)
    records, decisions, logs = [], [], []
    for t in range(n):
        decision = dict(control_index=t, state_before_sha256=report.state_sha(states[t]),
            proposed_action=actions[t].tolist(), action_absolute=actions[t].tolist(),
            current_state15=states[t].tolist(), planned_phase=report.planned_phase(t),
            actual_phase="mock-"+report.planned_phase(t), policy_sha256=POLICY, policy_rules_sha256=RULES)
        records.append(dict(control_index=t, completed=True, valid=True, valid_transition=True,
            substep_count=10, max_penetrations=[0., 0., 0.], agent_object_contact_substeps=0,
            object_wall_contact_substeps=0, agent_wall_contact_substeps=0,
            actual_contact_arc_ids=[], action_phase=decision["actual_phase"], decision=decision))
        decisions.append(deepcopy(decision))
        substeps = [dict(index=s, executed=True, dt=.01, post_state=states[t+1].tolist(),
            integrator_pre_state=states[t].tolist(), penetrations=[0.,0.,0.], contacts=[]) for s in range(10)]
        logs.append(dict(control_index=t, completed=True, action=actions[t].tolist(), substeps=substeps))
    metadata = dict(control_records=records, policy_identity=dict(policy_sha256=POLICY, rules_sha256=RULES),
        initial_identity=dict(physics=dict(object=dict(center_of_gravity=[0.,45.]))))
    return request, arrays, metadata, decisions, logs


class R2BReportTests(unittest.TestCase):
    def test_true_h1_h4_and_pushinternal_denominators(self):
        req, arrays, meta, decisions, logs = fixture()
        measured = report.measure_dynamic(req, {}, arrays, meta, logs, decisions)
        self.assertEqual({k:len(v) for k,v in measured["populations"].items()},
            dict(all_controls=24, h1_windows=21, h4_windows=18, push_h4_windows=7, push_full_history_h4_windows=4))
        push = measured["populations"]["push_h4_windows"]
        self.assertEqual([r["anchor"] for r in push], list(range(4,11)))
        self.assertEqual([r["anchor"] for r in measured["populations"]["push_full_history_h4_windows"]], list(range(7,11)))
        self.assertTrue(all(r["planned_phases"] == ["push"] for r in push))
        self.assertTrue(all(r["agent_object_contact"] is False for r in push))
        self.assertNotIn("source_actions_equal_request", measured)

    def test_dynamic_action_currentstate_and_independentlog_tamper_rejected(self):
        for tamper in ("action", "state", "external"):
            req, arrays, meta, decisions, logs = fixture(4)
            if tamper == "action":
                arrays["actions"][1,0] += 1.
            elif tamper == "state":
                arrays["states"][1,0] += 1.
            else:
                decisions[1]["actual_phase"] = "different"
            with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                report.measure_dynamic(req, {}, arrays, meta, logs, decisions)

    def test_partial_trajectory_preserves_missing_window_denominators(self):
        req, arrays, meta, decisions, logs = fixture(6)
        measured = report.measure_dynamic(req, {}, arrays, meta, logs, decisions)
        self.assertEqual({k:len(v) for k,v in measured["populations"].items()},
            dict(all_controls=6,h1_windows=3,h4_windows=0,push_h4_windows=0,push_full_history_h4_windows=0))
        self.assertEqual(report.PLANNED, dict(all_controls=1296,h1_windows=1134,h4_windows=972,push_h4_windows=378,push_full_history_h4_windows=216))
        self.assertFalse(measured["release"]["data_available"])

    def test_release_observed_threshold_and_no_formula_fill(self):
        _, arrays, _, _, logs = fixture()
        states = arrays["states"]
        states[14,7], states[14,9] = 100., 10.
        for log in logs[14:20]:
            for sub in log["substeps"]:
                i = (log["control_index"]-14)*10+sub["index"]
                entry, post = states[14].copy(), states[14].copy()
                entry[7], entry[9] = 100.*(.8**i), 10.*(.8**i)
                post[7], post[9] = 100.*(.8**(i+1)), 10.*(.8**(i+1))
                sub.update(integrator_pre_state=entry.tolist(), post_state=post.tolist())
        result = report.release_observation(states, logs, "T")
        threshold = result["free_segments"][0]["thresholds"]["linear_10pct"]
        self.assertEqual(threshold["status"], "observed_first_below_in_contact_free_segment")
        self.assertAlmostEqual(threshold["observed_time_since_release"], .11)
        self.assertEqual(threshold["reference_at_release"], 100.)
        self.assertTrue(result["complete_planned_release"])

    def test_recontact_censors_and_zero_release_never_defines_threshold(self):
        _, arrays, _, _, logs = fixture()
        arrays["states"][14,7] = 100.
        for log in logs[14:20]:
            for sub in log["substeps"]:
                sub["integrator_pre_state"][7] = sub["post_state"][7] = 100.
        logs[14]["substeps"][2]["contacts"] = [dict(kind="agent_object")]
        result = report.release_observation(arrays["states"], logs, "T")
        first = result["free_segments"][0]
        self.assertEqual(first["ended_by"], "agent_object_recontact")
        self.assertEqual(first["thresholds"]["linear_10pct"]["status"], "right_censored_agent_object_recontact")
        self.assertIsNone(first["thresholds"]["linear_10pct"]["observed_time_since_release"])
        self.assertEqual(first["thresholds"]["angular_10pct"]["status"], "not_assessable_zero_or_nearzero_release_speed")

    def test_contact_prefix_does_not_rebase_release_reference(self):
        _, arrays, _, _, logs = fixture()
        arrays["states"][14,7] = 100.
        for log in logs[14:20]:
            for sub in log["substeps"]:
                sub["integrator_pre_state"][7] = sub["post_state"][7] = 5.
        for sub in logs[14]["substeps"][:3]:
            sub["contacts"] = [dict(kind="agent_object")]
        result = report.release_observation(arrays["states"], logs, "T")
        self.assertAlmostEqual(result["actual_detachment_time_seconds"], .04)
        first = result["free_segments"][0]["thresholds"]["linear_10pct"]
        self.assertEqual(first["reference_at_release"], 100.)
        self.assertEqual(first["status"], "already_below_at_segment_start_crossing_unobserved")
        self.assertIsNone(first["observed_time_since_release"])

    def test_closed_missing_slots_keep_all_five_planned_denominators_and_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rules = dict(kind="mock-frozen-rules.v1")
            rules_bytes = (json.dumps(rules)+"\n").encode()
            source_file = "tdyssey/native_rgb_selected_physics.py"
            policy = dict(kind=rules["kind"], rules_sha256=hashlib.sha256(report.canonical(rules).encode()).hexdigest(),
                rules_file="frozen-rules.json", rules_file_sha256=hashlib.sha256(rules_bytes).hexdigest(),
                source_file=source_file, source_sha256=report._sha(report.ROOT/source_file))
            policy["policy_sha256"] = hashlib.sha256(report.canonical({k:policy[k] for k in ("kind","rules_sha256","source_sha256")}).encode()).hexdigest()
            requests = [dict(request_id=f"{letter}-{location}", family_id=f"family-{letter}-{location}", letter=letter,
                initial_space=location, max_controls=24, protocol=report.SELECTED_ARM,
                protocol_sha256=report.SELECTED_PROTOCOL_SHA256) for letter in report.LETTERS
                for location in ("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")]
            for name, value in (("frozen-requests.json",dict(requests=requests,rules=rules)),
                    ("frozen-config.json",dict(policy_identity=policy)),
                    ("index.json",dict(collection_closed=True,slots=[dict(request_id=r["request_id"],slot_id=r["request_id"],status="not_executed") for r in requests])),
                    ("resource-ledger.json",dict(closed=True))):
                (root/name).write_text(json.dumps(value), encoding="utf-8")
            (root/"frozen-rules.json").write_bytes(rules_bytes)
            sources = []
            for i, path in enumerate([root/"frozen-requests.json",root/"frozen-config.json",root/"frozen-rules.json",report.ROOT/source_file]):
                snapshot = root/"source-snapshot"/str(i)
                snapshot.parent.mkdir(exist_ok=True)
                snapshot.write_bytes(path.read_bytes())
                sources.append(dict(source=str(path),snapshot=snapshot.relative_to(root).as_posix(),sha256=report._sha(path)))
            (root/"source-identity.json").write_text(json.dumps(dict(files=sources)),encoding="utf-8")
            result = report.build_report(root)
            self.assertEqual(len(result["slots"]), 54)
            self.assertFalse(result["data_errors"])
            for name, planned in report.PLANNED.items():
                self.assertEqual(result["summaries"][name]["missing_planned"], planned)
            (root/"frozen-rules.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):
                report.build_report(root)


if __name__ == "__main__":
    unittest.main()
