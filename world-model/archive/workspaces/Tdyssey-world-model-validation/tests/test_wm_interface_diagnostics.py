"""Saved-array/fake-decoder checks, with no simulator or model dependency."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tdyssey.wm_interface_diagnostics import (analyse, confusion, measurement_view,
    rank_summary, ranking_table, sequence_table, weighted_measurements)
from tdyssey.wm_measurement_contract import coverage_measurement, measurement_frame, DEVELOPMENT_BOUND


POSE = {"position": [240, 245], "angle": 0.0}
GOAL = {"position": [240, 245], "angle": 0.0}


def fake_measurement(estimate=.4):
    return measurement_frame(state={"valid": True, "pose": deepcopy(POSE)},
        shape={"valid": True}, pusher={"valid": False, "position": None},
        coverage=coverage_measurement(estimate, half_width=.03,
            uncertainty_kind=DEVELOPMENT_BOUND, valid=estimate is not None))


class InterfaceOfflineChecks(unittest.TestCase):
    def test_unassessed_ambiguity_is_unknown_and_v1_has_no_interval_table(self):
        from tdyssey.wm_interface_diagnostics import stratified_tables
        measurement=fake_measurement()
        measurement["state"]["ambiguity"]="diagnostic note, not an ambiguity assessment"
        view=measurement_view(measurement,"v2")
        self.assertFalse(view["ambiguity_assessed"])
        self.assertIsNone(view["ambiguity"])
        self.assertEqual(measurement_view(None,"v2")["reason"],"missing_input")
        self.assertIsNone(measurement_view({"state":None},"v2")["ambiguity"])
        _, classifications=stratified_tables([])
        self.assertFalse(any(r["version"]=="v1" and r["method"]=="declared_interval" for r in classifications))
        self.assertTrue(any(r["version"]=="v2" and r["method"]=="declared_interval" for r in classifications))

    def test_invalid_object_pose_does_not_hide_independent_pusher(self):
        measurement=measurement_frame(state={"valid":False,"pose":None,"reason":"shape_mismatch"},
            shape={"valid":False,"reason":"shape_mismatch"},
            pusher={"valid":True,"position":[60,60],"reason":"ok"},
            coverage=coverage_measurement(None,valid=False))
        view=measurement_view(measurement,"v2")
        self.assertFalse(view["object_valid"])
        self.assertFalse(view["shape_valid"])
        self.assertIsNone(view["pose"])
        self.assertFalse(view["coverage_valid"])
        self.assertEqual(view["termination"],"unknown")
        self.assertTrue(view["pusher_valid"])
        self.assertEqual(view["pusher"],[60,60])

    def rank_fixture(self, *, raw_valid=False, sequence_valid=False):
        return [dict(version="v2",shape="L",anchor="L00",candidate=candidate,horizon=4,
                     noise_seed=seed,actual_coverage=value,raw_endpoint_coverage=value,
                     raw_endpoint_valid=raw_valid,sequence_score=value,
                     sequence_score_eligible=sequence_valid,physical_absorbing=False)
                for candidate,value in zip(("hold","toward","side","away"),(.1,.2,.3,.4))
                for seed in (5478,5479,5480)]

    def test_finite_invalid_diagnostic_scores_cannot_enter_ranking(self):
        fixture=self.rank_fixture()
        original=deepcopy(fixture)
        ranks=ranking_table(fixture)
        self.assertEqual(fixture,original)
        for row in ranks:
            self.assertTrue(row["selection_abstention"])
            self.assertFalse(row["prediction_complete"])
            self.assertIsNone(row["selected_candidate"])
            self.assertEqual(row["eligible_pairs"],6)
            self.assertEqual(row["scored_pairs"],0)
            self.assertEqual(row["abstained_pairs"],6)
            self.assertEqual(row["incorrect_scored_pairs"],0)
            self.assertIsNone(row["observed_pair_accuracy"])
            self.assertIsNone(row["pair_accuracy"])
            self.assertEqual(row["conservative_pair_accuracy"],0)
            self.assertEqual(row["diagnostic_values_away"],[.4,.4,.4])
            self.assertEqual(row["eligible_noises_away"],0)
        for row in rank_summary(ranks):
            self.assertIsNone(row["anchor_equal_observed_pair_accuracy"])
            self.assertEqual(row["anchors_with_scored_pairs"],0)
            self.assertEqual(row["abstained_pairs"],6)

    def test_noise_validity_is_mode_specific_and_all_three_are_required(self):
        fixture=self.rank_fixture(raw_valid=True,sequence_valid=False)
        fixture[-1]["raw_endpoint_valid"]=False
        ranks={r["scoring_mode"]:r for r in ranking_table(fixture)}
        raw=ranks["raw_endpoint"]
        self.assertTrue(raw["selection_abstention"])
        self.assertEqual(raw["eligible_noises_away"],2)
        self.assertEqual(raw["diagnostic_values_away"],[.4,.4,.4])
        self.assertEqual(raw["scored_pairs"],3)
        self.assertEqual(raw["abstained_pairs"],3)
        self.assertEqual(raw["observed_pair_accuracy"],1)
        self.assertEqual(raw["conservative_pair_accuracy"],.5)
        stopped=ranks["stopping_consistent"]
        self.assertEqual(stopped["scored_pairs"],0)
        self.assertIsNone(stopped["observed_pair_accuracy"])
        self.assertTrue(stopped["selection_abstention"])
        # A complete raw endpoint diagnostic still cannot authorize a stopped
        # score when the independently declared sequence eligibility is false.
        fixture[-1]["raw_endpoint_valid"]=True
        raw,stopped=ranking_table(fixture)
        self.assertEqual(raw["selected_candidate"],"away")
        self.assertEqual(raw["scored_pairs"],6)
        self.assertTrue(stopped["selection_abstention"])

    def test_fixed_inventory_and_cache_preserve_missing_and_original_v1(self):
        with tempfile.TemporaryDirectory(prefix="wm-interface-fake-") as directory:
            root=Path(directory)
            rgb=np.full((3,96,96),.5,dtype=np.float32)
            state={"object_body":{"pose":deepcopy(POSE)},"pusher_body":{"pose":{"position":[60,60],"angle":0}},"goal":deepcopy(GOAL)}
            truth={"pose":deepcopy(POSE),"pusher":[60,60],"coverage":.4}
            stored={"valid":False,"reason":"original_failure","pose":None,"pusher":None,"coverage":None}
            steps=[{"horizon":h,"decoded":stored,"score_coverage":None} for h in range(1,5)]
            branch={"anchor":"L00","candidate":"hold","shape":"L","status":"complete", "family":"noncontact",
                "initial_state":state,"latest":truth,"history":[{"state":state} for _ in range(3)],
                "actual":[{"horizon":h,"truth":truth,"contacts":0,"absorbing":False,"success":False} for h in range(1,5)],
                "predictions":[{"seed":seed,"steps":steps} for seed in (5478,5479,5480)]}
            report={"branches":[branch],"calibrations":[{"id":"cal-L-00","state":state,"truth":truth,"prediction":stored}]}
            path=root/"report.json"; path.write_text(json.dumps(report))
            np.savez_compressed(root/"cal-L-00.npz",rgb=rgb)
            np.savez_compressed(root/"L00-hold-physical.npz",history=np.stack([rgb]*4),actual=np.stack([rgb]*4))
            for seed in (5478,5479,5480):
                np.savez_compressed(root/f"L00-hold-noise-{seed}.npz",raw=np.stack([rgb]*4))
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            calls={"v1":0,"v2":0}
            def first(image,shape,goal):
                calls["v1"]+=1
                self.assertEqual(shape,"L"); self.assertEqual(goal,GOAL)
                return {"valid":True,"reason":"fake","pose":deepcopy(POSE),"pusher":[60,60],"coverage":.4}
            def second(image,shape,goal):
                calls["v2"]+=1
                self.assertEqual(shape,"L"); self.assertEqual(goal,GOAL)
                return fake_measurement()
            result=analyse(root,decoder_v1=first,decoder_v2=second)
            self.assertEqual(result["summary"]["expected_image_rows"],1304)
            self.assertEqual(result["summary"]["available_image_rows"],21)
            for operation in ("environment_creations","resets","restores","true_renders","controls",
                              "model_inference","policy_sampling","optimizer_updates"):
                self.assertEqual(result["summary"][operation],0)
            self.assertEqual(len(result["flat_images"]),2608)
            self.assertEqual(len(result["sequences"]),1536)
            self.assertEqual(len(result["baselines"]),256)
            self.assertEqual(calls,{"v1":1,"v2":1})
            original=next(r for r in result["images"] if r["id"]=="L00-hold-noise-5478-H1")
            self.assertEqual(original["v1"]["measurement"]["reason"],"original_failure")
            self.assertEqual(original["v1"]["origin"],"stored_report")
            self.assertTrue(original["v2"]["view"]["object_valid"])
            self.assertFalse(original["v2"]["view"]["pusher_valid"])
            sequence=next(r for r in result["sequences"] if r["version"]=="v2" and r["anchor"]=="L00" and r["candidate"]=="hold" and r["noise_seed"]==5478 and r["horizon"]==4)
            self.assertTrue(sequence["sequence_score_eligible"])
            self.assertEqual(sequence["sequence_score"],.4)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_endpoint_measurement_survives_unknown_prefix(self):
        rows=[]
        for horizon in range(1,5):
            measurement=fake_measurement(None if horizon==1 else .4)
            rows.append(dict(source="forecast",anchor="L00",candidate="hold",noise_seed=5478,horizon=horizon,
                shape="L",family="noncontact",truth={"coverage":.4},physical_absorbing=False,
                v1={"view":{"coverage_valid":False,"coverage":None,"termination":"unknown"}},
                v2={"measurement":measurement,"view":measurement_view(measurement,"v2")}))
        result=sequence_table(rows)
        endpoint=next(r for r in result if r["version"]=="v2" and r["horizon"]==4)
        self.assertTrue(endpoint["raw_endpoint_valid"])
        self.assertEqual(endpoint["raw_endpoint_coverage"],.4)
        self.assertFalse(endpoint["sequence_score_eligible"])
        self.assertIsNone(endpoint["sequence_score"])

    def test_anchor_equal_noise_nesting_and_missing_coverage(self):
        rows=[dict(anchor=anchor,candidate="hold",horizon=4,error=value) for anchor,values in
              (("L00",[1,1,1]),("L01",[3,None,None])) for value in values]
        result=weighted_measurements(rows,"error")
        self.assertEqual(result["anchor_equal_observed_mean"],2)
        self.assertEqual(result["measured_image_rows"],4)
        self.assertEqual(result["expected_image_rows"],6)
        self.assertEqual(result["fully_measured_branch_frames"],1)
        self.assertFalse(result["passes_validation_gate"])

    def test_no_positive_precision_unknowns_and_independent_baselines(self):
        rows=[dict(actual_success=True,nominal_predicted_success=False,termination="unknown"),
              dict(actual_success=False,nominal_predicted_success=False,termination="non_success")]
        nominal=confusion(rows)
        self.assertIsNone(nominal["precision"])
        self.assertEqual(nominal["FN"],1)
        interval=confusion(rows,interval=True)
        self.assertEqual(interval["unknown"],1)
        sequences=[]
        for candidate,actual in zip(("hold","toward","side","away"),(.1,.2,.4,.3)):
            for seed in (5478,5479,5480):
                sequences.append(dict(version="v2",shape="L",anchor="L00",candidate=candidate,horizon=4,
                    noise_seed=seed,actual_coverage=actual,raw_endpoint_coverage=None,sequence_score=None,physical_absorbing=False))
        ranks=ranking_table(sequences)
        self.assertEqual(len(ranks),2)
        for row in ranks:
            self.assertTrue(row["selection_abstention"])
            self.assertAlmostEqual(row["hold_regret"],.3)
            self.assertAlmostEqual(row["uniform_candidate_expectation_regret"],.15)
            self.assertEqual(row["regret_bound_upper"],1)


if __name__=="__main__":
    unittest.main()
