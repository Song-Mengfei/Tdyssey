"""Pure saved-array/mock index checks: no env, renderer, model or training iterator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import native_rgb_stage2c_indices as indices
from scripts.native_rgb_stage2c_report import select_examples


def record(tid="val-A-0",letter="A",n=64):
    states=np.zeros((n+1,15));states[:,:2]=[70,70];states[:,2:4]=[256,256]
    steps=[]
    for t in range(n):
        delta=3. if t<20 else 0. if t<45 else 1.
        states[t+1,2:4]=states[t,2:4]+[delta,0]
        states[t+1,4]=states[t,4]+(.02 if t==12 else 0.)
        steps.append(dict(phase="mock",intent="mock",approach_kind="normal"))
    arrays=dict(states=states,actions=np.full((n,2),256.),contact=np.asarray([1 if t%2 else 0 for t in range(n)]),
        valid_transition=np.ones(n,bool),penetrations=np.zeros((n+1,3)),substep_penetrations=np.zeros((n,10,3)))
    return dict(entry=dict(trajectory_id=tid,shape=letter,split="seen_val",stage="2C",file=tid+".npz",metadata_file=tid+".json",
        sha256="a"*64,metadata_sha256="b"*64,location="center",seed=1),arrays=arrays,steps=steps,
        scene=dict(selection_version="motion-room-planned-attempts-v2"))


def catalog(count=9):
    return indices.catalog_from_records([record(f"val-A-{i}") for i in range(count)])


class Stage2CIndicesTests(unittest.TestCase):
    def setUp(self):self.spec=indices.load_spec()

    def test_predeclared_seed_quotas_and_domains(self):
        self.assertEqual(self.spec["algorithm_seed"],2026102903)
        self.assertEqual(sum(self.spec["extra"]["quotas"].values()),9)
        self.assertEqual(self.spec["hash"]["domains"],["natural","extra","development-natural","development-extra-moving"])
        self.assertEqual(self.spec["complete_64_control_anchor_range_inclusive"],[3,60])

    def test_complete64_has_exact_H1_H4_bounds_and_original_hashes(self):
        value=catalog(1)
        self.assertEqual(value["valid_h1_candidates"],61)
        self.assertEqual(value["valid_h4_candidates"],58)
        self.assertEqual([row["anchor"] for row in value["candidates"]],list(range(3,61)))
        self.assertEqual(value["trajectories"][0]["npz_sha256"],"a"*64)
        self.assertTrue(all(row["h4"]["legal_transition_end_inclusive"]==row["anchor"]+3 for row in value["candidates"]))

    def test_invalid_history_and_last_future_transition_excluded(self):
        value=record();value["arrays"]["valid_transition"][[1,63]]=False
        actual=indices.catalog_from_records([value])
        anchors={row["anchor"] for row in actual["candidates"]}
        self.assertNotIn(3,anchors);self.assertNotIn(4,anchors);self.assertNotIn(60,anchors)
        self.assertIn(5,anchors);self.assertIn(59,anchors)
        self.assertEqual(actual["trajectories"][0]["invalid_h4_anchor_ids"],[3,4,60])

    def test_H1_boundary_uses_actual_negative_gap_and_not_initial_location(self):
        from tdyssey.envs.official import official_shape
        value=record(n=12)
        left=min(point[0] for component in official_shape("A").components for point in component.vertices)
        value["arrays"]["states"][:,2]=7-left-.1
        value["arrays"]["states"][:,4]=0
        value["arrays"]["states"][4:,2]+=2.
        actual=indices.catalog_from_records([value])
        anchor=next(row for row in actual["candidates"] if row["anchor"]==3)
        self.assertTrue(anchor["conditions"]["boundary_moving"])
        self.assertLess(anchor["h1"]["actual_boundary_min_world_px"],0)
        self.assertEqual(anchor["h1"]["actual_boundary_gap_band"],"negative")
        self.assertEqual(anchor["h1"]["requested_spatial_class"],"center")

    def test_H4_endpoint_and_path_labels_do_not_collapse_into_H1(self):
        value=record(n=12)
        value["arrays"]["states"][:,2:4]=[256,256]
        value["arrays"]["states"][:,4]=0
        value["arrays"]["states"][4:6,2]+=3
        actual=indices.catalog_from_records([value])
        anchor=next(row for row in actual["candidates"] if row["anchor"]==3)
        self.assertTrue(anchor["h1"]["moving"])
        self.assertTrue(anchor["h4"]["path_any_step_moving"])
        self.assertTrue(anchor["h4"]["endpoint_near_static"])
        self.assertFalse(anchor["h4"]["endpoint_moving"])

    def test_deterministic_natural_uniform_hash_and_permutation_stability(self):
        value=catalog()
        first=indices.select_indices(value,self.spec)
        shuffled=copy.deepcopy(value);shuffled["candidates"].reverse()
        second=indices.select_indices(shuffled,self.spec)
        self.assertEqual([row["window_id"] for row in first["combined"]],[row["window_id"] for row in second["combined"]])
        self.assertEqual(len(first["natural"]),9)
        self.assertGreater(len({row["anchor"] for row in first["natural"]}),1)
        self.assertTrue(all(3<=row["anchor"]<=60 for row in first["natural"]))
        trajectory={row["trajectory_id"]:row for row in value["trajectories"]}
        for chosen in first["natural"]:
            pool=[row for row in value["candidates"] if row["trajectory_id"]==chosen["trajectory_id"]]
            expected=min(pool,key=lambda row:indices._rank(row,trajectory[row["trajectory_id"]],self.spec,"natural"))
            self.assertEqual(chosen["window_id"],expected["window_id"])

    def test_rare_support_frozen_after_natural_and_missing_slots_never_replaced(self):
        value=catalog()
        actual=indices.select_indices(value,self.spec)
        natural={row["window_id"] for row in actual["natural"]}
        remaining=[row for row in value["candidates"] if row["window_id"] not in natural]
        support=actual["extra_support_after_natural"]["A"]
        for category in indices.CATEGORIES:
            self.assertEqual(support["counts"][category],sum(row["conditions"][category] for row in remaining))
        expected=sorted(indices.CATEGORIES,key=lambda category:(support["counts"][category],indices.CATEGORIES.index(category)))
        self.assertEqual(support["order"],expected)
        missing=[row for row in actual["missing_slots"] if row["panel"]=="extra" and row["letter"]=="A"]
        extra=[row for row in actual["extra"] if row["letter"]=="A"]
        self.assertEqual(len(missing)+len(extra),9)
        self.assertTrue(any(row["category"]=="likely_wall_blocked_proxy" for row in missing))
        self.assertFalse(natural & {row["window_id"] for row in actual["extra"]})

    def test_extra_trajectory_load_balancing_and_all_conditional_labels_retained(self):
        actual=indices.select_indices(catalog(),self.spec)
        for chosen in actual["extra"]:
            self.assertTrue(chosen["conditions"][chosen["selection_reason"]["category"]])
            self.assertEqual(chosen["conditional_panels"],[category for category in indices.CATEGORIES if chosen["conditions"][category]])
            self.assertGreaterEqual(chosen["selection_reason"]["selected_trajectory_load_before"],1)
        ids=[row["window_id"] for row in actual["combined"]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(actual["training_index_draws"],0)

    def test_development_uses_extra_moving_truth_even_when_allocated_rotation(self):
        value=catalog(1)
        for row in value["candidates"]:
            row["conditions"]={category:False for category in indices.CATEGORIES}
            row["conditions"].update(moving=True,rotation=True)
        actual=indices.select_indices(value,self.spec)
        dev=[row for row in actual["development"] if row["letter"]=="A"]
        self.assertEqual(len(dev),2)
        self.assertEqual(len({row["window_id"] for row in dev}),2)
        moving=next(row for row in dev if row["development_role"]=="extra_moving")
        extras=[row for row in actual["extra"] if row["conditions"]["moving"]]
        tr={row["trajectory_id"]:row for row in value["trajectories"]}
        expected=min(extras,key=lambda row:indices._rank(row,tr[row["trajectory_id"]],self.spec,"development-extra-moving"))
        self.assertEqual(moving["window_id"],expected["window_id"])

    def test_time_overlap_is_reported_without_merging_different_anchors(self):
        rows=[dict(trajectory_id="a",window_id="a:3",anchor=3),dict(trajectory_id="a",window_id="a:6",anchor=6),
              dict(trajectory_id="a",window_id="a:30",anchor=30),dict(trajectory_id="b",window_id="b:3",anchor=3)]
        result=indices._overlap(rows)
        self.assertEqual(result["counts"]["same_trajectory_pairs"],3)
        self.assertEqual(result["counts"]["H4_future_transition_overlap"],1)
        self.assertEqual(len(result["correlated_pairs"]),1)

    def test_wrong_split_held_letters_and_duplicate_windows_rejected(self):
        value=record();value["entry"]["split"]="train"
        with self.assertRaises(ValueError):indices.catalog_from_records([value])
        value=record(letter="D")
        with self.assertRaises(ValueError):indices.catalog_from_records([value])
        value=catalog(1);value["candidates"].append(value["candidates"][0])
        with self.assertRaises(ValueError):indices.select_indices(value,self.spec)

    def test_bounded_example_missing_and_same_letter_fallback_are_explicit(self):
        rules=dict(cases=[dict(letter="A",desired_category="likely_wall_blocked_proxy"),dict(letter="P",desired_category="rotation")],
                   max_cases=4,max_sheet_rows=4,max_gifs=2,max_consecutive_frames_per_case=8)
        result=select_examples([record()],rules)
        self.assertEqual(len(result["cases"]),1)
        self.assertTrue(result["cases"][0]["fallback_used"])
        self.assertEqual(len(result["missing"]),2)

    def _frozen_fixture(self,stage):
        data=stage/"data";data.mkdir()
        snapshots=stage/"source-snapshot";snapshots.mkdir()
        policy=snapshots/"frozen-index-spec.json";policy.write_bytes(indices.DEFAULT_SPEC.read_bytes())
        source=snapshots/"scripts/native_rgb_stage2c_indices.py";source.parent.mkdir()
        source.write_bytes(Path(indices.__file__).read_bytes())
        identity=dict(files=[dict(snapshot=p.relative_to(stage).as_posix(),sha256=indices.sha(p)) for p in (policy,source)])
        (stage/"source-identity.json").write_text(json.dumps(identity),encoding="utf-8")
        manifest=dict(stage="2C",split="seen_val",failure=None,unexecuted_request_ids=[],actual_controls=10368,saved_controls=10368,
            entries=[dict(controls=64,error=None,invalid_controls=0) for _ in range(162)],
            source_identity_sha256=indices.sha(stage/"source-identity.json"),index_spec_sha256=indices.sha(policy))
        (data/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        audit=dict(stage="2C",split="seen_val",passed=True,physics_collection_gate_passed=True,
            manifest_sha256=indices.sha(data/"manifest.json"),threshold_world_px=2,threshold_relaxed=False,
            invalid_controls=0,controls=10368,audited_substeps=103680)
        (stage/"physics-audit.json").write_text(json.dumps(audit),encoding="utf-8")
        return data,policy

    def test_production_gate_binds_policy_seed_to_prephysics_snapshot(self):
        with tempfile.TemporaryDirectory(dir=indices.ROOT/"artifacts",prefix="mock-2c-index-") as temporary:
            data,policy=self._frozen_fixture(Path(temporary))
            result=indices.verify_frozen_inputs(data,policy)
            self.assertEqual(result["policy_sha256"],indices.sha(indices.DEFAULT_SPEC))
            changed=copy.deepcopy(self.spec);changed["algorithm_seed"]+=1
            alternate=Path(temporary)/"alternate-policy.json";alternate.write_text(json.dumps(changed),encoding="utf-8")
            with self.assertRaises(ValueError):indices.verify_frozen_inputs(data,alternate)

    def test_production_gate_rejects_failed_audit_and_changed_snapshot(self):
        with tempfile.TemporaryDirectory(dir=indices.ROOT/"artifacts",prefix="mock-2c-index-") as temporary:
            data,policy=self._frozen_fixture(Path(temporary))
            audit_path=Path(temporary)/"physics-audit.json";audit=json.loads(audit_path.read_text())
            audit["physics_collection_gate_passed"]=False;audit_path.write_text(json.dumps(audit),encoding="utf-8")
            with self.assertRaises(ValueError):indices.verify_frozen_inputs(data,policy)
            audit["physics_collection_gate_passed"]=True;audit_path.write_text(json.dumps(audit),encoding="utf-8")
            snapshot=Path(temporary)/"source-snapshot/scripts/native_rgb_stage2c_indices.py"
            snapshot.write_text("changed",encoding="utf-8")
            with self.assertRaises(ValueError):indices.verify_frozen_inputs(data,policy)


if __name__=="__main__":unittest.main()
