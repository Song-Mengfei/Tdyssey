"""Reference-only merge and cohort statistics; no environment steps or model."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import native_rgb_stage2b_report as b
from tdyssey.native_rgb_env import CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS


def record(identifier,stage="2B",letter="M",location="center",category="moving",seed=1):
    states=np.zeros((65,15));states[:,:2]=(100+seed,100);states[:,2:4]=(220,220)
    if category=="moving":states[:,2]+=np.arange(65)*2
    if category=="rotation":states[:,4]=np.arange(65)*.02
    arrays=dict(states=states,actions=np.tile([256.,400.],(64,1)),contact=np.ones(64) if category=="near_static_contact" else np.zeros(64),
        valid_transition=np.ones(64,bool),penetrations=np.zeros((65,3)),substep_penetrations=np.zeros((64,10,3)))
    steps=[dict(phase="push",intent="push") for _ in range(64)]
    if category=="failure":steps[63]["observed"]=dict(recovery=dict(episode_id=1,status="failed",failure_reason="path_timeout"))
    entry=dict(trajectory_id=identifier,shape=letter,location=location,split="train",stage=stage,controls=64,seed=seed,
        file=identifier+".npz",metadata_file=identifier+".json",initial_scene_hash=hashlib.sha256(states[0].tobytes()).hexdigest())
    return dict(entry=entry,arrays=arrays,steps=steps)


def write_sources(root):
    originals={}
    for stage,records in (("2A",[record("old",stage="2A",category="static",seed=1)]),
                          ("2B",[record("new1",seed=2),record("new2",letter="O",seed=3)])):
        directory=root/b.COHORTS[stage][0];directory.mkdir(parents=True);entries=[]
        for item in records:
            entry=item["entry"].copy();np.savez_compressed(directory/entry["file"],**item["arrays"])
            scene={} if stage=="2A" else dict(selection_version=b.NEW_SELECTOR)
            (directory/entry["metadata_file"]).write_text(json.dumps(dict(scene=scene,steps=item["steps"])))
            entry.update(sha256=b.a.sha(directory/entry["file"]),metadata_sha256=b.a.sha(directory/entry["metadata_file"]))
            entries.append(entry)
        manifest=dict(entries=entries,split="train",stage=stage,state_columns=list(STATE_COLUMNS),source_root="physics-source",
            controller=CONTROLLER,protocol_sha256=PROTOCOL_SHA256)
        (directory/"manifest.json").write_text(json.dumps(manifest));originals[stage]=manifest
    return originals


def save_view(root):
    value=b.make_merged_manifest(root);path=root/"view.json";path.write_text(json.dumps(value));return path,value


class Stage2BReportTests(unittest.TestCase):
    def test_reference_view_preserves_headers_and_original_hashes_without_copying(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_sources(root);path,view=save_view(root)
            self.assertEqual(view["base_root"],str(root.resolve()));self.assertEqual(view["npz_files_copied"],0)
            self.assertEqual(view["state_columns"],list(STATE_COLUMNS));self.assertEqual(view["controller"],CONTROLLER)
            self.assertEqual(view["selector_versions"],{b.LEGACY_SELECTOR:1,b.NEW_SELECTOR:2})
            self.assertTrue(view["entries"][0]["file"].startswith("stage2a/data/"))
            records=b.read_merged_records(path,root)
            self.assertEqual(len(records),3);self.assertEqual(len(list(root.rglob("*.npz"))),3)
            self.assertFalse(view["training_consumption_authorized"])

    def test_original_npz_or_metadata_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_sources(root);path,view=save_view(root)
            target=root/view["entries"][0]["metadata_file"];target.write_text(target.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):b.read_merged_records(path,root)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_sources(root);path,view=save_view(root)
            with (root/view["entries"][0]["file"]).open("ab") as stream:stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):b.read_merged_records(path,root)

    def test_source_path_escape_and_nontrain_are_rejected_before_outside_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);values=write_sources(root);manifest=root/"stage2b/data/manifest.json"
            values["2B"]["entries"][0]["file"]="../../outside.npz";manifest.write_text(json.dumps(values["2B"]))
            with self.assertRaisesRegex(ValueError,"escapes"):b.make_merged_manifest(root)
            values["2B"]["entries"][0].update(split="seen_val",metadata_file="not-opened.json");manifest.write_text(json.dumps(values["2B"]))
            with self.assertRaisesRegex(ValueError,"formal2A/2B TRAIN"):b.make_merged_manifest(root)

    def test_forged_view_headers_identity_and_omission_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_sources(root);path,view=save_view(root)
            for mutation,pattern in ((lambda v:v.update(controller="other"),"header"),
                                     (lambda v:v["entries"][0].update(shape="L"),"identity"),
                                     (lambda v:v["entries"].pop(),"omitted")):
                changed=copy.deepcopy(view);mutation(changed);path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,pattern):b.read_merged_records(path,root)

    def test_source_manifest_stage_and_source_identity_evidence_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sources=write_sources(root);path,view=save_view(root)
            changed=copy.deepcopy(view);changed["source_manifests"]["2A"]["source_identity_sha256"]="forged"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"source identity differs"):b.read_merged_records(path,root)
            sources["2B"]["stage"]="2A";(root/"stage2b/data/manifest.json").write_text(json.dumps(sources["2B"]))
            with self.assertRaisesRegex(ValueError,"correct formal TRAIN cohort"):b.make_merged_manifest(root)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sources=write_sources(root);identity=root/"stage2a/source-identity.json";identity.write_text("{}")
            sources["2A"]["source_identity_sha256"]=b.a.sha(identity)
            (root/"stage2a/data/manifest.json").write_text(json.dumps(sources["2A"]))
            path,_=save_view(root);identity.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"source identity file changed"):b.read_merged_records(path,root)

    def test_duplicate_scenes_seeds_and_missing_selector_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);values=write_sources(root);manifest=root/"stage2b/data/manifest.json"
            values["2B"]["entries"][0]["seed"]=1;manifest.write_text(json.dumps(values["2B"]))
            with self.assertRaisesRegex(ValueError,"duplicate formal identity"):b.make_merged_manifest(root)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);values=write_sources(root);entry=values["2B"]["entries"][0]
            metadata=root/"stage2b/data"/entry["metadata_file"];metadata.write_text(json.dumps(dict(scene={},steps=[])))
            with self.assertRaisesRegex(ValueError,"selector version"):b.make_merged_manifest(root)

    def test_three_cohorts_keep_planned_denominators_and_old_failure_separate(self):
        old=[record("old",stage="2A",category="static")];new=[record("new",seed=2)]
        old_report=b.cohort_report(old,"2A",9);new_report=b.cohort_report(new,"2B",13);merged=b.cohort_report(old+new,"merged",22)
        self.assertEqual(new_report["planned_controls"],14976);self.assertEqual(new_report["planned_h1"],14274)
        self.assertEqual(new_report["planned_h4"],13572);self.assertEqual(merged["planned_h1"],24156)
        self.assertEqual(merged["planned_h4"],22968);self.assertEqual(merged["actual_h1"],122)
        self.assertFalse(old_report["per_letter"]["M"]["h1_gate"]["passed"])
        self.assertTrue(new_report["per_letter"]["M"]["h1_gate"]["passed"])
        self.assertEqual(merged["M_scenes"]["center"]["h1"]["motion"],dict(moving=61,near_static=61))
        self.assertEqual(merged["strata_support"]["index_draws"],0)

    def test_predeclared_examples_new_only_and_bounded_saved_frame_render(self):
        specifications=[("M","left-top","moving"),("M","center","failure"),("O","center","moving"),
            ("O","left","failure"),("P","center","rotation"),("P","right","failure"),
            ("V","center","near_static_contact"),("X","center","rotation")]
        records=[record(f"new{i}",letter=l,location=p,category=c,seed=i+1) for i,(l,p,c) in enumerate(specifications)]
        selection=b.select_examples(records);self.assertEqual(len(selection["cases"]),8)
        self.assertEqual(selection["cases"][0]["requested_spatial_class"],"left-top")
        self.assertFalse(any(v["desired_category_unavailable"] for v in selection["cases"]))
        calls=[]
        class Renderer:
            def reset_scene(self):calls.append("reset")
            def set_render_state(self,state):calls.append(float(state[0]))
            def render_clean(self):return np.zeros((3,96,96),np.uint8)
            def step(self,*args):raise AssertionError("No controls permitted")
        with tempfile.TemporaryDirectory() as folder,patch("tdyssey.native_rgb_env.make_env",return_value=Renderer()):
            output=Path(folder)/"examples";evidence=b.render_examples(records,"unused",output,selection)
            self.assertEqual(evidence["selection_rules"],b.EXAMPLE_RULES)
            self.assertEqual(len(list(output.glob("*.png"))),1);self.assertLessEqual(len(list(output.glob("*.gif"))),4)
            for v in evidence["sequences"]:self.assertEqual(v["frames"],list(range(v["frames"][0],v["frames"][0]+8)))
            self.assertEqual(len(calls),5+8*8);self.assertEqual(evidence["physics_controls_executed"],0)
        with self.assertRaisesRegex(ValueError,"new2B"):b.select_examples([record("old",stage="2A")])


if __name__=="__main__":unittest.main()
