"""JSON/marker and tiny synthetic RGB checks; no Torch, models or physics."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from unittest import TestCase, skipUnless
from unittest.mock import patch

from tdyssey import gpc_raw_fulltrain_data as raw
from tdyssey import gpc_training_schedule as legacy
from tdyssey import gpc_adaptation_data as adaptation
from tests import test_gpc_adaptation_data as image_fixtures
from tests.test_gpc_training_schedule import fixture, MarkerDataset


def source_fixture():
    data,payloads,opened,reader=fixture(controls=100)
    prefix=legacy.generate_schedule(data,record_reader=reader)
    return data,prefix,payloads


def development_fixture(contact=True):
    data,_prefix,payloads=source_fixture()
    rows=[row for row in data["episodes"] if row["role"] not in ("development",)]
    dev_payloads={}
    for shape in ("T","L"):
        source=next(row for row in rows if row.get("shape")==shape and row.get("role")=="train")
        row=deepcopy(source)
        row.update(trajectory_id="development-"+shape,role="development",split="validation")
        row["initial_state_sha256"]=hashlib.sha256(row["trajectory_id"].encode()).hexdigest()
        record=json.loads(payloads[source["trajectory_id"]])
        record["trajectory"]["trajectory_id"]=row["trajectory_id"]
        if not contact:
            for diagnostic in record["step_diagnostics"]:diagnostic["n_contacts"]=0
            row["contact_steps"]=0
        payload=legacy.schedule_bytes(record)
        row["entry"]["files"][0].update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        dev_payloads[row["trajectory_id"]]=payload
        rows.append(row)
    data["episodes"]=rows
    opened=[]
    def reader(row):
        if row["role"]!="development" or row["split"]!="validation":
            raise AssertionError("Opened a forbidden nondevelopment record")
        opened.append(row["trajectory_id"])
        return dev_payloads[row["trajectory_id"]]
    return data,opened,reader


class RawScheduleChecks(TestCase):
    def test_exact_prefix_then_unused_round_robin_visits_every_train_window_once(self):
        _data,prefix,_payloads=source_fixture()
        full=raw.build_full_schedule(prefix)
        entries=full["entries"]
        self.assertEqual(entries[:500],prefix["entries"])
        self.assertEqual(len(entries),2328)
        self.assertEqual(Counter(row["shape"] for row in entries),dict(T=1164,L=1164))
        self.assertEqual([row["shape"] for row in entries],["T","L"]*1164)
        self.assertEqual(len({(row["trajectory_id"],row["anchor"]) for row in entries}),2328)
        used={(row["trajectory_id"],row["anchor"]) for row in entries[:500]}
        for shape in ("T","L"):
            tail=[row for row in entries[500:] if row["shape"]==shape]
            identifiers=sorted({row["trajectory_id"] for row in entries if row["shape"]==shape})
            self.assertEqual([row["trajectory_id"] for row in tail[:12]],identifiers)
            self.assertEqual([row["trajectory_id"] for row in tail[12:24]],identifiers)
            for identifier in identifiers:
                anchors=[row["anchor"] for row in tail if row["trajectory_id"]==identifier]
                self.assertEqual(anchors,sorted(anchors))
                self.assertTrue(all((identifier,anchor) not in used for anchor in anchors))
            self.assertEqual(full["coverage"][shape]["appended_examples"],914)
        self.assertEqual(raw.schedule_bytes(full),raw.schedule_bytes(raw.build_full_schedule(prefix)))

    def test_consumer_keeps_finite_indices_and_rejects_order_changes(self):
        data,prefix,_payloads=source_fixture()
        full=raw.build_full_schedule(prefix)
        sha=hashlib.sha256(raw.schedule_bytes(full)).hexdigest()
        dataset=MarkerDataset(data)
        dataset.image_domain=raw.IMAGE_DOMAIN
        with patch.object(raw,"SOURCE_SHA256",data["sha256"]),patch.object(raw,"PREFIX_SHA256",full["prefix_sha256"]):
            consumer=raw.RawScheduleConsumer(full,sha,dataset)
            self.assertEqual(consumer.indices(0,500),[entry["dataset_index"] for entry in prefix["entries"]])
            self.assertEqual(len(consumer.indices(0,2328)),2328)
            expected=consumer.indices(0,1)
            consumer.value["entries"][0]["dataset_index"]+=1
            self.assertEqual(consumer.indices(0,1),expected)
            self.assertEqual(consumer.checkpoint_state(2328)["cursor"],2328)
            for cursor,count in ((-1,1),(2328,1),(2327,2),(0,0)):
                with self.assertRaises(ValueError):consumer.indices(cursor,count)
            changed=deepcopy(full)
            changed["entries"][500]["dataset_index"]+=1
            with self.assertRaises(ValueError):raw.validate_full_schedule(changed,data=data)
        with self.assertRaises(ValueError):raw.validate_full_schedule(full,data=data)

    def test_development_balances_contact_without_reading_other_roles_or_pixels(self):
        data,opened,reader=development_fixture()
        plan=raw.development_plan(data,record_reader=reader)
        self.assertEqual(opened,["development-T","development-L"])
        self.assertEqual(len(plan["entries"]),16)
        self.assertEqual(len({row["case_id"] for row in plan["entries"]}),16)
        for shape in ("T","L"):
            selected=[row for row in plan["entries"] if row["shape"]==shape]
            self.assertEqual(sum(row["contact"] for row in selected),4)
            self.assertEqual([row["anchor"] for row in selected],sorted(row["anchor"] for row in selected))
            self.assertEqual(plan["coverage"][shape]["source_scenes"],1)
            self.assertEqual(plan["coverage"][shape]["source_trajectories"],1)
        self.assertTrue(all(row["role"]=="development" for row in plan["entries"]))
        self.assertEqual(plan["image_payload_reads"],0)
        with self.assertRaises(ValueError):raw._development_record(dict(role="held_model_test"))
        for role in ("independent_validation","held_model_test","validation"):
            with self.assertRaises(ValueError):raw.RawDataset("must-not-open",role)

    def test_development_class_shortage_is_disclosed_and_stays_same_role(self):
        data,opened,reader=development_fixture(contact=False)
        plan=raw.development_plan(data,record_reader=reader)
        self.assertEqual(len(plan["entries"]),16)
        self.assertEqual(opened,["development-T","development-L"])
        for shape in ("T","L"):
            self.assertEqual(plan["coverage"][shape]["contact_quota_shortage"],4)
            self.assertEqual(plan["coverage"][shape]["noncontact_windows"],8)
            self.assertEqual(plan["coverage"][shape]["shape_window_shortage"],0)


@skipUnless(importlib.util.find_spec("numpy"),"NumPy required for synthetic RGB")
class RawWindowChecks(TestCase):
    def setUp(self):
        case=image_fixtures.AdaptationDataChecks("test_exact_four_control_window_next_target_t1_both_sides_and_shape_indices")
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.fixture=case

    def test_raw_frames_skip_T1_and_keep_original_action_loss_inputs(self):
        import numpy as np
        fixture=self.fixture
        legacy_dataset=adaptation.GPCAdaptationDataset(fixture.path)
        old=legacy_dataset[0]
        dataset=adaptation.GPCAdaptationDataset(fixture.path,image_domain=raw.IMAGE_DOMAIN)
        original=adaptation.load_episode
        with patch("tdyssey.wm_compat_transform.transform_t1",side_effect=AssertionError("Raw data applied T1")),patch.object(adaptation,"load_episode",wraps=original) as loader:
            sample=dataset[0]
            dataset[1]
            self.assertEqual(loader.call_count,1)
        trajectory,images,_=fixture.in_memory["T-train"]
        expected=images[:5].astype(np.float32)/np.float32(255)
        np.testing.assert_array_equal(sample["image"],expected)
        np.testing.assert_array_equal(sample["inputs"],expected[:4])
        np.testing.assert_array_equal(sample["target"],expected[4])
        np.testing.assert_array_equal(sample["action"],old["action"])
        np.testing.assert_array_equal(sample["absolute_action"],np.asarray([step.action for step in trajectory.transitions[:4]],dtype=np.float64))
        self.assertEqual(sample["absolute_action"].dtype,np.float64)
        self.assertEqual(len(sample["history_states"]),4)
        self.assertEqual(sample["history_states"][-1]["control_step"],3)
        self.assertEqual(sample["target_state"]["control_step"],4)
        self.assertIsNone(sample["t1_spec_sha256"])
        self.assertFalse(np.array_equal(old["image"],expected))
        dataset.image_domain="legacy_t1"
        with self.assertRaises(ValueError):dataset[0]

    def test_explicit_raw_role_refuses_independent_or_held_before_payload(self):
        with patch.object(adaptation,"read_adaptation_manifest",side_effect=AssertionError("Unexpected metadata read")):
            for role in ("independent_validation","held_model_test"):
                with self.assertRaises(ValueError):
                    adaptation.GPCAdaptationDataset("must-not-open",role,image_domain=raw.IMAGE_DOMAIN)
