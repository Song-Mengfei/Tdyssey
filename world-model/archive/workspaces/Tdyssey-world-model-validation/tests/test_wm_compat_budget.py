"""Pure stdlib/fake accounting checks; no decoder, model, or physical environment."""

import hashlib
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tdyssey.wm_compat_budget import (
    BudgetError, CompatBudget, EXACT_CHECKPOINT, IMAGE_BYTES,
    canonical_plan, call_key,
)
from tdyssey.wm_compat_storage import Coordinator, StorageError, create_lock_file
from tdyssey.wm_compat_identity import ATTEMPT_ID, REGISTRY


HASH = "a" * 64
BINDING = dict(protocol_sha256=HASH, code_sha256={"fake.py": HASH},
               asset_sha256={"fake-input.npz": HASH}, checkpoint_sha256=EXACT_CHECKPOINT)


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class CompatibilityBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.ledger = self.repo / "artifacts/wm-input-compat-test-ledger"
        self.output = self.repo / "artifacts/wm-input-compat-test-output/results"
        self.clock = Clock()

    def create(self, **kwargs):
        return CompatBudget.create(self.repo, self.ledger, self.output, BINDING,
                                   clock=self.clock, **kwargs)

    def ready(self, **kwargs):
        budget = self.create(**kwargs)
        budget.advance_m(True)
        budget.reserve_model_load()
        budget.complete_model_load()
        return budget

    def test_v3_claim_is_independent_once_only_and_preserves_old_spent_claims(self):
        sentinels = {}
        for version in ("v1","v2"):
            old = self.repo/("artifacts/wm-input-compat-eval-"+version+"-registry")
            old.mkdir(parents=True)
            for name in ("launcher-attempt.json","attempt.json"):
                payload = b"old permanently spent "+version.encode()+name.encode()
                (old/name).write_bytes(payload)
                sentinels[old/name] = payload
            invalid = dict(BINDING,attempt_id="wm-input-compat-eval-"+version)
            with self.assertRaisesRegex(BudgetError,"attempt identity"):
                CompatBudget.create(self.repo,self.ledger,self.output,invalid,clock=self.clock)
        self.assertFalse((self.repo/"artifacts"/REGISTRY).exists())
        budget = self.create()
        self.assertEqual(budget.claim["attempt_id"],ATTEMPT_ID)
        claim = self.repo/"artifacts"/REGISTRY/"attempt.json"
        original = claim.read_bytes()
        with self.assertRaisesRegex(BudgetError,"already claimed"):
            CompatBudget.create(self.repo,self.repo/"artifacts/wm-input-compat-alternate-ledger",
                self.repo/"artifacts/wm-input-compat-alternate-results",BINDING,clock=self.clock)
        self.assertEqual(claim.read_bytes(),original)
        self.assertFalse((self.repo/"artifacts/wm-input-compat-alternate-ledger").exists())
        for path,payload in sentinels.items():
            self.assertEqual(path.read_bytes(),payload)

    def fake_complete(self, budget, key):
        budget.reserve_call(key, HASH)
        budget.complete_call(key, HASH, HASH)

    def test_full_plan_keeps_missing_branch_and_exact_phase_caps(self):
        plan = canonical_plan()
        self.assertEqual(len(plan["branches"]), 64)
        self.assertEqual(len(plan["calls"]), 2304)
        self.assertEqual(sum(row["status"] == "missing" for row in plan["calls"]), 36)
        self.assertEqual([row for row in plan["branches"] if row["status"] == "missing"],
                         [dict(anchor="L01", shape="L", candidate="hold", status="missing")])
        for phase, count in (("T", 576), ("L", 1692)):
            self.assertEqual(sum(row["phase"] == phase and row["status"] == "planned"
                                 for row in plan["calls"]), count)
        with self.assertRaises(BudgetError):
            call_key("T12", "hold", 5478, "A", "TF", 1)

    def test_fixed_claim_prevents_changed_output_binding_and_resume(self):
        self.create()
        for suffix, binding in (("new", BINDING), ("different", {**BINDING, "protocol_sha256": "b" * 64})):
            with self.assertRaises(BudgetError):
                CompatBudget.create(self.repo, self.repo / f"artifacts/wm-input-compat-{suffix}-ledger",
                                    self.repo / f"artifacts/wm-input-compat-{suffix}-output", binding)
        self.assertFalse((self.repo / "artifacts/wm-input-compat-new-output").exists())

    def test_old_or_nonempty_paths_rejected_without_touching_them(self):
        old = self.repo / "artifacts/frozen-wm-validation-v1"
        old.mkdir(parents=True)
        marker = old / "ledger.json"
        marker.write_bytes(b"untouched")
        for bad in (old, self.repo / "artifacts/milestone-b-20261023",
                    self.repo / "artifacts/wm-input-compat-fake/milestone-c-20261024"):
            with self.assertRaises(BudgetError):
                CompatBudget.create(self.repo, bad, self.output, BINDING)
        self.assertEqual(marker.read_bytes(), b"untouched")
        self.output.mkdir(parents=True)
        (self.output / "old.bin").write_bytes(b"old")
        with self.assertRaises(BudgetError):
            self.create()

    def test_m_gate_failure_blocks_model_and_allows_only_final_export(self):
        budget = self.create()
        state = budget.advance_m(False, {"fake_measurement_pass": False})
        self.assertEqual(state["stage_status"], dict(M="failed", T="not_attempted", L="not_attempted"))
        result = budget.write_chunk("summary.json", b'{"status":"M_failed"}\n', final=True)
        self.assertEqual(result["status"], "complete")
        with self.assertRaises(BudgetError):
            budget.reserve_model_load()
        self.assertEqual(budget.snapshot()["counts"]["model_loads"], 0)

    def test_reservation_is_fsynced_before_fake_sampler_and_pending_is_charged(self):
        budget = self.ready()
        key = call_key("T12", "hold", 5478, "A", "AR", 1)
        with patch("tdyssey.wm_compat_budget.os.fsync", wraps=os.fsync) as sync:
            budget.reserve_call(key, HASH)
            self.assertGreater(sync.call_count, 0)
            # This is the fake sampler boundary: the durable record must exist first.
            journal = [json.loads(line) for line in (self.ledger / "journal.jsonl").read_text().splitlines()]
            self.assertEqual(journal[-1]["kind"], "call_reserved")
            self.assertEqual(journal[-1]["data"]["charged_euler_steps"], 3)
        with self.assertRaises(BudgetError):
            budget.fail_call(key, "synthetic sampler failure")
        self.assertEqual(budget.snapshot()["counts"]["images"], 1)
        self.assertEqual(budget.snapshot()["counts"]["euler_steps"], 3)
        self.assertTrue((self.ledger / "emergency-stop.json").exists())
        with self.assertRaises(BudgetError):
            budget.reserve_call(key, HASH)

    def test_io_failure_keeps_charge_and_cannot_reopen(self):
        flag = {"fail": False}

        def fault(operation, path):
            if flag["fail"] and operation == "journal_fsync":
                flag["fail"] = False
                raise OSError("injected fsync error")

        budget = self.ready(io_hook=fault)
        flag["fail"] = True
        with self.assertRaises(BudgetError):
            budget.reserve_call(call_key("T12", "hold", 5478, "A", "AR", 1), HASH)
        saved = json.loads((self.ledger / "emergency-stop.json").read_text())
        self.assertEqual(saved["state"]["counts"]["images"], 1)
        self.assertEqual(saved["state"]["counts"]["euler_steps"], 3)
        journal_before = (self.ledger / "journal.jsonl").read_bytes()
        with self.assertRaises(BudgetError):
            budget.write_chunk("after-io-failure.json", b"{}", final=True)
        self.assertEqual((self.ledger / "journal.jsonl").read_bytes(), journal_before)
        with self.assertRaises(BudgetError):
            CompatBudget.create(self.repo, self.repo / "artifacts/wm-input-compat-new-ledger",
                                self.repo / "artifacts/wm-input-compat-new-results", BINDING)

    def test_deadline_from_create_covers_measurement_and_load(self):
        budget = self.create()
        self.clock.value = 1799
        budget.advance_m(True)
        budget.reserve_model_load()
        budget.complete_model_load()
        self.clock.value = 1800
        with self.assertRaises(BudgetError):
            budget.reserve_call(call_key("T12", "hold", 5478, "A", "AR", 1), HASH)
        self.assertEqual(budget.snapshot()["counts"]["images"], 0)
        self.assertEqual(budget.snapshot()["deadline_monotonic"], 1800)
        with self.assertRaises(BudgetError):
            budget.write_chunk("late.json", b"{}", final=True)

    def test_fsync_crossing_deadline_does_not_release_sampler_entry(self):
        flag = {"advance": False}

        def fault(operation, path):
            if flag["advance"] and operation == "journal_fsync":
                self.clock.value = 1800

        budget = self.ready(io_hook=fault)
        flag["advance"] = True
        with self.assertRaises(BudgetError):
            budget.reserve_call(call_key("T12", "hold", 5478, "A", "AR", 1), HASH)
        self.assertEqual(budget.snapshot()["counts"]["images"], 1)
        self.assertEqual(budget.snapshot()["counts"]["euler_steps"], 3)
        self.assertEqual(budget.snapshot()["status"], "stopped")

    def test_duplicate_wrong_stage_and_second_load_are_permanent_failures(self):
        for invalid in ("second_load", "L", "duplicate"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                budget = CompatBudget.create(root, root / "artifacts/wm-input-compat-ledger",
                                             root / "artifacts/wm-input-compat-results", BINDING)
                budget.advance_m(True)
                budget.reserve_model_load()
                budget.complete_model_load()
                with self.assertRaises(BudgetError):
                    if invalid == "second_load":
                        budget.reserve_model_load()
                    elif invalid == "L":
                        budget.reserve_call(call_key("L00", "hold", 5478, "A", "AR", 1), HASH)
                    else:
                        key = call_key("T12", "hold", 5478, "A", "AR", 1)
                        self.fake_complete(budget, key)
                        budget.reserve_call(key, HASH)
                self.assertEqual(budget.snapshot()["status"], "stopped")

    def test_t_gate_fail_records_all_l_not_attempted(self):
        budget = self.ready()
        with patch("tdyssey.wm_compat_budget.os.fsync"):
            for row in budget.plan["calls"]:
                if row["phase"] == "T":
                    self.fake_complete(budget, row["key"])
        state = budget.advance_t(False, {"fake_shape_screen": False})
        self.assertEqual(state["counts"]["images"], 576)
        self.assertEqual(state["counts"]["euler_steps"], 1728)
        self.assertEqual(state["stage_status"]["L"], "not_attempted")
        self.assertEqual(sum(row["phase"] == "L" and row["status"] == "not_attempted"
                             for row in state["calls"]), 1692)

    def test_full_fake_plan_finishes_at_exact_caps(self):
        budget = self.ready()
        with patch("tdyssey.wm_compat_budget.os.fsync"):
            for phase in ("T", "L"):
                for row in budget.plan["calls"]:
                    if row["phase"] == phase and row["status"] == "planned":
                        self.fake_complete(budget, row["key"])
                if phase == "T":
                    budget.advance_t(True)
            state = budget.finish()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["counts"], dict(images=2268, euler_steps=6804, model_loads=1,
                                              T_images=576, L_images=1692))
        previous = "0" * 64
        for sequence, line in enumerate((self.ledger / "journal.jsonl").read_bytes().splitlines(keepends=True), 1):
            event = json.loads(line)
            self.assertEqual(event["sequence"], sequence)
            self.assertEqual(event["previous_sha256"], previous)
            previous = hashlib.sha256(line).hexdigest()

    def test_raw_worstcase_and_temporary_bounds_and_immutable_chunks(self):
        budget = self.create()
        self.assertEqual(budget.serialization_bound, 265502720)
        payload = b"x" * IMAGE_BYTES + b"header"
        result = budget.write_chunk("raw/frame.npy", payload, raw_bytes=IMAGE_BYTES, max_bytes=110848)
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertFalse(Path(result["temporary_path"]).exists())
        with self.assertRaises(BudgetError):
            budget.write_chunk("raw/frame.npy", b"overwrite")
        self.assertEqual(Path(result["path"]).read_bytes(), payload)

    def test_actual_temp_untracked_bytes_and_external_telemetry_are_counted(self):
        guardian = self.repo / "artifacts/wm-input-compat-test-output/guardian"
        guardian.mkdir(parents=True)
        (guardian / "handshake.json").write_bytes(b"{}")
        budget = self.create(telemetry_dirs=[guardian])
        oversized = self.output / "untracked.tmp"
        with oversized.open("wb") as stream:
            stream.seek(3 * 1024 * 1024)
            stream.write(b"x")
        with self.assertRaises(BudgetError):
            budget.audit_storage()
        self.assertIn("storage", budget.snapshot()["stop_reason"])

    def test_publish_fault_leaves_bounded_temp_and_stops(self):
        def fault(operation, path):
            if operation == "chunk_publish":
                raise OSError("injected publish failure")

        budget = self.create(io_hook=fault)
        with self.assertRaises(BudgetError):
            budget.write_chunk("record.json", b"{}")
        self.assertFalse((self.output / "record.json").exists())
        self.assertEqual(len(list(self.output.glob("*.pending-*"))), 1)
        self.assertEqual(budget.snapshot()["status"], "stopped")

    def test_preflight_rejects_insufficient_worstcase_or_reduced_raw_reserve(self):
        for limits in (dict(header_bytes=4 * 1024 * 1024, temporary_bytes=2 * 1024 * 1024),
                       dict(raw_bytes=IMAGE_BYTES), dict(result_bytes=265502719)):
            with self.subTest(limits=limits), self.assertRaises(BudgetError):
                self.create(limits=limits)

    def test_binding_drift_and_noncall_stop_are_persistent(self):
        budget = self.create()
        state = budget.stop("fake measurement IO error")
        self.assertEqual(state["stop_reason"], "fake measurement IO error")
        self.assertEqual(budget.stop("replacement reason")["stop_reason"], "fake measurement IO error")
        lines = (self.ledger / "journal.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(lines[-1])["kind"], "stopped")
        self.assertEqual(json.loads(lines[-1])["data"]["reason"], "fake measurement IO error")

    def test_actual_binding_drift_stops_without_new_authority(self):
        budget = self.create()
        with self.assertRaises(BudgetError):
            budget.check(binding={**BINDING, "code_sha256": {"fake.py": "b" * 64}})
        self.assertIn("binding changed", budget.snapshot()["stop_reason"])
        self.assertEqual(budget.snapshot()["counts"]["images"], 0)

    def test_gate_stop_survives_final_export_io_failure(self):
        def fault(operation, path):
            if operation == "chunk_fsync":
                raise OSError("injected final export IO failure")

        budget = self.create(io_hook=fault)
        budget.advance_m(False, {"fake_measurement_pass": False})
        with self.assertRaises(BudgetError):
            budget.write_chunk("summary.json", b"{}", final=True)
        state = budget.snapshot()
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stop_reason"], "M_gate_failed")
        self.assertIn("injected final export IO failure", state["secondary_failure"])
        self.assertEqual(state["counts"]["images"], 0)
        emergency = json.loads((self.ledger / "emergency-stop.json").read_text())
        self.assertEqual(emergency["reason"], "M_gate_failed")
        self.assertEqual(emergency["secondary"], state["secondary_failure"])

    def test_nested_abort_preserves_first_io_and_records_latest_secondary(self):
        fired = {"value": False}

        def fault(operation, path):
            if operation == "journal_fsync" and not fired["value"]:
                event = json.loads(path.read_text().splitlines()[-1])
                if event["kind"] == "output_complete":
                    fired["value"] = True
                    raise OSError("injected initial completion IO failure")

        budget = self.create(io_hook=fault)
        with self.assertRaises(BudgetError):
            budget.write_chunk("record.json", b"{}")
        state = budget.snapshot()
        self.assertTrue(state["stop_reason"].startswith("journal_failed: OSError:"))
        self.assertIn("injected initial completion IO failure", state["stop_reason"])
        self.assertTrue(state["secondary_failure"].startswith("chunk_failed:"))
        first_path = self.ledger / "emergency-stop.json"
        first_bytes = first_path.read_bytes()
        first = json.loads(first_bytes)
        self.assertEqual(first["reason"], state["stop_reason"])
        self.assertIsNone(first["secondary"])
        secondary_paths = sorted(self.ledger.glob("emergency-secondary-*.json"))
        self.assertGreaterEqual(len(secondary_paths), 2)
        last = json.loads(secondary_paths[-1].read_text())
        self.assertEqual(last["reason"], state["stop_reason"])
        self.assertEqual(last["secondary"], state["secondary_failure"])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(state["counts"]["images"], 0)
        self.assertEqual(state["status"], "stopped")

    def test_rng_pair_mismatch_is_rejected_at_entry_or_exit(self):
        for boundary in ("entry", "exit"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                budget = CompatBudget.create(root, root / "artifacts/wm-input-compat-ledger",
                                             root / "artifacts/wm-input-compat-results", BINDING)
                budget.advance_m(True)
                budget.reserve_model_load()
                budget.complete_model_load()
                self.fake_complete(budget, call_key("T12", "hold", 5478, "A", "AR", 1))
                key = call_key("T12", "hold", 5478, "B", "AR", 1)
                with self.assertRaises(BudgetError):
                    if boundary == "entry":
                        budget.reserve_call(key, "b" * 64)
                    else:
                        budget.reserve_call(key, HASH)
                        budget.complete_call(key, "b" * 64, HASH)
                self.assertEqual(budget.snapshot()["counts"]["images"], 1 if boundary == "entry" else 2)

    def test_completed_chunk_mutation_and_guardian_growth_are_detected(self):
        guardian = self.repo / "artifacts/wm-input-compat-test-output/guardian"
        guardian.mkdir(parents=True)
        budget = self.create(telemetry_dirs=[guardian])
        record = budget.write_chunk("sealed.json", b"{}")
        path = Path(record["path"])
        path.write_bytes(b"[]")
        os.utime(path, ns=(path.stat().st_atime_ns, record["mtime_ns"] + 1000000))
        with self.assertRaises(BudgetError):
            budget.audit_storage()
        self.assertIn("immutable", budget.snapshot()["stop_reason"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guardian = root / "artifacts/wm-input-compat-new-output/guardian"
            guardian.mkdir(parents=True)
            budget = CompatBudget.create(root, root / "artifacts/wm-input-compat-new-ledger",
                                         root / "artifacts/wm-input-compat-new-output/results", BINDING,
                                         telemetry_dirs=[guardian])
            with (guardian / "too-large.log").open("wb") as stream:
                stream.seek(16 * 1024 * 1024)
                stream.write(b"x")
            with self.assertRaises(StorageError):
                budget.storage.audit()


class StorageIntegrationTests(unittest.TestCase):
    # Keep these independent from the full-plan loop; they never use a sampler.
    setUp = CompatibilityBudgetTests.setUp
    create = CompatibilityBudgetTests.create
    ready = CompatibilityBudgetTests.ready

    def test_emergency_headroom_validation_and_shared_budget_requirement(self):
        for invalid in (-1,65537,True,1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Coordinator(emergency_headroom=invalid)
        self.assertEqual(Coordinator(emergency_headroom=0).emergency_headroom,0)
        shared = Coordinator(result_roots=(self.output,),
            telemetry_roots=(self.ledger,self.repo/"artifacts/wm-input-compat-eval-v3-registry"),
            append_only=(self.ledger/"journal.jsonl",))
        with self.assertRaises(BudgetError):
            self.create(storage=shared)
        self.assertFalse(self.ledger.exists())
        self.assertFalse((self.repo/"artifacts/wm-input-compat-eval-v3-registry/attempt.json").exists())

    def test_guardian_growth_preserves_emergency_after_stale_cache_lock_failure(self):
        guardian = self.repo/"artifacts/wm-input-compat-test-output/guardian"
        guardian.mkdir(parents=True)
        lock, log = guardian/"storage.lock",guardian/"worker-output.log"
        create_lock_file(lock)
        log.write_bytes(b"")
        cap = 1024*1024
        policy = dict(lock_path=lock,result_roots=(self.output,),
            telemetry_roots=(self.ledger,self.repo/"artifacts/wm-input-compat-eval-v3-registry",guardian),
            append_only=(self.ledger/"journal.jsonl",log),
            pending_roots=(self.output,self.ledger/"exports"),pending_reserve=2*1024*1024,
            emergency_headroom=65536,timeout=.1,limits=dict(telemetry_bytes=cap))
        shared, parent = Coordinator(**policy),Coordinator(**policy)
        self.addCleanup(shared.close)
        self.addCleanup(parent.close)
        emergency_written = threading.Event()
        def hook(operation,path):
            if operation == "emergency_fsync":
                emergency_written.set()
        budget = self.create(storage=shared,telemetry_dirs=[guardian],limits=dict(telemetry_bytes=cap),io_hook=hook)
        cached = budget._telemetry_bytes()
        boundary_written, abort_reached = threading.Event(),threading.Event()
        failures, rejected = [],[]

        def guardian_write():
            try:
                with parent.local_transaction("fixture.guardian_growth"):
                    before = parent.usage()["telemetry_bytes"]
                    admitted = cap-before
                    self.assertGreater(admitted,0)
                    with self.assertRaises(StorageError):
                        parent.reserve_bytes(log,admitted+1,phase="fixture.guardian_overflow")
                    self.assertEqual(log.stat().st_size,0,"Rejected growth reached the file")
                    rejected.append(True)
                    ticket = parent.reserve_bytes(log,admitted,phase="fixture.guardian_admitted")
                    with log.open("ab") as stream:
                        parent.validate_append(log,stream)
                        stream.write(b"x"*admitted);stream.flush();os.fsync(stream.fileno())
                        parent.record_append(log,b"x"*admitted,stream)
                    parent.commit(ticket,actual_bytes=admitted)
                    self.assertEqual(parent.usage()["telemetry_bytes"],cap)
                with parent.transaction("fixture.forced_quota_stall"):
                    boundary_written.set()
                    if not abort_reached.wait(3):
                        raise AssertionError("budget did not reach the simulated lock-failure abort")
                    if not emergency_written.wait(1):
                        raise AssertionError("emergency IO waited on the failed quota lock")
            except BaseException as exc:
                failures.append(exc)
                boundary_written.set()

        original_abort = budget._abort

        def signal_abort(*args,**kwargs):
            abort_reached.set()
            return original_abort(*args,**kwargs)

        thread = threading.Thread(target=guardian_write,daemon=True)
        thread.start()
        self.assertTrue(boundary_written.wait(3))
        with patch.object(budget,"_abort",side_effect=signal_abort), self.assertRaises(BudgetError) as raised:
            budget.check()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures,[])
        self.assertEqual(rejected,[True])
        self.assertEqual(raised.exception.details["exception_type"],"TimeoutError")
        self.assertEqual(budget._last_telemetry_bytes,cached,"Regression did not exercise a stale cache")
        first_reason = budget.snapshot()["stop_reason"]
        emergency_path = self.ledger/"emergency-stop.json"
        first_bytes = emergency_path.read_bytes()
        self.assertEqual(json.loads(first_bytes)["reason"],first_reason)
        self.assertGreater(budget._emergency_bytes,0)
        self.assertLessEqual(parent.usage(full_verify=True)["telemetry_bytes"],cap)
        with self.assertRaises(BudgetError):
            budget._abort("secondary synthetic IO fault")
        self.assertEqual(budget.snapshot()["stop_reason"],first_reason)
        self.assertEqual(emergency_path.read_bytes(),first_bytes)
        self.assertLessEqual(budget._emergency_bytes,65536)
        self.assertLessEqual(parent.usage(full_verify=True)["telemetry_bytes"],cap)

    def test_pending_telemetry_is_charged_once_at_the_exact_boundary(self):
        budget = self.create()
        budget.reserve_output("bounded.json",4096,category="telemetry")
        usage = budget.storage.usage(pending_files=budget._pending_files())
        self.assertEqual(usage["telemetry_bytes"]-usage["telemetry_observed_bytes"],8192+65536)
        # The charged bytes already include the fixed emergency allowance.
        # A second charge for either pending path would fail this boundary.
        budget.limits["telemetry_bytes"] = usage["telemetry_bytes"]
        budget.check()
        budget.limits["telemetry_bytes"] -= 1
        with self.assertRaises(BudgetError):
            budget.check()
        self.assertEqual(budget.snapshot()["counts"]["images"],0)

    def test_claim_capacity_failure_happens_before_claim_open(self):
        with self.assertRaises(BudgetError) as raised:
            self.create(limits=dict(telemetry_bytes=65537))
        claim = self.repo/"artifacts/wm-input-compat-eval-v3-registry/attempt.json"
        self.assertFalse(claim.exists())
        self.assertFalse(self.ledger.exists())
        self.assertEqual(raised.exception.details["path"],str(claim))
        self.assertIn("capacity",raised.exception.details["phase"])

    def test_declaration_io_failure_keeps_the_real_path_and_phase(self):
        def fault(operation,path):
            if operation == "declaration_fsync":
                raise OSError(errno.EIO,"injected declaration fsync failure")

        with self.assertRaises(BudgetError) as raised:
            self.create(io_hook=fault)
        details = raised.exception.details
        self.assertEqual(details["path"],str(self.ledger/"declaration.json"))
        self.assertEqual(details["phase"],"budget.declaration.fsync")
        self.assertEqual(details["errno"],errno.EIO)
        self.assertEqual(details["exception_type"],"OSError")
        emergency = json.loads((self.ledger/"emergency-stop.json").read_text())
        self.assertEqual(emergency["state"]["storage_error"],details)
        claim = self.repo/"artifacts/wm-input-compat-eval-v3-registry/attempt.json"
        self.assertTrue(claim.exists(),"A failed declaration must retain the fixed claim")

    def test_publish_io_failure_keeps_the_real_destination_and_phase(self):
        def fault(operation,path):
            if operation == "chunk_publish_fsync":
                raise OSError(errno.EIO,"injected publication fsync failure")

        budget = self.create(io_hook=fault)
        with self.assertRaises(BudgetError) as raised:
            budget.write_chunk("record.json",b"{}")
        details = raised.exception.details
        self.assertEqual(details["path"],str(self.output/"record.json"))
        self.assertEqual(details["phase"],"budget.chunk_publish.fsync")
        self.assertEqual(details["errno"],errno.EIO)
        emergency = json.loads((self.ledger/"emergency-stop.json").read_text())
        self.assertEqual(emergency["state"]["storage_error"],details)
        self.assertEqual(budget.snapshot()["counts"]["images"],0)

    def test_finish_full_verifies_parent_totals_before_complete_at_cap_boundary(self):
        for overflow in (None,"output_bytes","telemetry_bytes"):
            with self.subTest(overflow=overflow), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                budget = CompatBudget.create(repo,repo/"artifacts/wm-input-compat-ledger",
                    repo/"artifacts/wm-input-compat-results",BINDING,clock=Clock())
                # Populate terminal fake accounting directly, without issuing
                # calls, loading a model or creating any image payload.
                budget._state.update(phase="L",counts=dict(images=2268,euler_steps=6804,
                    model_loads=1,T_images=576,L_images=1692))
                budget._calls = {row["key"]:dict(status="complete") for row in budget.plan["calls"]
                                 if row["status"]=="planned"}
                original_usage, original_audit, verified_states = budget.storage.usage,budget.audit_storage,[]

                def audit():
                    result = original_audit()
                    verified_states.append(budget._state["status"])
                    return result

                def usage(**kwargs):
                    result = original_usage(**kwargs)
                    if verified_states and budget._state["status"]=="active":
                        for field in ("output_bytes","telemetry_bytes"):
                            result[field] = budget.storage.limits[field]+(field==overflow)
                    return result

                with patch.object(budget.storage,"usage",side_effect=usage), \
                        patch.object(budget,"audit_storage",side_effect=audit):
                    if overflow is None:
                        self.assertEqual(budget.finish()["status"],"complete")
                    else:
                        with self.assertRaises(BudgetError) as raised:
                            budget.finish()
                        self.assertEqual(budget.snapshot()["status"],"stopped")
                        self.assertIn(overflow,budget.snapshot()["stop_reason"])
                        self.assertEqual(raised.exception.details["phase"],"budget.finish.full_verify.capacity")
                        kinds=[json.loads(line)["kind"] for line in (budget.ledger_dir/"journal.jsonl").read_text().splitlines()]
                        self.assertNotIn("finished",kinds)
                self.assertEqual(verified_states,["active"])

    def test_completed_same_size_restored_mtime_mutation_is_detected(self):
        budget = self.create()
        record = budget.write_chunk("sealed.json",b"{}")
        path = Path(record["path"])
        original = path.stat()
        path.write_bytes(b"[]")
        os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
        self.assertEqual(path.stat().st_size,record["bytes"])
        self.assertEqual(path.stat().st_mtime_ns,record["mtime_ns"])
        with self.assertRaises(BudgetError):
            budget.audit_storage()
        state = budget.snapshot()
        self.assertEqual(state["status"],"stopped")
        self.assertEqual(state["counts"]["images"],0)
        emergency = json.loads((self.ledger/"emergency-stop.json").read_text())
        self.assertEqual(emergency["state"]["storage_error"]["path"],str(path))

    def test_missing_immutable_retains_structured_first_error(self):
        budget = self.create()
        record = budget.write_chunk("sealed.json",b"{}")
        path = Path(record["path"])
        path.unlink()
        with self.assertRaises(BudgetError):
            budget.audit_storage()
        details = budget.snapshot()["storage_error"]
        self.assertEqual(details["path"],str(path))
        self.assertTrue(set(("path","phase","exception_type","errno","message")).issubset(details))
        first = (self.ledger/"emergency-stop.json").read_bytes()
        self.assertEqual(json.loads(first)["state"]["storage_error"],details)
        with self.assertRaises(BudgetError):
            budget.reserve_call(call_key("T12","hold",5478,"A","AR",1),HASH)
        self.assertEqual((self.ledger/"emergency-stop.json").read_bytes(),first)
        self.assertEqual(budget.snapshot()["counts"]["images"],0)

    def test_permission_error_is_not_skipped_and_emergency_is_bounded(self):
        budget = self.create()
        path = Path(budget.write_chunk("sealed.json",b"{}")["path"])
        original_open = os.open

        def deny(selected,*args,**kwargs):
            if Path(selected) == path:
                raise PermissionError(errno.EACCES,"injected unreadable immutable file",str(selected))
            return original_open(selected,*args,**kwargs)

        with patch("tdyssey.wm_compat_storage.os.open",side_effect=deny), self.assertRaises(BudgetError):
            budget.audit_storage()
        details = budget.snapshot()["storage_error"]
        self.assertEqual(details["path"],str(path))
        self.assertEqual(details["exception_type"],"PermissionError")
        self.assertEqual(details["errno"],errno.EACCES)
        self.assertEqual(json.loads((self.ledger/"emergency-stop.json").read_text())["state"]["storage_error"],details)
        self.assertLessEqual(budget._emergency_bytes,65536)
        self.assertEqual(budget.snapshot()["counts"]["images"],0)

    def test_append_only_journal_prefix_rewrite_is_fatal(self):
        budget = self.create()
        journal = self.ledger/"journal.jsonl"
        original = journal.read_bytes()
        rewritten = original.replace(b'"kind":"created"',b'"kind":"creatfd"',1)
        self.assertNotEqual(original,rewritten)
        journal.write_bytes(rewritten)
        with self.assertRaises(BudgetError):
            budget.audit_storage()
        self.assertEqual(budget.snapshot()["storage_error"]["path"],str(journal))
        self.assertEqual(budget.snapshot()["counts"]["images"],0)

    def test_failed_emergency_fsync_still_consumes_bounded_headroom(self):
        def fault(operation,path):
            if operation == "emergency_fsync":
                raise OSError("injected emergency fsync failure")

        budget = self.create(io_hook=fault)
        with self.assertRaises(BudgetError):
            budget._abort("first IO fault")
        first_reserved = budget._emergency_bytes
        self.assertGreater(first_reserved,0)
        with self.assertRaises(BudgetError):
            budget._abort("secondary IO fault")
        files = list(self.ledger.glob("emergency-*.json"))
        self.assertGreater(budget._emergency_bytes,first_reserved)
        self.assertGreaterEqual(budget._emergency_bytes,sum(path.stat().st_size for path in files))
        self.assertLessEqual(budget._emergency_bytes,65536)
        self.assertEqual(budget.snapshot()["stop_reason"],"first IO fault")

    def test_heartbeat_progresses_during_reservation_journal_io(self):
        # This Windows sandbox denies rename in its system Temp directory;
        # keep this disposable fake repo inside the writable checkout instead.
        evidence = Path(__file__).resolve().parents[1]/"artifacts/wm-input-compat-eval-v3-audit-20261025"
        evidence.mkdir(parents=True,exist_ok=True)
        workspace_temp = tempfile.TemporaryDirectory(prefix="wm-compat-fixture-",dir=evidence)
        self.addCleanup(workspace_temp.cleanup)
        self.repo = Path(workspace_temp.name)
        self.ledger = self.repo/"artifacts/wm-input-compat-test-ledger"
        self.output = self.repo/"artifacts/wm-input-compat-test-output/results"
        guardian = self.repo/"artifacts/wm-input-compat-test-output/guardian"/("b"*32)
        guardian.mkdir(parents=True)
        status = guardian/"worker-status.json"
        shared = Coordinator(result_roots=(self.output,),
            telemetry_roots=(self.ledger,self.repo/"artifacts/wm-input-compat-eval-v3-registry",guardian),
            atomic_targets=(status,),append_only=(self.ledger/"journal.jsonl",),
            pending_roots=(self.output,self.ledger/"exports"),pending_reserve=2*1024*1024,emergency_headroom=65536)
        begin, attempted, published = (threading.Event() for _ in range(3))
        failures = []
        armed = {"value":False}

        def hook(operation,path):
            if armed["value"] and operation == "journal_append":
                armed["value"] = False
                self.assertFalse(shared.accounting_lock_held)
                begin.set()
                self.assertTrue(attempted.wait(1))
                self.assertTrue(published.wait(1),"guardian blocked behind budget journal IO")

        budget = self.ready(storage=shared,telemetry_dirs=[guardian],io_hook=hook)

        def publish():
            try:
                if not begin.wait(2):
                    raise AssertionError("reservation barrier was not reached")
                attempted.set()
                with shared.local_transaction("fixture.atomic_publish"):
                    self.assertFalse(shared.accounting_lock_held)
                    temporary = status.with_name(status.name+"."+"a"*32+".tmp")
                    with temporary.open("xb") as stream:
                        stream.write(b'{"fixture":true}\n');stream.flush();os.fsync(stream.fileno())
                    os.replace(temporary,status)
                    published.set()
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=publish,daemon=True)
        thread.start()
        armed["value"] = True
        budget.reserve_call(call_key("T12","hold",5478,"A","AR",1),HASH)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures,[])
        self.assertTrue(published.is_set())
        budget.check()
        self.assertEqual(budget.snapshot()["counts"]["images"],1)
        self.assertEqual(budget.snapshot()["counts"]["euler_steps"],3)
        self.assertTrue(status.exists())

    def test_whole_chunk_publication_blocks_a_concurrent_snapshot(self):
        begin, attempted, checked = (threading.Event() for _ in range(3))
        failures = []

        def hook(operation,path):
            if operation == "chunk_publish":
                begin.set()
                self.assertTrue(attempted.wait(1))
                self.assertFalse(checked.wait(.02),"snapshot escaped full publish transaction")

        budget = self.create(io_hook=hook)

        def check():
            try:
                if not begin.wait(2):
                    raise AssertionError("publication barrier was not reached")
                attempted.set()
                budget.check()
                checked.set()
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=check,daemon=True)
        thread.start()
        record = budget.write_chunk("record.json",b"{}")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures,[])
        self.assertTrue(checked.is_set())
        self.assertEqual(Path(record["path"]).read_bytes(),b"{}")
        self.assertFalse(Path(record["temporary_path"]).exists())
        self.assertEqual(budget.snapshot()["outputs"][record["path"]]["status"],"complete")

    def test_hot_checks_and_sampler_accounting_never_scan_or_rehash_old_files(self):
        budget = self.ready()
        key = call_key("T12","hold",5478,"A","AR",1)
        with patch("tdyssey.wm_compat_budget.scan_files",side_effect=AssertionError("hot directory scan")), \
                patch("tdyssey.wm_compat_storage.fingerprint",side_effect=AssertionError("hot file rehash")):
            budget.check()
            budget.reserve_call(key,HASH)
            budget.complete_call(key,HASH,HASH)
            budget.check()
        self.assertEqual(budget.snapshot()["counts"]["images"],1)
        self.assertEqual(budget.snapshot()["counts"]["euler_steps"],3)

    def test_hot_check_does_not_traverse_histories_but_explicit_snapshot_is_complete(self):
        budget = self.ready()
        key = call_key("T12","hold",5478,"A","AR",1)
        budget.reserve_call(key,HASH)
        budget.complete_call(key,HASH,HASH)
        record = budget.write_chunk("bounded.json",b"{}",max_bytes=4096)

        class UnreadHistory:
            def __getitem__(self,key):
                raise AssertionError("Hot check read historical state")
            def __iter__(self):
                raise AssertionError("Hot check traversed historical state")

        with patch.object(budget,"plan",UnreadHistory()), \
                patch.object(budget,"_calls",UnreadHistory()), \
                patch.object(budget,"_outputs",UnreadHistory()), \
                patch.object(budget,"snapshot",side_effect=AssertionError("Hot check built a full snapshot")):
            checked = budget.check()
        self.assertEqual(checked["counts"]["images"],1)
        self.assertEqual(checked["counts"]["euler_steps"],3)
        self.assertEqual(checked["phase"],"T")
        self.assertEqual(checked["status"],"active")
        self.assertNotIn("calls",checked)
        self.assertNotIn("outputs",checked)
        checked["counts"]["images"] = 99
        full = budget.snapshot()
        self.assertEqual(full["counts"]["images"],1)
        self.assertEqual(len(full["calls"]),2304)
        self.assertEqual(next(row for row in full["calls"] if row["key"]==key)["status"],"complete")
        self.assertEqual(full["outputs"][record["path"]]["status"],"complete")
        full["outputs"][record["path"]]["status"] = "changed"
        self.assertEqual(budget.snapshot()["outputs"][record["path"]]["status"],"complete")

    def test_chunk_target_is_fingerprinted_once_and_unused_maximum_stays_charged(self):
        budget = self.create()
        held=[]
        def hook(operation,path):
            held.append(budget.storage.accounting_lock_held)
        budget.io_hook = hook
        before = budget.storage.usage()["output_bytes"]
        with patch.object(budget.storage,"remember_file",wraps=budget.storage.remember_file) as remember, \
                patch("tdyssey.wm_compat_budget.scan_files",side_effect=AssertionError("publication tree scan")):
            record = budget.write_chunk("bounded.json",b"{}",max_bytes=4096)
        self.assertEqual(remember.call_count,1)
        self.assertEqual(remember.call_args.args,(Path(record["path"]),))
        self.assertTrue(held)
        self.assertFalse(any(held),"Quota mutex retained over file IO/fsync")
        usage = budget.storage.usage()
        self.assertEqual(usage["output_bytes"]-before,4096)
        self.assertEqual(usage["pending_result_publications"],0)

    def test_compact_output_events_keep_bounds_integrity_and_full_detached_snapshots(self):
        budget = self.create()
        records = [budget.write_chunk("nested/result.json",b"{}",max_bytes=128),
                   budget.write_chunk("nested/metadata.json",b"{}",max_bytes=128,category="telemetry")]
        snapshot = budget.snapshot()
        previous = "0"*64
        outputs = []
        for sequence,line in enumerate((self.ledger/"journal.jsonl").read_bytes().splitlines(keepends=True),1):
            event = json.loads(line)
            self.assertEqual(event["sequence"],sequence)
            self.assertEqual(event["previous_sha256"],previous)
            previous = hashlib.sha256(line).hexdigest()
            if event["kind"] not in ("output_reserved","output_complete"):
                continue
            outputs.append(event)
            row = event["data"]
            self.assertNotIn("path",row)
            self.assertNotIn("temporary_path",row)
            self.assertFalse(Path(row["name"]).is_absolute())
            self.assertNotIn(json.dumps(str(self.repo))[1:-1],json.dumps(row))
            complete = next(record for record in records if record["name"]==row["name"])
            self.assertEqual(row["max_bytes"],complete["max_bytes"])
            self.assertEqual(row["temp_bytes"],complete["temp_bytes"])
            self.assertEqual(row["raw_bytes"],complete["raw_bytes"])
            self.assertEqual(row["category"],complete["category"])
            if event["kind"]=="output_complete":
                self.assertEqual(row,{key:value for key,value in complete.items()
                                      if key not in ("path","temporary_path")})
            else:
                self.assertEqual(row["status"],"pending")
        self.assertEqual(len(outputs),4)
        self.assertEqual(previous,budget._previous)
        self.assertEqual(len(snapshot["calls"]),2304)
        for record in records:
            stored = snapshot["outputs"][record["path"]]
            self.assertEqual(stored,record)
            self.assertTrue(Path(stored["path"]).is_absolute())
            self.assertTrue(Path(stored["temporary_path"]).is_absolute())
            stored["path"] = "changed"
            stored["identity"][0] = -1
            self.assertEqual(budget.snapshot()["outputs"][record["path"]],record)


if __name__ == "__main__":
    unittest.main()
