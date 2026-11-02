"""Accounting is meaningful even when a real physics step raises/interruption occurs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from scripts.native_rgb_supervised_stage1 import StageLedger
from scripts.native_rgb_supervised_stage1 import main


class BudgetTests(unittest.TestCase):
    def test_failed_physical_audit_returns_failure_exit_status(self):
        with patch("scripts.native_rgb_supervised_stage1.read_supervision"), redirect_stdout(io.StringIO()):
            for passed, expected in ((False,1),(True,0)):
                with patch("scripts.native_rgb_supervised_stage1.audit_probe",return_value={"passed":passed,"issues":[]}):
                    self.assertEqual(main(["audit","--physics-root","unused"]),expected)

    def test_attempt_is_charged_before_execution_and_interruption_never_refunded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "artifacts") as d:
            ledger=StageLedger(d,cap=2);i=ledger.begin_trajectory("T-center",2)
            ledger.charge(i)
            disk=json.loads(ledger.path.read_text())
            self.assertEqual((disk["charged_controls"],disk["completed_controls"]),(1,0))
            reopened=StageLedger(d,cap=2)
            with self.assertRaisesRegex(RuntimeError,"unresolved"):
                reopened.charge(i)
            reopened.complete(i);reopened.charge(i)
            reopened.finish(i,status="physics_failed",reason="mock partial step")
            self.assertEqual(reopened.value["charged_controls"],2)
            self.assertEqual(reopened.value["completed_controls"],1)
            self.assertEqual(reopened.value["physics_substeps"],10)
            with self.assertRaises(RuntimeError):reopened.charge(i)

    def test_caps_and_terminal_callbacks_are_not_reusable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "artifacts") as d:
            ledger=StageLedger(d,cap=1);i=ledger.begin_trajectory("L-left",1)
            ledger.charge(i);ledger.complete(i)
            with self.assertRaises(RuntimeError):ledger.charge(i)
            ledger.finish(i)
            with self.assertRaises(RuntimeError):ledger.finish(i)
            with self.assertRaises(ValueError):StageLedger(d,cap=2)
            ledger.reject_reset("bad-scene","illegal exterior staging")
            self.assertEqual(ledger.value["charged_controls"],1)


if __name__=="__main__":unittest.main()
