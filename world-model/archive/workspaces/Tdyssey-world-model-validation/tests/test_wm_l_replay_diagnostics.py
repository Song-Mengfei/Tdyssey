"""Saved-array reconstruction and metadata regression; no learned runtime."""

import builtins
import copy
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey.wm_compat_engine import SavedBranch, action_window, arm_context, array_hash, compact_measurement
from tdyssey.wm_l_replay_diagnostics import replay_branch_inputs


class ReplayDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.branch = SavedBranch("L00", "hold", "L",
            np.stack([np.full((3,96,96), n/10, dtype=np.float32) for n in (1,2,3,4)]),
            np.array([[30,40],[50,60],[70,80]], dtype=float),
            np.array([[90,100],[110,120],[130,140],[150,160]], dtype=float),
            np.stack([np.full((3,96,96), n/10, dtype=np.float32) for n in (5,6,7,8)]))
        self.raw, self.rows = {}, []
        for arm in ("A", "B"):
            history = arm_context(self.branch.history,arm)
            generated = []
            for horizon in (1,2,3,4):
                key=f"L00/hold/5478/AR/{arm}/H{horizon}"
                output=np.full((3,96,96), .4+horizon/100, dtype=np.float32)
                output[0,0,0],output[1,0,0]=-0.2,1.2
                self.raw[key]=output
                context=np.stack((list(history)+generated)[horizon-1:horizon+3])
                self.rows.append(self.row(key,context,output,history,horizon))
                generated.append(output)
            for horizon, context in ((2,np.concatenate((self.branch.history[1:],self.branch.actual[:1]))),
                                      (4,np.concatenate((self.branch.history[-1:],self.branch.actual[:3])))):
                key=f"L00/hold/5478/TF/{arm}/H{horizon}"
                output=np.full((3,96,96),.6,dtype=np.float32);self.raw[key]=output
                self.rows.append(self.row(key,arm_context(context,arm),output,context,horizon))

    def row(self,key,context,raw,source,horizon):
        return dict(key=key,status="completed",input_sha256=array_hash(context),raw_sha256=array_hash(raw),
            action_sha256=array_hash(action_window(self.branch,horizon)),
            source_context_sha256=array_hash(self.branch.history if "/AR/" in key else source),
            source_history_sha256=array_hash(self.branch.history))

    def replay(self, rows=None, reader=None):
        return replay_branch_inputs(self.branch,5478,self.rows if rows is None else rows,
                                   reader or (lambda row:self.raw[row["key"]]))

    def test_preserves_raw_out_of_range_feedback_and_exact_TF_boundaries(self):
        before={key:value.copy() for key,value in self.raw.items()}
        checks=self.replay()
        self.assertEqual(len(checks),12)
        self.assertTrue(all(row["passed"] for row in checks))
        self.assertEqual(sum(row["raw_feedback_outside_unit_range"] for row in checks),8)
        for key in before:np.testing.assert_array_equal(before[key],self.raw[key])
        clipped=self.replay(reader=lambda row:np.clip(self.raw[row["key"]],0,1))
        self.assertFalse(all(row["passed"] for row in clipped))

    def test_actual_input_hash_is_checked_instead_of_original_history_hash(self):
        rows=copy.deepcopy(self.rows)
        target=next(row for row in rows if row["key"]=="L00/hold/5478/AR/B/H3")
        target["input_sha256"]=target["source_context_sha256"]
        result=next(row for row in self.replay(rows) if row["key"]==target["key"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["input_sha256"])
        self.assertTrue(result["checks"]["source_context_sha256"])

    def test_missing_AR_predecessor_blocks_later_AR_without_using_truth(self):
        rows=[row for row in self.rows if row["key"]!="L00/hold/5478/AR/B/H2"]
        results={row["key"]:row for row in self.replay(rows)}
        self.assertEqual(results["L00/hold/5478/AR/B/H3"]["status"],"blocked_by_missing_AR_predecessor")
        self.assertTrue(results["L00/hold/5478/TF/B/H4"]["passed"])
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            self.replay(self.rows+[self.rows[0]])

    def test_wrong_actions_are_detected_and_runtime_imports_forbidden(self):
        rows=copy.deepcopy(self.rows);rows[0]["action_sha256"]="0"*64
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split(".")[0] in {"torch","pymunk","pygame","diffusion_policy"}:
                raise AssertionError("Forbidden runtime import")
            return original(name,*args,**kwargs)
        with patch("builtins.__import__",guarded):results=self.replay(rows)
        self.assertFalse(results[0]["checks"]["action_sha256"])

    def test_compact_measurement_keeps_actual_pusher_fit_field(self):
        full=dict(pusher=dict(valid=True,position=[10.,20.],fit_iou=.93,area_ratio=1.0))
        compact=compact_measurement(full)
        self.assertEqual(compact["pusher"]["fit_iou"],.93)
        self.assertEqual(compact["pusher"]["position"],[10.,20.])
        self.assertEqual(compact_measurement(dict(pusher=dict(valid=False,reason="missing")))["pusher"],
                         dict(valid=False,reason="missing"))


if __name__=="__main__":unittest.main()
