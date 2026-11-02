"""Constructed arrays/labels only; no saved research RGB or learned model."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Offline figure diagnostics require installed NumPy/SciPy")
import numpy as np
from tdyssey.wm_l_diagnostic_figures import select_cases, display_rgb, color_line_residual, frame_diagnostics, _draw_pages


def branch(anchor,candidate,contact=False,rotation=0):
    return dict(anchor=anchor,candidate=candidate,shape="L",status="complete",latest={"pose":{"angle":0}},
                actual=[dict(contacts=int(contact),truth={"pose":{"angle":rotation if i==3 else 0}}) for i in range(4)])


def row(anchor,candidate,mode,valid,seed=5478):
    return dict(anchor=anchor,candidate=candidate,shape="L",arm="B",mode=mode,seed=seed,horizon=4,status="completed",
                measurement={"state":{"valid":valid},"shape":{"valid":valid}})


class OfflineFigureChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None,"Matplotlib is required for figure rendering")
    def test_rendered_H1_valid_and_H2_rejected_titles_keep_their_own_metadata(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict("os.environ",{"MPLCONFIGDIR":str(Path(directory)/"cache")}):
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            captures=[];zero=np.zeros((96,96));raw=np.ones((3,96,96),np.float32)
            def entry(valid,iou):
                return dict(raw=raw,truth={"pose":None,"pusher":None},
                    measurement={"state":{"valid":valid},"shape":{"valid":valid,"reason":"ok" if valid else "rigid_shape_mismatch","full_soft_iou":iou,"area_ratio":1.1},"pusher":{"valid":True}},
                    diagnostic={"alpha":zero,"blue":zero,"fit_residual":zero,"color_residual":zero,
                                "summary":{"raw_min":-.012,"raw_max":1.,"display_clipped_channel_fraction":.01,"offline_frozen_pusher":{"valid":True}}})
            def save(figure,path,**kwargs):
                captures.append((figure.axes[11].get_title(),figure.axes[12].get_title()))
                Path(path).write_bytes(b"synthetic figure stub")
            branch=dict(initial_state={"goal":{"position":[240,240],"angle":0}},actual=[{"action":[200,200]}]*4)
            with patch.object(Figure,"savefig",save):
                _draw_pages({("B AR",1):entry(True,.916919),("B AR",2):entry(False,.7284)},branch,Path(directory),"synthetic")
            self.assertEqual(len(captures),4)
            for h1,h2 in captures:
                self.assertIn("B AR H1 | S=OK P=OK\nok\nIoU=0.917",h1)
                self.assertIn("B AR H2 | S=REJECT P=OK\nrigid_shape_mismatch\nIoU=0.728",h2)

    def test_fixed_selection_preserves_contact_rotation_and_controls(self):
        branches=[branch("L00","toward",True,.2),branch("L00","hold"),branch("L01","side",True),branch("L02","hold")]
        rows=[]
        for anchor,candidate,ar,tf in (("L00","toward",False,True),("L00","hold",True,True),
                                      ("L01","side",False,False),("L02","hold",True,True)):
            rows.extend((row(anchor,candidate,"AR",ar),row(anchor,candidate,"TF",tf)))
        expected=select_cases(rows,branches)
        self.assertEqual([(r["anchor"],r["candidate"]) for r in expected],[("L00","hold"),("L00","toward"),("L01","side"),("L02","hold")])
        self.assertIn("rotation_gt5_h4_rejected",expected[1]["categories"])
        self.assertIn("AR_rejected_TF_recovered",expected[1]["categories"])
        self.assertIn("AR_TF_both_rejected",expected[2]["categories"])
        self.assertIn("matched_hold_control",expected[0]["categories"])
        rows.append(row("L00","toward","AR",True,5480))
        self.assertEqual(select_cases(list(reversed(rows)),list(reversed(branches))),expected)

    def test_display_is_fixed_and_leaves_raw_range_untouched(self):
        raw=np.zeros((3,96,96),np.float32);raw[:,0,0]=[-.1,.5,1.2];old=raw.copy()
        self.assertEqual(display_rgb(raw)[0,0].tolist(),[0,128,255])
        np.testing.assert_array_equal(raw,old)
        raw[0,0,0]=np.nan
        with self.assertRaises(ValueError):display_rgb(raw)

    def test_color_residual_is_independent_of_geometry(self):
        rgb=np.asarray([[[119,136,153],[255,255,255],[255,0,255]]],float)
        result=color_line_residual(rgb)
        self.assertEqual(result.shape,(1,3))
        self.assertLess(result[0,0],1e-12);self.assertLess(result[0,1],1e-12)
        self.assertGreater(result[0,2],.1)

    def test_rejected_fit_stays_rejected_and_pusher_diagnostics_survive(self):
        raw=np.ones((3,96,96),np.float32);yy,xx=np.mgrid[:96,:96]
        disk=(xx-20)**2+(yy-30)**2<=9
        raw[:,disk]=np.asarray([78,126,255],np.float32)[:,None]/255
        original=dict(state={"valid":False,"pose":None,"candidate_pose":{"position":[200,200],"angle":0}},
                      shape={"valid":False,"reason":"rigid_shape_mismatch","palette":"red","fitted_outline":2.,"fitted_blur_sigma":.25},
                      pusher={"valid":False,"position":None})
        before=deepcopy(original);result=frame_diagnostics(raw,original)
        self.assertEqual(original,before)
        self.assertIsNotNone(result["template"])
        self.assertFalse(result["summary"]["original_state_valid"])
        pusher=result["summary"]["offline_frozen_pusher"]
        self.assertIn("fit_iou",pusher);self.assertIn("candidate_position",pusher);self.assertIn("meaningful_components",pusher)
        self.assertIsInstance(result["summary"]["original_pusher_agreement"],bool)


if __name__=="__main__":unittest.main()
