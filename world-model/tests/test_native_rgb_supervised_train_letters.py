"""Stage2A entry expansion: pure geometry and pinned Stage1B source guards."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from tests.test_native_rgb_supervised_sampling import GeometricEnvironment
from tdyssey.native_rgb_data import contour_segments
from tdyssey.native_rgb_supervised_sampling import (LETTERS,LOCATIONS,ALLOWED_TRAINING_LETTERS,
    SIDE_SIGN_LETTERS,SIDE_SIGN_INDEX,plan_scenes,plan_scene,initialize_scene,ContinuousContactSampler)

BASELINE_SHA256="57fecfc81b7e6ef025b1154102e7c7a6e7ab370aa495ba14796052c3b42ec900"


def digest(source):return hashlib.sha256(source.encode()).hexdigest()


class FrozenStage1BSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(Path(__file__).resolve().parents[1]/"tdyssey/native_rgb_supervised_sampling.py").read_text(encoding="utf8")
        cls.tree=ast.parse(cls.source)
        cls.node=next(n for n in cls.tree.body if isinstance(n,ast.ClassDef) and n.name=="ContinuousContactSampler")
        cls.methods={n.name:n for n in cls.node.body if isinstance(n,ast.FunctionDef)}

    def test_all_recovery_methods_except_init_remain_byte_identical(self):
        self.assertEqual(list(self.methods),["__init__","_normal","_retreat","_recontact","_recovery_snapshot",
            "_begin_recovery","_end_recovery","_recovery_waypoint","_recovery_action","_observe_recovery","next_action","observe"])
        source="\n".join(ast.get_source_segment(self.source,node) for name,node in self.methods.items() if name!="__init__")
        self.assertEqual(digest(source),"42b75f4477fe208f4dbd227ceed2fdcd4712fbc37ce435d37ac1beda2e54523a")

    def test_init_and_scene_only_change_the_two_approved_entry_expressions(self):
        init=ast.get_source_segment(self.source,self.methods["__init__"])
        restored=init.replace("SIDE_SIGN_INDEX[self.letter]","LETTERS.index(self.letter)")
        self.assertEqual(digest(restored),"89dbd6392f6a28f1e0e428079e63c54176cdbbe3811f7e57094a9f7327e9c869")
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=="plan_scene")
        restored=ast.get_source_segment(self.source,node).replace("letter not in ALLOWED_TRAINING_LETTERS","letter not in LETTERS")
        self.assertEqual(digest(restored),"38247d2b295da8637aeb0e0f2d266391f8e6f50f6e225778840b829493de6516")

    def test_all_other_functions_and_old_constants_are_unchanged(self):
        source="\n".join(ast.get_source_segment(self.source,node) for node in self.tree.body
            if isinstance(node,ast.FunctionDef) and node.name!="plan_scene")
        self.assertEqual(digest(source),"4798c827f2ed92330cb854821fca0c1f94b7512c47fb72142579d3dd03a7379f")
        extra={"ALLOWED_TRAINING_LETTERS","SIDE_SIGN_LETTERS","SIDE_SIGN_INDEX"}
        source="\n".join(ast.get_source_segment(self.source,node) for node in self.tree.body
            if isinstance(node,ast.Assign) and not any(isinstance(t,ast.Name) and t.id in extra for t in node.targets))
        self.assertEqual(digest(source),"ceee3b339ea3553838dc0279ac24bfd295f44fd7f5bee2363fe2decbfcf2faee")


class TrainingLetterEntryTests(unittest.TestCase):
    def test_fixed_18_mapping_preserves_all_six_original_side_indices(self):
        self.assertEqual(LETTERS,tuple("TLOCXB"))
        self.assertEqual(ALLOWED_TRAINING_LETTERS,tuple("ABCEFHIJLMOPSTUVXY"))
        self.assertEqual(SIDE_SIGN_LETTERS,tuple("TLOCXBAEFHIJMPSUVY"))
        self.assertEqual(SIDE_SIGN_INDEX,{letter:index for index,letter in enumerate(SIDE_SIGN_LETTERS)})
        for index,letter in enumerate(LETTERS):
            self.assertEqual(SIDE_SIGN_INDEX[letter],index)
            for location_index in range(9):
                self.assertEqual((location_index+SIDE_SIGN_INDEX[letter])%2,(location_index+index)%2)

    def test_held_out_shapes_rejected_and_historical_probe_entry_stays_six_letters(self):
        for letter in "DGKNQRWZ":
            with self.assertRaises(ValueError):plan_scene(letter,"center",1)
        self.assertEqual(len(plan_scenes({})),54)
        with self.assertRaises(ValueError):plan_scenes(dict(letters=list(ALLOWED_TRAINING_LETTERS)))

    def test_all_162_initializations_are_pure_legal_json_and_preserve_controller_timeline(self):
        from shapely.geometry import Point
        gap_bands=((0,8),(8,32),(32,64));gaps=(4,18,48);count=0
        for location_index,location in enumerate(LOCATIONS):
            for letter_index,letter in enumerate(ALLOWED_TRAINING_LETTERS):
                tier=(location_index-1+letter_index)%3
                request=dict(letter=letter,spatial_class=location,seed=2027102901+location_index*1000+letter_index,
                    gap_band=None if location=="center" else list(gap_bands[tier]),
                    gap_world_px=None if location=="center" else gaps[tier])
                env=GeometricEnvironment(letter);scene=initialize_scene(env,request)
                self.assertTrue(scene["legal"],(letter,location,scene["issues"]))
                self.assertEqual(env.control_step,0);self.assertEqual(env.reset_calls,1)
                for wall in location.split("-"):
                    if wall!="center":self.assertAlmostEqual(scene["actual_wall_gaps"][wall],gaps[tier],places=6)
                segments=contour_segments(letter)
                self.assertEqual(segments[scene["staging_segment"]]["ring"],0)
                self.assertGreaterEqual(env.world_geometry().distance(Point(*env.state()[:2])),15.5-1e-6)
                policy=ContinuousContactSampler(env,segments,{},scene);before=env.state()
                _,metadata=policy.next_action(0)
                np.testing.assert_array_equal(env.state(),before)
                self.assertEqual(env.control_step,0)
                self.assertEqual(policy.side_sign,1 if (location_index+SIDE_SIGN_INDEX[letter])%2 else -1)
                json.dumps(scene,allow_nan=False);json.dumps(metadata,allow_nan=False);count+=1
        self.assertEqual(count,162)


if __name__=="__main__":unittest.main()
