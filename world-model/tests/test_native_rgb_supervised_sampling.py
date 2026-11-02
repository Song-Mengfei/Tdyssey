"""Pure geometry/mock tests: this file executes zero real physics controls."""

import json
import math
import unittest

from tdyssey.native_rgb_data import contour_segments
from tdyssey.native_rgb_env import shape_geometry,CONTROLLER,PROTOCOL_SHA256
from tdyssey.native_rgb_supervised_sampling import (LETTERS,LOCATIONS,plan_scenes,plan_scene,
    initialize_scene,ContinuousContactSampler,_area_center,_departure_route,_free_route)


class GeometricEnvironment:
    """Stores arrays only. No Pymunk space, body, or step implementation exists."""
    controller=CONTROLLER
    def __init__(self,letter):
        import numpy as np
        self.letter=letter;self.value=np.zeros(15);self.control_step=0;self.reset_calls=0
        self.local_geometry=shape_geometry(letter)
        self.block=type("PoseView",(),{})()
    def reset_scene(self,center,angle,pusher):
        import numpy as np
        from tdyssey.native_rgb_supervised_sampling import _rotation
        self.reset_calls+=1
        if self.reset_calls>1:raise AssertionError("A continuous trajectory cannot reset mid-rollout")
        cog=np.asarray((0,45) if self.letter=="T" else (0,0))
        self.value[:2]=pusher;self.value[2:4]=np.asarray(center)-_rotation(angle)@cog;self.value[4]=angle
        self.block.angle=angle
        self.block.local_to_world=lambda point:tuple(self.value[2:4]+_rotation(self.value[4])@np.asarray(point))
    def state(self):return self.value.copy()
    def world_geometry(self):
        from shapely import affinity
        return affinity.translate(affinity.rotate(self.local_geometry,float(self.value[4]),origin=(0,0),use_radians=True),*self.value[2:4])
    def penetrations(self):
        import numpy as np
        from shapely.geometry import Point
        p=Point(*self.value[:2]);g=self.world_geometry();xmin,ymin,xmax,ymax=g.bounds
        overlap=15+g.boundary.distance(p) if g.contains(p) else max(0,15-g.distance(p))
        return np.asarray((max(0,7-xmin,7-ymin,xmax-504,ymax-504),overlap,
            max(0,22-p.x,22-p.y,p.x-489,p.y-489)))
    def mock_transition(self,translation=(0,0)):
        import numpy as np
        # Deliberate synthetic test state change, never a real simulator step.
        self.value[:2]+=np.asarray(translation);self.value[2:4]+=np.asarray(translation);self.control_step+=1


class SupervisedSamplingTests(unittest.TestCase):
    def test_fixed_six_by_nine_requests_balance_boundary_gap_bands(self):
        from collections import Counter
        requests=plan_scenes({})
        self.assertEqual(len(requests),54)
        self.assertEqual(len({r["seed"] for r in requests}),54)
        self.assertEqual(Counter(r["letter"] for r in requests),{letter:9 for letter in LETTERS})
        self.assertEqual(Counter(r["spatial_class"] for r in requests),{location:6 for location in LOCATIONS})
        self.assertEqual(Counter(tuple(r["gap_band"]) for r in requests if r["gap_band"]),{(0,8):16,(8,32):16,(32,64):16})
        json.dumps(requests,allow_nan=False)

    def test_all_initial_plans_have_actual_gaps_and_exterior_contact_staging_without_steps(self):
        from shapely.geometry import Point
        for request in plan_scenes({}):
            with self.subTest(letter=request["letter"],location=request["spatial_class"]):
                env=GeometricEnvironment(request["letter"])
                scene=initialize_scene(env,request,coverage_counts={},historical_counts={})
                self.assertTrue(scene["legal"],scene["issues"])
                self.assertEqual(env.control_step,0);self.assertEqual(env.reset_calls,1)
                self.assertEqual(scene["physics_protocol_sha256"],PROTOCOL_SHA256)
                self.assertEqual(contour_segments(request["letter"])[scene["staging_segment"]]["ring"],0)
                self.assertGreaterEqual(env.world_geometry().distance(Point(*env.state()[:2])),15.5-1e-6)
                if request["spatial_class"]=="center":self.assertGreater(scene["initial_min_wall_gap_world_px"],64)
                for wall in request["spatial_class"].split("-"):
                    if wall!="center":self.assertAlmostEqual(scene["actual_wall_gaps"][wall],request["gap_world_px"],places=6)
                json.dumps(scene,allow_nan=False)

    def test_deterministic_scene_priorities_do_not_conflate_historical_and_new_hits(self):
        first=plan_scene("T","center",11,coverage_counts={},historical_counts={})
        repeated=plan_scene("T","center",11,coverage_counts={},historical_counts={})
        self.assertEqual(first,repeated)
        selected=first["staging_segment"]
        other=plan_scene("T","center",11,coverage_counts={selected:999},historical_counts={})
        self.assertNotEqual(selected,other["staging_segment"])

    def test_t_area_centroid_uses_area_centroid_not_original_cog_or_origin(self):
        import numpy as np
        state=np.zeros(15);state[2:4]=(200,200);state[4]=math.pi/2
        np.testing.assert_allclose(_area_center("T",state),(200-40.714285714285715,200),atol=1e-10)

    def test_64_mock_transitions_preserve_timeline_and_do_not_mutate_environment_in_policy(self):
        import numpy as np
        request=plan_scenes({})[0];env=GeometricEnvironment("T");scene=initialize_scene(env,request)
        segments=contour_segments("T");counts={row["id"]:0 for row in segments}
        policy=ContinuousContactSampler(env,segments,counts,scene);phases=[];distances=[]
        movement=-np.asarray(scene["staging"]["normal"])*2.1
        for step in range(64):
            before=env.state();action,metadata=policy.next_action(step)
            np.testing.assert_array_equal(env.state(),before)
            self.assertNotIn("step",metadata);json.dumps(metadata,allow_nan=False)
            self.assertTrue(((action>=0)&(action<=511)).all())
            env.mock_transition(movement);after=env.state()
            info=dict(contact=1,contact_points=np.asarray([segments[scene["staging_segment"]]["point"]]),valid=True)
            observed=policy.observe(before,action,after,info)
            np.testing.assert_array_equal(env.state(),after)
            self.assertTrue(observed["moving"]);phases.append(metadata["phase"]);distances.append(metadata["actual_target_distance_world_px"])
        self.assertEqual(env.reset_calls,1);self.assertEqual(env.control_step,64)
        self.assertTrue(set(("approach","push","sliding","retreat","recontact","push_again")).issubset(phases))
        self.assertGreater(max(distances),96)
        self.assertEqual(counts,{row["id"]:0 for row in segments})

    def test_blocked_mock_contact_is_reported_and_does_not_trigger_hidden_reset(self):
        import numpy as np
        env=GeometricEnvironment("O");scene=initialize_scene(env,plan_scenes({})[2]);segments=contour_segments("O")
        policy=ContinuousContactSampler(env,segments,{},scene);phases=[]
        for step in range(20):
            before=env.state();action,metadata=policy.next_action(step);env.mock_transition()
            observed=policy.observe(before,action,env.state(),dict(contact=1,contact_points=np.asarray([segments[scene["staging_segment"]]["point"]]),valid=True))
            self.assertTrue(observed["near_static"]);phases.append(metadata["phase"])
        self.assertIn("blocked_escape_retreat",phases)
        self.assertEqual(env.reset_calls,1);self.assertLessEqual(policy.escapes,2)

    def test_same_letter_has_both_planned_oblique_and_sliding_sides_across_locations(self):
        requests=plan_scenes({})
        for letter in LETTERS:
            signs=set()
            for request in [r for r in requests if r["letter"]==letter][:2]:
                env=GeometricEnvironment(letter);scene=initialize_scene(env,request)
                policy=ContinuousContactSampler(env,contour_segments(letter),{},scene)
                signs.add(policy.side_sign)
            self.assertEqual(signs,{-1,1})

    def test_contact_normal_tracks_a_real_majority_point_not_a_fictitious_mean(self):
        import numpy as np
        env=GeometricEnvironment("T");scene=initialize_scene(env,plan_scenes({})[0])
        segments=contour_segments("T");policy=ContinuousContactSampler(env,segments,{},scene)
        outside=[i for i,row in enumerate(segments) if not row["ring"]];first,second=outside[0],outside[7]
        policy.active=second
        before=env.state();action,_=policy.next_action(0);env.mock_transition()
        points=np.asarray([segments[first]["point"],segments[first]["point"],segments[second]["point"]])
        policy.observe(before,action,env.state(),dict(contact=3,contact_points=points,valid=True))
        self.assertEqual(policy.active,first)

    def test_retreat_and_recontact_do_not_take_thin_wall_shortcuts(self):
        from shapely.geometry import box
        from shapely.ops import unary_union
        geometry=unary_union((box(240,200,250,240),box(251,200,252,240)))
        self.assertFalse(_free_route(geometry,(268,220),(234,220)))
        self.assertFalse(_departure_route(geometry,(268,220),(234,220)))
        self.assertTrue(_departure_route(box(240,200,250,240),(264.5,220),(300,220)))


if __name__=="__main__":unittest.main()
