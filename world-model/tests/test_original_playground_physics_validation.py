"""No simulation: regression of initial-state copy and impulse accounting."""
from pathlib import Path
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_validation',ROOT/'scripts/validate_original_playground_physics.py')
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)

class Body:
    def __init__(self):
        self.position=(0.,0.);self._angle=0.;self.velocity=(0.,0.);self.angular_velocity=0.;self.force=(0.,0.);self.torque=0.
    @property
    def angle(self):return self._angle
    @angle.setter
    def angle(self,value):
        # Capture the nonzero-CoG assignment-order issue found before controls.
        self.position=(self.position[0]+value-self._angle,self.position[1]);self._angle=value

class OriginalValidationTests(unittest.TestCase):
    def test_initial_copy_angle_before_position_preserves_full_state(self):
        source=Body();source.angle=.4;source.position=(123.,257.);source.velocity=(2.,3.);source.angular_velocity=.7;source.force=(.2,.3);source.torque=.6
        target=Body();validation.copy_initial_body(source,target)
        for key in('angle','position','velocity','angular_velocity','force','torque'):self.assertEqual(getattr(source,key),getattr(target,key))
    def test_contacts_and_positive_impulses_are_per_substep_not_point_sum(self):
        rows=[dict(contacts=[dict(kind='agent_object',phase='pre_solve'),dict(kind='agent_object',phase='post_solve',total_impulse=[3,4],points=[1,2]),dict(kind='agent_object',phase='post_solve',total_impulse=[3,4],points=[1,2]),dict(kind='object_wall',phase='post_solve',total_impulse=[0,0])]),dict(contacts=[])]
        contacts,positive=validation.contact_counts(rows)
        self.assertEqual(contacts,{'agent_object':1,'object_wall':1,'agent_wall':0})
        self.assertEqual(positive,{'agent_object':1,'object_wall':0,'agent_wall':0})

if __name__=='__main__':unittest.main()
