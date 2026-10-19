from dataclasses import replace
import json
import unittest

from tdyssey.geometry import Pose, get_shape
from tdyssey.state import BodyState, Snapshot
from tdyssey.trajectories import Trajectory, Transition, split_trajectories


def snapshot(step=0):
    return Snapshot("I", get_shape("I").geometry_hash, "a" * 64,
                    BodyState(Pose((200, 220), 0.3), (2, 3), 0.4, (5, 6), 0.7),
                    BodyState(Pose((100, 120), 0), (8, 9)), Pose((256, 256), 0), step, (110, 130))


def trajectory(identifier="sample", terminated=False):
    before, after = snapshot(), snapshot(1)
    return Trajectory(identifier, "contact_play", "v1", 0, 0.1,
                      (Transition(before, (110, 130), after, 0.2, terminated),))


class StateAndDataChecks(unittest.TestCase):
    def test_trajectory_json_round_trip_preserves_provenance_and_flags(self):
        original = trajectory(terminated=True)
        self.assertEqual(Trajectory.from_json(original.to_json()), original)
        record = json.loads(original.to_json())
        record["schema_version"] = 2
        with self.assertRaises(ValueError):
            Trajectory.from_json(json.dumps(record))

    def test_snapshot_json_preserves_velocities_forces_and_target(self):
        original = snapshot(3)
        restored = Snapshot.from_json(original.to_json())
        self.assertEqual(original, restored)
        self.assertEqual(restored.object_body.velocity, (2, 3))
        self.assertEqual(restored.object_body.angular_velocity, 0.4)
        self.assertEqual(restored.object_body.force, (5, 6))
        self.assertEqual(restored.pusher_target, (110, 130))

    def test_invalid_schema_hash_or_numeric_state_is_rejected(self):
        for field, value in [("schema_version", 2), ("schema_version", True),
                             ("geometry_hash", "missing"), ("control_step", -1),
                             ("control_step", True)]:
            with self.assertRaises(ValueError):
                replace(snapshot(), **{field: value})
        record = json.loads(snapshot().to_json())
        record["object_body"]["velocity"][0] = float("nan")
        with self.assertRaises(ValueError):
            Snapshot.from_dict(record)

    def test_transition_rejects_action_and_step_mismatches(self):
        with self.assertRaises(ValueError):
            Transition(snapshot(), (9, 9), snapshot(1), 0)
        with self.assertRaises(ValueError):
            Transition(snapshot(), (110, 130), snapshot(8), 0)
        with self.assertRaises(ValueError):
            Transition(snapshot(), (110, 130), replace(snapshot(1), physics_hash="b" * 64), 0)

    def test_trajectory_stops_on_either_terminal_flag(self):
        first = trajectory().transitions[0]
        second = Transition(first.after, (110, 130), snapshot(2), 0.3)
        for flag in ("terminated", "truncated"):
            with self.assertRaises(ValueError):
                replace(trajectory(), transitions=(replace(first, **{flag: True}), second))

    def test_discontinuous_states_and_wrong_action_units_are_rejected(self):
        first = trajectory().transitions[0]
        second = Transition(replace(first.after, object_body=BodyState(Pose((1, 2), 0))),
                            (110, 130), snapshot(2), 0.3)
        with self.assertRaises(ValueError):
            replace(trajectory(), transitions=(first, second))
        with self.assertRaises(ValueError):
            replace(trajectory(), action_semantics="normalized_displacement")

    def test_splits_are_order_independent_complete_and_disjoint(self):
        records = [trajectory(str(i)) for i in range(20)]
        one, two = split_trajectories(records, 4), split_trajectories(reversed(records), 4)
        self.assertEqual(one, two)
        ids = [{r.trajectory_id for r in part} for part in one.values()]
        self.assertEqual([len(part) for part in ids], [16, 2, 2])
        self.assertEqual(len(set.union(*ids)), 20)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertNotEqual(one, split_trajectories(records, 5))

    def test_duplicate_trajectory_ids_and_invalid_fractions_are_rejected(self):
        with self.assertRaises(ValueError):
            split_trajectories([trajectory(), trajectory()])
        with self.assertRaises(ValueError):
            split_trajectories([trajectory()], fractions=(0.8, 0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
