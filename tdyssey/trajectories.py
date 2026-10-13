"""Control-step transition records and deterministic whole-trajectory splits."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .geometry import Point, finite, point
from .state import Snapshot


@dataclass(frozen=True)
class Transition:
    before: Snapshot
    action: Point
    after: Snapshot
    reward: float
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self):
        object.__setattr__(self, "action", point(self.action))
        object.__setattr__(self, "reward", finite(self.reward, "reward"))
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("Termination flags must be boolean")
        if self.after.control_step != self.before.control_step + 1:
            raise ValueError("A transition must represent exactly one control step")
        if self.after.pusher_target != self.action:
            raise ValueError("Recorded action differs from the executed pusher target")
        for name in ("shape", "geometry_hash", "physics_hash", "goal"):
            if getattr(self.before, name) != getattr(self.after, name):
                raise ValueError(f"Transition changes the episode's {name}")


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    collector: str
    collector_version: str
    seed: int
    control_dt: float
    transitions: tuple[Transition, ...]
    action_semantics: str = "world_frame_position_target_pixels"

    def __post_init__(self):
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.trajectory_id, self.collector, self.collector_version)):
            raise ValueError("Trajectory ID and collector provenance are required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Collector seed must be a nonnegative integer")
        object.__setattr__(self, "control_dt", finite(self.control_dt, "control_dt"))
        if self.control_dt <= 0 or not self.transitions:
            raise ValueError("Trajectory needs a positive time interval and transitions")
        if self.action_semantics != "world_frame_position_target_pixels":
            raise ValueError("Convert other action units explicitly before recording")
        for previous, current in zip(self.transitions, self.transitions[1:]):
            if previous.terminated or previous.truncated:
                raise ValueError("Trajectory continues after its episode ended")
            if previous.after != current.before:
                raise ValueError("Trajectory contains a discontinuity")

    def to_json(self) -> str:
        return json.dumps({"schema_version": 1, **asdict(self)}, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, message: str):
        value = json.loads(message)
        version = value.pop("schema_version")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError("Unsupported trajectory schema version")
        transitions = []
        for record in value.pop("transitions"):
            record = dict(record)
            record["before"] = Snapshot.from_dict(record["before"])
            record["after"] = Snapshot.from_dict(record["after"])
            transitions.append(Transition(**record))
        return cls(transitions=tuple(transitions), **value)


def split_trajectories(trajectories, seed: int = 0, fractions=(0.8, 0.1, 0.1)):
    """Return train/validation/model-test records without window-level leakage.

    Membership is invariant to input ordering. Freeze/export a split before adding
    data: changing the collection size can change membership. Policy test states
    and held-out shapes need separate manifests.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    fractions = tuple(finite(value, "fraction") for value in fractions)
    if len(fractions) != 3 or any(value < 0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ValueError("Three nonnegative fractions must sum to one")
    records = list(trajectories)
    if len({record.trajectory_id for record in records}) != len(records):
        raise ValueError("Duplicate trajectory IDs would compromise split provenance")
    records.sort(key=lambda record: hashlib.sha256(
        f"{seed}:{record.trajectory_id}".encode()).digest())
    exact = [len(records) * fraction for fraction in fractions]
    counts = [math.floor(value) for value in exact]
    for index in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[:len(records) - sum(counts)]:
        counts[index] += 1
    result, offset = {}, 0
    for name, count in zip(("train", "validation", "model_test"), counts):
        result[name] = tuple(records[offset:offset + count])
        offset += count
    return result
