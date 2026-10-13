"""Serializable physical/controller state at control-step boundaries.

These records do not serialize Chipmunk's contact solver cache. Restoring them
starts a fresh space; matched contact replay still requires backend validation.
"""

from dataclasses import asdict, dataclass
import json
import re

from .geometry import Point, Pose, finite, point


@dataclass(frozen=True)
class BodyState:
    pose: Pose
    velocity: Point = (0.0, 0.0)
    angular_velocity: float = 0.0
    force: Point = (0.0, 0.0)
    torque: float = 0.0

    def __post_init__(self):
        if not isinstance(self.pose, Pose):
            raise ValueError("Body pose must be a Pose record")
        object.__setattr__(self, "velocity", point(self.velocity))
        object.__setattr__(self, "force", point(self.force))
        object.__setattr__(self, "angular_velocity", finite(self.angular_velocity))
        object.__setattr__(self, "torque", finite(self.torque))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["pose"] = Pose(**value["pose"])
        return cls(**value)


@dataclass(frozen=True)
class Snapshot:
    shape: str
    geometry_hash: str
    physics_hash: str
    object_body: BodyState
    pusher_body: BodyState
    goal: Pose
    control_step: int
    pusher_target: Point | None = None
    schema_version: int = 1

    def __post_init__(self):
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != 1:
            raise ValueError("Unsupported snapshot schema version")
        for name in ("geometry_hash", "physics_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not isinstance(self.shape, str) or not self.shape.strip() or isinstance(self.control_step, bool) or not isinstance(self.control_step, int) or self.control_step < 0:
            raise ValueError("Snapshot needs a shape and a nonnegative integer control step")
        if not isinstance(self.object_body, BodyState) or not isinstance(self.pusher_body, BodyState) or not isinstance(self.goal, Pose):
            raise ValueError("Snapshot requires body and goal records")
        if self.pusher_target is not None:
            object.__setattr__(self, "pusher_target", point(self.pusher_target))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name in ("object_body", "pusher_body"):
            value[name] = BodyState.from_dict(value[name])
        value["goal"] = Pose(**value["goal"])
        return cls(**value)

    @classmethod
    def from_json(cls, value: str):
        return cls.from_dict(json.loads(value))
