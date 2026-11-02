"""Two R2A2 constant-damping candidates using the frozen R2A observer.

Only Space.damping changes, after the sole initial reset and before any control.
The inherited native step, force limits, speed cap and collision callbacks are
unchanged. No simulator is imported or constructed merely by importing this file.
"""
from __future__ import annotations

import hashlib
import json

from .native_rgb_r2a_env import R2AEnv, TELEMETRY_VERSION, make_r2a_env
from .native_rgb_env import PROTOCOL_SHA256

ARMS = {
    "D-half-life-0.1s": 0.0009765625,
    "D-half-life-0.3s": 0.09921256574801246,
}
HALF_LIFE_SECONDS = {"D-half-life-0.1s": .1, "D-half-life-0.3s": .3}
PROTOCOL_VERSION = "finite-force-pd-v1-r2a2-constant-damping-candidate-v1"


def candidate_protocol(arm):
    if arm not in ARMS:
        raise ValueError("Unknown R2A2 constant-damping candidate")
    return dict(version=PROTOCOL_VERSION, candidate_name=arm,
        base_protocol_sha256=PROTOCOL_SHA256,
        sole_physics_delta=dict(parameter="space.damping", original=0.0, candidate=ARMS[arm]),
        damping_rule="native constant Space.damping, velocity retention D**elapsed_seconds",
        velocity_half_life_seconds=HALF_LIFE_SECONDS[arm])


class R2A2Env(R2AEnv):
    """Initial-only damping override; every control uses the inherited observer."""
    def __init__(self, base, letter, arm, segments, rings):
        candidate_protocol(arm)  # Validate before any initialization attempt.
        self.candidate_arm = arm
        super().__init__(base, letter, "P0", segments, rings)

    def reset_scene(self, *args, **kwargs):
        rgb = super().reset_scene(*args, **kwargs)
        try:
            if self._base.control_step != 0 or self.last_substeps:
                raise RuntimeError("R2A2 damping must be set before any control")
            self._base.space.damping = ARMS[self.candidate_arm]
            if float(self._base.space.damping) != ARMS[self.candidate_arm]:
                raise ValueError("Actual Space.damping differs from the exact candidate constant")
            self.arm = self.candidate_arm
            # Re-read actual body/shape/space identity after the sole change.
            identity = super()._identity()
            protocol = candidate_protocol(self.candidate_arm)
            identity.update(kind="tdyssey.r2a2-initial-identity.v1",
                protocol_version=PROTOCOL_VERSION,
                protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                base_protocol_sha256=PROTOCOL_SHA256, candidate_protocol=protocol,
                inherited_telemetry_version=TELEMETRY_VERSION)
            self._initial_identity = identity
            return rgb
        except BaseException:
            self._terminal = True
            self.__dict__.pop("_initial_identity", None)
            raise


def make_r2a2_env(source_root, letter, arm):
    """Construct a candidate only for an explicitly authorized collector call."""
    candidate_protocol(arm)
    original = make_r2a_env(source_root, letter, "P0")
    return R2A2Env(original._base, letter, arm, original._segments, original._rings)
