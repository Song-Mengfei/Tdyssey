"""The explicit selected default for future native RGB data generation.

Historical data/rendering keep their recorded identities. This pointer delegates
to the existing protocol without changing its name, hash or implementation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .native_rgb_r2a2_env import candidate_protocol, make_r2a2_env

SELECTED_ARM = "D-half-life-0.1s"
SELECTED_DAMPING = .0009765625
SELECTED_PROTOCOL_SHA256 = "58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6"
DISPLAY_NAME = "0.1秒半衰期（已选）"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/"configs/native_rgb_selected_physics.json"
DEFAULT_POINTER = DEFAULT_CONFIG.with_name("native_rgb_default.json")


def load_selected_config():
    """Read and verify the selected pointer; this is not a choice/search API."""
    pointer = json.loads(DEFAULT_POINTER.read_text(encoding="utf-8"))
    if (pointer.get("kind") != "tdyssey.native-rgb-default.v1"
            or pointer.get("selected_physics_config") != "configs/native_rgb_selected_physics.json"
            or pointer.get("generation_factory") != "tdyssey.native_rgb_selected_physics.make_selected_env"
            or pointer.get("default_for_all_new_data_generation") is not True
            or pointer.get("new_generation_scopes") != ["collection", "train", "validation", "test"]):
        raise ValueError("Canonical new-generation default no longer points to selected physics")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    protocol = candidate_protocol(SELECTED_ARM)
    actual_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (config.get("kind") != "tdyssey.selected-native-physics.v1"
            or config.get("candidate_arm") != SELECTED_ARM
            or config.get("space_damping") != SELECTED_DAMPING
            or config.get("protocol_sha256") != SELECTED_PROTOCOL_SHA256
            or actual_sha != SELECTED_PROTOCOL_SHA256
            or config.get("protocol_version") != protocol["version"]
            or config.get("base_protocol_sha256") != protocol["base_protocol_sha256"]
            or config.get("display_name") != DISPLAY_NAME
            or config.get("default_for_all_new_data_generation") is not True
            or config.get("new_generation_scopes") != pointer["new_generation_scopes"]
            or config.get("factory") != "tdyssey.native_rgb_selected_physics.make_selected_env"):
        raise ValueError("Selected physics pointer differs from the authorized existing protocol")
    return config


def selected_metadata():
    """External display/provenance, separate from the unchanged protocol hash."""
    config = load_selected_config()
    return dict(display_name=DISPLAY_NAME, canonical_default="configs/native_rgb_default.json",
        canonical_default_sha256=hashlib.sha256(DEFAULT_POINTER.read_bytes()).hexdigest(),
        selected_physics_config="configs/native_rgb_selected_physics.json",
        selected_physics_config_sha256=hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest(),
        protocol_version=config["protocol_version"], protocol_sha256=SELECTED_PROTOCOL_SHA256,
        candidate_arm=SELECTED_ARM, space_damping=SELECTED_DAMPING)


def verify_selected_identity(identity):
    """Check a recorded actual initial identity, without reading any simulator."""
    load_selected_config()
    if (identity.get("arm") != SELECTED_ARM or identity.get("protocol_sha256") != SELECTED_PROTOCOL_SHA256
            or identity.get("physics", {}).get("space", {}).get("damping") != SELECTED_DAMPING):
        raise ValueError("Recorded physics is not the selected existing protocol")
    return True


def make_selected_env(source_root, letter):
    """Future collection entry point; no extra reset, control or physics change."""
    load_selected_config()
    return make_r2a2_env(source_root, letter, SELECTED_ARM)
