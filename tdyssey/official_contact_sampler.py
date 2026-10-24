"""Privileged official-T contact/miss play. Not a policy and not reward search.

Hit headings use official T rectangles and the pusher disk. The fallback aims
along pusher to mass center with a truncated normal offset. Collection writes
many official batch directories; each batch keeps the 1000-control pilot cap.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

from .collection import integer
from .envs.official import T_REFERENCE_COG, official_shape
from .geometry import finite, point
from .motion_replay import object_moved

HEADING_COUNT = 32
PUSHER_RADIUS = 15.0
RAY_LENGTH = 500.0
FALLBACK_SPREAD = math.radians(150)
TARGET_USEFUL = 0.6
BASE_CONTACT_PROBABILITY = 0.7
HISTORY = 16
WORKSPACE = (25.0, 486.0)
DEFAULT_BATCHES = 40
DEFAULT_SEED = 240000
ACTION_PRIOR = (
    "contact-balanced uses privileged official T rectangles, mass center and "
    "orientation to choose contact headings, with a truncated-normal fallback "
    "around the pusher-to-CoG axis and adaptive miss actions; no reward optimization"
)
ANGLES_CHECKPOINT = Path("artifacts/official-wm-T-angles-2026-10-23/model/denoiser.pt")
INIT_DIRECTORY = Path("artifacts/official-wm-T-init")


def object_cog(pose, shape_name="T"):
    """World mass center. Official T is offset from the body origin; L is not."""
    if shape_name == "T":
        return pose.transform(T_REFERENCE_COG)
    return pose.position


def _local_point(world, pose):
    dx = world[0] - pose.position[0]
    dy = world[1] - pose.position[1]
    cosine, sine = math.cos(pose.angle), math.sin(pose.angle)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _local_direction(heading, pose):
    cosine, sine = math.cos(pose.angle), math.sin(pose.angle)
    world = (math.cos(heading), math.sin(heading))
    return cosine * world[0] + sine * world[1], -sine * world[0] + cosine * world[1]


def _ray_aabb(origin, direction, x0, y0, x1, y1, reach=RAY_LENGTH):
    enter, leave = 0.0, reach
    for value, step, low, high in ((origin[0], direction[0], x0, x1),
                                   (origin[1], direction[1], y0, y1)):
        if abs(step) < 1e-12:
            if value < low or value > high:
                return None
            continue
        first = (low - value) / step
        second = (high - value) / step
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
        if enter > leave:
            return None
    return enter, leave


def classify_headings(pusher, pose, shape, radius=PUSHER_RADIUS, count=HEADING_COUNT):
    """Label equally spaced headings as hit or miss against the inflated T."""
    if count < 8:
        raise ValueError("Heading count must be at least 8")
    local = _local_point(pusher, pose)
    rows = []
    for index in range(count):
        heading = 2 * math.pi * index / count
        direction = _local_direction(heading, pose)
        hits = []
        for rect in shape.components:
            interval = _ray_aabb(local, direction, rect.x0 - radius, rect.y0 - radius,
                                 rect.x1 + radius, rect.y1 + radius)
            if interval is not None:
                hits.append(interval)
        if hits:
            rows.append({"heading": heading, "hit": True,
                         "t_enter": min(item[0] for item in hits),
                         "t_exit": max(item[1] for item in hits)})
        else:
            rows.append({"heading": heading, "hit": False, "t_enter": None, "t_exit": None})
    return rows


def fallback_offset(rng, spread=FALLBACK_SPREAD):
    """Truncated normal offset in radians, inside plus/minus the spread."""
    limit = finite(spread, "fallback spread")
    if limit <= 0:
        raise ValueError("fallback spread must be positive")
    return min(limit, max(-limit, rng.gauss(0.0, limit / 2)))


def fallback_heading(pusher, pose, rng, shape_name="T", spread=FALLBACK_SPREAD):
    """Heading along pusher to CoG plus a truncated normal offset."""
    center = object_cog(pose, shape_name)
    axis = math.atan2(center[1] - pusher[1], center[0] - pusher[0])
    return axis + fallback_offset(rng, spread)


def _clip_workspace(value):
    return min(WORKSPACE[1], max(WORKSPACE[0], value))


def _point_along(pusher, heading, distance):
    return (_clip_workspace(pusher[0] + distance * math.cos(heading)),
            _clip_workspace(pusher[1] + distance * math.sin(heading)))


def contact_target(pusher, pose, headings, rng, shape_name="T"):
    """Pick a geometrically hitting heading, or the CoG-axis fallback."""
    hits = [row for row in headings if row["hit"]]
    if not hits:
        heading = fallback_heading(pusher, pose, rng, shape_name)
        return _point_along(pusher, heading, rng.choice((40.0, 80.0, 150.0))), heading
    choice = rng.choice(hits)
    heading = choice["heading"]
    center = object_cog(pose, shape_name)
    through = max(20.0, math.hypot(center[0] - pusher[0], center[1] - pusher[1]))
    distances = (max(15.0, choice["t_enter"]), through, choice["t_exit"] + 20.0)
    return _point_along(pusher, heading, rng.choice(distances)), heading


def miss_target(pusher, pose, headings, rng, shape_name="T"):
    """Pick a heading that does not hit the T, or a point away from the CoG."""
    misses = [row for row in headings if not row["hit"]]
    if misses:
        heading = rng.choice(misses)["heading"]
        return _point_along(pusher, heading, rng.uniform(40.0, 180.0)), heading
    center = object_cog(pose, shape_name)
    heading = math.atan2(pusher[1] - center[1], pusher[0] - center[0])
    return _point_along(pusher, heading, rng.uniform(80.0, 180.0)), heading


class ContactBalancedPlay:
    """Adaptive contact-seeking and miss play from privileged official geometry."""

    def __init__(self, config, seed, target_useful=TARGET_USEFUL):
        self.config = config
        self.rng = random.Random(seed)
        self.target_useful = finite(target_useful, "target useful rate")
        if not 0 < self.target_useful < 1:
            raise ValueError("target useful rate must be in (0, 1)")
        self.contact_prob = BASE_CONTACT_PROBABILITY
        self.recent = []
        self.shape = official_shape(config.shape)

    def observe(self, before, after, n_contacts=0):
        """Update the contact/miss mix from the step that just ran in the simulator."""
        useful = object_moved(before, after) or integer(n_contacts, "contacts", 0, 100000) > 0
        self.recent.append(bool(useful))
        self.recent = self.recent[-HISTORY:]
        rate = sum(self.recent) / len(self.recent)
        if rate > self.target_useful + 0.05:
            self.contact_prob = max(0.35, self.contact_prob - 0.08)
        elif rate < self.target_useful - 0.05:
            self.contact_prob = min(0.90, self.contact_prob + 0.08)

    def action(self, observation, state, step):
        pusher = point(observation["agent_pos"])
        headings = classify_headings(pusher, state.object_body.pose, self.shape)
        if self.rng.random() < self.contact_prob:
            target, _heading = contact_target(pusher, state.object_body.pose, headings,
                                              self.rng, state.shape)
        else:
            target, _heading = miss_target(pusher, state.object_body.pose, headings,
                                           self.rng, state.shape)
        dx, dy = target[0] - pusher[0], target[1] - pusher[1]
        scale = min(1.0, self.config.max_target_delta / max(math.hypot(dx, dy), 1e-9))
        return (_clip_workspace(pusher[0] + scale * dx), _clip_workspace(pusher[1] + scale * dy))


def preserve_init_checkpoint(source=None, destination=None, root=None):
    """Copy the current best denoiser once. Later runs must not overwrite it."""
    base = Path(root) if root is not None else Path.cwd()
    source_file = Path(source) if source is not None else base / ANGLES_CHECKPOINT
    dest_dir = Path(destination) if destination is not None else base / INIT_DIRECTORY
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("denoiser.pt", "denoiser.json"):
        src = source_file.parent / name
        dest = dest_dir / name
        if dest.exists():
            continue
        if not src.is_file():
            if name == "denoiser.pt":
                raise FileNotFoundError(f"Best denoiser is missing: {src}")
            continue
        shutil.copy2(src, dest)
        copied.append(name)
    weights = dest_dir / "denoiser.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Init denoiser is missing: {weights}")
    return {"checkpoint": weights, "copied": copied}


def collect_contact_batches(output, source_root, provenance, batches=DEFAULT_BATCHES,
                            seed=DEFAULT_SEED, workers=2, seconds=300, episodes=8, steps=100,
                            resume=False):
    """Write many official T batches. Each directory stays inside the 1000-control cap."""
    from .official_collection import OfficialCollectionConfig
    from .official_collector import OfficialShapeCollector

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    integer(batches, "batches", 1, 20000)
    summaries = []
    skipped = 0
    for index in range(batches):
        root = output / f"batch-{index + 1:03d}"
        if root.exists():
            if not resume:
                raise FileExistsError(root)
            summaries.append(_finished_summary(root))
            skipped += 1
            continue
        if skipped:
            print(json.dumps({"resumed": True, "skipped": skipped, "next": root.name,
                              "target_batches": batches}), flush=True)
            skipped = 0
        config = OfficialCollectionConfig(
            shape="T", episodes=episodes, steps=steps, seed=seed + index * 100,
            mode="contact-balanced", max_target_delta=200, seconds=seconds)
        note = provenance(config)
        note["action_prior"] = ACTION_PRIOR
        manifest = OfficialShapeCollector(config, source_root, workers=workers).collect(root, note)
        row = {"batch": root.name, "controls": manifest["counts"]["control_steps"],
               "contact_steps": sum(entry["contact_steps"] for entry in manifest["episodes"]),
               "completed_batches": len(summaries) + 1, "target_batches": batches}
        summaries.append(row)
        print(json.dumps(row), flush=True)
    return {"status": "complete", "batches": summaries,
            "controls": sum(row["controls"] for row in summaries),
            "contact_steps": sum(row["contact_steps"] for row in summaries)}


def _finished_summary(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise FileExistsError(f"Incomplete batch cannot be resumed: {root}")
    return {"batch": Path(root).name, "controls": manifest["counts"]["control_steps"],
            "contact_steps": sum(entry["contact_steps"] for entry in manifest["episodes"])}
