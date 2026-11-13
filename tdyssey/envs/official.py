"""T-preserving official PushT extension with rigid capital-letter bodies."""

from dataclasses import replace
import hashlib
import json

from ..collection import integer, reset_issues
from ..geometry import Pose, Rectangle, Shape, point
from ..state import BodyState, Snapshot
from ..letters import NOVEL_PATTERNS, PATTERNS, block_letter

PROTOCOL = "official-t-l-com-v1"
T_REFERENCE_COG = (0.0, 45.0)

# Goal overlay (D066, off by default; docs/GOAL_OVERLAY.md). The part of the goal the block does not cover is painted
# red on the 512 px canvas after the goal and before the block and pusher. Parts thinner than R = `goal_overlay`
# simulator px (removed by a morphological opening of radius R/2; pieces at least R long) are grown by R, so a sliver
# far below one camera pixel (512/96 = 5.33 simulator px; the block's drawn outline hides another ~2 px) still reaches
# the 96 px image; wider parts keep their true outline, so the goal's letter shape stays readable far from the goal.
# Computed from the same collision polygons as the coverage metric: privileged simulator geometry, a rendering aid.
# Changing this definition changes what an overlay checkpoint was trained on: give it a new name, not the same flag.
OVERLAY_COLOR = (255, 0, 0)
OVERLAY_MIN_AREA = 1.0        # simulator px^2; smaller uncovered pieces (e.g. floating-point slivers) are not drawn
OVERLAY_HAIRLINE = 0.1        # simulator px; thin parts narrower than this (numerical; <= ~0.3% coverage) are not grown
OVERLAY_QUAD_SEGS = 4         # round joins: 4 segments per quarter circle (<= 0.17 px from the true arc at R = 9)


def _polygons(geometry, min_area):
    return [p for p in getattr(geometry, "geoms", [geometry]) if p.geom_type == "Polygon" and p.area >= min_area]


def _length(polygon):
    """Largest distance between two points of the polygon (diameter of its convex hull)."""
    import numpy as np
    c = np.asarray(polygon.convex_hull.exterior.coords)[:, :2]
    return float(np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)).max())


def uncovered_goal(env):
    """Shapely region of the goal not covered by the block: the coverage metric's polygons (pymunk_to_shapely, as in
    physical_metrics), each side merged into one polygon."""
    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
    from shapely.ops import unary_union

    goal = unary_union(list(pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes).geoms))
    block = unary_union(list(pymunk_to_shapely(env.block, env.block.shapes).geoms))
    return goal.difference(block)


def _uncovered_goal_fast(env):
    """uncovered_goal from the block's local outline (merged once per env) moved by the goal and block poses; the same
    rigid transform as pymunk's local_to_world, equal to uncovered_goal up to floating-point rounding."""
    import math
    from shapely.affinity import affine_transform
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    local = getattr(env, "_overlay_outline", None)
    if local is None:
        local = env._overlay_outline = unary_union([Polygon(list(s.get_vertices())) for s in env.block.shapes])

    def placed(x, y, angle):
        c, s = math.cos(angle), math.sin(angle)
        return affine_transform(local, (c, -s, s, c, x, y))
    goal = placed(float(env.goal_pose[0]), float(env.goal_pose[1]), float(env.goal_pose[2]))
    return goal.difference(placed(*env.block.position, env.block.angle))


def goal_overlay_region(env, radius, uncovered=None):
    """Uncovered goal plus its thin parts grown by R. Thin parts: what an opening of radius R/2 removes, keeping pieces at
    least R long (slivers along an edge or rotation wedges; this skips the small corner pieces an opening also cuts
    from wide parts and from the letter's own corners, which are under R long for corners above about 53 degrees).
    Returns None when nothing is left to draw."""
    from shapely.ops import unary_union

    parts = _polygons(_uncovered_goal_fast(env) if uncovered is None else uncovered, OVERLAY_MIN_AREA)
    if not parts:
        return None
    region = unary_union(parts)
    half = radius / 2
    q = OVERLAY_QUAD_SEGS
    cut = region.difference(region.buffer(-half, quad_segs=q).buffer(half, quad_segs=q))
    cut = cut.buffer(-OVERLAY_HAIRLINE / 2, quad_segs=1).buffer(OVERLAY_HAIRLINE / 2, quad_segs=1)   # numerical hairlines
    thin = [p for p in _polygons(cut, OVERLAY_MIN_AREA) if p.length >= 2 * radius and _length(p) >= radius]
    return unary_union([region] + [unary_union(thin).buffer(radius, quad_segs=q)]) if thin else region


def _rasterize(region, x0, y0, width, height):
    """uint8 (height, width) mask of a shapely region over canvas pixels [x0, x0 + width) x [y0, y0 + height)."""
    import cv2
    import numpy as np

    mask = np.zeros((height, width), np.uint8)
    rings = []
    for polygon in _polygons(region, 0.0):
        rings.append(polygon.exterior.coords)
        rings.extend(hole.coords for hole in polygon.interiors)
    # 1/8 px precision; holes are cut out because fillPoly fills overlapping contours with the even-odd rule.
    cv2.fillPoly(mask, [np.round((np.asarray(r)[:, :2] - (x0, y0)) * 8).astype(np.int32) for r in rings], 1,
                 cv2.LINE_8, shift=3)
    return mask


def goal_overlay_mask(env, radius, width=512, height=512, uncovered=None):
    """Boolean (height, width) canvas mask of goal_overlay_region (canvas pixel (x, y) = simulator point (x, y))."""
    import numpy as np

    region = goal_overlay_region(env, radius, uncovered)
    return np.zeros((height, width), bool) if region is None else _rasterize(region, 0, 0, width, height).astype(bool)


def paint_goal_overlay(env, surface, radius):
    """Paint goal_overlay_region on a pygame surface (called between drawing the goal and the bodies)."""
    import math
    import pygame

    region = goal_overlay_region(env, radius)
    if region is None:
        return
    width, height = surface.get_width(), surface.get_height()
    left, top, right, bottom = region.bounds
    x0, y0 = max(0, math.floor(left) - 1), max(0, math.floor(top) - 1)
    x1, y1 = min(width, math.ceil(right) + 2), min(height, math.ceil(bottom) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    mask = _rasterize(region, x0, y0, x1 - x0, y1 - y0).astype(bool)
    pixels = pygame.surfarray.pixels3d(surface)                       # (x, y, 3) view; released below
    pixels[x0:x1, y0:y1][mask.T] = OVERLAY_COLOR
    del pixels


def official_shape(name):
    """Local collision coordinates; screen y increases downward.

    T preserves original body coordinates. L has height 120, total foot width 90,
    and stroke thickness 30; its two nonoverlapping rectangles share one body.
    """
    if name == "T":
        return Shape("T", (Rectangle(-60, 0, 60, 30), Rectangle(-15, 30, 15, 120)), 1, "official_dp")
    if name == "L":
        # Before centering: stem [0,30]x[0,120], foot [30,90]x[90,120].
        # The union's area centroid is (30,75).
        return Shape("L", (Rectangle(-30, -75, 0, 45), Rectangle(0, 15, 60, 45)), 1, "official_l_area_centroid")
    if name == "T_narrow":
        # Same physical center and legacy inertia as T; only cap width changes.
        return Shape(name, (Rectangle(-54, -45, 54, -15), Rectangle(-15, -15, 15, 75)),
                     1, "official_t_reference_center")
    if name in PATTERNS or name in NOVEL_PATTERNS:                    # NOVEL_PATTERNS: held-out test shapes only
        return block_letter(name)
    raise ValueError(f"Unsupported official shape: {name}")


def matching_shape_state(reference, template):
    """Match physical centers/velocities and target centers, not different origins."""
    if (reference.shape != "T" or reference.geometry_hash != official_shape("T").geometry_hash
            or template.shape == "T" or template.geometry_hash != official_shape(template.shape).geometry_hash):
        raise ValueError("Expected official T reference and non-T template snapshots")
    object_center = reference.object_body.pose.transform(T_REFERENCE_COG)
    target_center = reference.goal.transform(T_REFERENCE_COG)
    return replace(template, object_body=replace(reference.object_body, pose=Pose(object_center, reference.object_body.pose.angle)),
                   pusher_body=reference.pusher_body, goal=Pose(target_center, reference.goal.angle),
                   control_step=reference.control_step, pusher_target=reference.pusher_target)



def matching_l_state(reference, template):
    """Compatibility name for the original T/L protocol."""
    if template.shape != "L":
        raise ValueError("Expected L template")
    return matching_shape_state(reference, template)


def _read_body(body):
    return BodyState(Pose(tuple(body.position), body.angle), tuple(body.velocity),
                     body.angular_velocity, tuple(body.force), body.torque)


def _write_body(body, state):
    body.angle = state.pose.angle
    body.position = state.pose.position
    body.velocity = state.velocity
    body.angular_velocity = state.angular_velocity
    body.force = state.force
    body.torque = state.torque


def make_official_shape_env_class(base_class):
    class OfficialShapeEnv(base_class):
        def __init__(self, *args, shape="T", goal_overlay=0, **kwargs):
            self.project_shape = official_shape(shape).name
            self.control_step = 0
            self.goal_overlay = int(goal_overlay)                        # simulator px; 0 = official rendering
            if self.goal_overlay < 0:
                raise ValueError("goal_overlay must be >= 0 simulator px")
            super().__init__(*args, **kwargs)
            if self.project_shape != "T" and self.block_cog is not None:
                raise ValueError("Non-T shapes fix their mass center; custom center of gravity is unsupported")

        def add_tee(self, position, angle, scale=30, color="LightSlateGray", mask=None):
            # Preserve every original T construction argument and its legacy inertia.
            if self.project_shape == "T":
                if mask is None:
                    return super().add_tee(position, angle, scale=scale, color=color)
                return super().add_tee(position, angle, scale=scale, color=color, mask=mask)
            if scale != 30:
                raise ValueError("The letter protocol fixes its geometry at scale 30")
            import pygame
            import pymunk

            shape = official_shape(self.project_shape)
            body = pymunk.Body(1.0, 3000.0 if self.project_shape == "T_narrow" else shape.inertia_about_origin(1.0))
            polygons = [pymunk.Poly(body, rectangle.vertices) for rectangle in shape.components]
            for polygon in polygons:
                polygon.color = pygame.Color(color)
                polygon.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() if mask is None else mask)
            body.center_of_gravity = (0.0, 0.0)
            body.angle, body.position = angle, position
            body.friction = 1  # Same upstream material-setting convention; no contact-law change.
            self.space.add(body, *polygons)
            return body

        def _setup(self):
            super()._setup()
            if self.project_shape != "T":
                # T's goal origin is not its mass center. Align the L target's
                # center with the original T target's mass center.
                goal = Pose(point(self.goal_pose[:2]), float(self.goal_pose[2]))
                self.goal_pose[:2] = goal.transform(T_REFERENCE_COG)
            if self.goal_overlay:
                # The official renderer draws the goal, then calls space.debug_draw for the bodies; painting the
                # overlay just before that keeps the rest of the official image pipeline (canvas, resize) unchanged.
                draw_bodies = self.space.debug_draw

                def debug_draw(options):
                    paint_goal_overlay(self, options.surface, self.goal_overlay)
                    draw_bodies(options)
                self.space.debug_draw = debug_draw

        def _set_state(self, state):
            if self.project_shape != "T":
                import numpy as np
                state = np.asarray(state, dtype=np.float64).copy()
                angle = float(state[4])
                offset = T_REFERENCE_COG if self.legacy else Pose((0, 0), angle).transform(T_REFERENCE_COG)
                state[2:4] += offset
            return super()._set_state(state)

        def reset(self):
            result = super().reset()
            self.control_step = 0
            return result

        def step(self, action):
            if action is not None:
                point(action)
            result = super().step(action)
            if action is not None:
                self.control_step += 1
            return result

        def _physics_hash(self):
            if self.space is None:
                raise RuntimeError("Reset before capturing physical state")
            data = {"protocol": PROTOCOL, "shape": self.project_shape,
                    "geometry_hash": official_shape(self.project_shape).geometry_hash,
                    "mass": self.block.mass, "moment": self.block.moment,
                    "cog": list(self.block.center_of_gravity), "legacy": self.legacy,
                    "sim_hz": self.sim_hz, "control_hz": self.control_hz,
                    "kp": self.k_p, "kv": self.k_v, "damping": self.space.damping,
                    "gravity": list(self.space.gravity), "pusher_radius": next(iter(self.agent.shapes)).radius,
                    "materials": sorted((p.friction, p.elasticity, p.sensor, p.filter.group,
                                         p.filter.categories, p.filter.mask) for p in self.block.shapes)}
            return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()

        def capture_state(self):
            shape = official_shape(self.project_shape)
            return Snapshot(shape.name, shape.geometry_hash, self._physics_hash(),
                            _read_body(self.block), _read_body(self.agent),
                            Pose(point(self.goal_pose[:2]), float(self.goal_pose[2])), self.control_step,
                            None if self.latest_action is None else point(self.latest_action))

        def restore_state(self, snapshot):
            shape = official_shape(self.project_shape)
            if snapshot.shape != shape.name or snapshot.geometry_hash != shape.geometry_hash:
                raise ValueError("Snapshot belongs to different official shape/geometry")
            if self.space is not None and snapshot.physics_hash != self._physics_hash():
                raise ValueError("Snapshot belongs to different official physics")
            self._setup()
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            if self.damping is not None:
                self.space.damping = self.damping
            if snapshot.physics_hash != self._physics_hash():
                raise ValueError("Snapshot belongs to different official physics")
            _write_body(self.block, snapshot.object_body)
            _write_body(self.agent, snapshot.pusher_body)
            self.space.reindex_shapes_for_body(self.block)
            self.space.reindex_shapes_for_body(self.agent)
            self.goal_pose[:] = (*snapshot.goal.position, snapshot.goal.angle)
            self.latest_action, self.control_step = snapshot.pusher_target, snapshot.control_step
            self.render_cache = None
            return self._get_obs()

    return OfficialShapeEnv


def matched_reset(environments, first_seed, counters, attempts=20):
    """Reject unsafe common scenes; count both shapes' resets, not just accepted ones."""
    integer(attempts, "reset candidates", 1, 20)
    integer(first_seed, "first reset seed", 0, 2**32 - attempts)
    if "T" not in environments or len(environments) < 2:
        raise ValueError("Matched evaluation requires T and at least one other shape")
    if any(name != env.project_shape for name, env in environments.items()):
        raise ValueError("Environment names must match their shapes")
    records = []
    for candidate in range(first_seed, first_seed + attempts):
        for env in environments.values():
            env.seed(candidate)
            env.reset()
            counters["reset_physics_substeps"] += 1
        reference = environments["T"].capture_state()
        if tuple(environments["T"].block.center_of_gravity) != T_REFERENCE_COG:
            raise ValueError("Matched scenes require the original T center of gravity")
        states = {name: reference if name == "T" else matching_shape_state(reference, env.capture_state())
                  for name, env in environments.items()}
        issues = {name: reset_issues(official_shape(name), state) for name, state in states.items()}
        records.append({"seed": candidate, "issues": issues})
        if not any(issues.values()):
            observations = {name: environments[name].restore_state(state) for name, state in states.items()}
            return observations, records
    raise ValueError(f"No valid common scene after {attempts} candidates: {records}")
