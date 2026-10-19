"""Small, dependency-free shape definitions in GPC's pixel/world coordinate frame.

T matches the pinned GPC local vertices. I and L use an area-centroid origin.
Components are nonoverlapping rectangles on one body, not independent objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

Point = tuple[float, float]
EPS = 1e-9


def finite(value: Real, name: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def point(values) -> Point:
    if len(values) != 2:
        raise ValueError("A point must contain exactly two coordinates")
    return finite(values[0], "x"), finite(values[1], "y")


@dataclass(frozen=True)
class Pose:
    position: Point
    angle: float

    def __post_init__(self):
        object.__setattr__(self, "position", point(self.position))
        object.__setattr__(self, "angle", finite(self.angle, "angle"))

    def transform(self, vertex: Point) -> Point:
        x, y = vertex
        c, s = math.cos(self.angle), math.sin(self.angle)
        return self.position[0] + c * x - s * y, self.position[1] + s * x + c * y


@dataclass(frozen=True)
class Rectangle:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Rectangle dimensions must be positive")

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def center(self) -> Point:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def vertices(self) -> tuple[Point, ...]:
        return ((self.x0, self.y0), (self.x1, self.y0),
                (self.x1, self.y1), (self.x0, self.y1))

    def overlap(self, other: Rectangle) -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0)) * max(
            0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def shares_edge(self, other: Rectangle) -> bool:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        return (abs(dx) <= EPS and dy > EPS) or (abs(dy) <= EPS and dx > EPS)


@dataclass(frozen=True)
class Shape:
    name: str
    components: tuple[Rectangle, ...]
    symmetry_order: int
    origin: str

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        if not self.name or not self.components:
            raise ValueError("A shape needs a name and components")
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValueError("Shape area must be finite and positive")
        if isinstance(self.symmetry_order, bool) or self.symmetry_order not in (1, 2, 4):
            raise ValueError("Supported rotational symmetry orders are 1, 2, and 4")
        for i, a in enumerate(self.components):
            for b in self.components[i + 1:]:
                if a.overlap(b) > EPS:
                    raise ValueError("Shape components must not overlap in area")
        connected = {0}
        while True:
            adjacent = {i for i, rect in enumerate(self.components)
                        if any(rect.shares_edge(self.components[j]) for j in connected)}
            new = connected | adjacent
            if new == connected:
                break
            connected = new
        if len(connected) != len(self.components):
            raise ValueError("Shape must be connected by edges, not only touching corners")
        if self.symmetry_order > 1:
            angle = 2 * math.pi / self.symmetry_order
            rotated = []
            for rect in self.components:
                vertices = [Pose((0, 0), angle).transform(v) for v in rect.vertices]
                rotated.append(Rectangle(min(v[0] for v in vertices), min(v[1] for v in vertices),
                                         max(v[0] for v in vertices), max(v[1] for v in vertices)))
            shared = sum(a.overlap(b) for a in self.components for b in rotated)
            if not math.isclose(shared, self.area, rel_tol=1e-10, abs_tol=EPS):
                raise ValueError("Declared rotational symmetry does not match the geometry")

    @property
    def area(self) -> float:
        return sum(rect.area for rect in self.components)

    @property
    def centroid(self) -> Point:
        return tuple(sum(rect.area * rect.center[i] for rect in self.components) / self.area
                     for i in (0, 1))

    @property
    def geometry_hash(self) -> str:
        payload = {"version": 1, "name": self.name, "origin": self.origin,
                   "symmetry_order": self.symmetry_order,
                   "rectangles": [[r.x0, r.y0, r.x1, r.y1] for r in self.components]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def polygons(self, pose: Pose) -> tuple[tuple[Point, ...], ...]:
        return tuple(tuple(pose.transform(v) for v in rect.vertices) for rect in self.components)

    def inertia_about_origin(self, mass: float) -> float:
        """Uniform density, area-weighted component masses, parallel-axis theorem.

        Used for new I/L bodies. Do not substitute this for upstream T inertia.
        """
        mass = finite(mass, "mass")
        if mass <= 0:
            raise ValueError("Mass must be positive")
        return sum(mass * rect.area / self.area * (
            ((rect.x1 - rect.x0) ** 2 + (rect.y1 - rect.y0) ** 2) / 12
            + rect.center[0] ** 2 + rect.center[1] ** 2) for rect in self.components)

    def angle_error(self, actual: float, goal: float) -> float:
        delta = (finite(actual) - finite(goal)) * self.symmetry_order
        return abs(math.atan2(math.sin(delta), math.cos(delta))) / self.symmetry_order


def get_shape(name: str, scale: float = 30.0) -> Shape:
    """Same scale meaning as GPC: bar thickness; bounding extent is four scales."""
    name = name.upper()
    scale = finite(scale, "scale")
    if scale <= 0:
        raise ValueError("Scale must be positive")
    if name == "T":
        raw, symmetry, origin = [(-2, -1.5, 2, -0.5), (-0.5, -0.5, 0.5, 2.5)], 1, "upstream_gpc"
    elif name == "I":
        raw, symmetry, origin = [(-0.5, -2, 0.5, 2)], 2, "area_centroid"
    elif name == "L":
        raw, symmetry, origin = [(-2, -2, -1, 2), (-1, 1, 2, 2)], 1, "area_centroid"
        total = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in raw)
        cx = sum((x1 - x0) * (y1 - y0) * (x0 + x1) / 2 for x0, y0, x1, y1 in raw) / total
        cy = sum((x1 - x0) * (y1 - y0) * (y0 + y1) / 2 for x0, y0, x1, y1 in raw) / total
        raw = [(x0 - cx, y0 - cy, x1 - cx, y1 - cy) for x0, y0, x1, y1 in raw]
    else:
        raise ValueError(f"Unsupported project shape: {name}; expected T, I, or L")
    return Shape(name, tuple(Rectangle(*(value * scale for value in rect)) for rect in raw),
                 symmetry, origin)


def polygon_area(vertices) -> float:
    if len(vertices) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(vertices, (*vertices[1:], vertices[0])))) / 2


def convex_intersection(subject, clip) -> tuple[Point, ...]:
    """Sutherland-Hodgman clipping for the counterclockwise component polygons."""
    output = list(subject)
    for a, b in zip(clip, (*clip[1:], clip[0])):
        if not output:
            break
        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        incoming, output = output, []
        previous = incoming[-1]
        previous_side = side(previous)
        for current in incoming:
            current_side = side(current)
            previous_inside, current_inside = previous_side >= -EPS, current_side >= -EPS
            if current_inside != previous_inside:
                denominator = previous_side - current_side
                if abs(denominator) > EPS:
                    t = previous_side / denominator
                    output.append((previous[0] + t * (current[0] - previous[0]),
                                   previous[1] + t * (current[1] - previous[1])))
            if current_inside:
                output.append(current)
            previous, previous_side = current, current_side
    return tuple(output)
