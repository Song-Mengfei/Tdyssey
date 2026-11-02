"""Exterior contact geometry for the policy-free simulator teacher.

Promoted from the preserved V2 geometry prototype; holes remain real obstacles.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math
from .simulator_search import _wrap

@dataclass(frozen=True)
class Edge:
    a: tuple
    b: tuple

    @property
    def normal(self):
        dx, dy = self.b[0]-self.a[0], self.b[1]-self.a[1]
        length = math.hypot(dx, dy)
        return dy/length, -dx/length

    def point(self, u):
        return tuple(x+u*(y-x) for x, y in zip(self.a, self.b))


def _rectangles(components):
    return tuple(tuple(float(getattr(r, n)) for n in ("x0", "y0", "x1", "y1"))
                 if hasattr(r, "x0") else tuple(map(float, r)) for r in components)


@lru_cache(maxsize=32)
def _exterior_cached(rectangles):
    """Boundary of occupied grid cells adjacent to outside-connected free cells.

    Splitting at every component coordinate handles partial shared seams. Flood
    filling excludes enclosed holes; rectangles need not form a convex polygon.
    """
    xs = sorted({v for r in rectangles for v in (r[0], r[2])})
    ys = sorted({v for r in rectangles for v in (r[1], r[3])})
    xs, ys = [xs[0]-1, *xs, xs[-1]+1], [ys[0]-1, *ys, ys[-1]+1]
    nx, ny = len(xs)-1, len(ys)-1
    occupied = set()
    for i in range(nx):
        for j in range(ny):
            x, y = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2
            if any(a < x < c and b < y < d for a, b, c, d in rectangles):
                occupied.add((i, j))
    outside, queue = {(0, 0)}, deque([(0, 0)])
    while queue:
        i, j = queue.popleft()
        for cell in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
            if (0 <= cell[0] < nx and 0 <= cell[1] < ny
                    and cell not in occupied and cell not in outside):
                outside.add(cell)
                queue.append(cell)
    edges = []
    for i, j in sorted(occupied):
        x0, x1, y0, y1 = xs[i], xs[i+1], ys[j], ys[j+1]
        for neighbor, a, b in (((i, j-1), (x0, y0), (x1, y0)),
                               ((i+1, j), (x1, y0), (x1, y1)),
                               ((i, j+1), (x1, y1), (x0, y1)),
                               ((i-1, j), (x0, y1), (x0, y0))):
            if neighbor in outside:
                edges.append(Edge(a, b))
    # Merge contiguous collinear subsegments while retaining their orientation.
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(edges):
            for j, second in enumerate(edges):
                if i != j and first.b == second.a and first.normal == second.normal:
                    merged = Edge(first.a, second.b)
                    edges = [e for k, e in enumerate(edges) if k not in (i, j)] + [merged]
                    changed = True
                    break
            if changed:
                break
    return tuple(sorted(edges, key=lambda e: (e.a, e.b)))


def rectangle_exterior_edges(components):
    rectangles = _rectangles(components)
    if not rectangles:
        raise ValueError("No rectangle components")
    return _exterior_cached(rectangles)


def signed_turn(actual, goal, symmetry_order=1):
    return _wrap(symmetry_order*(goal-actual))/symmetry_order


def torque_proxy(contact, center, inward):
    return ((contact[0]-center[0])*inward[1]
            -(contact[1]-center[1])*inward[0])


def _local(pose, p):
    x, y = p[0]-pose.position[0], p[1]-pose.position[1]
    c, s = math.cos(pose.angle), math.sin(pose.angle)
    return c*x+s*y, -s*x+c*y


def _point_segment(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    square = dx*dx+dy*dy
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/square)) if square else 0.
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def _cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def _intersects(a, b, c, d):
    return (max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))+1e-10
            and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))+1e-10
            and _cross(a, b, c)*_cross(a, b, d) <= 1e-10
            and _cross(c, d, a)*_cross(c, d, b) <= 1e-10)


def _segment_rectangle(a, b, rect):
    x0, y0, x1, y1 = rect
    def inside(p):
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1
    vertices = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if inside(a) or inside(b):
        return 0.
    pairs = tuple(zip(vertices, (*vertices[1:], vertices[0])))
    if any(_intersects(a, b, c, d) for c, d in pairs):
        return 0.
    return min([_point_segment(v, a, b) for v in vertices]
               + [_point_segment(p, c, d) for p in (a, b) for c, d in pairs])


def direct_clear(start, end, shape, pose, clearance=15.):
    if any(not 25 <= x <= 486 for p in (start, end) for x in p):
        return False
    a, b = _local(pose, start), _local(pose, end)
    return all(_segment_rectangle(a, b, r) >= clearance-1e-7
               for r in _rectangles(shape.components))


def _world_edge(shape, pose, index):
    edge = rectangle_exterior_edges(shape.components)[index]
    return Edge(pose.transform(edge.a), pose.transform(edge.b))

