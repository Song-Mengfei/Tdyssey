"""Isolated, unvalidated v2 geometric teacher. Never imported by the v1 CLI.

Only known simulator geometry/state is used. No policy, demonstrations or training.
The shared initial-history replay and physical forecast implementation is reused.
"""

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import time

from tdyssey.geometry import finite, point
from tdyssey.simulator_planning import PlannerDecision
from tdyssey.simulator_search import (SearchConfig, SimulatorActionSearch,
                                     primitive_actions, _boundary_route, _resample,
                                     _search_score, _wrap)


@dataclass(frozen=True)
class V2Config:
    switch_margin: float = .02
    correction_travel_min: float = 2.
    correction_travel_max: float = 80.
    correction_speed_min: float = 30.
    correction_speed_max: float = 140.
    staging_clearance_min: float = .5
    staging_clearance_max: float = 8.

    def __post_init__(self):
        for name, value in asdict(self).items():
            value = finite(value, name)
            if value < 0 or (name != "switch_margin" and value == 0):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, value)
        for low, high in (("correction_travel_min", "correction_travel_max"),
                          ("correction_speed_min", "correction_speed_max"),
                          ("staging_clearance_min", "staging_clearance_max")):
            if getattr(self, low) > getattr(self, high):
                raise ValueError(f"{low} must not exceed {high}")


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


def _center(shape, state):
    return (state.object_body.pose.transform((0., 45.)) if shape.name == "T"
            else state.object_body.pose.position)


def short_actions(shape, state, family, params, active_budget, max_target_step=32.):
    """Exterior-edge contact, direct visibility, then bounded inward movement.

    params=(categorical edge,u,clearance,travel,yaw,speed,categorical route).
    Approach geometry uses the current shape; actual simulation verifies motion.
    """
    edge_id, u, clearance, travel, yaw, speed, route = params
    edge = _world_edge(shape, state.object_body.pose, int(edge_id))
    q, normal = edge.point(u), edge.normal
    start = state.pusher_body.pose.position
    staged = tuple(q[i]+(15+clearance)*normal[i] for i in (0, 1))
    if not direct_clear(staged, staged, shape, state.object_body.pose, 15+clearance-.05):
        raise ValueError("Exterior contact is inaccessible to the radius-15 pusher")
    route_mode, path = "direct", [start, staged]
    # Already-touching correction omits an unnecessary retreat/full detour.
    normal_gap = sum((start[i]-q[i])*normal[i] for i in (0, 1))
    tangential_gap = abs(_cross(q, (q[0]+normal[0], q[1]+normal[1]), start))
    if 14.5 <= normal_gap <= 15+clearance+1. and tangential_gap <= 3.:
        path, staged, route_mode = [start], start, "already_staged"
    elif not direct_clear(start, staged, shape, state.object_body.pose, 15.1):
        retreat = tuple(start[i]+normal[i]*(clearance+2) for i in (0, 1))
        if (normal_gap >= 14.5 and direct_clear(start, retreat, shape, state.object_body.pose, 14.5)
                and direct_clear(retreat, staged, shape, state.object_body.pose, 15.1)):
            path, route_mode = [start, retreat, staged], "retreat_then_direct"
        else:
            vertices = [v for polygon in shape.polygons(state.object_body.pose) for v in polygon]
            bounds = (min(v[0] for v in vertices)-15-clearance,
                      min(v[1] for v in vertices)-15-clearance,
                      max(v[0] for v in vertices)+15+clearance,
                      max(v[1] for v in vertices)+15+clearance)
            path = _boundary_route(start, staged, bounds, 1 if route >= 0 else -1)
            route_mode = "box_fallback"
    approach_distance = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    approach = math.ceil(approach_distance/(speed*.1))
    if approach >= active_budget:
        raise ValueError("Contact approach cannot fit the active budget")
    inward = -normal[0], -normal[1]
    c, s = math.cos(yaw), math.sin(yaw)
    heading = c*inward[0]-s*inward[1], s*inward[0]+c*inward[1]
    # Account for normal progress at a biased heading; travel is inward progress,
    # independent of staging clearance and of goal-center translation distance.
    gap = max(0., sum((staged[i]-q[i])*normal[i] for i in (0, 1))-15)
    normal_progress = -sum(heading[i]*normal[i] for i in (0, 1))
    push_distance = (travel+gap)/max(.1, normal_progress)
    pushes = max(1, math.ceil(push_distance/(speed*.1)))
    if approach+pushes > active_budget:
        raise ValueError("Contact push cannot fit the active budget")
    finish = tuple(staged[i]+heading[i]*push_distance for i in (0, 1))
    commands = (_resample(path, approach) if approach else [])+_resample([staged, finish], pushes)
    if any(not 25 <= x <= 486 for command in commands for x in command):
        raise ValueError("Correction target leaves the safe pusher workspace")
    if any(math.dist(a, b) > max_target_step for a, b in zip([start, *commands], commands)):
        raise ValueError("Correction exceeds physical first-target or target-to-target bound")
    previous = state.pusher_target
    if previous is not None and math.dist(previous, commands[0]) > max_target_step:
        raise ValueError("Correction first target jumps too far from the previous controller target")
    return tuple(point(a) for a in commands), {
        "approach_controls": approach, "route_mode": route_mode, "edge_id": int(edge_id),
        "contact_point": q, "outward_normal": normal,
        "torque_proxy": torque_proxy(q, _center(shape, state), inward),
        "desired_turn": signed_turn(state.object_body.pose.angle, state.goal.angle, shape.symmetry_order),
    }


def select_candidate(records, best, incumbent, switch_margin):
    completed = [r for r in records if r.get("status") == "completed"]
    if not completed:
        return None, "no_completed_candidate"
    successes = [r for r in completed if r.get("native_success")]
    if successes:
        successful = max(successes, key=lambda r: (not r.get("unsafe_state_count", 0), r["score"]))
        return successful, "native_success"
    if incumbent is None or incumbent.get("status") != "completed":
        return best, "no_completed_incumbent"
    if incumbent.get("unsafe_state_count", 0):
        safe = [r for r in completed if not r.get("unsafe_state_count", 0)]
        if safe:
            return max(safe, key=lambda r: (bool(r.get("native_success")), r["score"])), "unsafe_incumbent"
    if best is incumbent:
        return incumbent, "incumbent_best"
    if best["score"] > incumbent["score"]+switch_margin+1e-12:
        return best, "score_margin_exceeded"
    return incumbent, "incumbent_margin"


class ProposalRejected(ValueError):
    pass


class SimulatorActionSearchV2(SimulatorActionSearch):
    def __init__(self, env_factory, initial_state, config=None, v2_config=None):
        super().__init__(env_factory, initial_state, config)
        self.v2_config = v2_config or V2Config()
        if not isinstance(self.v2_config, V2Config):
            raise ValueError("Expected V2Config")
        self._approach_remaining, self._phase_tail = 0, []

    def record_action(self, action):
        matched = bool(self._tail and math.dist(point(action), self._tail[0]) <= self.config.state_tolerance)
        super().record_action(action)
        if matched:
            if self._phase_tail:
                self._phase_tail.pop(0)
            self._approach_remaining = sum(self._phase_tail[:self._active_remaining])
        else:
            self._phase_tail, self._approach_remaining = [], 0

    def _rank_contacts(self, state, family):
        shape = self.branch.shape
        center = _center(shape, state)
        delta = tuple(g-x for g, x in zip(state.goal.position, center))
        norm = math.hypot(*delta)
        desired = signed_turn(state.object_body.pose.angle, state.goal.angle, shape.symmetry_order)
        result = []
        for index, _ in enumerate(rectangle_exterior_edges(shape.components)):
            edge = _world_edge(shape, state.object_body.pose, index)
            normal = edge.normal
            for u in (.2, .5, .8):
                q = edge.point(u)
                inward = -normal[0], -normal[1]
                stage = tuple(q[i]+17*normal[i] for i in (0, 1))
                if not direct_clear(stage, stage, shape, state.object_body.pose, 16.9):
                    continue
                torque = torque_proxy(q, center, inward)
                if family == "rotation":
                    alignment = (1 if desired >= 0 else -1)*torque/max(1., math.dist(q, center))
                else:
                    alignment = sum(delta[i]*inward[i] for i in (0, 1))/max(1., norm)
                score = alignment-.1*math.dist(state.pusher_body.pose.position, stage)/512
                result.append((score, index, u))
        return sorted(result, reverse=True)

    def _proposal(self, state, family, ordinal, elites=None, explore=False):
        if family == "long":
            if elites:
                base = elites[int(self.rng.choice(tuple(range(len(elites)))))]
                p = base["parameters"]
                floors = (.18, 8., .15, .15, 3., .06, .08)
                members = [r["parameters"] for r in elites if r["route"] == base["route"]]
                values = []
                limits = ((-math.pi, math.pi), (-55, 55), (-1.2, 1.2), (.25, 2.25), (15, 40), (.35, .8), (.6, 1.))
                for i, (low, high) in enumerate(limits):
                    mean = (math.atan2(sum(math.sin(v[i]) for v in members), sum(math.cos(v[i]) for v in members))
                            if i == 0 else sum(v[i] for v in members)/len(members))
                    sd = max(floors[i], math.sqrt(sum((_wrap(v[i]-mean) if i == 0 else v[i]-mean)**2 for v in members)/len(members)))
                    value = self.rng.normal(mean, sd)
                    values.append(_wrap(value) if i == 0 else max(low, min(high, float(value))))
                return {"family": family, "parameters": (*values, base["route"]), "route": base["route"]}
            p = (self._random() if explore else
                 self._seed(ordinal, signed_turn(state.object_body.pose.angle, state.goal.angle)))
            return {"family": family, "parameters": p, "route": p[-1]}
        ranked = self._rank_contacts(state, family)
        if not ranked:
            raise ProposalRejected("No accessible exterior contact")
        route = 1 if ordinal % 2 == 0 else -1
        recover = family == "rotation" and ordinal % 2 == 1
        if elites:
            base = elites[int(self.rng.choice(tuple(range(len(elites)))))]
            edge_id, *_, route = base["parameters"]
            recover = base.get("recover_translation", False)
            members = [r["parameters"] for r in elites if r["edge_id"] == edge_id
                       and r["route"] == route and r.get("recover_translation", False) == recover]
            if not members:
                members = [base["parameters"]]
            limits = ((.15, .85), (self.v2_config.staging_clearance_min, self.v2_config.staging_clearance_max),
                      (self.v2_config.correction_travel_min, self.v2_config.correction_travel_max),
                      (-.6, .6), (self.v2_config.correction_speed_min, self.v2_config.correction_speed_max))
            floors = (.1, 1., 6., .12, 15.)
            values = []
            for i, (low, high) in enumerate(limits, 1):
                mean = sum(p[i] for p in members)/len(members)
                sd = max(floors[i-1], math.sqrt(sum((p[i]-mean)**2 for p in members)/len(members)))
                values.append(max(low, min(high, float(self.rng.normal(mean, sd)))))
        elif explore:
            _, edge_id, _ = ranked[int(self.rng.choice(tuple(range(len(ranked)))))]
            values = [self.rng.uniform(.15, .85),
                      self.rng.uniform(self.v2_config.staging_clearance_min, self.v2_config.staging_clearance_max),
                      self.rng.uniform(self.v2_config.correction_travel_min, self.v2_config.correction_travel_max),
                      self.rng.uniform(-.6, .6),
                      self.rng.uniform(self.v2_config.correction_speed_min, self.v2_config.correction_speed_max)]
            route = int(self.rng.choice((-1, 1)))
            recover = family == "rotation" and bool(self.rng.choice((False, True)))
        else:
            # Alternate high-ranked distinct contacts and broad accessible edges.
            choices = ranked[:max(4, len(ranked)//2)]
            _, edge_id, u = choices[ordinal % len(choices)]
            distance = math.dist(state.goal.position, _center(self.branch.shape, state))
            travel = (10., 25., 45., 65.)[ordinal % 4] if family == "rotation" else min(60., max(5., distance))
            values = [u, 2., travel, (0., -.3, .3, 0.)[ordinal % 4], 100.]
        return {"family": family, "edge_id": int(edge_id), "route": route,
                "recover_translation": recover, "parameters": (int(edge_id), *values, route)}

    def choose(self, current_state, remaining_steps, deadline=None):
        if self.closed:
            raise RuntimeError("Search is closed")
        if type(remaining_steps) is not int or remaining_steps < 1:
            raise ValueError("remaining_steps must be positive")
        if current_state.control_step != self.branch.initial_state.control_step+len(self.branch.executed_actions):
            raise ValueError("Record every real executed action before search")
        if deadline is not None:
            deadline = finite(deadline, "deadline")
        self.branch._check_deadline(deadline)
        if self.rng is None:
            import numpy as np
            self.rng = np.random.default_rng(self.config.seed)
        began, counts_before = time.monotonic(), dict(self.counts)
        budget = min(self.config.horizon+self.config.settle_steps, remaining_steps, 80)
        max_active = min(self.config.horizon, max(1, budget-min(self.config.settle_steps, budget-1)))
        records, stored, completed = [], {}, []
        attempted, timed_out = 0, False

        def build(spec, state, active_budget):
            try:
                if spec["family"] == "long":
                    active = max(1, min(active_budget, round(active_budget*spec["parameters"][6])))
                    commands, extra = primitive_actions(self.branch.shape, state, spec["parameters"], active, self.config.max_target_step)
                    if state.pusher_target is not None and math.dist(state.pusher_target, commands[0]) > self.config.max_target_step:
                        raise ValueError("Long primitive first target jumps too far from previous target")
                else:
                    commands, extra = short_actions(self.branch.shape, state, spec["family"], spec["parameters"], active_budget, self.config.max_target_step)
                return tuple(commands), extra
            except ValueError as exc:
                raise ProposalRejected(str(exc)) from exc

        def evaluate(kind, spec=None):
            nonlocal attempted, timed_out
            if attempted >= self.config.max_evaluations:
                return False
            self.branch._check_deadline(deadline)
            attempted += 1
            before = dict(self.counts)
            record = {"candidate_index": len(records), "kind": kind, "generation": generation,
                      "family": spec["family"] if spec else kind,
                      "parameters": list(spec["parameters"]) if spec else None,
                      "edge_id": spec.get("edge_id") if spec else None,
                      "route": spec.get("route") if spec else None,
                      "recover_translation_requested": spec.get("recover_translation", False) if spec else False,
                      "recover_translation": False}
            try:
                extra, first_stage = {}, None
                if kind == "primitive":
                    stage_limit = min(32, max_active//2) if spec.get("recover_translation") and max_active >= 16 else max_active
                    commands, extra = build(spec, current_state, stage_limit)
                    active = len(commands)
                    phases = [True]*extra["approach_controls"]+[False]*(active-extra["approach_controls"])
                    if stage_limit < max_active:
                        first_stage = commands
                elif kind == "incumbent_tail":
                    active = min(self._active_remaining, max_active, len(self._tail))
                    commands = tuple(self._tail[:active])
                    phases = list(self._phase_tail[:active])
                    if len(phases) < active:
                        phases += [False]*(active-len(phases))
                else:
                    commands, phases, active = (), [], 0
                replay_error = self.branch._replay(current_state, self.counts, deadline)
                if first_stage:
                    a1, r1, s1 = self.branch._forecast(commands, active, current_state, active, self.counts, deadline)
                    if r1[-1]["native_done"]:
                        commands, rows, predictions = a1, r1, s1
                    else:
                        next_state = s1[-1]
                        # One transparent deterministic recovery, no hidden extra rollouts.
                        recovery, recovery_extra = None, None
                        for ordinal in range(4):
                            proposal = self._proposal(next_state, "translation", ordinal)
                            try:
                                recovery, recovery_extra = build(proposal, next_state, max_active-len(a1))
                                break
                            except ProposalRejected:
                                pass
                        if recovery is None:
                            raise ProposalRejected("Composite rotation has no fitting translation recovery")
                        a2, r2, s2 = self.branch._forecast(recovery, len(recovery), next_state, budget-len(a1), self.counts, deadline)
                        commands, rows, predictions = a1+a2, r1+r2, s1+s2
                        phases += ([True]*recovery_extra["approach_controls"]
                                   +[False]*(len(recovery)-recovery_extra["approach_controls"]))
                        active += len(recovery)
                        extra["second_stage"] = recovery_extra
                        record["recover_translation"] = True
                else:
                    commands, rows, predictions = self.branch._forecast(commands, active, current_state, budget, self.counts, deadline)
                phases = (phases+[False]*len(commands))[:len(commands)]
                score, diagnostics = _search_score(rows, predictions, self.branch.shape, self.config, 0)
                approaches = sum(phases)
                approach_contacts = sum(flag and row["n_contacts"] > 0 for flag, row in zip(phases, rows))
                score -= .02*approach_contacts/max(1, approaches)
                diagnostics["approach_contact_controls"] = approach_contacts
                self.branch._check_deadline(deadline)
                record.update(status="completed", score=score, policy_prefix=min(active, len(commands)),
                              forecast_steps=len(rows), replay_state_error=replay_error,
                              final_coverage=rows[-1]["coverage"], final_iou=rows[-1]["iou"],
                              peak_coverage=max(r["coverage"] for r in rows), native_success=rows[-1]["native_done"],
                              final_position_error=rows[-1]["position_error"], final_angle_error=rows[-1]["angle_error"],
                              contact_controls=sum(r["n_contacts"] > 0 for r in rows),
                              **{k: v for k, v in extra.items() if k != "approach_controls"},
                              approach_controls=approaches, **diagnostics)
                completed.append(record)
                stored[record["candidate_index"]] = commands, predictions, min(active, len(commands)), phases
            except ProposalRejected as exc:
                record.update(status="rejected", reason=str(exc))
            except TimeoutError:
                timed_out = True
                record.update(status="timed_out")
            finally:
                record["counts"] = {k: v-before[k] for k, v in self.counts.items()}
                records.append(record)
            return not timed_out

        generation = -1
        try:
            if self._tail and self._active_remaining:
                evaluate("incumbent_tail")
            if not timed_out and attempted < self.config.max_evaluations:
                evaluate("hold")
            for generation in range(self.config.generations):
                if timed_out or attempted >= self.config.max_evaluations:
                    break
                for index in range(self.config.population):
                    if timed_out or attempted >= self.config.max_evaluations:
                        break
                    self.branch._check_deadline(deadline)
                    family = ("long", "translation", "rotation")[index % 3]
                    elites = sorted((r for r in completed if r["family"] == family and r["parameters"]),
                                    key=lambda r: r["score"], reverse=True)[:self.config.elite_count]
                    refine = generation and len(elites) >= 2 and self.rng.uniform(0., 1.) >= self.config.exploration_fraction
                    try:
                        spec = self._proposal(current_state, family, index//3+generation*4,
                                              elites if refine else None, explore=bool(generation and not refine))
                    except ProposalRejected as exc:
                        attempted += 1
                        records.append({"candidate_index": len(records), "kind": "primitive", "family": family,
                                        "generation": generation, "parameters": None, "status": "rejected",
                                        "reason": str(exc), "counts": {k: 0 for k in self.counts}})
                        continue
                    evaluate("primitive", spec)
        except TimeoutError:
            timed_out = True
        if not completed:
            raise TimeoutError("No complete simulator-search candidate before deadline")
        raw_best = max(completed, key=lambda r: r["score"])
        incumbent = next((r for r in completed if r["kind"] == "incumbent_tail"), None)
        selected, rule = select_candidate(records, raw_best, incumbent, self.v2_config.switch_margin)
        commands, predictions, active, phases = stored[selected["candidate_index"]]
        selected["first_state"] = asdict(predictions[0])
        self._tail, self._active_remaining = list(commands), active
        self._phase_tail, self._approach_remaining = list(phases), sum(phases[:active])
        return PlannerDecision(commands, {
            "variant": "isolated_v2_draft", "config": asdict(self.config), "v2_config": asdict(self.v2_config),
            "counts": {k: v-counts_before[k] for k, v in self.counts.items()}, "selected": selected,
            "selected_index": selected["candidate_index"], "candidates": records,
            "selected_actions": [list(a) for a in commands], "selected_predictions": [asdict(s) for s in predictions],
            "timed_out": timed_out, "forecast_budget": budget, "forecast_hard_cap": 80,
            "completed_candidates": len(completed), "requested_candidates": attempted, "evaluations": attempted,
            "rejected_candidates": [r for r in records if r["status"] == "rejected"],
            "generations_completed": len({r["generation"] for r in completed if r["generation"] >= 0}),
            "elapsed_seconds": time.monotonic()-began, "wall_planning_seconds": time.monotonic()-began,
            "execute_steps": min(self.config.execute_steps, len(commands)),
            "selection_rule": rule, "switch_margin": self.v2_config.switch_margin,
            "raw_best_index": raw_best["candidate_index"], "raw_best_score": raw_best["score"],
            "incumbent_score": incumbent["score"] if incumbent else None,
            "score_gap": raw_best["score"]-incumbent["score"] if incumbent else None,
            "remaining_approach_controls": self._approach_remaining,
            "branch_protocol": "initial_snapshot_plus_all_executed_actions", "settling_uses_real_budget": True,
            "policy_samples": False, "demonstrations": False, "training": False, "privileged_physics": True,
            "search_method": "exterior_contacts_signed_rotation_composite_recovery_categorical_CEM",
        })


# Optional isolated launchers may use the original entry-class spelling.
SimulatorActionSearch = SimulatorActionSearchV2
