"""Bounded two-contact search with exact-history Pymunk evaluation.

Privileged geometry and state only. Inspired by contact-mode planning and iCEM
memory; this is not a convex relaxation or a learned controller.
"""
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import heapq
import math
import time

from .contact_geometry import (direct_clear, rectangle_exterior_edges, _world_edge,
                               torque_proxy, signed_turn)
from .geometry import finite, point
from .simulator_planning import PlannerConfig, PlannerDecision, SimulatorReranker, state_error
from .simulator_search import _resample


@dataclass(frozen=True)
class SequenceConfig:
    horizon: int = 80
    settle_steps: int = 4
    roots: int = 8
    beam_width: int = 4
    children: int = 4
    refinements: int = 8
    max_evaluations: int = 40
    max_global_searches: int = 8
    seed: int = 42
    state_tolerance: float = 1e-7
    max_target_step: float = 32.

    def __post_init__(self):
        for name, low, high in (("horizon", 4, 80), ("settle_steps", 0, 16),
                                ("roots", 1, 32), ("beam_width", 1, self.roots),
                                ("children", 1, 16), ("refinements", 0, 64),
                                ("max_evaluations", 1, 128), ("max_global_searches", 1, 32),
                                ("seed", 0, 2**32-1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        for name in ("state_tolerance", "max_target_step"):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"Invalid {name}")


@dataclass(frozen=True)
class ContactStage:
    edge: int
    u: float = .5
    clearance: float = 2.
    travel: float = 55.
    yaw: float = 0.
    speed: float = 160.
    route: int = 1
    bend: float = 0.
    approach_speed: float = 0.
    sustain: int = 0
    precision: bool = False

    def __post_init__(self):
        if type(self.edge) is not int or self.edge < 0 or self.route not in (-1, 1):
            raise ValueError("Invalid categorical contact/route")
        for name, lo, hi in (("u", .1, .9), ("clearance", .5, 8.), ("travel", .5, 140.),
                             ("yaw", -.7, .7), ("speed", 12., 220.), ("bend", -.6, .6)):
            if not lo <= finite(getattr(self, name), name) <= hi:
                raise ValueError(f"Invalid stage {name}")
        if self.approach_speed != 0 and not 60 <= finite(self.approach_speed, "approach_speed") <= 240:
            raise ValueError("Invalid approach speed")
        if type(self.sustain) is not int or not 0 <= self.sustain <= 4 or type(self.precision) is not bool:
            raise ValueError("Invalid precision press")


class ProposalRejected(ValueError):
    """An invalid proposal, never a hidden physics/protocol failure."""


class SearchBudgetExhausted(RuntimeError):
    pass


@lru_cache(maxsize=32)
def _material_points(shape):
    return tuple(sorted({p for rect in shape.components for p in rect.vertices}))


def pose_rms(shape, state):
    """Fixed material-point correspondence; rotation never fades with overlap."""
    local = _material_points(shape)
    # Symmetric shapes permit only their declared equivalent goal rotations.
    actual, goal = state.object_body.pose, state.goal
    dx, dy = actual.position[0]-goal.position[0], actual.position[1]-goal.position[1]
    errors = []
    for k in range(shape.symmetry_order):
        angle = goal.angle+2*math.pi*k/shape.symmetry_order
        c, s = math.cos(actual.angle)-math.cos(angle), math.sin(actual.angle)-math.sin(angle)
        errors.append(math.sqrt(sum((dx+c*x-s*y)**2+(dy+s*x+c*y)**2 for x, y in local)/len(local)))
    return min(errors)


def path_key(record):
    """Hard safety first, absorbing native success, then joint endpoint error."""
    return (bool(record["unsafe"]), not record["native_success"],
            not record.get("joint_feasible", True) if not record["native_success"] else False,
            record.get("joint_violation", 0.) if not record["native_success"] else 0.,
            len(record["rows"]) if record["native_success"] else record.get("balanced_score", record.get("completion_score", record["pose_rms"])),
            -record["final_coverage"], record["motion"], record["approach_controls"])


def joint_limits(state, shape, previous=None):
    """Ratchet terminal tolerances between macros, allowing bounded interim motion."""
    position = math.dist(state.object_body.pose.position, state.goal.position)
    angle = shape.angle_error(state.object_body.pose.angle, state.goal.angle)
    center_limit, angle_limit = max(3., position+.25), max(math.radians(2), angle+math.radians(.25))
    if previous is not None:
        center_limit = min(center_limit, previous["center_limit"])
        angle_limit = min(angle_limit, previous["angle_limit"])
    return dict(center_limit=center_limit, angle_limit=angle_limit,
        path_center_limit=max(position, center_limit)+3.,
        path_angle_limit=max(angle, angle_limit)+math.radians(8))


def joint_errors(states, shape, limits):
    positions = [math.dist(s.object_body.pose.position, s.goal.position) for s in states]
    angles = [shape.angle_error(s.object_body.pose.angle, s.goal.angle) for s in states]
    terminal_position, terminal_angle = max(positions[-3:]), max(angles[-3:])
    violation = (max(0., terminal_position-limits["center_limit"])/3
        +max(0., terminal_angle-limits["angle_limit"])/math.radians(2)
        +max(0., max(positions)-limits["path_center_limit"])/3
        +max(0., max(angles)-limits["path_angle_limit"])/math.radians(2))
    return dict(joint_feasible=violation <= 1e-9, joint_violation=violation,
        terminal_position_max=terminal_position, terminal_angle_max=terminal_angle,
        path_position_max=max(positions), path_angle_max=max(angles),
        balanced_score=max(terminal_position/3, terminal_angle/math.radians(2)))


def family_frontier(records):
    """Keep independent position, angle, joint and actually compensated routes."""
    pool = [r for r in records if r.get("stage_objects") and not r["unsafe"]]
    if not pool:
        return []
    result = []
    rankings = (sorted(pool, key=lambda r:r["final_position_error"]),
                sorted(pool, key=lambda r:r["final_angle_error"]), sorted(pool, key=path_key),
                sorted(pool, key=lambda r:(not r.get("actual_compensation", False), path_key(r))))
    for rows in rankings:
        row = next((r for r in rows if all(r is not p for p in result)), None)
        if row is not None:
            result.append(row)
    return result


def settling_reserve(state, budget, maximum):
    """Never reserve all but one of the last controls for mandatory holding."""
    moving = math.hypot(*state.object_body.velocity) > 5 or abs(state.object_body.angular_velocity) > .1
    return min(maximum, budget//(8 if moving else 12))


def correction_distance(shape, state):
    """Geometric lower-bound proxy for access to a useful next contact, not a proof."""
    center, goal = state.object_body.pose.position, state.goal.position
    delta = tuple(goal[k]-center[k] for k in (0, 1))
    turn = signed_turn(state.object_body.pose.angle, state.goal.angle, shape.symmetry_order)
    distances = []
    for i in range(len(rectangle_exterior_edges(shape.components))):
        edge = _world_edge(shape, state.object_body.pose, i)
        for u in (.2, .8):
            q, normal = edge.point(u), edge.normal
            inward = tuple(-x for x in normal)
            if (turn*torque_proxy(q, center, inward) <= 0
                    and sum(delta[k]*inward[k] for k in (0, 1)) <= 0):
                continue
            staged = tuple(q[k]+16*normal[k] for k in (0, 1))
            if direct_clear(staged, staged, shape, state.object_body.pose, 15.9):
                distances.append(math.dist(state.pusher_body.pose.position, staged))
    return min(distances, default=0.)


def diverse_prefixes(records, width):
    """Protect orientation progress and different contacts before leaf ranking."""
    records = [r for r in records if r.get("prefix_state") is not None and not r["unsafe"]]
    if not records:
        return []
    ordered = sorted(records, key=path_key)
    result = [ordered[0]]
    if width > 1 and any(r.get("prefix_position_error") is not None for r in records):
        position_best = min(records, key=lambda r:r.get("prefix_position_error", math.inf))
        if position_best not in result:
            result.append(position_best)
    if len(result) < width:
        angle_best = min(records, key=lambda r: (r["prefix_angle_error"], r["prefix_rms"]))
        if angle_best not in result:
            result.append(angle_best)
    used = {r["diversity_key"] for r in result}
    for row in sorted(records, key=lambda r: (r["prefix_rms"], r["approach_controls"])):
        if len(result) >= width:
            break
        if row["diversity_key"] not in used:
            result.append(row)
            used.add(row["diversity_key"])
    for row in ordered:
        if len(result) >= width:
            break
        if row not in result:
            result.append(row)
    return result


def approach_path(shape, state, target, route, clearance=16.):
    """Visibility graph around actual exterior corners, with physical release.

    Geometry is a proposal prior only; the whole path is subsequently simulated.
    The obstacle is never replaced by its convex hull or by an endpoint restore.
    """
    start, pose = state.pusher_body.pose.position, state.object_body.pose
    edges = rectangle_exterior_edges(shape.components)
    nearest = []
    for i, edge in enumerate(edges):
        world = _world_edge(shape, pose, i)
        dx, dy = world.b[0]-world.a[0], world.b[1]-world.a[1]
        u = max(0., min(1., sum((start[k]-world.a[k])*(dx, dy)[k] for k in (0, 1))/(dx*dx+dy*dy)))
        q = world.point(u)
        distance = math.dist(start, q)
        radial = tuple((start[k]-q[k])/max(1e-9, distance) for k in (0, 1))
        nearest.append((distance, world.normal, radial))
    distance, normal, radial = min(nearest)
    release = start
    if distance < clearance+.2:
        # At a corner two faces can be equally near. A face normal alone may
        # leave the disc intersecting the inflated obstacle. Validate the whole
        # release segment and its endpoint; prefer the actual closest-point ray.
        directions = [radial, normal]+[(math.cos(i*math.pi/8), math.sin(i*math.pi/8)) for i in range(16)]
        exits = []
        for length in (max(1., clearance+1-distance), 4., 8., 16.):
            for direction in directions:
                candidate = tuple(start[k]+length*direction[k] for k in (0, 1))
                if (direct_clear(candidate, candidate, shape, pose, clearance)
                        and direct_clear(start, candidate, shape, pose, max(0., distance-.1))):
                    exits.append(candidate)
            if exits:
                break
        if not exits:
            raise ProposalRejected("No nonpenetrating release from current contact")
        release = min(exits, key=lambda p: math.dist(start, p)+.01*math.dist(p, target))
    if direct_clear(release, target, shape, pose, clearance-.1):
        return [start, release, target], "release_direct" if release != start else "direct"
    vertices = {}
    for edge in edges:
        for vertex in (edge.a, edge.b):
            vertices.setdefault(vertex, []).append(edge.normal)
    nodes = [release, target]
    for vertex, normals in vertices.items():
        local = tuple(vertex[k]+(clearance+1)*sum(n[k] for n in normals) for k in (0, 1))
        p = pose.transform(local)
        if direct_clear(p, p, shape, pose, clearance):
            nodes.append(p)
    graph = [[] for _ in nodes]
    for i, a in enumerate(nodes):
        for j in range(i):
            if direct_clear(a, nodes[j], shape, pose, clearance-.1):
                cost = math.dist(a, nodes[j])
                graph[i].append((j, cost))
                graph[j].append((i, cost))

    def shortest(forbidden=None):
        queue, best = [(0., 0, (0,))], {0: 0.}
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == 1:
                return path
            if cost > best.get(node, math.inf):
                continue
            for other, step in graph[node]:
                if forbidden is not None and {node, other} == set(forbidden):
                    continue
                total = cost+step
                if total < best.get(other, math.inf):
                    best[other] = total
                    heapq.heappush(queue, (total, other, (*path, other)))
        return None

    indices = shortest()
    if indices and route < 0 and len(indices) > 2:
        alternative = shortest(indices[:2])
        indices = alternative or indices
    if indices is None:
        raise ProposalRejected("No radius-clear path to exterior contact")
    return [start, *(nodes[i] for i in indices)], "visibility_alternative" if route < 0 else "visibility_shortest"


def stage_actions(shape, state, stage, active_budget, max_target_step=32.):
    edge = _world_edge(shape, state.object_body.pose, stage.edge)
    q, normal = edge.point(stage.u), edge.normal
    staged = tuple(q[k]+(15+stage.clearance)*normal[k] for k in (0, 1))
    if not direct_clear(staged, staged, shape, state.object_body.pose, 15+stage.clearance-.05):
        raise ProposalRejected("Inaccessible exterior contact")
    start = state.pusher_body.pose.position
    nearby = (stage.precision and math.dist(start, staged) <= 12
              and sum((start[k]-q[k])*normal[k] for k in (0, 1)) >= 14.5
              and direct_clear(start, staged, shape, state.object_body.pose, 14.4))
    if nearby:
        path, route_mode = [start, staged], "maintain_nearby_contact"
    else:
        path, route_mode = approach_path(shape, state, staged, stage.route,
                                          min(16., 15+stage.clearance))
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    approach = math.ceil(length/((stage.approach_speed or stage.speed)*.1))
    inward = -normal[0], -normal[1]
    heading = math.atan2(inward[1], inward[0])+stage.yaw
    distance = (stage.travel+stage.clearance)/max(.3, math.cos(stage.yaw))
    middle = tuple(staged[k]+distance*.5*(math.cos(heading), math.sin(heading))[k] for k in (0, 1))
    finish = tuple(middle[k]+distance*.5*(math.cos(heading+stage.bend), math.sin(heading+stage.bend))[k] for k in (0, 1))
    pushes = max(1, math.ceil(distance/(stage.speed*.1)))
    if approach+pushes > active_budget:
        raise ProposalRejected("Approach and actual push exceed stage budget")
    sustain = min(stage.sustain, active_budget-approach-pushes)
    commands = ((_resample(path, approach) if approach else [])
                +_resample([staged, middle, finish], pushes)+[finish]*sustain)
    previous = state.pusher_target
    if (any(not 25 <= x <= 486 for a in commands for x in a)
            or any(math.dist(a, b) > max_target_step+1e-9 for a, b in zip([state.pusher_body.pose.position, *commands], commands))
            or previous is not None and math.dist(previous, commands[0]) > max_target_step+1e-9):
        raise ProposalRejected("Unsafe target or discontinuous PD target")
    phases = ["approach"]*approach+["push"]*(pushes+sustain)
    return tuple(map(point, commands)), phases, {"approach_controls": approach, "push_controls": pushes+sustain,
        "sustain_controls": sustain, "precision": stage.precision,
        "route_mode": route_mode, "contact_point": q, "outward_normal": normal}


class ContactSequenceSearch:
    def __init__(self, env_factory, initial_state, config=None):
        self.config = config or SequenceConfig()
        self.branch = SimulatorReranker(env_factory, initial_state,
            PlannerConfig(horizon=self.config.horizon, prefix_lengths=(self.config.horizon,),
                          settle_steps=0, include_hold=False, state_tolerance=self.config.state_tolerance))
        self.counts = self.branch.counts
        self._tail, self._predictions, self._phases = [], [], []
        self._active = 0
        self._expected = None
        self._memory = []
        self._selected = None
        self._observations = []
        self.global_searches = 0
        self.closed = False
        self.rng = None
        self._precision = False
        self._limits = None

    def check_execution(self, state):
        if self._limits is None or self._selected is None or self._selected["native_success"]:
            return None
        position = math.dist(state.object_body.pose.position, state.goal.position)
        angle = self.branch.shape.angle_error(state.object_body.pose.angle, state.goal.angle)
        if position > self._limits["path_center_limit"]+1e-7 or angle > self._limits["path_angle_limit"]+1e-7:
            raise ValueError("Executed macro left its joint pose corridor")
        return dict(position_error=position, angle_error=angle, corridor_ok=True)

    def record_action(self, action):
        action = point(action)
        self.branch.record_action(action)
        if self._tail and math.dist(action, self._tail[0]) <= self.config.state_tolerance:
            self._tail.pop(0)
            self._expected = self._predictions.pop(0)
            self._phases.pop(0)
            self._active = max(0, self._active-1)
        else:
            self._tail, self._predictions, self._phases = [], [], []
            self._active, self._expected = 0, None

    def _proposals(self, state, count):
        shape = self.branch.shape
        center = state.object_body.pose.position
        delta = tuple(g-x for g, x in zip(state.goal.position, center))
        desired = signed_turn(state.object_body.pose.angle, state.goal.angle, shape.symmetry_order)
        turn = 1 if desired >= 0 else -1
        contacts = []
        for i in range(len(rectangle_exterior_edges(shape.components))):
            edge = _world_edge(shape, state.object_body.pose, i)
            dx, dy = edge.b[0]-edge.a[0], edge.b[1]-edge.a[1]
            projected = max(.1, min(.9, sum((state.pusher_body.pose.position[k]-edge.a[k])*(dx, dy)[k]
                                        for k in (0, 1))/(dx*dx+dy*dy)))
            central = max(.1, min(.9, sum((center[k]-edge.a[k])*(dx, dy)[k]
                                        for k in (0, 1))/(dx*dx+dy*dy)))
            us = (.15, .35, .65, .85, projected, central) if self._precision else (.15, .35, .65, .85)
            for u in us:
                q, normal = edge.point(u), edge.normal
                staged = tuple(q[k]+17*normal[k] for k in (0, 1))
                if not direct_clear(staged, staged, shape, state.object_body.pose, 16.9):
                    continue
                inward = -normal[0], -normal[1]
                torque = torque_proxy(q, center, inward)/max(1., math.dist(q, center))
                translation = sum(delta[k]*inward[k] for k in (0, 1))/max(1., math.hypot(*delta))
                distance = math.dist(state.pusher_body.pose.position, staged)/512
                contacts.append((i, u, torque, translation, distance))
        if not contacts:
            raise ProposalRejected("No accessible exterior edges")
        proposals, used = [], set()
        for ordinal in range(count):
            family = ordinal % 4
            def priority(row):
                _, _, torque, translation, distance = row
                if self._precision:
                    return ((1.5*turn*torque+.2*translation if family == 0 else
                             turn*torque+.3*translation if family == 1 else
                             translation-.25*abs(torque) if family == 2 else
                             .8*turn*torque+.8*translation)
                            -( .8 if family == 0 else 2.)*distance)
                score = ((turn*torque if family in (0, 1) else
                          translation-.25*abs(torque) if family == 2 else
                          -turn*torque*.8+translation*.2)-.08*distance)
                return score
            ranked = sorted(contacts, key=priority, reverse=True)
            chosen = next((row for row in ranked if row[:2] not in used), ranked[ordinal % len(ranked)])
            used.add(chosen[:2])
            edge, u = chosen[:2]
            # Near the goal, fixed 45--85 px pushes overshoot the last few
            # degrees/pixels. Scale strokes by pose error while retaining broad
            # proposals far from alignment. No per-scene coordinates are used.
            radius = math.sqrt(sum(x*x+y*y for x, y in _material_points(shape))/len(_material_points(shape)))
            rotation_travel = max(2., min(110., radius*abs(desired)))
            travel = (max(2., rotation_travel*.65), min(140., rotation_travel*1.15),
                      max(2., min(100., math.hypot(*delta))), max(2., rotation_travel*.9))[family]
            if self._precision:
                travel = max(.5, min(18., travel*(.45 if ordinal < 4 else .8)))
            proposals.append(ContactStage(edge, u, 1. if self._precision else 2., travel,
                (0., .25*turn, 0., -.25*turn)[family], 180.,
                1 if ordinal % 2 == 0 else -1, (0., .25*turn, 0., -.25*turn)[family],
                approach_speed=220. if self._precision else 0.,
                sustain=3 if self._precision else 0, precision=self._precision))
            if self._precision:
                proposals[-1] = replace(proposals[-1], speed=35.)
        return proposals

    def _paired_proposals(self, state, first, count):
        """Compensate actual attained center error, with torque-aware alternatives."""
        choices = self._proposals(state, max(count, 8))
        if not self._precision or count < 2:
            return choices[:count]
        shape = self.branch.shape
        first_normal = _world_edge(shape, state.object_body.pose, first.edge).normal
        delta = tuple(g-x for g, x in zip(state.goal.position, state.object_body.pose.position))
        turn = signed_turn(state.object_body.pose.angle, state.goal.angle, shape.symmetry_order)
        length = math.hypot(*delta)
        for i in range(len(rectangle_exterior_edges(shape.components))):
            edge = _world_edge(shape, state.object_body.pose, i)
            dx,dy = edge.b[0]-edge.a[0],edge.b[1]-edge.a[1]
            center = state.object_body.pose.position
            central = max(.1,min(.9,sum((center[k]-edge.a[k])*(dx,dy)[k] for k in (0,1))/(dx*dx+dy*dy)))
            for u in (.2, central, .8):
                staged = tuple(edge.point(u)[k]+16*edge.normal[k] for k in (0, 1))
                if direct_clear(staged, staged, shape, state.object_body.pose, 15.9):
                    alignment = -sum(delta[k]*edge.normal[k] for k in (0,1))/max(1.,length)
                    choices.append(ContactStage(i, u, 1.,
                        max(.5, min(28., .8*length/max(.3,alignment))),
                        speed=35., approach_speed=220., sustain=3, precision=True))
        def pair_priority(stage):
            edge = _world_edge(shape, state.object_body.pose, stage.edge)
            inward = tuple(-x for x in edge.normal)
            q = edge.point(stage.u)
            rotation = torque_proxy(q, center, inward)/max(1.,math.dist(q,center))
            translation = sum(delta[k]*inward[k] for k in (0, 1))/max(1., math.hypot(*delta))
            return (3*translation-.6*abs(rotation)+.4*(1 if turn>=0 else -1)*rotation
                    -.2*math.dist(state.pusher_body.pose.position,q)/512)
        compensate = sorted(choices, key=pair_priority, reverse=True)
        result = [compensate[0]]
        opposite = next((s for s in compensate if sum(first_normal[k]*_world_edge(
            shape, state.object_body.pose, s.edge).normal[k] for k in (0, 1)) < -.5), None)
        if opposite is not None and opposite not in result:
            result.append(opposite)
        for stage in [choices[0], *compensate]:
            if len(result) >= count:
                break
            if stage not in result:
                result.append(stage)
        return result

    def _mutate(self, stages, peers=()):
        result = []
        for index, stage in enumerate(stages):
            members = [p[index] for p in peers if len(p) == len(stages)
                       and tuple(s.edge for s in p) == tuple(s.edge for s in stages)
                       and tuple(s.route for s in p) == tuple(s.route for s in stages)] or [stage]
            values = {}
            for name, low, high, floor in (("u", .1, .9, .08), ("clearance", .5, 8., 1.),
                    ("travel", .5 if stage.precision else 2., 140., 15.), ("yaw", -.7, .7, .15),
                    ("speed", 12. if stage.precision else 60., 80. if stage.precision else 220.,
                     8. if stage.precision else 20.), ("bend", -.6, .6, .15)):
                samples = [getattr(s, name) for s in members]
                mean = sum(samples)/len(samples)
                if name == "travel":
                    floor = max(.5 if stage.precision else 2., min(floor, .3*mean))
                sigma = max(floor, math.sqrt(sum((v-mean)**2 for v in samples)/len(samples)))
                values[name] = max(low, min(high, float(self.rng.normal(mean, sigma))))
            result.append(replace(stage, **values))
            if stage.precision:
                result[-1] = replace(result[-1], sustain=int(self.rng.integers(4)))
        return tuple(result)

    def _decision(self, report):
        commands = tuple(self._tail)
        next_phase = self._phases[0]
        block = min(2 if next_phase == "push" else 8, len(commands))
        # Never step through a phase boundary without checking the actual state.
        block = min(block, next((i for i, phase in enumerate(self._phases) if phase != next_phase), len(commands)))
        report.update(selected=self._selected, selected_index=self._selected["candidate_index"],
            selected_actions=[list(a) for a in commands], selected_predictions=[asdict(s) for s in self._predictions],
            execute_steps=block, global_searches=self.global_searches, config=asdict(self.config),
            policy_samples=False, demonstrations=False, training=False, privileged_physics=True,
            branch_protocol="initial_plus_real_history_plus_entire_hypothetical_prefix",
            search_method="guarded_joint_pose_contact_macros",
            precision_mode=self._precision, joint_limits=self._limits)
        return PlannerDecision(commands, report)

    def choose(self, current_state, remaining_steps, deadline=None):
        if self.closed or remaining_steps < 1:
            raise ValueError("Search closed or no remaining controls")
        if current_state.control_step != self.branch.initial_state.control_step+len(self.branch.executed_actions):
            raise ValueError("Record every real executed action before search")
        self.branch._check_deadline(deadline)
        if self._expected is not None and state_error(current_state, self._expected) > self.config.state_tolerance:
            raise ValueError("Selected sequence differs from actual state")
        began, before = time.monotonic(), dict(self.counts)
        error = pose_rms(self.branch.shape, current_state)
        self._observations.append((current_state.control_step, error))
        past = [v for step, v in self._observations if step <= current_state.control_step-16]
        stalled = (bool(past) and error >= past[-1]-.5 and self._predictions
                   and pose_rms(self.branch.shape, self._predictions[-1]) >= error-.5
                   and self._phases[0] == "push" and not self._selected["native_success"])
        if self._tail and self._active and not stalled:
            return self._decision({"continuation": True, "counts": {k: 0 for k in self.counts},
                "forecast_budget": len(self._tail), "forecast_hard_cap": self.config.horizon,
                "elapsed_seconds": time.monotonic()-began, "candidates": [], "evaluations": 0,
                "timed_out": False})
        if self.global_searches >= self.config.max_global_searches:
            raise SearchBudgetExhausted("Global search budget exhausted")
        self.global_searches += 1
        if self.rng is None:
            import numpy as np
            self.rng = np.random.default_rng(self.config.seed)
        budget = min(self.config.horizon, remaining_steps)
        self._precision = self._precision or error <= 25. or remaining_steps <= 40
        if self._precision:
            self._limits = joint_limits(current_state, self.branch.shape, self._limits)
        reserve_settle = settling_reserve(current_state, budget, self.config.settle_steps)
        active_budget = budget-reserve_settle
        records, completed, roots = [], [], []
        timed_out = False
        shape = self.branch.shape

        def evaluate(kind, stages=(), parent=None, saved=None):
            nonlocal timed_out
            if len(records) >= self.config.max_evaluations or timed_out:
                return None
            identifier, old_counts = len(records), dict(self.counts)
            record = {"candidate_index": identifier, "kind": kind,
                      "stages": [asdict(s) for s in stages], "status": "started"}
            try:
                self.branch._check_deadline(deadline)
                prepared = None
                prepared_index = len(parent["stage_objects"]) if parent is not None else 0
                if stages and saved is None:
                    geometry_state = parent["active_state"] if parent is not None else current_state
                    prefix_length = len(parent["active_commands"]) if parent is not None else 0
                    available = active_budget-prefix_length
                    reserve = min(12*(len(stages)-1), max(1, available//3)) if len(stages) > 1 and prepared_index == 0 else 0
                    prepared = stage_actions(shape, geometry_state, stages[prepared_index],
                        max(0, available-reserve), self.config.max_target_step)
                # Always reconstruct actual contact history; endpoint metadata is not a restore.
                replay_error = self.branch._replay(current_state, self.counts, deadline)
                commands, rows, predictions, phases, details = (), [], [], [], []
                prefix_state, prefix_commands, prefix_phases = None, (), ()
                if saved is not None:
                    active = min(self._active, active_budget, len(saved))
                    a, r, s = self.branch._forecast(saved, active, current_state, budget, self.counts, deadline)
                    commands, rows, predictions = a, list(r), list(s)
                    phases = (list(self._phases[:active])+["hold"]*budget)[:len(a)]
                elif stages:
                    if parent is not None:
                        # Sibling replay is hypothetical work, not additional real history.
                        a, r, s = self.branch._forecast(parent["active_commands"], len(parent["active_commands"]),
                            current_state, len(parent["active_commands"]), self.counts, deadline)
                        commands, rows, predictions = a, list(r), list(s)
                        phases = list(parent["active_phases"])
                        details = list(parent["stage_details"])
                        prefix_state, prefix_commands, prefix_phases = parent["prefix_state"], parent["prefix_commands"], parent["prefix_phases"]
                        if state_error(s[-1], parent["active_state"]) > self.config.state_tolerance:
                            raise ValueError("Hypothetical prefix replay mismatch")
                        start_index = prepared_index
                    else:
                        start_index = 0
                    for i in range(start_index, len(stages)):
                        if rows and rows[-1]["native_done"]:
                            break
                        state = predictions[-1] if predictions else current_state
                        available = active_budget-len(commands)
                        # Single-contact fallback uses the available budget.
                        # Only a true pair reserves controls for its second stage.
                        reserve = min(12*(len(stages)-i-1), max(1, available//3)) if i < len(stages)-1 else 0
                        actions, flags, extra = (prepared if i == prepared_index else
                            stage_actions(shape, state, stages[i], max(0, available-reserve), self.config.max_target_step))
                        a, r, s = self.branch._forecast(actions, len(actions), state, len(actions), self.counts, deadline)
                        commands += a
                        rows += list(r)
                        predictions += list(s)
                        phases += flags[:len(a)]
                        details.append(extra)
                        if i == 0:
                            prefix_state, prefix_commands, prefix_phases = s[-1], commands, tuple(phases)
                    active = len(commands)
                    if not rows[-1]["native_done"] and active < budget:
                        a, r, s = self.branch._forecast((), 0, predictions[-1], budget-active, self.counts, deadline)
                        commands += a
                        rows += list(r)
                        predictions += list(s)
                        phases += ["hold"]*len(a)
                else:
                    active = 0
                    a, r, s = self.branch._forecast((), 0, current_state, budget, self.counts, deadline)
                    commands, rows, predictions, phases = a, list(r), list(s), ["hold"]*len(a)
                tail = predictions[-min(3, len(predictions)):]
                unsafe = any(any(not 8 <= x <= 503 for poly in shape.polygons(s.object_body.pose) for p in poly for x in p)
                             or any(not 23 <= x <= 488 for x in s.pusher_body.pose.position) for s in predictions)
                rms = max(pose_rms(shape, s) for s in tail)
                # Express the modest access/time penalty in the same pixel units
                # as pose RMS. It is a disclosed heuristic, not a completion bound.
                access = correction_distance(shape, predictions[-1]) if self._precision else 0.
                radius = math.sqrt(sum(x*x+y*y for x, y in _material_points(shape))/len(_material_points(shape)))
                angle = shape.angle_error(predictions[-1].object_body.pose.angle, current_state.goal.angle)
                estimated_remaining = (access/22+rows[-1]["position_error"]/3.5+radius*angle/3.5+3
                                       if self._precision else 0.)
                pressure = min(3., 40/max(8, remaining_steps))
                cost = (pressure*(.02*phases.count("approach")+.005*active+.02*access/22
                                 +.05*max(0., estimated_remaining-(remaining_steps-active)))
                        if self._precision else 0.)
                record.update(status="completed", replay_state_error=replay_error, native_success=rows[-1]["native_done"],
                    unsafe=unsafe, pose_rms=rms, completion_score=rms+cost, completion_penalty=cost,
                    estimated_remaining_controls=estimated_remaining, next_contact_distance=access,
                    score=-(rms+cost), final_coverage=rows[-1]["coverage"],
                    final_iou=rows[-1]["iou"], final_position_error=rows[-1]["position_error"],
                    final_angle_error=rows[-1]["angle_error"], peak_coverage=max(r["coverage"] for r in rows),
                    forecast_steps=len(rows), policy_prefix=active, stage_details=details,
                    approach_controls=phases.count("approach"), contact_controls=sum(r["n_contacts"] > 0 for r in rows),
                    approach_contact_controls=sum(r["n_contacts"] > 0 for r, phase in zip(rows, phases) if phase == "approach"),
                    push_contact_controls=sum(r["n_contacts"] > 0 for r, phase in zip(rows, phases) if phase == "push"),
                    motion=sum(r["speed"]/512+abs(r["angular_speed"]) for r in rows[-3:]),
                    prefix_rms=pose_rms(shape, prefix_state) if prefix_state else None,
                    prefix_position_error=math.dist(prefix_state.object_body.pose.position, current_state.goal.position) if prefix_state and self._precision else None,
                    prefix_angle_error=shape.angle_error(prefix_state.object_body.pose.angle, current_state.goal.angle) if prefix_state else None,
                    diversity_key=(stages[0].edge, 1 if stages[0].u >= .5 else -1, stages[0].route) if stages else None)
                if self._precision:
                    record.update(joint_errors(predictions, shape, self._limits))
                    record["balanced_score"] += .02*cost
                    record["actual_compensation"] = bool(len(stages)>1 and prefix_state is not None
                        and record["final_position_error"] < record["prefix_position_error"]-.1)
                    record["joint_limits"] = dict(self._limits)
                internal = dict(record, commands=commands, predictions=predictions, rows=rows, phases=phases,
                    prefix_state=prefix_state, prefix_commands=prefix_commands, prefix_phases=prefix_phases,
                    stage_objects=tuple(stages), active=active,
                    active_commands=commands[:active], active_phases=phases[:active],
                    active_state=predictions[active-1] if active else current_state)
                self.branch._check_deadline(deadline)
                completed.append(internal)
                return internal
            except ProposalRejected as exc:
                record.update(status="rejected", reason=str(exc))
            except TimeoutError:
                timed_out = True
                record.update(status="timed_out")
            finally:
                record["counts"] = {k: self.counts[k]-old_counts[k] for k in self.counts}
                records.append(record)
            return None

        if self._tail:
            evaluate("incumbent", saved=tuple(self._tail))
        evaluate("hold")
        try:
            proposals = self._proposals(current_state, self.config.roots)
            # iCEM-style memory: retain a minority, not the entire old population.
            memory = [old for old in self._memory if old[0].precision == self._precision]
            for i, old in enumerate(memory[:min(1 if self._precision else 2, len(proposals))]):
                proposals[i] = self._mutate((old[0],))[0]
            for spec in proposals:
                result = evaluate("first_contact", (spec,))
                if result is not None and result["prefix_state"] is not None:
                    roots.append(result)
            for parent in diverse_prefixes(roots, self.config.beam_width):
                if timed_out or parent["native_success"]:
                    continue
                try:
                    children = self._paired_proposals(parent["prefix_state"], parent["stage_objects"][0], self.config.children)
                except ProposalRejected:
                    # Count a failed complete proposal through the same evaluator.
                    children = [parent["stage_objects"][0]]
                for second in children:
                    evaluate("second_contact", (*parent["stage_objects"], second), parent=parent)
            refinement_count = self.config.refinements//2 if self._precision else self.config.refinements
            for i in range(refinement_count):
                pairs = sorted((r for r in completed if len(r["stage_objects"]) == 2 and not r["unsafe"]), key=path_key)
                pool = pairs or sorted((r for r in completed if r["stage_objects"] and not r["unsafe"]), key=path_key)
                if not pool or timed_out:
                    break
                bank = family_frontier(completed) if self._precision else pool[:2]
                base = bank[i % len(bank)]
                peers = [r["stage_objects"] for r in pool[:4]]
                evaluate("joint_refinement", self._mutate(base["stage_objects"], peers))
            if self._precision:
                # Press duration is a searched action parameter, not three
                # mandatory future pushes. Every variant consumes an attempt.
                bank = family_frontier(completed)
                for i in range(self.config.refinements-refinement_count):
                    if not bank or timed_out:
                        break
                    base = bank[i % len(bank)]
                    evaluate("press_duration", tuple(replace(s, sustain=i % 4) for s in base["stage_objects"]))
                # Spend at most four spare slots on a third contact, only when
                # two-stage candidates do not already attain both planning scales.
                pairs = [r for r in completed if len(r["stage_objects"])==2 and not r["unsafe"]]
                if not any(r["native_success"] or r.get("joint_feasible",False)
                           and r.get("balanced_score",math.inf)<=1 for r in pairs):
                    for parent in family_frontier(pairs)[:2]:
                        if parent["native_success"] or parent["active"] >= active_budget-1:
                            continue
                        for third in self._paired_proposals(parent["active_state"],parent["stage_objects"][-1],2):
                            evaluate("third_contact",(*parent["stage_objects"],third),parent=parent)
            # Use spare attempts for fresh sequence exploration, preserving the hard cap.
            for _ in range(self.config.max_evaluations-len(records)):
                if timed_out:
                    break
                spec = self._mutate((proposals[int(self.rng.integers(len(proposals)))],))[0]
                evaluate("exploration", (spec,))
        except ProposalRejected as exc:
            if len(records) < self.config.max_evaluations:
                records.append({"candidate_index": len(records), "kind": "proposal_generation", "status": "rejected",
                                "reason": str(exc), "counts": {k: 0 for k in self.counts}})
        if not completed:
            raise TimeoutError("No complete sequence candidate before deadline")
        safe = [r for r in completed if not r["unsafe"]]
        if self._precision:
            safe = [r for r in safe if r["native_success"] or r["joint_feasible"]]
        if not safe:
            raise SearchBudgetExhausted("No jointly feasible completed plan" if self._precision else "No collision-safe completed plan")
        best = min(safe, key=path_key)
        elites = family_frontier(completed) if self._precision else sorted(safe, key=path_key)
        self._memory = [r["stage_objects"] for r in elites if r["stage_objects"]][:4]
        self._tail, self._predictions, self._phases = list(best["commands"]), list(best["predictions"]), list(best["phases"])
        self._active = len(self._tail) if best["native_success"] else best["active"]
        self._selected = dict(records[best["candidate_index"]])
        self._selected["first_state"] = asdict(self._predictions[0])
        report = {"continuation": False, "counts": {k: self.counts[k]-before[k] for k in self.counts},
                  "candidates": records, "evaluations": len(records), "completed_candidates": len(completed),
                  "forecast_budget": budget, "forecast_hard_cap": self.config.horizon,
                  "settling_reserve": reserve_settle, "active_budget": active_budget,
                  "retained_prefix_indices": [r["candidate_index"] for r in diverse_prefixes(roots, self.config.beam_width)],
                  "timed_out": timed_out, "stalled_trigger": bool(stalled),
                  "elapsed_seconds": time.monotonic()-began}
        return self._decision(report)

    def close(self):
        if not self.closed:
            self.branch.close()
            self.closed = True
