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

    def __post_init__(self):
        if type(self.edge) is not int or self.edge < 0 or self.route not in (-1, 1):
            raise ValueError("Invalid categorical contact/route")
        for name, lo, hi in (("u", .1, .9), ("clearance", .5, 8.), ("travel", 2., 140.),
                             ("yaw", -.7, .7), ("speed", 60., 220.), ("bend", -.6, .6)):
            if not lo <= finite(getattr(self, name), name) <= hi:
                raise ValueError(f"Invalid stage {name}")


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
            len(record["rows"]) if record["native_success"] else record["pose_rms"],
            -record["final_coverage"], record["motion"], record["approach_controls"])


def diverse_prefixes(records, width):
    """Protect orientation progress and different contacts before leaf ranking."""
    records = [r for r in records if r.get("prefix_state") is not None and not r["unsafe"]]
    if not records:
        return []
    ordered = sorted(records, key=path_key)
    result = [ordered[0]]
    if width > 1:
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
        nearest.append((math.dist(start, q), world.normal))
    distance, normal = min(nearest)
    release = start
    if distance < clearance+.2:
        release = tuple(start[k]+normal[k]*(clearance+1-distance) for k in (0, 1))
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
    path, route_mode = approach_path(shape, state, staged, stage.route,
                                      min(16., 15+stage.clearance))
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    approach = math.ceil(length/(stage.speed*.1))
    inward = -normal[0], -normal[1]
    heading = math.atan2(inward[1], inward[0])+stage.yaw
    distance = (stage.travel+stage.clearance)/max(.3, math.cos(stage.yaw))
    middle = tuple(staged[k]+distance*.5*(math.cos(heading), math.sin(heading))[k] for k in (0, 1))
    finish = tuple(middle[k]+distance*.5*(math.cos(heading+stage.bend), math.sin(heading+stage.bend))[k] for k in (0, 1))
    pushes = max(1, math.ceil(distance/(stage.speed*.1)))
    if approach+pushes > active_budget:
        raise ProposalRejected("Approach and actual push exceed stage budget")
    commands = (_resample(path, approach) if approach else [])+_resample([staged, middle, finish], pushes)
    previous = state.pusher_target
    if (any(not 25 <= x <= 486 for a in commands for x in a)
            or any(math.dist(a, b) > max_target_step+1e-9 for a, b in zip([state.pusher_body.pose.position, *commands], commands))
            or previous is not None and math.dist(previous, commands[0]) > max_target_step+1e-9):
        raise ProposalRejected("Unsafe target or discontinuous PD target")
    phases = ["approach"]*approach+["push"]*pushes
    return tuple(map(point, commands)), phases, {"approach_controls": approach, "push_controls": pushes,
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
            for u in (.15, .35, .65, .85):
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
                score = ((turn*torque if family in (0, 1) else
                          translation-.25*abs(torque) if family == 2 else
                          -turn*torque*.8+translation*.2)-.08*distance)
                return score
            ranked = sorted(contacts, key=priority, reverse=True)
            chosen = next((row for row in ranked if row[:2] not in used), ranked[ordinal % len(ranked)])
            used.add(chosen[:2])
            edge, u = chosen[:2]
            travel = (45., 85., max(12., min(100., math.hypot(*delta))), 65.)[family]
            proposals.append(ContactStage(edge, u, 2., travel,
                (0., .25*turn, 0., -.25*turn)[family], 180.,
                1 if ordinal % 2 == 0 else -1, (0., .25*turn, 0., -.25*turn)[family]))
        return proposals

    def _mutate(self, stages, peers=()):
        result = []
        for index, stage in enumerate(stages):
            members = [p[index] for p in peers if len(p) == len(stages)
                       and tuple(s.edge for s in p) == tuple(s.edge for s in stages)
                       and tuple(s.route for s in p) == tuple(s.route for s in stages)] or [stage]
            values = {}
            for name, low, high, floor in (("u", .1, .9, .08), ("clearance", .5, 8., 1.),
                    ("travel", 2., 140., 15.), ("yaw", -.7, .7, .15),
                    ("speed", 60., 220., 20.), ("bend", -.6, .6, .15)):
                samples = [getattr(s, name) for s in members]
                mean = sum(samples)/len(samples)
                sigma = max(floor, math.sqrt(sum((v-mean)**2 for v in samples)/len(samples)))
                values[name] = max(low, min(high, float(self.rng.normal(mean, sigma))))
            result.append(replace(stage, **values))
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
            search_method="two_contact_diverse_beam_joint_pose_elite_memory")
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
        active_budget = max(1, budget-min(self.config.settle_steps, budget-1))
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
                        a, r, s = self.branch._forecast(parent["prefix_commands"], len(parent["prefix_commands"]),
                            current_state, len(parent["prefix_commands"]), self.counts, deadline)
                        commands, rows, predictions = a, list(r), list(s)
                        phases = list(parent["prefix_phases"])
                        details = list(parent["stage_details"][:1])
                        if state_error(s[-1], parent["prefix_state"]) > self.config.state_tolerance:
                            raise ValueError("Hypothetical prefix replay mismatch")
                        start_index = 1
                    else:
                        start_index = 0
                    for i in range(start_index, len(stages)):
                        if rows and rows[-1]["native_done"]:
                            break
                        state = predictions[-1] if predictions else current_state
                        available = active_budget-len(commands)
                        # First stage cannot consume the whole two-contact horizon.
                        limit = min(available, max(1, active_budget-12)) if i == 0 else available
                        actions, flags, extra = stage_actions(shape, state, stages[i], limit, self.config.max_target_step)
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
                record.update(status="completed", replay_state_error=replay_error, native_success=rows[-1]["native_done"],
                    unsafe=unsafe, pose_rms=rms, score=-rms, final_coverage=rows[-1]["coverage"],
                    final_iou=rows[-1]["iou"], final_position_error=rows[-1]["position_error"],
                    final_angle_error=rows[-1]["angle_error"], peak_coverage=max(r["coverage"] for r in rows),
                    forecast_steps=len(rows), policy_prefix=active, stage_details=details,
                    approach_controls=phases.count("approach"), contact_controls=sum(r["n_contacts"] > 0 for r in rows),
                    motion=sum(r["speed"]/512+abs(r["angular_speed"]) for r in rows[-3:]),
                    prefix_rms=pose_rms(shape, prefix_state) if prefix_state else None,
                    prefix_angle_error=shape.angle_error(prefix_state.object_body.pose.angle, current_state.goal.angle) if prefix_state else None,
                    diversity_key=(stages[0].edge, 1 if stages[0].u >= .5 else -1, stages[0].route) if stages else None)
                internal = dict(record, commands=commands, predictions=predictions, rows=rows, phases=phases,
                    prefix_state=prefix_state, prefix_commands=prefix_commands, prefix_phases=prefix_phases,
                    stage_objects=tuple(stages), active=active)
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
            for i, old in enumerate(self._memory[:min(2, len(proposals))]):
                proposals[i] = self._mutate((old[0],))[0]
            for spec in proposals:
                result = evaluate("first_contact", (spec,))
                if result is not None and result["prefix_state"] is not None:
                    roots.append(result)
            for parent in diverse_prefixes(roots, self.config.beam_width):
                if timed_out or parent["native_success"]:
                    continue
                try:
                    children = self._proposals(parent["prefix_state"], self.config.children)
                except ProposalRejected:
                    # Count a failed complete proposal through the same evaluator.
                    children = [parent["stage_objects"][0]]
                for second in children:
                    evaluate("second_contact", (*parent["stage_objects"], second), parent=parent)
            for i in range(self.config.refinements):
                pairs = sorted((r for r in completed if len(r["stage_objects"]) == 2 and not r["unsafe"]), key=path_key)
                pool = pairs or sorted((r for r in completed if r["stage_objects"] and not r["unsafe"]), key=path_key)
                if not pool or timed_out:
                    break
                base = pool[i % min(2, len(pool))]
                peers = [r["stage_objects"] for r in pool[:4]]
                evaluate("joint_refinement", self._mutate(base["stage_objects"], peers))
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
        if not safe:
            raise SearchBudgetExhausted("No collision-safe completed plan")
        best = min(safe, key=path_key)
        self._memory = [r["stage_objects"] for r in sorted(safe, key=path_key) if r["stage_objects"]][:4]
        self._tail, self._predictions, self._phases = list(best["commands"]), list(best["predictions"]), list(best["phases"])
        self._active = len(self._tail) if best["native_success"] else best["active"]
        self._selected = dict(records[best["candidate_index"]])
        self._selected["first_state"] = asdict(self._predictions[0])
        report = {"continuation": False, "counts": {k: self.counts[k]-before[k] for k in self.counts},
                  "candidates": records, "evaluations": len(records), "completed_candidates": len(completed),
                  "forecast_budget": budget, "forecast_hard_cap": self.config.horizon,
                  "retained_prefix_indices": [r["candidate_index"] for r in diverse_prefixes(roots, self.config.beam_width)],
                  "timed_out": timed_out, "stalled_trigger": bool(stalled),
                  "elapsed_seconds": time.monotonic()-began}
        return self._decision(report)

    def close(self):
        if not self.closed:
            self.branch.close()
            self.closed = True
