"""Policy-free, bounded simulator search over geometric pushing primitives.

Known state, shape and dynamics are privileged inputs. No policy, demonstrations,
world model or training is used. NumPy is loaded only when search first runs.
"""

from dataclasses import asdict, dataclass
import math
import time

from .geometry import finite, point
from .simulator_planning import PlannerConfig, PlannerDecision, SimulatorReranker


@dataclass(frozen=True)
class SearchConfig:
    horizon: int = 64
    population: int = 12
    generations: int = 3
    elite_count: int = 3
    max_evaluations: int = 40
    execute_steps: int = 8
    settle_steps: int = 4
    seed: int = 42
    exploration_fraction: float = .25
    position_weight: float = .8
    angle_weight: float = .25
    state_tolerance: float = 1e-7
    max_target_step: float = 32.

    def __post_init__(self):
        for name, low, high in (("horizon", 4, 80), ("population", 2, 128),
                                ("generations", 1, 20), ("elite_count", 1, self.population),
                                ("max_evaluations", 1, 1024), ("execute_steps", 1, 8),
                                ("settle_steps", 0, 16), ("seed", 0, 2**32-1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name in ("position_weight", "angle_weight", "state_tolerance", "max_target_step"):
            value = finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        fraction = finite(self.exploration_fraction, "exploration_fraction")
        if not 0 < fraction <= 1:
            raise ValueError("exploration_fraction must be in (0, 1]")
        object.__setattr__(self, "exploration_fraction", fraction)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _boundary_route(start, finish, bounds, direction):
    """Route around an expanded axis-aligned box in either direction.

    The connectors may still touch a concave object; actual simulator contact and
    wall diagnostics determine the score. No waypoint is coordinate-clipped.
    """
    x0, y0, x1, y1 = bounds
    width, height = x1-x0, y1-y0
    perimeter = 2*(width+height)

    def project(p):
        x, y = p
        candidates = ((min(x1, max(x0, x)), y0), (x1, min(y1, max(y0, y))),
                      (min(x1, max(x0, x)), y1), (x0, min(y1, max(y0, y))))
        return min(candidates, key=lambda q: math.dist(p, q))

    def coordinate(p):
        x, y = p
        if y == y0:
            return x-x0
        if x == x1:
            return width+y-y0
        if y == y1:
            return width+height+x1-x
        return 2*width+height+y1-y

    entry, exit = project(start), project(finish)
    a, b = coordinate(entry), coordinate(exit)
    distance = ((b-a)*direction) % perimeter
    corners = [((x0, y0), 0.), ((x1, y0), width),
               ((x1, y1), width+height), ((x0, y1), 2*width+height)]
    middle = sorted((((s-a)*direction % perimeter, p) for p, s in corners), key=lambda row: row[0])
    return [start, entry, *(p for s, p in middle if 1e-9 < s < distance-1e-9), exit, finish]


def _resample(path, controls):
    segments = [(a, b, math.dist(a, b)) for a, b in zip(path, path[1:]) if math.dist(a, b) > 1e-9]
    if not segments:
        return [point(path[-1])]*controls
    total = sum(length for _, _, length in segments)
    result, segment, offset = [], 0, 0.
    for index in range(controls):
        distance = total*(index+1)/controls
        while segment < len(segments)-1 and distance > offset+segments[segment][2]:
            offset += segments[segment][2]
            segment += 1
        a, b, length = segments[segment]
        t = min(1., (distance-offset)/length)
        result.append(tuple(x+t*(y-x) for x, y in zip(a, b)))
    return result


def primitive_actions(shape, state, parameters, active_controls, max_target_step=32.):
    """Generate approach/contact/push commands from one low-dimensional sample.

    Parameters: contact angle relative to the rear of the goal translation,
    tangent offset (pixels), push yaw, distance scale, clearance beyond radius15,
    approach duration fraction, active duration fraction, and route direction.
    """
    angle, offset, yaw, scale, clearance, fraction, _, route = parameters
    center = state.object_body.pose.position
    displacement = tuple(g-x for g, x in zip(state.goal.position, center))
    goal_angle = math.atan2(displacement[1], displacement[0])
    inward_angle = goal_angle+angle
    inward = math.cos(inward_angle), math.sin(inward_angle)
    tangent = -inward[1], inward[0]
    vertices = [v for polygon in shape.polygons(state.object_body.pose) for v in polygon]
    support = min(sum((v[i]-center[i])*inward[i] for i in (0, 1)) for v in vertices)
    behind = tuple(center[i]+inward[i]*(support-15-clearance)+tangent[i]*offset for i in (0, 1))
    bounds = (min(v[0] for v in vertices)-15-clearance,
              min(v[1] for v in vertices)-15-clearance,
              max(v[0] for v in vertices)+15+clearance,
              max(v[1] for v in vertices)+15+clearance)
    start = state.pusher_body.pose.position
    if active_controls == 1:
        actions, approach = [start], 0
    else:
        approach = max(1, min(active_controls-1, round(active_controls*fraction)))
        path = _boundary_route(start, behind, bounds, 1 if route >= 0 else -1)
        actions = _resample(path, approach)
        heading = inward_angle+yaw
        distance = min(200., (math.hypot(*displacement)+clearance)*scale)
        finish = (behind[0]+distance*math.cos(heading), behind[1]+distance*math.sin(heading))
        actions += _resample([behind, finish], active_controls-approach)
    if any(not 25 <= v <= 486 for command in actions for v in command):
        raise ValueError("Primitive targets leave the collision-clear [25,486] pusher workspace")
    if any(math.dist(a, b) > max_target_step for a, b in zip([start, *actions], actions)):
        raise ValueError("Primitive exceeds the configured target displacement per control")
    return tuple(point(a) for a in actions), {"approach_controls": approach,
                                              "behind_target": behind, "route_direction": 1 if route >= 0 else -1}


def _search_score(rows, predictions, shape, config, approach_controls):
    tail = rows[-min(3, len(rows)):]
    coverage = min(r["coverage"] for r in tail)
    # Coverage already measures near-goal translation and angle alignment. Pose
    # shaping fades continuously rather than overriding a better near-goal fit.
    pose_factor = (1-coverage)**2
    position = sum(r["position_error"] for r in tail)/len(tail)
    angle = sum(r["angle_error"] for r in tail)/len(tail)
    motion = sum(r["speed"]/512+abs(r["angular_speed"]) for r in tail)/len(tail)
    unsafe = 0
    for state in predictions:
        vertices = [v for polygon in shape.polygons(state.object_body.pose) for v in polygon]
        unsafe += any(not 8 <= x <= 503 for v in vertices for x in v)
        unsafe += any(not 23 <= x <= 488 for x in state.pusher_body.pose.position)
    early_contacts = sum(r["n_contacts"] > 0 for r in rows[:approach_controls])
    penalty = 2.*unsafe/max(1, len(rows)) + .02*early_contacts/max(1, approach_controls)
    score = (2. if rows[-1]["native_done"] else
             coverage-pose_factor*(config.position_weight*position/512+config.angle_weight*angle/math.pi)-.005*motion)
    return score-penalty, {"terminal_min_coverage": coverage, "unsafe_state_count": unsafe,
                           "approach_contact_controls": early_contacts, "pose_shaping_factor": pose_factor}


class SimulatorActionSearch:
    """Cross-entropy refinement of independent geometric simulator candidates."""

    def __init__(self, env_factory, initial_state, config=None):
        self.config = config or SearchConfig()
        if not isinstance(self.config, SearchConfig):
            raise ValueError("Expected SearchConfig")
        self.branch = SimulatorReranker(env_factory, initial_state,
                                       PlannerConfig(horizon=self.config.horizon, prefix_lengths=(self.config.horizon,),
                                                     settle_steps=self.config.settle_steps, include_hold=False,
                                                     state_tolerance=self.config.state_tolerance))
        self.counts = self.branch.counts
        self.rng = None
        self._tail = []
        self._active_remaining = 0
        self.closed = False

    def record_action(self, action):
        action = point(action)
        self.branch.record_action(action)
        if self._tail and math.dist(action, self._tail[0]) <= self.config.state_tolerance:
            self._tail.pop(0)
            self._active_remaining = max(0, self._active_remaining-1)
        else:
            self._tail, self._active_remaining = [], 0

    def _random(self, means=None, deviations=None):
        limits = ((-math.pi, math.pi), (-55., 55.), (-1.2, 1.2), (.25, 2.25),
                  (15., 40.), (.35, .8), (.6, 1.))
        values = []
        for index, (low, high) in enumerate(limits):
            value = self.rng.uniform(low, high) if means is None else self.rng.normal(means[index], deviations[index])
            values.append(_wrap(value) if index == 0 else max(low, min(high, float(value))))
        route = int(self.rng.choice((-1, 1)))
        return (*values, route)

    def _seed(self, index, angle_error):
        # Diverse sides include both goal-directed translation and torque-producing
        # off-center contacts. They are generic geometric seeds, not a known D path.
        sides = (0., 0., math.pi/2, -math.pi/2, math.pi, math.pi/4, -math.pi/4, 3*math.pi/4)
        turn = 1 if angle_error >= 0 else -1
        offsets = (0., -35.*turn, 35.*turn, 0.)
        return (sides[index % len(sides)], offsets[index % 4],
                (0., -.5, .5, 0.)[index % 4], (1., 1.3, .8)[index % 3],
                20., .625, 1., 1 if index % 2 == 0 else -1)

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
        maximum_active = min(self.config.horizon, max(1, budget-min(self.config.settle_steps, budget-1)))
        records, completed, best, timed_out = [], [], None, False
        attempted = 0

        def evaluate(kind, parameters, generation, actions=None, active=None):
            nonlocal attempted, best, timed_out
            if attempted >= self.config.max_evaluations:
                return False
            self.branch._check_deadline(deadline)
            attempted += 1
            identifier, before = len(records), dict(self.counts)
            record = {"candidate_index": identifier, "kind": kind, "generation": generation,
                      "parameters": list(parameters) if parameters is not None else None}
            try:
                extra = {"approach_controls": 0}
                if kind == "primitive":
                    active = max(1, min(maximum_active, round(maximum_active*parameters[6])))
                    actions, extra = primitive_actions(self.branch.shape, current_state, parameters, active,
                                                        self.config.max_target_step)
                elif kind == "hold":
                    actions, active = (), 0
                replay_error = self.branch._replay(current_state, self.counts, deadline)
                commands, rows, predictions = self.branch._forecast(actions, active, current_state, budget,
                                                                    self.counts, deadline)
                score, diagnostics = _search_score(rows, predictions, self.branch.shape, self.config, extra["approach_controls"])
                self.branch._check_deadline(deadline)
                record.update(status="completed", score=score, policy_prefix=active, forecast_steps=len(rows),
                              replay_state_error=replay_error, final_coverage=rows[-1]["coverage"],
                              final_iou=rows[-1]["iou"], peak_coverage=max(r["coverage"] for r in rows),
                              native_success=rows[-1]["native_done"], final_position_error=rows[-1]["position_error"],
                              final_angle_error=rows[-1]["angle_error"], contact_controls=sum(r["n_contacts"] > 0 for r in rows),
                              **extra, **diagnostics)
                completed.append(record)
                if best is None or score > best[0]:
                    best = (score, identifier, commands, predictions, active)
            except ValueError as exc:
                # A failed history reconstruction is a protocol error, not an
                # unattractive search proposal: never hide it among rejections.
                if self.counts["restores"] != before["restores"]:
                    raise
                record.update(status="rejected", reason=str(exc))
            except TimeoutError:
                timed_out = True
                record.update(status="timed_out")
            finally:
                record["counts"] = {name: value-before[name] for name, value in self.counts.items()}
                records.append(record)
            return not timed_out

        try:
            if self._tail and self._active_remaining:
                evaluate("incumbent_tail", None, -1, tuple(self._tail),
                         min(self._active_remaining, maximum_active, len(self._tail)))
            if not timed_out and attempted < self.config.max_evaluations:
                evaluate("hold", None, -1)
            means = deviations = None
            angle_error = _wrap(current_state.goal.angle-current_state.object_body.pose.angle)
            for generation in range(self.config.generations):
                if timed_out or attempted >= self.config.max_evaluations:
                    break
                for index in range(self.config.population):
                    if attempted >= self.config.max_evaluations or timed_out:
                        break
                    parameters = (self._seed(index, angle_error) if generation == 0 else
                                  self._random() if self.rng.uniform(0., 1.) < self.config.exploration_fraction else
                                  self._random(means, deviations))
                    evaluate("primitive", parameters, generation)
                elites = sorted((r for r in completed if r["parameters"] is not None),
                                key=lambda r: r["score"], reverse=True)[:self.config.elite_count]
                if elites:
                    values = [r["parameters"] for r in elites]
                    means = [sum(p[i] for p in values)/len(values) for i in range(7)]
                    means[0] = math.atan2(sum(math.sin(p[0]) for p in values), sum(math.cos(p[0]) for p in values))
                    floors = (.18, 8., .15, .15, 3., .06, .08)
                    deviations = [max(floors[i], math.sqrt(sum((_wrap(p[i]-means[i]) if i == 0 else p[i]-means[i])**2 for p in values)/len(values))) for i in range(7)]
        except TimeoutError:
            timed_out = True
        if best is None:
            raise TimeoutError("No complete simulator-search candidate before the deadline")
        _, selected_index, commands, predictions, active = best
        selected = records[selected_index]
        selected["first_state"] = asdict(predictions[0])
        self._tail, self._active_remaining = list(commands), active
        return PlannerDecision(commands, {
            "config": asdict(self.config), "counts": {name: value-counts_before[name] for name, value in self.counts.items()},
            "selected": selected, "selected_index": selected_index, "candidates": records,
            "selected_predictions": [asdict(s) for s in predictions], "selected_actions": [list(a) for a in commands],
            "timed_out": timed_out, "forecast_budget": budget, "forecast_hard_cap": 80,
            "completed_candidates": len(completed),
            "requested_candidates": attempted, "evaluations": attempted,
            "generations_completed": len({r["generation"] for r in records if r["generation"] >= 0 and r["status"] == "completed"}),
            "rejected_candidates": [r for r in records if r["status"] == "rejected"],
            "elapsed_seconds": time.monotonic()-began, "execute_steps": min(self.config.execute_steps, len(commands)),
            "branch_protocol": "initial_snapshot_plus_all_executed_actions", "settling_uses_real_budget": True,
            "policy_samples": False, "demonstrations": False, "training": False, "privileged_physics": True,
            "search_method": "geometric_seeds_then_elite_CEM_with_uniform_exploration"})

    def close(self):
        if not self.closed:
            self.branch.close()
            self.closed = True
