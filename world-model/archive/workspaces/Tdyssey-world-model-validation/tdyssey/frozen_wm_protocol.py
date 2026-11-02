"""Fixed diagnostic construction and score semantics; no simulator imports."""

import math
from .geometry import Pose

CANDIDATES = ("hold", "toward", "side", "away")
NOISE_SEEDS = (5478, 5479, 5480)


def goal_for(shape):
    return Pose((256, 256), math.pi / 4) if shape == "T" else Pose(
        (256 - 45 / math.sqrt(2), 256 + 45 / math.sqrt(2)), math.pi / 4)


def anchors():
    rows = []
    def add(shape, family, position, degrees, pusher, direction):
        index = len(rows)
        pose = Pose(position, math.radians(degrees))
        rotated = Pose((0, 0), pose.angle).transform(direction)
        rows.append(dict(id=f"{shape}{index:02d}", shape=shape, family=family,
                         seed=700000 + index, pose=pose, pusher=pose.transform(pusher),
                         direction=rotated, goal=goal_for(shape)))
    for angle in (-15, 0, 15):
        add("L", "noncontact", (240, 245), angle, (-125, 0), (1, 0))
    for angle in (-15, 0, 15):
        add("L", "translation", (240, 245), angle, (-50, 0), (1, 0))
    add("L", "rotation_cw", (240, 245), 0, (-50, -50), (1, 0))
    add("L", "rotation_ccw", (240, 245), 0, (-50, 30), (1, 0))
    add("L", "rotation_ccw", (240, 245), 15, (20, 65), (0, -1))
    for angle in (44, 45, 46):
        position = Pose(goal_for("L").position, math.radians(angle)).transform((-3, 0))
        add("L", "near_goal", position, angle, (-50, 0), (1, 0))
    add("T", "noncontact", (250, 210), 0, (-145, 15), (1, 0))
    add("T", "translation", (250, 210), 0, (0, 140), (0, -1))
    add("T", "rotation", (250, 210), 0, (-35, -20), (0, 1))
    add("T", "near_goal", goal_for("T").transform((-3, 0)), 45, (0, 140), (0, -1))
    return rows


def calibration_states():
    result = []
    for shape in ("L", "T"):
        for k in range(12):
            if k < 8:
                pose = Pose((170 + 80 * (k % 3),
                             170 + 60 * ((k // 3) % 3) - (45 if shape == "T" else 0)),
                            math.radians(30 * k))
                pusher = (60, 60)
            else:
                pose = Pose(goal_for(shape).transform(((-4, -2, 2, 4)[k - 8], 0)), math.pi / 4)
                pusher = pose.transform((-50, 0) if shape == "L" else (0, 140))
            result.append(dict(id=f"cal-{shape}-{k:02d}", shape=shape, seed=710000 + len(result),
                               pose=pose, pusher=pusher, goal=goal_for(shape)))
    return result


def candidate_actions(anchor, name):
    if name not in CANDIDATES:
        raise ValueError("Unknown fixed candidate")
    dx, dy = anchor["direction"]
    direction = {"hold": (0, 0), "toward": (dx, dy), "side": (-dy, dx), "away": (-dx, -dy)}[name]
    return [[anchor["pusher"][i] + 12 * h * direction[i] for i in (0, 1)] for h in range(1, 5)]


def wrapped_degrees(angle):
    return math.degrees(math.atan2(math.sin(angle), math.cos(angle)))


def predicted_scores(decodes):
    """Predicted stopping only; a missing observation cannot hide a crossing."""
    scores, terminal, invalid = [], None, False
    for value in decodes:
        if terminal is not None:
            scores.append(terminal)
        elif invalid or not value["valid"] or value.get("coverage") is None:
            invalid = True
            scores.append(None)
        else:
            score = value["coverage"]
            scores.append(score)
            if score > .95:
                terminal = score
    return scores


def rank_scores(actual, predicted):
    """All candidates/noises required; invalid pairs stay in gate denominators."""
    if len(actual) != 4 or len(predicted) != 4:
        raise ValueError("Exactly four fixed candidates required")
    eligible = correct = 0
    for i in range(4):
        for j in range(i + 1, 4):
            difference = actual[i] - actual[j]
            if abs(difference) + 1e-12 < .05:
                continue
            eligible += 1
            if predicted[i] is not None and predicted[j] is not None:
                pdiff = predicted[i] - predicted[j]
                correct += int(abs(pdiff) > 1e-6 and difference * pdiff > 0)
    valid = all(value is not None and math.isfinite(value) for value in predicted)
    selected = None
    if valid:
        maximum = max(predicted)
        selected = next(i for i, score in enumerate(predicted) if maximum - score <= 1e-6)
    best = max(actual)
    return dict(eligible_pairs=eligible, correct_pairs=correct, selected=selected,
                selection_abstention=not valid,
                regret=None if selected is None else best - actual[selected],
                regret_conservative=1.0 if selected is None else best - actual[selected],
                hold_regret=best - actual[0], uniform_regret=best - sum(actual) / 4,
                actual_range=max(actual) - min(actual),
                optimism=None if selected is None else predicted[selected] - actual[selected])
