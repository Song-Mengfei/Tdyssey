"""Moved-T replay pools for official denoiser updates.

Training draws from saved windows after collection finishes. This is not an
online reinforcement-learning buffer and does not read model_test.
"""

from __future__ import annotations

import math

from .geometry import finite


TRANSLATION_THRESHOLD = 0.5
ROTATION_THRESHOLD = 0.01
DEFAULT_MOVED_FRACTION = 0.6


def wrap_angle(delta):
    """Smallest signed angle change in radians."""
    value = finite(delta, "angle delta")
    return math.atan2(math.sin(value), math.cos(value))


def object_displacement(before, after):
    """Translation in pixels and absolute wrapped rotation of the object body."""
    dx = after.object_body.pose.position[0] - before.object_body.pose.position[0]
    dy = after.object_body.pose.position[1] - before.object_body.pose.position[1]
    rotation = abs(wrap_angle(after.object_body.pose.angle - before.object_body.pose.angle))
    return math.hypot(dx, dy), rotation


def object_moved(before, after, translation=TRANSLATION_THRESHOLD, rotation=ROTATION_THRESHOLD):
    """True when the official object translated or rotated after one control."""
    moved, turned = object_displacement(before, after)
    return moved >= finite(translation, "translation") or turned >= finite(rotation, "rotation")


def split_motion_windows(windows):
    """Partition training windows using the predicted-step moved label."""
    moved, still = [], []
    for window in windows:
        if "moved" not in window:
            raise ValueError("Replay windows need a moved label on the predicted step")
        (moved if window["moved"] else still).append(window)
    return moved, still


def planned_replay_take(n_moved, n_still, batch_size, moved_fraction=DEFAULT_MOVED_FRACTION):
    """Count unique then replacement draws so a short pool does not shrink the batch."""
    if isinstance(n_moved, bool) or isinstance(n_still, bool) or isinstance(batch_size, bool):
        raise ValueError("Replay counts must be integers")
    if min(n_moved, n_still, batch_size) < 0 or batch_size < 1:
        raise ValueError("Replay batch size must be positive and pool sizes nonnegative")
    if n_moved + n_still < 1:
        raise ValueError("Replay needs at least one training window")
    fraction = finite(moved_fraction, "moved fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("moved fraction must be in [0, 1]")
    want_moved = min(batch_size, max(0, int(round(batch_size * fraction))))
    want_still = batch_size - want_moved
    take_moved = min(want_moved, n_moved)
    take_still = min(want_still, n_still)
    short = batch_size - take_moved - take_still
    extra_moved = min(short, n_moved - take_moved)
    take_moved += extra_moved
    short -= extra_moved
    extra_still = min(short, n_still - take_still)
    take_still += extra_still
    short -= extra_still
    replace_moved = replace_still = 0
    if short:
        if n_moved and (want_moved >= want_still or not n_still):
            replace_moved = short
        else:
            replace_still = short
    return {"moved": take_moved, "still": take_still, "replace_moved": replace_moved,
            "replace_still": replace_still, "requested_moved": want_moved}


def sample_replay_batch(moved, still, batch_size, moved_fraction, rng):
    """Draw one batch, oversampling moved-T windows when both pools are large."""
    plan = planned_replay_take(len(moved), len(still), batch_size, moved_fraction)
    batch = []
    batch.extend(_draw(moved, plan["moved"], rng, replace=False))
    batch.extend(_draw(still, plan["still"], rng, replace=False))
    batch.extend(_draw(moved, plan["replace_moved"], rng, replace=True))
    batch.extend(_draw(still, plan["replace_still"], rng, replace=True))
    if len(batch) != batch_size:
        raise ValueError("Replay batch did not reach the requested size")
    actual_moved = plan["moved"] + plan["replace_moved"]
    return batch, {"moved": actual_moved, "still": batch_size - actual_moved,
                   "requested_fraction": finite(moved_fraction, "moved fraction"),
                   "actual_fraction": actual_moved / batch_size}


def _draw(pool, count, rng, replace):
    if count <= 0:
        return []
    if not pool:
        raise ValueError("Cannot draw from an empty replay pool")
    if replace:
        return [_choice(pool, rng) for _ in range(count)]
    if count > len(pool):
        raise ValueError("Cannot draw more unique windows than the pool holds")
    order = _permutation(len(pool), rng)[:count]
    return [pool[index] for index in order]


def _permutation(size, rng):
    generator = getattr(rng, "manual_seed", None)
    if generator is not None or type(rng).__name__ == "Generator":
        import torch

        if isinstance(rng, torch.Generator):
            return [int(index) for index in torch.randperm(size, generator=rng)]
    values = list(range(size))
    rng.shuffle(values)
    return values


def _choice(pool, rng):
    generator = getattr(rng, "manual_seed", None)
    if generator is not None or type(rng).__name__ == "Generator":
        import torch

        if isinstance(rng, torch.Generator):
            index = int(torch.randint(len(pool), (1,), generator=rng).item())
            return pool[index]
    return rng.choice(pool)
