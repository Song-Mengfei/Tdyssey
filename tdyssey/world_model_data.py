"""Visual model windows with separate privileged supervision, never state inputs."""

import math
import random

from .dataset import load_episode
from .envs.official import official_shape
from .evaluation import goal_metrics
from .geometry import Pose
from .official_collection import ENVIRONMENT_FAMILY
from .training_data import WindowDataset


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def episode_order(offsets, seed):
    """Shuffle windows within shuffled episodes to retain a one-episode cache."""
    rng = random.Random(seed)
    episodes = list(range(len(offsets)-1))
    rng.shuffle(episodes)
    for index in episodes:
        windows = list(range(offsets[index], offsets[index+1]))
        rng.shuffle(windows)
        yield from windows


def object_masks(shape, poses):
    """Approximate geometry occupancy at policy-image pixel centers; label only."""
    import numpy as np

    y, x = np.mgrid[:96, :96].astype(np.float64)
    x, y = (x+.5)*512/96, (y+.5)*512/96
    masks = []
    for px, py, angle in poses:
        dx, dy = x-px, y-py
        local_x = math.cos(angle)*dx + math.sin(angle)*dy
        local_y = -math.sin(angle)*dx + math.cos(angle)*dy
        mask = np.zeros((96,96), dtype=np.bool_)
        for rect in shape.components:
            mask |= (local_x >= rect.x0) & (local_x <= rect.x1) & (local_y >= rect.y0) & (local_y <= rect.y1)
        # One-pixel dilation covers antialiasing at geometric edges.
        padded = np.pad(mask, 1)
        masks.append(np.logical_or.reduce([padded[i:i+96, j:j+96] for i in range(3) for j in range(3)]))
    return np.asarray(masks)[:, None]


class VisualDynamicsDataset(WindowDataset):
    """Attach labels from the already decoded episode without another disk read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.catalog.get("environment_family") != ENVIRONMENT_FAMILY:
            raise ValueError("The visual world-model pilot requires official data")
        if any(m["config"]["shape"] != "L" for m in self.manifests):
            raise ValueError("The first visual world-model pilot is L-only")
        self._physical_key, self._physical = None, None

    def __getitem__(self, index):
        import numpy as np

        sample = super().__getitem__(index)
        trajectory, record, arrays = self._cache
        if self._physical_key != self._cache_key:
            states = [trajectory.transitions[0].before, *(t.after for t in trajectory.transitions)]
            poses = np.asarray([(*s.object_body.pose.position, s.object_body.pose.angle) for s in states])
            motion = np.diff(poses, axis=0)
            motion[:, 2] = np.arctan2(np.sin(motion[:, 2]), np.cos(motion[:, 2]))
            agent_motion = np.diff(arrays["agent_pos"], axis=0)
            masks = object_masks(official_shape("L"), poses)
            self._physical = poses, motion, agent_motion, masks
            self._physical_key = self._cache_key
        poses, motion, agent_motion, masks = self._physical
        anchor, future = sample["anchor"], self.spec.future
        selection = slice(anchor, anchor+future)
        displacement = poses[anchor+1:anchor+future+1]-poses[anchor]
        displacement[:, 2] = np.arctan2(np.sin(displacement[:, 2]), np.cos(displacement[:, 2]))
        initial = trajectory.transitions[anchor].before
        prior = goal_metrics(official_shape("L"), initial.object_body.pose, initial.goal)
        sample["labels"].update({
            "motion": np.concatenate((agent_motion[selection], motion[selection]), axis=1).astype(np.float32),
            "object_displacement": displacement.astype(np.float32),
            "object_roi": (masks[anchor:anchor+future] | masks[anchor+1:anchor+future+1]).copy(),
            "previous_reward": np.float32(min(prior.coverage/.95, 1)),
            "previous_contact": np.float32(record["step_diagnostics"][anchor-1]["n_contacts"] > 0) if anchor else np.float32(0),
        })
        return sample


def fit_motion_stats(dataset):
    """Fit unique TRAIN transitions, not duplicated overlapping windows."""
    import numpy as np

    if dataset.split != "train":
        raise ValueError("Fit normalization on training episodes only")
    values, contacts, rewards, identifiers = [], [], [], []
    for run_index, _, entry in dataset.episodes:
        trajectory, record, _ = load_episode(dataset.roots[run_index], entry)
        identifiers.append(trajectory.trajectory_id)
        for transition, diagnostics in zip(trajectory.transitions, record["step_diagnostics"]):
            before, after = transition.before, transition.after
            agent = [after.pusher_body.pose.position[i]-before.pusher_body.pose.position[i] for i in (0,1)]
            obj = [after.object_body.pose.position[i]-before.object_body.pose.position[i] for i in (0,1)]
            values.append([*agent, *obj, wrap_angle(after.object_body.pose.angle-before.object_body.pose.angle)])
            contacts.append(diagnostics["n_contacts"] > 0)
            rewards.append(transition.reward)
    if not values:
        raise ValueError("Training episodes have no transitions")
    values = np.asarray(values, dtype=np.float64)
    scales = np.maximum(values.std(axis=0), [.25,.25,.25,.25,.005])
    return {"motion_mean": values.mean(axis=0).tolist(), "motion_scale": scales.tolist(),
            "contact_frequency": float(np.mean(contacts)), "mean_reward": float(np.mean(rewards)),
            "unique_transitions": len(values), "episode_ids": identifiers,
            "input_coordinates": "fixed world coordinates x/256-1; actions use (target-current_agent)/256",
            "target_order": ["agent_dx_px","agent_dy_px","object_dx_px","object_dy_px","object_dangle_rad"]}
