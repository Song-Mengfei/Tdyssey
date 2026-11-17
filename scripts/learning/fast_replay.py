"""Faster exact prefix replay: render only the two frames the policy history needs.

policy_branches.replay_prefix renders an image and recomputes privileged state at every replayed
control (~3.7 ms each); rendering alone is ~82% of a step. Physics is deterministic, so replaying the
same actions from the same restored initial snapshot reproduces the same state. This version steps
physics with rendering disabled except for the last two controls, then verifies the final control
against the recorded row with the same tolerances as replay_prefix (contacts, termination, rewards,
coverage, body states). Returns the same 2-frame history deque.
"""

from collections import deque

from tdyssey.policy_branches import step_record, verify_row
from tdyssey.state import Snapshot


def fast_replay_prefix(env, episode, counters=None, deadline=lambda: None):
    observation = env.restore_state(Snapshot.from_dict(episode["initial_state"]))
    steps = episode["steps"]
    history = deque([observation]*2, maxlen=2)
    real_get_obs = env._get_obs
    try:
        env._get_obs = lambda: None                       # physics and reward only
        for expected in steps[:-2]:
            deadline()
            _, _, done, _ = env.step(expected["action"])
            if done:
                raise ValueError("Cannot branch after terminal success")
    finally:
        env._get_obs = real_get_obs
    for expected in steps[-2:]:                            # rendered: these frames feed the policy
        observation, actual = step_record(env, expected["action"])
        if actual["done"]:
            raise ValueError("Cannot branch after terminal success")
        history.append(observation)
    if steps:
        verify_row(actual, steps[-1])
    if counters is not None:
        counters["initial_state_restores"] += 1
        counters["replay_controls"] += len(steps)
        counters["training_controls"] += len(steps)
    return history
