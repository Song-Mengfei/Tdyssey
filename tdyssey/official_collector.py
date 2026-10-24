"""Shape-parameterized official play collection for one world-model run.

Each worker owns one simulator. The parent writes finished episodes in index
order after every worker returns. Physics stays on CPU; this module does not
load a policy or a world model.
"""

import multiprocessing
import os
import time

from .collection import integer
from .dataset import DatasetWriter
from .official_collection import OfficialCollectionConfig, collect_official_episode, freeze_splits


def _quiet_sdl():
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)


def _collect_one(payload):
    """Collect one episode in a fresh process. SDL is set before pygame imports."""
    source_root, config, index, deadline = payload
    _quiet_sdl()
    from .envs.official import make_official_shape_env_class
    from .official import load_official_source

    counters = {"reset_attempts": 0, "reset_physics_substeps": 0,
                "control_steps": 0, "control_physics_substeps": 0}
    env = make_official_shape_env_class(load_official_source(source_root))(shape=config.shape, legacy=True)
    try:
        episode = collect_official_episode(env, config, index, counters, deadline)
        rates = {"simulation_hz": env.sim_hz, "control_hz": env.control_hz}
    finally:
        env.close()
    return index, episode, counters, rates


class OfficialShapeCollector:
    """Collect one official shape with object-directed, angle, or contact play.

    T is the first world-model run. A later letter uses the same class with
    another official shape name; it does not need a second collector.
    """

    def __init__(self, config, source_root, workers=1):
        if not isinstance(config, OfficialCollectionConfig):
            raise ValueError("Expected an official collection config")
        if config.mode not in ("object-directed", "angle-coverage", "contact-balanced"):
            raise ValueError("Official world-model collection uses object-directed, angle-coverage, or contact-balanced play")
        if config.episodes < 7:
            raise ValueError("Use at least seven episodes for nonempty whole-episode splits")
        self.config = config
        self.source_root = str(source_root)
        self.workers = integer(workers, "workers", 1, 8)

    def collect(self, output, provenance, deadline=None):
        """Write a finished dataset. Return the manifest after frozen splits."""
        _quiet_sdl()
        started = time.monotonic()
        config = self.config
        end = None if deadline is None else float(deadline)
        jobs = [(self.source_root, config, index, end) for index in range(config.episodes)]
        results, rates = {}, None
        totals = {"reset_attempts": 0, "reset_physics_substeps": 0, "control_steps": 0,
                  "control_physics_substeps": 0, "excluded_seed_candidates": 0}
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(self.workers, config.episodes)) as pool:
            for index, episode, counters, local_rates in pool.imap_unordered(_collect_one, jobs):
                if rates is None:
                    rates = local_rates
                elif rates != local_rates:
                    raise ValueError("Official workers disagree on simulation/control rates")
                results[index] = episode
                for key, value in counters.items():
                    totals[key] = totals.get(key, 0) + value
        if sorted(results) != list(range(config.episodes)) or rates is None:
            raise ValueError("Official collection did not return every episode")
        episodes = [results[index] for index in range(config.episodes)]
        writer = DatasetWriter(output, config, provenance)
        writer.manifest["rates"] = rates
        writer.manifest["counts"].update(totals)
        try:
            for episode in episodes:
                writer.add(episode)
            writer.manifest["elapsed_seconds"] = time.monotonic() - started
            writer.finish(freeze_splits([episode.trajectory for episode in episodes], config.seed))
        except BaseException as exc:
            if not writer.closed:
                writer.finish(error=f"{type(exc).__name__}: {exc}")
            raise
        return writer.manifest
