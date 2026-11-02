"""The four fixed official collection runs; no model or environment import."""

from dataclasses import asdict
import os
from pathlib import Path
import time

from .dataset import DatasetWriter, digest, json_bytes
from .gpc_adaptation_data import KIND, T1_SPEC_SHA256
from .official_collection import (DEVELOPMENT_SEEDS, ENVIRONMENT_FAMILY, VERSION,
    OfficialCollectionConfig, collect_official_episode, freeze_splits)
from .training_data import write_catalog


RUNS = (("T-A", "T", 410000), ("T-B", "T", 410001),
        ("L-A", "L", 410002), ("L-B", "L", 410003))


def collect_fixed_runs(budget, env_factory, *, exclusions, provenance_factory):
    """One pass only. A failed call leaves the account spent and aborts all runs.

    Scene exclusion is tested before the first control. New repeated scenes are
    a freeze error, never an excuse to replace an episode or reassign its split.
    Catalog and role manifests use already collected identities, including held.
    """
    excluded_states = frozenset(exclusions["reviewed_scene_sha256"])
    seeds = tuple(sorted(set(DEVELOPMENT_SEEDS) | set(exclusions["excluded_reset_seeds"])))
    roots, declarations, summaries, scenes = [], [], [], set()
    try:
        for run_id, shape, seed in RUNS:
            budget.begin_run(run_id)
            config = OfficialCollectionConfig(shape=shape, seed=seed, episodes=8, steps=100,
                reset_attempts=20, seconds=120, max_bytes=32*1024**2, excluded_reset_seeds=seeds)
            provenance = provenance_factory(config)
            provenance["excluded_initial_state_sha256"] = sorted(excluded_states)
            provenance["exclusion_manifest_sha256"] = exclusions["sha256"]
            writer = DatasetWriter(budget.output/run_id, config, provenance, budget=budget)
            env = None
            start = time.monotonic()
            try:
                env = env_factory(shape)
                writer.manifest["rates"] = dict(simulation_hz=env.sim_hz, control_hz=env.control_hz)
                if (env.sim_hz, env.control_hz) != (100, 10):
                    raise ValueError("Frozen official rates are 100 Hz / 10 Hz")
                trajectories = []
                for index in range(8):
                    episode = collect_official_episode(env, config, index, writer.manifest["counts"],
                        excluded_initial_state_sha256=excluded_states, budget=budget)
                    writer.add(episode)
                    trajectories.append(episode.trajectory)
                splits = freeze_splits(trajectories, config.seed)
                if sorted(map(len, splits.values())) != [1, 1, 6]:
                    raise ValueError("The original hash-sort source split must remain 6/1/1")
                # Use collection-time identity metadata, not a payload readback.
                for entry in writer.manifest["episodes"]:
                    scene = entry["collection_metadata"]["initial_state_sha256"]
                    if scene in scenes or scene in excluded_states:
                        raise ValueError("A repeated/reviewed scene blocks freeze; no replacement")
                    scenes.add(scene)
                    identifier = entry["trajectory_id"]
                    split = next(name for name, ids in splits.items() if identifier in ids)
                    role = ("held_model_test" if split == "model_test" else "train" if split == "train"
                            else "development" if run_id.endswith("-A") else "independent_validation")
                    declarations.append(dict(trajectory_id=identifier, role=role, prior_uses=[]))
                writer.manifest["elapsed_seconds"] = time.monotonic()-start
                writer.finish(splits)
                summaries.append(dict(run_id=run_id, seed=seed, shape=shape,
                    counts=dict(writer.manifest["counts"]),
                    splits={name:len(ids) for name,ids in splits.items()},
                    elapsed_seconds=writer.manifest["elapsed_seconds"]))
                roots.append(writer.root)
                budget.finish_run()
            except BaseException as error:
                # The prepaid stage failure slot remains writable even if a
                # quota/deadline prevents publishing another dataset manifest.
                budget.fail(type(error).__name__+": "+str(error))
                raise
            finally:
                if env is not None:
                    env.close()
        catalog_path = budget.output/"catalog.json"
        catalog = write_catalog(roots, catalog_path, budget=budget)
        manifest = dict(schema_version=1, kind=KIND, environment_family=ENVIRONMENT_FAMILY,
            t1_spec_sha256=T1_SPEC_SHA256,
            catalogs=[dict(path="catalog.json", sha256=digest(json_bytes(catalog)))],
            episodes=declarations, reviewed_scene_sha256=sorted(excluded_states),
            previously_used_episode_ids=list(exclusions.get("previously_used_episode_ids", ())))
        budget.write_bytes("adaptation-data.json", json_bytes(manifest))
        result = dict(kind="fixed_official_collection_result.v1", runs=summaries,
            total_episodes=len(declarations), distinct_scenes=len(scenes),
            role_counts={role:sum(row["role"]==role for row in declarations) for role in
                ("train","development","independent_validation","held_model_test")},
            held_payload_reads=0, models_loaded=0, optimizer_updates=0)
        budget.write_bytes("collection-result.json", json_bytes(result))
        return result
    except BaseException as error:
        budget.fail(type(error).__name__+": "+str(error))
        raise


def real_environment_factory(source_root):
    """Call only inside the claimed, native-supervised A/B worker."""
    for key, value in dict(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                            MPLBACKEND="Agg", PYGAME_HIDE_SUPPORT_PROMPT="1").items():
        os.environ.setdefault(key, value)
    from .official import load_official_source
    from .envs.official import make_official_shape_env_class
    cls = make_official_shape_env_class(load_official_source(source_root))
    return lambda shape: cls(shape=shape, legacy=True)
