"""One predeclared frozen diagnostic: calibration, shared physics, forecasts."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.02 * (attempt + 1))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    paths = [*(ROOT / "tdyssey").rglob("*.py"), Path(__file__),
             ROOT / "scripts/run_guarded_frozen_wm.py", ROOT / "scripts/guarded_c_child.py",
             ROOT / "manifests/gpc_world_model.json", ROOT / "manifests/official_dp.json"]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def unattempted_slots(report):
    """A failed slot is spent too; recovery cannot retry or replace it."""
    from tdyssey.frozen_wm_protocol import anchors, CANDIDATES
    spent = {(row["anchor"], row["candidate"]) for row in report.get("branches", [])}
    return {(spec["id"], candidate) for spec in anchors() for candidate in CANDIDATES} - spent


def state_label(state, shape):
    from tdyssey.evaluation import goal_metrics
    metric = goal_metrics(shape, state.object_body.pose, state.goal)
    return dict(pose=asdict(state.object_body.pose), pusher=list(state.pusher_body.pose.position),
                coverage=metric.coverage, iou=metric.iou)


def invoke(budget, costs, function):
    operation = budget.reserve(**costs)
    try:
        value = function()
    except BaseException as exc:
        budget.abort(operation, {"error": f"{type(exc).__name__}: {exc}"})
        raise
    budget.complete(operation)
    return value


def stopped_step(env, action, budget):
    """Reserve actual physics before stepping; never step after success."""
    import numpy as np
    return invoke(budget, {"controls": 1, "substeps": 10},
                  lambda: env.step(np.asarray(action, dtype=np.float64)))


def create_environment(cls, spec, budget):
    from tdyssey.collection import reset_issues
    from tdyssey.envs.official import official_shape
    from tdyssey.geometry import Pose
    from tdyssey.state import BodyState
    env = cls(shape=spec["shape"], legacy=True)
    try:
        env.seed(spec["seed"])
        invoke(budget, {"reset": 1, "settle": 1}, env.reset)
        state = replace(env.capture_state(), object_body=BodyState(spec["pose"]),
                        pusher_body=BodyState(Pose(spec["pusher"], 0)), goal=spec["goal"],
                        control_step=0, pusher_target=None)
        issues = [issue for issue in reset_issues(official_shape(spec["shape"]), state)
                  if issue != "already_near_goal"]
        if issues:
            raise ValueError(f"Fixed initial state invalid: {issues}")
        observation = invoke(budget, {"restore": 1}, lambda: env.restore_state(state))
        if env.sim_hz != 100 or env.control_hz != 10 or env.success_threshold != .95:
            raise ValueError("Unexpected official time/success interface")
        return env, observation
    except BaseException:
        env.close()
        raise


def collect_future(env, actions, budget, shape, steps=None, frames=None, on_step=None):
    """One executed prefix; terminal state is absorbing without more physics."""
    steps = [] if steps is None else steps
    frames = [] if frames is None else frames
    terminal = False
    for h, action in enumerate(actions, 1):
        if not terminal:
            observation, reward, done, info = stopped_step(env, action, budget)
            snapshot = env.capture_state()
            label = state_label(snapshot, shape)
            if abs(float(reward) - min(label["coverage"] / .95, 1)) > 1e-7 or bool(done) != (label["coverage"] > .95):
                raise ValueError("Raw coverage/upstream reward/strict success disagree")
            terminal = bool(done)
            step = dict(horizon=h, action=action, state=asdict(snapshot), truth=label,
                        contacts=int(info["n_contacts"]), upstream_clipped_reward=float(reward),
                        success=terminal, absorbing=False)
        else:
            step = dict(steps[-1], horizon=h, action=None, contacts=0, absorbing=True)
        steps.append(step)
        frames.append(observation["image"].copy())
        if on_step is not None:
            on_step()
    return steps, frames


def image_proxy(image):
    """Full-frame appearance proxy, with fixed colors and no truth ROI input."""
    import numpy as np
    rgb = np.moveaxis(np.clip(image, 0, 1), 0, -1) * 255
    gray = (rgb[..., 0] > 70) & (rgb[..., 0] < 190) & (rgb[..., 1] > rgb[..., 0] + 4) & (
        rgb[..., 2] > rgb[..., 1] + 4) & (rgb.max(-1) - rgb.min(-1) < 65)
    red = (rgb[..., 0] > 130) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 0] > 1.6 * rgb[..., 2])
    mask = gray | red
    y, x = np.nonzero(mask)
    center = None if len(x) < 12 else [float((x.mean() + .5) * 512 / 96 - .5),
                                      float((y.mean() + .5) * 512 / 96 - .5)]
    return mask, dict(object_pixels=int(mask.sum()), centroid=center)


def rgb_metrics(predicted, actual, latest):
    import numpy as np
    actual_mask, actual_proxy = image_proxy(actual)
    result = {}
    for name, frame in (("model", predicted), ("persistence", latest)):
        mask, proxy = image_proxy(frame)
        union = (mask | actual_mask).sum()
        result[name] = dict(image_mse=float(np.mean((np.clip(frame, 0, 1) - actual) ** 2)),
            object_mask_iou=float((mask & actual_mask).sum() / union) if union else None,
            centroid_error_pixels=None if proxy["centroid"] is None or actual_proxy["centroid"] is None
            else math.dist(proxy["centroid"], actual_proxy["centroid"]), **proxy)
    return result


def pose_metrics(prediction, truth, latest):
    from tdyssey.frozen_wm_protocol import wrapped_degrees
    if not prediction["valid"]:
        return {"valid": False, "reason": prediction["reason"]}
    delta = wrapped_degrees(truth["pose"]["angle"] - latest["pose"]["angle"])
    predicted_delta = wrapped_degrees(prediction["pose"]["angle"] - latest["pose"]["angle"])
    return dict(valid=True,
        position_error=math.dist(prediction["pose"]["position"], truth["pose"]["position"]),
        angle_error=abs(wrapped_degrees(prediction["pose"]["angle"] - truth["pose"]["angle"])),
        pusher_error=math.dist(prediction["pusher"], truth["pusher"]),
        coverage_error=abs(prediction["coverage"] - truth["coverage"]),
        coverage_bias=prediction["coverage"] - truth["coverage"],
        actual_rotation=delta, predicted_rotation=predicted_delta,
        rotation_direction_correct=bool(delta * predicted_delta > 0),
        persistence_position_error=math.dist(latest["pose"]["position"], truth["pose"]["position"]),
        persistence_angle_error=abs(delta),
        persistence_coverage_error=abs(latest["coverage"] - truth["coverage"]))


def run(args):
    for key, value in dict(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                           MPLBACKEND="Agg", PYGAME_HIDE_SUPPORT_PROMPT="1").items():
        os.environ.setdefault(key, value)
    from tdyssey.frozen_wm_budget import Budget
    from tdyssey.frozen_wm_protocol import anchors, calibration_states, CANDIDATES, NOISE_SEEDS, candidate_actions, predicted_scores
    started = time.monotonic()
    binding = dict(protocol_sha256=file_hash(args.protocol),
                   config_sha256=file_hash(ROOT / "configs/frozen_wm_decoder_v1.json"),
                   source_sha256=source_hashes(), checkpoint_sha256=file_hash(args.checkpoint),
                   official_root=str(args.official_root.resolve()), gpc_root=str(args.gpc_root.resolve()))
    budget = (Budget.resume(args.output_dir, args.ledger_dir, binding) if args.resume
              else Budget(args.output_dir, args.ledger_dir, binding))
    output = args.output_dir
    report = dict(schema_version=1, kind="tdyssey.frozen_world_model_diagnostic.v1", status="running",
        started_at=datetime.now(timezone.utc).isoformat(), binding=binding,
        source_commit=subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                            "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        policy_optimizer_updates=0, world_model_optimizer_updates=0, model_test_read=False,
        palette="unchanged official RGB; no privileged palette rendering", calibrations=[], branches=[],
        custom_model={"status": "unavailable", "reason": "historical exact best.pt/catalog not found; no replacement"},
        contact_head={"status": "unavailable", "reason": "GPC predicts RGB only; no direct contact probability"})
    if args.resume:
        prior_bytes = (output / "report.json").read_bytes()
        prior = json.loads(prior_bytes)
        if prior["status"] != "aborted" or prior.get("model") is not None:
            raise ValueError("Recovery is restricted to the recorded pre-model persistence abort")
        immutable = output / "attempt-1-aborted-report.json"
        with immutable.open("xb") as saved:
            saved.write(prior_bytes)
            saved.flush()
            os.fsync(saved.fileno())
        report = prior
        report["recovery"] = dict(reason="Windows ledger replace permission failure; same spent authority",
            active_binding=binding, source_commit=subprocess.check_output(
                ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            original_wall_seconds=prior["wall_seconds"], started_at=datetime.now(timezone.utc).isoformat(),
            no_physical_retries=True, skipped_attempted_branches=len(prior["branches"]))
        report["status"] = "running"
        report.pop("error", None)
        report.pop("traceback", None)
    write_json(output / "report.json", report)
    model, env = None, None
    try:
        import numpy as np
        import torch
        from tdyssey.envs.official import make_official_shape_env_class, official_shape
        from tdyssey.frozen_wm_decoder import decode, calibration_summary, CONFIG_SHA256
        from tdyssey.gpc_world_model import GPCWorldModel
        from tdyssey.official import load_official_source
        from tdyssey.policy_training import state_fingerprint
        torch.set_num_threads(2)
        cls = make_official_shape_env_class(load_official_source(args.official_root))
        report["decoder_config_sha256"] = CONFIG_SHA256
        if CONFIG_SHA256 != binding["config_sha256"]:
            raise ValueError("Decoder configuration differs from binding")
        for spec in ([] if args.resume else calibration_states()):
            budget.check()
            invoke(budget, {"calibration": 1}, lambda: None)
            env, observation = create_environment(cls, spec, budget)
            state = env.capture_state()
            truth = state_label(state, official_shape(spec["shape"]))
            prediction = decode(observation["image"], spec["shape"], spec["goal"])
            row = dict(id=spec["id"], shape_name=spec["shape"], seed=spec["seed"],
                       state=asdict(state), truth=truth, prediction=prediction)
            report["calibrations"].append(row)
            np.savez_compressed(output / (spec["id"] + ".npz"), rgb=observation["image"])
            env.close()
            env = None
            write_json(output / "report.json", report)
        report["calibration_gate"] = calibration_summary(report["calibrations"])
        write_json(output / "calibration.json", dict(rows=report["calibrations"], gate=report["calibration_gate"]))
        write_json(output / "report.json", report)
        print("CALIBRATION " + json.dumps(report["calibration_gate"]), flush=True)
        # Collect once, before any forecast; all model noises reuse these branches.
        remaining_slots = unattempted_slots(report)
        for anchor in anchors():
            reference = next((r["history_digest"] for r in report["branches"]
                if r["anchor"] == anchor["id"] and r["status"] == "complete"), None)
            for candidate in CANDIDATES:
                if (anchor["id"], candidate) not in remaining_slots:
                    continue
                budget.check()
                invoke(budget, {"branches": 1}, lambda: None)
                row = dict(anchor=anchor["id"], shape=anchor["shape"], family=anchor["family"],
                           candidate=candidate, seed=anchor["seed"], status="running", history=[], actual=[], predictions=[])
                report["branches"].append(row)
                try:
                    env, observation = create_environment(cls, anchor, budget)
                    row["initial_state"] = asdict(env.capture_state())
                    frames = [observation["image"].copy()]
                    past = []
                    actions = candidate_actions(anchor, candidate)
                    actual_frames = []
                    def save_branch():
                        write_json(output / f"{anchor['id']}-{candidate}.json", row)
                        np.savez_compressed(output / f"{anchor['id']}-{candidate}-physical.npz",
                                            history=np.stack(frames), past_actions=np.array(past),
                                            future_actions=np.array(actions), actual=np.array(actual_frames))
                    save_branch()
                    for _ in range(3):
                        observation, reward, done, info = stopped_step(env, anchor["pusher"], budget)
                        snapshot = env.capture_state()
                        row["history"].append(dict(action=list(anchor["pusher"]), state=asdict(snapshot),
                            contacts=int(info["n_contacts"]), upstream_clipped_reward=float(reward), success=bool(done)))
                        frames.append(observation["image"].copy())
                        past.append(list(anchor["pusher"]))
                        save_branch()
                        if done or int(info["n_contacts"]):
                            raise ValueError("History success or contact; fixed branch invalid, no retry")
                    current = env.capture_state()
                    history_digest = hashlib.sha256(np.stack(frames).tobytes() + current.to_json().encode()
                        + json.dumps(row["initial_state"], sort_keys=True).encode()).hexdigest()
                    row["history_digest"] = history_digest
                    if reference is None:
                        reference = history_digest
                    if history_digest != reference:
                        raise ValueError("Candidate histories/full states differ")
                    shape = official_shape(anchor["shape"])
                    row["latest"] = state_label(current, shape)
                    row["actions"] = actions
                    collect_future(env, actions, budget, shape, row["actual"], actual_frames, save_branch)
                    row["status"] = "complete"
                    save_branch()
                except Exception as exc:
                    row.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
                finally:
                    if env is not None:
                        env.close()
                        env = None
                    write_json(output / "report.json", report)
            print("PHYSICS " + anchor["id"], flush=True)
        report["physics_completed_seconds"] = budget.snapshot()["elapsed_seconds"]
        budget.check()
        # Keep constructors outside forecast counters, but within wall/resource cap.
        global_random = random.getstate()
        numpy_random = np.random.get_state()
        cpu_rng = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            model = GPCWorldModel(args.gpc_root, args.checkpoint)
        inference_cpu_rng = torch.get_rng_state().clone()
        before = state_fingerprint(model.network)
        gpu_rng = torch.cuda.get_rng_state().clone()
        torch.cuda.reset_peak_memory_stats()
        report["model"] = dict(parameter_count=model.parameter_count, state_entries=model.state_entries,
            network_fingerprint_before=before, checkpoint_hash_before=file_hash(args.checkpoint),
            device=str(model.device), gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            numpy_version=np.__version__, freeze_before=not any(p.requires_grad for p in model.network.parameters()),
            eval_before=not model.network.training)
        for row in report["branches"]:
            if row["status"] != "complete":
                continue
            budget.check()
            with np.load(output / f"{row['anchor']}-{row['candidate']}-physical.npz") as data:
                images, past, actions, actual = [data[key].copy() for key in ("history", "past_actions", "future_actions", "actual")]
            for seed in NOISE_SEEDS:
                began = time.monotonic()
                raw = invoke(budget, {"forecast_images": 4, "euler_steps": 12},
                             lambda: model.predict(images, past, actions, seed=seed))
                torch.cuda.synchronize()
                entry = dict(seed=seed, latency_seconds=time.monotonic()-began,
                    raw_min=float(raw.min()), raw_max=float(raw.max()),
                    clipped_fraction=float(((raw < 0) | (raw > 1)).mean()), steps=[])
                row["predictions"].append(entry)
                np.savez_compressed(output / f"{row['anchor']}-{row['candidate']}-noise-{seed}.npz", raw=raw)
                gate = report["calibration_gate"][row["shape"]]["passed"]
                for h in range(4):
                    budget.check()
                    decoded = decode(raw[h], row["shape"], row["initial_state"]["goal"])
                    if not gate:
                        # Preserve abstention reasons, never publish an uncalibrated reward.
                        decoded = {k: v for k, v in decoded.items() if k not in ("pose", "coverage", "pusher")}
                        decoded.update(valid=False, pose=None, coverage=None, pusher=None, contact_proxy=None,
                                       decoder_reason=decoded["reason"], reason="calibration_gate_failed",
                                       measurement_gate="failed_calibration")
                    entry["steps"].append(dict(horizon=h+1, decoded=decoded,
                        rgb=rgb_metrics(raw[h], actual[h], images[-1]),
                        physical_absorbing=row["actual"][h]["absorbing"],
                        motion=pose_metrics(decoded, row["actual"][h]["truth"], row["latest"])
                            if gate and not row["actual"][h]["absorbing"] else None))
                scores = predicted_scores([step["decoded"] for step in entry["steps"]])
                for h, score in enumerate(scores):
                    entry["steps"][h].update(score_coverage=score,
                        score_coverage_error=None if score is None else abs(score-row["actual"][h]["truth"]["coverage"]),
                        predicted_absorbing=any(value is not None and value > .95 for value in scores[:h]))
                write_json(output / f"{row['anchor']}-{row['candidate']}.json", row)
            write_json(output / "report.json", report)
            print("FORECAST " + row["anchor"] + " " + row["candidate"], flush=True)
        report["model"].update(network_fingerprint_after=state_fingerprint(model.network),
            checkpoint_hash_after=file_hash(args.checkpoint),
            freeze_after=not any(p.requires_grad or p.grad is not None for p in model.network.parameters()),
            eval_after=not model.network.training, cuda_rng_preserved=torch.equal(gpu_rng, torch.cuda.get_rng_state()),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        report["model"]["weights_preserved"] = (before == report["model"]["network_fingerprint_after"]
            and binding["checkpoint_sha256"] == report["model"]["checkpoint_hash_after"])
        # Constructor RNG is recorded separately; inference must preserve RNG exactly.
        report["rng"] = dict(python_preserved=global_random == random.getstate(),
            numpy_preserved=all(np.array_equal(a, b) for a, b in zip(numpy_random, np.random.get_state())),
            cpu_inference_preserved=torch.equal(inference_cpu_rng, torch.get_rng_state()),
            cpu_constructor_rng_changed=not torch.equal(cpu_rng, torch.get_rng_state()))
        reference = weakref.ref(model.network)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        report["model"]["network_released"] = reference() is None
        if (not all(report["model"][key] for key in ("weights_preserved", "cuda_rng_preserved", "freeze_before", "freeze_after",
                                                    "eval_before", "eval_after", "network_released"))
                or not all(report["rng"][key] for key in ("python_preserved", "numpy_preserved", "cpu_inference_preserved"))):
            raise ValueError("Frozen model or inference RNG preservation failed")
        from tdyssey.frozen_wm_summary import summarize, make_previews
        report["summary"] = summarize(report)
        make_previews(report, output)
        if (source_hashes() != binding["source_sha256"] or file_hash(args.protocol) != binding["protocol_sha256"]
                or file_hash(ROOT / "configs/frozen_wm_decoder_v1.json") != binding["config_sha256"]):
            raise ValueError("Source changed during attempt")
        budget.check()
        report["status"] = "complete" if all(r["status"] == "complete" for r in report["branches"]) else "complete_with_aborted_branches"
    except BaseException as exc:
        report.update(status="aborted", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        if env is not None:
            env.close()
        if model is not None:
            del model
            gc.collect()
        report["active_segment_wall_seconds"] = time.monotonic() - started
        report["wall_seconds"] = budget.snapshot()["elapsed_seconds"]
        report["budget"] = budget.snapshot()
        write_json(output / "report.json", report)
        budget.close()
    return 0 if report["status"] in ("complete", "complete_with_aborted_branches") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "ledger-dir", "official-root", "gpc-root", "checkpoint", "protocol"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue unused slots of the recorded persistence-aborted attempt only")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
