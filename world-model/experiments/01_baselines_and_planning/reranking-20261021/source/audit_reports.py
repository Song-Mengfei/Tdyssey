"""Read saved simulator reports and write a compact independent audit; no GPU use."""

from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tdyssey.envs.official import official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.simulator_planning import state_error
from tdyssey.state import Snapshot


def near(a, b, tolerance=1e-7):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tolerance


def mean(values):
    return sum(values)/len(values) if values else None


def audit(path):
    if not path.exists():
        return {"availability": "not_yet_written"}, None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"availability": "being_written"}, None
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    config = report["config"]
    complete = report["status"] == "complete"
    expected = {(pair, shape) for pair in range(config["pairs"]) for shape in config["shapes"]}
    census = report.get("requested_episodes", [])
    check({(e["pair_index"], e["shape"]) for e in census} == expected, "Requested-episode census differs from config")
    episodes = report["episodes"]
    seen = {(e["pair_index"], e["shape"]) for e in episodes}
    check(len(seen) == len(episodes), "Duplicate actual episode")
    if complete:
        check(seen == expected, "Complete run omits a requested episode")
        check(all(e["status"] == "completed" for e in census), "Complete run has unfinished census entries")
    summaries = []
    all_calls = 0
    for episode in episodes:
        label = f"{episode['shape']}/{episode['pair_index']}"
        rows = episode["steps"]
        initial = Snapshot.from_dict(episode["initial_state"])
        shape = official_shape(episode["shape"])
        check(episode.get("control_steps", len(rows)) == len(rows), f"{label}: control count differs")
        check(len(rows) <= config["steps"], f"{label}: real control budget exceeded")
        if episode["stop_reason"] == "control_budget":
            check(len(rows) == config["steps"], f"{label}: budget stop with incomplete controls")
        done = [bool(row["upstream_done"]) for row in rows]
        check(not any(done[:-1]), f"{label}: controls continued after native success")
        check(episode["success"] == bool(done and done[-1]), f"{label}: success flag differs from native done")
        check((episode["stop_reason"] == "upstream_success") == episode["success"], f"{label}: success stop reason differs")
        census_row = next((r for r in census if (r["pair_index"], r["shape"]) == (episode["pair_index"], episode["shape"])), {})
        if complete:
            check(census_row.get("control_steps") == len(rows), f"{label}: census control count differs")
        actual = {}
        geometry_error = 0.
        for index, row in enumerate(rows):
            state = Snapshot.from_dict(row["state"])
            actual[state.control_step] = row
            check(state.control_step == initial.control_step+index+1, f"{label}: discontinuous real control count")
            check(state.pusher_target == tuple(row["action"]), f"{label}: recorded target differs from action")
            metrics = goal_metrics(shape, state.object_body.pose, state.goal)
            for key in ("coverage", "iou"):
                difference = abs(row["metrics"][key]-getattr(metrics, key))
                geometry_error = max(geometry_error, difference)
                check(difference <= 1e-7, f"{label}: recorded {key} differs from independent geometry")
            check(near(row["reward"], min(metrics.coverage/.95, 1.)), f"{label}: reward differs from native coverage formula")
            check((metrics.coverage > .95) == bool(row["upstream_done"]), f"{label}: native success differs from coverage threshold")
        for metric in ("coverage", "iou"):
            maximum = max((r["metrics"][metric] for r in rows), default=0)
            final = rows[-1]["metrics"][metric] if rows else None
            check(near(episode.get(f"max_{metric}", maximum), maximum), f"{label}: max {metric} summary differs")
            check(episode.get(f"final_{metric}", final) == final, f"{label}: final {metric} summary differs")
        check(episode.get("contact_steps", 0) == sum(r["n_contacts"] > 0 for r in rows), f"{label}: contact summary differs")
        check(near(episode.get("max_reward", 0), max((r["reward"] for r in rows), default=0)), f"{label}: max reward differs")

        policy_times = episode["policy_seconds"]
        planning_times = episode["planning_seconds"]
        all_calls += len(policy_times)
        check(len(policy_times) == len(planning_times), f"{label}: policy/planning latency counts differ")
        plans = episode.get("simulator_plans", [])
        max_replay_error = max_prediction_error = 0.
        verified_next_steps = unexecuted_plans = verified_replays = 0
        selected_kinds, prefixes = Counter(), Counter()
        totals = Counter()
        for index, plan in enumerate(plans):
            candidates = plan["candidates"]
            selected = plan["selected"]
            selected_kinds[selected["kind"]] += 1
            prefixes[str(selected["policy_prefix"])] += 1
            totals.update(plan["counts"])
            check(plan["completed_candidates"] == len(candidates), f"{label}: completed-candidate count differs")
            check(plan["selected_index"] < len(candidates), f"{label}: invalid selected index")
            check(near(selected["score"], max(c["score"] for c in candidates)), f"{label}: did not select maximum score")
            elapsed_controls = plan["live_control_step"]-initial.control_step
            budget = min(plan["config"]["horizon"]+plan["config"]["settle_steps"], config["steps"]-elapsed_controls)
            check(plan["forecast_budget"] == budget, f"{label}: forecast exceeds remaining real budget")
            for candidate in candidates:
                verified_replays += 1
                max_replay_error = max(max_replay_error, candidate["replay_state_error"])
                check(near(candidate["replay_state_error"], 0), f"{label}: branch replay differs from live")
                forecast = candidate["steps"]
                check(len(forecast) <= budget, f"{label}: candidate forecast exceeds real budget")
                check(len(forecast) == candidate["forecast_steps"], f"{label}: candidate forecast count differs")
                check(len(forecast) == budget or forecast[-1]["native_done"], f"{label}: partial forecast was selectable")
                check(not any(r["native_done"] for r in forecast[:-1]), f"{label}: forecast continued after success")
                for metric in ("coverage", "iou"):
                    check(near(candidate[f"final_{metric}"], forecast[-1][metric]), f"{label}: candidate final metric differs")
            if not plan["timed_out"]:
                check(plan["counts"]["restores"] == len(candidates), f"{label}: restore count differs")
                check(plan["counts"]["replay_steps"] == elapsed_controls*len(candidates), f"{label}: replay-work count differs")
                check(plan["counts"]["candidate_steps"] == sum(c["forecast_steps"] for c in candidates), f"{label}: forecast-work count differs")
            next_row = actual.get(plan["live_control_step"]+1)
            if next_row is None:
                unexecuted_plans += 1
                check(index == len(plans)-1 and episode["stop_reason"] in ("wall_time_budget", "viewer_closed"), f"{label}: selected plan has no real continuation")
                continue
            predicted = Snapshot.from_dict(selected["first_state"])
            check(state_error(predicted, Snapshot.from_dict(plan["selected_predictions"][0])) <= 1e-7, f"{label}: selected first-state records differ")
            error = state_error(Snapshot.from_dict(next_row["state"]), predicted)
            max_prediction_error = max(max_prediction_error, error)
            check(error <= 1e-7, f"{label}: selected first prediction differs from next actual state")
            check(next_row["action"] == plan["selected_actions"][0], f"{label}: selected command differs from real command")
            first = selected["steps"][0]
            for key in ("coverage", "iou"):
                check(near(next_row["metrics"][key], first[key]), f"{label}: first predicted {key} differs from real")
            check(near(next_row["reward"], first["reward"]), f"{label}: predicted reward differs from real")
            check(next_row["upstream_done"] == first["native_done"], f"{label}: predicted done differs from real")
            check(next_row["n_contacts"] == first["n_contacts"], f"{label}: predicted contacts differ from real")
            verified_next_steps += 1
        saved_totals = episode.get("simulator_planning_counts", {})
        for key, count in totals.items():
            check(saved_totals[key] >= count, f"{label}: global simulator costs smaller than reported plans")
            if complete:
                check(saved_totals[key] == count, f"{label}: unreported simulator work in completed run")
        summaries.append({
            "shape": episode["shape"], "pair_index": episode["pair_index"], "scene_seed": episode["scene_seed"],
            "policy_seed": episode["policy_seed"], "control_steps": len(rows), "success": episode["success"],
            "stop_reason": episode["stop_reason"], "max_coverage": episode.get("max_coverage"),
            "final_coverage": episode.get("final_coverage"), "max_iou": episode.get("max_iou"),
            "final_iou": episode.get("final_iou"), "contact_steps": episode.get("contact_steps"),
            "first_max_coverage_control": max(range(len(rows)), key=lambda i: rows[i]["metrics"]["coverage"])+1 if rows else None,
            "contact_count_sum": sum(r["n_contacts"] for r in rows), "planning_counts": saved_totals,
            "policy_calls": len(policy_times), "policy_seconds_mean": mean(policy_times), "policy_seconds_total": sum(policy_times),
            "planning_seconds_mean": mean(planning_times), "planning_seconds_total": sum(planning_times),
            "timed_out_decisions": sum(p["timed_out"] for p in plans),
            "rejected_samples": sum(len(p["rejected_candidates"]) for p in plans),
            "selected_kinds": dict(selected_kinds), "selected_prefixes": dict(prefixes),
            "max_replay_state_error": max_replay_error, "max_next_state_prediction_error": max_prediction_error,
            "candidate_replays_verified": verified_replays,
            "verified_predicted_next_steps": verified_next_steps, "selected_plans_without_execution": unexecuted_plans,
            "inference_calls_without_completed_search_report": len(policy_times)-len(plans) if config["planner"] == "simulator" else 0,
            "max_stored_geometry_metric_error": geometry_error,
        })
    controls = sum(len(e["steps"]) for e in episodes)
    check(report["counts"]["control_steps"] == controls, "Top-level control count differs from episodes")
    check(report["counts"]["control_physics_substeps"] == controls*10, "Control physics-substep count differs")
    check(report["counts"]["reset_physics_substeps"] == sum(len(attempts) for attempts in report["reset_records"])*len(config["shapes"]), "Reset physics-substep count differs")
    check(report["counts"]["planning_calls"] == all_calls, "Planning-call count differs from actual inference calls")
    iterations = report["asset_spec"]["policy"]["num_diffusion_iters"]
    samples = config["candidates"] if config["planner"] == "simulator" else 1
    check(report["counts"]["candidate_sequences"] == all_calls*samples, "Candidate sampling count differs")
    check(report["counts"]["denoising_steps"] == all_calls*iterations, "Denoising iteration count differs")
    check(report["counts"]["denoising_sample_steps"] == all_calls*iterations*samples, "Sample-iteration count differs")
    return {"availability": "readable", "report_status": report["status"], "complete": complete,
            "audit_passed": not errors, "errors": errors, "requested_episodes": census,
            "config": {k: config[k] for k in ("steps", "planner", "execute_steps", "candidates", "prefix_lengths", "seed", "policy_seed")},
            "counts": report["counts"], "wall_seconds": report.get("evaluation_elapsed_seconds", report.get("elapsed_seconds")),
            "episodes": summaries}, report


def main():
    inputs = {"smoke": ROOT/"artifacts/td-simulator-smoke-20261021-r2/report.json",
              **{label: ROOT/f"artifacts/td-abc-20261021-{label}/report.json" for label in "ABC"},
              "A300": ROOT/"artifacts/td-baseline-300-20261021/report.json"}
    summary, raw = {"reports": {}, "matched_ABC": {"complete": False, "errors": []}}, {}
    for label, path in inputs.items():
        result, report = audit(path)
        summary["reports"][label] = result
        if report is not None:
            raw[label] = report
    available = [label for label in "ABC" if label in raw]
    if available:
        first = raw[available[0]]["config"]
        for label in available[1:]:
            for key in ("steps", "pairs", "seed", "policy_seed", "shapes"):
                if raw[label]["config"][key] != first[key]:
                    summary["matched_ABC"]["errors"].append(f"{label}: common comparison config {key} differs")
    reference = {}
    for label in available:
        for episode in raw[label]["episodes"]:
            key = (episode["pair_index"], episode["shape"])
            if key not in reference:
                reference[key] = (label, episode)
                continue
            other_label, other = reference[key]
            for field in ("scene_seed", "policy_seed"):
                if episode[field] != other[field]:
                    summary["matched_ABC"]["errors"].append(f"{label}/{other_label}/{key}: {field} differs")
            if state_error(Snapshot.from_dict(episode["initial_state"]), Snapshot.from_dict(other["initial_state"])) > 1e-7:
                summary["matched_ABC"]["errors"].append(f"{label}/{other_label}/{key}: initial state differs")
    summary["matched_ABC"]["complete"] = len(available) == 3 and all(summary["reports"][label].get("complete") for label in "ABC")
    summary["matched_ABC"]["audit_passed_for_available"] = not summary["matched_ABC"]["errors"]
    prefix_check = {"complete": False, "errors": [], "episodes": []}
    if "A" in raw and "A300" in raw:
        for short in raw["A"]["episodes"]:
            long = next((e for e in raw["A300"]["episodes"] if (e["pair_index"], e["shape"]) == (short["pair_index"], short["shape"])), None)
            if long is None:
                continue
            label = short["shape"]
            for key in ("scene_seed", "policy_seed", "initial_state"):
                if short[key] != long[key]:
                    prefix_check["errors"].append(f"{label}: A64/A300 {key} differs")
            count = min(len(short["steps"]), len(long["steps"]))
            maximum_error = maximum_score_error = 0.
            control_records_exact = score_records_exact = True
            for index, (left, right) in enumerate(zip(short["steps"], long["steps"])):
                error = state_error(Snapshot.from_dict(left["state"]), Snapshot.from_dict(right["state"]))
                maximum_error = max(maximum_error, error)
                for key in ("state", "action", "upstream_done", "n_contacts"):
                    if left[key] != right[key]:
                        control_records_exact = False
                        prefix_check["errors"].append(f"{label}: A64/A300 control {index+1} {key} differs")
                score_pairs = [(left["reward"], right["reward"]),
                               *((left["metrics"][key], right["metrics"][key]) for key in ("coverage", "iou"))]
                for a, b in score_pairs:
                    maximum_score_error = max(maximum_score_error, abs(a-b))
                    score_records_exact = score_records_exact and a == b
                    if not near(a, b):
                        prefix_check["errors"].append(f"{label}: A64/A300 control {index+1} reward/metric differs beyond tolerance")
            prefix_check["episodes"].append({"shape": label, "controls_compared": count,
                                              "expected_controls": len(short["steps"]),
                                              "max_snapshot_difference": maximum_error,
                                              "physical_action_done_contact_records_exact": control_records_exact,
                                              "score_records_exact": score_records_exact,
                                              "max_reward_coverage_iou_difference": maximum_score_error,
                                              "numeric_score_tolerance": 1e-7})
        prefix_check["complete"] = (len(prefix_check["episodes"]) == len(raw["A"]["episodes"])
                                    and all(e["controls_compared"] == e["expected_controls"] for e in prefix_check["episodes"]))
    prefix_check["audit_passed_for_available"] = not prefix_check["errors"]
    summary["A64_A300_prefix"] = prefix_check
    target = Path(__file__).with_name("review-summary.json")
    target.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"summary": str(target), "reports": {label: {k: result.get(k) for k in ("availability", "report_status", "audit_passed", "errors")} for label, result in summary["reports"].items()}, "matched_ABC": summary["matched_ABC"], "A64_A300_prefix": prefix_check}, indent=2))


if __name__ == "__main__":
    main()
