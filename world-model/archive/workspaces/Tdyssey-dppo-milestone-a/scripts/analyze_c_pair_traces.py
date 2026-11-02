"""Offline C coverage/action-target diagnosis from completed saved reports only.

Control 0 is the saved initial measurement. Controls 1..N are the saved executed
step order, not physics substeps. Action targets are commands, not measured
pusher/object paths. No simulator, tensor runtime or checkpoint is imported.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.policy_eval_comparison import compare_evaluation_reports

TOLERANCE = 1e-9
NEAR_SUCCESS = .90
ACTION_PREFIX = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, fraction=False):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    require(not fraction or 0 <= value <= 1, name + " must be in [0,1]")
    return value


def read_json(path):
    def reject(token):
        raise ValueError("Nonfinite JSON token " + token)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def key(case):
    return case["shape"], case["scene_seed"], case["policy_seed"]


def runs(values, predicate):
    """Inclusive executed control ranges; initial control zero is excluded."""
    result, start = [], None
    for index, value in enumerate(values, 1):
        if predicate(value) and start is None:
            start = index
        if start is not None and (not predicate(value) or index == len(values)):
            end = index if predicate(value) else index - 1
            result.append({"start_control": start, "end_control": end, "length": end - start + 1})
            start = None
    return result


def mean(values):
    return math.fsum(values) / len(values) if values else None


def analyze_case(case):
    require(case.get("status") == "completed", "Trace diagnosis requires completed cases")
    steps = case.get("steps")
    require(isinstance(steps, list) and steps, "Executed trace is missing")
    n = case["counts"]["control_steps"]
    require(type(n) is int and n == len(steps) and 1 <= n <= 300, "Executed trace length/count differs")
    initial = number(case["initial_metrics"]["coverage"], "initial coverage", fraction=True)
    initial_iou = number(case["initial_metrics"]["iou"], "initial IoU", fraction=True)
    coverage, iou, actions = [], [], []
    for step in steps:
        coverage.append(number(step.get("coverage"), "coverage", fraction=True))
        iou.append(number(step.get("iou"), "IoU", fraction=True))
        action = step.get("action")
        require(isinstance(action, list) and len(action) == 2, "Saved action target must have two coordinates")
        actions.append([number(value, "action target") for value in action])
    all_coverage, all_iou = [initial] + coverage, [initial_iou] + iou
    peak, peak_iou = max(all_coverage), max(all_iou)
    require(abs(case["final_metrics"]["coverage"] - coverage[-1]) <= 1e-12
            and abs(case["final_metrics"]["iou"] - iou[-1]) <= 1e-12, "Saved final metrics differ from trace")
    require(abs(case["peak_metrics"]["coverage"] - peak) <= 1e-12
            and abs(case["peak_metrics"]["iou"] - peak_iou) <= 1e-12, "Saved peak metrics differ from trace including control0")
    successes = [index for index, value in enumerate(coverage, 1) if value > .95]
    require(type(case.get("success")) is bool and bool(successes) == case["success"], "Strict executed success differs")
    require(case["success_step"] == (successes[0] if successes else None), "First success step differs")
    require(not successes or successes[0] == n, "Execution continued after first success")
    if case["stop_reason"] == "control_cap":
        require(n == 300, "A control-cap termination must execute300 controls")
    changes = [current - previous for previous, current in zip(all_coverage, coverage)]
    peak_control = all_coverage.index(peak)
    first_positive = next((index for index, value in enumerate(all_coverage) if value > 0), None)
    first_positive_executed = next((index for index, value in enumerate(coverage, 1) if value > 0), None)
    near = runs(coverage, lambda value: NEAR_SUCCESS <= value <= .95)
    plans = case["counts"]["planning_calls"]
    planning_latencies = case.get("latency", {}).get("planning_seconds")
    inferred = type(plans) is int and plans == math.ceil(n / ACTION_PREFIX)
    if planning_latencies is not None:
        require(isinstance(planning_latencies, list) and len(planning_latencies) == plans,
                "Saved planning latency/count differs")
    plan_starts = list(range(1, n + 1, ACTION_PREFIX)) if inferred else []
    target_jumps = [math.dist(previous, current) for previous, current in zip(actions, actions[1:])]
    summary = {"method": case["method"], "shape": case["shape"], "scene_seed": case["scene_seed"],
        "policy_seed": case["policy_seed"], "controls": n, "success": case["success"],
        "success_step": case["success_step"], "stop_reason": case["stop_reason"],
        "reached_control_cap": n == 300, "initial_coverage": initial,
        "first_positive_coverage_control": first_positive,
        "first_positive_executed_control": first_positive_executed,
        "peak_coverage": peak, "peak_control_first": peak_control,
        "final_coverage": coverage[-1], "peak_final_coverage_loss": peak - coverage[-1],
        "controls_after_first_peak": n - peak_control, "executed_coverage_mean": mean(coverage),
        "positive_coverage_count": sum(value > 0 for value in coverage),
        "positive_coverage_fraction": sum(value > 0 for value in coverage) / n,
        "initial_iou": initial_iou, "peak_iou": peak_iou, "final_iou": iou[-1],
        "peak_final_iou_loss": peak_iou - iou[-1],
        "coverage_total_variation": math.fsum(abs(value) for value in changes),
        "largest_single_control_gain": max(0, max(changes)),
        "largest_single_control_gain_index": changes.index(max(changes)) + 1 if max(changes) > 0 else None,
        "largest_single_control_drop": min(0, min(changes)),
        "largest_single_control_drop_index": changes.index(min(changes)) + 1 if min(changes) < 0 else None,
        "near_success_controls": sum(segment["length"] for segment in near),
        "near_success_segments": near,
        "coverage_at_least_0_50_controls": sum(value >= .5 for value in coverage),
        "coverage_at_least_0_90_controls": sum(value >= .9 for value in coverage),
        "coverage_at_least_0_94_controls": sum(value >= .94 for value in coverage),
        "coverage_strictly_above_0_95_controls": len(successes),
        "plan_count": plans, "plan_start_controls": plan_starts,
        "plan_boundaries_inferred": inferred,
        "plan_boundary_basis": "locked8_control_prefix_and_saved_count; not explicitly logged" if inferred else "unavailable",
        "action_target_variation_length": math.fsum(target_jumps),
        "action_target_mean_jump": mean(target_jumps),
        "action_target_max_jump": max(target_jumps) if target_jumps else None,
        "action_target_max_jump_control": target_jumps.index(max(target_jumps)) + 2 if target_jumps else None,
        "action_target_bounds": {axis: [min(a[index] for a in actions), max(a[index] for a in actions)]
                                 for index, axis in enumerate(("x", "y"))}}
    series = []
    for index, value in enumerate(all_coverage):
        series.append({"method": case["method"], "shape": case["shape"], "scene_seed": case["scene_seed"],
            "policy_seed": case["policy_seed"], "control": index, "coverage": value, "iou": all_iou[index],
            "coverage_change": None if index == 0 else changes[index - 1],
            "action_target_x": None if index == 0 else actions[index - 1][0],
            "action_target_y": None if index == 0 else actions[index - 1][1],
            "inferred_plan_start": index in plan_starts,
            "measurement": "initial" if index == 0 else "after_executed_control"})
    return summary, series


def paired_case(base, delta, base_series, delta_series):
    require(key(base) == key(delta), "Paired case keys differ")
    n = min(base["controls"], delta["controls"])
    contrasts, opposite = [], []
    for left, right in zip(base_series[:n + 1], delta_series[:n + 1]):
        control = left["control"]
        record = {"shape": base["shape"], "scene_seed": base["scene_seed"], "policy_seed": base["policy_seed"],
            "control": control, "baseline_coverage": left["coverage"], "delta_coverage": right["coverage"],
            "coverage_difference": right["coverage"] - left["coverage"],
            "baseline_coverage_change": left["coverage_change"], "delta_coverage_change": right["coverage_change"],
            "opposite_coverage_changes": False, "action_target_distance": None}
        if control:
            a, b = left["coverage_change"], right["coverage_change"]
            record["opposite_coverage_changes"] = (a > TOLERANCE and b < -TOLERANCE) or (a < -TOLERANCE and b > TOLERANCE)
            record["action_target_distance"] = math.dist([left["action_target_x"], left["action_target_y"]],
                                                         [right["action_target_x"], right["action_target_y"]])
            if record["opposite_coverage_changes"]:
                opposite.append(record)
        contrasts.append(record)
    metric_names = ("initial_coverage", "peak_coverage", "final_coverage", "peak_final_coverage_loss",
                    "positive_coverage_fraction", "near_success_controls", "executed_coverage_mean")
    row = {"shape": base["shape"], "scene_seed": base["scene_seed"], "policy_seed": base["policy_seed"],
        "baseline_success": base["success"], "delta_success": delta["success"],
        "baseline_controls": base["controls"], "delta_controls": delta["controls"],
        "shared_executed_controls": n, "success_difference": int(delta["success"]) - int(base["success"]),
        "baseline_peak_control": base["peak_control_first"], "delta_peak_control": delta["peak_control_first"],
        "baseline_first_positive": base["first_positive_coverage_control"],
        "delta_first_positive": delta["first_positive_coverage_control"],
        "both_success_step_difference": delta["success_step"] - base["success_step"] if base["success"] and delta["success"] else None,
        "opposite_change_control_count": len(opposite),
        "opposite_change_examples": sorted(opposite, key=lambda x: abs(x["delta_coverage_change"] - x["baseline_coverage_change"]), reverse=True)[:5],
        "shared_action_target_mean_distance": mean([x["action_target_distance"] for x in contrasts[1:]]),
        "shared_action_target_max_distance": max(x["action_target_distance"] for x in contrasts[1:]),
        "first_action_target_difference_control": next((x["control"] for x in contrasts[1:] if x["action_target_distance"] > TOLERANCE), None)}
    for name in metric_names:
        row["baseline_" + name] = base[name]
        row["delta_" + name] = delta[name]
        row[name + "_difference"] = delta[name] - base[name]
    return row, contrasts


def analyze_pair(baseline, delta):
    comparison = compare_evaluation_reports(baseline, delta)
    require(comparison["status"] == "completed", "Both complete groups are required")
    summaries, all_series, pairs, contrasts = [], [], [], []
    for left, right in zip(baseline["cases"], delta["cases"]):
        require(left["initial_state"] == right["initial_state"], "Paired saved initial states differ")
        a, a_series = analyze_case(left)
        b, b_series = analyze_case(right)
        summaries.extend((a, b)); all_series.extend(a_series + b_series)
        pair, joined = paired_case(a, b, a_series, b_series)
        pairs.append(pair); contrasts.extend(joined)
    scene_rows = []
    for shape in ("T", "L"):
        for scene in range(100000, 100005):
            selected = [row for row in pairs if row["shape"] == shape and row["scene_seed"] == scene]
            scene_rows.append({"shape": shape, "scene_seed": scene, "policy_seed_count": len(selected),
                "policy_seeds_correlated": True,
                "baseline_successes": sum(row["baseline_success"] for row in selected),
                "delta_successes": sum(row["delta_success"] for row in selected),
                **{name: mean([row[name] for row in selected]) for name in (
                    "final_coverage_difference", "peak_coverage_difference", "peak_final_coverage_loss_difference",
                    "positive_coverage_fraction_difference", "near_success_controls_difference")}})
    return {"schema_version": 1, "kind": "tdyssey.C.offline_trace_diagnosis", "status": "completed",
        "new_training_steps": 0, "new_physical_steps": 0, "new_policy_samples": 0,
        "definitions": {"control_index": "0 initial measurement;1..N saved executed-step order, not physics substeps",
            "positive_coverage": "strictly >0; fraction denominator is actually executed controls, excluding initial",
            "peak": "maximum includes initial control0; earliest exact maximum index is reported",
            "near_success": "0.90 <= coverage <=0.95 on executed controls; success remains strictly >0.95",
            "opposite_changes": "same executed-control ordinal: one coverage increment >1e-9 and the other <-1e-9; not matched physical states",
            "plan_starts": "inferred from locked8-action prefix only where saved planning count agrees; not explicitly logged",
            "action_targets": "absolute commanded targets; variation/distance is not measured pusher/object travel",
            "comparison_horizon": "each trace ends at actual termination; shared-prefix contrasts stop at min(Nbase,Ndelta), with no forward fill"},
        "case_summaries": summaries, "paired_cases": pairs, "per_scene": scene_rows,
        "series": all_series, "paired_series": contrasts,
        "limitations": ["No per-step object/pusher poses, contacts, RGB or explicit plan IDs are saved. Initial poses do not identify later motion or contact.",
            "Coverage and target-command variation can describe divergence and loss of overlap, not its physical cause.",
            "Two policy seeds within each of five scenes are correlated. These are fixed development traces, not independent-scene significance or unseen-shape evidence.",
            "Early success stops produce unequal horizons. Positive fractions and mean coverage describe the recorded horizon, not equal-duration counterfactual outcomes."]}


def csv_rows(path, rows):
    require(bool(rows), "Refusing an empty CSV")
    def value(item):
        return json.dumps(item, sort_keys=True, allow_nan=False) if isinstance(item, (dict, list)) else item
    with path.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({name: value(item) for name, item in row.items()} for row in rows)


def make_plots(result, directory):
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib-cache"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc), "files": []}
    series = result["series"]
    colors = {"base_training": "#1764ab", "delta_training": "#d95f02"}
    output = []
    def saved(fig, name):
        for extension in ("png", "pdf"):
            path = directory / (name + "." + extension)
            fig.savefig(path, dpi=170, bbox_inches="tight")
            output.append(path.name)
        plt.close(fig)
    for shape in ("T", "L"):
        for plot_kind in ("coverage", "action_targets"):
            fig, axes = plt.subplots(5, 2, figsize=(12, 14), squeeze=False)
            for row, scene in enumerate(range(100000, 100005)):
                for col, seed in enumerate((5478, 5479)):
                    ax = axes[row][col]
                    for method in colors:
                        points = [p for p in series if (p["shape"], p["scene_seed"], p["policy_seed"], p["method"]) == (shape, scene, seed, method)]
                        if plot_kind == "coverage":
                            ax.plot([p["control"] for p in points], [p["coverage"] for p in points], color=colors[method], label=method, lw=1.5)
                        else:
                            targets = points[1:]
                            x, y = [p["action_target_x"] for p in targets], [p["action_target_y"] for p in targets]
                            ax.plot(x, y, color=colors[method], label=method, lw=.9, alpha=.8)
                            ax.scatter(x[0], y[0], color=colors[method], marker="o", s=22)
                            ax.scatter(x[-1], y[-1], color=colors[method], marker="x", s=32)
                    ax.set_title(f"{shape} scene {scene}, policy seed {seed}", fontsize=10)
                    ax.grid(alpha=.18)
                    if plot_kind == "coverage":
                        ax.axhline(.95, color="black", ls="--", lw=.8)
                        ax.axhspan(.9, .95, color="gray", alpha=.09)
                        ax.set(xlim=(0, 300), ylim=(0, 1.01), xlabel="Executed control (0=initial)", ylabel="Coverage")
                    else:
                        ax.set(xlabel="Target x (units)", ylabel="Target y (units)")
                        ax.set_aspect("equal", adjustable="box")
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, .945))
            suffix = "No extrapolation beyond actual stop; dashed threshold .95; shaded .90..95." if plot_kind == "coverage" else "Command sequences in environment units, not measured paths. Circle=start, cross=end; no contact/pose inference."
            fig.suptitle(f"{shape}: saved {plot_kind.replace('_', ' ')}\n{suffix}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .915))
            saved(fig, shape + "_" + plot_kind)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for method in colors:
        points = [p for p in series if (p["shape"], p["scene_seed"], p["policy_seed"], p["method"]) == ("T", 100001, 5478, method)]
        axes[0].plot([p["control"] for p in points], [p["coverage"] for p in points], label=method, color=colors[method])
        for axis, name in zip(axes[1:], ("action_target_x", "action_target_y")):
            axis.plot([p["control"] for p in points[1:]], [p[name] for p in points[1:]], color=colors[method])
            for p in points[1:]:
                if p["inferred_plan_start"]:
                    axis.axvline(p["control"], color=colors[method], alpha=.08, lw=.5)
    axes[0].axhline(.95, color="black", ls="--", lw=.8)
    axes[0].set(ylabel="Coverage", ylim=(0, 1.01), xlim=(0, 300))
    axes[0].legend()
    axes[1].set_ylabel("Command target x")
    axes[2].set(ylabel="Command target y", xlabel="Actually executed control index")
    for ax in axes: ax.grid(alpha=.15)
    fig.suptitle("T scene100001 / seed5478: lost success\nTargets are commands, not measured paths. Faint target-panel lines are inferred8-control plan starts.")
    fig.tight_layout(rect=(0, 0, 1, .94))
    saved(fig, "T_lost_success_detail")
    return {"status": "completed", "matplotlib_version": matplotlib.__version__, "files": output}


def export(baseline_path, delta_path, output_dir, *, plots=False):
    output = Path(output_dir).resolve()
    require(not output.exists(), "Output must be a new directory; existing derived evidence is never overwritten")
    inputs = [Path(p).resolve() for p in (baseline_path, delta_path)]
    identities = {str(p): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs}
    result = analyze_pair(*(read_json(p) for p in inputs))
    output.mkdir(parents=True, exist_ok=False)
    for filename, field in (("cases.csv", "case_summaries"), ("pairs.csv", "paired_cases"),
                            ("scene_rows.csv", "per_scene"), ("series.csv", "series"), ("paired_series.csv", "paired_series")):
        csv_rows(output / filename, result[field])
    figure_status = make_plots(result, output) if plots else {"status": "not_requested", "files": []}
    summary = {k: v for k, v in result.items() if k not in ("series", "paired_series")}
    summary.update(inputs=identities, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   figures=figure_status, series_rows=len(result["series"]), paired_series_rows=len(result["paired_series"]))
    require(all(hashlib.sha256(p.read_bytes()).hexdigest() == identities[str(p)]["sha256"] for p in inputs), "Input changed during offline diagnosis")
    with (output / "analysis.json").open("x", encoding="utf-8") as out:
        json.dump(summary, out, sort_keys=True, indent=2, allow_nan=False); out.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Use installed Matplotlib if available; never install")
    args = parser.parse_args()
    try:
        result = export(args.baseline, args.delta, args.output_dir, plots=args.plots)
        print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir),
                          "cases": len(result["case_summaries"]), "pairs": len(result["paired_cases"]),
                          "figures": result["figures"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
