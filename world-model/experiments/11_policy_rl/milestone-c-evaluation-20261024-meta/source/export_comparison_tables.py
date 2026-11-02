"""Offline finite CSV/Markdown export; refuse absent or incomplete comparisons.

Use only after both physical evaluation groups have completed. This stdlib
helper validates the raw reports against the saved paired comparison before
creating its new output directory. It performs no runtime or ledger operations.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "tdyssey/policy_eval_comparison.py").is_file())
sys.path.insert(0, str(ROOT))
from tdyssey.policy_eval_comparison import METRICS, compare_evaluation_reports

META = Path(__file__).resolve().parent


def read_json(path):
    def reject(token):
        raise ValueError("Nonfinite JSON token: " + token)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Export refuses nonfinite data")
    if isinstance(value, dict):
        if "nonfinite" in value:
            raise ValueError("Export refuses explicit nonfinite diagnostics")
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty table")
    keys = list(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError("CSV rows have inconsistent columns")
    finite_tree(rows)
    with path.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summary_values(summary):
    result = {"n": summary["n"], "baseline_denominator": summary["baseline"]["denominator"],
        "delta_denominator": summary["delta"]["denominator"],
        "baseline_successes": summary["baseline"]["success_count"],
        "delta_successes": summary["delta"]["success_count"],
        "baseline_success_fraction": summary["baseline"]["success_fraction"],
        "delta_success_fraction": summary["delta"]["success_fraction"],
        "success_difference_pp": summary["success_difference_pp"]}
    result.update({"success_" + name: value for name, value in summary["paired_success"].items()})
    for metric in METRICS:
        values = summary["metrics"][metric]
        for owner in ("baseline", "delta", "paired_difference"):
            result.update({metric + "_" + owner + "_" + stat: number
                           for stat, number in values[owner].items()})
        result.update({metric + "_" + name: value for name, value in values["paired_outcomes"].items()})
    steps = summary["first_success_steps"]
    result.update(first_success_both_success_count=steps["both_success_count"],
                  first_success_censored_count=steps["censored_count"])
    result.update({"first_success_step_difference_" + stat: value
                   for stat, value in steps["paired_difference"].items()})
    result.update({"first_success_step_" + name: value for name, value in steps["paired_outcomes"].items()})
    return result


def case_values(case):
    row = {key: case[key] for key in ("shape", "scene_seed", "policy_seed", "success_difference")}
    for owner in ("baseline", "delta"):
        for key in ("status", "success", "success_step", "control_steps"):
            row[owner + "_" + key] = case[owner][key]
        row.update({owner + "_" + metric: case[owner][metric] for metric in METRICS})
    row.update({metric + "_difference": case["differences"][metric] for metric in METRICS})
    row.update(first_success_step_difference=case["success_step_difference"],
               first_success_censored=case["success_step_difference"] is None,
               first_success_comparison=case["success_step_comparison"])
    return row


def export(baseline_path, delta_path, comparison_path, output_path):
    paths = [Path(path).resolve() for path in (baseline_path, delta_path, comparison_path)]
    output = Path(output_path).resolve()
    if output == META or not output.is_relative_to(META):
        raise ValueError("Output must be a new subdirectory of this evaluation meta directory")
    if output.exists():
        raise FileExistsError("Output directory already exists; no previous export is overwritten")
    baseline, delta, saved = [read_json(path) for path in paths]
    for report in (baseline, delta, saved):
        finite_tree(report)
    if saved.get("status") != "completed":
        raise ValueError("A complete saved paired comparison is required; no partial export")
    computed = compare_evaluation_reports(baseline, delta)
    if computed["status"] != "completed" or computed != saved:
        raise ValueError("Raw baseline/delta reports differ from the completed saved comparison")
    summary = [{"shape": shape, **summary_values(saved["per_shape"][shape])} for shape in ("T", "L")]
    cases = [case_values(case) for case in saved["cases"]]
    scenes = []
    for scene in saved["scenes"]:
        row = {"scene_seed": scene["scene_seed"], "policy_seeds": ";".join(map(str, scene["policy_seeds"])),
               "policy_seeds_within_scene_correlated": True}
        for shape in ("T", "L"):
            row.update({shape + "_" + key: value for key, value in summary_values(scene["per_shape"][shape]).items()})
        scenes.append(row)
    if len(summary) != 2 or len(cases) != 20 or len(scenes) != 5:
        raise ValueError("Expected T/L summary, 20 paired cases and five scene blocks")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "shape_summary.csv", summary)
    write_csv(output / "paired_cases.csv", cases)
    write_csv(output / "scene_blocks.csv", scenes)
    lines = ["# Fixed C base/delta development comparison", "",
        "Both methods use the same training sampler with std floor 0.01. Differences are delta minus baseline.", "",
        "| Shape | Base successes/10 | Delta successes/10 | Difference (pp) | Success win/tie/loss | Mean final coverage difference | Mean peak coverage difference |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['shape']} | {row['baseline_successes']}/10 | {row['delta_successes']}/10 | "
            f"{row['success_difference_pp']:.6g} | {row['success_win']}/{row['success_tie']}/{row['success_loss']} | "
            f"{row['final_coverage_paired_difference_mean']:.6g} | {row['peak_coverage_paired_difference_mean']:.6g} |")
    lines += ["", "Coverage and IoU CSVs include method mean/median, paired difference mean/median and win/tie/loss at tolerance 1e-9.",
        "Success step differences use only pairs where both succeed; negative means delta succeeded earlier. Other pairs remain censored and blank, never assigned 300.",
        "There are five scene blocks per shape. Two policy seeds within a scene are correlated; these are not ten independent scenes.",
        "This is a fixed development comparison, not an untouched test or original-sampler baseline. No training or new evaluation is performed by this export.", ""]
    with (output / "summary.md").open("x", encoding="utf-8") as out:
        out.write("\n".join(lines))
    manifest = {"kind": "tdyssey.C.offline_comparison_tables", "status": "completed", "training_steps": 0,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "rows": {"shape_summary.csv": 2, "paired_cases.csv": 20, "scene_blocks.csv": 5},
        "metric_tolerance": saved["limits"]["metric_tolerance"], "limitations": saved["limits"]}
    with (output / "manifest.json").open("x", encoding="utf-8") as out:
        json.dump(manifest, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    return {"status": "completed", "output_dir": str(output), "rows": manifest["rows"], "training_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.baseline, args.delta, args.comparison, args.output_dir), sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
