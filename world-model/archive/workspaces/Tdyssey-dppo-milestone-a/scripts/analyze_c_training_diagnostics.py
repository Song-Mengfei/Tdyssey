"""Read saved C scalar evidence with the standard library; never load tensors.

Writes derived files only to the offline diagnosis's new training directory.
Per-shape physical measurements and mixed-buffer optimizer statistics are kept
separate. Repeated PPO passes reuse a buffer and are not new physical samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/milestone-c-offline-diagnosis-20261024/training-final")
STAT_NAMES = ("old_values", "raw_advantages", "advantages", "returns")
SUM_FIELDS = ("controls", "plans", "contact_controls", "unknown_contact_controls",
              "nonzero_raw_reward_controls", "nonzero_training_reward_controls",
              "nonzero_shaping_controls", "raw_reward_sum", "training_reward_sum",
              "shaping_sum", "coverage_delta_sum", "absolute_coverage_delta")
SOURCE_INPUTS = ("tdyssey/policy_returns.py", "tdyssey/policy_stream.py",
                 "tdyssey/policy_training.py", "tdyssey/policy_ppo.py",
                 "tdyssey/policy_experiment.py")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def input_identity(path, root):
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def scalar_summary(values):
    values = [float(x) for x in values if x is not None]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Scalar diagnostics must be finite")
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None}


def pool_moments(groups):
    """Combine population moments, counting each fresh plan exactly once."""
    groups = list(groups)
    count = sum(x["count"] for x in groups)
    if not count:
        raise ValueError("Cannot pool an empty sample")
    if any(type(x["count"]) is not int or x["count"] <= 0 or x["std"] < 0
           or any(not math.isfinite(x[k]) for k in ("mean", "std", "min", "max"))
           for x in groups):
        raise ValueError("Invalid population moments")
    mean = math.fsum(x["count"] * x["mean"] for x in groups) / count
    variance = math.fsum(x["count"] * (x["std"] ** 2 + (x["mean"] - mean) ** 2)
                         for x in groups) / count
    return {"count": count, "mean": mean, "std": math.sqrt(variance),
            "min": min(x["min"] for x in groups), "max": max(x["max"] for x in groups)}


def aggregate_shapes(batches):
    result = {}
    for shape in sorted({s for b in batches for s in b["collection"]["by_shape"]}):
        samples = [b["collection"]["by_shape"][shape] for b in batches]
        active = [s for s in samples if s["controls"]]
        stats = {k: math.fsum(x[k] for x in samples) for k in SUM_FIELDS}
        for k in SUM_FIELDS[:7]:
            stats[k] = int(stats[k])
        stats.update(max_coverage=max(s["max_coverage"] for s in samples),
                     last_coverage=active[-1]["last_coverage"] if active else None)
        stats["first_success_bonus_sum"] = stats["training_reward_sum"] - stats["shaping_sum"]
        controls = stats["controls"]
        stats["contact_fraction"] = stats["contact_controls"] / controls if controls else None
        stats["raw_reward_mean_per_control"] = stats["raw_reward_sum"] / controls if controls else None
        stats["training_reward_mean_per_control"] = stats["training_reward_sum"] / controls if controls else None
        stats["control_share"] = controls / sum(b["collection"]["controls"] for b in batches)
        result[shape] = stats
    return result


def optimizer_summary(passes, max_grad_norm):
    scalar_keys = ("actor_loss", "critic_loss", "critic_mse", "critic_explained_variance",
                   "actor_grad_norm", "critic_grad_norm")
    result = {k: scalar_summary(p[k] for p in passes) for k in scalar_keys}
    for key in ("sample_approx_kl", "clip_fraction"):
        result[key] = scalar_summary(p["after"][key] for p in passes)
    for owner in ("actor", "critic"):
        result[owner + "_step_delta_l2"] = scalar_summary(p[owner + "_weight_change"]["l2"] for p in passes)
        result[owner + "_positive_step_delta_count"] = sum(p[owner + "_weight_change"]["l2"] > 0 for p in passes)
        result[owner + "_preclip_norm_over_limit_count"] = sum(p[owner + "_grad_norm"] > max_grad_norm for p in passes)
    result["negative_explained_variance_count"] = sum(p["critic_explained_variance"] is not None
                                                        and p["critic_explained_variance"] < 0 for p in passes)
    result["undefined_explained_variance_count"] = sum(p["critic_explained_variance"] is None for p in passes)
    result["scope"] = "Unweighted summary of pass diagnostics on their own fixed mixed buffers; not held-out performance"
    return result


def physical_runs(shape_rows, gamma, beta):
    """Find consecutive logged plateaus, without inventing episode identities."""
    findings = {"flat_positive_coverage_no_contact": [], "zero_contact_zero_raw_reward": []}
    for name in findings:
        for shape in sorted({r["shape"] for r in shape_rows}):
            current = []

            def finish():
                if not current:
                    return
                summary = {"shape": shape, "batches": [r["batch"] for r in current],
                           "controls": sum(r["controls"] for r in current),
                           "contact_controls": sum(r["contact_controls"] for r in current),
                           "raw_reward_sum": math.fsum(r["raw_reward_sum"] for r in current),
                           "training_reward_sum": math.fsum(r["training_reward_sum"] for r in current),
                           "coverage": current[-1]["last_coverage"]}
                if name == "flat_positive_coverage_no_contact":
                    summary["expected_flat_training_sum"] = summary["controls"] * beta * (gamma - 1) * summary["coverage"]
                findings[name].append(summary)
                current.clear()

            for row in [r for r in shape_rows if r["shape"] == shape]:
                valid = (row["controls"] > 0 and row["unknown_contact_controls"] == 0
                         and row["contact_controls"] == 0 and row["absolute_coverage_delta"] == 0)
                if name == "flat_positive_coverage_no_contact":
                    valid = valid and row["last_coverage"] > 0
                else:
                    valid = valid and row["nonzero_raw_reward_controls"] == 0
                if not valid or (current and (row["batch"] != current[-1]["batch"] + 1
                                              or row["last_coverage"] != current[-1]["last_coverage"])):
                    finish()
                if valid:
                    current.append(row)
            finish()
    return findings


def reward_examples(gamma, beta, gae_lambda):
    """Exercise only the project's pure arithmetic, with explicit toy values."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tdyssey.policy_returns import ActionBlock, BlockBoundary, plan_gae, training_reward

    examples = {}
    for name, before, after in (("zero_stagnation", 0., 0.), ("positive_stagnation", .4, .4),
                                ("small_progress", .1, .101), ("larger_progress", .1, .102),
                                ("success", .94, .96)):
        r = training_reward(before, after, 17., gamma=gamma, beta=beta)
        examples[name] = {"current_coverage": before, "next_coverage": after,
                          "raw_reward_preserved": r.upstream_reward, "reward": r.reward,
                          "shaping": r.shaping, "next_potential": r.next_potential,
                          "first_success": r.first_success}
    rs = tuple(training_reward(a, b, 0., gamma=gamma, beta=beta).reward
               for a, b in ((.7, .8), (.8, .94), (.94, .96)))
    for name, boundary, rewards in (("success_third_control", BlockBoundary.SUCCESS, rs),
                                    ("five_control_timeout", BlockBoundary.TIMEOUT, (.01,) * 5),
                                    ("five_control_buffer_cutoff", BlockBoundary.BUFFER_CUTOFF, (.01,) * 5)):
        block = ActionBlock(episode_id="toy", rewards=rewards, value=.2,
                            final_observation_value=123. if boundary == BlockBoundary.SUCCESS else .7,
                            boundary=boundary)
        e = plan_gae([block], gamma=gamma, gae_lambda=gae_lambda)[0]
        examples[name] = {"actual_controls": len(rewards), "bootstrap_mask": block.bootstrap_mask,
                          "continuation_mask": block.continuation_mask, "physical_discount": e.physical_discount,
                          "discounted_reward": e.discounted_reward, "advantage": e.advantage,
                          "value_target": e.value_target}
    # A fully terminated shaped trajectory telescopes; local negative rewards
    # do not by themselves prove a wrong ideal discounted objective.
    examples["success_telescoping"] = {
        "measured_discounted_total": examples["success_third_control"]["discounted_reward"],
        "expected_discounted_bonus_minus_initial_potential": gamma ** 2 - beta * .7}
    examples["discount_horizon"] = {"gamma": gamma, "gamma_8": gamma ** 8,
                                     "gamma_300": gamma ** 300,
                                     "gamma_8_times_plan_lambda": gamma ** 8 * gae_lambda}
    return examples


def analyze_rows(rows, report, config, checkpoint_review=None):
    """Reject inconsistencies instead of inventing missing per-shape values."""
    batches = [r for r in rows if r.get("event") == "batch"]
    if not batches or [b["batch"] for b in batches] != list(range(1, len(batches) + 1)):
        raise ValueError("Expected one ordered fresh-batch record per batch")
    standalone = {(p["batch"], p["index"]): p for p in rows if p.get("event") == "ppo_pass"}
    if len(standalone) != sum(r.get("event") == "ppo_pass" for r in rows):
        raise ValueError("Duplicate standalone PPO pass")
    passes, batch_table, shape_table, pass_table = [], [], [], []
    cumulative = 0
    for ordinal, b in enumerate(batches):
        collection, update = b["collection"], b["update"]
        shapes = collection["by_shape"]
        if any(sum(s[k] for s in shapes.values()) != collection[k] for k in ("controls", "plans")):
            raise ValueError("Per-shape physical counts do not match mixed batch")
        if any(s["contact_controls"] + s["unknown_contact_controls"] > s["controls"] for s in shapes.values()):
            raise ValueError("Invalid contact accounting")
        stats = update["data_statistics"]
        if any(stats[k]["count"] != collection["plans"] for k in STAT_NAMES):
            raise ValueError("Mixed statistic count must equal fresh plans")
        if not math.isclose(stats["returns"]["mean"], stats["old_values"]["mean"] + stats["raw_advantages"]["mean"], abs_tol=1e-6):
            raise ValueError("Value targets do not match raw advantage plus old value")
        if update["completed_passes"] != len(update["passes"]):
            raise ValueError("PPO pass count mismatch")
        cumulative += collection["controls"]
        stage = ("early", "middle", "late")[min(2, ordinal * 3 // len(batches))]
        table = {"batch": b["batch"], "stage": stage, "controls": collection["controls"],
                 "cumulative_controls": cumulative, "plans": collection["plans"],
                 "passes": update["completed_passes"], "collection_seconds": b.get("collection_seconds"),
                 "update_seconds": b.get("update_seconds"), "entry_kl": update["entry"]["sample_approx_kl"],
                 "entry_clip_fraction": update["entry"]["clip_fraction"],
                 "entry_max_logprob_error": update["entry"]["max_logprob_error"],
                 "frozen_unchanged": update["frozen_unchanged"]}
        for k in STAT_NAMES:
            table.update({k + "_" + field: value for field, value in stats[k].items()})
        for shape, s in sorted(shapes.items()):
            table[shape + "_controls"] = s["controls"]
            bonus = s["training_reward_sum"] - s["shaping_sum"]
            if bonus < -1e-8 or not math.isclose(bonus, round(bonus), abs_tol=1e-8):
                raise ValueError("Training-minus-shaping must count first-success bonuses")
            shape_table.append({"batch": b["batch"], "stage": stage, "shape": shape, **s,
                                "first_success_bonus_sum": bonus,
                                "contact_fraction": s["contact_controls"] / s["controls"] if s["controls"] else None,
                                "raw_reward_mean_per_control": s["raw_reward_sum"] / s["controls"] if s["controls"] else None,
                                "training_reward_mean_per_control": s["training_reward_sum"] / s["controls"] if s["controls"] else None})
        for index, p in enumerate(update["passes"]):
            key = (b["batch"], index)
            if p["index"] != index or key not in standalone or {k: v for k, v in standalone[key].items() if k not in ("event", "batch")} != p:
                raise ValueError("Standalone and nested PPO diagnostics differ")
            passes.append(p)
            row = {"batch": b["batch"], "stage": stage, "pass_index": index,
                   "controls_in_reused_buffer": collection["controls"], "plans_in_reused_buffer": collection["plans"]}
            row.update({k: v for k, v in p.items() if not isinstance(v, (dict, list)) and k != "index"})
            row.update(p["after"])
            for owner in ("actor", "critic"):
                row.update({owner + "_step_delta_" + k: v for k, v in p[owner + "_weight_change"].items()
                            if not isinstance(v, (dict, list))})
            pass_table.append(row)
        batch_table.append(table)
    if len(passes) != len(standalone):
        raise ValueError("Orphaned standalone PPO diagnostic")
    result = report["result"]
    shapes = aggregate_shapes(batches)
    if result["controls"] != cumulative or result["batches"] != len(batches) or result["actor_steps"] != len(passes) or result["critic_steps"] != len(passes):
        raise ValueError("Saved final report and training log counts differ")
    if report["config_hash"] != canonical_hash(config):
        raise ValueError("Current canonical config differs from training binding")
    for shape, s in shapes.items():
        final = result["diagnostics_by_shape"][shape]
        if any(not math.isclose(s[k], final[k], abs_tol=1e-9) for k in SUM_FIELDS):
            raise ValueError("Saved per-shape totals differ from fresh-batch aggregation")
    stages = []
    for stage in ("early", "middle", "late"):
        indices = [i for i, row in enumerate(batch_table) if row["stage"] == stage]
        if not indices:
            continue
        selected = [batches[i] for i in indices]
        selected_passes = [p for i in indices for p in batches[i]["update"]["passes"]]
        stages.append({"stage": stage, "batches": [b["batch"] for b in selected],
                       "controls": sum(b["collection"]["controls"] for b in selected),
                       "by_shape": aggregate_shapes(selected),
                       "mixed_fresh_plan_statistics": {k: pool_moments(b["update"]["data_statistics"][k] for b in selected) for k in STAT_NAMES},
                       "pass_diagnostics": optimizer_summary(selected_passes, config["optimization"]["max_grad_norm"])})
    checkpoints = []
    for cp in (checkpoint_review or {}).get("checkpoints", []):
        m = cp["metadata"]
        if m["config_hash"] != report["config_hash"] or m["experiment_id"] != report["experiment_id"]:
            raise ValueError("Existing checkpoint metadata binding differs")
        checkpoints.append({k: cp[k] for k in ("path", "bytes", "sha256", "all_tensors_finite", "all_tensors_on_cpu", "optimizer_steps", "selected_actor_parameters", "critic_parameters")})
        checkpoints[-1].update(resumable_at_save=m["resumable"], terminal=m["terminal"],
                               controls=m["counts"]["completed"]["train"]["control"])
    limitations = [
        "Returns, raw/normalized advantages, old values and critic diagnostics are mixed T/L buffer statistics. Per-shape versions cannot be reconstructed from these scalar logs.",
        "Critic MSE and explained variance evaluate the same fixed buffer used for fitting, after each pass; they are not held-out estimates or task success rates.",
        "Each fresh batch is counted once. Repeated passes are correlated optimizer diagnostics, not additional controls or independent trajectories.",
        "Coverage deltas omit jumps across reset. Aggregate maxima and contact counts do not reconstruct trajectories or prove useful directed contacts.",
        "Three equal batch-index stages have different control counts, scene exposure and boundary-drain schedules; changes are descriptive, not causal learning curves.",
        "Per-step parameter L2 changes are update magnitudes. Their sum is not final net movement from the base; tensor weights are not loaded by this analyzer.",
        "This analysis uses prior checkpoint JSON metadata rather than re-verifying checkpoint tensor bytes. No model, simulator, sampler or optimizer is run."]
    return {"schema_version": 1, "kind": "tdyssey-C-offline-training-diagnosis", "experiment_id": report["experiment_id"],
            "config_hash": report["config_hash"], "counts": {"batches": len(batches), "controls": cumulative,
                "plans": sum(b["collection"]["plans"] for b in batches), "actor_steps": len(passes), "critic_steps": len(passes)},
            "by_shape": shapes, "episode_outcomes": result["collector"]["by_shape"],
            "mixed_fresh_plan_statistics": {k: pool_moments(b["update"]["data_statistics"][k] for b in batches) for k in STAT_NAMES},
            "pass_diagnostics": optimizer_summary(passes, config["optimization"]["max_grad_norm"]),
            "fresh_entry_exact_count": sum(b["update"]["entry"]["max_logprob_error"] == 0 for b in batches),
            "frozen_unchanged_batch_count": sum(b["update"]["frozen_unchanged"] for b in batches),
            "frozen_digests": sorted({b["update"]["frozen_digest"] for b in batches}),
            "kl_early_stop_batches": [b["batch"] for b in batches if b["update"]["stop_reason"] != "completed"],
            "stages": stages, "reward_arithmetic_examples": reward_examples(**{k: config["optimization"][k] for k in ("gamma", "beta", "gae_lambda")}),
            "physical_runs": physical_runs(shape_table, config["optimization"]["gamma"], config["optimization"]["beta"]),
            "first_success_bonus_batches": [{"batch": r["batch"], "shape": r["shape"],
                "bonus": r["first_success_bonus_sum"], "training_reward_sum": r["training_reward_sum"],
                "shaping_sum": r["shaping_sum"]} for r in shape_table if r["first_success_bonus_sum"] > .5],
            "checkpoint_metadata": checkpoints, "limitations": limitations,
            "tables": {"batches": batch_table, "shape_batches": shape_table, "passes": pass_table}}


def diagnosis_notes(analysis):
    d = analysis["pass_diagnostics"]
    lines = ["# Offline C training diagnosis", "", "Direct saved evidence:", "",
             f"- {analysis['counts']['controls']} controls, {analysis['counts']['plans']} plans, {analysis['counts']['batches']} fresh batches and {analysis['counts']['actor_steps']} actor/critic passes."]
    for shape, s in analysis["by_shape"].items():
        e = analysis["episode_outcomes"][shape]
        lines.append(f"- {shape}: {s['controls']} controls; {s['contact_controls']} contact controls; {e['successes']} successes; raw reward sum {s['raw_reward_sum']:.12g}; training reward sum {s['training_reward_sum']:.12g}; shaping sum {s['shaping_sum']:.12g}; maximum coverage {s['max_coverage']:.12g}.")
    lines.extend([f"- Negative mixed-buffer critic explained variance in {d['negative_explained_variance_count']}/{analysis['counts']['actor_steps']} passes; final value {analysis['tables']['passes'][-1]['critic_explained_variance']:.9g}. Maximum post-pass KL {d['sample_approx_kl']['max']:.9g}, clip fraction {d['clip_fraction']['max']:.9g}.",
                  f"- Actor pre-clip gradient norm exceeds configured limit in {d['actor_preclip_norm_over_limit_count']} passes. Every pass has positive actor/critic weight change; {analysis['frozen_unchanged_batch_count']} batch frozen-fingerprint checks passed.", "", "Reward and return semantics (source plus pure arithmetic):", "",
                  "- Optimized reward is first-success bonus + beta*(gamma*next_potential-current_coverage). The original upstream reward is retained for reporting and is not added to this objective. Success means coverage strictly >.95; its next potential is zero and its value bootstrap is disabled.",
                  "- Below success, zero coverage with no progress earns zero; flat positive coverage earns -(1-gamma)*beta*coverage. Small positive changes can still yield a negative local shaping reward. Timeout and buffer cutoffs retain the actual final coverage and final-observation value bootstrap.",
                  "- For m actual controls, block reward sums gamma**i*r[i], bootstrap uses gamma**m, and GAE continuation uses gamma**m*lambda once per plan. Success clears bootstrap/continuation; timeout/cutoff clear continuation only. Separate fragments prevent GAE crossing shape, episode or buffer boundaries.",
                  "- Potential shaping telescopes for a genuinely terminal trajectory. Negative local rewards or a negative successful-batch sum alone do not establish an incorrect ideal objective. Finite bootstrapping, sparse success and exploration remain possible practical causes.", "", "Specific stationary-data observations:", ""])
    for row in analysis["physical_runs"]["flat_positive_coverage_no_contact"]:
        lines.append(f"- {row['shape']} batches {row['batches']}: {row['controls']} controls, zero recorded contacts/coverage change, constant coverage {row['coverage']:.12g}, training reward {row['training_reward_sum']:.12g} (flat-coverage formula {row['expected_flat_training_sum']:.12g}).")
    for row in analysis["first_success_bonus_batches"]:
        lines.append(f"- Batch {row['batch']} {row['shape']} contains bonus {row['bonus']:.9g}; its undiscounted training sum is {row['training_reward_sum']:.12g}, with shaping sum {row['shaping_sum']:.12g}. A negative batch sum does not mean the success bonus was absent.")
    lines.extend(["", "Interpretation and hypotheses:", "",
                  "- High late mixed explained variance can reflect fitting stationary shaped values and between-shape offsets. Per-shape EV, advantage signs and the causal effect of those offsets are unobserved, so this remains a hypothesis.",
                  "- A near-zero actor loss at the fresh first pass is expected with identity likelihood ratios and mean-zero normalized advantages. Nonzero gradients and changed weights show that it does not mean no update. Small sampled conditional-transition KL is not a guarantee of task retention.",
                  "- Success is sparse and exposure is uneven. Training scenes change over time, so batch reward/coverage changes do not isolate policy learning.", "", "Priority single-factor follow-up proposal (requires a new authorized budget):", "",
                  "- Compare fresh-base training with 2,048 versus 4,096 physical controls, keeping gamma, beta, lambda, sampler floor, learning rates, passes per buffer, frozen subset and the ordered scene schedule fixed. This tests whether doubling exposure is sufficient; it does not presume undertraining caused the observed result. Use the same fixed paired development cases to compare outcomes. The current v1 configuration cannot run 4,096 controls: a future configuration/protocol version and explicit authorization are required. No additional run is authorized or performed by this analysis.",
                  "- The configured gamma=.99 gives gamma**300=.0490409. Reward horizon, finite bootstrapping and exploration remain unresolved hypotheses; discount arithmetic alone identifies no cause and is not an alternative experiment recommendation.", "", "Limits:", ""])
    lines.extend("- " + x for x in analysis["limitations"])
    return "\n".join(lines) + "\n"


def write_outputs(analysis, output):
    """Use exclusive creation; never replace any earlier evidence or analysis."""
    names = ["analysis.json", "batch_metrics.csv", "shape_batch_metrics.csv", "pass_metrics.csv", "notes.md"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing derived outputs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / names[0]).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(analysis, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for key, name in zip(("batches", "shape_batches", "passes"), names[1:4]):
        rows = analysis["tables"][key]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output / names[-1]).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(diagnosis_notes(analysis))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    allowed = (root / OUTPUT).resolve()
    output = (args.output_dir or allowed).resolve()
    if output != allowed and allowed not in output.parents:
        parser.error("Outputs must stay under " + str(allowed))
    paths = {"log": root / "artifacts/milestone-c-20261024/training.jsonl",
             "report": root / "artifacts/milestone-c-20261024/train-report-0000.json",
             "config": root / "configs/policy_c.example.json",
             "metadata": root / "artifacts/milestone-c-load-memory-20261024/checkpoint-metadata-verified.json"}
    rows = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            for line in paths["log"].read_text(encoding="utf-8").splitlines() if line.strip()]
    analysis = analyze_rows(rows, read_json(paths["report"]), read_json(paths["config"]), read_json(paths["metadata"]))
    inputs = [*paths.values(), *(root / p for p in SOURCE_INPUTS), Path(__file__).resolve()]
    for name in ("ppo-review-final.json", "train-reconciliation-verified.json", "resources-review-final.json"):
        path = root / "artifacts/milestone-c-load-memory-20261024" / name
        if path.exists():
            inputs.append(path)
    analysis["inputs"] = [input_identity(path, root) for path in inputs]
    analysis["provenance_scope"] = "Current source hashes document the arithmetic reviewed offline; launch-source verification remains in the separately hashed reconciliation. Existing tensor metadata is read as JSON only."
    write_outputs(analysis, output)
    print(json.dumps({"output_dir": str(output), "counts": analysis["counts"], "inputs": len(inputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
