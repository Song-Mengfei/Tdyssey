"""Read-only denoising capture and factual near-miss timing diagnostics."""

from contextlib import contextmanager


@contextmanager
def capture_scheduler_steps(scheduler):
    """Observe the real sampler; never resample, replace outputs, or consume RNG.

    Only this scheduler instance is wrapped, and its original attribute is restored
    even on failure. This is diagnostic instrumentation, not an alternate sampler.
    The retained tensors are detached copies on the original device until export.
    """
    original = scheduler.step
    had_override = "step" in vars(scheduler)
    override = vars(scheduler).get("step")
    record = {"samples": [], "timesteps": []}

    def observed_step(*args, **kwargs):
        timestep = kwargs["timestep"] if "timestep" in kwargs else args[1]
        sample = kwargs["sample"] if "sample" in kwargs else args[2]
        if not record["samples"]:
            record["samples"].append(sample.detach().clone())
        result = original(*args, **kwargs)
        record["samples"].append(result.prev_sample.detach().clone())
        record["timesteps"].append(int(timestep))
        return result

    scheduler.step = observed_step
    try:
        yield record
    finally:
        if had_override:
            scheduler.step = override
        else:
            del scheduler.step


def near_miss_facts(episode, action_horizon=8, success_threshold=0.95):
    """Posthoc facts, not a success override or a claim about network reasoning.

    Control zero is the initial scene; row zero is after control one. This
    function assumes the recorded runner replans every action_horizon controls.
    It preserves the earliest best-IoU tie and the strict upstream coverage rule.
    """
    rows = episode["steps"]
    if not rows:
        raise ValueError("Near-miss analysis requires a nonempty recorded rollout")
    index = max(range(len(rows)), key=lambda i: rows[i]["metrics"]["iou"])
    peak = index + 1
    start = index // action_horizon * action_horizon
    end = min(start + action_horizon, len(rows))
    first_drop = next((i + 1 for i in range(index + 1, end)
                       if rows[i]["metrics"]["iou"] < rows[index]["metrics"]["iou"] - 1e-9), None)
    return {
        "best_control": peak,
        "best_iou": rows[index]["metrics"]["iou"],
        "best_coverage": rows[index]["metrics"]["coverage"],
        "best_upstream_done": rows[index]["upstream_done"],
        "success_threshold": success_threshold,
        "best_exceeds_success_threshold": rows[index]["metrics"]["coverage"] > success_threshold,
        "plan_start_control": start,
        "planned_controls": [start + 1, end],
        "best_action_slot_1_based": peak - start,
        "first_iou_drop_before_replan": first_drop,
        "remaining_controls_before_replan": end - peak,
        "final_iou": rows[-1]["metrics"]["iou"],
    }


def post_success_facts(rows):
    """Separate ever reaching success from retaining it during continuation.

    Use original per-control done flags. The first successful control itself is
    excluded from all post-success counts. No crossing means no retention sample.
    """
    first = next((i for i, row in enumerate(rows) if row["done"]), None)
    after = [] if first is None else rows[first + 1:]
    drop = None if first is None else next((i + 1 for i in range(first + 1, len(rows))
                                           if not rows[i]["done"]), None)
    return {
        "first_success_control": None if first is None else first + 1,
        "post_success_controls": len(after),
        "first_loss_of_success_control": drop,
        "post_success_successful_controls": sum(row["done"] for row in after),
        "post_success_success_fraction": sum(row["done"] for row in after) / len(after) if after else None,
        "post_success_min_coverage": min((row["metrics"]["coverage"] for row in after), default=None),
        "post_success_min_iou": min((row["metrics"]["iou"] for row in after), default=None),
        "final_coverage": rows[-1]["metrics"]["coverage"] if rows else None,
        "final_success": bool(rows[-1]["done"]) if rows else None,
    }
