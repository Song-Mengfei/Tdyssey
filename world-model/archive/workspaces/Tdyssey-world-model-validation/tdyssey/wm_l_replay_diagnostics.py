"""Read-only reconstruction of saved predictor inputs; no predictor execution.

The reconstruction is an audit of the recorded local interface, not evidence
about the checkpoint's training cadence. Raw AR feedback stays unclipped and
untransformed. Truth is never used to build an AR context.
"""

import json
from pathlib import Path

import numpy as np

from .wm_compat_appearance import _read_raw
from .wm_compat_engine import SavedBranch, action_window, arm_context, array_hash, real_context


SEEDS = (5478, 5479, 5480)
CANDIDATES = ("hold", "toward", "side", "away")


def replay_branch_inputs(branch, seed, rows, read_raw):
    """Compare actual saved input/action/output hashes without a model callback.

    Missing predecessors block reconstruction; actual future images never fill
    an AR gap. TF has only H2/H4 and deliberately excludes its target frame.
    ``read_raw`` loads an existing array, it is not a sampler interface.
    """
    if seed not in SEEDS:
        raise ValueError("Undeclared saved noise seed")
    lookup = {}
    for row in rows:
        key = row["key"]
        if key in lookup:
            raise ValueError("Duplicate saved record")
        lookup[key] = row
    checks = []
    for arm in ("A", "B"):
        context = arm_context(branch.history, arm)
        for mode, horizons in (("AR", (1, 2, 3, 4)), ("TF", (2, 4))):
            for horizon in horizons:
                key = f"{branch.anchor}/{branch.candidate}/{seed}/{mode}/{arm}/H{horizon}"
                row = lookup.get(key)
                if mode == "TF":
                    selected = arm_context(real_context(branch, horizon), arm)
                else:
                    selected = context
                if row is None or row.get("status") != "completed":
                    checks.append(dict(key=key, status="missing_record", passed=False))
                    if mode == "AR":
                        context = None
                    continue
                if selected is None:
                    checks.append(dict(key=key, status="blocked_by_missing_AR_predecessor", passed=False))
                    continue
                raw = read_raw(row)
                if (not isinstance(raw, np.ndarray) or raw.dtype != np.float32
                        or raw.shape != (3, 96, 96) or not np.isfinite(raw).all()):
                    raise ValueError("Saved raw prediction is not finite float32 CHW")
                source = real_context(branch, horizon) if mode == "TF" else branch.history
                expected = dict(input_sha256=array_hash(selected), action_sha256=array_hash(action_window(branch, horizon)),
                    raw_sha256=array_hash(raw), source_context_sha256=array_hash(source),
                    source_history_sha256=array_hash(branch.history))
                fields = {name: row.get(name) == value for name, value in expected.items()}
                checks.append(dict(key=key, status="checked", passed=all(fields.values()), checks=fields,
                    raw_feedback_outside_unit_range=bool((raw < 0).any() or (raw > 1).any())))
                if mode == "AR":
                    # Preserve native float32 values, including values outside
                    # display range; never apply T1 a second time to predictions.
                    context = np.concatenate((context[1:], raw[None]), axis=0)
    return checks


def audit_saved_conditioning(run_dir, physical_dir):
    """Audit every available L record, keeping the original missing branch."""
    root, physical = Path(run_dir).resolve(), Path(physical_dir).resolve()
    results = root if root.name == "results" else root/"results"
    rows = [json.loads(path.read_text()) for path in sorted((results/"records").glob("L*.json"))]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["anchor"], row["candidate"], row["seed"]), []).append(row)
    checks, branches, missing = [], [], []
    for number in range(12):
        anchor = f"L{number:02d}"
        for candidate in CANDIDATES:
            if (anchor, candidate) == ("L01", "hold"):
                missing.append(dict(anchor=anchor, candidate=candidate, planned_forecasts=36,
                    reason="original_missing_branch_preserved"))
                continue
            path = physical/f"{anchor}-{candidate}-physical.npz"
            with np.load(path, allow_pickle=False) as saved:
                branch = SavedBranch(anchor, candidate, "L", *(saved[name].copy() for name in
                    ("history", "past_actions", "future_actions", "actual")))
            branches.append(dict(anchor=anchor, candidate=candidate))
            for seed in SEEDS:
                checks.extend(replay_branch_inputs(branch, seed, grouped.get((anchor, candidate, seed), []),
                    lambda row: _read_raw(row, results)))
    declared = {row["key"] for row in checks}
    extra = sorted(row["key"] for row in rows if row["key"] not in declared)
    return dict(kind="saved_L_conditioning_reconstruction.v1", passed=all(r["passed"] for r in checks) and not extra,
        available_records=len(rows), checked_records=sum(r["status"] == "checked" for r in checks),
        planned_records=1728, original_missing_records=36, physical_branches=len(branches), anchors=12,
        missing_branches=missing, unexpected_records=extra, failures=[r for r in checks if not r["passed"]],
        checks=checks, source_context_semantics=dict(AR="original saved history provenance, not evolved input",
            TF="saved real context immediately before target", actual_model_window="input_sha256"),
        limitations="Verifies saved local input/action ordering; not checkpoint training timestamps or model adequacy.",
        new_model_loads=0, new_predictions=0, new_physical_operations=0, optimizer_updates=0)
