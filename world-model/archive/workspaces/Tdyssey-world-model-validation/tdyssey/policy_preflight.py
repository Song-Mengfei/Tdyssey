"""Deterministic reset-only scene repair; no runtime, ledger or clock imports.

The caller owns reset execution and durable per-attempt journaling. This module
only orders probes and proposes a configuration; it never writes a config or
initializes a C experiment. Only explicitly classified geometry failures can
consume the predeclared replacement sequence.
"""

from __future__ import annotations

import math

from tdyssey.policy_protocol import CConfig


SHAPES = ("T", "L")
MAX_PROBE_CALLS = 212
COUNTERS = ("reset_attempts", "reset_physics_substeps", "control_steps")
DEFAULT_CANDIDATES = {"T": tuple(range(400032, 400064)),
                      "L": tuple(range(500032, 500064))}


def _primitive(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _primitive(item) for key, item in value.items()}
    raise ValueError("Probe results must contain only finite JSON primitives")


def _candidate_order(candidates):
    source = DEFAULT_CANDIDATES if candidates is None else candidates
    if not isinstance(source, dict) or set(source) != set(SHAPES):
        raise ValueError("Candidates must declare exactly T and L sequences")
    result = {}
    for shape in SHAPES:
        values = source[shape]
        if not isinstance(values, (list, tuple)) or len(values) > 32:
            raise ValueError("Each predeclared candidate sequence has at most 32 entries")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values):
            raise ValueError("Candidate seeds must be uint32 integers")
        result[shape] = list(values)
    return result


class _StopPreflight(Exception):
    pass


class _Run:
    def __init__(self, config, probe, candidates):
        self.config, self.probe = config, probe
        self.report = {
            "schema_version": 1, "status": "blocked", "reason": None,
            "original_config_hash": config.config_hash,
            "proposed_config": None, "proposed_config_hash": None,
            "candidate_order": candidates, "candidate_decisions": [],
            "replacements": [], "original_invalid_training": [],
            "original_invalid_evaluation": [], "records": [],
            "counts": {"probe_calls": 0, **dict.fromkeys(COUNTERS, 0),
                       "counts_complete": True, "unknown_counter_records": 0},
            "final_verification": {"status": "not_started", "config_hash": None,
                                   "expected_records": 74, "records": 0,
                                   "invalid_records": []},
        }

    def stop(self, reason):
        self.report["reason"] = reason
        raise _StopPreflight()

    def call(self, shape, seed, phase, config_hash, slot_index=None):
        counts = self.report["counts"]
        if counts["probe_calls"] >= MAX_PROBE_CALLS:
            self.stop("reset_probe_cap_exhausted")
        entry = {"call_index": counts["probe_calls"] + 1, "shape": shape,
                 "seed": seed, "phase": phase, "config_hash": config_hash}
        if slot_index is not None:
            entry["slot_index"] = slot_index
        self.report["records"].append(entry)
        counts["probe_calls"] += 1
        if phase.startswith("final_"):
            self.report["final_verification"]["records"] += 1
        try:
            raw = self.probe(shape, seed, phase)
        except Exception as exc:
            entry.update(result=None, counts_unknown=True,
                         error={"type": type(exc).__name__, "message": str(exc)})
            counts["counts_complete"] = False
            counts["unknown_counter_records"] += 1
            self.stop("probe_exception")

        # Preserve every available count even when a malformed result fails
        # validation. An absent count is unknown, never fabricated as zero.
        known = {key: raw.get(key) if isinstance(raw, dict) else None for key in COUNTERS}
        unknown = any(type(value) is not int or value < 0 for value in known.values())
        for key, value in known.items():
            if type(value) is int and value >= 0:
                counts[key] += value
        entry["reported_counts"] = {key: value if type(value) is int and value >= 0 else None
                                    for key, value in known.items()}
        entry["counts_unknown"] = unknown
        if unknown:
            counts["counts_complete"] = False
            counts["unknown_counter_records"] += 1
        try:
            result = _primitive(raw)
        except ValueError as exc:
            entry.update(result=None, error={"type": type(exc).__name__, "message": str(exc)})
            self.stop("malformed_probe_result")
        entry["result"] = result
        if not isinstance(result, dict) or unknown:
            self.stop("malformed_probe_result")
        if (known["reset_attempts"] > 1 or known["reset_physics_substeps"] > known["reset_attempts"]
                or known["control_steps"] != 0):
            self.stop("probe_counter_guard_violation")
        if result.get("error") is not None or result.get("blocked") is True:
            self.stop("probe_runtime_error")
        classification = result.get("classification")
        valid, issues = result.get("valid"), result.get("issues")
        if (type(valid) is not bool or not isinstance(issues, list)
                or any(type(issue) is not str for issue in issues)):
            self.stop("malformed_probe_result")
        if (classification == "valid" and valid and not issues
                and known["reset_attempts"] == 1 and known["reset_physics_substeps"] == 1):
            return entry
        if (classification == "geometry_invalid" and not valid and issues
                and known["reset_attempts"] == 1 and known["reset_physics_substeps"] == 1):
            return entry
        self.stop("unrecognized_or_inconsistent_probe_classification")

    def execute(self):
        original = self.config.to_dict()
        train = original["scenes"]["training"]
        evaluation = original["evaluation"]["scenes"]
        original_hash = self.config.config_hash
        for shape in SHAPES:
            for index, seed in enumerate(train[shape]):
                entry = self.call(shape, seed, "original_train", original_hash, index)
                if not entry["result"]["valid"]:
                    self.report["original_invalid_training"].append(
                        {"shape": shape, "slot_index": index, "seed": seed,
                         "call_index": entry["call_index"]})
        # T must precede L for each seed: the injected runtime uses this initial
        # T state for the matched COM geometry check, without another reset.
        for seed in evaluation:
            for shape in SHAPES:
                entry = self.call(shape, seed, "original_evaluation", original_hash)
                if not entry["result"]["valid"]:
                    self.report["original_invalid_evaluation"].append(entry["call_index"])
        if self.report["original_invalid_evaluation"]:
            self.stop("fixed_evaluation_geometry_invalid")

        excluded = set(original["scenes"]["excluded_evaluation"])
        original_seeds = set(train["T"]) | set(train["L"])
        selected, probed = set(), set()
        cursors = dict.fromkeys(SHAPES, 0)
        for invalid in self.report["original_invalid_training"]:
            shape, slot = invalid["shape"], invalid["slot_index"]
            pool = self.report["candidate_order"][shape]
            while cursors[shape] < len(pool):
                candidate_index = cursors[shape]
                seed = pool[candidate_index]
                cursors[shape] += 1
                decision = {"shape": shape, "slot_index": slot,
                            "candidate_index": candidate_index, "seed": seed}
                self.report["candidate_decisions"].append(decision)
                if seed in excluded:
                    reason = "evaluation_excluded"
                elif seed in original_seeds:
                    reason = "original_training"
                elif seed in selected:
                    reason = "already_selected"
                elif seed in probed:
                    reason = "already_probed"
                else:
                    reason = None
                if reason is not None:
                    decision.update(action="skipped", reason=reason)
                    continue
                probed.add(seed)
                decision.update(action="probed")
                entry = self.call(shape, seed, "candidate_train", original_hash, slot)
                decision["call_index"] = entry["call_index"]
                decision["valid"] = entry["result"]["valid"]
                if not decision["valid"]:
                    continue
                selected.add(seed)
                train[shape][slot] = seed
                self.report["replacements"].append({**invalid, "replacement_seed": seed,
                    "candidate_call_index": entry["call_index"]})
                break
            else:
                self.stop("candidate_sequence_exhausted")

        proposed = CConfig.from_dict(original)
        proposed_hash = proposed.config_hash
        self.report.update(proposed_config=proposed.to_dict(), proposed_config_hash=proposed_hash)
        final = self.report["final_verification"]
        final.update(status="in_progress", config_hash=proposed_hash)
        for shape in SHAPES:
            for index, seed in enumerate(train[shape]):
                entry = self.call(shape, seed, "final_train", proposed_hash, index)
                if not entry["result"]["valid"]:
                    final["invalid_records"].append(entry["call_index"])
        for seed in evaluation:
            for shape in SHAPES:
                entry = self.call(shape, seed, "final_evaluation", proposed_hash)
                if not entry["result"]["valid"]:
                    final["invalid_records"].append(entry["call_index"])
        if final["invalid_records"]:
            final["status"] = "blocked"
            self.stop("final_verification_geometry_invalid")
        final["status"] = "complete"
        self.report.update(status="complete", reason=None)


def run_reset_preflight(config, probe, *, candidates=None):
    """Audit 64 original training and 10 fixed evaluation resets, then verify.

    Results are deterministic given the callback results and declared candidate
    order. An exception or ambiguous probe result blocks immediately. Counters
    are known lower bounds when counts_complete is false; the caller's durable
    probe journal remains authoritative for an interrupted runtime operation.
    """
    if not isinstance(config, CConfig):
        raise TypeError("config must be a validated CConfig")
    config.validate()
    if any(len(config.to_dict()["scenes"]["training"][shape]) != 32 for shape in SHAPES):
        raise ValueError("Reset preflight requires exactly 32 training seeds per shape")
    if not callable(probe):
        raise TypeError("probe must be callable")
    runner = _Run(config, probe, _candidate_order(candidates))
    try:
        runner.execute()
    except _StopPreflight:
        if runner.report["final_verification"]["status"] == "in_progress":
            runner.report["final_verification"]["status"] = "blocked"
    return runner.report
