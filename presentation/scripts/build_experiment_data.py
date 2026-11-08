"""Data tables for the two later experiments (no GPU): the architecture experiment (D059: a 505M flow-matching
transformer against the main model on the same data and camera) and last-mile RL on the frozen main model (D058).

    python3 presentation/scripts/build_experiment_data.py [--runs <folder with the run store's runs/>]

Writes presentation/data/architecture_test_by_shape.csv, architecture_test_groups.csv,
architecture_validation_by_shape.csv, architecture_training_loss.csv and lastmile_rl.csv.
"""
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EARLY = ["T", "T_narrow", "Y", "D", "L", "E", "F", "J", "P", "U"]
UNSEEN = ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]


def episodes(pattern, key=("shape", "requested_seed")):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f))["episodes"]:
            out[tuple(r[k] for k in key)] = r
    return out


def paired(a, b, keys, field):
    d = [float(b[k][field]) - float(a[k][field]) for k in keys]
    n = len(d); mu = sum(d) / n; sd = (sum((x - mu) ** 2 for x in d) / (n - 1)) ** 0.5
    half = 1.96 * sd / math.sqrt(n)
    return n, mu, mu - half, mu + half


def write(name, rows):
    with open(DATA / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote data/{name} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.environ.get("TDYSSEY_WORK", "/tmp/Tdyssey"), "runs"))
    a = ap.parse_args()
    R = Path(a.runs)
    # architecture experiment, new test scenes (seeds 13000000 + 20*i, 200 per shape)
    A, B = episodes(str(R / "fairtest-d059/r50-part*.json")), episodes(str(R / "fairtest-d059/flow-part*.json"))
    keys = sorted(set(A) & set(B))
    shapes = sorted({k[0] for k in keys}, key=lambda s: (s in UNSEEN, s not in EARLY, s))
    rows = []
    for s in shapes:
        ks = [k for k in keys if k[0] == s]
        for name, D in (("main model (official network, 96 px, after 50 rounds)", A), ("505M transformer (DINOv2 + flow matching)", B)):
            rows.append({"shape": s, "group": "never seen" if s in UNSEEN else ("earlier 10" if s in EARLY else "newer 17"),
                         "policy": name, "episodes": len(ks), "successes": sum(D[k]["success"] for k in ks),
                         "success_rate": round(sum(D[k]["success"] for k in ks) / len(ks), 4),
                         "mean_final_coverage": round(sum(D[k]["final_coverage"] for k in ks) / len(ks), 4)})
    write("architecture_test_by_shape.csv", rows)
    groups = {"all 27 trained shapes": [s for s in shapes if s not in UNSEEN], "10 earlier shapes": EARLY,
              "17 newer letters": [s for s in shapes if s not in UNSEEN and s not in EARLY], "6 never-seen shapes": UNSEEN,
              "T": ["T"], "narrow T": ["T_narrow"]}
    rows = []
    for g, ss in groups.items():
        ks = [k for k in keys if k[0] in ss]
        n, mu, lo, hi = paired(A, B, ks, "success")
        rows.append({"group": g, "episodes": n, "main_model": round(sum(A[k]["success"] for k in ks) / n, 4),
                     "transformer": round(sum(B[k]["success"] for k in ks) / n, 4), "difference": round(mu, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "main_model_final_coverage": round(sum(A[k]["final_coverage"] for k in ks) / n, 4),
                     "transformer_final_coverage": round(sum(B[k]["final_coverage"] for k in ks) / n, 4)})
    write("architecture_test_groups.csv", rows)
    # validation (50 selection scenes per shape): transformer at step 120k vs the main line's validation rows
    V = episodes(str(R / "flow26/v1/val-step120000/eval-in-x8-part*.json"))
    rows = []
    for s in sorted({k[0] for k in V}, key=lambda s: (s not in EARLY, s)):
        ks = [k for k in V if k[0] == s]
        rows.append({"shape": s, "policy": "505M transformer, step 120k", "episodes": len(ks),
                     "successes": sum(V[k]["success"] for k in ks),
                     "mean_final_coverage": round(sum(V[k]["final_coverage"] for k in ks) / len(ks), 4)})
    write("architecture_validation_by_shape.csv", rows)
    # training loss, 500-step windows
    import numpy as np
    loss = np.load(R / "flow26/v1/loss_per_step.npy")
    rows = [{"step_end": int(min(i + 500, len(loss))), "mean_loss": round(float(loss[i:i + 500].mean()), 5)}
            for i in range(0, len(loss), 500)]
    write("architecture_training_loss.csv", rows)
    # last-mile RL (D058): finishing from near-goal states of held-out scenes, paired with the frozen main model
    L = R / "lastmile-rl"
    def ev(name):
        return {(r["shape"], r["scene_index"], r["control"]): r for r in json.load(open(L / name))["episodes"]}
    rows = []
    for version, base_f, tests in (
            ("v1 (48 controls, scenes 20030-20039)", "eval-heldout-base.json",
             [("noise steering, camera only", "eval-heldout-dsrl.json"), ("residual, camera only", "eval-heldout-residual.json")]),
            ("v2 (96 controls, scenes 20040-20049)", "eval-v2-heldout-base.json",
             [("noise steering, camera only", "eval-v2-heldout-dsrl.json"),
              ("noise steering, camera only, sampled", "eval-v2-heldout-dsrl-stochastic.json"),
              ("residual, camera only", "eval-v2-heldout-residual.json"),
              ("noise steering + exact block state (diagnostic)", "eval-v2-heldout-dsrl-privileged.json")])):
        base = ev(base_f)
        rows.append({"version": version, "policy": "frozen main model", "states": len(base),
                     "finished": round(sum(r["solved"] for r in base.values()) / len(base), 4),
                     "difference": "", "ci_low": "", "ci_high": ""})
        for name, f in tests:
            t = ev(f); ks = [k for k in t if k in base]
            n, mu, lo, hi = paired(base, t, ks, "solved")
            rows.append({"version": version, "policy": name, "states": n,
                         "finished": round(sum(t[k]["solved"] for k in ks) / n, 4),
                         "difference": round(mu, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    write("lastmile_rl.csv", rows)


if __name__ == "__main__":
    main()
