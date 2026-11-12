"""Data tables for the round-50 results (no GPU): every policy on ONE pool of new test scenes, the D064 pool (seeds
15000000 + 20*i, 200 scenes per shape, run store runs/fairtest-d064/).

    python3 presentation/scripts/build_round50_data.py [--runs <folder with the run store's runs/>] [--out <folder>]

Policies (file prefix): orig (the original T-only policy), ten (ten-shape model, dagger10m round 13), r50 (26-letter 96 px,
dagger26m round 50, the main model), h192 (26-letter 192 px camera, dagger26h round 2) and base (Flow-DiT 16x16,
flow26b16 at 160k steps). Each policy has <prefix>-part*.json (the 33 shapes of the main group: 26 letters, narrow T and
six never-seen shapes) and <prefix>new-part*.json (T, digit2, digit5, digit8; only the three digits are used, the T
there is a separate replicate of the main group's T).

Only scenes present for every policy with data for that shape are used (one paired set of scenes per shape), so a
partial run limits all policies to the scenes finished by the slowest one; the script prints the count per shape and
policy (the finished test has 200 everywhere). Groups follow the talk: all 26 letters, 9 earlier letters, 17 new
letters, 9 never-seen shapes (digit3 digit4 digit7 plus hash arrow + digit2 digit5 digit8) and T; the narrow T keeps its
own per-shape rows but is in no group. Paired differences (vs the original and vs round 50) use the same normal
approximation as build_data.py: mean per-scene difference +- 1.96 sd / sqrt(n).

Writes d064_test_by_shape.csv, d064_test_groups.csv and d064_test_paired.csv (rates as fractions, like the other tables).
"""
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = [("orig", "original"), ("ten", "10-shape 96 px (round 13)"), ("r50", "26-letter 96 px (round 50)"),
          ("h192", "26-letter 192 px (round 2)"), ("base", "Flow-DiT 16x16 (160k steps)")]
EARLY9 = ["T", "Y", "D", "L", "E", "F", "J", "P", "U"]
NEW17 = list("ABCGHIKMNOQRSVWXZ")
UNSEEN_MAIN = ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]
UNSEEN_DIGITS = ["digit2", "digit5", "digit8"]                 # from the <prefix>new files
NARROW_T = "T_narrow"
GROUPS = {"all 26 letters": EARLY9 + NEW17, "9 earlier letters": EARLY9, "17 new letters": NEW17,
          "9 never-seen shapes": UNSEEN_MAIN + UNSEEN_DIGITS, "T": ["T"]}
SHAPES = EARLY9 + NEW17 + [NARROW_T] + UNSEEN_MAIN + UNSEEN_DIGITS


def group_of(s):
    if s == NARROW_T:
        return "narrow T (not in the letter groups)"
    return next(g for g in ("9 earlier letters", "17 new letters", "9 never-seen shapes") if s in GROUPS[g])


def load(folder, prefix):
    """{(shape, scene seed): episode} from the main-group files and the three digits of the digit-group files."""
    out = {}
    for pattern, keep in ((f"{prefix}-part*.json", None), (f"{prefix}new-part*.json", UNSEEN_DIGITS)):
        for f in sorted(glob.glob(str(folder / pattern))):
            for r in json.load(open(f))["episodes"]:
                if keep is None or r["shape"] in keep:
                    out[(r["shape"], int(r["requested_seed"]))] = r
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired(a, b):
    """Paired difference b - a (lists of 0/1 per scene): n, base rate, model rate, mean, 95% interval."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    sd = (sum((x - mu) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return n, sum(a) / n, sum(b) / n, mu, mu - h, mu + h


def write(name, rows):
    with open(DATA / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {DATA.name}/{name} ({len(rows)} rows)")


def r4(x):
    return round(x, 4)


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.environ.get("TDYSSEY_WORK", "/tmp/Tdyssey"), "runs"))
    ap.add_argument("--out", default=str(DATA), help="folder for the CSV tables (default: presentation/data)")
    a = ap.parse_args()
    DATA = Path(a.out)
    DATA.mkdir(parents=True, exist_ok=True)
    folder = Path(a.runs) / "fairtest-d064"
    E = {k: load(folder, k) for k, _ in MODELS}
    # one set of scenes per shape: seeds present for every policy that has any episode of that shape
    scenes = {}
    print("paired scenes per shape (policies without data for a shape are left out of it):")
    print("  " + "shape".ljust(9) + "".join(k.rjust(6) for k, _ in MODELS) + "  used")
    for s in SHAPES:
        have = {k: {q for (sh, q) in E[k] if sh == s} for k, _ in MODELS}
        with_data = [k for k in have if have[k]]
        common = set.intersection(*(have[k] for k in with_data)) if with_data else set()
        scenes[s] = sorted(common)
        print("  " + s.ljust(9) + "".join(str(len(have[k])).rjust(6) for k, _ in MODELS) + str(len(common)).rjust(6))
    missing = [(k, s) for k, _ in MODELS for s in SHAPES if not any(sh == s for sh, _ in E[k])]
    if missing:
        print(f"  NOTE {len(missing)} (policy, shape) pairs have no episodes yet, e.g. {missing[:4]}")
    short = [s for s in SHAPES if len(scenes[s]) < 200]
    print("COMPLETE: 200 paired scenes per shape for all five policies" if not short and not missing else
          f"INCOMPLETE: {len(short)} of {len(SHAPES)} shapes below 200 paired scenes; do not keep these tables")

    def ok(k, s):
        return bool(scenes[s]) and all((s, q) in E[k] for q in scenes[s])

    rows = []
    for k, name in MODELS:
        for s in SHAPES:
            if not ok(k, s):
                continue
            ep = [E[k][(s, q)] for q in scenes[s]]
            n = len(ep)
            rows.append({"model_key": k, "model": name, "shape": s, "group": group_of(s), "episodes": n,
                         "successes": sum(int(e["success"]) for e in ep),
                         "success_rate": r4(sum(int(e["success"]) for e in ep) / n),
                         "mean_final_coverage": r4(sum(float(e["final_coverage"]) for e in ep) / n),
                         "mean_peak_coverage": r4(sum(float(e["peak_coverage"]) for e in ep) / n)})
    write("d064_test_by_shape.csv", rows)
    rows = []
    for k, name in MODELS:
        for g, ss in GROUPS.items():
            ep = [E[k][(s, q)] for s in ss if ok(k, s) for q in scenes[s]]
            if not ep:
                continue
            n = len(ep)
            succ = sum(int(e["success"]) for e in ep)
            lo, hi = wilson(succ, n)
            rows.append({"model_key": k, "model": name, "group": g, "shapes": sum(ok(k, s) for s in ss),
                         "of_shapes": len(ss), "episodes": n, "successes": succ, "success_rate": r4(succ / n),
                         "wilson_low": r4(lo), "wilson_high": r4(hi),
                         "mean_final_coverage": r4(sum(float(e["final_coverage"]) for e in ep) / n)})
    write("d064_test_groups.csv", rows)
    rows = []
    units = [("group", g, ss) for g, ss in GROUPS.items()] + [("shape", s, [s]) for s in SHAPES]
    for base in ("orig", "r50"):
        for k, name in MODELS:
            if k == base or (base == "r50" and k == "orig"):      # original vs round 50 is in the "orig" block
                continue
            for kind, unit, ss in units:
                keys = [(s, q) for s in ss if ok(base, s) and ok(k, s) for q in scenes[s]]
                if len(keys) < 2:
                    continue
                n, br, mr, mu, lo, hi = paired([int(E[base][x]["success"]) for x in keys],
                                               [int(E[k][x]["success"]) for x in keys])
                rows.append({"base_model_key": base, "model_key": k, "model": name, "kind": kind, "unit": unit,
                             "paired_episodes": n, "base_rate": r4(br), "model_rate": r4(mr), "difference": r4(mu),
                             "ci95_low": r4(lo), "ci95_high": r4(hi)})
    write("d064_test_paired.csv", rows)


if __name__ == "__main__":
    main()
