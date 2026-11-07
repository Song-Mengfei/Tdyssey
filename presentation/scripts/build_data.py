"""Build the CSV tables behind the presentation figures from the saved evaluation files (no GPU, no simulator).

Input: a folder with the run store's evaluation JSONs, laid out as runs/<run>/... (default
$TDYSSEY_WORK/slides_data/raw/runs). Each JSON is a base_rollouts.py result: the evaluated checkpoint ("policy"), the
scene seeds and one record per episode. Output: presentation/data/*.csv.

    python3 presentation/scripts/build_data.py [--raw DIR]
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"
FIVE = ["T", "T_narrow", "Y", "D", "L"]
OLD10 = FIVE + ["E", "F", "J", "P", "U"]
NEW17 = list("ABCGHIKMNOQRSVWXZ")
HELD = ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]
GROUPS = {"5 first shapes": FIVE, "10 earlier shapes": OLD10, "17 other letters": NEW17,
          "27 trained shapes": OLD10 + NEW17, "6 held-out shapes": HELD}

# Validation lines: (stage, line label, run folder, file pattern relative to the run folder)
LINES = [
    ("5 shapes", "copy the teacher, no corrections (v1)", "bc/v1", "eval/step*-x8.json"),
    ("5 shapes", "copy the teacher, no corrections (long)", "bc/v2-long", "eval/step*-x8.json"),
    ("5 shapes", "correction rounds", "dagger-loop", "round*/eval-x8.json"),
    ("10 shapes", "correction rounds, executed-push labels", "dagger10", "round*/eval-in-x8*.json"),
    ("10 shapes", "correction rounds, intended-plan labels", "dagger10p", "round*/eval-in-x8*.json"),
    ("10 shapes", "96 px main line", "dagger10m", "round*/eval-in-x8*.json"),
    ("10 shapes", "192 px side line", "dagger10h", "round*/eval-in-x8*.json"),
    ("27 shapes", "96 px line", "dagger26m", "round*/eval-in-x8*.json"),
    ("27 shapes", "192 px line", "dagger26h", "round*/eval-in-x8*.json"),
]

# Final tests: (test, description, folder, {file prefix: model label}, base prefix or None)
TESTS = [
    ("D040", "encoder blends, 1000 fresh scenes per letter", "final-test",
     {"base": "original", "blend-r4e60-s0.7": "blend r4e60 s0.7", "blend-big4-unet0-enc0.7": "blend encoder 0.7"}, "base"),
    ("D046", "5 shapes, correction round 4", "fairtest-d046", {"base": "original", "round4": "5-shape round 4"}, "base"),
    ("D047", "5 shapes, correction round 7", "fairtest-d047", {"base": "original", "round7": "5-shape round 7"}, "base"),
    ("D049/D050", "10 shapes, 1000 fresh scenes per shape", "fairtest-d049",
     {"base": "original", "main13": "10-shape 96 px", "side192": "10-shape 192 px"}, "base"),
    ("D052", "17 untrained capitals, 200 fresh scenes each", "unseen17-d052",
     {"base": "original", "main96": "10-shape 96 px", "side192": "10-shape 192 px"}, "base"),
    ("D051", "26 letters, 300 fresh scenes per shape (headline)", "fairtest-d051",
     {"base": "original", "ten96": "10-shape 96 px", "m26": "26-letter 96 px (round 44)", "h26": "26-letter 192 px (round 2)"}, "base"),
    ("D054", "round 50 vs round 44, 200 fresh scenes per shape", "fairtest-d054",
     {"r44": "26-letter 96 px round 44", "r50": "26-letter 96 px round 50"}, "r44"),
    ("D056", "steps executed per plan, same scenes as D054", "fairtest-d054",
     {"r50": "execute 8 (trained)", "e12": "execute 12", "e15": "execute 15"}, "r50"),
]


def episodes(paths):
    """Episodes from several part files, keyed by (shape, scene seed); the evaluated checkpoint path."""
    eps, policy = {}, None
    for p in paths:
        j = json.load(open(p))
        policy = policy or j.get("policy")
        for e in j["episodes"]:
            eps[(e["shape"], e["requested_seed"])] = e
    return eps, policy


def rel_policy(p):
    if not p:
        return "original"
    m = re.search(r"/runs/(.*)", p)
    return m.group(1) if m else os.path.basename(p)


def summarize(eps, shape):
    xs = [e for (s, _), e in eps.items() if s == shape]
    if not xs:
        return None
    return (len(xs), sum(bool(e["success"]) for e in xs), sum(e["final_coverage"] for e in xs) / len(xs),
            sum(e["peak_coverage"] for e in xs) / len(xs))


def build_validation(raw, w):
    w.writerow(["stage", "line", "run", "step", "policy", "shape", "episodes", "successes", "success_rate",
                "mean_final_coverage", "mean_peak_coverage"])
    rows = 0
    for stage, line, run, pattern in LINES:
        by_step = defaultdict(list)
        for p in glob.glob(str(raw / run / pattern)):
            m = re.search(r"round(\d+)", p) if "round" in pattern else re.search(r"step(\d+)", p)
            by_step[int(m.group(1))].append(p)
        for step in sorted(by_step):
            files = by_step[step]
            merged = [p for p in files if p.endswith("-merged.json")]
            eps, policy = episodes(merged or files)
            for shape in sorted({s for s, _ in eps}):
                n, k, fc, pc = summarize(eps, shape)
                w.writerow([stage, line, run, step, rel_policy(policy), shape, n, k, round(k / n, 4), round(fc, 4), round(pc, 4)])
                rows += 1
    return rows


def paired(base, cand, shapes):
    keys = [k for k in base if k in cand and k[0] in shapes]
    if not keys:
        return None
    d = [float(cand[k]["success"]) - float(base[k]["success"]) for k in keys]
    n = len(d); mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    b = sum(float(base[k]["success"]) for k in keys) / n; c = sum(float(cand[k]["success"]) for k in keys) / n
    return n, b, c, mean, mean - 1.96 * sd / math.sqrt(n), mean + 1.96 * sd / math.sqrt(n)


def build_tests(raw, w_shape, w_pair, w_eps):
    w_shape.writerow(["test", "description", "model", "policy", "shape", "episodes", "successes", "success_rate",
                      "mean_final_coverage", "mean_peak_coverage"])
    w_pair.writerow(["test", "base_model", "model", "group", "paired_episodes", "base_rate", "model_rate",
                     "difference", "ci95_low", "ci95_high"])
    w_eps.writerow(["test", "model", "shape", "scene_seed", "success", "controls", "final_coverage", "peak_coverage"])
    for test, desc, folder, models, base_prefix in TESTS:
        data = {}
        for prefix, label in models.items():
            files = sorted(glob.glob(str(raw / folder / f"{prefix}-part*.json")))
            if not files:
                print(f"missing {folder}/{prefix}"); continue
            eps, policy = episodes(files)
            data[prefix] = eps
            for shape in sorted({s for s, _ in eps}):
                n, k, fc, pc = summarize(eps, shape)
                w_shape.writerow([test, desc, label, rel_policy(policy) if prefix != "base" else "original", shape, n, k,
                                  round(k / n, 4), round(fc, 4), round(pc, 4)])
            if test in ("D051", "D054", "D056", "D049/D050"):
                for (shape, seed), e in sorted(eps.items()):
                    w_eps.writerow([test, label, shape, seed, int(bool(e["success"])), e["controls"],
                                    round(e["final_coverage"], 4), round(e["peak_coverage"], 4)])
        if base_prefix in data:
            shapes_present = {s for s, _ in data[base_prefix]}
            for prefix, label in models.items():
                if prefix == base_prefix or prefix not in data:
                    continue
                groups = {g: s for g, s in GROUPS.items() if set(s) <= shapes_present}   # only groups the test fully covers
                groups.update({s: [s] for s in sorted(shapes_present)})
                for g, shapes in groups.items():
                    r = paired(data[base_prefix], data[prefix], shapes)
                    if r:
                        n, b, c, m, lo, hi = r
                        w_pair.writerow([test, models[base_prefix], label, g, n, round(b, 4), round(c, 4), round(m, 4),
                                         round(lo, 4), round(hi, 4)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(os.environ.get("TDYSSEY_WORK", "/tmp/Tdyssey"), "slides_data/raw/runs"))
    raw = Path(ap.parse_args().raw)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "validation_rounds.csv", "w", newline="") as f:
        print("validation rows:", build_validation(raw, csv.writer(f)))
    with open(OUT / "final_tests_by_shape.csv", "w", newline="") as a, open(OUT / "final_tests_paired.csv", "w", newline="") as b, \
            open(OUT / "final_tests_episodes.csv", "w", newline="") as c:
        build_tests(raw, csv.writer(a), csv.writer(b), csv.writer(c))
    for p in sorted(OUT.glob("*.csv")):
        print(f"{p.name}: {sum(1 for _ in open(p)) - 1} rows, {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
