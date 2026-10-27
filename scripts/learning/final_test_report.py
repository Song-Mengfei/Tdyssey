"""Pre-registered final-test analysis (D040): candidate vs base, paired by scene, on the 1000-scene final pool.

Reads base parts (<runs>/base-part*.json) and candidate parts (<runs>/<name>-part*.json), pairs
episodes by (shape, requested_seed), and reports per shape the success rates, the paired difference with a 95%
normal-approximation interval, and the four pass conditions:
  (1) T lower bound >= -5 points, (2) T_narrow lower bound >= -5 points,
  (3) at least one of Y, D, L has a lower bound > 0, (4) no new letter has an upper bound < 0.
  python3 -I final_test_report.py --runs $RUNS/final-test --candidate blend-r4e60-s0.7
"""

import argparse
import glob
import json
import math
from pathlib import Path



def load(runs, prefix):
    out = {}
    for f in sorted(glob.glob(str(runs/f"{prefix}-part*.json"))):
        for e in json.load(open(f))["episodes"]:
            out[(e["shape"], e["requested_seed"])] = e
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True)
    p.add_argument("--runs", type=Path, required=True, help="Folder with base-part*.json and <candidate>-part*.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    base, cand = load(args.runs, "base"), load(args.runs, args.candidate)
    rows, result = [], {"candidate": args.candidate, "shapes": {}}
    for shape in ("T", "T_narrow", "Y", "D", "L"):
        keys = sorted(k for k in base if k[0] == shape and k in cand)
        n = len(keys)
        if n == 0:
            continue
        b = [float(bool(base[k]["success"])) for k in keys]; c = [float(bool(cand[k]["success"])) for k in keys]
        d = [ci - bi for ci, bi in zip(c, b)]
        mean = sum(d)/n; sd = math.sqrt(sum((x-mean)**2 for x in d)/(n-1)) if n > 1 else 0.0
        lo, hi = mean - 1.96*sd/math.sqrt(n), mean + 1.96*sd/math.sqrt(n)
        cov_b = sum(base[k]["final_coverage"] for k in keys)/n; cov_c = sum(cand[k]["final_coverage"] for k in keys)/n
        result["shapes"][shape] = {"n": n, "base": sum(b)/n, "candidate": sum(c)/n, "diff": mean, "lo": lo, "hi": hi,
                                   "base_coverage": cov_b, "candidate_coverage": cov_c,
                                   "both": sum(1 for x, y in zip(b, c) if x and y), "only_base": sum(1 for x, y in zip(b, c) if x and not y),
                                   "only_candidate": sum(1 for x, y in zip(b, c) if y and not x)}
        rows.append(f"{shape:9s} n={n:4d}  base {100*sum(b)/n:5.1f}%  candidate {100*sum(c)/n:5.1f}%  diff {100*mean:+5.1f} "
                    f"[{100*lo:+5.1f}, {100*hi:+5.1f}]  coverage {cov_b:.3f} -> {cov_c:.3f}")
    s = result["shapes"]
    new = [x for x in ("Y", "D", "L") if x in s]
    conds = {"1_T_noninferior": "T" in s and s["T"]["lo"] >= -0.05,
             "2_T_narrow_noninferior": "T_narrow" in s and s["T_narrow"]["lo"] >= -0.05,
             "3_new_letter_better": any(s[x]["lo"] > 0 for x in new),
             "4_no_new_letter_worse": all(s[x]["hi"] >= 0 for x in new),
             "complete": all(s.get(x, {}).get("n") == 1000 for x in ("T", "T_narrow", "Y", "D", "L"))}
    result["conditions"] = conds
    result["pass"] = all(conds.values())
    print("\n".join(rows)); print(json.dumps(conds)); print("PASS" if result["pass"] else "NOT PASSED")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
