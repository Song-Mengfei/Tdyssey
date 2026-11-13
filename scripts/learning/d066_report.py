"""D066 test report: goal-overlay student vs control (same rounds without the overlay) vs round 50, paired by shape and
scene on the D064 pool (seeds 15000000 + 20*i). Groups as D064: 26 letters (near-T excluded), 9 never-seen shapes, T.
Primary (fixed before results): overlay vs control, success over the 26 letters (mean over shapes; paired difference
with a 95% interval). Secondary: both vs round 50; among failures, the share whose best coverage reached 0.90-0.95
(near misses), and among episodes that reached 0.90, the share solved.
Files: <dir>/<m>-part*.json (33 shapes) and <dir>/<m>new-part*.json (T + digits 2, 5, 8), base_rollouts.py eval.
    python -I d066_report.py <dir> overlay control r50
Validation progress curve (monitoring only): every round's student of both arms on the selection scenes (seeds
400000 + 20*i, i < 50), paired by scene; <d066 dir>/<arm>/round<k>/val-part*.json and <d066 dir>/start/ (round 50).
    python -I d066_report.py validation <d066 dir>
"""
import glob
import json
import math
import sys

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NEVER = ["digit2", "digit3", "digit4", "digit5", "digit7", "digit8", "plus", "hash", "arrow"]


def load(d, m):
    out = {}
    for pat in (f"{d}/{m}-part*.json", f"{d}/{m}new-part*.json"):
        for f in glob.glob(pat):
            for e in json.load(open(f))["episodes"]:
                if pat.endswith("new-part*.json") and e["shape"] == "T":
                    continue                      # T in the digit group repeats the main T scenes
                out[(e["shape"], e["requested_seed"])] = e
    return out


def mean_ci(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def shape_mean(eps, keys, shapes):
    per = [[k for k in keys if k[0] == s] for s in shapes]
    per = [v for v in per if v]
    return sum(sum(eps[k]["success"] for k in v) / len(v) for v in per) / len(per), len(per)


def near_miss(eps, keys):
    fails = [k for k in keys if not eps[k]["success"]]
    near = [k for k in fails if 0.90 <= eps[k]["peak_coverage"] < 0.95]
    reached = [k for k in keys if eps[k]["peak_coverage"] >= 0.90]
    return (len(near) / len(fails) if fails else float("nan"),
            sum(eps[k]["success"] for k in reached) / len(reached) if reached else float("nan"))


def validation(root):
    def load_val(folder):
        out = {}
        for f in glob.glob(f"{folder}/val-part*.json"):
            for e in json.load(open(f))["episodes"]:
                out[(e["shape"], e["requested_seed"])] = e
        return out
    start = load_val(f"{root}/start")
    rounds = sorted({int(p.rsplit("round", 1)[1]) for p in glob.glob(f"{root}/*/round*") if p.rsplit("round", 1)[1].isdigit()})
    print("D066 validation (monitoring only): 26 letters, mean success over shapes; all 27 trained shapes, solved of 1350;")
    print("overlay - control paired by scene with a 95% interval")
    print(f"  {'round':>5s}  {'overlay':>16s}  {'control':>16s}  overlay - control")
    if start:
        ks = [k for k in start if k[0] in LETTERS]
        print(f"  start  round 50 (official images) {100 * shape_mean(start, ks, LETTERS)[0]:5.1f}%  "
              f"{sum(e['success'] for e in start.values())}/{len(start)}")
    for r in rounds:
        arms = {a: load_val(f"{root}/{a}/round{r}") for a in ("overlay", "control")}
        cells, ok = [], {a: len(v) == 1350 for a, v in arms.items()}
        for a in ("overlay", "control"):
            v = arms[a]
            ks = [k for k in v if k[0] in LETTERS]
            cells.append(f"{100 * shape_mean(v, ks, LETTERS)[0]:5.1f}% {sum(e['success'] for e in v.values()):4d}/{len(v)}"
                         if ks else f"{'-':>16s}")
        diff = ""
        if all(ok.values()):
            ks = [k for k in arms["overlay"] if k[0] in LETTERS and k in arms["control"]]
            m, h = mean_ci([int(arms["overlay"][k]["success"]) - int(arms["control"][k]["success"]) for k in ks])
            diff = f"{100 * m:+5.1f} [{100 * (m - h):+5.1f}, {100 * (m + h):+5.1f}]"
        print(f"  {r:5d}  {cells[0]:>16s}  {cells[1]:>16s}  {diff}")


def main():
    if sys.argv[1] == "validation":
        return validation(sys.argv[2])
    d, models = sys.argv[1], sys.argv[2:]
    eps = {m: load(d, m) for m in models}
    keys = set.intersection(*(set(e) for e in eps.values()))
    print(f"D066 test on seeds 15000000 + 20*i: {', '.join(models)}; {len(keys)} episodes common to all")
    for name, shapes in (("26 letters", LETTERS), ("9 never-seen shapes", NEVER), ("T", ["T"])):
        ks = [k for k in keys if k[0] in shapes]
        if not ks:
            continue
        print(f"\n== {name} ({len(ks)} scenes per model)")
        for m in models:
            s, n_sh = shape_mean(eps[m], ks, shapes)
            nm, conv = near_miss(eps[m], ks)
            cov = sum(eps[m][k]["final_coverage"] for k in ks) / len(ks)
            print(f"  {m:10s} success {100 * s:5.1f}% ({n_sh} shapes)  final coverage {cov:.3f}  near misses among "
                  f"failures {100 * nm:5.1f}%  solved once >= 0.90 {100 * conv:5.1f}%")
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                m, h = mean_ci([int(eps[a][k]["success"]) - int(eps[b][k]["success"]) for k in ks])
                print(f"  {a} - {b}: {100 * m:+5.1f} points [{100 * (m - h):+5.1f}, {100 * (m + h):+5.1f}] (paired, pooled)")
    print("\nPer letter (success %):", " ".join(m for m in models))
    for s in LETTERS:
        ks = [k for k in keys if k[0] == s]
        if ks:
            print(f"  {s:2s} " + " ".join(f"{100 * sum(eps[m][k]['success'] for k in ks) / len(ks):5.1f}" for m in models))


if __name__ == "__main__":
    main()
