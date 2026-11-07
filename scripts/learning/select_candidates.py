"""Pick each 26-letter line's final-test candidate (D051 rule): the best validation test (50 scenes x 27 shapes = 1350
episodes) finished by a cut time; ties go to the later round. Test round k evaluates the student trained in round k-1.
    python -I select_candidates.py "YYYY-MM-DD HH:MM" out.json [line ...]     (default lines: dagger26m dagger26h)
Reads <TDYSSEY_RUNS>/<line>/round*/eval-in-x8-part*.json (default /tmp/Tdyssey/runs)."""
import glob, json, os, re, sys, time
from collections import defaultdict
R = os.environ.get("TDYSSEY_RUNS", "/tmp/Tdyssey/runs")
cut = time.mktime(time.strptime(sys.argv[1], "%Y-%m-%d %H:%M"))
lines = sys.argv[3:] or ["dagger26m", "dagger26h"]
def tests(line):
    g = defaultdict(list)
    for f in glob.glob(f"{R}/{line}/round*/eval-in-x8-part*.json"):
        g[int(re.search(r"round(\d+)/", f).group(1))].append(f)
    out = {}
    for k, files in g.items():
        if any(os.path.getmtime(f) > cut for f in files):
            continue                                                       # finished after the cut
        tot = n = 0
        for f in files:
            for v in json.load(open(f))["summary"].values(): tot += v["success"]; n += v["n"]
        if n == 1350: out[k] = tot
    return out
def student(line, k):
    for steps in (54000, 20000, 30000):
        f = f"{R}/{line}/round{k-1}/train/ema-step{steps}.pt"
        if os.path.exists(f): return f
    raise SystemExit(f"no student for {line} test round {k}")
cands = {}
for line in lines:
    t = tests(line); best = max(t.values()); k = max(r for r, v in t.items() if v == best)
    cands[line] = {"test_round": k, "validation": best, "of": 1350, "student": student(line, k), "tests": dict(sorted(t.items()))}
    print(line, cands[line]["student"], best, "of 1350 (test round", k, ")")
json.dump(cands, open(sys.argv[2], "w"), indent=1)
