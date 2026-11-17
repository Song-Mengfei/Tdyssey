"""Paired fresh-scene comparison by shape group: python -I report_pair.py <dir> <base prefix> <candidate prefix> [...]
Files: <dir>/<prefix>-part*.json (base_rollouts.py eval). Groups: ten earlier shapes, 17 new letters, 27 trained, 6 held-out."""
import glob, json, math, sys
import numpy as np
D, base, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
def load(p): return {(x["shape"], x["requested_seed"]): x for f in glob.glob(f"{D}/{p}-part*.json") for x in json.load(open(f))["episodes"]}
OLD = ["T", "T_narrow", "Y", "D", "L", "E", "F", "J", "P", "U"]; NEW = list("ABCGHIKMNOQRSVWXZ")
HELD = ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]
b = load(base)
for c in cands:
    e = load(c); keys = sorted(set(b) & set(e)); print(f"== {c} vs {base}: {len(keys)} paired episodes")
    for name, sh in (("ten earlier shapes", OLD), ("17 new letters", NEW), ("all 27 trained shapes", OLD + NEW), ("6 held-out shapes", HELD)):
        ks = [k for k in keys if k[0] in sh]
        if not ks: continue
        x = np.array([b[k]["success"] for k in ks], float); y = np.array([e[k]["success"] for k in ks], float); d = y - x
        se = d.std(ddof=1) / math.sqrt(len(d))
        cx, cy = np.mean([b[k]["final_coverage"] for k in ks]), np.mean([e[k]["final_coverage"] for k in ks])
        print(f"  {name:22s} {100*x.mean():5.1f}% -> {100*y.mean():5.1f}%  diff {100*d.mean():+5.1f} [{100*(d.mean()-1.96*se):+5.1f}, {100*(d.mean()+1.96*se):+5.1f}]  coverage {cx:.3f} -> {cy:.3f}")
