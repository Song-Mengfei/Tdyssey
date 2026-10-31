"""Paired summary of oracle best-of-N vs base episodes (same scenes) from runs/oracle/*.json."""
import glob, json, sys
from collections import defaultdict
import statistics as st
rows = defaultdict(list)
for f in glob.glob(sys.argv[1] + "/*.json"):
    for e in json.load(open(f))["episodes"]:
        rows[(e["shape"], e["n"])].append(e)
for (shape, n), eps in sorted(rows.items()):
    base = {e["seed"]: e for e in eps if not e["oracle"]}
    orc = {e["seed"]: e for e in eps if e["oracle"]}
    pairs = [(base[s], orc[s]) for s in orc if s in base]
    if not pairs:
        continue
    bs, os_ = sum(b["success"] for b, _ in pairs), sum(o["success"] for _, o in pairs)
    print(f"{shape:9s} N={n:<3d} scenes {len(pairs):2d}  success base {bs}/{len(pairs)} -> oracle {os_}/{len(pairs)}  "
          f"final cov {st.mean(b['final_coverage'] for b, _ in pairs):.3f} -> {st.mean(o['final_coverage'] for _, o in pairs):.3f}  "
          f"peak {st.mean(b['peak_coverage'] for b, _ in pairs):.3f} -> {st.mean(o['peak_coverage'] for _, o in pairs):.3f}")
