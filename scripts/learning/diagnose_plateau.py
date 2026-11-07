"""Why did the 96 px main line stop improving? (2026-11-02) Reads only stored files; no GPU.

A. training loss per round (start and end of each round)
B. how test episodes end, per round (success / close >= 0.85 / middle / far < 0.6), main vs 192 px side line
C. student vs teacher on the student's own visited states (stored corrections: the student's 8 sampled plans and the
   teacher's label for the same observation), per round
D. churn: per-scene success flips between consecutive students vs between two tests of the SAME student (noise floor)
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

R = Path(os.environ.get("TDYSSEY_RUNS", "/tmp/Tdyssey/runs"))   # experiment outputs (outside Git)


def episodes(d):
    eps = []
    for f in sorted(glob.glob(str(d/"eval-in-x8-part*.json"))) or sorted(glob.glob(str(d)+"-x8-part*.json")):
        eps += json.load(open(f))["episodes"]
    return {(e["shape"], e["requested_seed"]): e for e in eps}


def rounds(line):
    return sorted(int(p.name[5:]) for p in (R/line).glob("round*") if (p/"eval-in-x8-part0.json").exists() and (p/"eval-in-x8-part25.json").exists())


print("A. training loss (mean of first 500 / last 2000 steps of each round)")
for line in ("dagger10m", "dagger10h"):
    row = []
    for k in sorted(int(p.name[5:]) for p in (R/line).glob("round*")):
        f = R/line/f"round{k}/train/loss_per_step.npy"
        if f.exists():
            x = np.load(f)
            if len(x) >= 19000:
                row.append(f"r{k} {x[:500].mean():.4f}->{x[-2000:].mean():.4f}")
    print(f"  {line}: " + "  ".join(row))

print("\nB. how test episodes end (500 per test; student trained in round k-1 is tested at the start of round k)")
print("  line   test  success  close(.85-.95)  middle(.6-.85)  far(<.6)  mean final cov  'had it, lost it' (peak>=.9, final<.85)")
for line in ("dagger10m", "dagger10h"):
    for k in rounds(line):
        e = list(episodes(R/line/f"round{k}").values())
        fc = np.array([x["final_coverage"] for x in e]); pk = np.array([x["peak_coverage"] for x in e]); s = np.array([x["success"] for x in e])
        close = (~s) & (fc >= 0.85); mid = (~s) & (fc >= 0.6) & (fc < 0.85); far = (~s) & (fc < 0.6); lost = (~s) & (pk >= 0.9) & (fc < 0.85)
        print(f"  {line[-1]:>4}  r{k:<3}  {s.sum():5d}  {close.sum():10d}  {mid.sum():14d}  {far.sum():9d}  {fc.mean():12.3f}  {lost.sum():10d}")

print("\nC. student vs teacher on the student's own states (first 8 targets, pixels; corrections collected in round k)")
print("  round  records  student-to-teacher (median / mean)  student's own spread (median)  teacher label vs student's best sample (median)")
SCALE = 256.0                                                       # normalized [-1, 1] -> [0, 512] px
for line in ("dagger10m", "dagger10h"):
    for k in sorted(int(p.name.split("-r")[-1]) for p in (R/"mpc-data").glob(f"{line}-r*")):
        fs = glob.glob(str(R/f"mpc-data/{line}-r{k}/data/*.npz"))
        if not fs:
            continue
        d_mean, spread, d_best, n = [], [], [], 0
        for f in fs:
            z = np.load(f)
            if "candidate_x" not in z.files or len(z["choice"]) == 0:
                continue
            c = z["candidate_x"].astype(np.float32)[:, :, :8]          # (n, 9, 8, 2): 8 student samples + teacher label
            stu, lab = c[:, :8], c[:, 8]
            d = np.linalg.norm(stu - lab[:, None], axis=-1).mean(-1) * SCALE        # (n, 8) per sample, mean over steps
            d_mean.append(np.linalg.norm(stu.mean(1) - lab, axis=-1).mean(-1) * SCALE)
            d_best.append(d.min(1))
            spread.append(np.linalg.norm(stu - stu.mean(1, keepdims=True), axis=-1).mean((1, 2)) * SCALE)
            n += len(c)
        dm, db, sp = np.concatenate(d_mean), np.concatenate(d_best), np.concatenate(spread)
        print(f"  {line[-1]} r{k:<3} {n:7d}   {np.median(dm):6.1f} / {dm.mean():6.1f}            {np.median(sp):6.1f}                       {np.median(db):6.1f}")

print("\nD. churn: per-scene success flips")
def flips(a, b):
    keys = sorted(set(a) & set(b))
    sa = np.array([a[x]["success"] for x in keys]); sb = np.array([b[x]["success"] for x in keys])
    return len(keys), int((sa & ~sb).sum()), int((~sa & sb).sum())
same_a = episodes(R/"dagger10p/round1")                               # student dagger10/round16, tested by the plan-label loop
same_b = episodes(Path(str(R/"dagger10/final-round16")))              # the same student, tested again by the old loop
if same_a and same_b:
    n, lost, won = flips(same_a, same_b)
    print(f"  SAME student tested twice: {n} scenes, {lost} lost + {won} gained = {lost+won} flips (noise floor)")
ks = rounds("dagger10m")
tot = []
for a, b in zip(ks, ks[1:]):
    n, lost, won = flips(episodes(R/f"dagger10m/round{a}"), episodes(R/f"dagger10m/round{b}"))
    tot.append(lost+won)
    print(f"  main r{a} -> r{b}: {lost:3d} lost, {won:3d} gained ({lost+won} flips of {n})")
print(f"  main line mean flips between consecutive students: {np.mean(tot):.0f}")
always = None
for k in ks[2:]:
    e = episodes(R/f"dagger10m/round{k}")
    v = {x: e[x]["success"] for x in e}
    always = {x: always.get(x, 0) + v[x] for x in v} if always else {x: int(v[x]) for x in v}
m = len(ks[2:]); c = np.array(list(always.values()))
print(f"  over the {m} main-line tests since r{ks[2]}: scenes solved every time {int((c == m).sum())}, never {int((c == 0).sum())}, "
      f"sometimes {int(((c > 0) & (c < m)).sum())} (of {len(c)})")
