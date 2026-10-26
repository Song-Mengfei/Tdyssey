"""Evidence for why long forward runs degrade (standard library only; reads local run files).

Usage: python3 -I degrade_report.py RUN_DIR [RUN_DIR ...]   (each runs/<group>/<job>/<job>)
Per run:
  retention   weighted T-retention term vs the main loss (is the anchor to the start policy active?)
  saturation  fraction of clipped advantages |a| = 1 and tied groups (detailed recent updates)
  probe       fixed-noise error per denoising timestep: start vs latest (where did the policy change?)
  eval T      per-scene T outcomes: success, peak and final coverage at each evaluation
"""

import json
from pathlib import Path
import statistics as st
import sys


def main():
    for run in map(Path, sys.argv[1:]):
        report = json.loads((run/"report.json").read_text())
        cfg = report["config"]
        rows = [json.loads(l)["metrics"] for l in (run/"metrics.jsonl").read_text().splitlines() if l.strip()]
        print(f"=== {run.name}  ({cfg['objective']}, {cfg['trainable']}, lr {cfg['actor_lr']:g}, "
              f"reference_weight {cfg['reference_weight']:g})")
        losses = [m["train/loss"] for m in rows if "train/loss" in m]
        retention = [m["train/T_retention_loss"] for m in rows if "train/T_retention_loss" in m]
        if losses and retention:
            print(f"   retention  main loss mean {st.mean(losses):.4f}; weighted retention mean "
                  f"{cfg['reference_weight']*st.mean(retention):.2e} ({cfg['reference_weight']*st.mean(retention)/max(st.mean(map(abs, losses)), 1e-12):.1e} of main)")
        adv, tied, groups = [], 0, 0
        for update in report.get("updates", []):
            for group in update.get("groups") or []:
                feedback = group.get("feedback")
                if not feedback:
                    continue
                groups += 1
                tied += len(set(c["score"] for c in group["candidates"])) == 1
                adv += feedback["advantages"]
        if adv:
            print(f"   saturation {sum(abs(a) >= 1 for a in adv)/len(adv):.0%} of {len(adv)} advantages clipped to +-1; "
                  f"{tied}/{groups} groups tied (last detailed updates)")
        probes = [m for m in rows if "probe/T/noise_mse_t0" in m]
        if len(probes) > 1:
            cells = []
            for t in (0, 25, 50, 99):
                a, b = probes[0][f"probe/T/noise_mse_t{t}"], probes[-1][f"probe/T/noise_mse_t{t}"]
                cells.append(f"t{t}: {a:.2e}->{b:.2e} ({b/a:.2f}x)")
            print("   probe T   " + "  ".join(cells))
        evals = [(e["update"], e["episodes"]) for e in report.get("evaluations", [])]
        if report.get("baseline"):
            evals = [(0, report["baseline"])] + evals
        for update, episodes in evals:
            t = [e for e in episodes if e["shape"] == "T"]
            if t:
                print(f"   eval T u{update:<4} success {sum(e['success'] for e in t)}/{len(t)}  "
                      f"peak {st.mean(e['peak_coverage'] for e in t):.3f}  final {st.mean(e['final_coverage'] for e in t):.3f}  "
                      f"peak>=0.95 but not success {sum(e['peak_coverage'] >= .95 and not e['success'] for e in t)}")


if __name__ == "__main__":
    main()
