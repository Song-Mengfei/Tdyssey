"""Render existing scalar evidence only; no model or physical operations."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(__file__).resolve().parent
OUT = LOCAL / "review-final"
if OUT.exists():
    raise SystemExit("Refusing to overwrite review directory")
OUT.mkdir()
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

inputs = {}
def identity(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def read(path):
    inputs[str(path.relative_to(ROOT))] = identity(path)
    return json.loads(path.read_text(encoding="utf-8"))

trace = read(LOCAL / "traces-final/analysis.json")
training = read(LOCAL / "training-final/analysis.json")
base, delta = [read(ROOT / "artifacts/milestone-c-evaluation-20261024" / (m + ".json"))
               for m in ("base_training", "delta_training")]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = {"T": "#2166ac", "L": "#d6604d"}
for shape in ("T", "L"):
    rows = [r for r in training["tables"]["shape_batches"] if r["shape"] == shape]
    x = [r["batch"] for r in rows]
    axes[0, 0].plot(x, [100*r["contact_fraction"] if r["contact_fraction"] is not None else math.nan for r in rows], "o-", label=shape, color=colors[shape])
    axes[0, 1].plot(x, [r["max_coverage"] if r["controls"] else math.nan for r in rows], "o-", label=shape, color=colors[shape])
axes[0, 0].set(xlabel="Fresh batch index", ylabel="Recorded contact controls (%)", ylim=(-2, 102))
axes[0, 1].set(xlabel="Fresh batch index", ylabel="Within-batch maximum coverage", ylim=(-.02, 1.02))
for ax in axes[0]:
    ax.legend(); ax.set_xticks(range(1, 19, 2))
passes = training["tables"]["passes"]
axes[1, 0].plot(range(1, len(passes)+1), [r["critic_explained_variance"] for r in passes], ".-", color="#756bb1")
axes[1, 0].axhline(0, color="black", lw=.7)
axes[1, 0].set(xlabel="Actor / critic update ordinal", ylabel="Mixed in-buffer explained variance")
batches = training["tables"]["batches"]
for prefix, label, color in (("returns", "Value target", "#238b45"), ("raw_advantages", "Raw advantage", "#88419d")):
    axes[1, 1].errorbar([r["batch"] for r in batches], [r[prefix+"_mean"] for r in batches],
                        yerr=[r[prefix+"_std"] for r in batches], fmt="o-", capsize=2, label=label, color=color)
axes[1, 1].axhline(0, color="black", lw=.7)
axes[1, 1].set(xlabel="Fresh batch index", ylabel="Mixed mean +/- population SD")
axes[1, 1].legend()
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle("Saved C training diagnostics\nChanging scenes and reused buffers: descriptive exposure, not policy learning curves", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .94))
for extension in ("png", "pdf"):
    fig.savefig(OUT / ("training_diagnostics." + extension), dpi=170)
plt.close(fig)

contrasts = []
for shape, scene, seed in [("T",100001,5478), ("L",100000,5478), ("L",100001,5479), ("L",100002,5479)]:
    a, b = [next(c for c in g["cases"] if (c["shape"],c["scene_seed"],c["policy_seed"]) == (shape,scene,seed))
            for g in (base, delta)]
    distances = [math.dist(x["action"],y["action"]) for x,y in zip(a["steps"],b["steps"])]
    contrasts.append({"shape":shape,"scene_seed":scene,"policy_seed":seed,
        "initial_states_equal":a["initial_state"] == b["initial_state"],
        "first_action_difference_control":next((i+1 for i,d in enumerate(distances) if d>1e-9),None),
        "first_eight_target_distance_mean_pixels":mean(distances[:8]),
        "first_eight_target_distance_max_pixels":max(distances[:8]),
        "shared_prefix_mean_target_distance_pixels":mean(distances),
        "shared_prefix_max_target_distance_pixels":max(distances),
        "caveat":"Action targets only. Same ordinal after divergence does not imply the same physical state."})
(OUT / "early_action_contrasts.json").write_text(json.dumps(contrasts,indent=2)+"\n",encoding="utf-8")

lines = ["# All saved paired cases", "", "Coverage entries are percentages; step0 means initial measurement, not policy-generated progress.", ""]
for shape in ("L", "T"):
    lines += [f"## {shape}", "", "| Scene / seed | First positive base / delta | Peak base @ control | Peak delta @ control | Final base / delta | Positive executed fraction base / delta | Controls base / delta |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for pair in trace["paired_cases"]:
        if pair["shape"] != shape:
            continue
        a,b = [next(c for c in trace["case_summaries"] if (c["shape"],c["scene_seed"],c["policy_seed"],c["method"]) ==
                    (shape,pair["scene_seed"],pair["policy_seed"],m)) for m in ("base_training","delta_training")]
        first = lambda c: "never" if c["first_positive_coverage_control"] is None else str(c["first_positive_coverage_control"])
        lines.append(f"| {a['scene_seed']} / {a['policy_seed']} | {first(a)} / {first(b)} | {100*a['peak_coverage']:.2f}% @ {a['peak_control_first']} | {100*b['peak_coverage']:.2f}% @ {b['peak_control_first']} | {100*a['final_coverage']:.2f}% / {100*b['final_coverage']:.2f}% | {100*a['positive_coverage_fraction']:.1f}% / {100*b['positive_coverage_fraction']:.1f}% | {a['controls']} / {b['controls']} |")
    lines.append("")
(OUT / "all_case_tables.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
inputs[str(Path(__file__).relative_to(ROOT))] = identity(Path(__file__))
manifest = {"kind":"offline_review_render","inputs":inputs,"matplotlib_version":matplotlib.__version__,
            "new_controls":0,"new_resets":0,"new_policy_samples":0,"new_optimizer_steps":0,
            "outputs":{str(p.relative_to(OUT)):identity(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
(OUT / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"output":str(OUT),"input_count":len(inputs),"output_count":len(manifest["outputs"])}))
