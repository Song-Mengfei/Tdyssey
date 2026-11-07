#!/usr/bin/env python3
"""Regenerate every presentation figure, the LaTeX tables and figures/README.md from presentation/data/*.csv.

No GPU, simulator or network. Needs matplotlib and numpy only.

    MPLBACKEND=Agg python presentation/scripts/make_figures.py            # everything
    MPLBACKEND=Agg python presentation/scripts/make_figures.py --only final_ heldout_

Every figure is written twice into presentation/figures/: a vector PDF (Beamer) and a 200 dpi PNG (PowerPoint).
Tables go to presentation/data/tables/*.tex (booktabs). A few small charts use numbers typed in from
docs/LEARNING_FINDINGS.md, docs/LEARNING_EXPERIMENT_LOG.md and docs/DECISIONS.md; each such block is marked TYPED-IN
with the document section it comes from.

Colours follow the model in every figure (validated with the dataviz palette checker, white surface):
original = light grey, 5-shape students = raspberry, 10-shape students = orange (192 px camera: dark orange),
26-letter 96 px = blue, 26-letter 192 px = aqua, planning teacher = gold.
"""
import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")      # reproducible PDFs: no creation time, so reruns do not change files
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TYPED = json.load(open(DATA / "typed_in_numbers.json"))   # numbers typed in from the docs (with sources)
FIG = ROOT / "figures"
TAB = DATA / "tables"

# ----------------------------------------------------------------------------------------------------------------------
# Palette and style
# ----------------------------------------------------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f4f3ef"
ORIG, ORIG_LINE = "#b4b2ab", "#8f8d86"          # original policy: fills / lines and markers
FIVE = "#c2408a"                                  # 5-shape students
TEN, TEN192 = "#eb6834", "#a8431a"                # 10-shape students, 96 px / 192 px camera
M96, M192 = "#2a78d6", "#1baf7a"                  # 26-letter students, 96 px / 192 px camera
TEACHER = "#eda100"                               # planning teacher (reads exact simulator state)
BLEND = "#e87ba4"                                 # early weight blends (5 shapes)
RL = "#6250d6"                                    # forward-process RL runs
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6"]     # ordinal light -> dark (validated --ordinal)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def tint(hex_color, amount):
    """Mix a colour with white; amount 0 = unchanged, 1 = white."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


FULL, HALF, TALL, BIG = (10, 5.6), (6, 4.5), (7.5, 9.5), (10, 7)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16, "axes.titleweight": "bold",
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 2.2, "lines.markersize": 6,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.3, "errorbar.capsize": 3,
    })


PCT = FuncFormatter(lambda v, _: f"{round(v, 1):g}%")
SIGNED = FuncFormatter(lambda v, _: (f"{v:+g}".replace("-", "−")) if abs(v) > 1e-9 else "0")

# ----------------------------------------------------------------------------------------------------------------------
# Shapes, groups and model labels
# ----------------------------------------------------------------------------------------------------------------------
FIVE_SHAPES = ["T", "T_narrow", "Y", "D", "L"]
OLD10 = FIVE_SHAPES + ["E", "F", "J", "P", "U"]
NEW17 = list("ABCGHIKMNOQRSVWXZ")
HELD = ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]
TRAINED27 = OLD10 + NEW17
ALL33 = TRAINED27 + HELD
GROUP_LABEL = {"10 earlier shapes": "10 earlier shapes", "17 other letters": "17 newer letters",
               "27 trained shapes": "All 27 trained shapes", "6 held-out shapes": "6 never-seen shapes",
               "5 first shapes": "5 first shapes"}
GROUP_SHAPES = {"5 first shapes": FIVE_SHAPES, "10 earlier shapes": OLD10, "17 other letters": NEW17,
                "27 trained shapes": TRAINED27, "6 held-out shapes": HELD}
DISP = {"T_narrow": "near-T", "digit3": "digit 3", "digit4": "digit 4", "digit7": "digit 7"}
SHORT = {"T_narrow": "nT", "digit3": "3", "digit4": "4", "digit7": "7", "plus": "+", "hash": "#", "arrow": "→"}


def disp(s):
    return DISP.get(s, s)


# model key -> (label in figures, colour); CSV names are mapped onto these keys per test
LAB = {
    "original": ("Original (T demos only)", ORIG),
    "five": ("Ours, 5 shapes", FIVE),
    "ten96": ("Ours, 10 shapes", TEN),
    "ten192": ("Ours, 10 shapes, 192 px camera", TEN192),
    "m96": ("Ours, 26 letters", M96),
    "m192": ("Ours, 26 letters, 192 px camera", M192),
    "teacher": ("Planning teacher (exact state)", TEACHER),
}
D051 = {"original": "original", "ten96": "10-shape 96 px", "m96": "26-letter 96 px (round 44)",
        "m192": "26-letter 192 px (round 2)"}
D049 = {"original": "original", "ten96": "10-shape 96 px", "ten192": "10-shape 192 px"}
TEST_NOTE = "final test, 300 fresh scenes per shape"


def lab(key):
    return LAB[key][0]


def col(key):
    return LAB[key][1]


# ----------------------------------------------------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------------------------------------------------
def read(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


VAL = read("validation_rounds.csv")
BYSHAPE = defaultdict(dict)
for _r in read("final_tests_by_shape.csv"):
    BYSHAPE[(_r["test"], _r["model"])][_r["shape"]] = _r
PAIRED = {}
for _r in read("final_tests_paired.csv"):
    PAIRED[(_r["test"], _r["base_model"], _r["model"], _r["group"])] = _r


def _load_episodes():
    tmp = defaultdict(lambda: defaultdict(list))
    for r in read("final_tests_episodes.csv"):
        d = tmp[(r["test"], r["model"])]
        d["shape"].append(r["shape"]); d["seed"].append(int(r["scene_seed"]))
        d["success"].append(int(r["success"])); d["controls"].append(int(r["controls"]))
        d["final"].append(float(r["final_coverage"])); d["peak"].append(float(r["peak_coverage"]))
    out = {}
    for k, d in tmp.items():
        order = np.lexsort((np.array(d["seed"]), np.array(d["shape"])))
        out[k] = {c: np.array(v)[order] for c, v in d.items()}
    return out


EPIS = _load_episodes()


def eps(test, model, shapes=None):
    d = EPIS[(test, model)]
    if shapes is None:
        return d
    m = np.isin(d["shape"], shapes)
    return {c: v[m] for c, v in d.items()}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def rate(test, model, shapes):
    """Pooled success % over shapes, successes, episodes, Wilson interval."""
    rows = [BYSHAPE[(test, model)][s] for s in shapes if s in BYSHAPE[(test, model)]]
    k = sum(int(r["successes"]) for r in rows)
    n = sum(int(r["episodes"]) for r in rows)
    lo, hi = wilson(k, n)
    return 100 * k / n, k, n, lo, hi


def coverage(test, model, shapes, col_="mean_final_coverage"):
    rows = [BYSHAPE[(test, model)][s] for s in shapes if s in BYSHAPE[(test, model)]]
    n = sum(int(r["episodes"]) for r in rows)
    return 100 * sum(float(r[col_]) * int(r["episodes"]) for r in rows) / n


def paired_row(test, base, model, group):
    """Paired difference vs base for a group name or a single shape; recomputed from episodes when available
    (avoids the 4-decimal rounding of final_tests_paired.csv), otherwise read from that file."""
    if (test, base) in EPIS and (test, model) in EPIS:
        return paired_eps(test, base, model, GROUP_SHAPES.get(group, [group]))
    r = PAIRED[(test, base, model, group)]
    return {k: 100 * float(r[k]) for k in ("base_rate", "model_rate", "difference", "ci95_low", "ci95_high")} | \
        {"n": int(r["paired_episodes"])}


def paired_eps(test, base, model, shapes, test_model=None):
    """Paired difference (model - base) in points with the same normal-approximation 95% interval as build_data.py."""
    a = eps(test, base, shapes)
    b = eps(test_model or test, model, shapes)
    ka = {(s, q): v for s, q, v in zip(a["shape"], a["seed"], a["success"])}
    kb = {(s, q): v for s, q, v in zip(b["shape"], b["seed"], b["success"])}
    keys = [k for k in ka if k in kb]
    d = np.array([kb[k] - ka[k] for k in keys], dtype=float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    return {"n": n, "base_rate": 100 * np.mean([ka[k] for k in keys]), "model_rate": 100 * np.mean([kb[k] for k in keys]),
            "difference": 100 * m, "ci95_low": 100 * (m - 1.96 * se), "ci95_high": 100 * (m + 1.96 * se)}


def val_line(stage, line, shapes=None, metric="success"):
    """One validation line: steps, value (% success or % mean final coverage), successes, episodes."""
    by = defaultdict(list)
    for r in VAL:
        if r["stage"] == stage and r["line"] == line and (shapes is None or r["shape"] in shapes):
            by[int(r["step"])].append(r)
    steps = sorted(by)
    ks = np.array([sum(int(r["successes"]) for r in by[s]) for s in steps])
    ns = np.array([sum(int(r["episodes"]) for r in by[s]) for s in steps])
    if metric == "success":
        ys = 100 * ks / ns
    else:
        ys = np.array([100 * sum(float(r["mean_final_coverage"]) * int(r["episodes"]) for r in by[s]) / sum(
            int(r["episodes"]) for r in by[s]) for s in steps])
    return np.array(steps), ys, ks, ns


def val_matrix(stage, line, shapes, metric="success_rate"):
    by = defaultdict(dict)
    for r in VAL:
        if r["stage"] == stage and r["line"] == line:
            by[r["shape"]][int(r["step"])] = 100 * float(r[metric])
    steps = sorted(next(iter(by.values())))
    return np.array(steps), np.array([[by[s][t] for t in steps] for s in shapes])


def rolling(y, w):
    h = w // 2
    return np.array([np.mean(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def wilson_band(ks, ns):
    lo, hi = zip(*[wilson(k, n) for k, n in zip(ks, ns)])
    return np.array(lo), np.array(hi)


# ----------------------------------------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------------------------------------
REGISTRY = []
ONLY = []


def wanted(name):
    return not ONLY or any(name.startswith(p) for p in ONLY)


def save(fig, name, shows, data, use, title=None):
    """Write <name>.pdf/.png; with title also write <name>_titled.* carrying that title."""
    FIG.mkdir(parents=True, exist_ok=True)
    txt = getattr(fig, "_tdy_note", None)
    if txt:   # footnote: wrapped to the drawing's width and placed just below everything else
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        fig.text(max(bb.x0, 0) / w, (bb.y0 - 0.06) / h, textwrap.fill(txt, width=max(40, int(bb.width * 12.5))),
                 fontsize=12, color=INK2, ha="left", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    REGISTRY.append((name, shows, data, use))
    if title:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        sup = fig.suptitle(title, fontsize=18, fontweight="bold", x=max(bb.x0, 0) / w, ha="left", va="bottom",
                           y=(bb.y1 + 0.12) / h)
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"{name}_titled.{ext}", dpi=200)
        sup.remove()
        REGISTRY.append((f"{name}_titled", "Same as above with a title for standalone reading.", data, "standalone use"))
    plt.close(fig)


def note(fig, text, y=None):
    fig._tdy_note = text


LABEL_NEAR = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"), (-5, -3, "right", "top"),
              (9, 0, "left", "center"), (-9, 0, "right", "center"), (0, 8, "center", "bottom"),
              (0, -8, "center", "top")]
LABEL_FAR = [(18, 12, "left", "bottom"), (-18, 12, "right", "bottom"), (18, -12, "left", "top"),
             (-18, -12, "right", "top"), (26, 0, "left", "center"), (-26, 0, "right", "center"),
             (0, 20, "center", "bottom"), (0, -20, "center", "top")]


def place_labels(ax, names, xs, ys, size=12, only=None, text=disp):
    """Greedy non-overlapping point labels: try 8 nearby spots, then 8 farther ones with a leader line; skip if none
    fits. Avoids other points, earlier labels, the legend and the axes edge. Returns the names left unlabelled."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    pad = 4.5 * fig.dpi / 72
    obst = [Bbox.from_extents(px - pad, py - pad, px + pad, py + pad) for px, py in pts]
    placed = []
    leg = ax.get_legend()
    if leg is not None:
        placed.append(leg.get_window_extent(r))
    axbb = ax.get_window_extent(r)
    skipped = []
    for i, nm in enumerate(names):
        if only is not None and nm not in only:
            continue
        done = False
        for far, cands in ((False, LABEL_NEAR), (True, LABEL_FAR)):
            for dx, dy, ha, va in cands:
                t = ax.annotate(text(nm), (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                fontsize=size, color=INK, zorder=6,
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=1, shrinkB=4) if far else None)
                t.update_positions(r)
                bb = Text.get_window_extent(t, r)
                ok = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
                ok = ok and not any(bb.overlaps(o) for j, o in enumerate(obst) if j != i)
                ok = ok and not any(bb.overlaps(p) for p in placed)
                if ok:
                    placed.append(bb)
                    done = True
                    break
                t.remove()
            if done:
                break
        if not done:
            skipped.append(nm)
    return skipped


def bars_with_pooled(groups, keys, getter, pooled_label, size=FULL, colors=None, ratio=2.4):
    """Per-item grouped bars on the left and the pooled group in its own narrow panel on the right (shared y).
    getter(key, item_or_None) -> (rate, lo, hi); None means pooled."""
    fig, (ax, axp) = plt.subplots(1, 2, figsize=size, sharey=True,
                                  gridspec_kw={"width_ratios": [len(groups), ratio], "wspace": 0.03})
    vals, los, his, pv, pl, ph = [], [], [], [], [], []
    for k in keys:
        rows = [getter(k, g) for g in groups]
        vals.append([r[0] for r in rows]); los.append([r[1] for r in rows]); his.append([r[2] for r in rows])
        p = getter(k, None)
        pv.append([p[0]]); pl.append([p[1]]); ph.append([p[2]])
    grouped_bars(ax, groups, keys, vals, los, his, labels=False, colors=colors)
    grouped_bars(axp, [pooled_label], keys, pv, pl, ph, labels=True, colors=colors, width=0.86, size=12)
    for t in axp.texts:
        t.set_fontweight("bold")
    axp.spines["left"].set_visible(False)
    axp.tick_params(axis="y", left=False)
    axp.set_xlim(-0.5, 0.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    return fig, ax, axp






def errbars(ax, x, y, lo, hi, horizontal=False, color=INK2, lw=1.2):
    lo, hi, y = np.asarray(lo), np.asarray(hi), np.asarray(y)
    err = np.vstack([y - lo, hi - y])
    if horizontal:
        ax.errorbar(y, x, xerr=err, fmt="none", ecolor=color, elinewidth=lw, capsize=3, zorder=4)
    else:
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, elinewidth=lw, capsize=3, zorder=4)


def legend_top(ax, handles=None, ncol=4, y=1.02, **kw):
    kw.setdefault("loc", "lower left")
    kw.setdefault("columnspacing", 1.4)
    return ax.legend(handles=handles, ncol=ncol, bbox_to_anchor=(0, y), handletextpad=0.5, **kw)


def patch(key_or_color, label=None, hatch=None):
    if key_or_color in LAB:
        return Patch(facecolor=col(key_or_color), edgecolor="white", label=label or lab(key_or_color), hatch=hatch)
    return Patch(facecolor=key_or_color, edgecolor="white", label=label, hatch=hatch)


def fmt_signed(v, nd=1):
    s = f"{abs(v):.{nd}f}"
    return ("+" if v >= 0 else "$-$") + s


# ----------------------------------------------------------------------------------------------------------------------
# curves_*: validation success over training
# ----------------------------------------------------------------------------------------------------------------------
def story_lines():
    """All validation lines placed on one x-axis: stage 1 at 1-10, stage 2 at 13-55, stage 3 at 59-108."""
    out = []
    s, y, k, n = val_line("5 shapes", "correction rounds")
    out.append(dict(key="five", label="5 shapes", color=FIVE, stage=0, x=s, y=y, k=k, n=n))
    s, y, k, n = val_line("10 shapes", "correction rounds, executed-push labels")
    out.append(dict(key="exec", label="10 shapes, executed-push labels", color=tint(TEN, 0.45), stage=1, x=s - 1, y=y,
                    k=k, n=n))
    s, y, k, n = val_line("10 shapes", "correction rounds, intended-plan labels")
    out.append(dict(key="plan", label="10 shapes, plan labels", color=TEN, stage=1, x=s - 1 + 16, y=y, k=k, n=n))
    s, y, k, n = val_line("10 shapes", "96 px main line")
    out.append(dict(key="ten96", label="10 shapes, 96 px", color=TEN, stage=1, x=s - 1 + 21, y=y, k=k, n=n))
    s, y, k, n = val_line("10 shapes", "192 px side line")
    out.append(dict(key="ten192", label="10 shapes, 192 px", color=TEN192, stage=1, x=s - 1 + 21, y=y, k=k, n=n))
    s, y, k, n = val_line("27 shapes", "96 px line")
    out.append(dict(key="m96", label="26 letters, 96 px", color=M96, stage=2, x=s - 1, y=y, k=k, n=n))
    s, y, k, n = val_line("27 shapes", "192 px line")
    out.append(dict(key="m192", label="26 letters, 192 px", color=M192, stage=2, x=s - 1, y=y, k=k, n=n))
    return out


STAGE_OFF = [0, 13, 59]
STAGE_SPAN = [(0.5, 10.5), (12.5, 55.5), (58.5, 108.5)]
STAGE_NAME = ["5 shapes", "10 shapes", "27 shapes (26 letters + near-T)"]
# picks that went to a fresh-scene test: (line key, within-line x)
PICKS = [("five", 4), ("five", 7), ("ten96", 21 + 13), ("ten192", 21 + 6), ("m96", 44), ("m192", 2)]
# original policy on each stage's shapes, fresh-scene tests (D047 pool, D049 pool, D051 pool)


def original_stage_levels():
    return [paired_row("D047", "original", "5-shape round 7", "5 first shapes")["base_rate"],
            paired_row("D049/D050", "original", "10-shape 96 px", "10 earlier shapes")["base_rate"],
            paired_row("D051", "original", "26-letter 96 px (round 44)", "27 trained shapes")["base_rate"]]


def fig_curves_all_stages(smooth=False):
    lines = story_lines()
    fig, ax = plt.subplots(figsize=FULL)
    for i, (a, b) in enumerate(STAGE_SPAN):
        ax.axvspan(a, b, color=BAND if i % 2 == 0 else "#fbfbf9", zorder=0, lw=0)
        ax.text((a + b) / 2, 51.5, STAGE_NAME[i], ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
    orig = original_stage_levels()
    for i, (a, b) in enumerate(STAGE_SPAN):
        ax.plot([a + 0.3, b - 0.3], [orig[i]] * 2, color=ORIG_LINE, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(STAGE_SPAN[2][1] - 0.5, orig[2] + 0.8, "original policy", ha="right", va="bottom", fontsize=12,
            color=INK2)
    ax.text(STAGE_SPAN[1][1] - 0.5, orig[1] - 0.8, "original policy", ha="right", va="top", fontsize=12,
            color=INK2)
    # sub-phases of stage 2 (labels at the bottom, where the plot is empty)
    for cx in (16, 20.5):
        ax.axvline(STAGE_OFF[1] + cx, color=AXIS, lw=0.9, ymax=0.9, zorder=1)
    for cx, y, t in ((7.5, 1.0, "executed-push\nlabels"), (18.3, 11.5, "plan\nlabels"), (32, 1.0, "96 px and\n192 px lines")):
        ax.text(STAGE_OFF[1] + cx, y, t, ha="center", va="bottom", fontsize=12, color=INK2)
    bx = STAGE_OFF[2] + 13.5
    ax.axvline(bx, color=AXIS, lw=0.9, ymax=0.9, zorder=1)
    ax.text(bx + 0.8, 7.0, "larger training\nbudget per round", ha="left", va="bottom", fontsize=12, color=INK2)
    for L in lines:
        pos = STAGE_OFF[L["stage"]] + L["x"]
        if smooth:
            ax.plot(pos, L["y"], color=L["color"], lw=1.2, alpha=0.35, zorder=2)
            w = 3 if len(L["y"]) <= 10 else 5
            ax.plot(pos, rolling(L["y"], w), color=L["color"], lw=2.8, zorder=3, label=L["label"])
        else:
            ax.plot(pos, L["y"], color=L["color"], lw=2.0, marker="o", ms=3.6, zorder=3, label=L["label"])
    if not smooth:
        byk = {L["key"]: L for L in lines}
        for key, x in PICKS:
            L = byk[key]
            i = int(np.where(L["x"] == x)[0][0])
            ax.plot(STAGE_OFF[L["stage"]] + x, L["y"][i], marker="o", ms=11, mfc="none", mec=INK, mew=1.6, zorder=5)
    ax.set_xlim(0, 109.5)
    ax.set_ylim(0, 52)
    ax.yaxis.set_major_formatter(PCT)
    ticks, labels = [], []
    for st, vals in ((0, [1, 5, 10]), (1, [0, 10, 20, 30, 40]), (2, [0, 10, 20, 30, 40, 49])):
        for v in vals:
            ticks.append(STAGE_OFF[st] + v)
            labels.append(str(v))
    ax.set_xticks(ticks, labels)
    ax.grid(False, axis="x")
    ax.set_xlabel("Correction round (counted within each stage)")
    ax.set_ylabel("Validation success")
    handles = [Line2D([], [], color=FIVE, lw=2.4, label="5 shapes"),
               Line2D([], [], color=tint(TEN, 0.45), lw=2.4, label="10 shapes, executed-push labels"),
               Line2D([], [], color=TEN, lw=2.4, label="10 shapes, plan labels, 96 px"),
               Line2D([], [], color=TEN192, lw=2.4, label="10 shapes, 192 px"),
               Line2D([], [], color=M96, lw=2.4, label="26 letters, 96 px"),
               Line2D([], [], color=M192, lw=2.4, label="26 letters, 192 px")]
    if not smooth:
        handles.append(Line2D([], [], marker="o", ms=10, mfc="none", mec=INK, mew=1.6, ls="none",
                              label="sent to a fresh-scene test"))
    legend_top(ax, handles=handles, ncol=4 if not smooth else 3, fontsize=12, columnspacing=1.0)
    note(fig, "Validation: 50 fixed scenes per shape, averaged over the stage's own shapes. Dashed: original policy "
              "on fresh test scenes of the same shapes.")
    return fig


def fig_curves_all_stages_panels():
    lines = story_lines()
    fig, axes = plt.subplots(1, 3, figsize=FULL, sharey=True, gridspec_kw={"width_ratios": [14, 43, 50], "wspace": 0.06})
    orig = original_stage_levels()
    for st, ax in enumerate(axes):
        for L in [L for L in lines if L["stage"] == st]:
            ax.plot(L["x"], L["y"], color=L["color"], lw=2.0, marker="o", ms=3.4, label=L["label"])
        ax.axhline(orig[st], color=ORIG_LINE, lw=1.6, ls=(0, (4, 3)))
        ax.set_title(STAGE_NAME[st].replace(" (26 letters + near-T)", ""), loc="left", fontsize=15)
        ax.grid(False, axis="x")
    axes[0].set_xticks([1, 5, 10])
    axes[0].set_xlim(0.3, 10.7)
    axes[1].set_xticks([0, 10, 20, 30, 40])
    axes[1].set_xlim(-1, 43)
    axes[1].axvline(16, color=AXIS, lw=0.9)
    axes[1].axvline(20.5, color=AXIS, lw=0.9)
    axes[1].text(8, 2, "executed-push\nlabels", ha="center", fontsize=12, color=INK2)
    axes[1].text(31.5, 2, "96 px / 192 px lines", ha="center", fontsize=12, color=INK2)
    axes[2].set_xticks([0, 10, 20, 30, 40, 49])
    axes[2].set_xlim(-1, 50)
    axes[2].axvline(13.5, color=AXIS, lw=0.9)
    axes[2].text(14.3, 6, "larger budget\nper round", fontsize=12, color=INK2)
    axes[2].text(49, orig[2] + 0.8, "original", ha="right", va="bottom", fontsize=12, color=INK2)
    axes[0].set_ylim(0, 47)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success")
    fig.supxlabel("Correction round (counted within each stage)", fontsize=16, y=0.0)
    handles = [Line2D([], [], color=FIVE, lw=2.4, label="5 shapes"),
               Line2D([], [], color=tint(TEN, 0.45), lw=2.4, label="10 shapes, executed-push labels"),
               Line2D([], [], color=TEN, lw=2.4, label="10 shapes, plan labels / 96 px"),
               Line2D([], [], color=TEN192, lw=2.4, label="10 shapes, 192 px"),
               Line2D([], [], color=M96, lw=2.4, label="26 letters, 96 px"),
               Line2D([], [], color=M192, lw=2.4, label="26 letters, 192 px")]
    axes[0].legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.1), fontsize=12)
    return fig


def fig_curves_26letters(smooth=False):
    fig, ax = plt.subplots(figsize=FULL)
    ax.axvspan(-0.6, 13.5, color=BAND, lw=0, zorder=0)
    ax.text(6.5, 45.5, "old budget:\n20k training steps per round", ha="center", va="top", fontsize=12, color=INK2)
    ax.text(14.2, 45.5, "54k steps per round + 60 practice scenes per new letter (from round 14)", ha="left", va="top",
            fontsize=12, color=INK2)
    for key, line, c in (("m96", "96 px line", M96), ("m192", "192 px line", M192)):
        s, y, k, n = val_line("27 shapes", line)
        x = s - 1
        lo, hi = wilson_band(k, n)
        ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
        if smooth:
            ax.plot(x, y, color=c, lw=1.2, alpha=0.4)
            ax.plot(x, rolling(y, 5), color=c, lw=2.8, label=lab(key))
        else:
            ax.plot(x, y, color=c, lw=2.0, marker="o", ms=4, label=lab(key))
    for key, line, x, txt, dy in (("m96", "96 px line", 44, "picked for the final test\n(round 44, 34.8%)", 5.5),
                                  ("m192", "192 px line", 2, "picked (round 2, 37.9%)", 3.0)):
        s, y, *_ = val_line("27 shapes", line)
        v = y[list(s - 1).index(x)]
        ax.plot(x, v, marker="o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=5)
        ax.annotate(txt, (x, v), xytext=(x + (-2 if key == "m96" else 3), v + dy), fontsize=12, color=INK,
                    ha="right" if key == "m96" else "left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlim(-0.6, 49.6)
    ax.set_ylim(15, 46)
    ax.yaxis.set_major_formatter(PCT)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(False, axis="x")
    ax.set_xlabel("Correction rounds completed")
    ax.set_ylabel("Validation success (27 shapes)")
    ax.legend(loc="lower right")
    note(fig, "50 fixed validation scenes per shape (1350 per round). Shaded bands: 95% binomial intervals."
         + (" Thick lines: 5-round rolling mean." if smooth else ""))
    return fig


def fig_curves_26letters_groups():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"width_ratios": [50, 16], "wspace": 0.08})
    for ax, key, line in ((axes[0], "m96", "96 px line"), (axes[1], "m192", "192 px line")):
        c = col(key)
        for g, shapes, ls, cc in (("10 earlier shapes", OLD10, (0, (5, 2)), tint(c, 0.25)),
                                  ("17 newer letters", NEW17, "-", c)):
            s, y, *_ = val_line("27 shapes", line, shapes)
            ax.plot(s - 1, y, color=cc, ls=ls, lw=2.2, marker="o", ms=3.2, label=g)
        ax.set_title(lab(key).replace("Ours, ", ""), loc="left", fontsize=15)
        ax.grid(False, axis="x")
    axes[0].axvline(13.5, color=AXIS, lw=0.9)
    axes[0].text(14.2, 5, "larger budget per round", fontsize=12, color=INK2)
    axes[0].set_xlim(-0.6, 49.6)
    axes[1].set_xlim(-0.6, 12.6)
    axes[0].set_ylim(0, 50)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success")
    axes[0].set_xlabel("Correction rounds completed")
    axes[0].legend(loc="upper left", ncol=2)
    note(fig, "50 fixed validation scenes per shape. The 17 newer letters start far behind and catch up once the "
              "per-round budget grows.")
    return fig


def fig_curves_26letters_coverage():
    fig, ax = plt.subplots(figsize=FULL)
    ax.axvspan(-0.6, 13.5, color=BAND, lw=0, zorder=0)
    for key, line in (("m96", "96 px line"), ("m192", "192 px line")):
        s, y, *_ = val_line("27 shapes", line, metric="coverage")
        ax.plot(s - 1, y, color=col(key), lw=2.0, marker="o", ms=4, label=lab(key))
    ax.text(6.5, 86.5, "old budget", ha="center", va="top", fontsize=12, color=INK2)
    ax.set_xlim(-0.6, 49.6)
    ax.set_ylim(70, 87)
    ax.yaxis.set_major_formatter(PCT)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(False, axis="x")
    ax.set_xlabel("Correction rounds completed")
    ax.set_ylabel("Mean final goal coverage")
    ax.legend(loc="lower right")
    note(fig, "Validation, 50 fixed scenes per shape, 27 shapes. Success needs more than 95% coverage.")
    return fig


def fig_curves_10shapes():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.08})
    ax = axes[0]
    s, y, *_ = val_line("10 shapes", "correction rounds, executed-push labels")
    ax.plot(s - 1, y, color=tint(TEN, 0.45), marker="o", ms=4, label="executed-push labels")
    s2, y2, *_ = val_line("10 shapes", "correction rounds, intended-plan labels")
    ax.plot(s2 - 1 + 16, y2, color=TEN, marker="o", ms=4, label="intended-plan labels")
    ax.axvline(16, color=AXIS, lw=0.9)
    ax.set_title("Which labels to copy", loc="left", fontsize=15)
    ax.set_xlabel("Correction rounds completed")
    ax.set_xlim(-0.8, 21)
    ax = axes[1]
    for key, line in (("ten96", "96 px main line"), ("ten192", "192 px side line")):
        s, y, *_ = val_line("10 shapes", line)
        ax.plot(s - 1, y, color=col(key), marker="o", ms=4, label="96 px camera" if key == "ten96" else "192 px camera")
    for key, line, x in (("ten96", "96 px main line", 13), ("ten192", "192 px side line", 6)):
        s, y, *_ = val_line("10 shapes", line)
        ax.plot(x, y[x], marker="o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=5)
    ax.plot([], [], marker="o", ms=10, mfc="none", mec=INK, mew=1.6, ls="none", label="sent to fresh-scene test")
    ax.set_title("Plan labels: camera resolution", loc="left", fontsize=15)
    ax.set_xlabel("Correction rounds completed")
    ax.set_xlim(-0.8, 21.8)
    for ax in axes:
        ax.axhline(78, color=TEACHER, lw=1.8, ls=(0, (4, 3)))
        ax.grid(False, axis="x")
        ax.legend(loc="lower right", fontsize=12)
    axes[0].text(0, 76, "teacher (reads exact state): 78%", fontsize=12, color=INK2, va="top")
    axes[0].set_ylim(0, 82)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success (10 shapes)")
    note(fig, "50 fixed validation scenes per shape. Teacher 390 of 500 on the same scenes (LEARNING_FINDINGS.md "
              "section 7).")
    return fig


def fig_curves_5shapes():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.08})
    ax = axes[0]
    for line, c, lbl in (("copy the teacher, no corrections (v1)", tint(FIVE, 0.45), "first run"),
                         ("copy the teacher, no corrections (long)", FIVE, "longer run")):
        s, y, *_ = val_line("5 shapes", line)
        ax.plot(s / 1000, y, color=c, marker="o", ms=5, label=lbl)
    ax.set_xlabel("Training steps (thousands)")
    ax.set_title("Copy the teacher, no corrections", loc="left", fontsize=15)
    ax.set_xlim(0, 155)
    ax = axes[1]
    s, y, *_ = val_line("5 shapes", "correction rounds")
    ax.plot(s, y, color=FIVE, marker="o", ms=5, label="correction rounds")
    for x in (4, 7):
        ax.plot(x, y[x - 1], marker="o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=5)
    ax.plot([], [], marker="o", ms=10, mfc="none", mec=INK, mew=1.6, ls="none", label="sent to fresh-scene test")
    ax.set_xlabel("Correction round")
    ax.set_title("With correction rounds", loc="left", fontsize=15)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    # TYPED-IN: DECISIONS.md D046 (original on the teacher's 50 validation scenes: 20+17+4+0+0 of 250) and
    # LEARNING_FINDINGS.md section 5 (final planner, same 50 scenes: 5+12+36+50+21 = 124 of 250).
    o5 = TYPED["original_on_teacher_validation_5shapes"]; t5 = TYPED["teacher_first_planner_5shapes_total"]
    o5p, t5p = o5["successes"] / o5["of"] * 100, t5["successes"] / t5["of"] * 100
    for ax in axes:
        ax.axhline(o5p, color=ORIG_LINE, lw=1.6, ls=(0, (4, 3)))
        ax.axhline(t5p, color=TEACHER, lw=1.8, ls=(0, (4, 3)))
        ax.grid(False, axis="x")
        ax.legend(loc="lower right", fontsize=12)
    axes[0].text(152, t5p + 1, f"teacher (exact state) {t5p:.1f}%", fontsize=12, color=INK2, ha="right")
    axes[0].text(152, o5p - 1.2, f"original {o5p:.1f}%", fontsize=12, color=INK2, ha="right", va="top")
    axes[0].set_ylim(0, 55)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success (5 shapes)")
    note(fig, "T, near-T, Y, D, L; 50 fixed validation scenes per shape. Original and teacher on the same scenes "
              "(DECISIONS.md D046; LEARNING_FINDINGS.md section 5).")
    return fig


def _heat(ax, M, steps, shapes, vmax, budget=True, vmin=0):
    im = ax.imshow(M, aspect="auto", cmap=SEQ, vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=(steps[0] - 0.5, steps[-1] + 0.5, len(shapes) - 0.5, -0.5))
    ax.set_yticks(range(len(shapes)), [disp(s) for s in shapes], fontsize=12)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.axhline(len(OLD10) - 0.5, color="white", lw=3)
    if budget:
        ax.axvline(13.5, color=INK, lw=1.2, ls=(0, (3, 2)))
    return im


def fig_heatmap_26(metric="success_rate", smooth=False, line="96 px line"):
    steps, M = val_matrix("27 shapes", line, TRAINED27, metric)
    x = steps - 1
    if smooth:
        M = np.array([rolling(r, 5) for r in M])
    big = line == "96 px line"
    fig, ax = plt.subplots(figsize=(10, 7) if big else (6.5, 7))
    vmin, vmax = (0, 60) if metric == "success_rate" else (40, 100)   # data range: success 0-60%, coverage 45-97%
    im = _heat(ax, M, x, TRAINED27, vmax, budget=big, vmin=vmin)
    ax.set_xlabel("Correction rounds completed")
    ax.xaxis.set_major_locator(MultipleLocator(5 if big else 2))
    for frac, t in ((1 - 5 / 27, "10 earlier shapes"), (1 - 18.5 / 27, "17 newer letters")):
        ax.text(-0.105 if big else -0.2, frac, t, transform=ax.transAxes, rotation=90, ha="center", va="center",
                fontsize=12, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.yaxis.set_major_formatter(PCT)
    cb.set_label(("Validation success" if metric == "success_rate" else "Mean final coverage")
                 + " (50 scenes)", fontsize=14)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=12)
    if big:
        ax.text(13.8, -1.0, "larger budget from round 14", fontsize=12, color=INK2, va="bottom")
    return fig


def fig_curves_small_multiples():
    steps, M = val_matrix("27 shapes", "96 px line", TRAINED27)
    steps2, M2 = val_matrix("27 shapes", "192 px line", TRAINED27)
    fig, axes = plt.subplots(5, 6, figsize=(12, 9), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(TRAINED27):
            ax.axis("off")
            continue
        ax.plot(steps - 1, M[i], color=M96, lw=0.8, alpha=0.35)
        ax.plot(steps - 1, rolling(M[i], 5), color=M96, lw=2)
        ax.plot(steps2 - 1, rolling(M2[i], 3), color=M192, lw=2)
        ax.text(0.04, 0.93, disp(TRAINED27[i]), transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        ax.axvline(13.5, color=AXIS, lw=0.8)
        ax.grid(False, axis="x")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PCT)
        ax.tick_params(labelsize=12)
    fig.supxlabel("Correction rounds completed", fontsize=16)
    fig.supylabel("Validation success (50 scenes)", fontsize=16)
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], color=M96, lw=2, label="26 letters, 96 px (5-round mean; faint: raw)"),
                        Line2D([], [], color=M192, lw=2, label="26 letters, 192 px (3-round mean)"),
                        Line2D([], [], color=AXIS, lw=1, label="larger budget from round 14")],
               loc="center", bbox_to_anchor=(0.76, 0.12), fontsize=13)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# final_*: fresh-scene final tests
# ----------------------------------------------------------------------------------------------------------------------
def grouped_bars(ax, groups, keys, values, lo=None, hi=None, labels=True, width=0.8, fmt="{:.1f}", size=12,
                 colors=None, hatches=None):
    nk = len(keys)
    w = width / nk
    xs = np.arange(len(groups))
    for j, k in enumerate(keys):
        x = xs - width / 2 + w * (j + 0.5)
        c = colors[j] if colors else col(k)
        bars = ax.bar(x, values[j], w, color=c, edgecolor="white", linewidth=1.5, zorder=3,
                      hatch=hatches[j] if hatches else None)
        if lo is not None:
            errbars(ax, x, values[j], lo[j], hi[j])
        if labels:
            tops = np.asarray(hi[j]) if hi is not None else np.asarray(values[j])
            for xi, t, v in zip(x, tops, values[j]):
                ax.text(xi, t + 0.6, fmt.format(v), ha="center", va="bottom", fontsize=size, color=INK)
    ax.set_xticks(xs, groups)
    ax.grid(False, axis="x")
    return xs


def fig_final_grouped_bars(groups=("10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"),
                           size=FULL, metric="success"):
    keys = ["original", "ten96", "m96", "m192"]
    vals, los, his = [], [], []
    for k in keys:
        v, lo, hi = [], [], []
        for g in groups:
            if metric == "success":
                r, _, _, l_, h_ = rate("D051", D051[k], GROUP_SHAPES[g])
                v.append(r); lo.append(l_); hi.append(h_)
            else:
                v.append(coverage("D051", D051[k], GROUP_SHAPES[g]))
        vals.append(v); los.append(lo); his.append(hi)
    fig, ax = plt.subplots(figsize=size)
    if metric == "success":
        grouped_bars(ax, [GROUP_LABEL[g] for g in groups], keys, vals, los, his)
        ax.set_ylabel(f"Success\n({TEST_NOTE})")
        ax.set_ylim(0, 42)
    else:
        grouped_bars(ax, [GROUP_LABEL[g] for g in groups], keys, vals, fmt="{:.0f}")
        ax.set_ylabel(f"Mean final goal coverage\n({TEST_NOTE})")
        ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PCT)
    legend_top(ax, handles=[patch(k) for k in keys], ncol=2)
    return fig


def fig_final_stage_progression():
    # fresh-scene tests at each stage, each paired with the original on the same scenes
    rows = [("5 shapes\n(round 7)", "D047", "5-shape round 7", "5 first shapes", "five"),
            ("10 shapes\n(96 px)", "D049/D050", "10-shape 96 px", "10 earlier shapes", "ten96"),
            ("27 shapes\n(26 letters, 96 px)", "D051", "26-letter 96 px (round 44)", "27 trained shapes", "m96")]
    fig, ax = plt.subplots(figsize=FULL)
    xs = np.arange(len(rows))
    w = 0.36
    for i, (name, test, model, group, key) in enumerate(rows):
        shapes = GROUP_SHAPES[group]
        rb, _, _, lb, hb = rate(test, "original", shapes)
        rm, _, _, lm, hm = rate(test, model, shapes)
        ax.bar(xs[i] - w / 2, rb, w, color=ORIG, edgecolor="white", lw=1.5, zorder=3)
        ax.bar(xs[i] + w / 2, rm, w, color=col(key), edgecolor="white", lw=1.5, zorder=3)
        errbars(ax, [xs[i] - w / 2], [rb], [lb], [hb])
        errbars(ax, [xs[i] + w / 2], [rm], [lm], [hm])
        ax.text(xs[i] - w / 2, hb + 0.7, f"{rb:.1f}", ha="center", fontsize=13)
        ax.text(xs[i] + w / 2, hm + 0.7, f"{rm:.1f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.grid(False, axis="x")
    ax.set_ylim(0, 42)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success on the stage's shapes\n(fresh-scene final test)")
    legend_top(ax, handles=[patch("original", "Original (T demos only)"), patch("five", "Ours, 5 shapes"),
                            patch("ten96", "Ours, 10 shapes"), patch("m96", "Ours, 26 letters")], ncol=4, fontsize=12)
    note(fig, "1000 fresh scenes per shape (5 and 10 shapes) and 300 per shape (27 shapes); bars paired with the "
              "original on the same scenes. 95% intervals.")
    return fig


def _sorted_shapes(model_key="m96", shapes=TRAINED27):
    return sorted(shapes, key=lambda s: rate("D051", D051[model_key], [s])[0])


def fig_final_per_shape_bars_horizontal(keys=("original", "m96"), size=TALL):
    order = _sorted_shapes()[::-1] + sorted(HELD, key=lambda s: -rate("D051", D051["m96"], [s])[0])
    fig, ax = plt.subplots(figsize=size)
    ys = np.arange(len(order))
    nk = len(keys)
    h = 0.8 / nk
    for j, k in enumerate(keys):
        v = [rate("D051", D051[k], [s])[0] for s in order]
        y = ys - 0.4 + h * (j + 0.5)
        ax.barh(y, v, h, color=col(k), edgecolor="white", lw=1.0, zorder=3, label=lab(k))
        if k == "m96" and nk <= 2:
            for yi, vi in zip(y, v):
                ax.text(vi + 0.6, yi, f"{vi:.0f}", va="center", fontsize=11, color=INK2)
    ax.set_yticks(ys, [disp(s) for s in order], fontsize=12)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.axhline(len(TRAINED27) - 0.5, color=INK2, lw=1)
    ax.set_xlim(0, 55)
    ax.text(54, len(TRAINED27) - 0.35, "never seen in training ↓", fontsize=12, color=INK2, va="top", ha="right")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"Success ({TEST_NOTE})")
    legend_top(ax, ncol=2 if nk <= 2 else 2, fontsize=12)
    return fig


def fig_final_per_shape_bars_vertical():
    order = _sorted_shapes()[::-1] + sorted(HELD, key=lambda s: -rate("D051", D051["m96"], [s])[0])
    fig, ax = plt.subplots(figsize=FULL)
    xs = np.arange(len(order))
    keys = ["original", "m96"]
    w = 0.42
    for j, k in enumerate(keys):
        v = [rate("D051", D051[k], [s])[0] for s in order]
        ax.bar(xs - w / 2 + w * j, v, w, color=col(k), edgecolor="white", lw=1.0, zorder=3, label=lab(k))
    ax.set_xticks(xs, [SHORT.get(s, s) for s in order], fontsize=13)
    ax.axvline(len(TRAINED27) - 0.5, color=INK2, lw=1)
    ax.text(len(TRAINED27) - 0.2, 57, "never seen\nin training", fontsize=12, color=INK2, va="top")
    ax.grid(False, axis="x")
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel(f"Success\n({TEST_NOTE})")
    legend_top(ax, ncol=2)
    note(fig, "nT = near-T; 3, 4, 7 = digits; + plus, # hash, → arrow.")
    return fig


def fig_final_dumbbell(vertical=False, key="m96"):
    order = _sorted_shapes(key)[::-1] + sorted(HELD, key=lambda s: -rate("D051", D051[key], [s])[0])
    a = np.array([rate("D051", "original", [s])[0] for s in order])
    b = np.array([rate("D051", D051[key], [s])[0] for s in order])
    if vertical:
        fig, ax = plt.subplots(figsize=FULL)
        xs = np.arange(len(order))
        ax.vlines(xs, a, b, color=tint(col(key), 0.5), lw=2.2, zorder=2)
        ax.scatter(xs, a, s=55, color=ORIG_LINE, zorder=3, label=lab("original"), edgecolor="white", lw=1.5)
        ax.scatter(xs, b, s=70, color=col(key), zorder=4, label=lab(key), edgecolor="white", lw=1.5)
        ax.set_xticks(xs, [SHORT.get(s, s) for s in order], fontsize=13)
        ax.axvline(len(TRAINED27) - 0.5, color=INK2, lw=1)
        ax.text(len(TRAINED27) - 0.2, 58, "never seen\nin training", fontsize=12, color=INK2, va="top")
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylim(-2, 60)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylabel(f"Success\n({TEST_NOTE})")
        ax.grid(False, axis="x")
        legend_top(ax, ncol=2)
        note(fig, "nT = near-T; 3, 4, 7 = digits; + plus, # hash, → arrow.")
        return fig
    fig, ax = plt.subplots(figsize=TALL)
    ys = np.arange(len(order))
    ax.hlines(ys, a, b, color=tint(col(key), 0.5), lw=2.2, zorder=2)
    ax.scatter(a, ys, s=55, color=ORIG_LINE, zorder=3, label=lab("original"), edgecolor="white", lw=1.5)
    ax.scatter(b, ys, s=70, color=col(key), zorder=4, label=lab(key), edgecolor="white", lw=1.5)
    ax.set_yticks(ys, [disp(s) for s in order], fontsize=12)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.axhline(len(TRAINED27) - 0.5, color=INK2, lw=1)
    ax.set_xlim(-1.5, 56)
    ax.text(55, len(TRAINED27) - 0.35, "never seen in training ↓", fontsize=12, color=INK2, va="top", ha="right")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"Success ({TEST_NOTE})")
    legend_top(ax, ncol=1, fontsize=12)
    return fig


SHAPE_GROUPS = ((OLD10, "o", "10 earlier shapes", 0.45, True), (NEW17, "s", "17 newer letters", 0.0, True),
                (HELD, "^", "6 never-seen shapes", 0.0, False))


def shape_scatter(ax, xfun, yfun, c, lim, legend_loc="lower right", labels=None, groups=SHAPE_GROUPS):
    """Per-shape scatter: colour = the y-axis model, marker = shape group (tinted circles: 10 earlier shapes,
    squares: 17 newer letters, hollow triangles: never-seen shapes); labels placed without overlaps."""
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1.2, zorder=1)
    names, xs, ys = [], [], []
    for shapes, marker, label, tnt, filled in groups:
        x = [xfun(s) for s in shapes]
        y = [yfun(s) for s in shapes]
        cc = tint(c, tnt)
        ax.scatter(x, y, s=70, marker=marker, color=cc if filled else "white", edgecolor=cc if filled else c,
                   lw=1.6 if not filled else 1.0, zorder=3, label=label)
        names += shapes; xs += x; ys += y
    ax.set_xlim(-1.5, lim)
    ax.set_ylim(-1.5, lim)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc=legend_loc, fontsize=12, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    return place_labels(ax, names, xs, ys, only=labels)


def fig_final_scatter(xkey="original", ykey="m96", size=(7, 6.2)):
    fig, ax = plt.subplots(figsize=size)
    lim = 62
    ax.text(lim - 4, lim - 1, "no change", fontsize=12, color=INK2, ha="right", va="top")
    pick = {"T", "T_narrow", "Y", "P", "C", "G", "I", "B", "digit7", "hash", "arrow", "W", "H", "O", "R"}
    shape_scatter(ax, lambda s: rate("D051", D051[xkey], [s])[0], lambda s: rate("D051", D051[ykey], [s])[0],
                  col(ykey), lim, labels=pick)
    ax.set_xlabel(f"{lab(xkey)}: success")
    ax.set_ylabel(f"{lab(ykey)}: success")
    note(fig, f"Each point is one shape; {TEST_NOTE}. Only some shapes are labelled; every letter except T, "
              "near-T and Y starts at 0-2% for the original.")
    return fig


def forest(ax, rows, colors, offsets=None, labels=None, ms=8):
    """rows: list of (row label, [dict per series or None]); draws difference points with 95% intervals."""
    n = len(rows)
    ys = np.arange(n)
    ns = len(colors)
    offsets = offsets if offsets is not None else np.linspace(-0.18, 0.18, ns) if ns > 1 else [0]
    for j, c in enumerate(colors):
        for i, (_, ser) in enumerate(rows):
            d = ser[j]
            if d is None:
                continue
            y = ys[i] + offsets[j]
            ax.plot([d["ci95_low"], d["ci95_high"]], [y, y], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
            ax.plot(d["difference"], y, "o", color=c, ms=ms, mec="white", mew=1.4, zorder=4,
                    label=labels[j] if labels and i == 0 else None)
    ax.axvline(0, color=INK2, lw=1.1, zorder=2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(n - 0.5, -0.6)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.xaxis.set_major_formatter(SIGNED)
    return ys


def fig_final_forest_groups():
    groups = ["10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"]
    keys = ["ten96", "m96", "m192"]
    rows = [(GROUP_LABEL[g], [paired_row("D051", "original", D051[k], g) for k in keys]) for g in groups]
    fig, ax = plt.subplots(figsize=FULL)
    forest(ax, rows, [col(k) for k in keys], labels=[lab(k) for k in keys], offsets=[-0.22, 0, 0.22])
    ax.set_xlabel("Change in success vs. the original (points), same scenes, 95% interval")
    ax.set_xlim(-2, 37)
    for i, g in enumerate(groups):
        d = rows[i][1][1]
        ax.text(d["ci95_high"] + 0.6, i, f"{fmt_signed(d['difference']).replace('$-$', '−')}", va="center",
                fontsize=12, color=M96, fontweight="bold")
    legend_top(ax, ncol=3, fontsize=12, markerscale=1.4)
    note(fig, f"Paired by scene against the original policy; {TEST_NOTE}.")
    return fig


def fig_final_forest_shapes():
    order = _sorted_shapes()[::-1] + sorted(HELD, key=lambda s: -rate("D051", D051["m96"], [s])[0])
    keys = ["m96", "m192"]
    groups = ["27 trained shapes", "6 held-out shapes"]
    rows = [(GROUP_LABEL[g], [paired_row("D051", "original", D051[k], g) for k in keys]) for g in groups]
    rows += [(disp(s), [paired_row("D051", "original", D051[k], s) for k in keys]) for s in order]
    fig, ax = plt.subplots(figsize=(7.5, 11))
    forest(ax, rows, [col(k) for k in keys], labels=[lab(k) for k in keys], offsets=[-0.17, 0.17], ms=6)
    ax.axhline(1.5, color=INK2, lw=1)
    ax.axhline(1.5 + len(TRAINED27), color=INK2, lw=1)
    ax.tick_params(axis="y", labelsize=12)
    for t in ax.get_yticklabels()[:2]:
        t.set_fontweight("bold")
    ax.set_xlabel("Change in success vs. original (points)\nsame 300 scenes per shape, 95% interval")
    ax.legend(loc="lower right", fontsize=12, frameon=True, facecolor="white", edgecolor="none")
    return fig


def _getter(test, models, pool):
    def g(k, s):
        r = rate(test, models[k], pool if s is None else [s])
        return r[0], r[3], r[4]
    return g


def fig_final_10shapes_bars():
    keys = ["original", "ten96", "ten192"]
    fig, ax, axp = bars_with_pooled(OLD10, keys, _getter("D049/D050", D049, OLD10), "All 10\nshapes")
    ax.set_xticks(range(len(OLD10)), [disp(s) for s in OLD10])
    ax.set_ylim(0, 65)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success\n(final test, 1000 fresh scenes per shape)")
    legend_top(ax, handles=[patch(k) for k in keys], ncol=3, fontsize=12)
    return fig


def fig_final_10shapes_forest():
    keys = ["ten96", "ten192"]
    rows = [("All 10 shapes", [paired_row("D049/D050", "original", D049[k], "10 earlier shapes") for k in keys])]
    rows += [(disp(s), [paired_row("D049/D050", "original", D049[k], s) for k in keys]) for s in OLD10]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    forest(ax, rows, [col(k) for k in keys], labels=[lab(k) for k in keys])
    ax.axhline(0.5, color=INK2, lw=1)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("Change in success vs. original (points)\n1000 paired scenes per shape, 95% interval")
    legend_top(ax, ncol=1, fontsize=12)
    return fig


def fig_final_5shapes_bars():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, test, model, title in ((axes[0], "D046", "5-shape round 4", "After 4 correction rounds"),
                                   (axes[1], "D047", "5-shape round 7", "After 7 correction rounds")):
        groups = [disp(s) for s in FIVE_SHAPES] + ["All 5"]
        vals, los, his = [], [], []
        for m in ("original", model):
            v, lo, hi = [], [], []
            for s in FIVE_SHAPES + [None]:
                r, _, _, l_, h_ = rate(test, m, FIVE_SHAPES if s is None else [s])
                v.append(r); lo.append(l_); hi.append(h_)
            vals.append(v); los.append(lo); his.append(hi)
        grouped_bars(ax, groups, ["original", "five"], vals, los, his, labels=True, fmt="{:.0f}", size=11)
        ax.axvline(4.5, color=AXIS, lw=1)
        ax.set_title(title, loc="left", fontsize=15)
    axes[0].set_ylim(0, 52)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Success\n(1000 fresh scenes per shape)")
    legend_top(axes[0], handles=[patch("original"), patch("five")], ncol=2, y=1.1)
    note(fig, "Each panel is paired with the original on its own fresh scene pool (two different pools).")
    return fig


def fig_final_blends():
    keys = [("original", ORIG, "Original"), ("blend r4e60 s0.7", tint(BLEND, 0.25), "70% weight blend"),
            ("blend encoder 0.7", FIVE, "70% encoder-only blend")]
    groups = [disp(s) for s in FIVE_SHAPES]
    vals, los, his = [], [], []
    for m, _, _ in keys:
        v, lo, hi = [], [], []
        for s in FIVE_SHAPES:
            r, _, _, l_, h_ = rate("D040", m, [s])
            v.append(r); lo.append(l_); hi.append(h_)
        vals.append(v); los.append(lo); his.append(hi)
    fig, ax = plt.subplots(figsize=FULL)
    grouped_bars(ax, groups, [k[0] for k in keys], vals, los, his, labels=True, fmt="{:.1f}", size=11,
                 colors=[k[1] for k in keys])
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success\n(1000 fresh scenes per shape)")
    legend_top(ax, handles=[patch(c, l) for _, c, l in keys], ncol=3, fontsize=12)
    note(fig, "Early attempt: blending a planner-copying student's weights into the original. Small gains on Y and D, "
              "a loss on T.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# shapes_*: per-shape comparisons
# ----------------------------------------------------------------------------------------------------------------------


def fig_shapes_96_vs_192():
    fig, ax = plt.subplots(figsize=(7, 6.2))
    lim = 62
    skipped = shape_scatter(ax, lambda s: rate("D051", D051["m96"], [s])[0],
                            lambda s: rate("D051", D051["m192"], [s])[0], M192, lim)
    ax.set_xlabel("26 letters, 96 px camera: success")
    ax.set_ylabel("26 letters, 192 px camera: success")
    note(fig, f"Each point is one shape; {TEST_NOTE}. Above the line: the sharper camera helps."
         + (" Unlabelled: " + ", ".join(disp(s) for s in skipped) + "." if skipped else ""))
    return fig


def fig_shapes_10_vs_26():
    fig, ax = plt.subplots(figsize=(7, 6.2))
    lim = 62
    groups = ((OLD10, "o", "10 earlier shapes (both trained on them)", 0.45, True),
              (NEW17, "s", "17 newer letters (only the 26-letter policy)", 0.0, True),
              (HELD, "^", "6 never-seen shapes", 0.0, False))
    skipped = shape_scatter(ax, lambda s: rate("D051", D051["ten96"], [s])[0],
                            lambda s: rate("D051", D051["m96"], [s])[0], M96, lim, legend_loc="lower right",
                            groups=groups)
    ax.set_xlabel("Ours, 10 shapes: success")
    ax.set_ylabel("Ours, 26 letters: success")
    note(fig, f"Each point is one shape; {TEST_NOTE}. Adding 17 letters did not cost the earlier ten."
         + (" Unlabelled: " + ", ".join(disp(s) for s in skipped) + "." if skipped else ""))
    return fig


def fig_shapes_validation_vs_test():
    steps, M = val_matrix("27 shapes", "96 px line", TRAINED27)
    v = M[:, list(steps).index(45)]       # validation of the round-44 student (step 45)
    t = np.array([rate("D051", D051["m96"], [s])[0] for s in TRAINED27])
    fig, ax = plt.subplots(figsize=(7, 6.2))
    lim = 80
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1.2)
    ax.scatter(v, t, s=70, color=M96, edgecolor="white", lw=1.4, zorder=3)
    r = np.corrcoef(v, t)[0, 1]
    ax.text(2, lim - 3, f"correlation {r:.2f} across 27 shapes", fontsize=12, color=INK2, va="top")
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    place_labels(ax, TRAINED27, v, t)
    ax.set_xlabel("Validation success at round 44\n(50 fixed scenes per shape)")
    ax.set_ylabel("Final-test success\n(300 fresh scenes per shape)")
    note(fig, "26-letter 96 px student picked for the final test. 50 scenes give about ±13 points of chance spread.")
    return fig


# TYPED-IN: LEARNING_FINDINGS.md section 7, "26-letter preparation (D051)": teacher (world model wm2 v3) successes of
# 50 validation scenes per letter. Earlier ten-letter shapes other than T, D and L were not re-checked.
TEACHER26 = TYPED["teacher26_successes_of_50"]["values"]
# TYPED-IN: LEARNING_FINDINGS.md section 5 table "Final planner (50)": first sampling-planner teacher, 50 validation
# scenes per letter; and the original policy on the 1000-scene final-test pool (same section, last column).
TEACHER5 = TYPED["teacher5_successes_of_50"]["values"]
ORIG5_FINALPOOL = TYPED["original_5shapes_final_pool_percent"]["values"]


def fig_shapes_teacher_vs_student():
    shapes = list(TEACHER26)
    x = np.array([TEACHER26[s] * 2 for s in shapes], dtype=float)
    y = np.array([rate("D051", D051["m96"], [s])[0] for s in shapes])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1.2)
    ax.text(70, 74, "student = teacher", ha="right", va="bottom", fontsize=12, color=INK2)
    ax.scatter(x, y, s=75, color=M96, edgecolor="white", lw=1.4, zorder=3)
    ax.text(3, 97, f"the student keeps {np.mean(y) / np.mean(x) * 100:.0f}% of the teacher's success\n"
                   f"(mean {np.mean(y):.0f}% vs {np.mean(x):.0f}%, these 20 letters)", fontsize=12, color=INK2, va="top")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    place_labels(ax, shapes, x, y)
    ax.set_xlabel("Planning teacher, reads exact state\n(50 validation scenes)")
    ax.set_ylabel("Camera student, 26 letters\n(300 fresh test scenes)")
    note(fig, "Teacher numbers typed in from LEARNING_FINDINGS.md section 7; different scene pools.")
    return fig


def fig_shapes_new_letters_dotplot():
    order = sorted(NEW17, key=lambda s: rate("D051", D051["m96"], [s])[0])
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ys = np.arange(len(order))
    for key, mk, sz in (("original", "o", 55), ("ten96", "D", 60), ("m96", "o", 80)):
        v = [rate("D051", D051[key], [s])[0] for s in order]
        if key == "m96":
            v0 = [rate("D051", D051["ten96"], [s])[0] for s in order]
            ax.hlines(ys, v0, v, color=tint(M96, 0.55), lw=2, zorder=2)
        ax.scatter(v, ys, s=sz, marker=mk, color=col(key) if key != "original" else ORIG_LINE, edgecolor="white",
                   lw=1.4, zorder=3, label={"original": "Original", "ten96": "Ours, 10 shapes (never saw it)",
                                            "m96": "Ours, 26 letters (trained on it)"}[key])
    ax.set_yticks(ys, order, fontsize=13)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"Success on the 17 newer letters\n({TEST_NOTE})")
    legend_top(ax, ncol=1, fontsize=12)
    return fig


def fig_shapes_table_heatmap():
    keys = ["original", "ten96", "m96", "m192"]
    order = OLD10 + sorted(NEW17) + HELD
    M = np.array([[rate("D051", D051[k], [s])[0] for k in keys] for s in order])
    fig, ax = plt.subplots(figsize=(6.5, 11))
    im = ax.imshow(M, aspect="auto", cmap=SEQ, vmin=0, vmax=70)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > 38 else INK)
    ax.set_yticks(range(len(order)), [disp(s) for s in order], fontsize=12)
    ax.set_xticks(range(len(keys)), ["Original", "Ours,\n10 shapes", "Ours,\n26 letters", "Ours, 26,\n192 px"],
                  fontsize=12)
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for yline in (len(OLD10) - 0.5, len(TRAINED27) - 0.5):
        ax.axhline(yline, color="white", lw=4)
    ax.text(-1.25, len(OLD10) / 2 - 0.5, "10 earlier", rotation=90, ha="center", va="center", fontsize=12, color=INK2)
    ax.text(-1.25, len(OLD10) + 8, "17 newer", rotation=90, ha="center", va="center", fontsize=12, color=INK2)
    ax.text(-1.25, len(TRAINED27) + 2.5, "never seen", rotation=90, ha="center", va="center", fontsize=12, color=INK2)
    note(fig, f"Success %, {TEST_NOTE}.", y=0.06)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# heldout_*: never-seen shapes
# ----------------------------------------------------------------------------------------------------------------------
def fig_heldout_scaling():
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"wspace": 0.32})
    xs = [1, 10, 27]
    for ax, metric in ((axes[0], "success"), (axes[1], "coverage")):
        pts = []
        for key in ("original", "ten96", "m96"):
            if metric == "success":
                r, _, _, lo, hi = rate("D051", D051[key], HELD)
            else:
                r, lo, hi = coverage("D051", D051[key], HELD), None, None
            pts.append((r, lo, hi))
        y = [p[0] for p in pts]
        ax.plot(xs, y, color=M96, lw=2.2, zorder=2)
        for x, key, p in zip(xs, ("original", "ten96", "m96"), pts):
            ax.scatter([x], [p[0]], s=110, color=col(key) if key != "original" else ORIG_LINE, edgecolor="white",
                       lw=1.5, zorder=4)
            if p[1] is not None:
                errbars(ax, [x], [p[0]], [p[1]], [p[2]])
        if metric == "success":
            r, _, _, lo, hi = rate("D051", D051["m192"], HELD)
        else:
            r, lo, hi = coverage("D051", D051["m192"], HELD), None, None
        ax.scatter([27.9], [r], s=110, color=M192, edgecolor="white", lw=1.5, zorder=4, marker="D")
        if lo is not None:
            errbars(ax, [27.9], [r], [lo], [hi])
        for x, v in zip(xs + [27.9], y + [r]):
            ax.text(x + (-1.2 if x == 27 else 0.8), v + (0.4 if metric == "success" else 0.8), f"{v:.1f}",
                    fontsize=12, ha="right" if x == 27 else "left", va="bottom", color=INK)
        ax.set_xticks(xs, ["1\n(original:\nT only)", "10", "27"])
        ax.set_xlim(-1, 31)
        ax.grid(False, axis="x")
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("Shapes in training")
    axes[0].set_ylim(0, 8)
    axes[0].set_ylabel("Success on 6 never-seen shapes")
    axes[1].set_ylim(40, 62)
    axes[1].yaxis.set_major_locator(MultipleLocator(5))
    axes[1].set_ylabel("Mean final goal coverage")
    handles = [Line2D([], [], marker="o", color=ORIG_LINE, ls="none", ms=10, label="Original"),
               Line2D([], [], marker="o", color=TEN, ls="none", ms=10, label="Ours, 10 shapes"),
               Line2D([], [], marker="o", color=M96, ls="none", ms=10, label="Ours, 26 letters"),
               Line2D([], [], marker="D", color=M192, ls="none", ms=9, label="Ours, 26 letters, 192 px")]
    axes[0].legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=12)
    note(fig, "digits 3, 4, 7, plus, hash, arrow; 300 fresh scenes per shape; 95% intervals.")
    return fig


def fig_heldout_per_shape_bars():
    keys = ["original", "ten96", "m96", "m192"]
    order = sorted(HELD, key=lambda s: -rate("D051", D051["m192"], [s])[0])
    fig, ax, axp = bars_with_pooled(order, keys, _getter("D051", D051, HELD), "All 6", ratio=1.8)
    ax.set_xticks(range(len(order)), [disp(s) for s in order])
    ax.set_ylim(0, 30)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel(f"Success on never-seen shapes\n({TEST_NOTE})")
    legend_top(ax, handles=[patch(k) for k in keys], ncol=2, fontsize=12)
    return fig


def fig_heldout_scaling_per_shape():
    fig, axes = plt.subplots(2, 3, figsize=FULL, sharex=True)
    xs = [1, 10, 27]
    for ax, s in zip(axes.flat, sorted(HELD, key=lambda s: -rate("D051", D051["m192"], [s])[0])):
        pts = [rate("D051", D051[k], [s]) for k in ("original", "ten96", "m96")]
        ax.plot(xs, [p[0] for p in pts], color=M96, lw=2, zorder=2)
        for x, k, p in zip(xs, ("original", "ten96", "m96"), pts):
            ax.scatter([x], [p[0]], s=60, color=col(k) if k != "original" else ORIG_LINE, zorder=3, edgecolor="white")
            errbars(ax, [x], [p[0]], [p[3]], [p[4]], lw=1)
        p = rate("D051", D051["m192"], [s])
        ax.scatter([28.5], [p[0]], s=60, color=M192, marker="D", zorder=3, edgecolor="white")
        errbars(ax, [28.5], [p[0]], [p[3]], [p[4]], lw=1)
        ax.set_title(disp(s), loc="left", fontsize=15)
        top = max(6, max(p[4] for p in pts + [p]) * 1.1)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xticks(xs)
        ax.set_xlim(-2, 31)
        ax.grid(False, axis="x")
    fig.supxlabel("Shapes in training (diamond: 192 px camera)", fontsize=16)
    fig.supylabel("Success", fontsize=16)
    fig.tight_layout()
    return fig


def fig_heldout_unseen_capitals():
    keys = ["original", "ten96", "ten192"]
    order = sorted(NEW17, key=lambda s: -rate("D052", D049["ten192"], [s])[0])
    g = _getter("D052", D049, NEW17)
    fig, (ax, axp) = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"width_ratios": [17, 3.2],
                                                                                 "wspace": 0.03})
    vals = [[g(k, s)[0] for s in order] for k in keys]
    grouped_bars(ax, order, keys, vals, labels=False, width=0.84)      # per letter: no interval (200 scenes each)
    pooled = [g(k, None) for k in keys]
    grouped_bars(axp, ["All 17"], keys, [[p[0]] for p in pooled], [[p[1]] for p in pooled], [[p[2]] for p in pooled],
                 labels=True, width=0.86, size=12)
    axp.spines["left"].set_visible(False)
    axp.tick_params(axis="y", left=False)
    axp.set_xlim(-0.5, 0.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 45)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success on letters never trained on\n(200 fresh scenes per letter)")
    legend_top(ax, handles=[patch(k) for k in keys], ncol=3, fontsize=12)
    note(fig, "10-shape students (trained on T, near-T, Y, D, L, E, F, J, P, U) on the other 17 capitals.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# coverage_*: how close episodes get
# ----------------------------------------------------------------------------------------------------------------------
KEYS4 = ["original", "ten96", "m96", "m192"]


def fig_coverage_hist(which="final", shapes=TRAINED27, keys=("original", "m96")):
    fig, ax = plt.subplots(figsize=FULL)
    bins = np.linspace(0, 1, 41)
    for k in keys:
        v = eps("D051", D051[k], shapes)[which]
        w = np.ones_like(v) * 100 / len(v)
        c = col(k) if k != "original" else ORIG_LINE
        ax.hist(v * 100, bins=bins * 100, weights=w, histtype="stepfilled", color=c, alpha=0.18, lw=0)
        ax.hist(v * 100, bins=bins * 100, weights=w, histtype="step", color=c, lw=2.2, label=lab(k))
    ax.axvline(95, color=INK, lw=1.2, ls=(0, (3, 2)))
    ax.text(94, ax.get_ylim()[1] * 0.97, "success\nthreshold 95%", ha="right", va="top", fontsize=12, color=INK2)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel(("Final" if which == "final" else "Peak") + " goal coverage of an episode")
    ax.set_ylabel("Share of episodes")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left")
    note(fig, f"27 trained shapes, {TEST_NOTE} ({len(eps('D051', 'original', shapes)['final'])} episodes per model).")
    return fig


def fig_coverage_cdf():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.16})
    for ax, which in ((axes[0], "final"), (axes[1], "peak")):
        for k in KEYS4:
            v = np.sort(eps("D051", D051[k], TRAINED27)[which]) * 100
            y = 100 * np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, color=col(k) if k != "original" else ORIG_LINE, lw=2.2, label=lab(k))
        ax.axvline(95, color=INK, lw=1.1, ls=(0, (3, 2)))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.grid(True, axis="x")
        ax.set_xlabel(("Final" if which == "final" else "Peak") + " goal coverage")
    axes[0].set_ylabel("Episodes at or below this coverage")
    legend_top(axes[0], ncol=2, fontsize=12, y=1.03)
    note(fig, f"27 trained shapes, {TEST_NOTE}. Further right = closer to the goal; dashed: 95% success threshold.")
    return fig


def fig_coverage_violin():
    groups = ["10 earlier shapes", "17 other letters", "6 held-out shapes"]
    keys = KEYS4
    fig, ax = plt.subplots(figsize=FULL)
    w = 0.8 / len(keys)
    for gi, g in enumerate(groups):
        for j, k in enumerate(keys):
            v = eps("D051", D051[k], GROUP_SHAPES[g])["final"] * 100
            x = gi - 0.4 + w * (j + 0.5)
            parts = ax.violinplot([v], positions=[x], widths=w * 0.92, showextrema=False, bw_method=0.15)
            for b in parts["bodies"]:
                b.set_facecolor(col(k))
                b.set_edgecolor("white")
                b.set_alpha(0.9)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(x, q1, q3, color=INK, lw=2.5, zorder=3)
            ax.scatter([x], [med], color="white", s=18, zorder=4, edgecolor=INK, lw=0.8)
    ax.axhline(95, color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.set_xticks(range(len(groups)), [GROUP_LABEL[g] for g in groups])
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("Final goal coverage per episode")
    legend_top(ax, handles=[patch(k) for k in keys] + [Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2)),
                                                              label="success threshold (95%)")], ncol=3, fontsize=12)
    note(fig, f"{TEST_NOTE.capitalize()}. Black bar: middle half of episodes; dot: median.")
    return fig


def _outcome_shares(test, model, shapes):
    d = eps(test, model, shapes)
    n = len(d["success"])
    solved = d["success"] == 1
    near = (~solved) & (d["peak"] >= 0.9)
    close = (~solved) & (d["peak"] >= 0.8) & (d["peak"] < 0.9)
    return 100 * solved.sum() / n, 100 * near.sum() / n, 100 * close.sum() / n, 100 * (n - solved.sum() - near.sum()
                                                                                         - close.sum()) / n


def fig_coverage_near_misses():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.16})
    keys = KEYS4[::-1]
    for ax, shapes, title in ((axes[0], TRAINED27, "27 trained shapes"), (axes[1], HELD, "6 never-seen shapes")):
        for i, k in enumerate(keys):
            parts = _outcome_shares("D051", D051[k], shapes)
            c = col(k) if k != "original" else ORIG_LINE
            colors = [c, tint(c, 0.5), tint(c, 0.78), "#efeee9"]
            left = 0
            for j, (v, cc) in enumerate(zip(parts, colors)):
                ax.barh(i, v, left=left, color=cc, edgecolor="white", lw=1.5, height=0.7, zorder=3)
                if v >= 6:
                    ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=12,
                            color="white" if j == 0 and k != "original" else INK)
                left += v
        ax.set_title(title, loc="left", fontsize=15)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PCT)
        ax.grid(False)
        ax.set_xlabel("Share of episodes")
    axes[0].set_yticks(range(len(keys)), [lab(k).replace(", 192 px camera", ",\n192 px camera") for k in keys],
                       fontsize=12)
    shades = [Patch(facecolor=M96, label="solved (>95% coverage)"),
              Patch(facecolor=tint(M96, 0.5), label="near miss: peak 90-95%"),
              Patch(facecolor=tint(M96, 0.78), label="peak 80-90%"),
              Patch(facecolor="#efeee9", label="peak below 80%")]
    legend_top(axes[0], handles=shades, ncol=4, fontsize=12, y=1.1)
    note(fig, f"{TEST_NOTE.capitalize()}. 'Peak' is the best coverage reached at any moment of the episode; "
              "shades of each policy's colour show the same four outcomes.")
    return fig


def fig_coverage_success_vs_coverage():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in ("original", "ten96", "m96"):
        x = [100 * float(BYSHAPE[("D051", D051[k])][s]["mean_final_coverage"]) for s in ALL33]
        y = [rate("D051", D051[k], [s])[0] for s in ALL33]
        ax.scatter(x, y, s=55, color=col(k) if k != "original" else ORIG_LINE, edgecolor="white", lw=1.2, zorder=3,
                   label=lab(k))
    ax.set_xlim(10, 100)
    ax.set_ylim(-2, 62)
    ax.grid(True, axis="both")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Mean final goal coverage of a shape")
    ax.set_ylabel("Success on that shape")
    ax.legend(loc="upper left", fontsize=12)
    note(fig, f"Each point is one shape and model; 33 shapes, {TEST_NOTE}. Success appears only above ~70% coverage.")
    return fig


def fig_coverage_controls():
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"wspace": 0.3})
    t = np.arange(0, 301)
    for k in KEYS4:
        d = eps("D051", D051[k], TRAINED27)
        c = col(k) if k != "original" else ORIG_LINE
        solved = d["controls"][d["success"] == 1]
        axes[0].plot(t, 100 * np.array([(solved <= x).sum() for x in t]) / len(d["success"]), color=c, lw=2.2,
                     label=lab(k))
        s = np.sort(solved)
        axes[1].plot(s, 100 * np.arange(1, len(s) + 1) / len(s), color=c, lw=2.2, label=lab(k))
    axes[0].set_ylabel("Episodes solved by then")
    axes[0].set_ylim(0, 36)
    axes[1].set_ylabel("Share of the solved episodes")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlim(0, 300)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter(PCT)
        ax.grid(True, axis="x")
        ax.set_xlabel("Controls used (limit 300)")
    legend_top(axes[0], ncol=2, fontsize=12)
    note(fig, f"27 trained shapes, {TEST_NOTE}. Left: all episodes; right: only solved ones (how long success takes).")
    return fig


def fig_coverage_steps_executed():
    keys = [("execute 8 (trained)", BLUE_RAMP[2], "8 (as trained)"), ("execute 12", BLUE_RAMP[1], "12"),
            ("execute 15", BLUE_RAMP[0], "15")]
    fig, ax = plt.subplots(figsize=HALF)
    for m, c, l in keys:
        v = np.sort(eps("D056", m, TRAINED27)["final"]) * 100
        ax.plot(v, 100 * np.arange(1, len(v) + 1) / len(v), color=c, lw=2.4, label=l)
    ax.axvline(95, color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.set_xlabel("Final goal coverage")
    ax.set_ylabel("Episodes at or below")
    ax.legend(title="Steps executed per plan", loc="upper left", fontsize=12, title_fontsize=12)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# analysis_*
# ----------------------------------------------------------------------------------------------------------------------
def fig_analysis_steps_executed():
    groups = ["10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"]
    models = ["execute 8 (trained)", "execute 12", "execute 15"]
    vals, los, his = [], [], []
    for m in models:
        v, lo, hi = [], [], []
        for g in groups:
            r, _, _, l_, h_ = rate("D056", m, GROUP_SHAPES[g])
            v.append(r); lo.append(l_); hi.append(h_)
        vals.append(v); los.append(lo); his.append(hi)
    fig, ax = plt.subplots(figsize=FULL)
    grouped_bars(ax, [GROUP_LABEL[g] for g in groups], models, vals, los, his, colors=BLUE_RAMP[::-1])
    for gi, g in enumerate(groups):
        for j, m in enumerate(models[1:], start=1):
            d = paired_row("D056", "execute 8 (trained)", m, g)
            if d["model_rate"] < 6:
                continue
            ax.text(gi - 0.4 + 0.8 / 3 * (j + 0.5), 1.0, f"{d['difference']:+.1f}".replace("-", "−"), ha="center",
                    va="bottom", fontsize=11, color="white" if j == 1 else INK, zorder=5)
    ax.set_ylim(0, 42)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success\n(200 fresh scenes per shape)")
    legend_top(ax, handles=[patch(BLUE_RAMP[2], "execute 8 of 16 planned steps (as trained)"),
                            patch(BLUE_RAMP[1], "execute 12"), patch(BLUE_RAMP[0], "execute 15")], ncol=3, fontsize=12)
    note(fig, "26-letter 96 px student (round 50), same scenes for every setting. Numbers at the bar foot: change vs 8 "
              "(points).")
    return fig


# TYPED-IN: DECISIONS.md D048 "Teacher outcome": ten-letter teacher, 50 validation scenes per letter, successes of 500.
TEACHER_COMMIT = TYPED["teacher_commit_successes_of_500"]["values"]


def fig_analysis_commitment():
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    x = np.arange(3)
    v = [n / 5 for _, n in TEACHER_COMMIT]
    lo, hi = zip(*[wilson(n, 500) for _, n in TEACHER_COMMIT])
    bars = ax.bar(x, v, 0.6, color=[TEACHER, tint(TEACHER, 0.3), tint(TEACHER, 0.5)], edgecolor="white", zorder=3)
    errbars(ax, x, v, lo, hi)
    for xi, vi, h in zip(x, v, hi):
        ax.text(xi, h + 1, f"{vi:.0f}", ha="center", fontsize=13)
    ax.set_xticks(x, [t for t, _ in TEACHER_COMMIT], fontsize=12)
    ax.set_title("Teacher (exact state), 10 shapes", loc="left", fontsize=15)
    ax.set_ylabel("Validation success")
    ax.set_ylim(0, 100)
    ax = axes[1]
    models = [("execute 8 (trained)", "8\n(as trained)"), ("execute 12", "12"), ("execute 15", "15")]
    v, lo, hi = [], [], []
    for m, _ in models:
        r, _, _, l_, h_ = rate("D056", m, TRAINED27)
        v.append(r); lo.append(l_); hi.append(h_)
    ax.bar(x, v, 0.6, color=BLUE_RAMP[::-1], edgecolor="white", zorder=3)
    errbars(ax, x, v, lo, hi)
    for xi, vi, h in zip(x, v, hi):
        ax.text(xi, h + 0.6, f"{vi:.1f}", ha="center", fontsize=13)
    ax.set_xticks(x, [t for _, t in models], fontsize=12)
    ax.set_title("Camera student, 27 shapes", loc="left", fontsize=15)
    ax.set_xlabel("Steps executed per 16-step plan")
    ax.set_ylabel("Fresh-scene success")
    ax.set_ylim(0, 42)
    for ax in axes:
        ax.yaxis.set_major_formatter(PCT)
        ax.grid(False, axis="x")
    note(fig, "Left: typed in from DECISIONS.md D048 (390, 317, 303 of 500). Right: round-50 student, 200 fresh scenes "
              "per shape.")
    return fig


def fig_analysis_round50_vs_44(per_shape=False):
    base, model = "26-letter 96 px round 44", "26-letter 96 px round 50"
    groups = ["10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"]
    rows = [(GROUP_LABEL[g], [paired_row("D054", base, model, g)]) for g in groups]
    if per_shape:
        order = sorted(TRAINED27, key=lambda s: -paired_row("D054", base, model, s)["difference"]) + HELD
        rows += [(disp(s), [paired_row("D054", base, model, s)]) for s in order]
    fig, ax = plt.subplots(figsize=(7.5, 11) if per_shape else FULL)
    forest(ax, rows, [M96], ms=7 if per_shape else 10)
    if per_shape:
        ax.axhline(3.5, color=INK2, lw=1)
        ax.axhline(3.5 + 27, color=INK2, lw=1)
        for t in ax.get_yticklabels()[:4]:
            t.set_fontweight("bold")
        ax.tick_params(axis="y", labelsize=12)
    else:
        for i, (_, ser) in enumerate(rows):
            d = ser[0]
            ax.text(d["ci95_high"] + 0.3, i, f"{d['base_rate']:.1f}% → {d['model_rate']:.1f}%   "
                                              f"({d['difference']:+.1f})".replace("-", "−"), va="center", fontsize=12)
        ax.set_xlim(-6, 12)
    ax.set_xlabel("Round 50 minus round 44 (points), same scenes, 95% interval")
    note(fig, "26-letter 96 px student after six more correction rounds; 200 fresh scenes per shape.")
    return fig


def fig_analysis_t_tradeoff(diff=False):
    entries = [("5 shapes, round 4", "D046", "5-shape round 4", "five"),
               ("5 shapes, round 7", "D047", "5-shape round 7", "five"),
               ("10 shapes", "D049/D050", "10-shape 96 px", "ten96"),
               ("10 shapes, 192 px", "D049/D050", "10-shape 192 px", "ten192"),
               ("26 letters", "D051", "26-letter 96 px (round 44)", "m96"),
               ("26 letters, 192 px", "D051", "26-letter 192 px (round 2)", "m192")]
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.16})
    ys = np.arange(len(entries))
    for ax, shape in ((axes[0], "T"), (axes[1], "T_narrow")):
        for i, (name, test, model, key) in enumerate(entries):
            if diff:
                d = paired_row(test, "original", model, shape)
                ax.plot([d["ci95_low"], d["ci95_high"]], [i, i], color=col(key), lw=2.4)
                ax.plot(d["difference"], i, "o", color=col(key), ms=10, mec="white", mew=1.4, zorder=4)
            else:
                ro = rate(test, "original", [shape])
                rm = rate(test, model, [shape])
                ax.plot([ro[0], rm[0]], [i - 0.14, i + 0.14], color=AXIS, lw=1.3, zorder=2)
                ax.errorbar(ro[0], i - 0.14, xerr=[[ro[0] - ro[3]], [ro[4] - ro[0]]], fmt="o", color=ORIG_LINE, ms=8,
                            mec="white", elinewidth=1.4, capsize=0, zorder=3)
                ax.errorbar(rm[0], i + 0.14, xerr=[[rm[0] - rm[3]], [rm[4] - rm[0]]], fmt="o", color=col(key),
                            ms=10, mec="white", elinewidth=1.8, capsize=0, zorder=4)
        ax.set_title(disp(shape), loc="left", fontsize=16)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        if diff:
            ax.axvline(0, color=INK2, lw=1.1)
            ax.xaxis.set_major_formatter(SIGNED)
            ax.set_xlim(-20, 10)
            ax.set_xticks([-20, -10, 0, 10])
        else:
            ax.xaxis.set_major_formatter(PCT)
            ax.set_xlim(20, 55)
    axes[0].set_yticks(ys, [e[0] for e in entries])
    axes[0].set_ylim(len(entries) - 0.5, -0.6)
    if diff:
        fig.supxlabel("Change vs. the original on the same scenes (points, 95% interval)", fontsize=16, y=0.0)
    else:
        fig.supxlabel("Fresh-scene success (95% interval)", fontsize=16, y=0.0)
        legend_top(axes[0], handles=[Line2D([], [], marker="o", color=ORIG_LINE, ls="none", ms=8,
                                            label="original, same scenes"),
                                     Line2D([], [], marker="o", color=INK2, ls="none", ms=10,
                                            label="our student (colour = policy)")], ncol=2, fontsize=12, y=1.08)
    note(fig, "Each test has its own fresh scene pool: 1000 scenes per shape (5 and 10 shapes), 300 (26 letters).")
    return fig


def fig_analysis_192_vs_96(per_shape=False):
    if not per_shape:
        rows = [("10-shape models:\nall 10 shapes", [paired_eps("D049/D050", "10-shape 96 px", "10-shape 192 px", OLD10),
                                                     None])]
        for g in ("10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"):
            rows.append((f"26-letter models:\n{GROUP_LABEL[g].lower()}" if g != "27 trained shapes" else
                         "26-letter models:\nall 27 trained shapes",
                         [None, paired_eps("D051", D051["m96"], D051["m192"], GROUP_SHAPES[g])]))
        fig, ax = plt.subplots(figsize=FULL)
        forest(ax, rows, [TEN192, M192], offsets=[0, 0], ms=10)
        ax.tick_params(axis="y", labelsize=12)
        for i, (_, ser) in enumerate(rows):
            d = ser[0] or ser[1]
            ax.text(d["ci95_high"] + 0.3, i, f"{d['difference']:+.1f}".replace("-", "−"), va="center", fontsize=12,
                    fontweight="bold" if i == 0 else None)
        ax.set_xlim(-4, 8)
        ax.set_xlabel("192 px camera minus 96 px camera (points), same scenes, 95% interval")
        note(fig, "10 shapes: 1000 paired scenes per shape. 26 letters: 300 paired scenes per shape "
                  "(192 px student picked at round 2).")
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35})
    rows = [(disp(s), [paired_eps("D049/D050", "10-shape 96 px", "10-shape 192 px", [s])]) for s in OLD10]
    forest(axes[0], rows, [TEN192], ms=8)
    axes[0].set_title("10-shape models", loc="left", fontsize=15)
    order = OLD10 + sorted(NEW17) + HELD
    rows = [(disp(s), [paired_eps("D051", D051["m96"], D051["m192"], [s])]) for s in order]
    forest(axes[1], rows, [M192], ms=6)
    axes[1].set_title("26-letter models", loc="left", fontsize=15)
    axes[1].tick_params(axis="y", labelsize=11)
    axes[1].axhline(9.5, color=INK2, lw=1)
    axes[1].axhline(26.5, color=INK2, lw=1)
    for ax in axes:
        ax.set_xlabel("192 px minus 96 px (points)")
    return fig


# TYPED-IN: LEARNING_FINDINGS.md section 7 "What helps" table (successes of 1000: 100 validation scenes x 10 shapes).
LABEL_AB = TYPED["label_ab_successes_of_1000"]["values"]


def fig_analysis_label_ab():
    fig, ax = plt.subplots(figsize=FULL)
    ys = np.arange(len(LABEL_AB))
    h = 0.36
    for j, (lbl, c) in enumerate((("without the change", tint(TEN, 0.6)), ("with the change", TEN))):
        v = [r[1 + j] / 10 for r in LABEL_AB]
        lo, hi = zip(*[wilson(r[1 + j], 1000) for r in LABEL_AB])
        y = ys - h / 2 + h * j
        ax.barh(y, v, h, color=c, edgecolor="white", lw=1.5, zorder=3, label=lbl)
        errbars(ax, y, v, lo, hi, horizontal=True)
        for yi, vi, hh in zip(y, v, hi):
            ax.text(hh + 0.3, yi, f"{vi:.1f}", va="center", fontsize=12)
    ax.set_yticks(ys, [r[0] for r in LABEL_AB], fontsize=12.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlim(0, 19.5)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Validation success (100 scenes x 10 shapes), 95% interval")
    ax.text(17.0, 0.18, "+68%", fontsize=14, fontweight="bold", color=TEN192, va="center")
    legend_top(ax, ncol=2, fontsize=12)
    note(fig, "Matched A/B tests, 30k steps from the official checkpoint (last row: 50k steps from scratch). Typed in "
              "from LEARNING_FINDINGS.md section 7.")
    return fig


def fig_analysis_budget():
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.35})
    # TYPED-IN: DECISIONS.md D053: training samples per shape per correction round.
    ax = axes[0]
    (l1, n1), (l2, n2), (l3, n3) = TYPED["budget_samples_per_shape_per_round_thousands"]["values"]
    rows = [(l1, n1, TEN), (l2, n2, tint(M96, 0.5)), (l3, n3, M96)]
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], 0.6, color=[r[2] for r in rows], edgecolor="white", zorder=3)
    for xi, r in zip(x, rows):
        ax.text(xi, r[1] + 10, f"{r[1]}k", ha="center", fontsize=13)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=12)
    ax.set_ylabel("Training samples per shape\nper round (thousands)")
    ax.set_ylim(0, 600)
    ax.grid(False, axis="x")
    ax = axes[1]
    s, *_ = val_line("27 shapes", "96 px line")
    phases = [("start\n(round 0)", [1]), ("old budget\n(rounds 10-13)", [11, 12, 13, 14]),
              ("larger budget\n(rounds 40-49)", list(range(41, 51)))]
    groups = [("10 earlier shapes", OLD10), ("17 newer letters", NEW17)]
    w = 0.26
    for j, (pn, steps) in enumerate(phases):
        v, lo, hi = [], [], []
        for gn, shapes in groups:
            k = n = 0
            for r in VAL:
                if r["line"] == "96 px line" and r["stage"] == "27 shapes" and int(r["step"]) in steps and \
                        r["shape"] in shapes:
                    k += int(r["successes"]); n += int(r["episodes"])
            v.append(100 * k / n)
            l_, h_ = wilson(k, n)
            lo.append(l_); hi.append(h_)
        xx = np.arange(len(groups)) - w + w * j
        ax.bar(xx, v, w, color=BLUE_RAMP[j], edgecolor="white", lw=1.5, zorder=3, label=pn.replace("\n", " "))
        errbars(ax, xx, v, lo, hi)
        for xi, vi, h in zip(xx, v, hi):
            ax.text(xi, h + 0.6, f"{vi:.0f}", ha="center", fontsize=12)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylim(0, 45)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Validation success (26-letter, 96 px)")
    ax.grid(False, axis="x")
    legend_top(ax, ncol=1, fontsize=12)
    note(fig, "Left: typed in from DECISIONS.md D053. Right: pooled validation rounds (50 scenes per shape per round).")
    return fig


def fig_analysis_plateau_noise():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=False, gridspec_kw={"wspace": 0.28})
    for ax, stage, line, key, sel, title in (
            (axes[0], "10 shapes", "96 px main line", "ten96", slice(1, None), "10 shapes, 96 px: rounds 1-21"),
            (axes[1], "27 shapes", "96 px line", "m96", slice(30, None), "26 letters, 96 px: rounds 30-49")):
        s, y, k, n = val_line(stage, line)
        x, y, k, n = s[sel] - 1, y[sel], k[sel], n[sel]
        p = k.sum() / n.sum()
        sd_chance = 100 * math.sqrt(p * (1 - p) / n[0])
        ax.axhspan(100 * p - 1.96 * sd_chance, 100 * p + 1.96 * sd_chance, color=tint(col(key), 0.85), lw=0)
        ax.axhline(100 * p, color=col(key), lw=1.2)
        ax.plot(x, y, "o-", color=col(key), lw=1.4, ms=6)
        ax.set_title(title, loc="left", fontsize=14)
        ax.text(0.02, 0.04, f"observed spread {np.std(y, ddof=1):.1f} points\nchance alone {sd_chance:.1f} points",
                transform=ax.transAxes, fontsize=12, color=INK2)
        ax.set_ylim(100 * p - 10, 100 * p + 10)
        ax.yaxis.set_major_formatter(PCT)
        ax.grid(False, axis="x")
        ax.set_xlabel("Correction rounds completed")
    axes[0].set_ylabel("Validation success")
    note(fig, "Shaded: 95% range expected from scene-to-scene chance alone around the mean. Round-to-round changes on "
              "the plateau look like noise.")
    return fig


def fig_analysis_selection_bias():
    # validation of the picked student (50 scenes per shape) vs its fresh-scene final test on the same shape set
    entries = [("5 shapes, round 4", "5 shapes", "correction rounds", 4, "D046", "5-shape round 4", FIVE_SHAPES, "five"),
               ("5 shapes, round 7", "5 shapes", "correction rounds", 7, "D047", "5-shape round 7", FIVE_SHAPES, "five"),
               ("10 shapes, 96 px", "10 shapes", "96 px main line", 14, "D049/D050", "10-shape 96 px", OLD10, "ten96"),
               ("10 shapes, 192 px", "10 shapes", "192 px side line", 7, "D049/D050", "10-shape 192 px", OLD10,
                "ten192"),
               ("26 letters, 96 px", "27 shapes", "96 px line", 45, "D051", "26-letter 96 px (round 44)", TRAINED27,
                "m96"),
               ("26 letters, 192 px", "27 shapes", "192 px line", 3, "D051", "26-letter 192 px (round 2)", TRAINED27,
                "m192")]
    fig, ax = plt.subplots(figsize=FULL)
    for i, (name, stage, line, step, test, model, shapes, key) in enumerate(entries):
        s, y, *_ = val_line(stage, line)
        v = y[list(s).index(step)]
        t, _, _, lo, hi = rate(test, model, shapes)
        ax.plot([v, t], [i, i], color=tint(col(key), 0.5), lw=2.4, zorder=2)
        ax.scatter([v], [i], s=90, facecolor="white", edgecolor=col(key), lw=2, zorder=3)
        ax.scatter([t], [i], s=100, color=col(key), edgecolor="white", lw=1.4, zorder=4)
        ax.plot([lo, hi], [i, i], color=col(key), lw=1, zorder=3)
        ax.text(v + 0.4, i - 0.22, f"{v:.1f}", fontsize=12, color=INK2)
        ax.text(t - 0.4, i - 0.22, f"{t:.1f}", fontsize=12, color=INK, ha="right")
    ax.set_yticks(range(len(entries)), [e[0] for e in entries])
    ax.set_ylim(len(entries) - 0.5, -0.8)
    ax.set_xlim(28, 46)
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Success on the stage's shapes")
    legend_top(ax, handles=[Line2D([], [], marker="o", mfc="white", mec=INK2, mew=2, ls="none", ms=9,
                                   label="validation at the picked round (50 scenes per shape)"),
                            Line2D([], [], marker="o", color=INK2, ls="none", ms=9,
                                   label="fresh-scene final test")], ncol=2, fontsize=12)
    note(fig, "Picking the best of many noisy validation rounds overstates success by 1-6 points; fresh tests "
              "correct for it.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# rl_*: forward-process RL on the simulator (all TYPED-IN)
# ----------------------------------------------------------------------------------------------------------------------
# TYPED-IN: LEARNING_EXPERIMENT_LOG.md section 2, "Long runs" table: T success at update 0 and at the last evaluation
# (small development evaluation). (setting, AdvantageFlow last T, NFT last T, last update)
RL_LONG = TYPED["rl_long_runs"]["values"]
RL_T0 = TYPED["rl_long_runs"]["t_success_at_update_0"]
# TYPED-IN: same table, (T drift ratio, last-evaluation T success) per long run with an evaluation.
RL_DRIFT = TYPED["rl_drift"]["values"]
# TYPED-IN: LEARNING_FINDINGS.md section 2 table (score_snr.py): plan share of score variance, split-half reliability.
RL_SNR = TYPED["rl_signal_share"]["values"]
# TYPED-IN: LEARNING_EXPERIMENT_LOG.md section 3, value function as candidate ranker: mean correlation with the true
# full-episode return (trainer's 32-control score, value-function score, one full-episode draw).
RL_RANK = TYPED["rl_candidate_ranking"]["values"]


def fig_rl_t_collapse():
    fig, ax = plt.subplots(figsize=FULL)
    ys = np.arange(len(RL_LONG))
    ax.axvline(RL_T0 * 100, color=ORIG_LINE, lw=1.6, ls=(0, (4, 3)))
    ax.text(RL_T0 * 100 + 1, -0.75, "T success before RL: 55%", fontsize=12, color=INK2, va="bottom")
    for i, (name, af, nft, upd) in enumerate(RL_LONG):
        for v, mk, off in ((af, "o", -0.12), (nft, "s", 0.12)):
            if v is None:
                continue
            ax.plot([RL_T0 * 100, v * 100], [i + off, i + off], color=tint(RL, 0.6), lw=2, zorder=2)
            ax.plot(v * 100, i + off, mk, color=RL, ms=10, mec="white", mew=1.4, zorder=3)
        ax.text(102, i, f"update {upd}", va="center", fontsize=12, color=INK2)
    ax.set_yticks(ys, [r[0] for r in RL_LONG])
    ax.set_ylim(len(RL_LONG) - 0.4, -0.9)
    ax.set_xlim(-3, 115)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("T success at the last evaluation")
    legend_top(ax, handles=[Line2D([], [], marker="o", color=RL, ls="none", ms=9, label="AdvantageFlow, after RL"),
                            Line2D([], [], marker="s", color=RL, ls="none", ms=9, label="NFT, after RL")], ncol=2,
               fontsize=12, y=1.06)
    note(fig, "Forward-process RL on the physical simulator, up to ~1M controls per run; small development "
              "evaluation. Typed in from LEARNING_EXPERIMENT_LOG.md section 2.")
    return fig


def fig_rl_drift_vs_t():
    fig, ax = plt.subplots(figsize=HALF)
    for alg, mk in (("AF", "o"), ("NFT", "s")):
        pts = [(d, t) for a, d, t in RL_DRIFT if a == alg]
        ax.scatter([p[0] for p in pts], [100 * p[1] for p in pts], marker=mk, s=80, color=RL, edgecolor="white",
                   lw=1.4, zorder=3, label="AdvantageFlow" if alg == "AF" else "NFT")
    ax.axhline(RL_T0 * 100, color=ORIG_LINE, lw=1.6, ls=(0, (4, 3)))
    ax.text(1.2e-2, RL_T0 * 100 + 2, "before RL", fontsize=12, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(-5, 75)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.set_xlabel("How far the policy moved\n(drift ratio, log scale)")
    ax.set_ylabel("T success after RL")
    ax.legend(loc="upper right", fontsize=12)
    return fig


def fig_rl_signal():
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.1})
    x = np.arange(len(RL_SNR))
    for ax, j, title in ((axes[0], 1, "Share of the training score\nexplained by the plan itself"),
                         (axes[1], 2, "Split-half reliability\nof the score")):
        v = [r[j] * 100 for r in RL_SNR]
        ax.bar(x, v, 0.6, color=RL, edgecolor="white", zorder=3)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}%", ha="center", fontsize=13)
        ax.set_xticks(x, [r[0] for r in RL_SNR])
        ax.set_title(title, loc="left", fontsize=14)
        ax.grid(False, axis="x")
        ax.axhline(100, color=AXIS, lw=1)
    axes[0].set_ylim(0, 105)
    axes[0].yaxis.set_major_formatter(PCT)
    note(fig, "8 candidate plans x 8 independent follow-up draws per context. Most of the score is follow-up noise. "
              "Typed in from LEARNING_FINDINGS.md section 2.")
    return fig


def fig_rl_candidate_ranking():
    fig, ax = plt.subplots(figsize=FULL)
    names = ["trainer's 32-control score", "learned value function", "one full-episode rollout"]
    colors = [RL, tint(RL, 0.55), INK2]
    vals = [[r[1 + j] for r in RL_RANK] for j in range(3)]
    xs = np.arange(len(RL_RANK))
    w = 0.26
    for j in range(3):
        b = ax.bar(xs - w + w * j, vals[j], w, color=colors[j], edgecolor="white", lw=1.5, zorder=3, label=names[j])
        for xi, v in zip(xs - w + w * j, vals[j]):
            ax.text(xi, v + (0.015 if v >= 0 else -0.015), f"{v:.2f}".replace("-", "−"), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=11.5)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(xs, [r[0] for r in RL_RANK])
    ax.grid(False, axis="x")
    ax.set_ylim(-0.2, 0.62)
    ax.set_ylabel("Correlation with the true\nfull-episode return")
    legend_top(ax, ncol=3, fontsize=12)
    note(fig, "How well each score ranks candidate plans (6 contexts per shape). Typed in from "
              "LEARNING_EXPERIMENT_LOG.md section 3.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# method_*: the teacher
# ----------------------------------------------------------------------------------------------------------------------
def fig_method_teacher_5shapes():
    fig, ax = plt.subplots(figsize=FULL)
    shapes = FIVE_SHAPES
    x = np.arange(len(shapes))
    w = 0.38
    v_o = [ORIG5_FINALPOOL[s] for s in shapes]
    v_t = [TEACHER5[s] * 2 for s in shapes]
    lo_t, hi_t = zip(*[wilson(TEACHER5[s], 50) for s in shapes])
    ax.bar(x - w / 2, v_o, w, color=ORIG, edgecolor="white", lw=1.5, zorder=3, label="Original (1000 test scenes)")
    ax.bar(x + w / 2, v_t, w, color=TEACHER, edgecolor="white", lw=1.5, zorder=3,
           label="Planning teacher in the world model (50 validation scenes)")
    errbars(ax, x + w / 2, v_t, lo_t, hi_t)
    for xi, v in zip(x - w / 2, v_o):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=12)
    for xi, v, h in zip(x + w / 2, v_t, hi_t):
        ax.text(xi, h + 1, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(x, [disp(s) for s in shapes])
    ax.grid(False, axis="x")
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success")
    legend_top(ax, ncol=1, fontsize=12)
    note(fig, "First sampling-planner teacher; reads exact simulator state, plans in the learned world model. Typed in "
              "from LEARNING_FINDINGS.md section 5.")
    return fig


def fig_method_teacher_26():
    shapes = sorted(TEACHER26, key=lambda s: -TEACHER26[s])
    fig, ax = plt.subplots(figsize=FULL)
    x = np.arange(len(shapes))
    w = 0.27
    v_o = [rate("D051", "original", [s])[0] for s in shapes]
    v_s = [rate("D051", D051["m96"], [s])[0] for s in shapes]
    v_t = [TEACHER26[s] * 2 for s in shapes]
    ax.bar(x - w, v_o, w, color=ORIG, edgecolor="white", lw=1, zorder=3, label="Original (test)")
    ax.bar(x, v_s, w, color=M96, edgecolor="white", lw=1, zorder=3, label="Ours, 26 letters (test)")
    ax.bar(x + w, v_t, w, color=TEACHER, edgecolor="white", lw=1, zorder=3, label="Teacher (validation)")
    ax.set_xticks(x, [disp(s) for s in shapes], fontsize=13)
    ax.grid(False, axis="x")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success")
    legend_top(ax, ncol=3, fontsize=12)
    note(fig, "Teacher: 50 validation scenes per letter (typed in from LEARNING_FINDINGS.md section 7). Original and "
              f"ours: {TEST_NOTE}.")
    return fig


def fig_method_teacher_gap():
    # TYPED-IN: teacher totals (LEARNING_FINDINGS.md sections 5 and 7); students from validation_rounds.csv
    tt = TYPED["teacher_totals"]["values"]
    rows = [("5 shapes", tt["5 shapes"][0] / tt["5 shapes"][1] * 100, "5 shapes", "correction rounds", 7, FIVE),
            ("10 shapes", tt["10 shapes"][0] / tt["10 shapes"][1] * 100, "10 shapes", "96 px main line", 14, TEN)]
    fig, ax = plt.subplots(figsize=HALF)
    x = np.arange(len(rows))
    w = 0.36
    for i, (name, t, stage, line, step, c) in enumerate(rows):
        s, y, *_ = val_line(stage, line)
        v = y[list(s).index(step)]
        ax.bar(i - w / 2, t, w, color=TEACHER, edgecolor="white", zorder=3)
        ax.bar(i + w / 2, v, w, color=c, edgecolor="white", zorder=3)
        ax.text(i - w / 2, t + 1.5, f"{t:.0f}", ha="center", fontsize=13)
        ax.text(i + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=13)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("Validation success")
    ax.legend(handles=[patch(TEACHER, "teacher (exact state)"), patch(FIVE, "camera student, 5 shapes"),
                       patch(TEN, "camera student, 10 shapes")], loc="upper left", fontsize=12)
    note(fig, "Students: best validation round (round 7; round 13). Teacher totals typed in from "
              "LEARNING_FINDINGS.md sections 5 and 7.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# LaTeX tables
# ----------------------------------------------------------------------------------------------------------------------
def _ci(d):
    return f"{fmt_signed(d['difference'])} [{fmt_signed(d['ci95_low'])}, {fmt_signed(d['ci95_high'])}]"


def write_tables():
    TAB.mkdir(parents=True, exist_ok=True)
    head = "% Generated by presentation/scripts/make_figures.py from presentation/data/*.csv. Needs \\usepackage{booktabs}.\n"
    # headline
    groups = ["10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"]
    lines = [head, "\\begin{tabular}{lrrrr}", "\\toprule",
             "Shapes & Original & Ours, 10 shapes & Ours, 26 letters & Ours, 26 letters (192 px) \\\\", "\\midrule"]
    for g in groups:
        vals = [rate("D051", D051[k], GROUP_SHAPES[g])[0] for k in KEYS4]
        cells = [f"{v:.1f}" for v in vals]
        if g == "27 trained shapes":
            cells[2:] = [f"\\textbf{{{c}}}" for c in cells[2:]]
        lines.append(f"{GROUP_LABEL[g]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for g in ("27 trained shapes", "6 held-out shapes"):
        ds = [paired_row("D051", "original", D051[k], g) for k in KEYS4[1:]]
        lines.append(f"Change vs.\\ original, {GROUP_LABEL[g].lower()} & -- & " + " & ".join(_ci(d) for d in ds) + " \\\\")
    cov = [coverage("D051", D051[k], TRAINED27) / 100 for k in KEYS4]
    lines.append("Final coverage, 27 trained shapes & " + " & ".join(f"{c:.2f}" for c in cov) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %; final test, 300 fresh scenes per shape, paired by scene; "
              "differences in points with 95% intervals."]
    (TAB / "final_headline.tex").write_text("\n".join(lines) + "\n")

    # per shape, one long table
    def shape_row(s):
        vals = [rate("D051", D051[k], [s])[0] for k in KEYS4]
        d = paired_row("D051", "original", D051["m96"], s)
        return f"{disp(s)} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {fmt_signed(d['difference'])} \\\\"
    lines = [head, "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Shape & Original & 10 shapes & 26 letters & 26 letters, 192 px & Change (26 letters) \\\\", "\\midrule"]
    for block, shapes in (("10 earlier shapes", OLD10), ("17 newer letters", NEW17), ("6 never-seen shapes", HELD)):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{block}}}}} \\\\")
        lines += [shape_row(s) for s in shapes]
        if block != "6 never-seen shapes":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %, final test, 300 fresh scenes per shape; change = 26-letter "
              "96 px minus original, points."]
    (TAB / "final_per_shape.tex").write_text("\n".join(lines) + "\n")

    # per shape, two blocks side by side (fits one slide at \scriptsize)
    order = OLD10 + NEW17 + HELD
    half = math.ceil(len(order) / 2)
    left, right = order[:half], order[half:]
    lines = [head, "\\begin{tabular}{lrrrr@{\\hspace{1.5em}}lrrrr}", "\\toprule",
             "Shape & Orig. & 10 sh. & 26 l. & 192 px & Shape & Orig. & 10 sh. & 26 l. & 192 px \\\\", "\\midrule"]
    for i in range(half):
        cells = []
        for blk in (left, right):
            if i < len(blk):
                s = blk[i]
                cells.append(f"{disp(s)} & " + " & ".join(f"{rate('D051', D051[k], [s])[0]:.1f}" for k in KEYS4))
            else:
                cells.append(" & & & & ")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %, final test, 300 fresh scenes per shape. Order: 10 earlier "
              "shapes, 17 newer letters, 6 never-seen shapes (digit 3 ... arrow)."]
    (TAB / "final_per_shape_compact.tex").write_text("\n".join(lines) + "\n")

    # ten-shape test
    lines = [head, "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Shape & Original & 96 px & Change & 192 px & Change \\\\", "\\midrule"]
    for s in OLD10 + [None]:
        shapes = OLD10 if s is None else [s]
        g = "10 earlier shapes" if s is None else s
        vals = [rate("D049/D050", D049[k], shapes)[0] for k in ("original", "ten96", "ten192")]
        d96 = paired_row("D049/D050", "original", "10-shape 96 px", g)
        d192 = paired_row("D049/D050", "original", "10-shape 192 px", g)
        name = "\\textbf{All 10}" if s is None else disp(s)
        if s is None:
            lines.append("\\midrule")
        lines.append(f"{name} & {vals[0]:.1f} & {vals[1]:.1f} & {_ci(d96)} & {vals[2]:.1f} & {_ci(d192)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %, final test, 1000 fresh scenes per shape, paired by scene; "
              "change in points with 95% interval."]
    (TAB / "final_10shapes.tex").write_text("\n".join(lines) + "\n")

    # round 50 vs 44 and steps executed
    lines = [head, "\\begin{tabular}{lrrr}", "\\toprule", "Shapes & Round 44 & Round 50 & Change \\\\", "\\midrule"]
    for g in groups:
        d = paired_row("D054", "26-letter 96 px round 44", "26-letter 96 px round 50", g)
        lines.append(f"{GROUP_LABEL[g]} & {d['base_rate']:.1f} & {d['model_rate']:.1f} & {_ci(d)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% 26-letter 96 px student; 200 fresh scenes per shape, paired by scene."]
    (TAB / "round50_vs_44.tex").write_text("\n".join(lines) + "\n")

    lines = [head, "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Shapes & Execute 8 & Execute 12 & Change & Execute 15 & Change \\\\", "\\midrule"]
    for g in groups:
        d12 = paired_row("D056", "execute 8 (trained)", "execute 12", g)
        d15 = paired_row("D056", "execute 8 (trained)", "execute 15", g)
        lines.append(f"{GROUP_LABEL[g]} & {d12['base_rate']:.1f} & {d12['model_rate']:.1f} & {_ci(d12)} & "
                     f"{d15['model_rate']:.1f} & {_ci(d15)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Steps of each 16-step plan executed before re-planning; round-50 "
              "student, 200 fresh scenes per shape, paired by scene."]
    (TAB / "steps_executed.tex").write_text("\n".join(lines) + "\n")

    # D052 unseen capitals
    lines = [head, "\\begin{tabular}{lrrr}", "\\toprule", "Letter & Original & 10 shapes, 96 px & 10 shapes, 192 px \\\\",
             "\\midrule"]
    for s in sorted(NEW17) + [None]:
        shapes = NEW17 if s is None else [s]
        vals = [rate("D052", D049[k], shapes)[0] for k in ("original", "ten96", "ten192")]
        if s is None:
            lines.append("\\midrule")
        lines.append(("\\textbf{All 17}" if s is None else s) + " & " + " & ".join(f"{v:.1f}" for v in vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success % of 10-shape students on the 17 capitals they never trained "
              "on; 200 fresh scenes per letter."]
    (TAB / "unseen_capitals.tex").write_text("\n".join(lines) + "\n")

    # typed-in A/B
    lines = [head, "\\begin{tabular}{lrr}", "\\toprule", "Change & Without & With \\\\", "\\midrule"]
    for name, a, b in LABEL_AB:
        lines.append(f"{name.replace(chr(10), ' ')} & {a / 10:.1f} & {b / 10:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Validation success %, 100 scenes x 10 shapes; typed in from "
              "docs/LEARNING_FINDINGS.md section 7."]
    (TAB / "label_ab.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------------------------------
# README
# ----------------------------------------------------------------------------------------------------------------------
def write_readme():
    lines = ["# Presentation figures", "",
             "Generated by `presentation/scripts/make_figures.py` (do not edit by hand). Every figure exists as a vector",
             "PDF (for Beamer) and a 200 dpi PNG (for PowerPoint) with the same name. Regenerate with:", "",
             "```bash", "MPLBACKEND=Agg python presentation/scripts/make_figures.py", "```", "",
             "Colours follow the model everywhere: original = light grey, 5-shape students = raspberry, 10-shape",
             "students = orange (192 px camera: dark orange), 26-letter 96 px = blue, 26-letter 192 px = aqua,",
             "planning teacher = gold, forward RL = violet. \"Validation\" = 50 fixed scenes per shape used to pick",
             "checkpoints; \"final test\" = fresh scenes never used before, paired by scene with the original.",
             "Charts marked *typed-in* use numbers copied from the project docs (the script cites the section).", "",
             "LaTeX tables (booktabs) are in `presentation/data/tables/`: `final_headline.tex`, `final_per_shape.tex`,",
             "`final_per_shape_compact.tex`, `final_10shapes.tex`, `round50_vs_44.tex`, `steps_executed.tex`,",
             "`unseen_capitals.tex`, `label_ab.tex`.", "",
             "| File | What it shows | Data | Suggested use |", "| --- | --- | --- | --- |"]
    for name, shows, data, use in REGISTRY:
        lines.append(f"| `{name}` | {shows} | {data} | {use} |")
    (FIG / "README.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
V = "validation_rounds.csv"
S = "final_tests_by_shape.csv"
P = "final_tests_paired.csv"
E = "final_tests_episodes.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=[], help="figure name prefixes to regenerate")
    ONLY.extend(ap.parse_args().only)
    style()
    jobs = [
        ("curves_all_stages", lambda: fig_curves_all_stages(),
         "Whole training story: validation success per correction round, 5 -> 10 -> 27 shapes on one axis, picks circled, "
         "original policy dashed.", f"{V}; original levels from {P}", "story slide (main)",
         "Validation success through all three stages"),
        ("curves_all_stages_smoothed", lambda: fig_curves_all_stages(smooth=True),
         "Same story with faint raw lines and a rolling mean (3 rounds for 5 shapes, 5 rounds otherwise).", V,
         "story slide, cleaner look", None),
        ("curves_all_stages_panels", fig_curves_all_stages_panels,
         "Same story as three side-by-side panels with a shared y-axis.", V, "story slide / paper", None),
        ("curves_26letters", lambda: fig_curves_26letters(),
         "26-letter lines (96 px 50 rounds, 192 px 13 rounds) with 95% bands, larger-budget start (round 14) and the "
         "picked rounds (44 and 2).", V, "26-letter training slide", "Training on all 26 letters"),
        ("curves_26letters_smoothed", lambda: fig_curves_26letters(smooth=True),
         "Same with a 5-round rolling mean.", V, "26-letter training slide", None),
        ("curves_26letters_groups", fig_curves_26letters_groups,
         "26-letter lines split into the 10 earlier shapes vs the 17 newer letters.", V, "budget / catch-up slide", None),
        ("curves_26letters_coverage", fig_curves_26letters_coverage,
         "Mean final goal coverage per round for both 26-letter lines.", V, "backup", None),
        ("curves_26letters_small_multiples", fig_curves_small_multiples,
         "One small panel per shape: 26-letter validation success over rounds (rolling means).", V, "paper appendix",
         None),
        ("curves_10shapes", fig_curves_10shapes,
         "10-shape stage: executed-push vs intended-plan labels (left) and 96 px vs 192 px lines (right); teacher's 78% "
         "dashed.", f"{V}; teacher typed-in", "10-shape slide", None),
        ("curves_5shapes", fig_curves_5shapes,
         "5-shape stage: copying the teacher without corrections vs training steps (left), correction rounds (right); "
         "original and teacher on the same scenes dashed.", f"{V}; reference lines typed-in", "5-shape slide", None),
        ("curves_heatmap_26letters", lambda: fig_heatmap_26(),
         "Shapes x rounds heatmap of validation success, 26-letter 96 px line.", V, "per-shape progress / paper", None),
        ("curves_heatmap_26letters_smoothed", lambda: fig_heatmap_26(smooth=True),
         "Same heatmap with a 5-round rolling mean per shape (less noise).", V, "per-shape progress", None),
        ("curves_heatmap_26letters_coverage", lambda: fig_heatmap_26(metric="mean_final_coverage"),
         "Shapes x rounds heatmap of mean final coverage, 26-letter 96 px line.", V, "backup / paper", None),
        ("curves_heatmap_26letters_192", lambda: fig_heatmap_26(line="192 px line"),
         "Shapes x rounds heatmap of validation success, 26-letter 192 px line (13 rounds).", V, "backup", None),
        ("final_grouped_bars", lambda: fig_final_grouped_bars(),
         "HEADLINE final test by shape group (10 earlier, 17 newer, 27 trained, 6 never-seen) for the four policies, "
         "95% Wilson intervals.", S, "results slide (main)", "Final test: one camera policy for 26 letters"),
        ("final_headline_bars", lambda: fig_final_grouped_bars(groups=("27 trained shapes", "6 held-out shapes"),
                                                               size=HALF),
         "Compact headline: 27 trained shapes and 6 never-seen shapes only.", S, "half-slide headline", None),
        ("final_stage_progression", fig_final_stage_progression,
         "Each stage's fresh-scene test vs the original on the same scenes: success stays ~34% as shapes grow 5 -> 10 -> "
         "27.", f"{P}, {S}", "summary slide", None),
        ("final_per_shape_bars", lambda: fig_final_per_shape_bars_horizontal(),
         "Horizontal bars for all 33 shapes (sorted), original vs 26-letter 96 px; never-seen shapes below the line.",
         S, "paper / tall slide", None),
        ("final_per_shape_bars_vertical", fig_final_per_shape_bars_vertical,
         "Same as vertical bars across a 16:9 slide.", S, "per-shape slide", None),
        ("final_per_shape_bars_all_models", lambda: fig_final_per_shape_bars_horizontal(KEYS4, size=(7.5, 13)),
         "Horizontal bars for all 33 shapes and all four policies.", S, "paper appendix", None),
        ("final_dumbbell", lambda: fig_final_dumbbell(),
         "Dumbbell per shape: original -> 26-letter 96 px (tall).", S, "paper", None),
        ("final_dumbbell_vertical", lambda: fig_final_dumbbell(vertical=True),
         "Dumbbell per shape across a 16:9 slide.", S, "per-shape slide", None),
        ("final_dumbbell_192", lambda: fig_final_dumbbell(key="m192"),
         "Dumbbell per shape: original -> 26-letter 192 px.", S, "backup", None),
        ("final_scatter", lambda: fig_final_scatter(),
         "Original vs 26-letter 96 px success per shape with the no-change line.", S, "half-slide / paper", None),
        ("final_scatter_10shape", lambda: fig_final_scatter(ykey="ten96"),
         "Original vs 10-shape 96 px success per shape (17 newer letters barely move).", S, "backup", None),
        ("final_forest_groups", fig_final_forest_groups,
         "Paired change vs original with 95% intervals per shape group for the three students.", P, "results slide",
         None),
        ("final_forest_shapes", fig_final_forest_shapes,
         "Paired change vs original with 95% intervals for every shape (26-letter 96 px and 192 px).", P,
         "paper appendix", None),
        ("final_10shapes_bars", fig_final_10shapes_bars,
         "10-shape final test per shape, 1000 fresh scenes per shape, original vs 96 px vs 192 px.", S,
         "10-shape results slide", None),
        ("final_10shapes_forest", fig_final_10shapes_forest,
         "10-shape paired change vs original per shape with 95% intervals.", P, "backup / paper", None),
        ("final_5shapes_bars", fig_final_5shapes_bars,
         "5-shape final tests after 4 and 7 correction rounds vs original (two fresh pools).", S, "5-shape slide", None),
        ("final_blends_5shapes", fig_final_blends,
         "Early weight-blend attempt (5 shapes): small Y/D gains, T loss.", S, "what did not work / backup", None),
        ("shapes_96_vs_192", fig_shapes_96_vs_192,
         "Per-shape success, 26-letter 96 px vs 192 px camera.", S, "camera slide / backup", None),
        ("shapes_10_vs_26", fig_shapes_10_vs_26,
         "Per-shape success, 10-shape vs 26-letter policy: earlier ten unchanged, newer 17 jump.", S, "scaling slide",
         None),
        ("shapes_validation_vs_test", fig_shapes_validation_vs_test,
         "Per-shape validation (round 44) vs final test success: how noisy 50-scene validation is.", f"{V}, {S}",
         "methods / backup", None),
        ("shapes_teacher_vs_student", fig_shapes_teacher_vs_student,
         "Teacher (exact state) vs camera student success per letter.", f"{S}; teacher typed-in", "teacher-gap slide",
         None),
        ("shapes_new_letters_dotplot", fig_shapes_new_letters_dotplot,
         "17 newer letters: original, 10-shape model (never saw them), 26-letter model (trained on them).", S,
         "transfer vs training slide", None),
        ("shapes_table_heatmap", fig_shapes_table_heatmap,
         "Table-like heatmap of success for 33 shapes x 4 policies with numbers.", S, "paper / backup", None),
        ("heldout_scaling", fig_heldout_scaling,
         "Never-seen shapes: success and final coverage vs number of training shapes (1 -> 10 -> 27).", S,
         "generalisation slide (main)", "More training shapes help on never-seen shapes"),
        ("heldout_per_shape_bars", fig_heldout_per_shape_bars,
         "Each never-seen shape for the four policies.", S, "generalisation slide", None),
        ("heldout_scaling_per_shape", fig_heldout_scaling_per_shape,
         "Small multiples: each never-seen shape vs number of training shapes.", S, "backup / paper", None),
        ("heldout_unseen_capitals", fig_heldout_unseen_capitals,
         "10-shape students on the 17 capitals they never trained on, per letter and pooled.", S,
         "transfer slide", None),
        ("coverage_final_hist", lambda: fig_coverage_hist("final"),
         "Distribution of final goal coverage per episode, original vs 26-letter 96 px (27 trained shapes).", E,
         "near-miss slide", None),
        ("coverage_peak_hist", lambda: fig_coverage_hist("peak"),
         "Distribution of peak goal coverage per episode, original vs 26-letter 96 px.", E, "near-miss slide", None),
        ("coverage_cdf", fig_coverage_cdf,
         "Cumulative distributions of final and peak coverage for the four policies.", E, "backup / paper", None),
        ("coverage_violin", fig_coverage_violin,
         "Violins of final coverage per shape group and policy.", E, "backup / paper", None),
        ("coverage_near_misses", fig_coverage_near_misses,
         "Episode outcomes: solved / near miss (peak 90-95%) / peak 80-90% / below, trained and never-seen shapes.", E,
         "last-mile precision slide", None),
        ("coverage_success_vs_coverage", fig_coverage_success_vs_coverage,
         "Per shape and policy: mean final coverage vs success.", S, "backup", None),
        ("coverage_controls", fig_coverage_controls,
         "Solved episodes vs controls used: cumulative over all episodes and among solved ones.", E, "backup / paper",
         None),
        ("coverage_steps_executed", fig_coverage_steps_executed,
         "Final coverage CDF when executing 8, 12 or 15 of 16 planned steps.", E, "backup", None),
        ("analysis_steps_executed", fig_analysis_steps_executed,
         "Executing 8 (trained) vs 12 vs 15 planned steps per shape group with 95% intervals and paired changes.", f"{S}, "
         f"{P}", "ablation slide", None),
        ("analysis_commitment", fig_analysis_commitment,
         "Committing to more steps hurts both the exact-state teacher (typed-in) and the camera student.",
         f"{S}; teacher typed-in", "ablation slide", None),
        ("analysis_round50_vs_44", lambda: fig_analysis_round50_vs_44(),
         "Six more rounds: round 50 minus round 44 per group, paired, 95% intervals.", P, "results / backup", None),
        ("analysis_round50_vs_44_shapes", lambda: fig_analysis_round50_vs_44(per_shape=True),
         "Round 50 minus round 44 for every shape.", P, "paper appendix", None),
        ("analysis_t_tradeoff", lambda: fig_analysis_t_tradeoff(),
         "T and near-T success of each student next to the original on the same scenes.", S, "trade-off slide", None),
        ("analysis_t_tradeoff_diff", lambda: fig_analysis_t_tradeoff(diff=True),
         "T and near-T paired change vs original with 95% intervals.", P, "trade-off slide", None),
        ("analysis_192_vs_96", lambda: fig_analysis_192_vs_96(),
         "192 px minus 96 px camera, paired by scene, 10-shape and 26-letter models.", E, "camera slide", None),
        ("analysis_192_vs_96_shapes", lambda: fig_analysis_192_vs_96(per_shape=True),
         "192 px minus 96 px for every shape.", E, "paper appendix", None),
        ("analysis_label_ab", fig_analysis_label_ab,
         "Typed-in matched A/B tests: plan labels, camera, demo count, network size.", "typed-in (LEARNING_FINDINGS "
         "section 7)", "design-choices slide", None),
        ("analysis_budget", fig_analysis_budget,
         "26-letter budget diagnosis: samples per shape per round (typed-in) and validation before/after.",
         f"{V}; D053 typed-in", "budget slide", None),
        ("analysis_plateau_noise", fig_analysis_plateau_noise,
         "Plateau rounds against the chance band: changes between late rounds look like noise.", V, "backup / paper",
         None),
        ("analysis_selection_bias", fig_analysis_selection_bias,
         "Validation at the picked round vs fresh-scene test for every picked student.", f"{V}, {S}",
         "methods slide / backup", None),
        ("rl_t_collapse", fig_rl_t_collapse,
         "Forward-process RL: T success before vs after for each step size (typed-in).", "typed-in (EXPERIMENT_LOG "
         "section 2)", "what did not work slide", None),
        ("rl_drift_vs_t", fig_rl_drift_vs_t,
         "Forward RL: the more the policy moved, the more T it lost (typed-in).", "typed-in (EXPERIMENT_LOG section 2)",
         "what did not work slide", None),
        ("rl_signal_variance", fig_rl_signal,
         "Learning-signal diagnosis: plan share of score variance and split-half reliability (typed-in).",
         "typed-in (LEARNING_FINDINGS section 2)", "why RL failed slide", None),
        ("rl_candidate_ranking", fig_rl_candidate_ranking,
         "How well the trainer's score, a value function and one full rollout rank candidate plans (typed-in).",
         "typed-in (EXPERIMENT_LOG section 3)", "backup", None),
        ("method_teacher_success", fig_method_teacher_5shapes,
         "First planning teacher vs original on T, near-T, Y, D, L (typed-in).", "typed-in (LEARNING_FINDINGS "
         "section 5)", "method slide (teacher)", None),
        ("method_teacher_success_26letters", fig_method_teacher_26,
         "26-letter teacher (typed-in) vs original and the 26-letter student per letter.", f"{S}; teacher typed-in",
         "teacher-gap slide", None),
        ("method_teacher_gap", fig_method_teacher_gap,
         "Teacher vs best camera student on validation, 5 and 10 shapes.", f"{V}; teacher typed-in", "half-slide",
         None),
    ]
    for name, fn, shows, data, use, title in jobs:
        if not wanted(name):
            continue
        fig = fn()
        save(fig, name, shows, data, use, title)
        print("wrote", name)
    if not ONLY:
        write_tables()
        write_readme()
        print("wrote tables and README")


if __name__ == "__main__":
    main()
