#!/usr/bin/env python3
"""Regenerate every presentation figure, the LaTeX tables and figures/README.md from presentation/data/*.csv.

No GPU, simulator or network. Needs matplotlib and numpy only.

    MPLBACKEND=Agg python presentation/scripts/make_figures.py            # everything
    MPLBACKEND=Agg python presentation/scripts/make_figures.py --only final_ heldout_

Every figure is written twice into presentation/figures/: a vector PDF (Beamer) and a 200 dpi PNG (PowerPoint); the
method diagrams also as SVG. Slide versions without footnote or title go to figures/slide/ (6.3 x 3.5 in, full slide
width at 16:9, fonts >= 16 pt) and figures/slide_half/ (4.2 x 3.2 in, two-column slides, fonts >= 14 pt):

    MPLBACKEND=Agg python presentation/scripts/make_figures.py --variant slide      # only the slide versions
    MPLBACKEND=Agg python presentation/scripts/make_figures.py --variant none       # only the default versions

Tables go to presentation/data/tables/*.tex (booktabs). Every number a chart uses lives in presentation/data/: the CSV
tables built from the saved test files, typed_in_numbers.json and the typed-in tables (early_attempts.csv,
teacher_validation.csv, teacher_planner.csv, world_model_accuracy.csv, dataset_sizes.csv, timeline.csv), each row with
the document section it comes from (docs/LEARNING_FINDINGS.md, docs/LEARNING_EXPERIMENT_LOG.md, docs/DECISIONS.md).

Colours follow the model in every figure (validated with the dataviz palette checker, white surface):
original = grey, 5-shape students = raspberry, 10-shape students = orange (192 px camera: dark orange),
26-letter 96 px = blue, 26-letter 192 px = green, planning teacher = gold, forward RL = violet.
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
matplotlib.rcParams["svg.hashsalt"] = "tdyssey"      # reproducible SVG ids

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle  # noqa: E402
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
ORIG = ORIG_LINE = "#8f8d86"                    # original policy (3.3:1 on white; was #b4b2ab, 2.1:1)
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
class _SlideDict(dict):
    """Long labels by default; the short ones in the slide versions."""
    def __init__(self, long, short):
        super().__init__(long)
        self.short = short

    def __getitem__(self, k):
        return self.short[k] if MODE["name"] != "default" and k in self.short else dict.__getitem__(self, k)


GROUP_LABEL = _SlideDict({"10 earlier shapes": "10 earlier shapes", "17 other letters": "17 newer letters",
                          "27 trained shapes": "All 27 trained shapes", "6 held-out shapes": "6 never-seen shapes",
                          "5 first shapes": "5 first shapes"},
                         {"10 earlier shapes": "10 earlier", "17 other letters": "17 newer", "27 trained shapes": "All 27",
                          "6 held-out shapes": "6 unseen", "5 first shapes": "5 first"})
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
    "m96": ("Ours, 27 shapes, 96 px (round 44)", M96),          # the D051 final-test picks; round 50 (the video
    "m192": ("Ours, 27 shapes, 192 px (round 2)", M192),        # model) is only in the separate D054 check
    "teacher": ("Planning teacher (exact state)", TEACHER),
}
D051 = {"original": "original", "ten96": "10-shape 96 px", "m96": "26-letter 96 px (round 44)",
        "m192": "26-letter 192 px (round 2)"}
D049 = {"original": "original", "ten96": "10-shape 96 px", "ten192": "10-shape 192 px"}
TEST_NOTE = "final test, 300 fresh scenes per shape"


LAB_SHORT = {"original": "Original", "five": "Ours, 5 shapes", "ten96": "Ours, 10 shapes",
             "ten192": "Ours, 10 shapes, 192 px", "m96": "Ours, 96 px (round 44)", "m192": "Ours, 192 px (round 2)",
             "teacher": "Teacher"}
# whole validation lines (every round), not one picked student
LINE_LAB = {"m96": ("26 letters + near-T, 96 px", "96 px"), "m192": ("26 letters + near-T, 192 px", "192 px")}


def lab(key):
    return LAB_SHORT[key] if slide() else LAB[key][0]


def line_lab(key):
    return LINE_LAB[key][1 if slide() else 0]


def SL(default, short):
    """Pick the short text in the slide versions."""
    return short if slide() else default


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
VARIANT_REGISTRY = []     # (mode, name, what it shows) for the slide and half-slide versions
ONLY = []
# Output mode: "default" (figures/), "slide" (figures/slide/, 6.3 x 3.5 in, fonts >= 16 pt, no footnote or title) or
# "half" (figures/slide_half/, 4.2 x 3.2 in, fonts >= 14 pt). Figure functions check slide() to drop clutter.
MODE = {"name": "default"}
VARIANT_SIZE = {"slide": (6.3, 3.5), "half": (4.2, 3.2)}
VARIANT_FLOOR = {"slide": 16, "half": 14}
VARIANT_DIR = {"slide": "slide", "half": "slide_half"}


def slide():
    return MODE["name"] != "default"


def half():
    return MODE["name"] == "half"


def FS(size):
    """Font size respecting the variant's minimum (used where text is measured while the figure is built)."""
    return max(size, VARIANT_FLOOR.get(MODE["name"], 0))


def wanted(name):
    return not ONLY or any(name.startswith(p) for p in ONLY)


def _wrapped_note(fig, txt):
    """Footnote wrapped so it never extends past the drawing's right edge, placed just below everything else."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    w, h = fig.get_size_inches()
    x0 = max(bb.x0, 0)
    right = bb.x1
    t = fig.text(x0 / w, (bb.y0 - 0.06) / h, "", fontsize=12, color=INK2, ha="left", va="top")
    width = max(40, int((right - x0) * 12.5))
    while True:
        t.set_text(textwrap.fill(txt, width=width))
        if t.get_window_extent(r).x1 / fig.dpi <= right - 0.02 or width <= 30:
            break
        width -= 2


def variant_size(fig, w, h):
    """Ask for a different canvas than the variant's standard one (only used in the slide versions)."""
    if slide():
        fig._tdy_variant_size = (w, h)


def save_variant(fig, name, shows=""):
    """Slide / half-slide version: exact size, constrained layout, every text at least the variant's floor."""
    mode = MODE["name"]
    floor = VARIANT_FLOOR[mode]
    size = getattr(fig, "_tdy_variant_size", None) or VARIANT_SIZE[mode]
    fig.set_size_inches(*size)
    for ax in fig.axes:      # "Success\n(final test, 300 fresh scenes per shape)" -> "Success"
        for get, set_ in ((ax.get_ylabel, ax.set_ylabel), (ax.get_xlabel, ax.set_xlabel)):
            if "\n(" in get():
                set_(get().split("\n(")[0])
    for t in fig.findobj(Text):
        if t.get_text() and t.get_fontsize() < floor:
            t.set_fontsize(floor)
    out = FIG / VARIANT_DIR[mode]
    out.mkdir(parents=True, exist_ok=True)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200)
    if any("constrained_layout" in str(w.message) for w in caught):
        print(f"  WARNING {VARIANT_DIR[mode]}/{name}: content does not fit the canvas (constrained layout gave up)")
    shows = VARIANT_SHOWS.get((mode, name), shows).replace(" Also SVG.", "")
    if tuple(size) != tuple(VARIANT_SIZE[mode]):
        shows += f" Canvas {size[0]:g} x {size[1]:g} in."
    VARIANT_REGISTRY.append((mode, name, shows))
    plt.close(fig)


def save(fig, name, shows, data, use, title=None, svg=False):
    """Write <name>.pdf/.png (and .svg); with title also write <name>_titled.* carrying that title."""
    if slide():
        return save_variant(fig, name, shows)
    FIG.mkdir(parents=True, exist_ok=True)
    txt = getattr(fig, "_tdy_note", None)
    if txt:
        _wrapped_note(fig, txt)
    for ext in ("pdf", "png") + (("svg",) if svg else ()):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, **({"metadata": {"Date": None}} if ext == "svg" else {}))
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
    if not slide():          # slide versions carry no footnote
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
                                fontsize=FS(size), color=INK, zorder=6,
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


def round_labels(ax):
    """Half-slide bar labels: whole numbers, but keep one decimal below 1 (0.2 must not read as 0)."""
    for t in ax.texts:
        try:
            v = float(t.get_text())
        except ValueError:
            continue
        t.set_text(f"{v:.0f}" if v >= 1 else f"{v:.1f}")


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
    if half():                          # whole numbers, so neighbouring labels never touch
        round_labels(axp)
    for t in axp.texts:
        t.set_fontweight("bold" if not slide() or half() else "normal")
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
    if slide():   # one legend across the top of the canvas, as many columns as fit
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        else:
            labels = [h.get_label() for h in handles]
        return slide_legend(ax.figure, handles, labels, title=kw.get("title"))
    kw.setdefault("loc", "lower left")
    kw.setdefault("columnspacing", 1.4)
    return ax.legend(handles=handles, ncol=ncol, bbox_to_anchor=(0, y), handletextpad=0.5, **kw)


def stagger(labels):
    """Alternate tick labels between two rows (slide versions with many narrow categories)."""
    return [("\n" if i % 2 else "") + str(t) for i, t in enumerate(labels)] if slide() else list(labels)


def slide_legend(fig, handles, labels, title=None, ncol=None):
    """Figure legend above the axes (constrained layout makes room); columns chosen to fit the canvas width."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    fs = VARIANT_FLOOR[MODE["name"]]
    prop = FontProperties(family="sans-serif", size=fs)
    ttp = TextToPath()
    widths = [ttp.get_text_width_height_descent(str(l), prop, ismath=False)[0] + fs * (1.2 + 0.4 + 1.0)
              for l in labels]
    avail = VARIANT_SIZE[MODE["name"]][0] * 72 - 8
    if ncol is None:
        ncol = len(labels)
        while ncol > 1:      # matplotlib fills columns with heights [q + 1] * r + [q] * (ncol - r)
            q, r = divmod(len(labels), ncol)
            heights = [q + 1] * r + [q] * (ncol - r)
            starts = np.cumsum([0] + heights)
            cols = [max(widths[starts[c]:starts[c + 1]] or [0]) for c in range(ncol)]
            if 1.05 * sum(cols) <= avail:
                break
            ncol -= 1
    return fig.legend(handles, labels, loc="outside upper center", ncol=ncol, frameon=False, title=title,
                      handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)


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
    """All validation lines placed on one x-axis (rounds completed within each stage, shifted by STAGE_OFF)."""
    out = []
    s, y, k, n = val_line("5 shapes", "correction rounds")
    # round 0 of the 5-shape stage: the copy-the-teacher student the correction rounds started from (typed_in JSON)
    r0 = TYPED["five_shape_round0"]
    s0, y0, k0, n0 = val_line(r0["stage"], r0["line"])
    i0 = list(s0).index(r0["step"])
    out.append(dict(key="five", label="5 shapes", color=FIVE, stage=0, x=np.r_[0, s], y=np.r_[y0[i0], y],
                    k=np.r_[k0[i0], k], n=np.r_[n0[i0], n]))
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


STAGE_OFF = [1, 14, 60]
STAGE_SPAN = [(0.5, 11.5), (13.5, 56.5), (59.5, 110.8)]
STAGE_NAME = ["5 shapes", "10 shapes", "27 shapes (26 letters + near-T)"]
# picks that went to a fresh-scene test: (line key, within-line x); rounds from typed_in_numbers.json "picked_rounds"
_PR = TYPED["picked_rounds"]["values"]
PICKS = [("five", _PR["5-shape round 4"][2]), ("five", _PR["5-shape round 7"][2]),
         ("ten96", 21 + _PR["10-shape 96 px"][2]), ("ten192", 21 + _PR["10-shape 192 px"][2]),
         ("m96", _PR["26-letter 96 px (round 44)"][2]), ("m192", _PR["26-letter 192 px (round 2)"][2])]
MAIN_ROUND = _PR["26-letter 96 px round 50 (main model)"][2]


def main_model_rate():
    """Round-50 student (the 96 px model) on the 27 trained shapes, D054 fresh-scene test."""
    return rate("D054", "26-letter 96 px round 50", TRAINED27)[0]


def original_stage_levels():
    """Level of the original policy for each stage's band on the validation-success axis: [(percent, validated)].
    5 and 10 shapes: the original on the same 50 validation scenes per shape (typed_in_numbers.json: 41 of 250 and
    41 of 500). 27 shapes: the original was never run on those validation scenes, so the only number is its D051
    fresh-scene test (3.7%), drawn and labelled differently."""
    o5, o10 = TYPED["original_on_teacher_validation_5shapes"], TYPED["original_on_validation_10shapes"]
    return [(100 * o5["successes"] / o5["of"], True), (100 * o10["successes"] / o10["of"], True),
            (rate("D051", "original", TRAINED27)[0], False)]


ORIG_VAL_LS, ORIG_TEST_LS = (0, (4, 3)), (0, (1, 2.2))      # dashed: validation; dotted: fresh test
STAR = dict(marker="*", ms=17, mfc=M96, mec=INK, mew=1.1, ls="none", zorder=6)


def fig_curves_all_stages(smooth=False):
    lines = story_lines()
    sl = slide()
    fig, ax = plt.subplots(figsize=FULL)
    top = 52
    for i, (a, b) in enumerate(STAGE_SPAN):
        ax.axvspan(a, b, color=BAND if i % 2 == 0 else "#fbfbf9", zorder=0, lw=0)
        if sl:      # stage names above the plot, so the top of each band is free for labels
            ax.text((a + b) / 2, 1.015, ["5 shapes", "10 shapes", "26 letters + near-T"][i], ha="center",
                    va="bottom", fontweight="bold", color=INK, transform=ax.get_xaxis_transform(), clip_on=False)
        else:
            ax.text((a + b) / 2, top - 0.5, STAGE_NAME[i], ha="center", va="top", fontsize=14, fontweight="bold",
                    color=INK)
    orig = original_stage_levels()
    for i, (a, b) in enumerate(STAGE_SPAN):
        v, validated = orig[i]
        ax.plot([a + 0.3, b - 0.3], [v] * 2, color=ORIG_LINE, lw=1.6 if validated else 2.0,
                ls=ORIG_VAL_LS if validated else ORIG_TEST_LS, zorder=1)
    (o5, _), (o10, _), (o27, _) = orig
    if sl:
        ax.text(STAGE_SPAN[1][1] - 0.4, o10 - 0.9, "original", ha="right", va="top", color=INK2)
        ax.text(STAGE_SPAN[2][1] - 0.2, o27 + 0.9, "original: fresh test", ha="right", va="bottom", color=INK2)
    else:
        ax.text(STAGE_SPAN[0][0] + 0.6, o5 - 0.8, f"{o5:.1f}%", ha="left", va="top", fontsize=12, color=INK2)
        ax.text(STAGE_SPAN[1][1] - 0.5, o10 + 0.8, f"original (validation): {o10:.1f}%", ha="right", va="bottom",
                fontsize=12, color=INK2)
        ax.text(STAGE_SPAN[2][1] - 0.5, o27 + 0.8, f"original: {o27:.1f}% on fresh test\nscenes (no validation run)",
                ha="right", va="bottom", fontsize=12, color=INK2)
    # sub-phases of stage 2: executed-push labels | intended-plan labels | 96 px main line and 192 px side line
    for cx in ((16, 20.5) if not sl else (16,)):
        ax.axvline(STAGE_OFF[1] + cx, color=AXIS if not sl else INK2, lw=0.9 if not sl else 1.0,
                   ymin=0 if not sl else 0.2, ymax=0.9 if not sl else 1, zorder=1)
    if not sl:
        for cx, y, t in ((7.5, 1.0, "executed-push\nlabels"), (18.3, 12.5, "plan\nlabels"),
                         (32, 1.0, "96 px and\n192 px lines")):
            ax.text(STAGE_OFF[1] + cx, y, t, ha="center", va="bottom", fontsize=12, color=INK2)
    bx = STAGE_OFF[2] + 13.5
    ax.axvline(bx, color=AXIS if not sl else INK2, lw=0.9 if not sl else 1.0, ymin=0 if not sl else 0.22,
               ymax=0.9 if not sl else 1, zorder=1)
    if not sl:
        ax.text(bx + 0.8, 37.5, "larger training\nbudget per round", ha="left", va="bottom", fontsize=12, color=INK2)
    for L in lines:
        pos = STAGE_OFF[L["stage"]] + L["x"]
        if smooth:
            ax.plot(pos, L["y"], color=L["color"], lw=1.2, alpha=0.35, zorder=2)
            w = 3 if len(L["y"]) <= 11 else 5
            ax.plot(pos, rolling(L["y"], w), color=L["color"], lw=2.8, zorder=3, label=L["label"])
        else:
            ax.plot(pos, L["y"], color=L["color"], lw=1.6 if sl else 2.0, marker="o", ms=2.6 if sl else 3.6, zorder=3,
                    label=L["label"])
    byk = {L["key"]: L for L in lines}
    if not smooth:
        for key, x in PICKS:
            L = byk[key]
            i = int(np.where(L["x"] == x)[0][0])
            ax.plot(STAGE_OFF[L["stage"]] + x, L["y"][i], marker="o", ms=9 if sl else 11, mfc="none", mec=INK,
                    mew=1.6, zorder=5)
    # round 50: the 96 px model; never validated (the line stopped after training it), so its fresh-scene test is shown
    xm, ym = STAGE_OFF[2] + MAIN_ROUND, main_model_rate()
    ax.plot(xm, ym, **(STAR | ({"ms": 14} if sl else {})))
    if not sl:
        ax.annotate(f"round {MAIN_ROUND}, fresh test {ym:.1f}%\n(the model in the videos; never validated)",
                    (xm, ym), xytext=(xm - 1.5, 44.5), ha="right", va="bottom", fontsize=12, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkB=7))
    if not smooth and not sl:
        m96 = byk["m96"]
        ib = int(np.argmax(m96["y"]))
        xb, yb = STAGE_OFF[2] + m96["x"][ib], m96["y"][ib]
        ax.annotate(f"round {int(m96['x'][ib])}: most validation successes;\nthe final-test pick was fixed by rule "
                    f"at round {_PR['26-letter 96 px (round 44)'][2]}", (xb, yb), xytext=(xb + 1, 20.5), ha="right",
                    va="top", fontsize=11.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                                                         shrinkB=4))
    ax.set_xlim(0, STAGE_SPAN[2][1] + 0.5)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(PCT)
    if sl:
        ax.set_yticks([0, 10, 20, 30, 40, 50])
    ticks, labels = [], []
    for st, vals in ((0, [0, 5] if sl else [0, 5, 10]), (1, [0, 20, 40] if sl else [0, 10, 20, 30, 40]),
                     (2, [0, 25, 50] if sl else [0, 10, 20, 30, 40, 50])):
        for v in vals:
            ticks.append(STAGE_OFF[st] + v)
            labels.append(str(v))
    ax.set_xticks(ticks, labels)
    ax.grid(False, axis="x")
    ax.set_xlabel("Correction rounds completed (within each stage)")
    ax.set_ylabel("Validation success")
    if sl:
        # direct labels (data coordinates; 16 pt text) instead of a line legend
        t = dict(color=INK2, clip_on=False)
        sw = STAGE_OFF[1] + 16                                     # label switch inside the 10-shape stage
        ax.text(sw - 1.2, top - 0.6, "executed", ha="right", va="top", **t)
        ax.text(sw + 1.2, top - 0.6, "plan labels", ha="left", va="top", **t)
        L = byk["ten192"]
        ax.text(STAGE_OFF[1] + L["x"][-1] + 1.0, L["y"][-1] + 0.4, "192 px", ha="left", va="bottom", **t)
        L = byk["ten96"]
        ax.text(STAGE_OFF[1] + L["x"][-1], min(L["y"][-6:]) - 2.5, "96 px", ha="right", va="top", **t)
        ax.text(bx + 1.2, top - 0.6, "larger budget", ha="left", va="top", **t)
        L = byk["m192"]
        ax.text((STAGE_SPAN[2][0] + bx) / 2 - 0.4, max(L["y"]) + 2.2, "192 px", ha="center", va="bottom", **t)
        L = byk["m96"]
        j = int(np.argmin(np.abs(L["x"] - 33)))
        ax.text(STAGE_OFF[2] + 33, min(L["y"][j - 3:j + 4]) - 2.5, "96 px", ha="center", va="top", **t)
        handles = [Line2D([], [], marker="o", ms=11, mfc="none", mec=INK, mew=1.6, ls="none",
                          label="sent to a fresh test"),
                   Line2D([], [], **(STAR | {"ms": 14}), label=f"round {MAIN_ROUND}, fresh test {ym:.1f}%")]
        slide_legend(fig, handles, [h.get_label() for h in handles])
        return fig
    handles = [Line2D([], [], color=FIVE, lw=2.4, label="5 shapes"),
               Line2D([], [], color=tint(TEN, 0.45), lw=2.4, label="10 shapes, executed-push labels"),
               Line2D([], [], color=TEN, lw=2.4, label="10 shapes, plan labels, 96 px"),
               Line2D([], [], color=TEN192, lw=2.4, label="10 shapes, 192 px"),
               Line2D([], [], color=M96, lw=2.4, label="26 letters + near-T, 96 px"),
               Line2D([], [], color=M192, lw=2.4, label="26 letters + near-T, 192 px")]
    if not smooth:
        handles.append(Line2D([], [], marker="o", ms=10, mfc="none", mec=INK, mew=1.6, ls="none",
                              label="sent to a fresh-scene test"))
    handles.append(Line2D([], [], **(STAR | {"ms": 13}), label=f"round {MAIN_ROUND}, fresh test {ym:.1f}%"))
    legend_top(ax, handles=handles, ncol=4 if not smooth else 3, fontsize=12, columnspacing=1.0)
    note(fig, "Validation: 50 fixed scenes per shape, averaged over the stage's own shapes; x = correction rounds completed "
              "in that stage (round 0 = the student before any correction round; for 5 shapes, the copy-the-teacher "
              "student the rounds started from). Dashed: the original policy on the same validation scenes (5 shapes: "
              "41 of 250; 10 shapes: 41 of 500). The original was never run on the 27-shape validation scenes, so the "
              "dotted level there is its fresh-scene final test (D051). Round 50 was never validated either: the star is "
              "its own fresh-scene test (D054, a different scene pool).")
    return fig


def fig_curves_all_stages_panels():
    lines = story_lines()
    fig, axes = plt.subplots(1, 3, figsize=FULL, sharey=True, gridspec_kw={"width_ratios": [15, 43, 52], "wspace": 0.06})
    orig = original_stage_levels()
    byk = {L["key"]: L for L in lines}
    for st, ax in enumerate(axes):
        for L in [L for L in lines if L["stage"] == st]:
            ax.plot(L["x"], L["y"], color=L["color"], lw=2.0, marker="o", ms=3.4, label=L["label"])
        v, validated = orig[st]
        ax.axhline(v, color=ORIG_LINE, lw=1.6 if validated else 2.0, ls=ORIG_VAL_LS if validated else ORIG_TEST_LS)
        ax.set_title(STAGE_NAME[st].replace(" (26 letters + near-T)", ""), loc="left", fontsize=15)
        ax.grid(False, axis="x")
    for key, x in PICKS:
        L = byk[key]
        i = int(np.where(L["x"] == x)[0][0])
        axes[L["stage"]].plot(x, L["y"][i], marker="o", ms=10, mfc="none", mec=INK, mew=1.5, zorder=5)
    axes[0].set_xticks([0, 5, 10])
    axes[0].set_xlim(-0.7, 10.7)
    axes[1].set_xticks([0, 10, 20, 30, 40])
    axes[1].set_xlim(-1, 43)
    axes[1].axvline(16, color=AXIS, lw=0.9)
    axes[1].axvline(20.5, color=AXIS, lw=0.9)
    axes[1].text(8, 2, "executed-push\nlabels", ha="center", fontsize=12, color=INK2)
    axes[1].text(31.5, 2, "96 px / 192 px lines", ha="center", fontsize=12, color=INK2)
    axes[2].set_xticks([0, 10, 20, 30, 40, 50])
    axes[2].set_xlim(-1, 51.5)
    axes[2].axvline(13.5, color=AXIS, lw=0.9)
    axes[2].text(14.3, 10, "larger budget\nper round", fontsize=12, color=INK2)
    axes[0].text(10.5, orig[0][0] - 0.8, f"{orig[0][0]:.1f}%", ha="right", va="top", fontsize=12, color=INK2)
    wb = dict(fc="white", ec="none", pad=0.6)
    axes[1].text(42.5, orig[1][0] + 0.8, f"original (validation) {orig[1][0]:.1f}%", ha="right", va="bottom",
                 fontsize=12, color=INK2, bbox=wb, zorder=3)
    axes[2].text(51, orig[2][0] + 0.8, f"original: {orig[2][0]:.1f}% fresh test (no validation run)", ha="right",
                 va="bottom", fontsize=12, color=INK2, bbox=wb, zorder=3)
    ym = main_model_rate()
    axes[2].plot(MAIN_ROUND, ym, **(STAR | {"ms": 15}))
    axes[0].set_ylim(0, 47)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success")
    fig.supxlabel("Correction rounds completed (within each stage)", fontsize=16, y=0.0)
    handles = [Line2D([], [], color=FIVE, lw=2.4, label="5 shapes"),
               Line2D([], [], color=tint(TEN, 0.45), lw=2.4, label="10 shapes, executed-push labels"),
               Line2D([], [], color=TEN, lw=2.4, label="10 shapes, plan labels / 96 px"),
               Line2D([], [], color=TEN192, lw=2.4, label="10 shapes, 192 px"),
               Line2D([], [], color=M96, lw=2.4, label="26 letters + near-T, 96 px"),
               Line2D([], [], color=M192, lw=2.4, label="26 letters + near-T, 192 px"),
               Line2D([], [], marker="o", ms=10, mfc="none", mec=INK, mew=1.5, ls="none",
                      label="sent to a fresh-scene test"),
               Line2D([], [], **(STAR | {"ms": 13}), label=f"round {MAIN_ROUND}, fresh test {ym:.1f}%")]
    axes[0].legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.1), fontsize=12)
    note(fig, "Dashed: the original policy on the same validation scenes (5 shapes 41 of 250, 10 shapes 41 of 500). "
              "Dotted: the original's fresh-scene final test (D051), since it was never run on the 27-shape validation "
              "scenes. The star is round 50's own fresh-scene test (D054); round 50 was never validated.")
    return fig


def fig_curves_26letters(smooth=False):
    fig, ax = plt.subplots(figsize=FULL)
    ax.axvspan(-0.6, 13.5, color=BAND, lw=0, zorder=0)
    if slide():
        ax.text(6.5, 16, "20k steps", ha="center", va="bottom", fontsize=16, color=INK2)
        ax.text(14.2, 16, "54k steps per round", ha="left", va="bottom", fontsize=16, color=INK2)
    else:
        ax.text(6.5, 45.5, "old budget:\n20k training steps per round", ha="center", va="top", fontsize=12, color=INK2)
        ax.text(14.2, 45.5, "54k steps per round + 60 practice scenes per new letter (from round 14)", ha="left",
                va="top", fontsize=12, color=INK2)
    for key, line, c in (("m96", "96 px line", M96), ("m192", "192 px line", M192)):
        s, y, k, n = val_line("27 shapes", line)
        x = s - 1
        lo, hi = wilson_band(k, n)
        ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
        if smooth:
            ax.plot(x, y, color=c, lw=1.2, alpha=0.4)
            ax.plot(x, rolling(y, 5), color=c, lw=2.8, label=line_lab(key))
        else:
            ax.plot(x, y, color=c, lw=2.0, marker="o", ms=4, label=line_lab(key))
    for key, line, x, txt, dy in (("m96", "96 px line", _PR["26-letter 96 px (round 44)"][2],
                                   SL("picked for the final test\n(round 44, 34.8%)", "picked"), 5.5),
                                  ("m192", "192 px line", _PR["26-letter 192 px (round 2)"][2],
                                   SL("picked (round 2, 37.9%)", "picked"), 3.0)):
        s, y, *_ = val_line("27 shapes", line)
        v = y[list(s - 1).index(x)]
        ax.plot(x, v, marker="o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=5)
        ax.annotate(txt, (x, v), xytext=(x + (-2 if key == "m96" else 3), v + dy), fontsize=12, color=INK,
                    ha="right" if key == "m96" else "left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ym = main_model_rate()
    ax.plot(MAIN_ROUND, ym, **STAR)
    ax.annotate(SL(f"round {MAIN_ROUND}: the 96 px model\n{ym:.1f}% on fresh test scenes\n(not validated)",
                   f"round {MAIN_ROUND},\nfresh test\n{ym:.1f}%"), (MAIN_ROUND, ym),
                xytext=(MAIN_ROUND - 0.5, 19.5), ha="right", va="bottom", fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkB=7))
    ax.set_xlim(-0.6, 50.8)
    ax.set_ylim(15, 46)
    ax.yaxis.set_major_formatter(PCT)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(False, axis="x")
    ax.set_xlabel("Correction rounds completed")
    ax.set_ylabel("Validation success (27 shapes)")
    if slide():
        legend_top(ax)
    else:
        ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0))
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
        ax.set_title(line_lab(key).replace("26 letters + near-T", "26 letters"), loc="left", fontsize=15)
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
        ax.plot(s - 1, y, color=col(key), lw=2.0, marker="o", ms=4, label=line_lab(key))
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
    tk, tn = TYPED["teacher_totals"]["values"]["10 shapes"]
    tp = 100 * tk / tn
    for ax in axes:
        ax.axhline(tp, color=TEACHER, lw=1.8, ls=(0, (4, 3)))
        ax.grid(False, axis="x")
        ax.legend(loc="lower right", fontsize=12)
    axes[0].text(0, tp - 2, f"teacher (reads exact state): {tp:.0f}%", fontsize=12, color=INK2, va="top")
    axes[0].set_ylim(0, 82)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success (10 shapes)")
    note(fig, f"50 fixed validation scenes per shape. Teacher {tk} of {tn} on the same scenes (LEARNING_FINDINGS.md "
              "section 7; typed_in_numbers.json).")
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
    L5 = story_lines()[0]          # 5-shape line with round 0 = the first-run student at 30k steps it started from
    ax.plot(L5["x"], L5["y"], color=FIVE, marker="o", ms=5, label="correction rounds")
    ax.plot(L5["x"][0], L5["y"][0], marker="o", ms=7, mfc="white", mec=FIVE, mew=1.6, zorder=4)
    axes[0].plot(TYPED["five_shape_round0"]["step"] / 1000, L5["y"][0], marker="o", ms=7, mfc="white", mec=FIVE,
                 mew=1.6, zorder=4)
    for _, x in [p for p in PICKS if p[0] == "five"]:
        ax.plot(x, L5["y"][list(L5["x"]).index(x)], marker="o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=5)
    ax.plot([], [], marker="o", ms=10, mfc="none", mec=INK, mew=1.6, ls="none", label="sent to fresh-scene test")
    ax.set_xlabel("Correction rounds completed")
    ax.set_title("With correction rounds", loc="left", fontsize=15)
    ax.set_xticks(range(0, 11))
    ax.set_xlim(-0.5, 10.5)
    ax.annotate("start: first run\nat 30k steps", (0, L5["y"][0]), xytext=(0.3, L5["y"][0] - 9), fontsize=12,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkB=4))
    # TYPED-IN (typed_in_numbers.json): DECISIONS.md D046 (original on the teacher's 50 validation scenes: 20+17+4+0+0
    # of 250) and LEARNING_FINDINGS.md section 6 (the teacher that labelled these students: exact state read with 1 px /
    # 1 degree noise, 34+22+46+50+39 = 191 of 250 on the same scenes).
    o5 = TYPED["original_on_teacher_validation_5shapes"]; t5 = TYPED["teacher_noisy_planner_5shapes"]
    o5p, t5p = o5["successes"] / o5["of"] * 100, t5["successes"] / t5["of"] * 100
    for ax in axes:
        ax.axhline(o5p, color=ORIG_LINE, lw=1.6, ls=(0, (4, 3)))
        ax.axhline(t5p, color=TEACHER, lw=1.8, ls=(0, (4, 3)))
        ax.grid(False, axis="x")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.89), fontsize=12)
    axes[0].text(152, t5p - 1.5, f"teacher (exact state + 1 px noise) {t5p:.1f}%", fontsize=12, color=INK2, ha="right",
                 va="top")
    axes[0].text(152, o5p - 1.2, f"original {o5p:.1f}%", fontsize=12, color=INK2, ha="right", va="top")
    axes[0].set_ylim(0, 82)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Validation success (5 shapes)")
    note(fig, "T, near-T, Y, D, L; 50 fixed validation scenes per shape. Original and teacher on the same scenes "
              f"({o5['successes']} and {t5['successes']} of 250; DECISIONS.md D046, LEARNING_FINDINGS.md section 6). The "
              "teacher here is the one that labelled these students, not the first exact-state planner (124 of 250).")
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
                 colors=None, hatches=None, label_keys=None):
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
        if labels and (label_keys is None or k in label_keys):
            tops = np.asarray(hi[j]) if hi is not None else np.asarray(values[j])
            f = "{:.0f}" if slide() and nk * len(groups) > 6 else fmt
            for xi, t, v in zip(x, tops, values[j]):
                ax.text(xi, t + 0.6, f.format(v), ha="center", va="bottom", fontsize=size, color=INK)
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
        xs = grouped_bars(ax, [GROUP_LABEL[g] for g in groups], keys, vals, los, his, labels=not slide())
        if slide():     # one-decimal labels on the original and the 96 px bars, clear of the 192 px interval
            w = 0.8 / len(keys)
            for j, k in enumerate(keys):
                if k not in ("original", "m96"):
                    continue
                for gi in range(len(groups)):
                    top = max(his[j][gi], his[j + 1][gi]) if k == "m96" else his[j][gi]
                    ax.text(xs[gi] - 0.4 + w * (j + 0.5), top + 0.6, f"{vals[j][gi]:.1f}", ha="center",
                            va="bottom", color=INK, fontweight="bold" if k == "m96" else None)
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
    rows = [(SL("5 shapes\n(round 7)", "5 shapes"), "D047", "5-shape round 7", "5 first shapes", "five"),
            (SL("10 shapes\n(96 px)", "10 shapes"), "D049/D050", "10-shape 96 px", "10 earlier shapes", "ten96"),
            (SL("27 shapes\n(26 letters, 96 px)", "27 shapes"), "D051", "26-letter 96 px (round 44)", "27 trained shapes",
             "m96")]
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
    ax.set_ylabel(SL("Success on the stage's shapes\n(fresh-scene final test)", "Success"))
    legend_top(ax, handles=[patch("original", SL("Original (T demos only)", "Original")), patch("five"), patch("ten96"),
                            patch("m96")], ncol=4, fontsize=12)
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
    ax.set_xticks(xs, stagger([SHORT.get(s, s) for s in order]), fontsize=13)
    ax.axvline(len(TRAINED27) - 0.5, color=INK2, lw=1)
    ax.text(len(TRAINED27) - 0.2, 57, SL("never seen\nin training", "unseen"), fontsize=12, color=INK2, va="top")
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
    if vertical or slide():      # slide version: shapes along x, 6.3 x 3.6 in (fits height=0.8\textheight)
        fig, ax = plt.subplots(figsize=FULL)
        variant_size(fig, 6.3, 3.6)
        xs = np.arange(len(order))
        ax.vlines(xs, a, b, color=tint(col(key), 0.5), lw=2.2 if not slide() else 2.0, zorder=2)
        ax.scatter(xs, a, s=55 if not slide() else 30, color=ORIG_LINE, zorder=3, label=lab("original"),
                   edgecolor="white", lw=1.5 if not slide() else 0.8)
        ax.scatter(xs, b, s=70 if not slide() else 40, color=col(key), zorder=4, label=lab(key), edgecolor="white",
                   lw=1.5 if not slide() else 0.8)
        ax.set_xticks(xs, stagger([SHORT.get(s, s) for s in order]), fontsize=13)
        if slide():
            ax.tick_params(axis="x", length=0, pad=3)
            for t, s_ in zip(ax.get_xticklabels(), order):
                if s_ in ("T", "T_narrow"):
                    t.set_fontweight("bold")
                    t.set_color(INK)
        ax.axvline(len(TRAINED27) - 0.5, color=INK2, lw=1)
        ax.text(len(TRAINED27) - 0.2, 58, SL("never seen\nin training", "never\nseen"), fontsize=12, color=INK2,
                va="top")
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
    ax.set_xlabel(SL("Change in success vs. the original (points), same scenes, 95% interval",
                     "Change vs. original (points)"))
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
    fig, ax, axp = bars_with_pooled(OLD10, keys, _getter("D049/D050", D049, OLD10), SL("All 10\nshapes", "All 10"),
                                    ratio=2.4 if not slide() else 3.6 if not half() else 4.6)
    ax.set_xticks(range(len(OLD10)), [disp(s) if not slide() else SHORT.get(s, s) for s in OLD10])
    ax.set_ylim(0, 65)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success\n(final test, 1000 fresh scenes per shape)")
    r96, r192 = _PR["10-shape 96 px"][2], _PR["10-shape 192 px"][2]
    legend_top(ax, handles=[patch("original"), patch("ten96", SL(f"Ours, 10 shapes, 96 px (round {r96})",
                                                                 f"Ours, 96 px (round {r96})")),
                            patch("ten192", SL(f"Ours, 10 shapes, 192 px (round {r192})", f"Ours, 192 px (round {r192})"))],
               ncol=3, fontsize=12)
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
    note(fig, "Early attempt (D040): blending a planner-copying student's weights into the original. Small gains on Y and "
              "D, a loss on T. The later blend tests (D041-D043) are in final_blends_all.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# shapes_*: per-shape comparisons
# ----------------------------------------------------------------------------------------------------------------------


def fig_shapes_96_vs_192():
    fig, ax = plt.subplots(figsize=(7, 6.2))
    lim = 62
    skipped = shape_scatter(ax, lambda s: rate("D051", D051["m96"], [s])[0],
                            lambda s: rate("D051", D051["m192"], [s])[0], M192, lim)
    ax.set_xlabel(f"{lab('m96')}: success")
    ax.set_ylabel(f"{lab('m192')}: success")
    note(fig, f"Each point is one shape; {TEST_NOTE}. Above the line: the sharper camera helps."
         + (" Unlabelled: " + ", ".join(disp(s) for s in skipped) + "." if skipped else ""))
    return fig


def fig_shapes_10_vs_26():
    fig, ax = plt.subplots(figsize=(7, 6.2))
    lim = 62
    groups = ((OLD10, "o", "10 earlier shapes (both trained on them)", 0.45, True),
              (NEW17, "s", "17 newer letters (only the 27-shape policy)", 0.0, True),
              (HELD, "^", "6 never-seen shapes", 0.0, False))
    skipped = shape_scatter(ax, lambda s: rate("D051", D051["ten96"], [s])[0],
                            lambda s: rate("D051", D051["m96"], [s])[0], M96, lim, legend_loc="lower right",
                            groups=groups)
    ax.set_xlabel("Ours, 10 shapes: success")
    ax.set_ylabel(f"{lab('m96')}: success")
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


# TYPED-IN (teacher_validation.csv): the planner teacher on the 50 validation scenes per shape for all 27 trained shapes,
# with the world-model version of each check (wm2 v3 for the 17 new letters and T, D, L; v2 for E, F, J, P, U; v1 for
# near-T and Y). LEARNING_FINDINGS.md sections 6 and 7.
TEACHER_VAL = {r["shape"]: (int(r["teacher_successes"]), int(r["of"]), r["world_model_version"])
               for r in read("teacher_validation.csv")}
assert sorted(TEACHER_VAL) == sorted(TRAINED27)
# TYPED-IN: LEARNING_FINDINGS.md section 5 table "Final planner (50)": first sampling-planner teacher, 50 validation
# scenes per letter; and the original policy on the 1000-scene final-test pool (same section, last column).
TEACHER5 = TYPED["teacher5_successes_of_50"]["values"]
TEACHER5_NOISY = TYPED["teacher_noisy_planner_5shapes"]["values"]
ORIG5_FINALPOOL = TYPED["original_5shapes_final_pool_percent"]["values"]
STUDENT_VAL_ROUND = _PR["26-letter 96 px (round 44)"][2]       # picked round; round 50 was never validated


def student_val(shapes, rnd=None):
    """26-letter 96 px student on the same 50 validation scenes per shape as the teacher: {shape: (successes, n)}."""
    step = (STUDENT_VAL_ROUND if rnd is None else rnd) + 1      # validation step k evaluates the round k-1 student
    out = {}
    for r in VAL:
        if r["stage"] == "27 shapes" and r["line"] == "96 px line" and int(r["step"]) == step and r["shape"] in shapes:
            out[r["shape"]] = (int(r["successes"]), int(r["episodes"]))
    return out


def fig_shapes_teacher_vs_student():
    shapes = TRAINED27
    sv = student_val(shapes)
    x = np.array([100 * TEACHER_VAL[s][0] / TEACHER_VAL[s][1] for s in shapes])
    y = np.array([100 * sv[s][0] / sv[s][1] for s in shapes])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1.2)
    ax.text(70, 74, "student = teacher", ha="right", va="bottom", fontsize=12, color=INK2)
    for grp, mk in ((OLD10, "o"), (NEW17, "s")):
        m = np.isin(shapes, grp)
        ax.scatter(x[m], y[m], s=75, marker=mk, color=M96 if mk == "s" else tint(M96, 0.45), edgecolor="white", lw=1.4,
                   zorder=3, label="17 newer letters" if mk == "s" else "10 earlier shapes")
    kt = sum(TEACHER_VAL[s][0] for s in shapes); nt = sum(TEACHER_VAL[s][1] for s in shapes)
    ks = sum(sv[s][0] for s in shapes); ns = sum(sv[s][1] for s in shapes)
    ax.text(3, 97, f"the student keeps {100 * (ks / ns) / (kt / nt):.0f}% of the teacher's successes\n"
                   f"({ks} vs {kt} of {nt}, same 50 scenes per shape)", fontsize=12, color=INK2, va="top")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="lower right", fontsize=12, frameon=True, facecolor="white", edgecolor="none")
    place_labels(ax, shapes, x, y)
    ax.set_xlabel("Planning teacher, reads exact state\n(50 validation scenes per shape)")
    ax.set_ylabel(f"Camera student, 27 shapes, 96 px, round {STUDENT_VAL_ROUND}\n(same 50 scenes)")
    note(fig, "Teacher from teacher_validation.csv (LEARNING_FINDINGS.md sections 6-7; world model v1-v3 by shape); "
              f"student = the round-{STUDENT_VAL_ROUND} pick, the best validation round, so it is slightly flattered.")
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
                                            "m96": "Ours, 27 shapes, 96 px, round 44 (trained on it)"}[key])
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
    ax.set_xticks(range(len(keys)), ["Original", "Ours,\n10 shapes", "Ours, 96 px\n(round 44)", "Ours, 192 px\n(round 2)"],
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
    if half():
        fig, ax0 = plt.subplots(figsize=HALF)
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"wspace": 0.32})
    xs = [1, 10, 27]
    xd = 27.9 if not slide() else 29.8          # the 192 px student (also 27 shapes), drawn just right of 27
    for ax, metric in zip(axes, ("success", "coverage")):
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
        ax.scatter([xd], [r], s=110, color=M192, edgecolor="white", lw=1.5, zorder=4, marker="D")
        if lo is not None:
            errbars(ax, [xd], [r], [lo], [hi])
        for x, v in zip(xs + [xd], y + [r]):
            if x in (1, 10):        # below-right, off the rising line
                ax.text(x + 1.0, v - (0.25 if metric == "success" else 0.6), f"{v:.1f}", fontsize=12, ha="left",
                        va="top", color=INK)
            else:
                ax.text(x + (-1.2 if x == 27 else 0.8), v + (0.4 if metric == "success" else 0.8), f"{v:.1f}",
                        fontsize=12, ha="right" if x == 27 else "left", va="bottom", color=INK)
        ax.set_xticks(xs, [SL("1\n(original:\nT only)", "1"), "10", "27"])
        ax.set_xlim(-1, 31) if not slide() else ax.set_xlim(-2, 36)
        ax.grid(False, axis="x")
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("Shapes in training")
    axes[0].set_ylim(0, 8)
    axes[0].set_yticks([0, 2, 4, 6, 8])
    axes[0].set_ylabel(SL("Success on 6 never-seen shapes", "Success, 6 unseen"))
    if not half():
        axes[1].set_ylim(40, 62)
        axes[1].yaxis.set_major_locator(MultipleLocator(5))
        axes[1].set_ylabel(SL("Mean final goal coverage", "Final coverage"))
    handles = [Line2D([], [], marker="o", color=ORIG_LINE, ls="none", ms=10, label="Original"),
               Line2D([], [], marker="o", color=TEN, ls="none", ms=10, label=lab("ten96")),
               Line2D([], [], marker="o", color=M96, ls="none", ms=10, label=lab("m96")),
               Line2D([], [], marker="D", color=M192, ls="none", ms=9, label=lab("m192"))]
    legend_top(axes[0], handles=handles, ncol=4, fontsize=12)
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
    fig, (ax, axp) = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={
        "width_ratios": [17, 3.2 if not slide() else 4.6 if not half() else 5.6], "wspace": 0.03})
    vals = [[g(k, s)[0] for s in order] for k in keys]
    grouped_bars(ax, order, keys, vals, labels=False, width=0.84)      # per letter: no interval (200 scenes each)
    if half():
        ax.set_xticks(range(len(order)), stagger(order))
    pooled = [g(k, None) for k in keys]
    grouped_bars(axp, ["All 17"], keys, [[p[0]] for p in pooled], [[p[1]] for p in pooled], [[p[2]] for p in pooled],
                 labels=True, width=0.86, size=12)
    if half():
        round_labels(axp)
    axp.spines["left"].set_visible(False)
    axp.tick_params(axis="y", left=False)
    axp.set_xlim(-0.5, 0.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 45)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel(SL("Success on letters never trained on\n(200 fresh scenes per letter)",
                     "Success, unseen letters" if not half() else "Success"))
    r96, r192 = _PR["10-shape 96 px"][2], _PR["10-shape 192 px"][2]
    legend_top(ax, handles=[patch("original"), patch("ten96", (f"Ours, 10 shapes, 96 px (round {r96})" if not slide()
                                                               or half() else f"Ours, 96 px (round {r96})")),
                            patch("ten192", (f"Ours, 10 shapes, 192 px (round {r192})" if not slide() or half() else
                                             f"Ours, 192 px (round {r192})"))], ncol=3, fontsize=12)
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
    if half():
        fig, ax0 = plt.subplots(figsize=HALF)
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=FULL, sharey=True, gridspec_kw={"wspace": 0.16})
    keys = KEYS4[::-1]
    for ax, shapes, title in zip(axes, (TRAINED27, HELD), (SL("27 trained shapes", "27 trained"),
                                                           SL("6 never-seen shapes", "6 unseen"))):
        for i, k in enumerate(keys):
            parts = _outcome_shares("D051", D051[k], shapes)
            c = col(k) if k != "original" else ORIG_LINE
            colors = [c, tint(c, 0.5), tint(c, 0.78), "#efeee9"]
            left = 0
            for j, (v, cc) in enumerate(zip(parts, colors)):
                ax.barh(i, v, left=left, color=cc, edgecolor="white", lw=1.5, height=0.7, zorder=3)
                if v >= (6 if not slide() else 14 if not half() else 16):
                    ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=12,
                            color="white" if j == 0 and k != "original" else INK)
                left += v
        ax.set_title(title, loc="left", fontsize=15)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PCT)
        ax.grid(False)
        ax.set_xlabel("Share of episodes")
    nm = ({"original": "Original", "ten96": "Ours, 10 shapes", "m96": "Ours, 27 shapes,\n96 px (round 44)",
           "m192": "Ours, 27 shapes,\n192 px (round 2)"} if not slide() else
          {"original": "Original", "ten96": "10 shapes", "m96": "96 px\n(round 44)", "m192": "192 px\n(round 2)"})
    axes[0].set_yticks(range(len(keys)), [nm[k] for k in keys], fontsize=12)
    shades = [Patch(facecolor=M96, label=SL("solved (>95% coverage)", "solved")),
              Patch(facecolor=tint(M96, 0.5), label=SL("near miss: peak 90-95%", "peak 90-95%")),
              Patch(facecolor=tint(M96, 0.78), label="peak 80-90%"),
              Patch(facecolor="#efeee9", label=SL("peak below 80%", "below 80%"))]
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
    if half():
        groups = groups[2:]
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
    for gi, g in enumerate(groups if not slide() else []):
        for j, m in enumerate(models[1:], start=1):
            d = paired_row("D056", "execute 8 (trained)", m, g)
            if d["model_rate"] < 6:
                continue
            ax.text(gi - 0.4 + 0.8 / 3 * (j + 0.5), 1.0, f"{d['difference']:+.1f}".replace("-", "−"), ha="center",
                    va="bottom", fontsize=11, color="white" if j == 1 else INK, zorder=5)
    ax.set_ylim(0, 42)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success\n(200 fresh scenes per shape)")
    legend_top(ax, handles=[patch(BLUE_RAMP[2], SL("execute 8 of 16 planned steps (as trained)", "execute 8 (trained)")),
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
    elif slide():   # rates in a column right of the plot
        for i, (_, ser) in enumerate(rows):
            d = ser[0]
            ax.text(1.04, i, f"{d['base_rate']:.1f} → {d['model_rate']:.1f}%", va="center", ha="left",
                    transform=ax.get_yaxis_transform(), fontweight="bold" if i == 2 else None)
        ax.text(1.04, -0.75, "round 44 → 50", va="bottom", ha="left", color=INK2, transform=ax.get_yaxis_transform())
        ax.get_yticklabels()[2].set_fontweight("bold")
        ax.set_xlim(-4, 6)
        ax.set_ylim(len(rows) - 0.5, -0.9)
        ax.set_xticks([-4, 0, 4] if half() else [-4, -2, 0, 2, 4, 6])
    else:
        for i, (_, ser) in enumerate(rows):
            d = ser[0]
            ax.text(d["ci95_high"] + 0.3, i, f"{d['base_rate']:.1f}% → {d['model_rate']:.1f}%   "
                                              f"({d['difference']:+.1f})".replace("-", "−"), va="center", fontsize=12)
        ax.set_xlim(-6, 12)
    ax.set_xlabel(SL("Round 50 minus round 44 (points), same scenes, 95% interval", "Round 50 − round 44 (points)"))
    note(fig, "26-letter 96 px student after six more correction rounds; 200 fresh scenes per shape.")
    return fig


def fig_analysis_t_tradeoff(diff=False):
    entries = [("5 shapes, round 4", "D046", "5-shape round 4", "five"),
               ("5 shapes, round 7", "D047", "5-shape round 7", "five"),
               ("10 shapes", "D049/D050", "10-shape 96 px", "ten96"),
               ("10 shapes, 192 px", "D049/D050", "10-shape 192 px", "ten192"),
               (SL("27 shapes, 96 px (round 44)", "27 shapes (round 44)"), "D051", "26-letter 96 px (round 44)", "m96"),
               (SL("27 shapes, 192 px (round 2)", "27 shapes, 192 px"), "D051", "26-letter 192 px (round 2)", "m192")]
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
        fig.supxlabel(SL("Change vs. the original on the same scenes (points, 95% interval)", "Change vs. original"),
                      fontsize=16, y=None if slide() else 0.0)
    else:
        fig.supxlabel(SL("Fresh-scene success (95% interval)", "Fresh-scene success"), fontsize=16,
                      y=None if slide() else 0.0)
        legend_top(axes[0], handles=[Line2D([], [], marker="o", color=ORIG_LINE, ls="none", ms=8,
                                            label="original, same scenes"),
                                     Line2D([], [], marker="o", color=INK2, ls="none", ms=10,
                                            label="our student (colour = policy)")], ncol=2, fontsize=12, y=1.08)
    note(fig, "Each test has its own fresh scene pool: 1000 scenes per shape (5 and 10 shapes), 300 (27 shapes).")
    return fig


def _192_vs_96_slide():
    """Slide versions: group headings name the two picked rounds (the cameras' students also differ in training
    rounds: 10 shapes round 6 vs 13, 27 shapes round 2 vs 44)."""
    from matplotlib.transforms import blended_transform_factory
    r10 = (_PR["10-shape 192 px"][2], _PR["10-shape 96 px"][2])
    r27 = (_PR["26-letter 192 px (round 2)"][2], _PR["26-letter 96 px (round 44)"][2])
    groups = ["27 trained shapes", "6 held-out shapes"] if half() else \
        ["10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"]
    rows = [(0.0, "All 10", paired_eps("D049/D050", "10-shape 96 px", "10-shape 192 px", OLD10), TEN192)]
    y = 2.15
    for g in groups:
        rows.append((y, GROUP_LABEL[g], paired_eps("D051", D051["m96"], D051["m192"], GROUP_SHAPES[g]), M192))
        y += 1
    fig, ax = plt.subplots(figsize=FULL)
    for yy, name, d, c in rows:
        ax.plot([d["ci95_low"], d["ci95_high"]], [yy, yy], color=c, lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot(d["difference"], yy, "o", color=c, ms=9, mec="white", mew=1.4, zorder=4)
        if name.startswith("All") or half():
            ax.text(d["ci95_high"] + 0.3, yy, f"{d['difference']:+.1f}".replace("-", "−"), va="center",
                    fontweight="bold" if name.startswith("All") else None)
    ax.axvline(0, color=INK2, lw=1.1, zorder=2)
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
    ax.set_ylim(rows[-1][0] + 0.5, -1.1)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    px = "" if half() else "192 px "
    px96 = "" if half() else "96 px "
    wb = dict(fc="white", ec="none", pad=0.3)
    for yh, t, bold in ((-0.75, f"10 shapes: {px}round {r10[0]} vs {px96}round {r10[1]}", True),
                        (0.95, f"27 shapes: {px}round {r27[0]} vs {px96}round {r27[1]}", True),
                        (1.45, "(also differs in training rounds)", False)):
        ax.text(0.01, yh, t, transform=tr, ha="left", va="center", fontweight="bold" if bold else None,
                color=INK if bold else INK2, bbox=wb, zorder=5)
    ax.set_xlim(-3, 7)
    ax.set_xticks([-2, 0, 2, 4, 6])
    ax.xaxis.set_major_formatter(SIGNED)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("192 px minus 96 px (points)")
    return fig


def fig_analysis_192_vs_96(per_shape=False):
    if slide() and not per_shape:
        return _192_vs_96_slide()
    if not per_shape:
        rows = [("10-shape models:\nall 10 shapes", [paired_eps("D049/D050", "10-shape 96 px", "10-shape 192 px", OLD10),
                                                     None])]
        for g in ("10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"):
            rows.append((f"27-shape models:\n{GROUP_LABEL[g].lower()}" if g != "27 trained shapes" else
                         "27-shape models:\nall 27 trained shapes",
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
        note(fig, "10 shapes: 1000 paired scenes per shape. 27 shapes (26 letters + near-T): 300 paired scenes per "
                  "shape. Each pair compares the two cameras' picked students, which also differ in training rounds "
                  "(10 shapes: 192 px round 6 vs 96 px round 13; 27 shapes: 192 px round 2 vs 96 px round 44).")
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35})
    rows = [(disp(s), [paired_eps("D049/D050", "10-shape 96 px", "10-shape 192 px", [s])]) for s in OLD10]
    forest(axes[0], rows, [TEN192], ms=8)
    axes[0].set_title("10-shape models (round 6 vs 13)", loc="left", fontsize=15)
    order = OLD10 + sorted(NEW17) + HELD
    rows = [(disp(s), [paired_eps("D051", D051["m96"], D051["m192"], [s])]) for s in order]
    forest(axes[1], rows, [M192], ms=6)
    axes[1].set_title("27-shape models (round 2 vs 44)", loc="left", fontsize=15)
    axes[1].tick_params(axis="y", labelsize=11)
    axes[1].axhline(9.5, color=INK2, lw=1)
    axes[1].axhline(26.5, color=INK2, lw=1)
    for ax in axes:
        ax.set_xlabel("192 px minus 96 px (points)")
    return fig


# TYPED-IN: LEARNING_FINDINGS.md section 7 "What helps" table (successes of 1000: 100 validation scenes x 10 shapes).
LABEL_AB = TYPED["label_ab_successes_of_1000"]["values"]


LABEL_AB_SHORT = ["Intended-plan labels", "192 px camera", "All teacher demos", "990M network"]


def fig_analysis_label_ab():
    fig, ax = plt.subplots(figsize=FULL)
    ys = np.arange(len(LABEL_AB))
    h = 0.36
    for j, (lbl, c) in enumerate(((SL("without the change", "without"), tint(TEN, 0.6)),
                                  (SL("with the change", "with"), TEN))):
        v = [r[1 + j] / 10 for r in LABEL_AB]
        lo, hi = zip(*[wilson(r[1 + j], 1000) for r in LABEL_AB])
        y = ys - h / 2 + h * j
        ax.barh(y, v, h, color=c, edgecolor="white", lw=1.5, zorder=3, label=lbl)
        errbars(ax, y, v, lo, hi, horizontal=True)
        for i, (yi, vi, hh) in enumerate(zip(y, v, hi)):      # success rates, not relative gains
            ax.text(hh + 0.3, yi, f"{vi:.1f}%", va="center", fontsize=12,
                    fontweight="bold" if (i == 0 and j == 1) else None)
    ax.set_yticks(ys, [r[0] for r in LABEL_AB] if not slide() else LABEL_AB_SHORT if not half() else
                  ["Plan labels"] + LABEL_AB_SHORT[1:], fontsize=12.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlim(0, 21 if not slide() else 23 if not half() else 26)
    ax.xaxis.set_major_locator(MultipleLocator(4 if not slide() else 5 if not half() else 10))
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(SL("Validation success (100 scenes x 10 shapes = 1000 episodes), 95% interval",
                     "Validation success (1000 episodes)" if not half() else "Validation success,\n1000 episodes"))
    legend_top(ax, ncol=2, fontsize=12)
    note(fig, "Matched A/B tests, 30k steps from the official checkpoint (last row: 50k steps from scratch). Typed in "
              "from LEARNING_FINDINGS.md section 7.")
    return fig


def fig_analysis_budget():
    if half():      # the half-slide version shows only the validation panel
        fig, ax = plt.subplots(figsize=HALF)
    else:
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
    if half():
        phases = [("start", [1]), ("old budget", [11, 12, 13, 14]), ("larger", list(range(41, 51)))]
    groups = [(SL("10 earlier shapes", "10 earlier"), OLD10), (SL("17 newer letters", "17 newer"), NEW17)]
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
    ax.set_ylabel(SL("Validation success (26-letter, 96 px)", "Validation success"))
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
               ("27 shapes, 96 px (round 44)", "27 shapes", "96 px line", 45, "D051", "26-letter 96 px (round 44)",
                TRAINED27, "m96"),
               ("27 shapes, 192 px (round 2)", "27 shapes", "192 px line", 3, "D051", "26-letter 192 px (round 2)",
                TRAINED27, "m192")]
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
    ax.text(RL_T0 * 100 + (1 if not half() else -1), -0.75, SL(f"T success before RL: {RL_T0 * 100:.0f}%",
                                                               f"before: {RL_T0 * 100:.0f}%"),
            fontsize=12, color=INK2, va="bottom", ha="left" if not half() else "right")
    for i, (name, af, nft, upd) in enumerate(RL_LONG):
        for v, mk, off in ((af, "o", -0.12), (nft, "s", 0.12)):
            if v is None:
                continue
            ax.plot([RL_T0 * 100, v * 100], [i + off, i + off], color=tint(RL, 0.6), lw=2, zorder=2)
            ax.plot(v * 100, i + off, mk, color=RL, ms=10, mec="white", mew=1.4, zorder=3)
        # how many RL updates the run had made at its last evaluation (right of the plot, with a column header)
        xu = 1.03 if not slide() else 1.03 + (0.17 if not half() else 0.22)      # slide: numbers right-aligned
        ax.text(xu, i, f"after {upd} updates" if not slide() else upd, va="center",
                ha="left" if not slide() else "right", fontsize=12, color=INK2, transform=ax.get_yaxis_transform(),
                clip_on=False)
    if slide():
        ax.text(1.03, -0.75, "updates", ha="left", va="bottom", color=INK2, transform=ax.get_yaxis_transform(),
                clip_on=False)
    ax.set_yticks(ys, [r[0] for r in RL_LONG])
    ax.set_ylim(len(RL_LONG) - 0.4, -0.9)
    ax.set_xlim(-3, 103)
    ax.set_xticks(range(0, 101, 20) if not slide() else range(0, 101, 25) if not half() else range(0, 101, 50))
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel(SL("T success at the last evaluation", "T success after RL"))
    legend_top(ax, handles=[Line2D([], [], marker="o", color=RL, ls="none", ms=9,
                                   label=SL("AdvantageFlow, after RL", "AdvantageFlow")),
                            Line2D([], [], marker="s", color=RL, ls="none", ms=9, label=SL("NFT, after RL", "NFT"))],
               ncol=2, fontsize=12, y=1.06)
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
    for ax, j, title in ((axes[0], 1, "Share of the training score\nexplained by the plan itself" if not half() else
                          "Plan's share"), (axes[1], 2, "Split-half reliability\nof the score" if not half() else
                                            "Reliability")):
        v = [r[j] * 100 for r in RL_SNR]
        ax.bar(x, v, 0.6, color=RL, edgecolor="white", zorder=3)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}%", ha="center", fontsize=13)
        ax.set_xticks(x, [r[0].replace("near-T", "nT") if half() else r[0] for r in RL_SNR])
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
    w = 0.27
    v_o = [ORIG5_FINALPOOL[s] for s in shapes]
    series = [(TEACHER5, tint(TEACHER, 0.55), SL("First planner teacher, exact state (50 validation scenes)",
                                                  "first planner (exact state)")),
              (TEACHER5_NOISY, TEACHER, SL("Teacher reading the state with 1 px / 1° noise (same scenes)",
                                           "teacher, 1 px / 1° noise"))]
    ax.bar(x - w, v_o, w, color=ORIG, edgecolor="white", lw=1.5, zorder=3,
           label=SL("Original policy (1000 test scenes)", "original"))
    for xi, v in zip(x - w, v_o):
        ax.text(xi, v + 1, f"{v:.0f}" if slide() else f"{v:.1f}", ha="center", fontsize=12)
    if half():
        for t in ax.texts:
            t.set_visible(False)
    for j, (vals, c, lbl) in enumerate(series):
        v_t = [vals[s] * 2 for s in shapes]
        lo_t, hi_t = zip(*[wilson(vals[s], 50) for s in shapes])
        if slide():
            hi_t = v_t
        ax.bar(x + w * j, v_t, w, color=c, edgecolor="white", lw=1.5, zorder=3, label=lbl)
        if not slide():
            errbars(ax, x + w * j, v_t, lo_t, hi_t)
        for xi, v, h in zip(x + w * j, v_t, hi_t):
            if slide() and j == 0:
                continue
            ax.text(xi, h + 1, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold" if j == 1 else None)
    ax.set_xticks(x, [disp(s) for s in shapes])
    ax.grid(False, axis="x")
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Success")
    legend_top(ax, ncol=1, fontsize=12)
    note(fig, "Both teachers read the simulator's exact state and plan in the learned world model (training data only). "
              "Small state noise stopped the planner repeating the same nearly-right push (LEARNING_FINDINGS.md sections "
              "5-6; typed_in_numbers.json).")
    return fig


def _teacher_student_rows(shapes):
    sv = student_val(shapes)
    t = [(TEACHER_VAL[s][0], TEACHER_VAL[s][1]) for s in shapes]
    st = [sv[s] for s in shapes]
    return t, st


TEACHER_HATCH = {"wm2 v3": None, "wm2 v2": "//", "wm2 v1": ".."}


def fig_method_teacher_26():
    shapes = sorted(TRAINED27, key=lambda s: (-TEACHER_VAL[s][0], s))
    t, st = _teacher_student_rows(shapes)
    if slide():     # wide per-letter version: the letters get the full width, pooled rates go into the legend
        fig, ax = plt.subplots(figsize=FULL)
        axp = None
    else:
        fig, (ax, axp) = plt.subplots(1, 2, figsize=FULL, sharey=True,
                                      gridspec_kw={"width_ratios": [27, 2.6], "wspace": 0.03})
    x = np.arange(len(shapes))
    w = 0.4
    for i, s in enumerate(shapes):      # slide: no hatching (which world model checked the teacher: README)
        ax.bar(x[i] - w / 2, 100 * t[i][0] / t[i][1], w, color=TEACHER, edgecolor="white", lw=0.8, zorder=3,
               hatch=TEACHER_HATCH[TEACHER_VAL[s][2]] if not slide() else None)
    ax.bar(x + w / 2, [100 * k / n for k, n in st], w, color=M96, edgecolor="white", lw=0.8, zorder=3)
    ax.set_xticks(x, stagger([SHORT.get(s, s) if slide() else disp(s).replace("near-T", "nT") for s in shapes]),
                  fontsize=12 if not slide() else None)     # slide: two rows (a 16 pt W is wider than one slot)
    if slide():
        ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.6, len(shapes) - 0.4)
    ax.grid(False, axis="x")
    kt, nt = sum(a for a, _ in t), sum(b for _, b in t)
    ks, ns = sum(a for a, _ in st), sum(b for _, b in st)
    if axp is not None:
        for j, (k, n, c) in enumerate(((kt, nt, TEACHER), (ks, ns, M96))):
            v = 100 * k / n
            lo, hi = wilson(k, n)
            axp.bar(j, v, 0.8, color=c, edgecolor="white", zorder=3)
            errbars(axp, [j], [v], [lo], [hi])
            axp.text(j, hi + 1.5, f"{v:.0f}", ha="center", fontsize=13, fontweight="bold")
        axp.set_xticks([0.5], ["All 27"])
        axp.set_xlim(-0.6, 1.6)
        axp.spines["left"].set_visible(False)
        axp.tick_params(axis="y", left=False)
        axp.grid(False, axis="x")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PCT)
    if slide():
        ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Validation success\n(same 50 scenes per shape)" if not slide() else "Validation success")
    handles = [patch(TEACHER, SL("Planning teacher (reads exact state)",
                                 f"Teacher (exact state): {100 * kt / nt:.0f}% overall")),
               patch(M96, SL(f"Camera student, 27 shapes, 96 px (round {STUDENT_VAL_ROUND})",
                             f"Camera student (round {STUDENT_VAL_ROUND}): {100 * ks / ns:.0f}% overall"))]
    if not slide():
        handles += [Patch(facecolor=TEACHER, edgecolor="white", hatch="//", label="teacher checked with world model v2"),
                    Patch(facecolor=TEACHER, edgecolor="white", hatch="..", label="... with world model v1")]
    legend_top(ax, handles=handles, ncol=2, fontsize=12)
    note(fig, "Teacher: teacher_validation.csv (LEARNING_FINDINGS.md sections 6-7); plain bars = world model v3 (26-letter "
              "check), hatched = earlier checks not repeated with v3. Student: the round-"
              f"{STUDENT_VAL_ROUND} pick on the same scenes (validation_rounds.csv); round 50 was never validated. "
              "nT = near-T.")
    return fig


def _teacher_26_summary():
    groups = [("10 earlier", OLD10), ("17 newer", NEW17), ("All 27", TRAINED27)]
    fig, ax = plt.subplots(figsize=HALF)
    w = 0.38
    for gi, (gname, shapes) in enumerate(groups):
        t, st = _teacher_student_rows(shapes)
        for j, (k, n, c) in enumerate(((sum(a for a, _ in t), sum(b for _, b in t), TEACHER),
                                       (sum(a for a, _ in st), sum(b for _, b in st), M96))):
            v = 100 * k / n
            ax.bar(gi - w / 2 + w * j, v, w, color=c, edgecolor="white", zorder=3)
            ax.text(gi - w / 2 + w * j, v + 2, f"{v:.0f}", ha="center", fontsize=14)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("Validation success")
    legend_top(ax, handles=[patch(TEACHER, "teacher (exact state)"),
                            patch(M96, f"camera student (round {STUDENT_VAL_ROUND})")], ncol=2)
    note(fig, "Same 50 validation scenes per shape for both. Teacher: latest check per shape (teacher_validation.csv; "
              "the 10 earlier shapes give 400 of 500 here, while the run that labelled the 10-shape students scored 390 "
              "of 500 with an older world model).")
    return fig


def fig_method_teacher_gap():
    # teacher totals: typed_in_numbers.json (5 and 10 shapes) and teacher_validation.csv (27 shapes, summed);
    # students: the picked validation round of each stage (validation_rounds.csv)
    tt = TYPED["teacher_totals"]["values"]
    k27 = sum(v[0] for v in TEACHER_VAL.values()); n27 = sum(v[1] for v in TEACHER_VAL.values())
    rows = [("5 shapes", tt["5 shapes"], "5 shapes", "correction rounds", _PR["5-shape round 7"][2], FIVE),
            ("10 shapes", tt["10 shapes"], "10 shapes", "96 px main line", _PR["10-shape 96 px"][2] + 1, TEN),
            ("27 shapes", (k27, n27), "27 shapes", "96 px line", STUDENT_VAL_ROUND + 1, M96)]
    fig, ax = plt.subplots(figsize=HALF)
    x = np.arange(len(rows))
    w = 0.36
    for i, (name, (tk, tn), stage, line, step, c) in enumerate(rows):
        s, y, *_ = val_line(stage, line)
        v = y[list(s).index(step)]
        t = 100 * tk / tn
        ax.bar(i - w / 2, t, w, color=TEACHER, edgecolor="white", zorder=3)
        ax.bar(i + w / 2, v, w, color=c, edgecolor="white", zorder=3)
        ax.text(i - w / 2, t + 1.5, f"{t:.0f}", ha="center", fontsize=13)
        ax.text(i + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=13)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("Validation success")
    if slide():
        for xx, t, c in ((-w / 2, "teacher", INK), (w / 2, "student", "white")):
            ax.text(xx, 4, t, rotation=90, ha="center", va="bottom", color=c, fontsize=14)
    else:
        legend_top(ax, handles=[patch(TEACHER, "teacher (exact state)"), patch(FIVE, "student, 5"),
                                patch(TEN, "student, 10"), patch(M96, "student, 27")], ncol=2, fontsize=12)
    note(fig, "Same 50 validation scenes per shape for teacher and camera student. Teachers: the noisy-state planner that "
              "labelled each stage (5 shapes 191/250, 10 shapes 390/500; 27 shapes: latest check per shape, "
              "teacher_validation.csv). Students: the picked round of each stage (rounds 7, 13, 44).")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Typed-in tables (every number from the docs lives in presentation/data/, each row with its source)
# ----------------------------------------------------------------------------------------------------------------------
EARLY = read("early_attempts.csv")
WMACC = read("world_model_accuracy.csv")
TPLAN = read("teacher_planner.csv")
TIMELINE = read("timeline.csv")
DSIZES = read("dataset_sizes.csv")
TRAIN = read("training_rounds.csv")
LOSS = read("training_loss_curves.csv")
MF = TYPED["method_pipeline"]


def _f(x):
    return None if x in ("", None) else float(x)


def ea_conditions(attempt):
    out = []
    for r in EARLY:
        if r["attempt"] == attempt and r["condition"] not in out:
            out.append(r["condition"])
    return out


def ea(attempt, condition, shape, metric, field="value"):
    rows = [r for r in EARLY if r["attempt"] == attempt and r["condition"] == condition and r["shape"] == shape
            and r["metric"] == metric]
    assert len(rows) == 1, (attempt, condition, shape, metric, len(rows))
    return _f(rows[0][field])


def ea_row(attempt, condition, shape, metric):
    rows = [r for r in EARLY if r["attempt"] == attempt and r["condition"] == condition and r["shape"] == shape
            and r["metric"] == metric]
    assert len(rows) == 1, (attempt, condition, shape, metric, len(rows))
    return rows[0]


def tp(experiment, condition=None, shape=None):
    return [r for r in TPLAN if r["experiment"] == experiment and (condition is None or r["condition"] == condition)
            and (shape is None or r["shape"] == shape)]


def tp_conditions(experiment):
    out = []
    for r in TPLAN:
        if r["experiment"] == experiment and r["condition"] not in out:
            out.append(r["condition"])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# final_blends_all: every pre-registered blend test (D040-D043)
# ----------------------------------------------------------------------------------------------------------------------
BLEND_LABEL = {"D040 primary": ("D040: 70% blend, whole network", "D040 whole 70%"),
               "D040 secondary": ("D040, 2nd look: 70% encoder", "D040 enc. 70%"),
               "D041": ("D041: 70% encoder, fresh pool", "D041 enc. 70%"),
               "D042: original action network + 30%": ("D042: 30% encoder, fresh pool", "D042 enc. 30%"),
               "D042: original action network + 50%": ("D042: 50% encoder, fresh pool", "D042 enc. 50%"),
               "D043": ("D043: 50% encoder, replication", "D043 enc. 50%")}


def _blend_label(cond, short=False):
    for k, v in BLEND_LABEL.items():
        if cond.startswith(k):
            return v[1 if short else 0]
    return cond


def fig_final_blends_all():
    tests = ea_conditions("blend_tests")
    shapes = FIVE_SHAPES if not slide() else ["T", "Y", "D"]
    lims = {"T": (-10, 8), "T_narrow": (-10, 8), "Y": (-1, 13), "D": (-1, 7), "L": (-1.2, 1.2)}
    ratios = {"T": 18, "T_narrow": 18, "Y": 14, "D": 8 if not slide() else 10, "L": 5}
    fig, axes = plt.subplots(1, len(shapes), figsize=FULL, sharey=True,
                             gridspec_kw={"width_ratios": [ratios[s] for s in shapes], "wspace": 0.12})
    ys = np.arange(len(tests))
    for ax, s in zip(axes, shapes):
        for i, t in enumerate(tests):
            passed = ea("blend_tests", t, "all", "passed") == 1
            r = ea_row("blend_tests", t, s, "difference_pts")
            d, lo, hi = _f(r["value"]), _f(r["ci_low"]), _f(r["ci_high"])
            c = FIVE if passed else ORIG_LINE
            if lo is not None:
                ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
            ax.plot(d, i, "o", ms=8 if not slide() else 7, mfc=c if passed else "white", mec=c, mew=1.8, zorder=4)
        ax.axvline(0, color=INK2, lw=1.0, zorder=2)
        if s in ("T", "T_narrow"):
            ax.axvline(-5, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=2)
            if s == "T" and not slide():
                ax.text(-4.7, len(tests) - 0.5, "margin −5", ha="left", va="bottom", fontsize=11, color=INK2)
        ax.set_xlim(*lims[s])
        ax.set_title(disp(s), loc="center", fontsize=15)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.xaxis.set_major_formatter(SIGNED)
        ax.xaxis.set_major_locator(MultipleLocator({"T": 5, "T_narrow": 5, "Y": 5, "D": 3, "L": 1}[s] if not slide()
                                                   else {"T": 10, "Y": 10, "D": 5}[s]))
        if s != "T":
            ax.tick_params(axis="y", left=False)
    axes[0].set_yticks(ys, [_blend_label(t, short=slide()) for t in tests], fontsize=None if slide() else 12)
    axes[0].set_ylim(len(tests) - 0.5, -0.6)
    for i, t in enumerate(tests):
        passed = ea("blend_tests", t, "all", "passed") == 1
        axes[-1].text(1.15, i, "passed" if passed else "failed", transform=axes[-1].get_yaxis_transform(), va="center",
                      ha="left", fontsize=12, fontweight="bold" if passed else None, color=INK if passed else INK2)
    fig.supxlabel(SL("Change in success vs. the original policy (points), paired by scene", "Change vs. original (points)"),
                  fontsize=14 if not slide() else 16, y=0.0 if not slide() else None)
    if slide():     # the dashed line on T is the pre-registered margin
        h = Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), label="pass rule: T interval above −5")
        slide_legend(fig, [h], [h.get_label()])
    note(fig, "Pre-registered rule (D040): T and near-T lower bounds at least −5 points, at least one of Y, D, L clearly "
              "better, none clearly worse. 1000 fresh scenes per letter (D040), 2000 on new pools (D041-D043); 95% "
              "intervals (D042: 97.5%, two candidates). Filled = passed. Data: early_attempts.csv (D040 also in "
              "final_tests_paired.csv).")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Diagrams: method pipeline, policy interface, world model structure (numbers from typed_in_numbers.json)
# ----------------------------------------------------------------------------------------------------------------------
KIND = {"privileged": (tint(TEACHER, 0.78), TEACHER), "deployed": (tint(M96, 0.82), M96),
        "data": ("#f1f0eb", AXIS), "copied": ("#f1f0eb", INK2), "learned": (tint(TEACHER, 0.78), TEACHER),
        "io": ("white", AXIS)}


def _diagram_axes(size=None):
    """Axes filling the whole canvas, 100 units wide, equal aspect (1 unit = 1% of the width)."""
    if slide():
        size = VARIANT_SIZE[MODE["name"]]
    fig = plt.figure(figsize=size or FULL, layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    w, h = fig.get_size_inches()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _units_per_pt(ax):
    return 100 / (ax.figure.get_size_inches()[0] * 72)


def _box(ax, x, y, w, h, title, body="", kind="data", title_fs=13, body_fs=11, lw=1.6):
    fc, ec = KIND[kind]
    title_fs, body_fs = FS(title_fs), FS(body_fs)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=lw, zorder=2))
    if body:
        u = _units_per_pt(ax)
        pad = 0.55 * title_fs * u
        th = title.count("\n") + 1
        ax.text(x + w / 2, y + h - pad, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color=INK,
                zorder=3, linespacing=1.1)
        top = y + h - pad - th * 1.18 * title_fs * u
        ax.text(x + w / 2, (top + y + 0.4 * pad) / 2, body, ha="center", va="center", fontsize=body_fs, color=INK,
                zorder=3, linespacing=1.22)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=INK,
                zorder=3, linespacing=1.15)


def _arrow(ax, p0, p1, label=None, lpos=None, rad=0.0, color=INK2, fs=11.5, ha="center", va="bottom", lw=1.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        x, y = lpos if lpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.8)
        ax.text(x, y, label, ha=ha, va=va, fontsize=fs, color=INK2, zorder=5, linespacing=1.15)


def _kind_legend(ax, items, x, y, dy=4.2, fs=12, horizontal=False, dx=None):
    for i, (kind, text) in enumerate(items):
        fc, ec = KIND[kind]
        xx, yy = (x + i * dx, y) if horizontal else (x, y - i * dy)
        ax.add_patch(FancyBboxPatch((xx, yy - 1.2), 3.6, 2.4, boxstyle="round,pad=0,rounding_size=0.6", fc=fc, ec=ec,
                                    lw=1.4, zorder=2))
        ax.text(xx + 4.8, yy, text, ha="left", va="center", fontsize=fs, color=INK, zorder=3)


def fig_method_pipeline():
    m = MF
    fig, ax = _diagram_axes()
    cand = f"{m['planner_candidates_per_step']:,}"      # 4 x 256 scored plans; 128 of the first 256 are structured pushes
    if slide():
        bw, bh = 31, 21
        X = [0.5, 34.5, 68.5]
        yt, yb = 32.5, 7
        _box(ax, X[0], yt, bw, bh, "Rollouts", "original policy\n+ action noise", "data")
        _box(ax, X[1], yt, bw, bh, "World model", "exact pusher,\nlearned block", "privileged")
        _box(ax, X[2], yt, bw, bh, "Planner teacher", f"{cand} plans per\nstep, exact state", "privileged")
        _box(ax, X[2], yb, bw, bh, "Demos", f"intended\n{m['plan_steps']}-step plans", "data")
        _box(ax, X[1], yb, bw, bh, "Camera policy", f"{m['camera_frames']} frames + pusher\n→ {m['plan_steps']} steps, "
             f"runs {m['executed_steps']}", "deployed")
        _box(ax, X[0], yb, bw, bh, "Corrections", "policy drives,\nteacher labels", "privileged")
        my = yt + bh / 2
        _arrow(ax, (X[0] + bw, my), (X[1], my))
        _arrow(ax, (X[1] + bw, my), (X[2], my))
        _arrow(ax, (X[2] + bw / 2, yt), (X[2] + bw / 2, yb + bh))
        _arrow(ax, (X[2], yb + bh * 0.5), (X[1] + bw, yb + bh * 0.5))
        _arrow(ax, (X[1], yb + bh * 0.72), (X[0] + bw, yb + bh * 0.72))
        _arrow(ax, (X[0] + bw, yb + bh * 0.28), (X[1], yb + bh * 0.28))
        _arrow(ax, (X[2] + 4, yt), (X[0] + bw - 5, yb + bh), color=TEACHER)
        _kind_legend(ax, [("privileged", "training only"), ("deployed", "deployed"), ("data", "data")], 1.5, 2.6,
                     horizontal=True, dx=33, fs=16)
        return fig
    yt, ht = 30, 24
    tops = [(0.5, 21, "Machine rollouts",
             f"original policy +\naction noise ({'/'.join(str(v) for v in m['action_noise_px'])} px):\n"
             f"{m['dynamics_episodes']:,} episodes, {m['dynamics_controls_thousands']}k\ncontrols; no human demos", "data"),
            (26.5, 21, "Structured\nworld model", "pusher motion: exact\ncontact gate: exact\nblock motion: learned",
             "privileged"),
            (52.5, 24, "Sampling planner\nteacher",
             f"scores {cand} candidate plans\nper step in the world model\n({m['planner_random_search_iterations']} "
             f"rounds × {m['planner_samples_per_iteration']}; {m['planner_structured_pushes']} of the\nfirst "
             f"{m['planner_samples_per_iteration']} are structured pushes;\n{m['planner_horizon_controls']} steps "
             f"ahead); reads the exact\nstate with {m['teacher_state_noise_px']} px / "
             f"{m['teacher_angle_noise_deg']}° noise", "privileged"),
            (81.5, 18, "Teacher\ndemos", f"runs in the\nsimulator; labels\n= its intended\n{m['plan_steps']}-step plans",
             "data")]
    for x, w, t, b, k in tops:
        _box(ax, x, yt, w, ht, t, b, k)
    my = yt + ht / 2
    for (x0, w0, *_), (x1, *_) in zip(tops[:-1], tops[1:]):
        _arrow(ax, (x0 + w0, my), (x1, my))
    ax.text(24.0, my + 1.0, "fit", ha="center", va="bottom", fontsize=11, color=INK2)
    yb, hb = 2, 21
    _box(ax, 71, yb, 28.5, hb, "Camera policy (deployed)",
         f"official Diffusion Policy recipe:\n{m['camera_frames']} frames {m['camera_px']} × {m['camera_px']} + pusher "
         f"position\n→ {m['plan_steps']} future targets,\nexecutes the first {m['executed_steps']}", "deployed")
    _box(ax, 29, yb, 33, hb, "Correction rounds (repeated)",
         f"the policy drives; the teacher takes\n"
         f"{m['teacher_share_of_decisions_in_corrections']:.0%} of decisions and labels every\nvisited state; retrain "
         f"on everything so\nfar, incl. {m['human_T_episodes']} human T demos", "privileged")
    _arrow(ax, (90.5, yt), (90.5, yb + hb), "copy the\nplans", (91.7, (yt + yb + hb) / 2), ha="left", va="center")
    _arrow(ax, (71, yb + hb * 0.66), (62, yb + hb * 0.66), "drives new\nscenes", (66.5, yb + hb * 0.66 + 0.8), fs=11)
    _arrow(ax, (62, yb + hb * 0.24), (71, yb + hb * 0.24), "retrain", (66.5, yb + hb * 0.24 - 0.8), va="top", fs=11)
    _arrow(ax, (59, yt), (53, yb + hb), "labels\nvisited states", (58.6, (yt + yb + hb) / 2 + 0.2), ha="left",
           va="center", color=TEACHER, fs=11)
    _kind_legend(ax, [("privileged", "training only:\nreads exact simulator\nstate"), ("deployed", "deployed:\ncamera only"),
                      ("data", "data")], 1.0, 20.0, dy=7.6, fs=11.5)
    return fig


def fig_method_pipeline_simple():
    fig, ax = _diagram_axes((10, 3.4) if not slide() else None)
    if slide():
        ax.set_ylim(0, 100 * VARIANT_SIZE[MODE["name"]][1] / VARIANT_SIZE[MODE["name"]][0])
    H = ax.get_ylim()[1]
    n = 4
    gap = 4.5
    bw = (100 - 1 - gap * (n - 1)) / n
    bh = 15 if not slide() else 20
    y0 = (H - bh) / 2 + (2.5 if not slide() else 4)
    items = [("Machine\nrollouts", "data"), ("World\nmodel", "privileged"), ("Planner\nteacher", "privileged"),
             (SL("Camera policy,\n26 letters", "Camera\npolicy"), "deployed")]
    xs = [0.5 + i * (bw + gap) for i in range(n)]
    for x, (t, k) in zip(xs, items):
        _box(ax, x, y0, bw, bh, t, kind=k, title_fs=17 if not slide() else 16)
    for i in range(n - 1):
        _arrow(ax, (xs[i] + bw, y0 + bh / 2), (xs[i + 1], y0 + bh / 2), lw=2)
    # correction loop between teacher and policy
    a, b = xs[2] + bw / 2, xs[3] + bw / 2
    ax.add_patch(FancyArrowPatch((b, y0), (a, y0), arrowstyle="-|>", mutation_scale=16, lw=2, color=TEACHER,
                                 connectionstyle="arc3,rad=-0.45", zorder=4))
    ax.text((a + b) / 2, y0 - (7.5 if not slide() else 9.5), "correction rounds", ha="center", va="top",
            fontsize=14 if not slide() else 16, color=INK2)
    if not slide():
        ax.text(xs[1] + bw + gap / 2, y0 + bh + 2.5, "training only (reads exact simulator state)", ha="center",
                va="bottom", fontsize=13, color=INK2)
        ax.plot([xs[1], xs[2] + bw], [y0 + bh + 1.6] * 2, color=TEACHER, lw=2)
        ax.text(xs[3] + bw / 2, y0 + bh + 2.5, "deployed: camera only", ha="center", va="bottom", fontsize=13,
                color=INK2)
        ax.plot([xs[3], xs[3] + bw], [y0 + bh + 1.6] * 2, color=M96, lw=2)
    else:
        ax.text(xs[1] + bw + gap / 2, y0 + bh + 2, "training only", ha="center", va="bottom", fontsize=16, color=INK2)
        ax.plot([xs[1], xs[2] + bw], [y0 + bh + 1.2] * 2, color=TEACHER, lw=2)
        ax.text(xs[3] + bw / 2, y0 + bh + 2, "deployed", ha="center", va="bottom", fontsize=16, color=INK2)
        ax.plot([xs[3], xs[3] + bw], [y0 + bh + 1.2] * 2, color=M96, lw=2)
    return fig


# PushT-style T block (two rectangles), in block coordinates; used only for schematic camera frames
T_POLY = np.array([(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)], float)
T_POLY[:, 1] -= 45


def _t_poly(cx, cy, ang_deg, scale):
    a = math.radians(ang_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (T_POLY * scale) @ R.T + np.array([cx, cy])


SCENE = dict(goal=(48, 46, 45), block=(42, 58, 15), scale=0.19)


def _scene(ax, pusher, plan=None, executed=8, dot=11):
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 96)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), 96, 96, fc="#f6f6f3", ec=AXIS, lw=1.2, zorder=0))
    gx, gy, ga = SCENE["goal"]
    bx, by, ba = SCENE["block"]
    ax.add_patch(Polygon(_t_poly(gx, gy, ga, SCENE["scale"]), closed=True, fc="none", ec="#2fa84f", lw=1.8, zorder=1))
    ax.add_patch(Polygon(_t_poly(bx, by, ba, SCENE["scale"]), closed=True, fc="#9a9993", ec="none", zorder=2))
    if plan is not None:
        ax.scatter(plan[executed:, 0], plan[executed:, 1], s=dot, color="#f3a6c0", zorder=3, lw=0)
        ax.scatter(plan[:executed, 0], plan[:executed, 1], s=dot * 1.2, color="#d6232a", zorder=4, lw=0)
    ax.add_patch(Circle(pusher, 3.0, fc="#2a62d6", ec="white", lw=0.8, zorder=5))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _plan_points(start, n):
    t = np.linspace(0, 1, n + 1)[1:]
    p0, p1, p2 = np.array(start), np.array([23, 50]), np.array([35, 52])
    pts = ((1 - t) ** 2)[:, None] * p0 + (2 * (1 - t) * t)[:, None] * p1 + (t ** 2)[:, None] * p2
    return pts


def fig_policy_interface():
    m = MF
    sl = slide()
    fig, ax = _diagram_axes((10, 3.9) if not sl else None)
    H = ax.get_ylim()[1]
    pusher0, pusher1 = (20, 30), (22, 34)
    plan = _plan_points(pusher1, m["plan_steps"])
    handles = [Line2D([], [], marker="o", color="#d6232a", ls="none", ms=8, label=f"executed (first {m['executed_steps']})"),
               Line2D([], [], marker="o", color="#f3a6c0", ls="none", ms=8, label="rest of the plan"),
               Line2D([], [], marker="o", color="#2a62d6", ls="none", ms=9, label="pusher"),
               Line2D([], [], color="#2fa84f", lw=2, label="goal"),
               Patch(facecolor="#9a9993", label="block")]
    if sl:
        fw, fy = 25, 20
        for (x, y), p in zip(((4.0, fy + 3.0), (0.5, fy)), (pusher0, pusher1)):
            _scene(ax.inset_axes([x, y, fw, fw], transform=ax.transData), p)
        ax.text(0.5 + fw / 2 + 1.5, fy - 1.0, f"{m['camera_frames']} frames\n+ pusher position", ha="center", va="top",
                fontsize=16, color=INK)
        x_net, nw = 36, 26
        _box(ax, x_net, fy + 3, nw, fw - 3, "Policy", f"{m['denoising_steps']} denoising\nsteps", "deployed")
        _arrow(ax, (x_net - 6, fy + fw / 2), (x_net - 0.8, fy + fw / 2), lw=2)
        x_out = x_net + nw + 5.5
        _arrow(ax, (x_net + nw + 0.5, fy + fw / 2), (x_out - 0.6, fy + fw / 2), lw=2)
        fo = 100 - x_out - 0.5
        _scene(ax.inset_axes([x_out, fy + fw - fo, fo, fo], transform=ax.transData), pusher1, plan,
               executed=m["executed_steps"], dot=24)
        ax.text(x_out + fo / 2, fy + fw + 0.8, f"{m['plan_steps']} targets", ha="center", va="bottom", fontsize=16,
                color=INK)
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, handletextpad=0.3,
                  columnspacing=0.8, borderaxespad=0.2)
        return fig
    fw = 20
    fy = H - fw - 6
    for i, (x, p) in enumerate(zip([1, 1 + fw + 2], (pusher0, pusher1))):
        _scene(ax.inset_axes([x, fy, fw, fw], transform=ax.transData), p)
        ax.text(x + fw / 2, fy + fw + 1, "camera frame t−1" if i == 0 else "camera frame t", ha="center",
                va="bottom", fontsize=12, color=INK2)
    ax.text(1 + fw + 1, fy - 2.5, f"{m['camera_frames']} frames, {m['camera_px']} × {m['camera_px']} px\n"
            f"+ pusher position (x, y)", ha="center", va="top", fontsize=13, color=INK)
    x_net, nw = 1 + 2 * fw + 2 + 6, 22
    _box(ax, x_net, fy + fw * 0.12, nw, fw * 0.76, "Diffusion policy",
         f"image encoder + U-Net\n{m['denoising_steps']} denoising steps", "deployed", title_fs=13, body_fs=12)
    _arrow(ax, (x_net - 5.5, fy + fw / 2), (x_net - 0.5, fy + fw / 2), lw=2)
    x_out = x_net + nw + 6
    _arrow(ax, (x_net + nw + 0.5, fy + fw / 2), (x_out - 0.5, fy + fw / 2), lw=2)
    _scene(ax.inset_axes([x_out, fy, fw, fw], transform=ax.transData), pusher1, plan, executed=m["executed_steps"],
           dot=18)
    ax.text(x_out + fw / 2, fy + fw + 1, f"output: {m['plan_steps']} future pusher targets", ha="center",
            va="bottom", fontsize=12, color=INK2)
    ax.text(x_out + fw + 2, fy + fw * 0.62, f"first {m['executed_steps']} executed,\nthen plan again\n"
            f"(every {m['executed_steps']} controls)", ha="left", va="center", fontsize=12, color=INK)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=12, handletextpad=0.4,
              columnspacing=1.2)
    return fig


def fig_wm_structure():
    m = MF
    sl = slide()
    fig, ax = _diagram_axes((10, 4.0) if not sl else None)
    H = ax.get_ylim()[1]
    if sl:
        bw, bh = 31, 18
        X = [0.5, 34.5, 68.5]
        yt, yb = H - bh - 1, 6
        _box(ax, X[0], yt, bw, bh, "State + action", kind="io", title_fs=16)
        _box(ax, X[1], yt, bw, bh, "Pusher motion", "copied, exact", "copied")
        _box(ax, X[2], yt, bw, bh, "Contact gate", "copied, exact", "copied")
        _box(ax, X[2], yb, bw, bh, "Block motion", "learned", "learned")
        _box(ax, X[1], yb, bw, bh, "Next state", kind="io", title_fs=16)
        _arrow(ax, (X[0] + bw, yt + bh / 2), (X[1], yt + bh / 2))
        _arrow(ax, (X[1] + bw, yt + bh / 2), (X[2], yt + bh / 2))
        _arrow(ax, (X[2] + bw * 0.7, yt), (X[2] + bw * 0.7, yb + bh), "contact", (X[2] + bw * 0.7 - 1.2, (yt + yb + bh) / 2),
               ha="right", va="center", fs=16)
        _arrow(ax, (X[2], yb + bh / 2), (X[1] + bw, yb + bh / 2))
        _arrow(ax, (X[2] + bw * 0.2, yt), (X[1] + bw * 0.8, yb + bh), "no contact", (X[1] + bw * 0.55, (yt + yb + bh) / 2),
               ha="right", va="center", fs=16)
        _kind_legend(ax, [("copied", "copied"), ("learned", "learned")], 1.5, yb + bh - 4, dy=8, fs=16)
        return fig
    bh = 21
    y = H - bh - 4.5
    _box(ax, 0.5, y, 15, bh, "Now", "block pose\nand speed,\npusher position\n+ pusher target", "io")
    _box(ax, 20, y, 21, bh, "Pusher motion",
         f"copied from the simulator:\nPD law, {m['world_model_pd_substeps']} substeps\n(matches it to 0.0001 px)",
         "copied")
    _box(ax, 45.5, y, 21, bh, "Contact gate",
         "copied: exact distance from\nthe pusher to the letter's\nrectangles; no contact →\nthe block does not move",
         "copied", body_fs=11)
    _box(ax, 71, y, 21, bh, "Block motion",
         "learned: small network in\nthe block's own frame,\nonly while in contact", "learned")
    _box(ax, 94, y, 5.5, bh, "Next", kind="io", title_fs=12)
    for x0, x1 in ((15.5, 20), (41, 45.5), (66.5, 71), (92, 94)):
        _arrow(ax, (x0, y + bh / 2), (x1, y + bh / 2))
    _arrow(ax, (56, y), (96.75, y), "no contact: motion exactly zero", (76, y - 5.4), rad=0.18, va="top")
    ax.text(81.5, y + bh + 1, f"trained on {m['dynamics_episodes']:,} machine rollouts of the original policy\n"
            f"({m['dynamics_controls_thousands']}k controls, action noise "
            f"{', '.join(str(v) for v in m['action_noise_px'])} px; no human demos)", ha="center", va="bottom",
            fontsize=11, color=INK2)
    _kind_legend(ax, [("copied", "copied from the simulator (exact)"), ("learned", "learned from data"),
                      ("io", "input / output state")], 1, 2.5, horizontal=True, dx=33, fs=12)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# World-model accuracy (world_model_accuracy.csv, LEARNING_FINDINGS.md section 4 and Open problems)
# ----------------------------------------------------------------------------------------------------------------------
def wm(item, model=None, shape=None, horizon=None, statistic=None, unit=None):
    rows = [r for r in WMACC if r["item"] == item and (model is None or r["model"] == model)
            and (shape is None or r["shape"] == shape) and (horizon is None or int(r["horizon_controls"]) == horizon)
            and (statistic is None or r["statistic"] == statistic) and (unit is None or r["unit"] == unit)]
    return rows


def wm1(*a, **k):
    rows = wm(*a, **k)
    assert len(rows) == 1, (a, k, len(rows))
    return float(rows[0]["value"])


def fig_wm_accuracy():
    sl = slide()
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1, 1.7] if not sl else [1.1, 1.6],
                                                               "wspace": 0.32})
    need = wm1("requirement", shape="all", unit="px")
    ax = axes[0]
    S = "structured (wm2 v1)"
    w = 0.36
    for j, h in enumerate((8, 32) if not sl else (8,)):
        g = wm1("contact_error", model="generic MLP", horizon=h, statistic="median", unit="px")
        per = [wm1("contact_error", model=S, shape=s, horizon=h, statistic="median", unit="px") for s in FIVE_SHAPES]
        xg, xs_ = (j - w / 2, j + w / 2) if not sl else (0, 1)
        ww = w if not sl else 0.6
        ax.bar(xg, g, ww, color=ORIG, edgecolor="white", zorder=3)
        ax.bar(xs_, np.mean(per), ww, color=TEACHER, edgecolor="white", zorder=3)
        ax.scatter(np.full(5, xs_), per, s=14, color=INK2, zorder=4, lw=0)
        ax.text(xg, g + 0.08, f"{g:.2f}" if not sl else f"{g:.1f}", ha="center", va="bottom", fontsize=12)
        ax.text(xs_, max(per) + 0.08, f"{min(per):.2f}-\n{max(per):.2f}" if not sl else f"{np.mean(per):.1f}",
                ha="center", va="bottom", fontsize=11.5)
    ax.axhline(need, color=INK, lw=1.1, ls=(0, (3, 2)))
    if sl:
        ax.set_xticks([0, 1], ["generic", "ours"])       # ours = the structured world model
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 2.9)
        ax.text(1.5, need + 0.06, f"≈{need:g} px\nneeded", ha="right", va="bottom", fontsize=16, color=INK2)
        ax.set_ylabel("Median error (px)")
        ax.set_title("8 controls ahead", loc="left")
    else:
        ax.set_xticks([0, 1], ["8 controls", "32 controls"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 4.9)
        ax.set_ylabel("Median contact error (px)")
        ax.set_title("Typical error", loc="left", fontsize=15)
        legend_top(ax, handles=[patch(ORIG, "generic network"), patch(TEACHER, "structured (dots: shapes)"),
                                Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2)),
                                       label=f"≈{need:g} px needed for success")], ncol=1, fontsize=12, y=1.08)
    ax.grid(False, axis="x")
    ax = axes[1]
    xs = np.arange(len(FIVE_SHAPES) + 1)
    for j, (h, c) in enumerate(((8, TEACHER), (32, tint(TEACHER, 0.45)))):
        if sl and h == 32:
            continue
        off = (-0.16 + 0.32 * j) if not sl else 0
        for i, s in enumerate(FIVE_SHAPES):
            md = wm1("contact_error", model=S, shape=s, horizon=h, statistic="median", unit="px")
            p9 = wm1("contact_error", model=S, shape=s, horizon=h, statistic="p90", unit="px")
            ax.plot([i + off, i + off], [md, p9], color=c, lw=3, solid_capstyle="butt", zorder=3)
            ax.plot(i + off, md, "o", color=c, ms=7, mec="white", mew=1, zorder=4)
            ax.plot(i + off, p9, "_", color=INK2, ms=10, mew=2, zorder=4)
    i = len(FIVE_SHAPES)
    md = wm1("rare_large_errors", statistic="median")
    p95 = wm1("rare_large_errors", statistic="95th percentile (1 in 20 at or above)")
    worst = wm1("rare_large_errors", statistic="worst")
    ax.plot([i, i], [md, worst], color=ORIG, lw=3, solid_capstyle="butt", zorder=3)
    ax.plot(i, md, "o", color=ORIG, ms=7, mec="white", zorder=4)
    for v, t in ((p95, f"1 in 20 ≥ {p95:g} px"), (worst, f"worst {worst:g} px"))[:2 if not sl else 1]:
        ax.plot(i, v, "_", color=INK2, ms=10, mew=2, zorder=4)
        ax.text(i - (0.12 if not sl else 0.4), v, t, ha="right", va="center", fontsize=11.5, color=INK2)
    if sl:          # what the marks mean (the default version has a legend)
        ax.text(-0.45, 95, "dot: typical error\nline top: 1 in 10", ha="left", va="top", color=INK2)
    ax.axhline(need, color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_ylim(0.08, 150)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(xs, [disp(s).replace("near-T", "nT") for s in FIVE_SHAPES] + ["all,\nnear walls" if not sl else "all"])
    ax.set_xlim(-0.6, len(xs) - 0.3)
    ax.grid(False, axis="x")
    ax.set_ylabel(SL("Contact error (px, log scale)", "Error (px, log)"))
    ax.set_title(SL("Structured model: typical and rare errors", "Rare errors"), loc="left",
                 fontsize=15)
    if not sl:
        legend_top(ax, handles=[Line2D([], [], color=TEACHER, lw=3, marker="o", label="8 controls"),
                                Line2D([], [], color=tint(TEACHER, 0.45), lw=3, marker="o", label="32 controls"),
                                Line2D([], [], color=INK2, marker="_", ls="none", ms=10, mew=2,
                                       label="dot: median, bar: 90th percentile"),
                                Line2D([], [], color=ORIG, lw=3, marker="o", label="500 16-control windows")],
                   ncol=2, fontsize=12, y=1.08)
    note(fig, f"Held-out base-policy rollouts with real contact (world_model_accuracy.csv; LEARNING_FINDINGS.md section 4 "
              f"and Open problems). Near the goal the one-step error is {wm1('near_goal_one_step'):g} px; for the "
              f"planner's own best push the model matched the simulator within 1 px in "
              f"{wm1('planner_push_check', statistic='trials within 1 px'):.0f} of 15 trials. Large errors happen mostly "
              "when the letter touches the arena wall, which the model has no input for.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Earlier approaches (early_attempts.csv)
# ----------------------------------------------------------------------------------------------------------------------
EARLY_RAMP = [tint(FIVE, 0.62), tint(FIVE, 0.42), tint(FIVE, 0.22), FIVE, "#8a2c62"]


def _oracle_bon_panel(ax):
    """Original vs simulator best-of-N per letter; labels 'b -> o of n' (slide: left of the dots when they are far
    right, so the text stays inside the panel)."""
    shapes = FIVE_SHAPES
    for i, s in enumerate(shapes):
        n = ea("oracle_bon", "base policy", s, "successes", "of")
        b = ea("oracle_bon", "base policy", s, "successes")
        o = ea("oracle_bon", "simulator best-of-N (upper bound)", s, "successes")
        pb, po = 100 * b / n, 100 * o / n
        ax.plot([pb, po], [i, i], color=tint(TEACHER, 0.4), lw=2.4, zorder=2)
        ax.plot(pb, i, "o", color=ORIG, ms=10, mec="white", zorder=3)
        ax.plot(po, i, "D", color=TEACHER, ms=9, mec="white", zorder=4)
        txt = f"{b:.0f} → {o:.0f} of {n:.0f}"
        if slide() and max(pb, po) > 45:
            ax.text(min(pb, po) - 5, i, txt, va="center", ha="right", fontsize=12)
        else:
            ax.text(max(pb, po) + (4 if not slide() else 5), i, txt, va="center", fontsize=12)
    ax.set_yticks(range(len(shapes)), [disp(s) if half() or not slide() else SHORT.get(s, s) for s in shapes])
    ax.set_ylim(len(shapes) - 0.5, -0.7)
    ax.set_xlim(-3, 100)
    if slide():
        ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel(SL("Success (same validation scenes)", "Success (validation)"))


ORACLE_HANDLES = lambda: [Line2D([], [], marker="o", color=ORIG, ls="none", ms=9, label="original policy"),
                          Line2D([], [], marker="D", color=TEACHER, ls="none", ms=8,
                                 label=SL("simulator best-of-N", "best of 16, true simulator"))]


def fig_oracle_bon():
    if half():      # half slide: the upper-bound panel only
        fig, ax = plt.subplots(figsize=HALF)
        _oracle_bon_panel(ax)
        legend_top(ax, handles=ORACLE_HANDLES())
        return fig
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1.15, 1] if not slide() else [2.3, 1],
                                                               "wspace": 0.42 if not slide() else 0.12})
    ax = axes[0]
    _oracle_bon_panel(ax)
    ax.set_title(SL("Pick the best of 16 own plans\nwith the true simulator", "Upper bound"), loc="left", fontsize=14)
    if not slide():
        ax.legend(handles=ORACLE_HANDLES(), loc="lower right", fontsize=12)
    ax = axes[1]
    conds = ea_conditions("oracle_imitation")
    base = [c for c in conds if c.startswith("base")]
    studs = [c for c in conds if not c.startswith("base")]
    sh3 = ["T", "T_narrow", "Y"]
    for i, s in enumerate(sh3):
        bv = [ea("oracle_imitation", c, s, "successes") for c in base]
        ax.add_patch(Rectangle((i - 0.38, min(bv) - 0.4), 0.76, max(bv) - min(bv) + 0.8, fc=tint(ORIG, 0.6), ec="none",
                               zorder=1))
        for j, c in enumerate(studs):
            v = ea("oracle_imitation", c, s, "successes")
            ax.plot(i - 0.27 + 0.18 * j, v, "o", color=EARLY_RAMP[j + 1] if not slide() else FIVE, ms=8,
                    mec="white", zorder=3)
    ax.set_xticks(range(3), [disp(s) if not slide() else SHORT.get(s, s) for s in sh3])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 78 if not slide() else 60)
    ax.yaxis.set_major_locator(MultipleLocator(10 if not slide() else 20))
    ax.grid(False, axis="x")
    ax.set_ylabel(SL("Successes of 100 (monitor test)", "Successes of 100"))
    ax.set_title(SL("Imitating those choices", "Imitating it"), loc="left", fontsize=14)
    if slide():
        hs = ORACLE_HANDLES() + [Patch(facecolor=tint(ORIG, 0.6), label="original, 2 seeds"),
                                 Line2D([], [], marker="o", color=FIVE, ls="none", ms=8, label="imitating students")]
        slide_legend(fig, hs, [h.get_label() for h in hs])
        return fig
    ax.legend(handles=[Patch(facecolor=tint(ORIG, 0.6), label="original, two noise seeds")] +
              [Line2D([], [], marker="o", color=EARLY_RAMP[j + 1], ls="none", ms=8, label=c.replace("dense, ", ""))
               for j, c in enumerate(studs)], loc="upper right", fontsize=11.5, ncol=1)
    note(fig, "Left: an upper bound that uses the real simulator at decision time, not a policy (LEARNING_FINDINGS.md "
              "section 3). Right: students imitating its choices stay inside the original's run-to-run range; D and L "
              "stay at 0. The original's own plans are too similar to contain new behaviour. Data: early_attempts.csv.")
    return fig


def fig_distill_early():
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.22})
    ax = axes[0]
    sh3 = ["T", "T_narrow", "Y"]
    studs = ["lr 3e-5", "lr 1e-4", "lr 3e-5 + 0.1 anchor"]
    for i, s in enumerate(sh3):
        b = ea("distill_round1", "base", s, "successes")
        ax.plot([i - 0.38, i + 0.38], [b, b], color=ORIG, lw=3, zorder=2)
        for j, st in enumerate(studs):
            x = i - 0.22 + 0.22 * j
            v10 = ea("distill_round1", f"{st}, epoch 10", s, "successes")
            v20 = ea("distill_round1", f"{st}, epoch 20", s, "successes")
            ax.plot([x, x], [v10, v20], color=EARLY_RAMP[j + 1], lw=1.4, zorder=2)
            ax.plot(x, v10, "o", mfc="white", mec=EARLY_RAMP[j + 1], mew=1.6, ms=7, zorder=3)
            ax.plot(x, v20, "o", color=EARLY_RAMP[j + 1], ms=8, mec="white", zorder=4)
    ax.set_xticks(range(3), [disp(s) for s in sh3])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 64)
    ax.grid(False, axis="x")
    ax.set_ylabel("Successes of 100")
    ax.set_title("1. First distillation\n(validation)", loc="left", fontsize=14)
    ax.legend(handles=[Line2D([], [], color=ORIG, lw=3, label="original")] +
              [Line2D([], [], marker="o", color=EARLY_RAMP[j + 1], ls="none", ms=8, label=st)
               for j, st in enumerate(studs)] +
              [Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=7, label="hollow: epoch 10")],
              loc="upper right", fontsize=11, ncol=1)
    ax = axes[1]
    conds = ea_conditions("distill_round3_4_dagger")
    base, studs = conds[0], conds[1:]
    nice = {"round 3, encoder trained, 30 epochs": "round 3: train the encoder too",
            "round 3, frozen encoder, stronger anchor": "round 3: frozen encoder",
            "round 4, encoder trained, 60 epochs (r4e60)": "round 4: encoder, 60 epochs",
            "DAgger iteration 1": "DAgger iteration 1", "DAgger iteration 2": "DAgger iteration 2"}
    for i, s in enumerate(FIVE_SHAPES):
        r = ea_row("distill_round3_4_dagger", base, s, "successes")
        lo, hi = float(r["value"]), float(r["value_high"] or r["value"])
        ax.add_patch(Rectangle((i - 0.42, lo - 0.4), 0.84, hi - lo + 0.8, fc=tint(ORIG, 0.6), ec="none", zorder=1))
        for j, c in enumerate(studs):
            r = ea_row("distill_round3_4_dagger", c, s, "successes")
            v, vh = float(r["value"]), _f(r["value_high"])
            x = i - 0.32 + 0.16 * j
            if vh is not None:
                ax.plot([x, x], [v, vh], color=EARLY_RAMP[j], lw=2.5, zorder=3)
            ax.plot(x, v if vh is None else (v + vh) / 2, "o", color=EARLY_RAMP[j], ms=7.5, mec="white", zorder=4)
    ax.set_xticks(range(5), [disp(s) for s in FIVE_SHAPES])
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(0, 64)
    ax.grid(False, axis="x")
    ax.set_title("2. Train the encoder, longer, DAgger (monitor test)", loc="left", fontsize=14)
    ax.legend(handles=[Patch(facecolor=tint(ORIG, 0.6), label="original (two seeds)")] +
              [Line2D([], [], marker="o", color=EARLY_RAMP[j], ls="none", ms=8, label=nice.get(c, c))
               for j, c in enumerate(studs)], loc="upper left", bbox_to_anchor=(0.01, 1.0), fontsize=11, ncol=2,
              columnspacing=0.8)
    ax.annotate("first real D and L\nsuccesses from images", (3.0, 5.5), xytext=(2.35, 20), fontsize=12, color=INK,
                ha="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    note(fig, "Distilling the planner teacher into the camera policy with the policy's own loss (5 shapes, Oct 26-27). "
              "First every student was worse than the original, also on the letters the teacher solves; training the "
              "image encoder too, longer training and DAgger gave the first D and L successes but 10-15 points less T. "
              "Data: early_attempts.csv (LEARNING_FINDINGS.md section 6).")
    return fig


def fig_bptt_teacher():
    fig, ax = plt.subplots(figsize=(10, 4.8) if not slide() else FULL)
    runs = ea_conditions("bptt_teacher")
    for i, s in enumerate(FIVE_SHAPES):
        cs = [ea("bptt_teacher", r, s, "final_coverage") for r in runs]
        ax.plot([min(cs), max(cs)], [i, i], color=INK2, lw=6, alpha=0.25, solid_capstyle="round", zorder=2)
        ax.scatter(cs, [i] * len(cs), s=26, color=INK2, zorder=3, lw=0)
        o = float(BYSHAPE[("D040", "original")][s]["mean_final_coverage"])
        p = ea("sampling_planner", "sampling planner teacher (final, exact state)", s, "final_coverage")
        ax.plot(o, i, "o", color=ORIG, ms=11, mec="white", zorder=4)
        ax.plot(p, i, "D", color=TEACHER, ms=10, mec="white", zorder=4)
    ax.axvline(0.95, color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.text(0.945, -0.75, "success needs >95%", ha="right", va="bottom", fontsize=12, color=INK2)
    ax.set_yticks(range(5), [disp(s) for s in FIVE_SHAPES])
    ax.set_ylim(4.5, -0.9)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Mean final goal coverage in the real simulator")
    legend_top(ax, handles=[Line2D([], [], marker="o", color=INK2, ls="none", ms=6,
                                   label=f"gradient-based teacher ({len(runs)} runs, 250-2000 iterations)"),
                            Line2D([], [], marker="o", color=ORIG, ls="none", ms=10, label="original policy"),
                            Line2D([], [], marker="D", color=TEACHER, ls="none", ms=9, label="sampling planner teacher")],
               ncol=2 if not slide() else 1, fontsize=12)
    note(fig, "Training a teacher network by backpropagating through the learned world model (SuperMimic-style BPTT): it "
              "learned to approach the block but never pushed it anywhere (0 of 20 successes per shape in every run). "
              "The sampling planner in the same model solves most D, Y and L scenes. Data: early_attempts.csv "
              "(LEARNING_EXPERIMENT_LOG.md section 6, LEARNING_FINDINGS.md section 5); original: final_tests_by_shape.csv "
              "(D040 pool).")
    return fig


def fig_copying_gap():
    conds = ea_conditions("copying_gap")
    pick = [(conds[0], TEACHER, "teacher (exact state + 1 px noise)", None),
            (conds[4], "white", "student reading exact state, re-plans every control", INK),
            (conds[3], INK2, "student reading exact state, 8 controls per plan", None),
            (conds[7], FIVE, "camera student, official recipe (30k steps)", None),
            (conds[1], ORIG, "original camera policy", None)]
    groups = FIVE_SHAPES + ["all"]
    fig, ax = plt.subplots(figsize=FULL)
    nk = len(pick)
    w = 0.84 / nk
    totals = []
    for j, (c, fc, lbl, ec) in enumerate(pick):
        vals = [ea("copying_gap", c, s, "successes") for s in FIVE_SHAPES]
        tot = int(sum(vals))
        totals.append(tot)
        v = [100 * x / 50 for x in vals] + [100 * tot / 250]
        x = np.arange(len(groups)) - 0.42 + w * (j + 0.5)
        ax.bar(x, v, w, color=fc, edgecolor=ec or "white", lw=1.3 if ec else 1.0, zorder=3, label=lbl)
        ax.text(x[-1], v[-1] + 1.2, f"{v[-1]:.0f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold" if j == 0 else None)
    ax.axvline(len(FIVE_SHAPES) - 0.5, color=AXIS, lw=1)
    ax.set_xticks(range(len(groups)), [disp(s) for s in FIVE_SHAPES] + ["All 5"])
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("Validation success\n(same 50 scenes per shape)")
    legend_top(ax, ncol=2, fontsize=12)
    note(fig, f"D044: even a student that reads the exact state, like the teacher, keeps only {totals[2]} or {totals[1]} of "
              f"the teacher's {totals[0]} successes ({100 * totals[2] / totals[0]:.0f}-{100 * totals[1] / totals[0]:.0f}%): "
              "most of the skill is lost in copying, not in seeing. Data: early_attempts.csv (LEARNING_FINDINGS.md "
              "section 6).")
    return fig


def fig_lastmile_replanning():
    rows = []
    for c in ea_conditions("lastmile"):
        r = [x for x in EARLY if x["attempt"] == "lastmile" and x["condition"] == c][0]
        lbl = (SL(f"last-mile vs plain, comparison {c.split('comparison ')[1][0]}",
                  f"last-mile {c.split('comparison ')[1][0]}") if "comparison" in c
               else SL(f"last-mile, 26-letter pool, check {c[-1]}", f"last-mile, 26 l. {c[-1]}"))
        rows.append((lbl, _f(r["value"]), _f(r["ci_low"]), _f(r["ci_high"]), FIVE))
    r = [x for x in EARLY if x["attempt"] == "replanning"][0]
    rows.append((SL("re-plan every 4 controls (5 shapes)", "re-plan every 4"), _f(r["value"]), _f(r["ci_low"]),
                 _f(r["ci_high"]), FIVE))
    for k in (12, 15):
        d = paired_row("D056", "execute 8 (trained)", f"execute {k}", "27 trained shapes")
        rows.append((SL(f"execute {k} of 16 steps (26 letters)", f"execute {k}"), d["difference"], d["ci95_low"],
                     d["ci95_high"], M96))
    fig, ax = plt.subplots(figsize=(10, 4.8) if not slide() else FULL)
    for i, (lbl, d, lo, hi, c) in enumerate(rows):
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=c, lw=2.4, solid_capstyle="butt", zorder=3)
            ax.plot(d, i, "o", color=c, ms=9, mec="white", mew=1.4, zorder=4)
        else:
            ax.plot(d, i, "o", mfc="white", mec=c, mew=1.8, ms=9, zorder=4)
        ax.text((hi if hi is not None else d) + (0.8 if not slide() else 1.8), i, f"{d:+.1f}".replace("-", "−"),
                va="center", fontsize=12)
    ax.axvline(0, color=INK2, lw=1.1)
    ax.axhline(4.5, color=AXIS, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=None if slide() else 12.5)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(-26, 11)
    ax.xaxis.set_major_formatter(SIGNED)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel(SL("Change in success (points) vs. the matched plain run, 95% interval", "Change (points)"))
    if slide():
        ax.set_xticks([-20, -10, 0, 10])
    if not slide():
        ax.text(10.5, 1.0, "near-goal steps 3x,\ntwice the corrections:\nno detectable gain", ha="right", va="center",
                fontsize=12, color=INK2)
        ax.text(10.5, 6.0, "executing 8 planned\nsteps is best in\nboth directions", ha="right", va="center",
                fontsize=12, color=INK2)
    note(fig, "Hollow: the interval includes zero but its bounds were not recorded. Last-mile and re-planning: 5-shape "
              "students, 100 validation scenes per letter (early_attempts.csv, LEARNING_FINDINGS.md section 6). Executing "
              "12 or 15 steps: round-50 26-letter student, 200 fresh scenes per shape (final_tests_episodes.csv, D056).")
    return fig


VERDICT_COLOR = {"worked": M96, "partly": tint(M96, 0.55), "did not work": ORIG, "diagnosis": "white"}


def fig_project_timeline():
    import datetime as dt
    rows = [r for r in TIMELINE if not slide() or r["on_slide"] == "1"]
    d0 = dt.date(2026, 10, 24)
    day = lambda s: (dt.date.fromisoformat(s) - d0).days
    sl = slide()
    fig, ax = plt.subplots(figsize=(10, 6.4) if not sl else FULL)
    for i, r in enumerate(rows):
        a, b = day(r["start"]), day(r["end"]) + 1
        fc = VERDICT_COLOR[r["verdict"]]
        ax.barh(i, b - a, left=a, height=0.62, color=fc, edgecolor=INK2 if r["verdict"] == "diagnosis" else "white",
                lw=1.2, zorder=3)
        if not sl:
            ax.text(15.3, i, r["outcome"].replace("->", "→"), va="center", ha="left", fontsize=11.5, color=INK)
    ax.set_yticks(range(len(rows)), [r["slide_label"] if sl else r["approach"] for r in rows], fontsize=None if sl else 12)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ticks = list(range(0, 15, 4))
    ax.set_xticks(ticks, [(d0 + dt.timedelta(days=t)).strftime("%b %-d") for t in ticks])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, 15 if sl else 15.2)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    if not sl:
        ax.set_xlabel("Date (2026)", x=0.33)
    handles = [Patch(facecolor=VERDICT_COLOR[k], edgecolor=INK2 if k == "diagnosis" else "white", label=k)
               for k in ("worked", "partly", "did not work", "diagnosis") if any(r["verdict"] == k for r in rows)]
    legend_top(ax, handles=handles, ncol=4, fontsize=12)
    if not sl:
        fig.subplots_adjust(left=0.3, right=0.585)
    note(fig, "Each bar spans the days the approach was worked on (inclusive). Data: timeline.csv (dates from "
              "LEARNING_EXPERIMENT_LOG.md timeline, LEARNING_FINDINGS.md and DECISIONS.md D033-D056).")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Teacher: state noise and planner fixes (teacher_planner.csv)
# ----------------------------------------------------------------------------------------------------------------------
GOLD_RAMP = [tint(TEACHER, 0.6), TEACHER, "#a26e00"]       # ordinal light -> dark (more state noise)


def fig_teacher_state_noise():
    sl = slide()
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1, 1.15] if not sl else [1, 1],
                                                                "wspace": 0.28})
    ax = axes[0]
    conds = tp_conditions("state_noise_50")
    shapes = ["T", "T_narrow", "L"]
    w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        k = [int(tp("state_noise_50", c, s)[0]["successes"]) for s in shapes]
        v = [2 * x for x in k]
        lo, hi = zip(*[wilson(x, 50) for x in k])
        x = np.arange(3) - 0.4 + w * (j + 0.5)
        ax.bar(x, v, w, color=GOLD_RAMP[j], edgecolor="white", zorder=3,
               label={"exact simulator state": "exact state"}.get(c, c.replace(", ", " / ").replace(" deg", "°")))
        if not sl:
            errbars(ax, x, v, lo, hi)
        for xi, vi, h in zip(x, v, hi if not sl else []):
            ax.text(xi, h + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(range(3), [disp(s) for s in shapes])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("Planner success (validation)")
    ax.set_title("50 scenes per shape" if not sl else "Noise helps", loc="left", fontsize=15)
    legend_top(ax, ncol=1 if not sl else 3, fontsize=12, y=1.08 if not sl else 1.12,
               title="state the planner reads:" if not sl else None, title_fontsize=12)
    ax = axes[1]
    conds = tp_conditions("state_noise_20")
    xpos = {"exact simulator state": 0, "0.5 px, 0.5 deg": 0.5, "1 px, 1 deg": 1, "2 px, 2 deg": 2, "4 px, 4 deg": 4}
    # letters by marker and shade of grey (the gold shades on the left mean noise levels), labelled at the line end
    styles = {"Y": (INK, "o", "-"), "D": (INK2, "s", (0, (5, 2))), "L": (MUTED, "^", (0, (1.5, 1.5)))}
    for s, (c, mk, ls) in styles.items():
        xs = [xpos[cc] for cc in conds if cc in xpos]
        ys = [100 * int(tp("state_noise_20", cc, s)[0]["successes"]) / 20 for cc in conds if cc in xpos]
        ax.plot(xs, ys, color=c, marker=mk, ls=ls, lw=2.0, ms=8, mec="white", zorder=3)
        ax.text(4.14, ys[-1] + {"Y": 1.5, "D": -3.5, "L": 0}[s], disp(s), ha="left", va="center", fontsize=13,
                color=c, fontweight="bold")
    ax.axvspan(-0.15, 2.15, color=BAND, lw=0, zorder=0)
    if not sl:
        ax.text(1.0, 6, "no loss up to\n2 px / 2°", ha="center", va="bottom", fontsize=12, color=INK2)
        ax.set_xticks([0, 0.5, 1, 2, 4], ["exact", "0.5", "1", "2", "4"])
    else:
        ax.set_xticks([0, 1, 2, 4], ["0", "1", "2", "4"])
    ax.set_xlim(-0.25, 4.5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_xlabel(SL("Noise per reading (px and degrees)", "Noise (px and °)"))
    ax.set_title("Estimator accuracy sweep, 20 scenes" if not sl else "Accuracy needed", loc="left", fontsize=15)
    note(fig, "The planner teacher reads the simulator state with independent Gaussian noise each control. Left: 1 px / "
              "1 degree noise turns T from 5 into 34 of 50 (the exact-state planner repeats the same nearly-right push). "
              "Right: an image-based estimator would need about 2 px / 2 degrees (0.4 of a 96 px camera pixel). Data: "
              "teacher_planner.csv (LEARNING_FINDINGS.md section 6).")
    return fig


FIX_SHORT = ["start", "gate\nmargin 0", "push\ncand.", "horizon\n48", "no dist.\npenalty", "sharp\nkernel",
             "area\npoints"]


def fig_method_planner_fixes():
    sl = slide()
    fig, axes = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1, 1.75], "wspace": 0.3})
    ax = axes[0]
    order = ["D", "Y", "L", "T_narrow", "T"]
    first, final = tp_conditions("first_vs_final_planner")[:2]
    for i, s in enumerate(order):
        a = tp("first_vs_final_planner", first, s)[0]
        b = tp("first_vs_final_planner", final, s)[0]
        va, vb = 100 * int(a["successes"]) / int(a["of"]), 100 * int(b["successes"]) / int(b["of"])
        ax.plot([va, vb], [i, i], color=tint(TEACHER, 0.45), lw=2.4, zorder=2)
        ax.plot(va, i, "o", mfc="white", mec=TEACHER, mew=1.8, ms=9, zorder=3)
        ax.plot(vb, i, "o", color=TEACHER, ms=10, mec="white", zorder=4)
        ax.text(vb + 7, i, f"{a['successes']} → {b['successes']} of {b['of']}", va="center", fontsize=12)
    ax.set_yticks(range(len(order)), [disp(s) for s in order])
    ax.set_ylim(len(order) - 0.5, -0.7)
    ax.set_xlim(-4, 150 if not sl else 165)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Success, 20 validation scenes")
    ax.set_title("First → final planner", loc="left", fontsize=15)
    ax = axes[1]
    rows = tp("t_fix_progression")
    cov = [100 * float(r["final_coverage"]) for r in rows]
    x = np.arange(len(rows))
    ax.plot(x, cov, color=TEACHER, marker="o", ms=8, mec="white", lw=2.2, zorder=3)
    for xi, r, c in zip(x, rows, cov):
        if int(r["successes"]) > 0:
            ax.text(xi, c - 1.6, f"{r['successes']}/{r['of']}\nsolved", ha="center", va="top", fontsize=11.5,
                    color=INK)
    ax.axhline(95, color=INK, lw=1.1, ls=(0, (3, 2)))
    ax.text(0, 95.4, "success needs >95%", ha="left", va="bottom", fontsize=12, color=INK2)
    ax.set_xticks(x, FIX_SHORT if not sl else [str(i) for i in x], fontsize=11.5 if not sl else None)
    ax.set_ylim(60, 101)
    ax.yaxis.set_major_formatter(PCT)
    ax.grid(False, axis="x")
    ax.set_ylabel("T final coverage (12 scenes)")
    ax.set_title("T: one fix per measured stall cause", loc="left", fontsize=15)
    note(fig, "Each fix on the right removed one stall cause the logs showed (planner_fixes.tex): random paths never go "
              "around the letter, the elite average blurred distinct pushes, phantom pushes while hovering, a distance "
              "penalty that made going around too costly, a weak reward for the last 1-2 px, and a pose error measured at "
              "corners. Data: teacher_planner.csv (LEARNING_EXPERIMENT_LOG.md section 7, LEARNING_FINDINGS.md section 5).")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Data and training (training_rounds.csv, training_loss_curves.csv)
# ----------------------------------------------------------------------------------------------------------------------
TRAIN_LINES_SHOW = [("5 shapes", "correction rounds", "5 shapes"), ("10 shapes", "96 px main line", "10 shapes (96 px)"),
                    ("27 shapes", "96 px line", "26 letters + near-T (96 px)")]
SOURCE_COLORS = [("human_T_examples", ORIG, "human T demonstrations (original data)"),
                 ("teacher_examples", TEACHER, "teacher demonstrations"),
                 ("correction_examples", M96, "corrections (policy's own states, teacher labels)")]


def train_rows(stage, line):
    rows = [r for r in TRAIN if r["stage"] == stage and r["line"] == line]
    return sorted(rows, key=lambda r: int(r["round"]))


def complete_round(stage, line, rnd, steps):
    n = sum(1 for r in LOSS if r["stage"] == stage and r["line"] == line and int(r["round"]) == rnd)
    return n * 500 >= int(steps)


def fig_data_growth():
    sl = slide()
    show = TRAIN_LINES_SHOW if not sl else TRAIN_LINES_SHOW[1:]
    widths = [len(train_rows(s, l)) + 1 for s, l, _ in show]
    fig, axes = plt.subplots(1, len(show), figsize=FULL, sharey=True, gridspec_kw={"width_ratios": widths, "wspace": 0.08})
    for ax, (stage, line, title) in zip(axes, show):
        rows = train_rows(stage, line)
        x = np.array([int(r["round"]) for r in rows])
        ys = [np.array([int(r[c]) for r in rows]) / 1e6 for c, _, _ in SOURCE_COLORS]
        ax.stackplot(x, *ys, colors=[c for _, c, _ in SOURCE_COLORS], edgecolor="white", lw=0.8, zorder=2)
        ax.set_title(title if not sl else title.split(" (")[0].replace(" + near-T", ""), loc="left", fontsize=14)
        ax.set_xlim(x[0], x[-1])
        ax.grid(False, axis="x")
        tot = sum(y[-1] for y in ys)
        ax.text(x[-1], tot + 0.05, f"{tot:.2g}M", ha="right", va="bottom", fontsize=12, color=INK)
        if stage == "27 shapes":      # extra correction scenes from round 14: the line spans the corrections band
            ax.axvline(13.5, color=INK2, lw=1, ls=(0, (3, 2)), zorder=3, ymin=(ys[0][-1] + ys[1][-1]) / 3.6)
            if not sl:
                ax.text(14.2, 3.45, "60 practice scenes per\nnew letter from round 14", fontsize=11.5, color=INK2,
                        va="top")
            # the human T band (24k examples) is too thin to see at this scale: label it directly
            hT = ys[0][-1]
            ax.annotate(f"human T: {hT * 1000:.0f}k", (x[-1] - 2.5, hT / 2), xytext=(x[-1] - 2.5, 0.36 if sl else 0.3),
                        ha="right", va="center", fontsize=12, color=INK,
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=10))
            if sl:     # direct labels instead of a legend
                ax.text(14.2, 3.55, "more scenes for\nthe new letters", ha="left", va="top", color=INK2)
                ax.text(2, ys[0][-1] + ys[1][-1] / 2, "teacher demos", ha="left", va="center", color=INK)
                ax.text(30, (ys[0][-1] + ys[1][-1]) + 0.9, "corrections", ha="center", va="center", color="white")
    axes[0].set_ylim(0, 3.6)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}M" if v else "0"))
    axes[0].set_ylabel("Training examples")
    fig.supxlabel("Training round", fontsize=16, y=0.0 if not sl else None)
    if not sl:
        legend_top(axes[0], handles=[patch(c, l) for _, c, l in SOURCE_COLORS[::-1]], ncol=3, fontsize=12, y=1.1)
    note(fig, "Examples are training samples (one per recorded step). Every round retrains on everything collected so far: "
              "the teacher's demonstrations, all correction rounds of all stages and the 206 original human T "
              "demonstrations. Each stage starts with the corrections of the stage before. Data: training_rounds.csv.")
    return fig


def fig_train_loss_rounds():
    sl = slide()
    cols = [("10 shapes", "96 px main line", "10 shapes, 96 px", TEN), ("27 shapes", "96 px line", "26 letters, 96 px", M96)]
    if sl:
        cols = cols[1:]
    fig, axes = plt.subplots(2, len(cols), figsize=(10, 6.2) if not sl else FULL, sharex="col",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.22, "width_ratios": [23, 51][-len(cols):]},
                             squeeze=False)
    for j, (stage, line, title, c) in enumerate(cols):
        rows = [r for r in train_rows(stage, line) if r["loss_last_1000"] and int(r["round"]) > 0
                and complete_round(stage, line, int(r["round"]), r["steps"])]
        x = np.array([int(r["round"]) for r in rows])
        y = np.array([float(r["loss_last_1000"]) for r in rows])
        ax = axes[0, j]
        ax.plot(x, y, color=c, marker="o", ms=3.5, lw=2)
        ax.set_title(title, loc="left", fontsize=15)
        ax.grid(False, axis="x")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
        s, v, k, n = val_line(stage, line)
        ax2 = axes[1, j]
        ax2.plot(s - 1, v, color=c, lw=1, alpha=0.45, marker="o", ms=2.5)
        ax2.plot(s - 1, rolling(v, 5), color=c, lw=2.6)
        ax2.yaxis.set_major_formatter(PCT)
        ax2.grid(False, axis="x")
        ax2.set_ylim(0, 45)
        ax2.set_xlabel("Correction rounds completed")
        if stage == "27 shapes":
            for a in (ax, ax2):
                a.axvline(13.5, color=INK2, lw=1, ls=(0, (3, 2)))
            if not sl:
                ax.text(14.2, ax.get_ylim()[1], "54k steps per round from round 14", va="top", fontsize=11.5,
                        color=INK2)
    axes[0, 0].set_ylabel("Training loss\n(end of round)")
    axes[1, 0].set_ylabel("Validation\nsuccess")
    note(fig, "Top: mean training loss over the last 1000 steps of each round (round 0, a 30k-step retraining from the "
              "previous stage's student, and rounds that stopped early are left out). "
              "Bottom: validation success (faint) and its 5-round rolling mean. The loss keeps falling while validation "
              "stops improving: the student fits its training data better without acting better on new scenes. Data: "
              "training_rounds.csv, validation_rounds.csv.")
    return fig


def fig_train_loss_within_round():
    stage, line = "27 shapes", "96 px line"
    picks = [1, 13, 14, 30, 50]
    cols = [SEQ(v) for v in np.linspace(0.38, 1.0, len(picks))]
    fig, ax = plt.subplots(figsize=FULL if slide() else (10, 5))
    for rnd, c in zip(picks, cols):
        rows = [r for r in LOSS if r["stage"] == stage and r["line"] == line and int(r["round"]) == rnd]
        x = np.array([int(r["step_end"]) for r in rows]) / 1000
        y = np.array([float(r["mean_loss"]) for r in rows])
        ax.plot(x, y, color=c, lw=2.2, label=f"round {rnd}")
        ax.text(x[-1] + 0.6, y[-1], f"round {rnd}", va="center", fontsize=12, color=INK)
    ax.axvline(20, color=AXIS, lw=1)
    if not slide():
        ax.text(19.5, ax.get_ylim()[1], "rounds 1-13 end\nat 20k steps", ha="right", va="top", fontsize=11.5, color=INK2)
    ax.set_xlim(0, 62)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(False, axis="x")
    ax.set_xlabel("Training steps within the round (thousands)")
    ax.set_ylabel("Training loss (mean per 500 steps)")
    note(fig, "26-letter 96 px line. Each round restarts the learning-rate schedule (500 warm-up steps, cosine decay; "
              "scripts/learning/bc_train.py) from the previous round's student. Data: training_loss_curves.csv.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# failure_categories: how far the unsolved episodes got (final_tests_episodes.csv)
# ----------------------------------------------------------------------------------------------------------------------
FAIL_BINS = [("solved", None), ("near miss: ≥ 90%", 0.9), ("partial: 50-90%", 0.5), ("far: < 50%", 0.0)]


def _fail_shares(test, model, shapes, which):
    d = eps(test, model, shapes)
    n = len(d["success"])
    solved = d["success"] == 1
    v = d[which]
    near = (~solved) & (v >= 0.9)
    part = (~solved) & (v >= 0.5) & (v < 0.9)
    far = (~solved) & (v < 0.5)
    return [100 * m.sum() / n for m in (solved, near, part, far)], n


def fig_failure_categories():
    """Original vs the round-44 student on the same D051 scenes; in the default version also round 44 vs round 50 on
    their own (different) D054 pool, in a separate, labelled block, never next to the original."""
    sl = slide()
    blocks = [("Final test: same 300 fresh scenes per shape (D051)",
               [("Original", "D051", "original", ORIG_LINE),
                (SL("Ours, 96 px (round 44)", "Ours, 96 px\n(round 44)"), "D051", D051["m96"], M96)])]
    if not sl:
        blocks.append(("Different scene pool (D054): round 44 vs round 50, 200 fresh scenes per shape",
                       [("Ours, round 44", "D054", "26-letter 96 px round 44", M96),
                        ("Ours, round 50", "D054", "26-letter 96 px round 50", M96)]))
    rows, heads, y = [], [], 0.0
    for head, entries in blocks:
        heads.append((y - 0.62, head))
        for e in entries:
            rows.append((y, *e))
            y += 1
        y += 0.75
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4) if not sl else FULL, sharey=True, gridspec_kw={"wspace": 0.2})
    for ax, which, title in ((axes[0], "peak", SL("Best coverage reached", "Best coverage")),
                             (axes[1], "final", SL("Coverage at the end", "Final coverage"))):
        for yy, name, test, model, c in rows:
            parts, n = _fail_shares(test, model, TRAINED27, which)
            shades = [c, tint(c, 0.45), tint(c, 0.72), "#efeee9"]
            left = 0
            for j, (v, cc) in enumerate(zip(parts, shades)):
                ax.barh(yy, v, left=left, color=cc, edgecolor="white", lw=1.5, height=0.66, zorder=3)
                if v >= (6 if not sl else 15):
                    ax.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=12,
                            color="white" if j == 0 and c != ORIG_LINE else INK)
                left += v
        ax.set_title(title, loc="left", fontsize=15)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PCT)
        ax.grid(False)
        ax.set_xlabel(SL("Share of episodes, 27 trained shapes", "Share of episodes"))
        if sl:
            ax.set_xticks([0, 50, 100])
    axes[0].set_yticks([r[0] for r in rows], [r[1] for r in rows])
    axes[0].set_ylim(rows[-1][0] + 0.5, -0.5 if sl else -1.0)
    if not sl:
        for ax in axes:                      # block headers run across both panels
            ax.patch.set_alpha(0)
            ax.axhline(heads[1][0] - 0.42, color=AXIS, lw=1.0)
        for yh, head in heads:
            axes[0].text(0, yh, head, ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK2,
                         clip_on=False, zorder=6)
    shades = [Patch(facecolor=INK2, label=SL("solved (> 95%)", "solved")),
              Patch(facecolor=tint(INK2, 0.45), label=SL("near miss ≥ 90%", "≥ 90%")),
              Patch(facecolor=tint(INK2, 0.72), label=SL("partial 50-90%", "50-90%")),
              Patch(facecolor="#efeee9", label=SL("far < 50%", "< 50%"))]
    legend_top(axes[0], handles=shades, ncol=4 if not sl else 2, fontsize=12, y=1.08)
    d44 = _fail_shares("D051", D051["m96"], TRAINED27, "peak")[0]
    note(fig, f"Unsolved episodes split by coverage: of the round-44 student's unsolved episodes, "
              f"{100 * d44[1] / (100 - d44[0]):.0f}% got within 90% coverage at some moment. Top: original and round 44 on "
              "the final-test scenes (D051), 300 fresh scenes per shape. Bottom: round 44 and round 50 on their own, "
              "different fresh pool (D054, 200 per shape); compare round 50 only with round 44 there. Shades of each "
              "policy's colour show the same four outcomes. Data: final_tests_episodes.csv.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# LaTeX tables
# ----------------------------------------------------------------------------------------------------------------------
def _ci(d):
    return f"{fmt_signed(d['difference'])} [{fmt_signed(d['ci95_low'])}, {fmt_signed(d['ci95_high'])}]"


def _tex(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("→", "$\\rightarrow$").replace("≥", "$\\geq$").replace("≈", "$\\approx$")
            .replace("°", "$^\\circ$").replace("×", "$\\times$").replace("−", "$-$"))


def write_new_tables(head):
    # world model accuracy
    S = "structured (wm2 v1)"
    lines = [head, "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "& \\multicolumn{4}{c}{8 controls} & \\multicolumn{2}{c}{32 controls} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
             "Model, shape & median px & p90 px & median deg & p90 deg & median px & p90 px \\\\", "\\midrule"]
    g = [wm1("contact_error", model="generic MLP", horizon=h, statistic="median", unit=u) for h, u in
         ((8, "px"), (8, "deg"), (32, "px"))]
    lines.append(f"Generic network, all 5 shapes & {g[0]:.2f} & -- & {g[1]:.2f} & -- & {g[2]:.2f} & -- \\\\")
    lines.append("\\midrule")
    for s in FIVE_SHAPES:
        v = [wm1("contact_error", model=S, shape=s, horizon=h, statistic=st, unit=u) for h, st, u in
             ((8, "median", "px"), (8, "p90", "px"), (8, "median", "deg"), (8, "p90", "deg"), (32, "median", "px"),
              (32, "p90", "px"))]
        lines.append(f"Structured, {disp(s)} & " + " & ".join(f"{x:.2f}" for x in v) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"% Held-out contact error (LEARNING_FINDINGS.md section 4). Near the goal: one-step error "
              f"{wm1('near_goal_one_step'):g} px. Planner's best push: {wm1('planner_push_check', statistic='trials within 1 px'):.0f} "
              f"of 15 within 1 px. 500 16-control windows: median {wm1('rare_large_errors', statistic='median'):g} px, "
              f"1 in 20 at or above {wm1('rare_large_errors', statistic='95th percentile (1 in 20 at or above)'):g} px, "
              f"worst {wm1('rare_large_errors', statistic='worst'):g} px (mostly at the arena wall). Success needs about "
              f"{wm1('requirement', shape='all', unit='px'):g} px."]
    (TAB / "world_model.tex").write_text("\n".join(lines) + "\n")

    # earlier approaches (one line each)
    def blend_cell(t):
        d = {s: ea("blend_tests", t, s, "difference_pts") for s in ("T", "Y", "D")}
        lo = ea("blend_tests", t, "T", "difference_pts", "ci_low")
        ok = ea("blend_tests", t, "all", "passed") == 1
        return (f"Y {fmt_signed(d['Y'])}, D {fmt_signed(d['D'])}, T {fmt_signed(d['T'])} (lower bound "
                f"{fmt_signed(lo)})", "passed" if ok else "failed (T)")
    ob = {s: (ea("oracle_bon", "base policy", s, "successes"), ea("oracle_bon", "simulator best-of-N (upper bound)", s,
                                                                    "successes"), ea("oracle_bon", "base policy", s,
                                                                                     "successes", "of"))
          for s in ("Y", "T_narrow")}
    bp = [ea("bptt_teacher", r, s, "final_coverage") for r in ea_conditions("bptt_teacher") for s in FIVE_SHAPES]
    cg = {c: sum(ea("copying_gap", c, s, "successes") for s in FIVE_SHAPES) for c in ea_conditions("copying_gap")}
    cgk = ea_conditions("copying_gap")
    lm = [ea("lastmile", c, "5 first shapes", "difference_pts") for c in ea_conditions("lastmile")
          if "comparison" in c]
    rp = [r for r in EARLY if r["attempt"] == "replanning"][0]
    d1 = [ea("distill_round1", f"{st}, epoch {e}", "T", "successes") for st in ("lr 3e-5", "lr 1e-4",
                                                                                "lr 3e-5 + 0.1 anchor") for e in (10, 20)]
    dd = {s: max(ea("distill_round3_4_dagger", c, s, "successes") for c in ea_conditions("distill_round3_4_dagger")[1:])
          for s in ("D", "L")}
    rows = [("Simulator best-of-N (upper bound)", "Oct 25",
             f"Y {ob['Y'][0]:.0f} $\\rightarrow$ {ob['Y'][1]:.0f} of {ob['Y'][2]:.0f}, near-T {ob['T_narrow'][0]:.0f} "
             f"$\\rightarrow$ {ob['T_narrow'][1]:.0f} of {ob['T_narrow'][2]:.0f}; imitation within noise", "no new skill"),
            ("First planner distillation", "Oct 26",
             f"T {min(d1):.0f}-{max(d1):.0f} of 100 vs {ea('distill_round1', 'base', 'T', 'successes'):.0f}; worse everywhere",
             "failed"),
            ("Train encoder, longer, DAgger", "Oct 26-27",
             f"first D and L successes (up to {dd['D']:.0f} and {dd['L']:.0f} of 100); T 10-15 points lower", "partly"),
            ("Gradient-based teacher (BPTT)", "Oct 26", f"real coverage {min(bp):.2f}-{max(bp):.2f}, 0 of 20 solved",
             "failed")]
    for t in ea_conditions("blend_tests"):
        cell, out = blend_cell(t)
        day = int(ea_row("blend_tests", t, "all", "passed")["date"][-2:])
        rows.append((_blend_label(t), f"Oct {day}", cell, out))
    rows += [("State-input student (D044)", "Oct 30",
              f"keeps {cg[cgk[3]]:.0f}-{cg[cgk[4]]:.0f} of the teacher's {cg[cgk[0]]:.0f} successes", "diagnosis"),
             ("Last-mile variant", "Oct 31", ", ".join(fmt_signed(v) for v in lm) + " points (intervals include 0)",
              "no gain"),
             ("Re-plan every 4 controls", "Oct 31",
              f"{fmt_signed(float(rp['value']))} [{fmt_signed(float(rp['ci_low']))}, {fmt_signed(float(rp['ci_high']))}] "
              "points", "worse")]
    lines = [head, "\\begin{tabular}{llp{7.2cm}l}", "\\toprule", "Approach & When & Result & Outcome \\\\", "\\midrule"]
    lines += [f"{_tex(a)} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines += ["\\bottomrule", "\\end{tabular}",
              "% Typed in from docs/LEARNING_FINDINGS.md sections 3-6 and docs/DECISIONS.md D040-D044 "
              "(presentation/data/early_attempts.csv). Blend rows: paired change vs. the original in points; D042 uses "
              "97.5% intervals."]
    (TAB / "early_attempts.tex").write_text("\n".join(lines) + "\n")

    # blend tests in full
    lines = [head, "\\begin{tabular}{lrrrrrl}", "\\toprule", "Test & T & near-T & Y & D & L & Outcome \\\\", "\\midrule"]
    for t in ea_conditions("blend_tests"):
        cells = [fmt_signed(ea("blend_tests", t, s, "difference_pts")) for s in FIVE_SHAPES]
        lines.append(f"{_tex(_blend_label(t))} & " + " & ".join(cells) + " & "
                     + ("passed" if ea("blend_tests", t, "all", "passed") == 1 else "failed") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Paired change in success vs. the original policy (points); 1000 "
              "(D040) or 2000 (D041-D043) fresh scenes per letter. Intervals in presentation/data/early_attempts.csv."]
    (TAB / "blend_tests.tex").write_text("\n".join(lines) + "\n")

    # datasets
    lines = [head, "\\begin{tabular}{p{6.4cm}lrrr}", "\\toprule", "Data & Stage & Episodes & Examples & Per shape \\\\",
             "\\midrule"]
    for r in DSIZES:
        ep = f"{int(r['episodes']):,}" if r["episodes"] else "--"
        ex = f"{int(r['examples']):,}" if r["examples"] else "--"
        ps = r["per_shape"]
        ps = (f"{int(ps) // 1000}k" if ps.isdigit() and int(ps) >= 10000 else _tex(ps)) or "--"
        lines.append(f"{_tex(r['item'])} & {_tex(r['stage'])} & {ep} & {ex} & {ps} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% presentation/data/dataset_sizes.csv (sources per row): episodes = "
              "scenes, examples = training samples (one per recorded step); per shape = episodes, scenes per round or "
              "samples per round as the row says."]
    (TAB / "datasets.tex").write_text("\n".join(lines) + "\n")

    # teacher vs student per shape on the same validation scenes
    sv = student_val(TRAINED27)
    lines = [head, "\\begin{tabular}{lrrl@{\\hspace{1.5em}}lrrl}", "\\toprule",
             "Shape & Teacher & Student & WM & Shape & Teacher & Student & WM \\\\", "\\midrule"]
    order = OLD10 + NEW17
    half_n = math.ceil(len(order) / 2)
    for i in range(half_n):
        cells = []
        for blk in (order[:half_n], order[half_n:]):
            if i < len(blk):
                s = blk[i]
                cells.append(f"{disp(s)} & {TEACHER_VAL[s][0]} & {sv[s][0]} & {TEACHER_VAL[s][2].replace('wm2 ', '')}")
            else:
                cells.append(" & & & ")
        lines.append(" & ".join(cells) + " \\\\")
    kt = sum(v[0] for v in TEACHER_VAL.values()); ks = sum(v[0] for v in sv.values())
    lines += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{All 27: teacher {kt}, student {ks} of 1350}} \\\\",
              "\\bottomrule", "\\end{tabular}",
              f"% Successes of 50 validation scenes per shape (seeds 400000 + 20*i). Teacher: teacher_validation.csv; "
              f"WM = world-model version of the teacher check. Student: 26-letter 96 px round {STUDENT_VAL_ROUND}."]
    (TAB / "teacher_vs_student.tex").write_text("\n".join(lines) + "\n")

    # planner fixes (stall causes)
    lines = [head, "\\begin{tabular}{p{6.2cm}p{6.2cm}rr}", "\\toprule",
             "Change (cumulative) & Stall cause it removed & T solved & T coverage \\\\", "\\midrule"]
    for r in tp("t_fix_progression"):
        lines.append(f"{_tex(r['condition'].split(': ', 1)[1])} & {_tex(r['note']) or '--'} & {r['successes']}/{r['of']} & "
                     f"{float(r['final_coverage']):.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% 12 T validation scenes per row (LEARNING_EXPERIMENT_LOG.md section 7); "
              "stall causes from LEARNING_FINDINGS.md section 5 (presentation/data/teacher_planner.csv)."]
    (TAB / "planner_fixes.tex").write_text("\n".join(lines) + "\n")


def write_tables():
    TAB.mkdir(parents=True, exist_ok=True)
    head = "% Generated by presentation/scripts/make_figures.py from presentation/data/*.csv. Needs \\usepackage{booktabs}.\n"
    # headline
    groups = ["10 earlier shapes", "17 other letters", "27 trained shapes", "6 held-out shapes"]
    lines = [head, "\\begin{tabular}{lrrrr}", "\\toprule",
             "Shapes & Original & Ours, 10 shapes & Ours, 96 px (round 44) & Ours, 192 px (round 2) \\\\", "\\midrule"]
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
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %; final test (D051), 300 fresh scenes per shape, paired by "
              "scene; differences in points with 95% intervals. 96 px and 192 px: the 27-shape students of rounds 44 and "
              "2 (round 50 was tested separately on another pool, D054: round50_vs_44.tex)."]
    (TAB / "final_headline.tex").write_text("\n".join(lines) + "\n")

    # per shape, one long table
    def shape_row(s):
        vals = [rate("D051", D051[k], [s])[0] for k in KEYS4]
        d = paired_row("D051", "original", D051["m96"], s)
        return f"{disp(s)} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {fmt_signed(d['difference'])} \\\\"
    lines = [head, "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Shape & Original & 10 shapes & 96 px (r.~44) & 192 px (r.~2) & Change (96 px) \\\\", "\\midrule"]
    for block, shapes in (("10 earlier shapes", OLD10), ("17 newer letters", NEW17), ("6 never-seen shapes", HELD)):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{block}}}}} \\\\")
        lines += [shape_row(s) for s in shapes]
        if block != "6 never-seen shapes":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %, final test (D051), 300 fresh scenes per shape; 96 px = 27-shape student round 44, 192 px = "
              "round 2; change = 27-shape "
              "96 px minus original, points."]
    (TAB / "final_per_shape.tex").write_text("\n".join(lines) + "\n")

    # per shape, two blocks side by side (fits one slide at \scriptsize)
    order = OLD10 + NEW17 + HELD
    half = math.ceil(len(order) / 2)
    left, right = order[:half], order[half:]
    lines = [head, "\\begin{tabular}{lrrrr@{\\hspace{1.5em}}lrrrr}", "\\toprule",
             "Shape & Orig. & 10 sh. & 96 px & 192 px & Shape & Orig. & 10 sh. & 96 px & 192 px \\\\", "\\midrule"]
    for i in range(half):
        cells = []
        for blk in (left, right):
            if i < len(blk):
                s = blk[i]
                cells.append(f"{disp(s)} & " + " & ".join(f"{rate('D051', D051[k], [s])[0]:.1f}" for k in KEYS4))
            else:
                cells.append(" & & & & ")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "% Success %, final test (D051), 300 fresh scenes per shape; 96 px and 192 px = the 27-shape "
              "students of rounds 44 and 2. Order: 10 earlier "
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
    lines += ["\\bottomrule", "\\end{tabular}", "% 27-shape 96 px student; separate fresh pool (D054), 200 scenes per shape, paired by scene; not "
              "comparable with the D051 final-test numbers."]
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
    write_new_tables(head)


# ----------------------------------------------------------------------------------------------------------------------
# README
# ----------------------------------------------------------------------------------------------------------------------
def write_readme():
    lines = ["# Presentation figures", "",
             "Generated by `presentation/scripts/make_figures.py` (do not edit by hand). Every figure exists as a vector",
             "PDF (for Beamer) and a 200 dpi PNG (for PowerPoint) with the same name; the method diagrams also as SVG.",
             "Regenerate with:", "",
             "```bash", "MPLBACKEND=Agg python presentation/scripts/make_figures.py                 # everything",
             "MPLBACKEND=Agg python presentation/scripts/make_figures.py --variant slide # only figures/slide*/",
             "```", "",
             "Colours follow the model everywhere: original = grey, 5-shape students = raspberry, 10-shape",
             "students = orange (192 px camera: dark orange), 26-letter 96 px = blue, 26-letter 192 px = green,",
             "planning teacher = gold, forward RL = violet. \"Validation\" = 50 fixed scenes per shape used to pick",
             "checkpoints; \"final test\" = fresh scenes never used before, paired by scene with the original.",
             "Every number a chart uses lives in `presentation/data/`: the tables built from the saved test files",
             "(`validation_rounds.csv`, `final_tests_*.csv`, `training_rounds.csv`, `training_loss_curves.csv`) and the",
             "typed-in tables, each row with the document section it comes from (`typed_in_numbers.json`,",
             "`early_attempts.csv`, `teacher_validation.csv`, `teacher_planner.csv`, `world_model_accuracy.csv`,",
             "`dataset_sizes.csv`, `timeline.csv`). Restyle a chart from those files alone.", "",
             "LaTeX tables (booktabs) are in `presentation/data/tables/`: " + ", ".join(
                 f"`{p.name}`" for p in sorted(TAB.glob("*.tex"))) + ".", "",
             "## Default versions (`figures/`)", "",
             "Sized for the paper and for standalone reading (10 x 5.6 in for full-width charts); most carry a short",
             "footnote with the test protocol and data source.", "",
             "| File | What it shows | Data | Suggested use |", "| --- | --- | --- | --- |"]
    for name, shows, data, use in REGISTRY:
        lines.append(f"| `{name}` | {shows} | {data} | {use} |")
    lines += ["", "## Reading the numbers", "",
              "- Model names. The headline final test (D051, 300 fresh scenes per shape) compares the original, ours with "
              "10 shapes (96 px, round 13), and ours with 27 shapes (26 letters + near-T) at 96 px (round 44) and 192 px "
              "(round 2). Round 50, the model in the videos, was tested separately on another fresh pool (D054, 200 scenes "
              "per shape) only against round 44 (27 shapes: 32.9% -> 34.8%). Charts never put D051 and D054 numbers side "
              "by side: curves_all_stages marks round 50 as 'round 50, fresh test 34.8%'; failure_categories keeps it in "
              "its own labelled block.",
              "- Original policy on the training curves. Dashed = the original on the same 50 validation scenes per "
              "shape (5 shapes: 41 of 250 = 16.4%; 10 shapes: 41 of 500 = 8.2%; typed_in_numbers.json). It was never run "
              "on the 27-shape validation scenes, so the 27-shape level is its D051 fresh-test 3.7%, drawn dotted and "
              "labelled 'fresh test'.",
              "- Teacher on the 10 earlier shapes. method_teacher_success_26letters and "
              "slide_half/method_teacher_success_groups use the latest check per shape (teacher_validation.csv): 400 of "
              "500 on the 10 earlier shapes, because T, D and L were re-checked with world model v3 (T 30, D 49, L 34) "
              "while near-T and Y keep their v1 checks and E, F, J, P, U their v2 checks. method_teacher_gap and "
              "curves_10shapes use 390 of 500, the single run (world model v2) whose teacher labelled the 10-shape "
              "students (DECISIONS.md D048). Both are on the same 50 validation scenes per shape.",
              "- Planner size. The teacher scores 1,024 plans per control: 4 iterations x 256 samples; the 128 structured "
              "push candidates replace 128 of the first iteration's 256 samples (scripts/learning/mpc_teacher.py), so "
              "they are not extra.",
              "- Validation vs fresh test. Validation numbers (50 fixed scenes per shape, used to pick checkpoints, e.g. "
              "analysis_label_ab) are not comparable with fresh-scene final tests."]
    for mode, head, txt in (("slide", "Slide versions (`figures/slide/`)",
                             "6.3 x 3.5 in (full slide width at 16:9) unless the table says otherwise, every text at least "
                             "16 pt, no footnote and no title (say the protocol aloud or put it in the slide's footer). Same "
                             "data as the default version, with shorter labels and sometimes fewer panels or value labels "
                             "(for example final_blends_all shows T, Y and D; data_growth leaves out the 5-shape panel; the "
                             "timeline keeps 8 rows). final_dumbbell is 6.3 x 3.6 in so it fits height=0.8\\textheight; "
                             "curves_all_stages keeps 6.3 x 3.5 in (aspect 1.8), which also fits at that height."),
                            ("half", "Half-slide versions (`figures/slide_half/`)",
                             "4.2 x 3.2 in for two-column slides, every text at least 14 pt, no footnote or title. "
                             "Two-panel charts keep their main panel (heldout_scaling: success; coverage_near_misses: "
                             "27 trained shapes; analysis_budget: validation by phase; analysis_steps_executed: 27 "
                             "trained and 6 unseen; oracle_bon: the best-of-N panel). The grouped teacher-vs-student chart "
                             "is slide_half/method_teacher_success_groups; method_teacher_success_26letters has no half "
                             "version, so \\halffig falls back to the wide per-letter slide version.")):
        rows = [(n, sh) for m, n, sh in VARIANT_REGISTRY if m == mode]
        if not rows:
            continue
        lines += ["", f"## {head}", "", txt, "", "| File | Shows |", "| --- | --- |"]
        assert not any("|" in sh for _, sh in rows), "a '|' would break the README table"
        lines += [f"| `{VARIANT_DIR[mode]}/{n}` | {sh} |" for n, sh in rows]
    (FIG / "README.md").write_text("\n".join(lines) + "\n")


def style_variant(mode):
    """Bigger fonts and an exact canvas for the slide versions; constrained layout keeps everything on the canvas."""
    style()
    f = VARIANT_FLOOR[mode]
    plt.rcParams.update({"font.size": f, "axes.labelsize": f, "axes.titlesize": f, "xtick.labelsize": f,
                         "ytick.labelsize": f, "legend.fontsize": f, "legend.title_fontsize": f,
                         "figure.constrained_layout.use": True, "savefig.bbox": None, "lines.markersize": 5,
                         "legend.handlelength": 1.2, "legend.columnspacing": 1.0, "axes.titlepad": 4})
    global FULL, HALF, TALL, BIG
    FULL = HALF = TALL = BIG = VARIANT_SIZE[mode]


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
    ap.add_argument("--variant", choices=("all", "none", "slide"), default="all",
                    help="all: default figures, slide versions, tables and README; none: default figures only; "
                         "slide: only figures/slide/ and figures/slide_half/")
    args = ap.parse_args()
    ONLY.extend(args.only)
    variant = args.variant
    style()
    jobs = [
        ("curves_all_stages", lambda: fig_curves_all_stages(),
         "Whole training story: validation success per correction round (rounds completed, from 0 in every stage), 5 -> "
         "10 -> 27 shapes on one axis, picks circled, round 50 starred (its own fresh test, 34.8%, D054), round-47 note. "
         "Original policy: dashed at its validation level on the same scenes (5 shapes 16.4%, 10 shapes 8.2%), dotted "
         "and labelled 'fresh test' for 27 shapes (3.7%, D051; it was never run on those validation scenes).",
         f"{V}; original validation levels and picks in typed_in_numbers.json; 27-shape original level and round 50 "
         f"from {S}",
         "story slide (main)",
         "Validation success through all three stages"),
        ("curves_all_stages_smoothed", lambda: fig_curves_all_stages(smooth=True),
         "Same story with faint raw lines and a rolling mean (3 rounds for 5 shapes, 5 rounds otherwise).", V,
         "story slide, cleaner look", None),
        ("curves_all_stages_panels", fig_curves_all_stages_panels,
         "Same story as three side-by-side panels with a shared y-axis (same original levels, circles and star).", V,
         "story slide / paper", None),
        ("curves_26letters", lambda: fig_curves_26letters(),
         "26-letter lines (96 px 50 rounds, 192 px 13 rounds) with 95% bands, larger-budget start (round 14), the "
         "picked rounds (44 and 2) and the round-50 model (starred, fresh test).", f"{V}; round 50 from {S}",
         "26-letter training slide", "Training on all 26 letters"),
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
         "5-shape stage: copying the teacher without corrections vs training steps (left), correction rounds from round 0 "
         "(right); original (16.4%) and the 1 px-noise teacher that labelled these students (76.4%) dashed.",
         f"{V}; reference lines from typed_in_numbers.json", "5-shape slide", None),
        ("curves_heatmap_26letters", lambda: fig_heatmap_26(),
         "Shapes x rounds heatmap of validation success, 26-letter 96 px line.", V, "per-shape progress / paper", None),
        ("curves_heatmap_26letters_smoothed", lambda: fig_heatmap_26(smooth=True),
         "Same heatmap with a 5-round rolling mean per shape (less noise).", V, "per-shape progress", None),
        ("curves_heatmap_26letters_coverage", lambda: fig_heatmap_26(metric="mean_final_coverage"),
         "Shapes x rounds heatmap of mean final coverage, 26-letter 96 px line.", V, "backup / paper", None),
        ("curves_heatmap_26letters_192", lambda: fig_heatmap_26(line="192 px line"),
         "Shapes x rounds heatmap of validation success, 26-letter 192 px line (13 rounds).", V, "backup", None),
        ("final_grouped_bars", lambda: fig_final_grouped_bars(),
         "HEADLINE final test (D051) by shape group (10 earlier, 17 newer, 27 trained, 6 never-seen) for the four "
         "policies: original, ours 10 shapes, ours 27 shapes 96 px (round 44), ours 27 shapes 192 px (round 2); 95% "
         "Wilson intervals.", S, "results slide (main)", "Final test: one camera policy for 26 letters"),
        ("final_headline_bars", lambda: fig_final_grouped_bars(groups=("27 trained shapes", "6 held-out shapes"),
                                                               size=HALF),
         "Compact headline: 27 trained shapes and 6 never-seen shapes only.", S, "half-slide headline", None),
        ("final_stage_progression", fig_final_stage_progression,
         "Each stage's fresh-scene test vs the original on the same scenes: success stays ~34% as shapes grow 5 -> 10 -> "
         "27.", f"{P}, {S}", "summary slide", None),
        ("final_per_shape_bars", lambda: fig_final_per_shape_bars_horizontal(),
         "Horizontal bars for all 33 shapes (sorted), original vs ours, 27 shapes, 96 px (round 44); never-seen shapes "
         "below the line.",
         S, "paper / tall slide", None),
        ("final_per_shape_bars_vertical", fig_final_per_shape_bars_vertical,
         "Same as vertical bars across a 16:9 slide.", S, "per-shape slide", None),
        ("final_per_shape_bars_all_models", lambda: fig_final_per_shape_bars_horizontal(KEYS4, size=(7.5, 13)),
         "Horizontal bars for all 33 shapes and all four policies.", S, "paper appendix", None),
        ("final_dumbbell", lambda: fig_final_dumbbell(),
         "Dumbbell per shape: original -> ours, 27 shapes, 96 px (round 44) (tall).", S, "paper / per-shape slide",
         None),
        ("final_dumbbell_vertical", lambda: fig_final_dumbbell(vertical=True),
         "Dumbbell per shape across a 16:9 slide.", S, "per-shape slide", None),
        ("final_dumbbell_192", lambda: fig_final_dumbbell(key="m192"),
         "Dumbbell per shape: original -> ours, 27 shapes, 192 px (round 2).", S, "backup", None),
        ("final_scatter", lambda: fig_final_scatter(),
         "Original vs ours (27 shapes, 96 px, round 44) success per shape with the no-change line.", S,
         "half-slide / paper", None),
        ("final_scatter_10shape", lambda: fig_final_scatter(ykey="ten96"),
         "Original vs 10-shape 96 px success per shape (17 newer letters barely move).", S, "backup", None),
        ("final_forest_groups", fig_final_forest_groups,
         "Paired change vs original with 95% intervals per shape group for the three students.", P, "results slide",
         None),
        ("final_forest_shapes", fig_final_forest_shapes,
         "Paired change vs original with 95% intervals for every shape (27-shape students, 96 px round 44 and 192 px "
         "round 2).", P,
         "paper appendix", None),
        ("final_10shapes_bars", fig_final_10shapes_bars,
         "10-shape final test per shape, 1000 fresh scenes per shape, original vs 96 px (round 13) vs 192 px (round 6).",
         S,
         "10-shape results slide", None),
        ("final_10shapes_forest", fig_final_10shapes_forest,
         "10-shape paired change vs original per shape with 95% intervals.", P, "backup / paper", None),
        ("final_5shapes_bars", fig_final_5shapes_bars,
         "5-shape final tests after 4 and 7 correction rounds vs original (two fresh pools).", S, "5-shape slide", None),
        ("final_blends_5shapes", fig_final_blends,
         "First weight-blend test (D040, 5 shapes): small Y/D gains, T loss.", S, "backup (superseded by final_blends_all)",
         None),
        ("final_blends_all", fig_final_blends_all,
         "Every pre-registered blend test, D040 -> D043: paired change per letter with intervals, non-inferiority margin, "
         "passed/failed. Two failed on T; the 50% encoder blend passed twice (D042, D043).", "early_attempts.csv",
         "what we tried first / backup", None),
        ("shapes_96_vs_192", fig_shapes_96_vs_192,
         "Per-shape success, 26-letter 96 px vs 192 px camera.", S, "camera slide / backup", None),
        ("shapes_10_vs_26", fig_shapes_10_vs_26,
         "Per-shape success, 10-shape vs 26-letter policy: earlier ten unchanged, newer 17 jump.", S, "scaling slide",
         None),
        ("shapes_validation_vs_test", fig_shapes_validation_vs_test,
         "Per-shape validation (round 44) vs final test success: how noisy 50-scene validation is.", f"{V}, {S}",
         "methods / backup", None),
        ("shapes_teacher_vs_student", fig_shapes_teacher_vs_student,
         "Teacher (exact state) vs camera student per shape, all 27 trained shapes, on the same 50 validation scenes "
         "(student: round 44).", f"teacher_validation.csv, {V}", "teacher-gap slide", None),
        ("shapes_new_letters_dotplot", fig_shapes_new_letters_dotplot,
         "17 newer letters: original, 10-shape model (never saw them), 26-letter model (trained on them).", S,
         "transfer vs training slide", None),
        ("shapes_table_heatmap", fig_shapes_table_heatmap,
         "Table-like heatmap of success for 33 shapes x 4 policies with numbers.", S, "paper / backup", None),
        ("heldout_scaling", fig_heldout_scaling,
         "Never-seen shapes: success (1.2% -> 2.8% -> 4.7%; 192 px 5.9%, all labelled) and final coverage vs number of "
         "training shapes (1 -> 10 -> 27).", S,
         "generalisation slide (main)", "More training shapes help on never-seen shapes"),
        ("heldout_per_shape_bars", fig_heldout_per_shape_bars,
         "Each never-seen shape for the four policies.", S, "generalisation slide", None),
        ("heldout_scaling_per_shape", fig_heldout_scaling_per_shape,
         "Small multiples: each never-seen shape vs number of training shapes.", S, "backup / paper", None),
        ("heldout_unseen_capitals", fig_heldout_unseen_capitals,
         "10-shape students (96 px round 13, 192 px round 6) on the 17 capitals they never trained on, per letter and "
         "pooled (D052).", S,
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
         "192 px minus 96 px camera, paired by scene: 10-shape picks (round 6 vs 13) and 27-shape picks (round 2 vs "
         "44); each pair also differs in training rounds.", E, "camera slide", None),
        ("analysis_192_vs_96_shapes", lambda: fig_analysis_192_vs_96(per_shape=True),
         "192 px minus 96 px for every shape.", E, "paper appendix", None),
        ("analysis_label_ab", fig_analysis_label_ab,
         "Typed-in matched A/B tests on validation (1000 episodes each): plan labels 7.7% -> 12.9%, camera, demo count, "
         "network size; success rates with 95% intervals, no relative gains.", "typed-in (LEARNING_FINDINGS "
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
         "Forward-process RL: T success before (55%) vs after for each step size, with the number of RL updates at the "
         "last evaluation (typed-in).", "typed-in (EXPERIMENT_LOG "
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
         "Original vs the first exact-state planner teacher vs the teacher reading the state with 1 px / 1° noise, on T, "
         "near-T, Y, D, L.", "typed_in_numbers.json (LEARNING_FINDINGS sections 5-6)", "method slide (teacher)", None),
        ("method_teacher_success_26letters", fig_method_teacher_26,
         "Teacher vs the 27-shape camera student (96 px, round 44) for all 27 trained shapes on the same 50 validation "
         "scenes, plus the pooled rate; hatched teacher bars were checked with an older world model.",
         f"teacher_validation.csv, {V}", "teacher-gap slide", None),
        ("method_teacher_success_groups", _teacher_26_summary,
         "Teacher vs 26-letter camera student (round 44) by shape group on the same 50 validation scenes per shape: 10 "
         "earlier shapes, 17 newer letters, all 27. The teacher's 10-earlier-shapes number is the latest check per "
         "shape (teacher_validation.csv: 400 of 500, T, D and L re-checked with world model v3), not the 390 of 500 "
         "of the run that labelled the 10-shape students (world model v2, method_teacher_gap).",
         f"teacher_validation.csv, {V}", "teacher-gap half slide", None),
        ("method_teacher_gap", fig_method_teacher_gap,
         "Teacher vs picked camera student on the same validation scenes: 5 shapes (76% vs 40%), 10 shapes, 27 shapes.",
         f"{V}; typed_in_numbers.json; teacher_validation.csv", "half-slide", None),
        ("method_pipeline", fig_method_pipeline,
         "Method block diagram: rollouts -> structured world model -> sampling planner teacher (scores 1,024 plans per "
         "control: 4 x 256 samples, 128 of the first 256 are structured pushes) -> demonstrations -> camera policy <-> "
         "correction rounds; training-only (privileged) boxes gold, deployed policy blue. Also SVG.",
         "typed_in_numbers.json (method_pipeline)", "method slide", None),
        ("method_pipeline_simple", fig_method_pipeline_simple,
         "Four-box version of the method (rollouts, world model, planner teacher, camera policy) with the correction loop. "
         "Also SVG.", "-", "title / overview slide", None),
        ("policy_interface", fig_policy_interface,
         "What the policy sees and outputs: two 96 px camera frames + pusher position -> network (100 denoising steps) -> "
         "16 future targets, first 8 executed (red), rest pink; colours match the videos. Schematic scene. Also SVG.",
         "typed_in_numbers.json (method_pipeline)", "setup / method slide", None),
        ("wm_structure", fig_wm_structure,
         "World model: what is copied from the simulator (pusher motion, contact gate) and what is learned (block motion in "
         "contact). Also SVG.", "typed_in_numbers.json (method_pipeline)", "method slide / backup", None),
        ("wm_accuracy", fig_wm_accuracy,
         "World-model error: generic network vs structured model (median, 8 and 32 controls) and the structured model's "
         "typical vs rare errors (90th percentile; 1 in 20 windows >= 14 px near walls).", "world_model_accuracy.csv",
         "world-model slide / backup", None),
        ("teacher_state_noise", fig_teacher_state_noise,
         "Teacher reading the state with noise: exact vs 1 px vs 2 px on T, near-T, L (50 scenes; gold shades = noise "
         "level) and the estimator accuracy sweep on Y, D, L (20 scenes; letters by grey shade, marker and direct "
         "label).", "teacher_planner.csv", "teacher slide / backup", None),
        ("method_planner_fixes", fig_method_planner_fixes,
         "Planner teacher development: first vs final planner per letter (20 scenes) and T coverage as each stall cause "
         "was fixed (12 scenes); stall causes in planner_fixes.tex.", "teacher_planner.csv", "backup", None),
        ("oracle_bon", fig_oracle_bon,
         "Simulator best-of-N (upper bound): Y 3 -> 8 of 20, and students imitating it within the original's noise.",
         "early_attempts.csv", "what we tried first / backup", None),
        ("distill_early", fig_distill_early,
         "Early planner distillation: first worse everywhere; training the encoder, longer training and DAgger gave the "
         "first D and L successes with lower T.", "early_attempts.csv", "what we tried first / backup", None),
        ("bptt_teacher", fig_bptt_teacher,
         "Gradient-based (BPTT) teacher: real-simulator coverage 0.04-0.12 vs the sampling planner's 0.93-0.97.",
         f"early_attempts.csv; original from {S} (D040)", "what we tried first / backup", None),
        ("copying_gap", fig_copying_gap,
         "D044: a student reading the exact state keeps only 30-41% of the teacher's successes; the camera student and "
         "the original on the same scenes.", "early_attempts.csv", "why corrections / backup", None),
        ("lastmile_and_replanning", fig_lastmile_replanning,
         "Last-mile variant (-3.8, -4.8, +0.0; no gain), re-planning every 4 controls (-13.6) and executing 12 or 15 "
         "steps (-16.1, -22.5) against the matched runs.", f"early_attempts.csv, {E}", "ablation / backup", None),
        ("project_timeline", fig_project_timeline,
         "Project timeline Oct 24 -> Nov 7: each approach as a bar over its days, coloured by outcome, with the result.",
         "timeline.csv", "overview / what we tried slide", None),
        ("data_growth", fig_data_growth,
         "Training examples per round by source (human T, teacher demonstrations, corrections), 5 shapes, 10 shapes and "
         "26 letters (96 px lines); the human T band (24k examples, too thin to see) is labelled directly.", "training_rounds.csv", "data slide / backup", None),
        ("train_loss_rounds", fig_train_loss_rounds,
         "End-of-round training loss vs validation success per round (10 shapes, 26 letters): the loss keeps falling "
         "while validation plateaus.", f"training_rounds.csv, {V}", "plateau slide / backup", None),
        ("train_loss_within_round", fig_train_loss_within_round,
         "Training loss within five rounds of the 26-letter 96 px line (rounds 1, 13: 20k steps; 14, 30, 50: 54k).",
         "training_loss_curves.csv", "backup", None),
        ("failure_categories", fig_failure_categories,
         "Episodes split into solved / near miss (>= 90%) / partial (50-90%) / far (< 50%) by best and by final coverage. "
         "Top: original vs ours, 96 px (round 44) on the same final-test scenes (D051). Bottom, a separate labelled "
         "block: round 44 vs round 50 on their own different pool (D054).", E, "near-miss / limits slide", None),
    ]
    byname = {j[0]: j for j in jobs}
    if variant in ("all", "none"):
        for name, fn, shows, data, use, title in jobs:
            if not wanted(name):
                continue
            fig = fn()
            save(fig, name, shows, data, use, title, svg=name in SVG_FIGURES)
            print("wrote", name)
    if variant in ("all", "slide"):
        for mode, names in (("slide", SLIDE_FIGURES), ("half", HALF_FIGURES)):
            MODE["name"] = mode
            style_variant(mode)
            for name in names:
                if wanted(name):
                    _, fn, shows, *_ = byname[name]
                    save(fn(), name, shows, None, None)
                    print(f"wrote {VARIANT_DIR[mode]}/{name}")
        MODE["name"] = "default"
        style()
    if not ONLY and variant == "all":
        write_tables()
        write_readme()
        print("wrote tables and README")


SVG_FIGURES = {"method_pipeline", "method_pipeline_simple", "policy_interface", "wm_structure"}
# slide versions: full slide width (6.3 x 3.5 in, >= 16 pt) and half slide (4.2 x 3.2 in, >= 14 pt)
SLIDE_FIGURES = ["method_pipeline_simple", "method_pipeline", "policy_interface", "wm_structure", "wm_accuracy",
                 "rl_t_collapse", "rl_signal_variance", "method_teacher_success", "method_teacher_success_26letters",
                 "teacher_state_noise", "curves_all_stages", "curves_26letters", "data_growth", "train_loss_rounds",
                 "final_grouped_bars", "final_stage_progression", "final_per_shape_bars_vertical", "final_forest_groups",
                 "heldout_scaling", "coverage_near_misses", "failure_categories", "analysis_label_ab",
                 "analysis_steps_executed", "analysis_t_tradeoff", "final_blends_all", "project_timeline",
                 "final_dumbbell", "oracle_bon", "analysis_round50_vs_44", "analysis_192_vs_96", "final_10shapes_bars",
                 "heldout_unseen_capitals"]
HALF_FIGURES = ["analysis_label_ab", "analysis_budget", "analysis_steps_executed", "heldout_scaling",
                "coverage_near_misses", "rl_t_collapse", "rl_signal_variance", "method_teacher_success",
                "method_teacher_success_groups", "method_teacher_gap", "final_headline_bars", "lastmile_and_replanning",
                "oracle_bon", "final_blends_all", "analysis_round50_vs_44", "analysis_192_vs_96", "final_10shapes_bars",
                "heldout_unseen_capitals"]
# README text for slide versions that show something different from the default version
VARIANT_SHOWS = {
    ("slide", "curves_all_stages"):
        "Whole training story on one axis with direct labels: stage names above the plot, 'executed' and 'plan labels' at "
        "the label switch (10 shapes), 'larger budget' (27 shapes), 96 px / 192 px tags; a legend for the circles (sent "
        "to a fresh test) and the star (round 50, fresh test 34.8%). Dashed: original on the same validation scenes; "
        "dotted 'original: fresh test' for 27 shapes (no validation run of the original there).",
    ("slide", "final_grouped_bars"):
        "HEADLINE final test (D051) by shape group; legend Original, Ours 10 shapes, Ours 96 px (round 44), Ours 192 px "
        "(round 2); one-decimal labels on the original and 96 px bars (3.7, 33.4, 0.4, ...).",
    ("half", "final_headline_bars"):
        "Compact headline (27 trained, 6 never-seen); one-decimal labels on the original and 96 px (round 44) bars.",
    ("slide", "final_dumbbell"):
        "All 33 shapes along x (sorted; never-seen shapes right of the line), original vs ours 96 px (round 44); T and "
        "near-T labels bold. Taller canvas so it fits height=0.8\\textheight.",
    ("slide", "failure_categories"):
        "Original vs ours, 96 px (round 44) on the same final-test scenes (D051) only; round 50 (other pool, D054) is in "
        "the default version's separate block.",
    ("slide", "oracle_bon"):
        "Upper bound (simulator best-of-16 per letter, left) and the imitating students in one colour (right), one legend.",
    ("half", "oracle_bon"):
        "Upper-bound panel only: original vs simulator best-of-16 per letter (Y 3 -> 8 of 20).",
    ("slide", "final_blends_all"):
        "Blend tests D040-D043 on T, Y and D, passed/failed; the dashed T line is explained in the legend (pass rule: T "
        "interval above -5).",
    ("half", "final_blends_all"):
        "Blend tests D040-D043 on T, Y and D, passed/failed; legend: pass rule (T interval above -5).",
    ("slide", "analysis_round50_vs_44"):
        "Round 50 minus round 44 per shape group on their own fresh pool (D054, 200 scenes per shape), 95% intervals; the "
        "two rates in a column (All 27: 32.9 -> 34.8%).",
    ("half", "analysis_round50_vs_44"):
        "Round 50 minus round 44 per shape group (D054 pool), 95% intervals, rates in a column.",
    ("slide", "analysis_192_vs_96"):
        "192 px minus 96 px, paired by scene; group headings name the compared picks (10 shapes: 192 px round 6 vs 96 px "
        "round 13; 27 shapes: round 2 vs round 44, also differs in training rounds).",
    ("half", "analysis_192_vs_96"):
        "192 px minus 96 px: all 10 (round 6 vs 13), all 27 and 6 unseen (round 2 vs 44; also differs in training "
        "rounds).",
    ("slide", "final_10shapes_bars"):
        "10-shape final test per shape (1000 fresh scenes each) and pooled: original, ours 96 px (round 13), 192 px "
        "(round 6).",
    ("half", "final_10shapes_bars"):
        "10-shape final test per shape and pooled (whole-number pooled labels).",
    ("slide", "heldout_unseen_capitals"):
        "10-shape students (96 px round 13, 192 px round 6) on the 17 capitals they never trained on, per letter and "
        "pooled (0.2 / 6.6 / 10.6%).",
    ("half", "heldout_unseen_capitals"):
        "10-shape students on the 17 capitals they never trained on, per letter (two rows of labels) and pooled "
        "(whole numbers; 0.2 kept).",
    ("slide", "method_teacher_success_26letters"):
        "WIDE per-letter version: teacher vs camera student (96 px, round 44) for all 27 trained shapes on the same 50 "
        "validation scenes; pooled rates in the legend (88% vs 35% overall); no hatching (world-model version per "
        "shape: default version and teacher_validation.csv).",
    ("half", "method_teacher_success_groups"):
        "Teacher vs camera student (round 44) by shape group: 10 earlier, 17 newer, all 27 (see the note on 400 vs 390 "
        "of 500 below).",
    ("slide", "data_growth"):
        "10 shapes and 26 letters; bands labelled directly (corrections, teacher demos, human T: 24k) instead of a legend.",
    ("slide", "wm_accuracy"):
        "Median error 8 controls ahead (generic network vs ours = structured model) and the structured model's typical "
        "vs 1-in-10 errors per letter plus the rare large errors; the marks are explained inside the panel.",
    ("half", "analysis_label_ab"):
        "Matched A/B tests (validation, 1000 episodes each) as success rates: plan labels 7.7% -> 12.9%, camera, demos, "
        "network size; no relative gain label.",
    ("half", "rl_t_collapse"):
        "T success after RL per setting, AdvantageFlow and NFT, against 55% before; right column = RL updates at the last "
        "evaluation.",
    ("slide", "rl_t_collapse"):
        "T success after RL per setting, AdvantageFlow and NFT, against 55% before; right column = RL updates at the last "
        "evaluation.",
}


if __name__ == "__main__":
    main()
