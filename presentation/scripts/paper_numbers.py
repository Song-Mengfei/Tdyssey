"""Numbers and results tables for the course paper (no GPU, standard library only).

    python3 presentation/scripts/paper_numbers.py [--data <folder>] [--history <folder>] [--paper <folder>]

Reads the round-50 test tables that build_round50_data.py writes (d064_test_by_shape.csv, d064_test_groups.csv,
d064_test_paired.csv: every policy on ONE pool of new scenes, seeds 15000000 + 20*i, 200 per shape) from --data
(default presentation/data; point it at a scratch folder to check a test that is still running), and the earlier
tests, which are final, from --history (default presentation/data: final_tests_by_shape.csv for the history table,
final_tests_episodes.csv for the steps-executed test D056).

Writes, into --paper (default presentation/paper):
  numbers.tex          one \\newcommand per number the paper's text uses, each with a comment naming its source
  tables/main_test.tex     success by shape group for the five policies, change of the main model vs the original
  tables/flowdit.tex       Flow-DiT against the main model (paired)
  tables/per_shape.tex     every shape, every policy (appendix)
  tables/steps.tex         steps executed per plan, D056 (26 letters, narrow T left out)
  tables/history.tex       the earlier fixed tests, averaged over letters only

Groups follow the talk: all 26 letters, 9 earlier letters (T Y D L E F J P U), 17 new letters, 9 never-seen shapes
(digits 2, 3, 4, 5, 7, 8, plus, hash, arrow) and T; the narrow T (an extra training shape) is in no group. Rates are
pooled successes / episodes; paired changes and their 95% intervals come from d064_test_paired.csv (normal
approximation on per-scene differences, as in build_round50_data.py). If the test is not finished (fewer than 200
paired scenes for some shape or policy), numbers.tex defines \\paperdatanote as a red line under the abstract.

Flow-DiT numbers come from the same d064 files (policy key "base", paired against round 50), so the
architecture_test_*.csv files of build_experiment_data.py are not read here (they feed the deck's arch_groups graph).

Typical use after the test:
  python3 presentation/scripts/build_round50_data.py && python3 presentation/scripts/build_experiment_data.py && \\
  python3 presentation/scripts/paper_numbers.py
"""
import argparse
import csv
import math
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # presentation/
MODELS = [("orig", "Orig", "Original"), ("ten", "Ten", "10 shapes"), ("r50", "Main", "Ours, 96 px"),
          ("h192", "HiRes", "Ours, 192 px"), ("base", "Flow", "Flow-DiT")]
MODEL_DESC = {"orig": "original policy", "ten": "ten-shape model (round 13)", "r50": "main model (round 50, 96 px)",
              "h192": "192 px camera model (round 2)", "base": "Flow-DiT 16x16 (160k steps)"}
EARLY9 = ["T", "Y", "D", "L", "E", "F", "J", "P", "U"]
NEW17 = list("ABCGHIKMNOQRSVWXZ")
LETTERS26 = EARLY9 + NEW17
UNSEEN9 = ["digit2", "digit3", "digit4", "digit5", "digit7", "digit8", "plus", "hash", "arrow"]
UNSEEN6 = ["digit3", "digit4", "digit7", "plus", "hash", "arrow"]        # the never-seen shapes of the earlier tests
NARROW_T = "T_narrow"
GROUPS = [("all 26 letters", "Head", "All 26 letters", LETTERS26), ("9 earlier letters", "Early", "9 earlier letters", EARLY9),
          ("17 new letters", "New", "17 new letters", NEW17), ("9 never-seen shapes", "Unseen", "9 never-seen shapes", UNSEEN9),
          ("T", "T", "T", ["T"])]
FULL = 200                                                       # scenes per shape in the finished test


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dec1(x):
    """A number as text with one decimal, rounded half-even on its exact decimal value (as the deck figures do)."""
    return Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN)


def pct(k, n):
    return f"{dec1(Decimal(k) * 100 / Decimal(n))}" if n else "--"


def pct_frac(s):
    return f"{dec1(Decimal(str(s)) * 100)}"


def signed(text):
    """'+12.4', '$-$3.1' or '0.0' from a one-decimal string."""
    v = Decimal(text)
    if v > 0:
        return "+" + text
    if v < 0:
        return "$-$" + text.lstrip("-")
    return "0.0"


def shape_name(s):
    return {"T_narrow": "narrow T", "plus": "+", "hash": "\\#", "arrow": "arrow"}.get(s, s.replace("digit", ""))


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name, value, comment):
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}%  {comment}")

    def text(self):
        return "\n".join(self.lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"), help="folder with d064_test_*.csv (default presentation/data)")
    ap.add_argument("--history", default=str(ROOT / "data"),
                    help="folder with final_tests_by_shape.csv and final_tests_episodes.csv (default presentation/data)")
    ap.add_argument("--paper", default=str(ROOT / "paper"), help="paper folder (numbers.tex and tables/ go here)")
    a = ap.parse_args()
    data, hist, paper = Path(a.data), Path(a.history), Path(a.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    need = [data / f"d064_test_{x}.csv" for x in ("by_shape", "groups", "paired")]
    if not all(f.exists() for f in need):
        raise SystemExit(f"missing {[f.name for f in need if not f.exists()]} in {data}: run "
                         "scripts/build_round50_data.py first (or pass --data <folder with its CSV files>)")
    G = {(r["model_key"], r["group"]): r for r in read(data / "d064_test_groups.csv")}
    S = {(r["model_key"], r["shape"]): r for r in read(data / "d064_test_by_shape.csv")}
    P = {(r["base_model_key"], r["model_key"], r["kind"], r["unit"]): r for r in read(data / "d064_test_paired.csv")}
    src = f"{data.name}/d064_test"

    # ---------------------------------------------------------------- completeness
    shapes_all = LETTERS26 + [NARROW_T] + UNSEEN9
    counts = [int(S[(k, s)]["episodes"]) if (k, s) in S else 0 for k, _, _ in MODELS for s in shapes_all]
    n_min, n_max = min(counts), max(counts)
    complete = n_min == FULL and n_max == FULL
    M = Macros()
    M.add("numTestScenes", str(n_min), f"paired new scenes per shape ({src}_by_shape.csv; finished test: {FULL})")
    M.add("numTestLetterEpisodes", f"{26 * n_min:,}".replace(",", "{,}"), "paired episodes per policy over the 26 letters")
    M.add("numTestStatus", "complete" if complete else "partial", f"{n_min}-{n_max} scenes per shape and policy")
    note = "" if complete else (f"\\textcolor{{red}}{{Partial data, {n_min} of {FULL} scenes per shape. Rerun "
                                "build\\_round50\\_data.py, build\\_experiment\\_data.py and paper\\_numbers.py after the "
                                "test.}")
    M.add("paperdatanote", note, "empty once every shape has 200 paired scenes for every policy")
    print(f"D064 test: {n_min}-{n_max} paired scenes per shape and policy -> {'COMPLETE' if complete else 'PARTIAL'}")

    # ---------------------------------------------------------------- group rates and coverage
    rate = {}
    for gkey, gname, _, shapes in GROUPS:
        for k, kname, _ in MODELS:
            r = G.get((k, gkey))
            if r is None:
                M.add(f"num{gname}{kname}", "--", f"{MODEL_DESC[k]}, {gkey}: missing in {src}_groups.csv")
                continue
            v = pct(int(r["successes"]), int(r["episodes"]))
            rate[(k, gkey)] = Decimal(v)
            M.add(f"num{gname}{kname}", v, f"{MODEL_DESC[k]}, {gkey}, success % ({r['successes']}/{r['episodes']}; "
                                         f"{src}_groups.csv)")
    for gkey, gname in (("all 26 letters", "Cov"), ("9 never-seen shapes", "CovUnseen")):
        for k, kname, _ in MODELS:
            r = G.get((k, gkey))
            M.add(f"num{gname}{kname}", f"{float(r['mean_final_coverage']):.2f}" if r else "--",
                  f"{MODEL_DESC[k]}, {gkey}, mean final coverage ({src}_groups.csv)")
    for k, kname, _ in MODELS:                                     # mean peak coverage over the 26 letters
        rows = [S[(k, s)] for s in LETTERS26 if (k, s) in S]
        n = sum(int(r["episodes"]) for r in rows)
        v = sum(float(r["mean_peak_coverage"]) * int(r["episodes"]) for r in rows) / n if n else float("nan")
        M.add(f"numPeak{kname}", f"{v:.2f}", f"{MODEL_DESC[k]}, 26 letters, mean peak coverage, episode-weighted "
                                          f"({src}_by_shape.csv)")

    # ---------------------------------------------------------------- paired changes
    def paired(base, k, unit, kind="group"):
        r = P.get((base, k, kind, unit))
        if r is None:
            return None
        d, lo, hi = (pct_frac(r[c]) for c in ("difference", "ci95_low", "ci95_high"))
        sig = "excludes" if Decimal(lo) > 0 or Decimal(hi) < 0 else "includes"
        return signed(d), signed(lo), signed(hi), sig, r["paired_episodes"]

    for base, prefix, models in (("orig", "numDiff", ["ten", "r50", "h192", "base"]),
                                 ("r50", "numVsMain", ["ten", "h192", "base"])):
        for gkey, gname, _, _ in GROUPS:
            for k in models:
                kname = dict((m[0], m[1]) for m in MODELS)[k]
                p = paired(base, k, gkey)
                name = f"{prefix}{gname}{kname}"
                what = f"{MODEL_DESC[k]} minus {MODEL_DESC[base]}, {gkey}"
                if p is None:
                    for suf in ("", "Lo", "Hi", "Sig"):
                        M.add(name + suf, "--", what + ": missing")
                    continue
                M.add(name, p[0], f"{what}, paired change in points over {p[4]} scenes ({src}_paired.csv)")
                M.add(name + "Lo", p[1], "  95% interval, low end")
                M.add(name + "Hi", p[2], "  95% interval, high end")
                M.add(name + "Sig", p[3], "  'excludes' or 'includes' (does the interval include zero?)")
    # main model minus the ten-shape model (the negated ten-minus-round-50 rows, interval ends swapped)
    for gkey, gname, _, _ in GROUPS:
        r = P.get(("r50", "ten", "group", gkey))
        name = f"numMainOverTen{gname}"
        if r is None:
            for suf in ("", "Lo", "Hi", "Sig"):
                M.add(name + suf, "--", f"main model minus ten-shape model, {gkey}: missing")
            continue
        d, lo, hi = (pct_frac(-Decimal(r[c])) for c in ("difference", "ci95_high", "ci95_low"))
        M.add(name, signed(d), f"main model minus ten-shape model, {gkey}, paired change in points over "
                               f"{r['paired_episodes']} scenes (negated r50/ten rows of {src}_paired.csv)")
        M.add(name + "Lo", signed(lo), "  95% interval, low end")
        M.add(name + "Hi", signed(hi), "  95% interval, high end")
        M.add(name + "Sig", "excludes" if Decimal(lo) > 0 or Decimal(hi) < 0 else "includes", "  interval vs zero")

    # ---------------------------------------------------------------- per-shape facts
    def srate(k, s):
        r = S.get((k, s))
        return Decimal(pct(int(r["successes"]), int(r["episodes"]))) if r else None

    up_main = sum(1 for s in LETTERS26 if srate("r50", s) is not None and srate("r50", s) > srate("orig", s))
    up_flow = sum(1 for s in LETTERS26 if srate("base", s) is not None and srate("base", s) > srate("r50", s))
    M.add("numLettersUpMain", str(up_main), f"letters (of 26) where the main model beats the original ({src}_by_shape.csv)")
    M.add("numLettersUpFlow", str(up_flow), f"letters (of 26) where Flow-DiT beats the main model ({src}_by_shape.csv)")
    main_rates = sorted((srate("r50", s), s) for s in LETTERS26 if srate("r50", s) is not None)
    if main_rates:
        M.add("numMainLetterMin", str(main_rates[0][0]), f"lowest letter rate of the main model ({main_rates[0][1]})")
        M.add("numMainLetterMinName", main_rates[0][1], "  its letter")
        M.add("numMainLetterMax", str(main_rates[-1][0]), f"highest letter rate of the main model ({main_rates[-1][1]})")
        M.add("numMainLetterMaxName", main_rates[-1][1], "  its letter")
    for s, sname in (("digit7", "DigitSeven"), (NARROW_T, "NarrowT")):
        for k, kname, _ in MODELS:
            v = srate(k, s)
            M.add(f"num{sname}{kname}", str(v) if v is not None else "--", f"{MODEL_DESC[k]}, {s}, success % "
                                                                          f"({src}_by_shape.csv)")
    t_o, t_m = rate.get(("orig", "T")), rate.get(("r50", "T"))
    M.add("numTDirMain", "--" if t_o is None or t_m is None else ("below" if t_m < t_o else "above" if t_m > t_o
                                                                    else "equal to"),
          "main model on T compared with the original ('below', 'above' or 'equal to')")
    u = [rate.get((k, "9 never-seen shapes")) for k in ("orig", "ten", "r50")]
    trend = ("rises with the number of training shapes" if None not in u and u[0] < u[1] < u[2] else
             "is higher after training on all 26 letters" if None not in u and u[2] > u[0] else
             "does not rise with the number of training shapes")
    M.add("numUnseenTrend", trend, "never-seen shapes: original -> ten-shape model -> main model")

    # ---------------------------------------------------------------- earlier tests (final data)
    FB = read(hist / "final_tests_by_shape.csv")

    def hrate(test, model, shapes):
        rows = [r for r in FB if r["test"] == test and r["model"] == model and r["shape"] in shapes]
        k, n = sum(int(r["successes"]) for r in rows), sum(int(r["episodes"]) for r in rows)
        per = sorted({int(r["episodes"]) for r in rows})
        return pct(k, n), (per[0] if per else 0), len(rows)

    HIST = [  # (test, label, shapes label, shapes, seed base, reference model, reference label, model)
        ("D046", "5 shapes, round 4", "T Y D L", ["T", "Y", "D", "L"], "7{,}000{,}000", "original", "", "5-shape round 4"),
        ("D047", "5 shapes, round 7", "T Y D L", ["T", "Y", "D", "L"], "8{,}000{,}000", "original", "", "5-shape round 7"),
        ("D049/D050", "10 shapes, 96 px (round 13)", "9 earlier letters", EARLY9, "9{,}000{,}000", "original", "",
         "10-shape 96 px"),
        ("D049/D050", "10 shapes, 192 px (round 6)", "9 earlier letters", EARLY9, "9{,}000{,}000", "original", "",
         "10-shape 192 px"),
        ("D052", "10 shapes, 96 px, untrained letters", "17 new letters", NEW17, "11{,}000{,}000", "original", "",
         "10-shape 96 px"),
        ("D052", "10 shapes, 192 px, untrained letters", "17 new letters", NEW17, "11{,}000{,}000", "original", "",
         "10-shape 192 px"),
        ("D051", "26 letters, 96 px (round 44)", "26 letters", LETTERS26, "10{,}000{,}000", "original", "",
         "26-letter 96 px (round 44)"),
        ("D051", "26 letters, 192 px (round 2)", "26 letters", LETTERS26, "10{,}000{,}000", "original", "",
         "26-letter 192 px (round 2)"),
        ("D051", "26 letters, 96 px (round 44)", "6 never-seen shapes", UNSEEN6, "10{,}000{,}000", "original", "",
         "26-letter 96 px (round 44)"),
        ("D054", "26 letters, round 50", "26 letters", LETTERS26, "12{,}000{,}000", "26-letter 96 px round 44",
         " (round 44)", "26-letter 96 px round 50"),
    ]
    hist_rows = []
    for test, label, slabel, shapes, seed, ref, reflab, model in HIST:
        rr, n_ref, c_ref = hrate(test, ref, shapes)
        rm, n_m, c_m = hrate(test, model, shapes)
        if c_ref != len(shapes) or c_m != len(shapes):
            print(f"  history {test} {model}: {c_m} of {len(shapes)} shapes found")
        hist_rows.append(f"{label} & {slabel} & {n_m:,} & {seed} & {rr}{reflab} & {rm} \\\\".replace("1,000", "1{,}000"))
        if test == "D051" and slabel == "26 letters" and "96 px" in label:
            M.add("numHistFinalOrig", rr, "earlier final test D051 (seeds 10000000 + 20*i, 300 scenes), original, 26 "
                                          "letters (final_tests_by_shape.csv)")
            M.add("numHistFinalMain", rm, "  D051, round 44, 26 letters")
        if test == "D054":
            M.add("numHistRoundFortyFour", rr, "D054 (seeds 12000000 + 20*i), round 44, 26 letters "
                                               "(final_tests_by_shape.csv)")
            M.add("numHistRoundFifty", rm, "  D054, round 50, 26 letters")
    with open(paper / "tables" / "history.tex", "w") as f:
        f.write("% Generated by presentation/scripts/paper_numbers.py from final_tests_by_shape.csv (earlier tests, "
                "letters only;\n% the narrow T is left out). Success %, pooled over the shapes named.\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Model & Shapes & Scenes per shape & Seed base & Reference & Ours \\\\\n\\midrule\n")
        f.write("\n".join(hist_rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------- steps executed (D056, final data)
    EP = [r for r in read(hist / "final_tests_episodes.csv") if r["test"] == "D056"]
    by = {}
    for r in EP:
        by.setdefault(r["model"], {})[(r["shape"], r["scene_seed"])] = int(r["success"])
    m8, m12, m15 = "execute 8 (trained)", "execute 12", "execute 15"
    step_groups = [("9 earlier letters", EARLY9), ("17 new letters", NEW17), ("All 26 letters", LETTERS26),
                   ("6 never-seen shapes", UNSEEN6), ("T", ["T"])]
    lines = []
    for gl, shapes in step_groups:
        keys = sorted(k for k in by[m8] if k[0] in shapes and k in by[m12] and k in by[m15])
        n = len(keys)
        cells = [pct(sum(by[m8][k] for k in keys), n)]
        for m, word in ((m12, "Twelve"), (m15, "Fifteen")):
            d = [by[m][k] - by[m8][k] for k in keys]
            mu = sum(d) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
            h = 1.96 * sd / math.sqrt(n)
            v = pct(sum(by[m][k] for k in keys), n)
            ch = [signed(str(dec1(100 * x))) for x in (mu, mu - h, mu + h)]
            cells += [v, f"{ch[0]} [{ch[1]}, {ch[2]}]"]
            if gl == "All 26 letters":
                M.add(f"numSteps{word}", v, f"D056 (seeds 12000000 + 20*i, 200 scenes), round 50 executing "
                                            f"{word.lower()} of 16 steps, 26 letters (final_tests_episodes.csv)")
                M.add(f"numSteps{word}Diff", ch[0], "  paired change vs executing 8, points")
                M.add(f"numSteps{word}DiffLo", ch[1], "  95% interval, low end")
                M.add(f"numSteps{word}DiffHi", ch[2], "  95% interval, high end")
        if gl == "All 26 letters":
            M.add("numStepsEight", cells[0], "D056, round 50 executing 8 of 16 steps (trained), 26 letters")
        lines.append(f"{gl} & " + " & ".join(cells) + " \\\\")
    with open(paper / "tables" / "steps.tex", "w") as f:
        f.write("% Generated by presentation/scripts/paper_numbers.py from final_tests_episodes.csv (test D056, round 50,"
                "\n% seeds 12000000 + 20*i, 200 scenes per shape, paired by scene). Success %; change in points [95% "
                "interval].\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Shapes & Execute 8 & Execute 12 & Change & Execute 15 & Change \\\\\n\\midrule\n")
        f.write("\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------- tables from the D064 test
    def cell(k, gkey):
        r = G.get((k, gkey))
        return pct(int(r["successes"]), int(r["episodes"])) if r else "--"

    def change(base, k, gkey):
        p = paired(base, k, gkey)
        return f"{p[0]} [{p[1]}, {p[2]}]" if p else "--"

    head = (f"% Generated by presentation/scripts/paper_numbers.py from {src}_*.csv (seeds 15000000 + 20*i, "
            f"{n_min}-{n_max} paired scenes per shape;\n% {'complete' if complete else 'PARTIAL, rerun after the test'}). "
            "Success %; change in points [95% interval], paired by scene.\n")
    with open(paper / "tables" / "main_test.tex", "w") as f:
        f.write(head)
        f.write("\\begin{tabular}{lrrrlrr}\n\\toprule\n")
        f.write("Shapes & Original & 10 shapes & Ours, 96 px & Change vs original & Ours, 192 px & Flow-DiT \\\\\n"
                "\\midrule\n")
        for gkey, _, glabel, _ in GROUPS:
            bold = gkey == "all 26 letters"
            c = [cell(k, gkey) for k, _, _ in MODELS]
            label = glabel
            if bold:
                c[2] = "\\textbf{" + c[2] + "}"
                label = "\\textbf{" + glabel + "}"
            f.write(f"{label} & {c[0]} & {c[1]} & {c[2]} & {change('orig', 'r50', gkey)} & {c[3]} & {c[4]} \\\\\n")
        cov = [G.get((k, "all 26 letters")) for k, _, _ in MODELS]
        f.write("\\midrule\nFinal coverage, 26 letters & " + " & ".join(
            (f"{float(r['mean_final_coverage']):.2f}" if r else "--") for r in cov[:3]) + " & & " + " & ".join(
            (f"{float(r['mean_final_coverage']):.2f}" if r else "--") for r in cov[3:]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(paper / "tables" / "flowdit.tex", "w") as f:
        f.write(head)
        f.write("\\begin{tabular}{@{}lrrl@{}}\n\\toprule\n")
        f.write("Shapes & Main & Flow-DiT & Change \\\\\n\\midrule\n")
        for gkey, _, glabel, _ in GROUPS:
            f.write(f"{glabel} & {cell('r50', gkey)} & {cell('base', gkey)} & {change('r50', 'base', gkey)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    # per shape: two blocks side by side
    left = [("9 earlier letters", EARLY9), ("17 new letters", NEW17[:9])]
    right = [("17 new letters (continued)", NEW17[9:]), ("Extra training shape", [NARROW_T]),
             ("9 never-seen shapes", UNSEEN9)]

    def block(parts):
        out = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Shape & Orig. & 10 sh. & 96 px & 192 px & Flow-DiT \\\\"]
        for title, shapes in parts:
            out += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\"]
            for s in shapes:
                vals = [str(srate(k, s)) if srate(k, s) is not None else "--" for k, _, _ in MODELS]
                out.append(f"{shape_name(s)} & " + " & ".join(vals) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    with open(paper / "tables" / "per_shape.tex", "w") as f:
        f.write(head + block(left) + "\n\\hspace{6mm}\n" + block(right) + "\n")

    try:                                                       # no machine-specific paths in the written file
        shown = str(data.resolve().relative_to(ROOT.parent))
    except ValueError:
        shown = f"<scratch>/{data.name}"
    with open(paper / "numbers.tex", "w") as f:
        f.write("% Generated by presentation/scripts/paper_numbers.py; do not edit by hand. Rerun it after the test:\n"
                "%   python3 presentation/scripts/build_round50_data.py && python3 presentation/scripts/"
                "build_experiment_data.py && python3 presentation/scripts/paper_numbers.py\n"
                f"% Source of the round-50 test numbers: {shown}/d064_test_*.csv ({n_min}-{n_max} paired scenes per "
                f"shape and policy; {'complete' if complete else 'PARTIAL'}).\n"
                "% Rates in %, one decimal; changes in points; coverage as a share of the goal.\n")
        f.write(M.text())
    print(f"wrote {paper / 'numbers.tex'} ({len(M.lines)} macros) and 5 tables in {paper / 'tables'}")


if __name__ == "__main__":
    main()
