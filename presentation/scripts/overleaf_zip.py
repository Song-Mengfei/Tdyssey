"""Build a small zip for Overleaf (its upload limit is about 50 MB): the deck, the paper, every figure as PDF, the
tables, and only the images and video stills that slides/main.tex or paper/main.tex mention. Videos are left out
(Overleaf cannot play them; link or embed them in PowerPoint instead).

    python3 presentation/scripts/overleaf_zip.py --out /path/to/presentation-overleaf.zip

Upload the zip as a new Overleaf project and set slides/main.tex (or paper/main.tex) as the main file.
"""
import argparse
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    texts = "".join((ROOT / f).read_text() for f in ("slides/main.tex", "paper/main.tex") if (ROOT / f).exists())
    names = set(re.findall(r"[\w./-]+\.(?:png|jpg|pdf)", texts))
    names |= {n + ".png" for n in re.findall(r"\\(?:video|mdl)\{([^}]+?)(?:\.png)?\}", texts)}
    models = re.findall(r"\\mdl\{([^}]+)\}", texts)
    files = [ROOT / "figures" / "README.md"]
    files += sorted((ROOT / "figures").rglob("*.pdf"))
    files += sorted((ROOT / "data" / "tables").glob("*.tex"))
    files += sorted(p for d in ("slides", "paper") for p in (ROOT / d).rglob("*") if p.is_file())
    for p in sorted((ROOT / "images").rglob("*.png")) + sorted((ROOT / "videos" / "posters").glob("*.png")):
        rel = p.relative_to(ROOT).as_posix()
        if p.name in names or any(rel.endswith(n) for n in names) or any(p.name == f"{m}_9shapes_end.png" for m in models):
            files.append(p)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(files):
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{a.out}: {len(set(files))} files, {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
