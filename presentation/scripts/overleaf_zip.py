"""Build a small zip for Overleaf (its upload limit is about 50 MB): the deck, the paper, every figure as PDF, the
tables, the TikZ diagrams (figures/tikz/), and only the images and video stills that the slide, paper
or TikZ sources mention, including the image-model section. The17 clips from the
approved PPTX are included for a portable local copy; PDF posters open their
public Drive links by default. Use --without-videos for a smaller online-only zip.

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
    ap.add_argument("--without-videos", action="store_true",
                    help="omit the17 current MP4 files; retain online PDF links")
    a = ap.parse_args()
    tikz = sorted((ROOT / "figures" / "tikz").glob("*.tex"))
    sources = sorted((ROOT / "slides").rglob("*.tex")) + sorted((ROOT / "paper").rglob("*.tex")) + tikz
    texts = "".join(f.read_text(encoding="utf-8") for f in sources if f.exists())
    names = set(re.findall(r"[\w./-]+\.(?:png|jpg|jpeg|pdf)", texts))
    names |= {n + ".png" for n in re.findall(r"\\(?:video|mdl)\{([^}]+?)(?:\.png)?\}", texts)}
    models = re.findall(r"\\mdl\{([^}]+)\}", texts)
    files = [ROOT / "figures" / "README.md"]
    files += sorted((ROOT / "figures").rglob("*.pdf"))
    files += sorted((ROOT / "data" / "tables").glob("*.tex")) + tikz
    files += sorted((ROOT / "data").glob("image_wm*_sources.json"))
    files += [ROOT / "data" / "pptx_slide_media.json", ROOT / "data" / "slide_video_bindings.tex"]
    files += sorted(p for d in ("slides", "paper") for p in (ROOT / d).rglob("*") if p.is_file())
    base = {Path(n).name for n in names}
    files += sorted((ROOT / "images" / "slide").glob("*.png"))                # the deck's crops (some names built by macros)
    for p in sorted((ROOT / "images").rglob("*.png")) + sorted((ROOT / "videos" / "posters").glob("*.png")):
        curated_image_model_asset = "image_wm_48m" in p.relative_to(ROOT).parts
        if curated_image_model_asset or p.name in base or any(p.name == f"{m}_9shapes_end.png" for m in models):
            files.append(p)
    files += sorted((ROOT / "videos" / "posters" / "slides").glob("*"))
    files += sorted((ROOT / "videos" / "slides").glob("*.md"))
    if not a.without_videos:
        files += sorted((ROOT / "videos" / "slides").glob("*.mp4"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(files):
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{a.out}: {len(set(files))} files, {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
