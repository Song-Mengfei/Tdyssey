#!/usr/bin/env bash
# Build the editable PowerPoint deck from slides/main.tex. From the repository root:
#
#   bash presentation/pptx/build.sh
#
# Output: presentation/pptx/out/tdyssey-slides.pptx (videos embedded). Steps: compile the default Beamer PDF and the
# step-by-step PDF, describe every page (pdf_to_deck.py), write the deck (build_deck.js), check its structure and that
# every slide's words occur on its Beamer page (qa.py inspect; the build fails otherwise).
# Environment (all optional):
#   TEX=tectonic|pdflatex   TeX engine (default: tectonic when on PATH or in $TECTONIC, else pdflatex via latexmk)
#   PDF=..., BUILDS_PDF=... use these PDFs instead of compiling (CI compiles them with the LaTeX action)
#   VIDEOS=embed|drive      embed the MP4 files (default) or link posters to the public Drive copies (Google Slides)
#   OUT=...                 output .pptx path
#   PYTHON=...              Python with requirements.txt installed (default: uv run with the pinned packages, else python3)
#   CARLITO=...             Carlito-Regular.ttf for the glyph check (default: the Debian/Ubuntu fonts-crosextra-carlito path)
# Needs ffmpeg on the PATH (the videos are re-muxed with their index first).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRES="$(dirname "$HERE")"
BUILD="$HERE/out/build"
mkdir -p "$BUILD"

# ---- 1. Beamer PDFs
if [[ -z "${PDF:-}" || -z "${BUILDS_PDF:-}" ]]; then
  TECTONIC="${TECTONIC:-$(command -v tectonic || true)}"
  ENGINE="${TEX:-$([[ -n "$TECTONIC" ]] && echo tectonic || echo pdflatex)}"
  echo "== compiling the Beamer deck ($ENGINE)"
  if [[ "$ENGINE" == tectonic ]]; then
    (cd "$PRES/slides" && "$TECTONIC" -X compile main.tex --outdir "$BUILD" >"$BUILD/tectonic-main.log" 2>&1)
    (cd "$HERE" && "$TECTONIC" -X compile tdyssey-builds.tex --outdir "$BUILD" >"$BUILD/tectonic-builds.log" 2>&1)
  else
    (cd "$PRES/slides" && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$BUILD" main.tex >/dev/null)
    (cd "$HERE" && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$BUILD" tdyssey-builds.tex >/dev/null)
  fi
  PDF="$BUILD/main.pdf"
  BUILDS_PDF="$BUILD/tdyssey-builds.pdf"
fi

# ---- 2. Python and Node dependencies
if [[ -n "${PYTHON:-}" ]]; then
  PY=("$PYTHON")
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run --no-project --python 3.12 --with-requirements "$HERE/requirements.txt" python)
else
  PY=(python3)
fi
if [[ ! -d "$HERE/node_modules/pptxgenjs" ]]; then
  (cd "$HERE" && npm ci --no-audit --no-fund)
fi

# ---- 3. describe the pages, write the deck, check it
echo "== extracting pages from $PDF"
"${PY[@]}" "$HERE/pdf_to_deck.py" --pdf "$PDF" --builds-pdf "$BUILDS_PDF" --out "$HERE/out"
echo "== writing the deck"
node "$HERE/build_deck.js" --deck "$HERE/out/deck.json" --out "${OUT:-$HERE/out/tdyssey-slides.pptx}" --videos "${VIDEOS:-embed}"
echo "== structure and text check"
CARLITO="${CARLITO:-/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf}"
"${PY[@]}" "$HERE/qa.py" inspect "${OUT:-$HERE/out/tdyssey-slides.pptx}" --pdf "$PDF" \
  $([[ -f "$CARLITO" ]] && echo --font "$CARLITO")
