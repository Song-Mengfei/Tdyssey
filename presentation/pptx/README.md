# Editable PowerPoint version of the talk deck

`build.sh` turns `slides/main.tex` into `out/tdyssey-slides.pptx`: every page of the Beamer PDF in order (main
slides, the closing slide, the image-model summary and all backup pages), at the Beamer page size, with editable
text, separate pictures, embedded videos and the Beamer overlay builds as click animations.

```bash
bash presentation/pptx/build.sh                      # from the repository root
VIDEOS=drive bash presentation/pptx/build.sh         # posters link to the public Drive videos (for Google Slides)
```

Needs Node 20+ (`npm ci` installs the pinned `pptxgenjs` 4.0.1 from `package-lock.json`), Python 3.12 with
`requirements.txt` (the script uses `uv run` when `uv` is on the PATH), `ffmpeg`, and Tectonic or a TeX Live with
`latexmk`.
A full build takes about 35 seconds. GitHub builds the deck on every push that changes `presentation/` (workflow
"Presentation PDFs", artifact `presentation-pptx`, with LibreOffice comparison sheets). Generated files stay in
`out/`, which Git ignores.

## How it works

1. TeX compiles the default PDF (`slides/main.tex`, one final build per page) and the step-by-step PDF
   (`tdyssey-builds.tex`, every overlay on its own page).
2. `pdf_to_deck.py` reads both PDFs with PyMuPDF and writes `out/deck.json` and `out/assets/`.
3. `build_deck.js` writes the deck with pptxgenjs, then edits the slide XML for what pptxgenjs cannot do (theme
   colours, picture crops, exact dash patterns, the slide number suffix, click animations).
4. `qa.py inspect --pdf` lists fonts, sizes, pictures, videos and animations, and fails the build on duplicate
   shape ids, animations that point to missing shapes, a video without its media node or click trigger, any object
   above a video or linked poster that overlaps it, ligature or other broken characters, and any slide word that
   does not occur on its Beamer page (the PDF's own text with ligatures expanded, words counted per page).

| Beamer element | PowerPoint object |
| --- | --- |
| Section kicker, frame title | placeholders of the content layout; size, weight, colour and position come from the layout only |
| Footer text, progress-bar track, slide number | on the content layout; the blue progress bar is one shape per slide; main slides show the slide number field and "/ 42", backup slides keep Beamer's own "1 / 15" numbering as text |
| Text | text boxes at the PDF baseline positions with the PDF sizes and colours; one paragraph per PDF line, so the line breaks stay as in Beamer |
| Stills, galleries, posters | pictures with the original pixels, each its own object |
| Video posters (`\slidevideo`) and scenes that open a local clip (`\video`) | embedded MP4 (re-muxed with `ffmpeg -c copy -movflags +faststart`, streams unchanged) with the poster as cover; the scene frame and play mark that Beamer draws over a clip are baked into the cover, so nothing lies above a video; a click on the video plays or pauses it (media node and click trigger in the slide timing, as PowerPoint writes them); with `VIDEOS=drive` the poster links to the public Drive copy (`data/pptx_slide_media.json`) or to the local clip |
| Graphs (`figures/deck/*.pdf`) | pictures rendered from the PDF page at 400 dpi |
| TikZ diagrams (regions in `regions.json`) | pictures rendered from the PDF page at 400 dpi, one transparent layer per build step |
| Frames, panels, rules, bullets, play marks, arrows outside diagrams | native shapes (rectangle, rounded rectangle, oval, line, free-form path) |
| `\onslide`, `\uncover`, `\only`, `\ovstep` and the other step macros | "Appear" entrance effects, one click per overlay step after the first; all objects of a step appear together |

Build steps are found by comparing each frame's overlay pages with its final page, so a new overlay in
`main.tex` or in a TikZ file needs no change here. A new TikZ diagram that should stay one picture needs an entry
in `regions.json` (page number and rectangle in mm); without one its boxes, arrows and labels become separate
native shapes and text boxes.

## Fonts and sizes

The whole deck uses one font family, Calibri, set once in the theme (heading and body). Calibri ships with
Office and is available in Google Slides. On this deck it is about 6% narrower than Latin Modern Sans (10% in
bold), so with the PDF's own line breaks no text runs past its Beamer extent; Arial would be about 3% wider and
push lines into neighbouring pictures. Calibri has every character of the slide text (arrows, multiplication
sign, quotes, dashes) except the check and cross marks from `\ding`; those runs name Segoe UI Symbol, the font
Office itself uses for them on Windows, and other viewers substitute a font that has them (LibreOffice: DejaVu
Sans). Ligatures in the PDF (fi, ff, fl) become plain letters. Graph and diagram pictures keep the Latin Modern text of the PDF.

Sizes are the Beamer sizes in points: frame title 16, section kicker 8, footer 7, body 11 and 10 (`\small`),
caption 9, `\large` and takeaway 12, numbered limitations 12.5, stat number 15, title-slide title and hero number
30, subtitle 17, the question 21, the two backup tables 7.4 and 6.8, the playable-video captions 7.5 and 8. The
teammates' frames keep the explicit sizes of their source (6.5 to 11.5 pt).

## Checks without PowerPoint

```bash
python presentation/pptx/qa.py inspect presentation/pptx/out/tdyssey-slides.pptx --pdf presentation/pptx/out/build/main.pdf
python presentation/pptx/qa.py compare presentation/slides/main.pdf tdyssey-slides.pdf    # LibreOffice render
python presentation/pptx/qa.py preview presentation/pptx/out/tdyssey-slides.pptx presentation/pptx/out/build/main.pdf \
  --builds-pdf presentation/pptx/out/build/tdyssey-builds.pdf --fonts /usr/share/fonts/truetype/crosextra
```

`compare` and `preview` write sheets with the Beamer page, the PowerPoint page and both overlaid (red: only in
Beamer, cyan: only in PowerPoint). `preview` draws the slides from the .pptx XML with Carlito (Calibri metrics) and
also shows every click state next to the matching overlay page. It is an approximation for machines without an
office renderer, not a PowerPoint render. LibreOffice and python-pptx do not play animations; the animation
timing XML is checked by `qa.py inspect` and by the click-state sheets.
