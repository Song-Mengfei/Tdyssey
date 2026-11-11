# Videos extracted from the current 62-slide deck

These 17MP4 files and their original posters in `../posters/slides/` were extracted
from the reviewed offline PowerPoint file without recompression. The source PPTX
SHA256 is `871d7d5735548bbe70e21ce19c90976a1110f7a82cd66e7c8b0b1096803c2e67`.
The MP4s total 6,744,124 bytes; the 17posters total 1,381,890 bytes.

[`INDEX.md`](INDEX.md) lists every current slide, its stable page ID and playback
duration. [`../../data/pptx_slide_media.json`](../../data/pptx_slide_media.json)
records the exact PPTX parts, SHA256 hashes, pixel sizes, original bounding boxes
on a 160x90 mm page, public Drive fallback URLs and scientific qualifications.
All 17Drive videos were verified as anyone-with-link readers with search discovery
disabled on 2026-11-11. The presentation and media folder were not made public.

Re-extract or verify from a copy of that exact offline PPTX, from the repository
root:

```bash
python -B presentation/scripts/extract_pptx_slide_media.py --pptx /path/to/Tdyssey-Offline-With-Videos.pptx
python -B presentation/scripts/extract_pptx_slide_media.py --pptx /path/to/Tdyssey-Offline-With-Videos.pptx --verify-only
```

The script uses the Python standard library, checks the input PPTX hash and every
extracted byte stream, and regenerates `data/slide_video_bindings.tex`. It refuses
to overwrite an existing media file whose contents differ. No model, simulator,
video decoder or video encoder is called.

The stable page IDs follow the current offline deck. Its current slide 41 is page
`p38`; the earlier local LaTeX order placed that page at15. These IDs keep the
media bindings valid when the talk order changes.

## One click per composite

Pages `p2`, `p19`, `p20`, `p21`, `p26`, `p47`, `p54` and `p55` each contain one
previously reviewed synchronized MP4. Every panel starts at zero. A shorter panel
holds its final frame while the longer panel finishes; it does not loop or change
speed. Six canvases retain RGB-v2 bytes; `p54` and `p55` retain the wider-height
wide-safe-v3 canvases, including the unchanged text and spacing around the panels.
The 30 fps output preserves the original sample timing. In particular, `p47` keeps
the timing of its 15 fps source clips by repeating frames, rather than speeding them
up. The original presentation recordings are usually shown at 3x simulator real
time; the world-model recordings run at half that playback speed. The composite
step did not change those source playback rates.

The three early image-model videos are saved-state displays, not continuous
model rollouts:

- `p27`: four saved historical 8.73M predictions, H1 through H4, held 2 seconds each.
  The left simulator image is a fixed H4 endpoint. Missing GT H1-H3 was not invented.
- `p32`: the matched intermediate-parent O027 comparison at saved H4/H8/H16
  snapshots, held 2 seconds each. It does not compare the completed 49,991-update
  original baseline.
- `p37`: two L cases, each visiting the three saved H4/H8/H16 snapshots. Completed
  original and mixed continuation have different budgets and branch points.
  L021 is a moving prefix, not a stopping interval.

`p62` is the earlier historical 8.73M C diagnostic at its original 10 fps, not a 48M
result. The remaining six single videos keep their original archived bytes.

By default, clicking a video poster in the PDF opens its public Drive URL.
Defining `\tdlocalvideos` when compiling switches that poster link to a relative
MP4 launch in an external viewer. Each build selects one link per poster; the PDF
does not offer a second visible fallback link or switch automatically if playback
fails. Each composite remains one file, preserving one-click synchronization.
Local launches depend on the PDF viewer, and require this `videos/` tree in the
source/PDF bundle. The offline PPTX remains the single-file deliverable with its
videos embedded.

The wider canvases are a narrow exception to the earlier 768 px guidance in D057:
these are the exact approved presentation videos, preserved without resizing.
Five MP4s also have byte-identical copies elsewhere under `presentation/`; the
dedicated17-file snapshot keeps this deck's media mapping complete and stable.
