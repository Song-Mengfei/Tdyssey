#!/usr/bin/env python3
"""Crop video stills and image grids to their scene content for the slides: presentation/images/slide/*.png.

The stills and grids made by make_videos.py carry small text bars (OpenCV captions) above every tile. On a slide that
text is far smaller than the slide's own text, so the deck shows only the scenes and sets every label in LaTeX at slide
size. This script cuts the scenes out of the existing files (no GPU, simulator or checkpoints; needs Pillow and numpy):

    python presentation/scripts/slide_images.py

Tile geometry follows make_videos.py: frames are 384 px (TILE); side-by-side clips have a 40 px title bar and a 32 px
caption per tile; 256 px grid tiles have a 32 px caption, smaller tiles 26 px; grids have a 44 px header (56 px with a
subtitle). The grey table walls (7 px at 384 px) are trimmed, so the slides can draw one consistent thin frame.
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "videos" / "posters"
IMG = ROOT / "images"
OUT = IMG / "slide"
TILE = 384


def load(path):
    return Image.open(path).convert("RGB")


def save(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, optimize=True)
    return name


def scene(img, x, y, size, wall=True):
    """One square scene starting at (x, y); wall=True trims the table's grey wall (7 px per 384 px)."""
    t = round(7 * size / TILE) if wall else 0
    return img.crop((x + t, y + t, x + size - t, y + size - t))


def no_inset(img):
    """Blank the camera inset in the top-left corner (30% of the tile plus its frame), e.g. for very small tiles."""
    a = np.asarray(img).copy()
    k = round(img.size[0] * 0.32)
    a[:k, :k] = 255
    return Image.fromarray(a)


def grid(tiles, cols, gap=6):
    """Tiles (equal size) in a grid with white gaps."""
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), "white")
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out.paste(t, (c * (w + gap), r * (h + gap)))
    return out


def icon(img, out_px=160, pad=0.18, skip_inset=True):
    """Square close-up of a scene's content (block, goal, pusher, plan dots) for small icons in Figure 1: the bounding
    box of non-white pixels (the camera inset in the top-left corner ignored), padded and made square."""
    a = np.asarray(img).copy()
    if skip_inset:                                   # blank the camera inset so it never shows in the close-up
        k = round(img.size[0] * 0.34)
        a[:k, :k] = 255
        img = Image.fromarray(a)
    ink = (255 * 3 - a.astype(int).sum(axis=2)) > 60
    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = [round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)]
    out = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "white")
    out.paste(img.crop(box), (0, 0))
    return out.resize((out_px, out_px), Image.LANCZOS)


def pair(stem, left, right):
    """Side-by-side clip still (compare_*, teacher_vs_student_*): two 384 px scenes below a 40 px title, 32 px captions."""
    img = load(POST / f"{stem}.png")
    save(scene(img, 0, 72, TILE), f"{left}.png")
    save(scene(img, TILE, 72, TILE), f"{right}.png")


def main():
    written = []
    # original vs ours (last frames: the outcome) and two mid-episode frames with the plan dots
    for s in ("P", "A", "M", "S_scene2", "digit7_scene2"):
        pair(f"compare_{s}", f"compare_{s}_orig", f"compare_{s}_ours")
        written += [f"compare_{s}_orig.png", f"compare_{s}_ours.png"]
    pair("compare_P_mid", "compare_P_mid_orig", "compare_P_mid_ours")
    # architecture experiment: main model (left) vs the 505M transformer (right), same scene, last frames
    for s_ in ("T", "W", "Z"):
        if (POST / f"frontier_{s_}.png").exists():
            pair(f"frontier_{s_}", f"frontier_{s_}_main", f"frontier_{s_}_transformer")
            written += [f"frontier_{s_}_main.png", f"frontier_{s_}_transformer.png"]
    pair("teacher_vs_student_K", "tvs_K_teacher", "tvs_K_policy")
    written += ["compare_P_mid_orig.png", "compare_P_mid_ours.png", "tvs_K_teacher.png", "tvs_K_policy.png"]

    # one scene through the project: 2 x 3 grid of 256 px tiles under a 44 px title, 32 px captions
    img = load(POST / "progression_E.png")
    for i, key in enumerate(("original", "five_r7", "ten_96", "l26_r0", "l26_r13", "l26_r50")):
        r, c = divmod(i, 3)
        written.append(save(scene(img, c * 256, 44 + r * 288 + 32, 256), f"prog_E_{key}.png"))

    # near miss (single 384 px tile with a 32 px caption) and the camera-pixel close-ups (two 576 px panels, 40 px bars)
    written.append(save(scene(load(POST / "near_miss_M_400020.png"), 0, 32, TILE), "near_miss_M.png"))
    img = load(IMG / "last_mile" / "near_miss_vs_solved_zoom.png")
    written.append(save(img.crop((0, 40, 576, 616)), "zoom_near_miss_M.png"))
    written.append(save(img.crop((576, 40, 1152, 616)), "zoom_solved_nearT.png"))

    # correction-round records (3 x 2 tiles of 384 px under 40 px bars) and intended vs executed labels (4 tiles)
    img = load(IMG / "training_data" / "correction_round_examples.png")
    for i, s in enumerate("KREWAD"):
        r, c = divmod(i, 3)
        written.append(save(scene(img, c * TILE, r * 424 + 40, TILE), f"correction_{s}.png"))
    img = load(IMG / "training_data" / "labels_intended_vs_executed.png")
    for i, s in enumerate("TYER"):
        written.append(save(scene(img, i * TILE, 40, TILE), f"labels_{s}.png"))

    # planner teacher, mid-episode (3 x 3 grid of 256 px tiles under a 56 px two-line header)
    img = load(POST / "teacher_9shapes_mid.png")
    for i, s in enumerate(("T", "nearT", "Y", "D", "L", "E", "K", "R", "W")):
        r, c = divmod(i, 3)
        written.append(save(scene(img, c * 256, 56 + r * 288 + 32, 256), f"teacher_{s}.png"))

    # all 33 shapes, round 50 (11 x 3 grid of 160 px tiles, 26 px captions, 44 px header) without the captions
    img = load(POST / "l26_r50_33shapes_wide.png")
    tiles = [no_inset(scene(img, c * 160, 44 + r * 186 + 26, 160)) for r in range(3) for c in range(11)]
    written.append(save(grid(tiles, 11, gap=5), "grid_r50_33shapes.png"))

    # every model on the same 9 shapes (3 x 3 grid of 256 px tiles, 32 px captions, 44 px header), last frames
    for key in ("original", "forward_rl", "five_r7", "blend", "ten_96", "l26_r50"):
        img = load(IMG / "models" / f"{key}_9shapes_end.png")
        tiles = [no_inset(scene(img, c * 256, 44 + r * 288 + 32, 256)) for r in range(3) for c in range(3)]
        written.append(save(grid(tiles, 3, gap=6), f"models_{key}.png"))

    # world model: imagined (red outline) vs simulator, posters (384 px under a 40 px bar) and one still for Figure 1
    for s in ("D", "E"):
        written.append(save(scene(load(POST / f"imagined_vs_real_{s}.png"), 0, 40, TILE), f"wm_imagined_{s}.png"))
    img = load(IMG / "world_model" / "imagined_vs_real_D.png")
    written.append(save(scene(img, TILE, 40, TILE), "wm_D_8ahead.png"))

    # Figure 1 thumbnails: noisy rollout of the original (4 x 4 grid of 192 px tiles, 26 px captions, 56 px header),
    # the 96 px camera input (third panel, 34 px bar) and a planned path (ours, mid-episode, red and pink dots)
    img = load(POST / "dynamics_data_action_noise.png")
    written.append(save(scene(img, 1 * 192, 56 + 1 * 218 + 26, 192), "ov_rollout.png"))
    img = load(IMG / "camera_views_512_192_96.png")
    written.append(save(scene(img, 2 * TILE, 34, TILE), "ov_camera96.png"))
    # close-ups for the icons in Figure 1 (figures/tikz/overview.tex)
    written.append(save(icon(load(OUT / "ov_rollout.png")), "ov_icon_data.png"))
    written.append(save(icon(load(OUT / "wm_D_8ahead.png"), skip_inset=False), "ov_icon_wm.png"))
    written.append(save(icon(load(OUT / "teacher_nearT.png"), skip_inset=False), "ov_icon_teacher.png"))
    cam = scene(load(IMG / "camera_views_512_192_96.png"), 2 * TILE + 14, 34 + 14, TILE - 28, wall=False)
    written.append(save(icon(cam, pad=0.08, skip_inset=False), "ov_icon_camera.png"))

    # shape gallery without labels (11 x 3 tiles of 180 px with 28 px captions); never-seen shapes drawn lighter
    img = load(IMG / "shape_gallery_33_upright.png")
    names = ["T", "nearT", "Y", "D", "L", "E", "F", "J", "P", "U", "A", "B", "C", "G", "H", "I", "K", "M", "N", "O",
             "Q", "R", "S", "V", "W", "X", "Z", "digit3", "digit4", "digit7", "plus", "hash", "arrow"]
    crops = []
    for i, nm in enumerate(names):
        r, c = divmod(i, 11)
        t = img.crop((c * 180 + 22, r * 208 + 28 + 22, c * 180 + 158, r * 208 + 28 + 158))   # 136 px around the glyph
        if i >= 27:
            a = np.asarray(t).astype(float)
            t = Image.fromarray((a + (255 - a) * 0.55).round().astype(np.uint8))
        crops.append(t)
        if nm in ("T", "A", "K", "R", "S", "W", "digit7"):
            written.append(save(t, f"shape_{nm}.png"))
    written.append(save(grid(crops, 11, gap=8), "shapes_33.png"))
    written.append(save(grid(crops[:27], 27, gap=10), "shapes_27_row.png"))
    print(f"wrote {len(written)} images to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
