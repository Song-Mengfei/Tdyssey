#!/usr/bin/env python3
"""Describe every page of the Beamer PDF as editable slide objects for build_deck.js.

Inputs (all built from slides/main.tex, see build.sh):
  * the default PDF (one final build per page), the reference the PowerPoint file must match;
  * the step-by-step PDF (\\def\\tdanimated{1}), whose overlay pages give the click order of every build.

Output: <out>/deck.json and <out>/assets/ (pictures, diagram and graph renders, build layers).

What becomes what:
  * text            -> one text box per line group, exact baseline positions, sizes and colours, line breaks as in
                       the PDF (the font is replaced by the theme font, see README.md);
  * raster images   -> separate pictures with the original pixels (stills, galleries, posters);
  * video posters   -> embedded MP4 files with the poster as cover (data/slide_video_bindings.tex);
  * graphs          -> pictures rendered from the PDF at high resolution (every included PDF figure);
  * TikZ diagrams   -> pictures rendered from the PDF (regions in regions.json), one transparent layer per build
                       step so that the click animations reproduce the Beamer overlays;
  * other drawing   -> native shapes (rectangles, rounded rectangles, circles, lines, free-form paths).
Build steps are found by comparing the overlay pages of each frame with its final state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

MM = 72 / 25.4  # PDF points per millimetre
TEXPT = 72.27 / 72  # TeX points per PDF point

HERE = Path(__file__).resolve().parent
PRES = HERE.parent  # presentation/

# Deck colours (slides/main.tex). Exact matches are written as theme colours where the theme has a slot.
PALETTE = {
    "1F2328": "tdink", "5F6670": "tdgrey", "9AA0A6": "tdmuted", "C9CDD3": "tdrule", "B3BAC3": "tdframe",
    "E6E9ED": "tdtrack", "F3F5F8": "tdpanel", "2A78D6": "tdblue", "EDA100": "tdgold", "8A5A00": "tdgolddark",
    "8F8D86": "tdorig", "D23C3C": "tdred", "1F8A4C": "tdgood", "B3261E": "tdbad", "FFF6E0": "tdgoldbg",
    "EAF2FC": "tdbluebg", "9CC3F0": "tdblueline", "9E2A8A": "tdtransf", "FFFFFF": "white", "000000": "black",
}
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


# ----------------------------------------------------------------------------------------------- small helpers
def hexcolor(rgb) -> str:
    if rgb is None:
        return None
    if isinstance(rgb, int):
        rgb = ((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255)
        vals = rgb
    else:
        if len(rgb) == 1:
            rgb = (rgb[0],) * 3
        if len(rgb) == 4:  # CMYK
            c, m, y, k = rgb
            rgb = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
        vals = tuple(int(round(max(0.0, min(1.0, v)) * 255)) for v in rgb)
    best, bestd = None, 99
    for h in PALETTE:
        pv = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        dist = max(abs(a - b) for a, b in zip(vals, pv))
        if dist < bestd:
            best, bestd = h, dist
    if bestd <= 2:
        return best
    return "%02X%02X%02X" % vals


def r_mm(rect) -> list:
    return [round(rect[0] / MM, 3), round(rect[1] / MM, 3), round(rect[2] / MM, 3), round(rect[3] / MM, 3)]


def contains(outer, inner, tol=0.4) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol
            and inner[3] <= outer[3] + tol)


def center_in(outer, inner) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def tex_size(bp: float) -> float:
    """Font size in TeX points (9.96 bp -> 10): half points when within 0.06 pt (10.95 -> 11), else 0.1 pt."""
    pt = bp * TEXPT
    half = round(pt * 2) / 2
    return half if abs(pt - half) <= 0.06 else round(pt * 10) / 10


def span_chars(span) -> list:
    """Characters of a PyMuPDF text-trace span in content-stream order, as dicts with text, box and baseline (mm).

    A ligature glyph comes either as one code point (U+FB00-FB06, expanded here) or, when the font's ToUnicode
    map says "fi", as several characters: the first carries the glyph's box, the rest have no glyph (-1) and a
    zero-width box at the glyph's left edge. Those followers are appended to the first character, so the letters
    stay in order and no gap (which would become a space) opens inside the word."""
    font = span["font"]
    dingbat = "Dingbat" in font or "Zapf" in font
    out = []
    for c in span["chars"]:
        u, glyph = c[0], c[1]
        ch = chr(u)
        if dingbat and u < 0x60:  # TeX's ZapfDingbats codes without a Unicode map: \ding{51} -> U+2713
            ch = chr(0x2700 + u)
        ch = LIGATURES.get(ch, ch)
        x0, y0, x1, y1 = (v / MM for v in c[3])
        base = c[2][1] / MM
        if out and (glyph == -1 or x1 - x0 < 1e-4) and out[-1]["x0"] - 0.05 <= x0 <= out[-1]["x1"] + 0.05 \
                and abs(base - out[-1]["base"]) < 0.05:
            out[-1]["ch"] += ch
            continue
        out.append({"ch": ch, "x0": x0, "x1": x1, "base": base})
    return out


def page_labels(doc) -> list:
    out = []
    for p in doc:
        lab = p.get_label() or ""
        m = re.fullmatch(r"<FEFF([0-9A-Fa-f]*)>", lab)
        if m:  # xdvipdfmx writes UTF-16 hex labels
            h = m.group(1)
            lab = "".join(chr(int(h[i:i + 4], 16)) for i in range(0, len(h), 4))
        out.append(lab)
    return out


def frame_groups(doc) -> list:
    """Consecutive pages with the same label are the overlays of one frame."""
    labs = page_labels(doc)
    groups, prev = [], None
    for i, lab in enumerate(labs):
        if lab != prev or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
        prev = lab
    return groups


# ---------------------------------------------------------------------------- content stream: placed PDF figures
_TOKEN = re.compile(rb"\s*(?:(\()|(<<|>>)|(<)|(\[|\])|(/[^\s/\[\]()<>{}%]*)|([-+]?(?:\d+\.?\d*|\.\d+))|"
                    rb"([A-Za-z'\"*][A-Za-z0-9'\"*]*)|(%[^\r\n]*)|(\S))")


def _skip_string(data: bytes, i: int) -> int:
    depth, n = 1, len(data)
    while i < n and depth:
        c = data[i]
        if c == 0x5C:  # backslash
            i += 2
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
        i += 1
    return i


def placed_forms(page) -> list:
    """Bounding boxes (PDF points) of the Form XObjects drawn directly on the page (the included PDF graphs)."""
    doc = page.parent
    forms = {}
    for xref, name, invoker, bbox in page.get_xobjects():
        if invoker == 0:
            mat = doc.xref_get_key(xref, "Matrix")
            m = [1, 0, 0, 1, 0, 0]
            if mat[0] == "array":
                m = [float(v) for v in mat[1].strip("[]").split()]
            forms[name] = (fitz.Rect(bbox), fitz.Matrix(*m))
    if not forms:
        return []
    data = page.read_contents()
    ctm, stack, nums, out, i = fitz.Matrix(1, 0, 0, 1, 0, 0), [], [], [], 0
    last_name = None
    while i < len(data):
        m = _TOKEN.match(data, i)
        if not m:
            break
        i = m.end()
        if m.group(1):
            i = _skip_string(data, i)
            nums = []
        elif m.group(3):
            j = data.find(b">", i)
            i = j + 1
            nums = []
        elif m.group(5):
            last_name = m.group(5)[1:].decode("latin-1")
            nums = []
        elif m.group(6):
            nums.append(float(m.group(6)))
        elif m.group(7):
            op = m.group(7)
            if op == b"q":
                stack.append(fitz.Matrix(ctm))
            elif op == b"Q":
                ctm = stack.pop() if stack else ctm
            elif op == b"cm" and len(nums) >= 6:
                ctm = fitz.Matrix(*nums[-6:]) * ctm
            elif op == b"Do" and last_name in forms:
                bbox, fm = forms[last_name]
                r = bbox * fm * ctm
                out.append((last_name, fitz.Rect(r)))
            elif op == b"BI":  # inline image: skip to EI
                j = data.find(b"EI", i)
                i = j + 2
            nums = []
        else:
            nums = []
    # PDF user space has y up; PyMuPDF pages use y down. Convert with the page's transformation.
    res = []
    tm = page.transformation_matrix
    for name, r in out:
        res.append((name, fitz.Rect(r * tm).normalize()))
    return res


# ------------------------------------------------------------------------------------------------- rendering

def norm_title(s: str) -> str:
    """Frame title as a lookup key: ligatures expanded, case and spaces ignored."""
    for lig, rep in (("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb03", "ffi"), ("\ufb04", "ffl"),
                     ("\ufb05", "st"), ("\ufb06", "st")):
        s = s.replace(lig, rep)
    return re.sub(r"\s+", "", s).replace("\u2019", "'").lower()


def frame_title(page, layout: dict) -> str:
    """The frame title text on a page (spans at the layout's title baseline and size), or '' if there is none."""
    ref = layout.get("title") if layout else None
    if not ref:
        return ""
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES
    parts = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if abs(span["size"] - ref["size"]) < 0.3 and abs(span["origin"][1] / MM - ref["baseline"]) < 0.6:
                    parts.append(span["text"])
    return " ".join(parts)

class Renderer:
    def __init__(self, doc):
        self.doc = doc
        self.cache = {}

    def page(self, idx: int, dpi: int) -> np.ndarray:
        key = (idx, dpi)
        if key not in self.cache:
            pix = self.doc[idx].get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            self.cache[key] = arr.copy()
        return self.cache[key]

    def crop(self, idx: int, rect_mm, dpi: int) -> np.ndarray:
        clip = fitz.Rect(rect_mm[0] * MM, rect_mm[1] * MM, rect_mm[2] * MM, rect_mm[3] * MM)
        pix = self.doc[idx].get_pixmap(dpi=dpi, alpha=False, clip=clip)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def px_box(rect_mm, dpi, shape):
    s = dpi / 25.4
    x0 = max(0, int(math.floor(rect_mm[0] * s)))
    y0 = max(0, int(math.floor(rect_mm[1] * s)))
    x1 = min(shape[1], int(math.ceil(rect_mm[2] * s)))
    y1 = min(shape[0], int(math.ceil(rect_mm[3] * s)))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


# ---------------------------------------------------------------------------------------------- the extractor
class Extractor:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.assets = self.out / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.doc = fitz.open(args.pdf)
        self.bdoc = fitz.open(args.builds_pdf)
        self.rend = Renderer(self.doc)
        self.brend = Renderer(self.bdoc)
        self.regions = json.loads(Path(args.regions).read_text())
        self.regions["titles"] = {norm_title(k): v for k, v in self.regions.get("titles", {}).items()}
        self.videos = self._read_video_bindings()
        self.clips = self._read_poster_clips()
        self.media = json.loads((PRES / "data" / "pptx_slide_media.json").read_text())
        self.drive = {v["page_id"]: v.get("drive_url") for v in self.media["videos"]}
        self.saved = {}
        self.warnings = []

    # ---- sources
    def _read_video_bindings(self) -> list:
        txt = (PRES / "data" / "slide_video_bindings.tex").read_text()
        out = []
        for m in re.finditer(r"tdclip@(p\d+)\\endcsname\{\\tdplacevideo\{([\d.]+)\}\{([\d.]+)\}\{([\d.]+)\}\{([\d.]+)\}"
                             r"\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\}", txt):
            pid, x, y, w, h, poster, mp4, url = m.groups()
            out.append({"id": pid, "x": float(x), "y": float(y), "w": float(w), "h": float(h), "poster": poster,
                        "mp4": mp4, "url": url})
        return out

    def _read_poster_clips(self) -> dict:
        txt = (PRES / "slides" / "video_poster_masks.tex").read_text()
        out = {}
        for m in re.finditer(r"tdposterclip@(p\d+)\\endcsname\{%\s*\\clip \(\[xshift=([\d.]+)mm,yshift=-([\d.]+)mm\]"
                             r"current page.north west\)\s*rectangle \(\[xshift=([\d.]+)mm,yshift=-([\d.]+)mm\]", txt):
            pid, x0, y0, x1, y1 = m.groups()
            out[pid] = [float(x0), float(y0), float(x1), float(y1)]
        return out

    # ---- asset writing (content-addressed, so shared pictures are stored once)
    def save_png(self, arr_or_img, stem: str) -> str:
        img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        h = hashlib.sha1(data).hexdigest()[:12]
        if h in self.saved:
            return self.saved[h]
        name = f"{stem}-{h}.png"
        (self.assets / name).write_bytes(data)
        self.saved[h] = "assets/" + name
        return self.saved[h]

    def save_bytes(self, data: bytes, stem: str, ext: str) -> str:
        h = hashlib.sha1(data).hexdigest()[:12]
        if h in self.saved:
            return self.saved[h]
        name = f"{stem}-{h}.{ext}"
        (self.assets / name).write_bytes(data)
        self.saved[h] = "assets/" + name
        return self.saved[h]

    def image_file(self, xref: int, smask: int) -> tuple:
        """Original pixels of an embedded image as PNG or JPEG; returns (path, has_alpha)."""
        info = self.doc.extract_image(xref)
        if smask:
            base = fitz.Pixmap(self.doc, xref)
            if base.n - base.alpha >= 4:
                base = fitz.Pixmap(fitz.csRGB, base)
            mask = fitz.Pixmap(self.doc, smask)
            pix = fitz.Pixmap(base, mask)
            img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            return self.save_png(img, "img"), True
        if info["ext"] in ("jpeg", "jpg") and info.get("cs-name", "DeviceRGB") in ("DeviceRGB", "DeviceGray", ""):
            return self.save_bytes(info["image"], "img", "jpg"), False
        if info["ext"] == "png":
            return self.save_bytes(info["image"], "img", "png"), False
        pix = fitz.Pixmap(self.doc, xref)
        if pix.n - pix.alpha >= 4 or pix.colorspace is None or pix.colorspace.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return self.save_bytes(pix.tobytes("png"), "img", "png"), False

    # ------------------------------------------------------------------------------------ per page
    def run(self):
        groups_h = frame_groups(self.doc)
        groups_b = frame_groups(self.bdoc)
        if len(groups_h) != len(self.doc):
            raise SystemExit("the default PDF must have one page per frame (it has repeated page labels)")
        if len(groups_b) != len(self.doc):
            raise SystemExit(f"frame count differs: {len(self.doc)} pages vs {len(groups_b)} frames in the builds PDF")
        labels = page_labels(self.doc)
        appendix_start = next((i for i in range(1, len(labels)) if labels[i] == "1"), len(labels))
        layout = self.measure_layout(1)
        slides = []
        for i in range(len(self.doc)):
            sl = self.page(i, groups_b[i], appendix=i >= appendix_start, layout=layout)
            sl["label"] = labels[i]
            slides.append(sl)
            print(f"page {i + 1:2d}: {len(sl['elements']):3d} objects, {sl['steps']} step(s)"
                  + (f", {sl['kind']}" if sl["kind"] != "content" else ""), file=sys.stderr)
        deck = {
            "source": {"pdf": str(Path(self.args.pdf).name), "builds_pdf": str(Path(self.args.builds_pdf).name),
                       "pages": len(self.doc)},
            "page_mm": [round(self.doc[0].rect.width / MM, 3), round(self.doc[0].rect.height / MM, 3)],
            "layout": layout,
            "main_total": appendix_start,
            "slides": slides,
            "warnings": self.warnings,
        }
        (self.out / "deck.json").write_text(json.dumps(deck, indent=1))
        print(f"wrote {self.out / 'deck.json'} ({len(slides)} slides, {len(self.saved)} asset files)", file=sys.stderr)

    # ---- the fixed frame parts, measured on a normal content page
    def measure_layout(self, idx: int) -> dict:
        page = self.doc[idx]
        lay = {}
        for s in page.get_texttrace():
            if not s["chars"]:
                continue
            txt = "".join(chr(c[0]) for c in s["chars"])
            base = s["chars"][0][2][1] / MM
            x0 = s["chars"][0][3][0] / MM
            size = tex_size(s["size"])
            col = hexcolor(s["color"])
            if base < 7 and "kicker" not in lay:
                lay["kicker"] = {"x": round(x0, 3), "baseline": round(base, 3), "size": size, "color": col,
                                 "bold": True}
            elif base < 15 and "title" not in lay and size > 12:
                lay["title"] = {"x": round(x0, 3), "baseline": round(base, 3), "size": size, "color": col,
                                "bold": True}
            elif base > 84 and "footer" not in lay:
                lay["footer"] = {"x": round(x0, 3), "baseline": round(base, 3), "size": size, "color": col,
                                 "text": "Tdyssey · Team 28"}
            elif base > 84 and "/" in txt:
                x1 = s["chars"][-1][3][2] / MM
                lay["number"] = {"x1": round(x1, 3), "baseline": round(base, 3), "size": size, "color": col}
        for dr in page.get_drawings():
            r = r_mm(dr["rect"])
            if r[1] > 89 and r[2] - r[0] > 159 and dr.get("fill"):
                lay["track"] = {"y": r[1], "h": round(r[3] - r[1], 3), "color": hexcolor(dr["fill"])}
            elif r[1] > 89 and dr.get("fill"):
                lay["progress_color"] = hexcolor(dr["fill"])
        return lay

    # ---- one page
    def page(self, idx: int, bpages: list, appendix: bool, layout: dict) -> dict:
        page = self.doc[idx]
        W, H = page.rect.width / MM, page.rect.height / MM
        spec = (self.regions.get("titles", {}).get(norm_title(frame_title(page, layout)))
                or self.regions.get("pages", {}).get(str(idx + 1), {}))
        log = page.get_bboxlog()
        drawings = page.get_drawings()
        trace = page.get_texttrace()
        images = page.get_image_info(xrefs=True)
        forms = placed_forms(page)

        # does this page carry the frame decoration (footer, progress bar)?
        has_footer = any(r_mm(d["rect"])[1] > 89 and r_mm(d["rect"])[2] - r_mm(d["rect"])[0] > 159
                         and d.get("fill") is not None for d in drawings)

        items = []  # every visible thing on the page, in paint order
        # images: bboxlog order matches get_image_info order
        img_log = [i for i, e in enumerate(log) if e[0] == "fill-image"]
        for k, im in enumerate(images):
            seq = img_log[k] if k < len(img_log) else 10 ** 6
            items.append({"kind": "image", "seq": seq, "bbox": r_mm(im["bbox"]), "src": im})
        for dr in drawings:
            items.append({"kind": "path", "seq": dr["seqno"], "bbox": r_mm(dr["rect"]), "src": dr})
        for s in trace:
            if s["type"] == 3 or s["opacity"] == 0 or not s["chars"]:
                continue
            items.append({"kind": "text", "seq": s["seqno"], "bbox": r_mm(s["bbox"]), "src": s})

        # regions drawn as pictures: placed PDF graphs (automatic) and TikZ diagrams (regions.json)
        regions = []
        for name, r in forms:
            regions.append({"rect": r_mm(r), "why": "graph", "name": name})
        for reg in spec.get("raster", []):
            regions.append({"rect": reg["rect"], "why": reg.get("why", "diagram"), "name": reg.get("name", "")})
        regions = self.merge_regions(regions)

        # frame decoration and titles
        kicker = title = None
        keep = []
        for it in items:
            b = it["bbox"]
            if it["kind"] == "path":
                fill = it["src"].get("fill")
                if b[0] <= 0.1 and b[1] <= 0.1 and b[2] >= W - 0.1 and b[3] >= H - 0.1:
                    continue  # page background
                if has_footer and b[1] >= 89.0:
                    continue  # progress bar (layout + per-slide bar)
            if it["kind"] == "text" and has_footer and b[1] > 84.3:
                continue  # footer text (layout)
            keep.append(it)
        items = keep

        # video posters (matched by bytes); a clipped poster only covers its clip area
        vids = self.match_videos(idx, items)
        for it in items:
            v = vids.get(id(it))
            if v and v["id"] in self.clips:
                c = self.clips[v["id"]]
                b = it["bbox"]
                it["vis"] = [max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3])]

        # occlusion: drop anything fully covered by a later opaque rectangle or image
        covers = []
        for it in items:
            if it["kind"] == "image" and not it["src"].get("has-mask") and not self._image_has_alpha(it["src"]):
                covers.append(it)
            elif it["kind"] == "path":
                dr = it["src"]
                if dr.get("fill") is not None and (dr.get("fill_opacity") or 1) >= 0.999 and self._is_rect(dr):
                    covers.append(it)
        visible = []
        for it in items:
            hidden = any(c["seq"] > it["seq"] and contains(c.get("vis", c["bbox"]), it["bbox"], tol=0.15)
                         and c is not it for c in covers)
            if not hidden:
                visible.append(it)
        items = visible

        # white masks that no longer cover anything visible are dropped (they only hid old content)
        keep = []
        for it in items:
            if it["kind"] == "path":
                dr = it["src"]
                if (dr.get("fill") is not None and hexcolor(dr["fill"]) == "FFFFFF" and dr.get("color") is None
                        and self._is_rect(dr)):
                    under = [o for o in items if o["seq"] < it["seq"] and intersects(o["bbox"], it["bbox"])]
                    if not under:
                        continue
            keep.append(it)
        items = keep

        # titles
        content_page = has_footer
        if content_page:
            for it in items:
                if it["kind"] != "text":
                    continue
                s = it["src"]
                base = s["chars"][0][2][1] / MM
                size = tex_size(s["size"])
                if kicker is None and abs(base - layout["kicker"]["baseline"]) < 0.4 and abs(size - layout["kicker"]["size"]) < 0.2:
                    kicker = it
                elif title is None and abs(base - layout["title"]["baseline"]) < 0.4 and abs(size - layout["title"]["size"]) < 0.2:
                    title = it
            # titles are single spans per line; collect all spans on those baselines
        kicker_txt = self.span_line_text(items, kicker) if kicker else None
        title_txt = self.span_line_text(items, title) if title else None
        if kicker:
            kb = layout["kicker"]["baseline"]
            items = [it for it in items if not (it["kind"] == "text" and abs(it["src"]["chars"][0][2][1] / MM - kb) < 0.4
                                                 and it["bbox"][1] < 8)]
        if title:
            tb = layout["title"]["baseline"]
            items = [it for it in items if not (it["kind"] == "text" and abs(it["src"]["chars"][0][2][1] / MM - tb) < 0.4
                                                 and it["bbox"][1] < 15)]

        # assign items to raster regions
        for reg in regions:
            reg["seq"] = min([it["seq"] for it in items if center_in(reg["rect"], it["bbox"])] or [10 ** 6])
        rest = []
        for it in items:
            if any(center_in(reg["rect"], it["bbox"]) and contains(reg["rect"], it["bbox"], tol=1.5)
                   for reg in regions):
                continue
            rest.append(it)
        items = rest

        # build steps: overlay pages of this frame (builds PDF)
        nsteps = len(bpages)

        elements = []

        # text -> lines -> boxes
        text_items = [it for it in items if it["kind"] == "text"]
        boxes = self.text_boxes(text_items, items)
        for b in boxes:
            elements.append(b)
        for it in items:
            if it["kind"] == "image":
                el = self.image_element(it, vids)
                if el:
                    elements.append(el)
            elif it["kind"] == "path":
                el = self.path_element(it)
                if el:
                    elements.append(el)
        for reg in regions:
            elements.extend(self.raster_elements(idx, bpages, reg))

        # step of every native element (raster layers already carry theirs)
        if nsteps > 1:
            self.assign_steps(idx, bpages, elements)
        for el in elements:
            el.setdefault("step", 1)
        elements.sort(key=lambda e: (e["seq"], e.get("sub", 0)))

        # launch links (local clips) not covered by an embedded video
        links = []
        for ln in page.get_links():
            r = r_mm(ln["from"])
            if ln["kind"] in (fitz.LINK_LAUNCH, fitz.LINK_GOTOR):
                if any(e["type"] == "video" and intersects([e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]], r)
                       for e in elements):
                    continue
                target = ln.get("file", "")
                target = str((PRES / "slides" / target).resolve().relative_to(PRES)) if target else ""
                links.append({"rect": r, "file": target})
            elif ln["kind"] == fitz.LINK_URI:
                links.append({"rect": r, "url": ln.get("uri")})
        self.attach_links(elements, links)
        # a scene that opens a local clip becomes an embedded video with the scene as its cover
        for el in elements:
            ln = el.get("link") or {}
            if el["type"] == "image" and ln.get("file", "").lower().endswith(".mp4") and (PRES / ln["file"]).exists():
                cover = el["path"]
                if not cover.endswith(".png"):
                    cover = self.save_png(Image.open(self.out / cover).convert("RGB"), "cover")
                el.update({"type": "video", "video_id": Path(ln["file"]).stem, "mp4": ln["file"], "cover": cover,
                           "poster": None, "local": True})
        elements = self.videos_on_top(idx, elements)

        # frame decoration of this page
        number_text, progress_w = None, None
        if content_page:
            for s in page.get_texttrace():
                if s["chars"] and s["chars"][0][2][1] / MM > 84 and "/" in "".join(chr(c[0]) for c in s["chars"]):
                    number_text = "".join(chr(c[0]) for c in s["chars"]).replace("/", " / ")
            for dr in drawings:
                r = r_mm(dr["rect"])
                if r[1] >= 89 and r[2] - r[0] < W - 0.1 and dr.get("fill") is not None:
                    progress_w = round(r[2] - r[0], 3)
        kind = "content" if content_page else ("title" if idx == 0 else "plain")
        plain_title = None
        if not content_page:
            plain_title = self.plain_title(elements)
        return {
            "page": idx + 1,
            "kind": kind,
            "appendix": appendix,
            "section": kicker_txt,
            "title": title_txt,
            "plain_title": plain_title,
            "number_text": number_text,
            "progress_w": progress_w,
            "steps": max([e["step"] for e in elements] or [1]),
            "elements": [{k: v for k, v in e.items() if k not in ("seq", "sub", "bbox")} | {"z": n}
                         for n, e in enumerate(elements)],
        }

    # ---- helpers for page()
    def _image_has_alpha(self, im) -> bool:
        xref = im.get("xref", 0)
        if not xref:
            return False
        try:
            return bool(self.doc.xref_get_key(xref, "SMask")[0] != "null")
        except Exception:
            return False

    @staticmethod
    def _is_rect(dr) -> bool:
        its = dr["items"]
        if len(its) == 1 and its[0][0] == "re":
            return True
        if len(its) == 1 and its[0][0] == "qu":
            q = its[0][1]
            return abs(q.ul.y - q.ur.y) < 0.01 and abs(q.ul.x - q.ll.x) < 0.01
        if all(i[0] == "l" for i in its) and 3 <= len(its) <= 4:
            pts = [i[1] for i in its] + [its[-1][2]]
            xs = {round(p.x, 1) for p in pts}
            ys = {round(p.y, 1) for p in pts}
            return len(xs) <= 2 and len(ys) <= 2
        return False

    @staticmethod
    def span_line_text(items, it) -> str:
        base = it["src"]["chars"][0][2][1]
        spans = [o["src"] for o in items if o["kind"] == "text" and abs(o["src"]["chars"][0][2][1] - base) < 1
                 and abs(o["bbox"][1] - it["bbox"][1]) < 3]
        # spans left to right, characters inside a span in stream order (never re-sorted)
        spans.sort(key=lambda sp: sp["chars"][0][3][0])
        out, prev = "", None
        for sp in spans:
            for c in span_chars(sp):
                if prev is not None and (c["x0"] - prev) * MM > 0.15 * sp["size"]:
                    out += " "
                out += c["ch"]
                prev = c["x1"]
        return out

    @staticmethod
    def merge_regions(regions):
        """Regions that overlap are merged (a diagram and an included graph inside it)."""
        out = []
        for r in sorted(regions, key=lambda r: (r["rect"][1], r["rect"][0])):
            for o in out:
                if intersects(o["rect"], r["rect"]):
                    a, b = o["rect"], r["rect"]
                    o["rect"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    o["why"] = o["why"] if o["why"] == r["why"] else "diagram+graph"
                    break
            else:
                out.append(dict(r))
        return out

    def plain_title(self, elements):
        best = None
        for e in elements:
            if e["type"] == "text":
                size = max(r["size"] for p in e["paragraphs"] for r in p["runs"])
                bold = all(r.get("bold") for p in e["paragraphs"] for r in p["runs"])
                if bold and size >= 20 and (best is None or size > best[0]):
                    best = (size, e)
        if best:
            best[1]["role"] = "title"
            return "".join(r["text"] for r in best[1]["paragraphs"][0]["runs"])
        return None

    # ---- videos
    def match_videos(self, idx, items) -> dict:
        """Poster images are recognised by their bytes (several posters share one position)."""
        found = {}
        for it in items:
            if it["kind"] != "image":
                continue
            b = it["bbox"]
            cands = [v for v in self.videos
                     if abs(b[0] - v["x"]) < 0.4 and abs(b[1] - v["y"]) < 0.4 and abs(b[2] - b[0] - v["w"]) < 0.4
                     and abs(b[3] - b[1] - v["h"]) < 0.4]
            if not cands:
                continue
            xref = it["src"].get("xref", 0)
            data = self.doc.extract_image(xref)["image"] if xref else b""
            exact = [v for v in cands if (PRES / v["poster"]).read_bytes() == data]
            if exact:
                found[id(it)] = exact[0]
            else:
                found[id(it)] = cands[0]
                self.warnings.append(f"page {idx + 1}: poster bytes differ from {cands[0]['poster']}; matched by position")
        return found

    # ---- images
    def image_element(self, it, vids):
        im = it["src"]
        b = it["bbox"]
        el = {"type": "image", "seq": it["seq"], "x": b[0], "y": b[1], "w": round(b[2] - b[0], 3),
              "h": round(b[3] - b[1], 3), "bbox": b}
        v = vids.get(id(it))
        if v:
            clip = self.clips.get(v["id"])
            el.update({"type": "video", "video_id": v["id"], "mp4": v["mp4"], "poster": v["poster"],
                       "url": self.drive.get(v["id"]) or v["url"]})
            # PowerPoint stores a video cover as PNG; the poster pixels are kept as they are
            el["cover"] = self.save_png(Image.open(PRES / v["poster"]).convert("RGB"), f"poster-{v['id']}")
            if clip:
                el["crop"] = self.crop_fractions(b, clip)
            return el
        xref = im.get("xref", 0)
        if not xref:
            self.warnings.append(f"page image without xref at {b}")
            return None
        smask = 0
        try:
            sm = self.doc.xref_get_key(xref, "SMask")
            if sm[0] == "xref":
                smask = int(sm[1].split()[0])
        except Exception:
            pass
        el["path"], el["alpha"] = self.image_file(xref, smask)
        return el

    @staticmethod
    def crop_fractions(b, clip):
        w, h = b[2] - b[0], b[3] - b[1]
        return {"l": max(0.0, (clip[0] - b[0]) / w), "t": max(0.0, (clip[1] - b[1]) / h),
                "r": max(0.0, (b[2] - clip[2]) / w), "b": max(0.0, (b[3] - clip[3]) / h)}

    # ---- vector drawing -> native shape
    def path_element(self, it):
        dr = it["src"]
        b = it["bbox"]
        fill = hexcolor(dr.get("fill")) if dr.get("fill") is not None else None
        stroke = hexcolor(dr.get("color")) if dr.get("color") is not None and "s" in dr["type"] else None
        if fill is None and stroke is None:
            return None
        el = {"type": "shape", "seq": it["seq"], "bbox": b}
        if fill:
            el["fill"] = fill
            fo = dr.get("fill_opacity")
            if fo is not None and fo < 0.999:
                el["fill_alpha"] = round(fo, 3)
        if stroke:
            el["line"] = stroke
            el["line_w"] = round(dr.get("width") or 1.0, 3)
            so = dr.get("stroke_opacity")
            if so is not None and so < 0.999:
                el["line_alpha"] = round(so, 3)
            dash = self.parse_dash(dr.get("dashes"))
            if dash:
                el["dash"] = dash
        its = dr["items"]
        w, h = b[2] - b[0], b[3] - b[1]
        el.update({"x": b[0], "y": b[1], "w": round(w, 3), "h": round(h, 3)})
        if self._is_rect(dr) and not (len(its) <= 2 and all(i[0] == "l" for i in its)):
            el["geom"] = "rect"
            return el
        if len(its) == 1 and its[0][0] == "l":
            p, q = its[0][1], its[0][2]
            el["geom"] = "line"
            el["x1"], el["y1"], el["x2"], el["y2"] = (round(p.x / MM, 3), round(p.y / MM, 3), round(q.x / MM, 3),
                                                      round(q.y / MM, 3))
            el.pop("fill", None)
            return el
        rr = self.rounded_rect(its, b)
        if rr is not None:
            el["geom"] = "roundRect"
            el["radius"] = rr
            return el
        if len(its) == 4 and all(i[0] == "c" for i in its) and w > 0 and abs(w - h) < 0.05 * max(w, h) + 0.05:
            el["geom"] = "ellipse"
            return el
        # free-form path, points relative to the bounding box (mm)
        pts = []
        cur = None
        for item in its:
            op = item[0]
            if op == "l":
                p, q = item[1], item[2]
                if cur is None or abs(cur.x - p.x) > 0.01 or abs(cur.y - p.y) > 0.01:
                    pts.append({"m": [self.rel(p, b)]})
                pts.append({"l": [self.rel(q, b)]})
                cur = q
            elif op == "c":
                p, c1, c2, q = item[1], item[2], item[3], item[4]
                if cur is None or abs(cur.x - p.x) > 0.01 or abs(cur.y - p.y) > 0.01:
                    pts.append({"m": [self.rel(p, b)]})
                pts.append({"c": [self.rel(c1, b), self.rel(c2, b), self.rel(q, b)]})
                cur = q
            elif op == "re":
                r = item[1]
                pts.append({"m": [self.rel(r.tl, b)]})
                pts.append({"l": [self.rel(r.tr, b)]})
                pts.append({"l": [self.rel(r.br, b)]})
                pts.append({"l": [self.rel(r.bl, b)]})
                pts.append({"z": []})
                cur = None
            elif op == "qu":
                q = item[1]
                pts.append({"m": [self.rel(q.ul, b)]})
                pts.append({"l": [self.rel(q.ur, b)]})
                pts.append({"l": [self.rel(q.lr, b)]})
                pts.append({"l": [self.rel(q.ll, b)]})
                pts.append({"z": []})
                cur = None
        if dr.get("closePath"):
            pts.append({"z": []})
        el["geom"] = "path"
        el["path"] = pts
        if w < 0.01 or h < 0.01:  # straight horizontal or vertical polyline
            el["w"] = max(el["w"], 0.01)
            el["h"] = max(el["h"], 0.01)
        return el

    @staticmethod
    def rel(p, b):
        return [round(p.x / MM - b[0], 4), round(p.y / MM - b[1], 4)]

    @staticmethod
    def parse_dash(d):
        if not d:
            return None
        m = re.match(r"\[\s*([^\]]*)\]\s*([-\d.]+)", d)
        if not m or not m.group(1).strip():
            return None
        vals = [float(v) for v in m.group(1).split()]
        return [round(v, 3) for v in vals]

    @staticmethod
    def rounded_rect(its, b):
        kinds = "".join(i[0][0] for i in its)
        if sorted(kinds) != sorted("llllcccc") and sorted(kinds) != sorted("cccc" + "lll"):
            return None
        if kinds.count("c") != 4:
            return None
        # corner radius: distance from a curve's start to its end along x
        radii = []
        for i in its:
            if i[0] == "c":
                p, q = i[1], i[4]
                radii.append(max(abs(p.x - q.x), abs(p.y - q.y)) / MM)
        r = sum(radii) / len(radii)
        if max(radii) - min(radii) > 0.2:
            return None
        return round(r, 3)

    # ---- text
    def text_boxes(self, text_items, all_items):
        chars = []
        for it in text_items:
            s = it["src"]
            font = s["font"]
            bold = "Bold" in font or "bold" in font or "Bx" in font
            italic = "Oblique" in font or "Italic" in font
            col = hexcolor(s["color"])
            for c in span_chars(s):
                chars.append(dict(c, size=s["size"], bold=bold, italic=italic, color=col, seq=it["seq"]))
        # lines: consecutive characters on one baseline (smaller raised/lowered characters join as super/subscripts)
        lines = []
        for c in chars:
            ln = lines[-1] if lines else None
            if ln is not None:
                last = ln["chars"][-1]
                same_base = abs(c["base"] - ln["base"]) < 0.25
                gap = c["x0"] - last["x1"]
                shift = abs(c["base"] - ln["base"])
                em_ln = ln["size"] / MM
                # a super- or subscript: clearly smaller, raised or lowered, and set right after the previous glyph
                script = (c["size"] < 0.8 * ln["size"] and 0.12 * em_ln < shift < 0.6 * em_ln
                          and abs(gap) < 0.25 * em_ln)
                back_script = (bool(last.get("script")) and shift < 0.25 and abs(gap) < 0.4 * c["size"] / MM)
                if (same_base or script or back_script) and -0.4 * c["size"] / MM < gap < 2.0 * max(c["size"], last["size"]) / MM \
                        and abs(c["seq"] - last["seq"]) < 50:
                    if script:
                        c["script"] = "sup" if c["base"] < ln["base"] else "sub"
                    ln["chars"].append(c)
                    continue
            lines.append({"chars": [c], "base": c["base"], "size": c["size"]})
        # split lines at large gaps (columns, \hfill), insert spaces at word gaps
        segs = []
        for ln in lines:
            cur = None
            for c in ln["chars"]:
                if cur is not None:
                    prev = cur["chars"][-1]
                    em = min(c["size"], prev["size"]) / MM
                    gap = c["x0"] - prev["x1"]
                    if gap > 0.6 * em:  # wider than any word space: separate boxes (columns, \hfill, \hspace)
                        segs.append(cur)
                        cur = None
                    elif gap > 0.12 * em:
                        cur["chars"].append(dict(c, ch=" ", x1=c["x0"], space=True))
                if cur is None:
                    cur = {"chars": [], "base": ln["base"], "size": ln["size"]}
                cur["chars"].append(c)
            if cur:
                segs.append(cur)
        for sg in segs:
            body = [c for c in sg["chars"] if not c.get("script")]
            sg["x0"] = min(c["x0"] for c in sg["chars"])
            sg["x1"] = max(c["x1"] for c in sg["chars"])
            sg["size"] = max(c["size"] for c in body) if body else sg["size"]
            sg["seq"] = min(c["seq"] for c in sg["chars"])
            sg["top"] = sg["base"] - 0.75 * sg["size"] / MM
        # anchors for single-line alignment: picture and box edges, page centre and text margins
        anchors_c = [80.0]
        anchors_r = [152.0]
        for it in all_items:
            if it["kind"] in ("image", "path"):
                b = it["bbox"]
                if b[2] - b[0] > 3:
                    anchors_c.append((b[0] + b[2]) / 2)
                    anchors_r.append(b[2])
        # group segments into blocks
        segs.sort(key=lambda s: (s["seq"], s["base"], s["x0"]))
        used = [False] * len(segs)
        blocks = []
        order = sorted(range(len(segs)), key=lambda k: (segs[k]["base"], segs[k]["x0"]))
        for k in order:
            if used[k]:
                continue
            used[k] = True
            blk = [segs[k]]
            while True:
                last = blk[-1]
                cand = None
                for j in order:
                    if used[j]:
                        continue
                    s = segs[j]
                    step = s["base"] - last["base"]
                    if step <= 0.3 or step > 2.4 * last["size"] / MM:
                        continue
                    if abs(s["size"] - last["size"]) > 0.6:
                        continue
                    if not self.same_alignment(blk, s):
                        continue
                    # nothing else may sit between the two lines in this column
                    if cand is None or s["base"] < cand["base"]:
                        cand = s
                        cj = j
                if cand is None:
                    break
                between = [s for j, s in enumerate(segs) if not used[j] and j != cj and last["base"] < s["base"] < cand["base"]
                           and s["x0"] < max(c["x1"] for c in blk) and s["x1"] > min(c["x0"] for c in blk)]
                if between:
                    break
                used[cj] = True
                blk.append(cand)
            blocks.append(blk)
        out = []
        for blk in blocks:
            out.append(self.block_element(blk, anchors_c, anchors_r))
        return out

    @staticmethod
    def _align_of(blk):
        if len(blk) < 2:
            return None
        x0s = [s["x0"] for s in blk]
        x1s = [s["x1"] for s in blk]
        cs = [(s["x0"] + s["x1"]) / 2 for s in blk]
        if max(x0s) - min(x0s) < 0.35:
            return "left"
        if max(cs) - min(cs) < 0.35:
            return "center"
        if max(x1s) - min(x1s) < 0.35:
            return "right"
        return None

    def same_alignment(self, blk, s):
        al = self._align_of(blk)
        cand = blk + [s]
        al2 = self._align_of(cand)
        if al2 is None:
            return False
        if al is not None and al2 != al:
            # a block that was left-aligned stays left-aligned
            x0s = [c["x0"] for c in cand]
            return al == "left" and max(x0s) - min(x0s) < 0.35
        return True

    def block_element(self, blk, anchors_c, anchors_r):
        align = self._align_of(blk)
        if align is None:  # single line: guess from the page geometry
            s = blk[0]
            c = (s["x0"] + s["x1"]) / 2
            if abs(s["x0"] - 8.0) < 0.3:
                align = "left"
            elif any(abs(c - a) < 0.5 for a in anchors_c) and s["x1"] - s["x0"] > 0:
                align = "center"
            elif any(abs(s["x1"] - a) < 0.6 for a in anchors_r):
                align = "right"
            else:
                align = "left"
        sizes = [s["size"] for s in blk]
        steps = [blk[i + 1]["base"] - blk[i]["base"] for i in range(len(blk) - 1)]
        size_pt = max(sizes)
        if steps and min(steps) <= 1.5 * size_pt / MM:
            lead_mm = min(steps)
        else:
            lead_mm = 1.2 * size_pt / MM
        paragraphs = []
        for i, s in enumerate(blk):
            runs = []
            for c in s["chars"]:
                fmt = (round(tex_size(c["size"]), 1), c["bold"], c["italic"], c["color"], c.get("script"))
                if runs and runs[-1]["fmt"] == fmt:
                    runs[-1]["text"] += c["ch"]
                elif c.get("space") and runs:
                    runs[-1]["text"] += " "
                else:
                    runs.append({"fmt": fmt, "text": c["ch"]})
            pr = {"runs": [{"text": r["text"], "size": r["fmt"][0], "bold": r["fmt"][1], "italic": r["fmt"][2],
                            "color": r["fmt"][3], **({"script": r["fmt"][4]} if r["fmt"][4] else {})}
                           for r in runs]}
            if i > 0:
                extra = blk[i]["base"] - blk[i - 1]["base"] - lead_mm
                if extra > 0.05:
                    pr["space_before_pt"] = round(extra * MM, 2)
            paragraphs.append(pr)
        x0 = min(s["x0"] for s in blk)
        x1 = max(s["x1"] for s in blk)
        width = x1 - x0
        slack = 0.08 * width + 1.0  # the theme font is narrower than Latin Modern Sans; never wrap
        if align == "left":
            bx = x0
        elif align == "right":
            bx = x1 - width - slack
        else:
            bx = (x0 + x1) / 2 - (width + slack) / 2
        lead_pt = lead_mm * MM
        first = blk[0]
        top = first["base"] - (lead_mm - 0.26 * tex_size(first["size"]) / TEXPT / MM)
        height = (blk[-1]["base"] - first["base"]) + lead_mm
        return {"type": "text", "seq": min(s["seq"] for s in blk), "x": round(bx, 3), "y": round(top, 3),
                "w": round(width + slack, 3), "h": round(height, 3), "align": align,
                "line_pt": round(lead_pt, 2), "paragraphs": paragraphs,
                "bbox": [x0, min(s["top"] for s in blk), x1, blk[-1]["base"] + 0.25 * size_pt / MM],
                "baseline": round(first["base"], 3)}

    # ---- videos: nothing may sit above a video, or it swallows the click that plays it
    def videos_on_top(self, idx, elements):
        """Drawings and pictures that Beamer draws over a video (frame, play mark) are baked into the cover: the
        cover becomes the PDF page rendered over the video's box, and those objects are dropped. Each video's MP4
        is re-muxed with its index first (ffmpeg -c copy -movflags +faststart; the streams are not re-encoded)."""
        order = sorted(range(len(elements)), key=lambda k: (elements[k]["seq"], elements[k].get("sub", 0)))
        rank = {k: r for r, k in enumerate(order)}
        drop = set()
        for k, v in enumerate(elements):
            if v["type"] != "video":
                continue
            box = [v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]]
            if v.get("crop"):
                c = v["crop"]
                box = [v["x"] + c["l"] * v["w"], v["y"] + c["t"] * v["h"], v["x"] + v["w"] * (1 - c["r"]),
                       v["y"] + v["h"] * (1 - c["b"])]
            above = []
            for j, e in enumerate(elements):
                if j == k or j in drop or rank[j] < rank[k]:
                    continue
                eb = [e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]]
                if e["type"] == "shape" and e.get("geom") == "line":
                    eb = [min(e["x1"], e["x2"]), min(e["y1"], e["y2"]), max(e["x1"], e["x2"]), max(e["y1"], e["y2"])]
                ix = min(box[2], eb[2]) - max(box[0], eb[0])
                iy = min(box[3], eb[3]) - max(box[1], eb[1])
                if ix > 0.02 and iy > -0.02 or iy > 0.02 and ix > -0.02:
                    above.append(j)
            if not above:
                continue
            bad = [elements[j] for j in above if elements[j]["type"] in ("text", "video") or
                   elements[j].get("step", 1) > v.get("step", 1) or not contains(box, self._box(elements[j]), tol=0.6)]
            if bad or v.get("crop"):
                self.warnings.append(f"page {idx + 1}: objects above video {v['video_id']} were left in place: "
                                     + ", ".join(e["type"] for e in bad))
                continue
            arr = self.rend.crop(idx, box, self.args.dpi)
            v["cover"] = self.save_png(arr, f"cover-{v['video_id']}")
            v["baked"] = len(above)
            drop.update(above)
        out = [e for j, e in enumerate(elements) if j not in drop]
        for v in out:
            if v["type"] == "video":
                v["media"] = self.faststart(PRES / v["mp4"])
        return out

    @staticmethod
    def _box(e):
        if e["type"] == "shape" and e.get("geom") == "line":
            return [min(e["x1"], e["x2"]), min(e["y1"], e["y2"]), max(e["x1"], e["x2"]), max(e["y1"], e["y2"])]
        return [e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]]

    def faststart(self, src: Path) -> str:
        import shutil
        import subprocess
        import tempfile

        data = src.read_bytes()
        h = hashlib.sha1(data).hexdigest()[:12]
        key = "mp4:" + h
        if key in self.saved:
            return self.saved[key]
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("ffmpeg is needed to re-mux the videos (-c copy -movflags +faststart)")
        name = f"video-{src.stem}-{h}.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / name
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-map", "0", "-c", "copy",
                            "-movflags", "+faststart", str(dst)], check=True)
            (self.assets / name).write_bytes(dst.read_bytes())
        self.saved[key] = "assets/" + name
        return self.saved[key]

    # ---- raster regions with build layers
    def raster_elements(self, idx, bpages, reg):
        rect = reg["rect"]
        dpi = self.args.dpi
        final = self.rend.crop(idx, rect, dpi)
        base_el = {"type": "picture", "role": reg["why"], "x": rect[0], "y": rect[1],
                   "w": round(rect[2] - rect[0], 3), "h": round(rect[3] - rect[1], 3), "seq": reg["seq"]}
        stem = "graph" if reg["why"] == "graph" else "diagram"
        if len(bpages) == 1:
            return [dict(base_el, path=self.save_png(self.white_to_alpha(final), stem), step=1, sub=0)]
        states = [self.brend.crop(b, rect, dpi) for b in bpages]
        if states[-1].shape == final.shape and np.abs(states[-1].astype(int) - final.astype(int)).max() > 8:
            self.warnings.append(f"page {idx + 1}: final overlay differs from the default PDF inside {rect}; "
                                 "the default PDF is used for the last step")
            states[-1] = final
        out = []
        prev = None
        for k, st in enumerate(states):
            if prev is None:
                layer = self.white_to_alpha(st)
                if layer[:, :, 3].max() == 0:
                    prev = st
                    continue
            else:
                diff = np.any(st != prev, axis=2)
                if not diff.any():
                    prev = st
                    continue
                layer = np.zeros((st.shape[0], st.shape[1], 4), dtype=np.uint8)
                layer[:, :, :3] = st
                layer[:, :, 3] = np.where(diff, 255, 0)
            out.append(dict(base_el, path=self.save_png(layer, f"{stem}-step{k + 1}"), step=k + 1, sub=k,
                            name=f"{stem} step {k + 1}"))
            prev = st
        return out

    @staticmethod
    def white_to_alpha(arr):
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = np.where(np.all(arr >= 254, axis=2), 0, 255)
        return rgba

    # ---- steps of native elements
    def assign_steps(self, idx, bpages, elements):
        dpi = 100
        states = [self.brend.page(b, dpi) for b in bpages]
        final = states[-1]
        for el in elements:
            if el["type"] == "picture" and "step" in el:
                continue
            b = el.get("bbox") or [el["x"], el["y"], el["x"] + el["w"], el["y"] + el["h"]]
            x0, y0, x1, y1 = px_box(b, dpi, final.shape)
            ref = final[y0:y1, x0:x1].astype(int)
            step = 1
            for k in range(len(states) - 1, -1, -1):
                cur = states[k][y0:y1, x0:x1].astype(int)
                mism = np.mean(np.any(np.abs(cur - ref) > 24, axis=2))
                if mism > 0.02:
                    step = k + 2
                    break
            el["step"] = min(step, len(states))

    # ---- hyperlinks
    @staticmethod
    def attach_links(elements, links):
        for ln in links:
            r = ln["rect"]
            best, area = None, 0
            for e in elements:
                if e["type"] not in ("image", "picture", "shape", "text"):
                    continue
                eb = [e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]]
                ix = max(0, min(eb[2], r[2]) - max(eb[0], r[0])) * max(0, min(eb[3], r[3]) - max(eb[1], r[1]))
                if ix > area:
                    best, area = e, ix
            if best is not None:
                best["link"] = {k: v for k, v in ln.items() if k != "rect"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", required=True, help="default Beamer PDF (one final build per page)")
    ap.add_argument("--builds-pdf", required=True, help="Beamer PDF with every overlay step (\\def\\tdanimated{1})")
    ap.add_argument("--regions", default=str(HERE / "regions.json"), help="diagram regions drawn as pictures")
    ap.add_argument("--out", default=str(HERE / "out"), help="output folder for deck.json and assets/")
    ap.add_argument("--dpi", type=int, default=400, help="resolution of graph and diagram pictures")
    args = ap.parse_args()
    Extractor(args).run()


if __name__ == "__main__":
    main()
