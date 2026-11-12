#!/usr/bin/env python3
"""Checks for the generated PowerPoint deck.

  qa.py inspect deck.pptx            structure: fonts and sizes used, pictures, videos, animations, slide count
  qa.py compare beamer.pdf pptx.pdf  side-by-side sheets: Beamer page | PowerPoint render | overlay of both
  qa.py preview deck.pptx beamer.pdf [--builds-pdf builds.pdf]
                                     the same sheets drawn from the .pptx XML with a Calibri-metric font (Carlito),
                                     plus every click state against the Beamer overlay pages. This is NOT a PowerPoint
                                     or LibreOffice render; it checks positions, line breaks, overflow and build steps
                                     when no office renderer is available.

The overlay column draws both pages on top of each other: ink in both is dark, ink only in the Beamer page is red,
ink only in the PowerPoint page is cyan.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


# ------------------------------------------------------------------------------------------------- inspect
def inspect(pptx: Path) -> int:
    from lxml import etree

    z = zipfile.ZipFile(pptx)
    names = z.namelist()
    slides = sorted([n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)],
                    key=lambda n: int(re.findall(r"\d+", n)[-1]))
    theme = etree.fromstring(z.read("ppt/theme/theme1.xml"))
    major = theme.find(".//a:majorFont/a:latin", NS).get("typeface")
    minor = theme.find(".//a:minorFont/a:latin", NS).get("typeface")
    faces = collections.Counter()
    sizes = collections.Counter()
    problems = []
    n_pic = n_vid = n_txt = n_anim_slides = n_effects = 0
    for sn in slides:
        x = etree.fromstring(z.read(sn))
        num = int(re.findall(r"\d+", sn)[-1])
        for lat in x.iter("{%s}latin" % NS["a"]):
            faces[lat.get("typeface")] += 1
        for tag in ("rPr", "endParaRPr", "defRPr"):
            for rp in x.iter("{%s}%s" % (NS["a"], tag)):
                if rp.get("sz"):
                    sizes[int(rp.get("sz")) / 100] += 1
        for pic in x.iter("{%s}pic" % NS["p"]):
            if pic.find(".//a:videoFile", NS) is not None:
                n_vid += 1
            else:
                n_pic += 1
        n_txt += len(x.findall(".//p:sp/p:txBody", NS))
        # pictures must not be slide backgrounds
        if x.find(".//p:bg/p:bgPr/a:blipFill", NS) is not None:
            problems.append(f"slide {num}: picture used as background")
        # animation targets must exist and ids must be unique
        ids = [int(c.get("id")) for c in x.iter("{%s}cNvPr" % NS["p"])]
        if len(ids) != len(set(ids)):
            problems.append(f"slide {num}: duplicate shape ids")
        timing = x.find("p:timing", NS)
        if timing is not None:
            tgts = [int(t.get("spid")) for t in timing.iter("{%s}spTgt" % NS["p"])]
            missing = sorted(set(tgts) - set(ids))
            if missing:
                problems.append(f"slide {num}: animation targets missing shapes {missing}")
            clicks = timing.findall(".//p:seq/p:cTn/p:childTnLst/p:par", NS)
            eff = [c for c in timing.iter("{%s}cTn" % NS["p"]) if c.get("presetClass") == "entr"]
            if eff:
                n_anim_slides += 1
                n_effects += len(eff)
                print(f"  slide {num:2d}: {len(clicks)} click(s), {len(eff)} appear effect(s)")
            ctn_ids = [int(c.get("id")) for c in timing.iter("{%s}cTn" % NS["p"])]
            if len(ctn_ids) != len(set(ctn_ids)):
                problems.append(f"slide {num}: duplicate timing node ids")
    print(f"slides: {len(slides)}")
    print(f"theme fonts: heading {major}, body {minor}")
    print(f"explicit font faces in slides: {dict(faces) or 'none (all text uses the theme font)'}")
    print("font sizes used (pt: count): " + ", ".join(f"{k:g}: {v}" for k, v in sorted(sizes.items())))
    print(f"text shapes: {n_txt}, pictures: {n_pic}, embedded videos: {n_vid}")
    print(f"slides with click builds: {n_anim_slides}, appear effects: {n_effects}")
    media = [n for n in names if n.startswith("ppt/media/")]
    total = sum(z.getinfo(n).file_size for n in media)
    print(f"media parts: {len(media)} ({total / 1e6:.1f} MB)")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


# ------------------------------------------------------------------------------------------------- sheets
def render_pdf(pdf: Path, dpi: int):
    import fitz
    from PIL import Image

    doc = fitz.open(pdf)
    out = []
    for p in doc:
        pix = p.get_pixmap(dpi=dpi, alpha=False)
        out.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return out


def overlay(a, b):
    from PIL import Image, ImageChops

    ga, gb = a.convert("L"), b.convert("L").resize(a.size)
    # red channel shows the PowerPoint ink, green/blue the Beamer ink
    return Image.merge("RGB", (gb, ga, ga))


def sheets(left, right, out_dir: Path, titles=("Beamer PDF", "PowerPoint"), per_sheet=4, labels=None, numbers=None):
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = left[0].size
    pad, head = 10, 26
    files = []
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(left), per_sheet):
        rows = list(range(start, min(start + per_sheet, len(left))))
        S = Image.new("RGB", (3 * w + 4 * pad, len(rows) * (h + head + pad) + pad), "#6b7078")
        d = ImageDraw.Draw(S)
        for r, i in enumerate(rows):
            y = pad + r * (h + head + pad)
            a = left[i]
            b = right[i].resize((w, h)) if i < len(right) else Image.new("RGB", (w, h), "#ddd")
            for c, (img, t) in enumerate(((a, titles[0]), (b, titles[1]), (overlay(a, b), "overlay"))):
                x = pad + c * (w + pad)
                S.paste(img, (x, y + head))
                num = numbers[i] if numbers else i + 1
                lab = f"page {num}  {t}" + (f"  {labels[i]}" if labels and c == 0 else "")
                d.text((x + 2, y + 4), lab, fill="white", font=font)
        a_, b_ = (numbers[start], numbers[rows[-1]]) if numbers else (start + 1, rows[-1] + 1)
        f = out_dir / f"compare-{str(a_).replace('.', '_')}-{str(b_).replace('.', '_')}.png"
        S.save(f)
        files.append(f)
    return files


def compare(beamer: Path, pptx_pdf: Path, out_dir: Path, dpi: int) -> int:
    left = render_pdf(beamer, dpi)
    right = render_pdf(pptx_pdf, dpi)
    if len(left) != len(right):
        print(f"page count differs: Beamer {len(left)}, PowerPoint {len(right)}")
    files = sheets(left, right, out_dir, titles=("Beamer PDF", "PowerPoint (LibreOffice render)"))
    print("\n".join(str(f) for f in files))
    return 0


# ------------------------------------------------------------------------------------------------- preview
class PptxPreview:
    """Draws the slides of a .pptx from its XML (layout shapes, placeholders, text boxes with word wrap, pictures with
    crops, preset and free-form shapes). It approximates PowerPoint with a Calibri-metric font (Carlito). It is NOT a
    PowerPoint or LibreOffice render: use it to check positions, line breaks, overflow and build steps."""

    EMU_MM = 36000.0

    def __init__(self, pptx: Path, dpi: int, fontdir: Path):
        from lxml import etree
        from PIL import ImageFont

        self.z = zipfile.ZipFile(pptx)
        self.etree = etree
        self.dpi = dpi
        self.s = dpi / 25.4
        self.fontdir = fontdir
        self.ImageFont = ImageFont
        self.fonts = {}
        self.wrapped = []
        th = etree.fromstring(self.z.read("ppt/theme/theme1.xml"))
        self.scheme = {}
        for el in th.find(".//a:clrScheme", NS):
            v = el.find("a:srgbClr", NS)
            if v is None:
                v = el.find("a:sysClr", NS)
                val = v.get("lastClr") if v is not None else "000000"
            else:
                val = v.get("val")
            self.scheme[etree.QName(el).localname] = val
        alias = {"tx1": "dk1", "tx2": "dk2", "bg1": "lt1", "bg2": "lt2"}
        for k, v in alias.items():
            self.scheme[k] = self.scheme[v]
        pres = etree.fromstring(self.z.read("ppt/presentation.xml"))
        sz = pres.find("p:sldSz", NS)
        self.W, self.H = int(sz.get("cx")) / self.EMU_MM, int(sz.get("cy")) / self.EMU_MM
        rels = self.rels("ppt/presentation.xml")
        self.slides = ["ppt/" + rels[s.get("{%s}id" % NS["r"])] for s in pres.find("p:sldIdLst", NS)]

    # ---- package helpers
    def rels(self, part):
        d, f = part.rsplit("/", 1)
        rp = f"{d}/_rels/{f}.rels"
        out = {}
        if rp in self.z.namelist():
            for r in self.etree.fromstring(self.z.read(rp)):
                out[r.get("Id")] = r.get("Target") if r.get("TargetMode") == "External" else self.norm(d, r.get("Target"))
        return out

    @staticmethod
    def norm(base, target):
        parts = (base + "/" + target).split("/")
        out = []
        for p in parts:
            if p == "..":
                out.pop()
            elif p and p != ".":
                out.append(p)
        return "/".join(out)[len("ppt/"):] if "/".join(out).startswith("ppt/") else "/".join(out)

    def font(self, size_pt, bold, italic):
        key = (round(size_pt * 4), bool(bold), bool(italic))
        if key not in self.fonts:
            name = "Carlito-" + {(False, False): "Regular", (True, False): "Bold", (False, True): "Italic",
                                 (True, True): "BoldItalic"}[(bool(bold), bool(italic))] + ".ttf"
            px = max(1, int(round(size_pt / 72 * self.dpi)))
            try:
                self.fonts[key] = self.ImageFont.truetype(str(self.fontdir / name), px)
            except OSError:
                self.fonts[key] = self.ImageFont.truetype("DejaVuSans.ttf", max(1, int(px * 0.88)))
        return self.fonts[key]

    def color(self, parent, default=None):
        if parent is None:
            return default
        c = parent.find("a:srgbClr", NS)
        val = None
        if c is not None:
            val = c.get("val")
        else:
            c = parent.find("a:schemeClr", NS)
            if c is not None:
                val = self.scheme.get(c.get("val"), "000000")
        if val is None:
            return default
        alpha = 255
        a = c.find("a:alpha", NS)
        if a is not None:
            alpha = int(int(a.get("val")) / 100000 * 255)
        return (int(val[0:2], 16), int(val[2:4], 16), int(val[4:6], 16), alpha)

    # ---- drawing
    def render(self, idx: int, step=None):
        """Slide idx (0-based) after `step` clicks (None: final state)."""
        from PIL import Image

        part = self.slides[idx]
        x = self.etree.fromstring(self.z.read(part))
        srels = self.rels(part)
        layout_part = "ppt/" + [v for k, v in srels.items() if "slideLayout" in v][0]
        lx = self.etree.fromstring(self.z.read(layout_part))
        lrels = self.rels(layout_part)
        img = Image.new("RGBA", (int(round(self.W * self.s)), int(round(self.H * self.s))), "white")
        hidden = self.hidden_ids(x, step)
        lay_ph = {}
        for sp in lx.find(".//p:spTree", NS):
            ph = sp.find(".//p:nvPr/p:ph", NS)
            if ph is not None:
                lay_ph[(ph.get("type"), ph.get("idx"))] = sp
                lay_ph[(ph.get("type"), None)] = sp
                continue
            self.draw(img, sp, lrels, None, idx)
        for sp in x.find(".//p:spTree", NS):
            cid = sp.find(".//p:cNvPr", NS)
            if cid is not None and cid.get("id") in hidden:
                continue
            ph = sp.find(".//p:nvPr/p:ph", NS)
            base = None
            if ph is not None:
                base = lay_ph.get((ph.get("type"), ph.get("idx")))
                if base is None:
                    base = lay_ph.get((ph.get("type"), None))
            self.draw(img, sp, srels, base, idx)
        return img.convert("RGB")

    def hidden_ids(self, x, step):
        """Shapes with an entrance effect stay hidden until their click."""
        timing = x.find("p:timing", NS)
        if timing is None or step is None:
            return set()
        clicks = timing.findall(".//p:seq/p:cTn/p:childTnLst/p:par", NS)
        hidden = set()
        for k, c in enumerate(clicks):
            if step is not None and k < step:
                continue
            for t in c.iter("{%s}spTgt" % NS["p"]):
                hidden.add(t.get("spid"))
        return hidden

    def xfrm(self, sp, base):
        for src in (sp, base):
            if src is None:
                continue
            xf = src.find(".//a:xfrm", NS)
            if xf is None:
                xf = src.find(".//p:xfrm", NS)
            if xf is not None and xf.find("a:off", NS) is not None:
                off, ext = xf.find("a:off", NS), xf.find("a:ext", NS)
                return (int(off.get("x")) / self.EMU_MM, int(off.get("y")) / self.EMU_MM,
                        int(ext.get("cx")) / self.EMU_MM, int(ext.get("cy")) / self.EMU_MM,
                        xf.get("flipH") == "1", xf.get("flipV") == "1")
        return None

    def draw(self, img, sp, rels, base, idx):
        tag = self.etree.QName(sp).localname
        if tag not in ("sp", "pic", "cxnSp"):
            return
        geo = self.xfrm(sp, base)
        if geo is None:
            return
        if tag == "pic":
            self.picture(img, sp, rels, geo)
            return
        self.shape(img, sp, geo)
        tx = sp.find("p:txBody", NS)
        if tx is not None:
            self.text(img, tx, geo, base, idx)

    def picture(self, img, sp, rels, geo):
        from PIL import Image

        blip = sp.find(".//a:blip", NS)
        if blip is None:
            return
        target = rels.get(blip.get("{%s}embed" % NS["r"]))
        if not target:
            return
        src = Image.open(__import__("io").BytesIO(self.z.read("ppt/" + target))).convert("RGBA")
        sr = sp.find(".//a:srcRect", NS)
        if sr is not None:
            w, h = src.size
            l, t = int(sr.get("l", 0)) / 1e5, int(sr.get("t", 0)) / 1e5
            r, b = int(sr.get("r", 0)) / 1e5, int(sr.get("b", 0)) / 1e5
            src = src.crop((int(l * w), int(t * h), int(w - r * w), int(h - b * h)))
        x, y, w, h, _, _ = geo
        tw, th = max(1, int(round(w * self.s))), max(1, int(round(h * self.s)))
        img.alpha_composite(src.resize((tw, th), Image.LANCZOS), (int(round(x * self.s)), int(round(y * self.s))))

    def shape(self, img, sp, geo):
        from PIL import Image, ImageDraw

        spPr = sp.find("p:spPr", NS)
        if spPr is None:
            return
        x, y, w, h, fh, fv = geo
        fill = None
        if spPr.find("a:solidFill", NS) is not None:
            fill = self.color(spPr.find("a:solidFill", NS))
        ln = spPr.find("a:ln", NS)
        line, lw = None, 0
        if ln is not None and ln.find("a:noFill", NS) is None and ln.find("a:solidFill", NS) is not None:
            line = self.color(ln.find("a:solidFill", NS))
            lw = max(1, int(round(int(ln.get("w", 12700)) / 12700 / 72 * self.dpi)))
        if fill is None and line is None:
            return
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        X0, Y0, X1, Y1 = x * self.s, y * self.s, (x + w) * self.s, (y + h) * self.s
        prst = spPr.find("a:prstGeom", NS)
        cust = spPr.find("a:custGeom", NS)
        if prst is not None:
            kind = prst.get("prst")
            if kind == "rect":
                d.rectangle([X0, Y0, X1, Y1], fill=fill, outline=line, width=lw)
            elif kind == "roundRect":
                gd = prst.find(".//a:gd", NS)
                adj = int(gd.get("fmla").split()[1]) / 100000 if gd is not None else 0.16667
                d.rounded_rectangle([X0, Y0, X1, Y1], radius=adj * min(X1 - X0, Y1 - Y0), fill=fill, outline=line, width=lw)
            elif kind == "ellipse":
                d.ellipse([X0, Y0, X1, Y1], fill=fill, outline=line, width=lw)
            elif kind == "line":
                a, b = (X0, Y0), (X1, Y1)
                if fh != fv:
                    a, b = (X1, Y0), (X0, Y1)
                d.line([a, b], fill=line, width=lw)
        elif cust is not None:
            for path in cust.findall(".//a:path", NS):
                pw, ph = int(path.get("w", 1)) or 1, int(path.get("h", 1)) or 1
                sx, sy = (X1 - X0) / pw, (Y1 - Y0) / ph
                polys, cur = [], []
                for seg in path:
                    nm = self.etree.QName(seg).localname
                    pts = [(X0 + int(p.get("x")) * sx, Y0 + int(p.get("y")) * sy) for p in seg.findall("a:pt", NS)]
                    if nm == "moveTo":
                        if cur:
                            polys.append((cur, False))
                        cur = [pts[0]]
                    elif nm == "lnTo":
                        cur.append(pts[0])
                    elif nm == "cubicBezTo" and cur:
                        p0 = cur[-1]
                        for k in range(1, 9):
                            t = k / 8
                            m = 1 - t
                            cur.append((m ** 3 * p0[0] + 3 * m * m * t * pts[0][0] + 3 * m * t * t * pts[1][0] + t ** 3 * pts[2][0],
                                        m ** 3 * p0[1] + 3 * m * m * t * pts[0][1] + 3 * m * t * t * pts[1][1] + t ** 3 * pts[2][1]))
                    elif nm == "close":
                        if cur:
                            polys.append((cur, True))
                        cur = []
                if cur:
                    polys.append((cur, False))
                for pts, closed in polys:
                    if fill and len(pts) > 2:
                        d.polygon(pts, fill=fill)
                    if line and len(pts) > 1:
                        d.line(pts + ([pts[0]] if closed else []), fill=line, width=lw)
        img.alpha_composite(layer)

    def text(self, img, tx, geo, base, idx):
        from PIL import ImageDraw

        d = ImageDraw.Draw(img)
        x, y, w, h, _, _ = geo
        bp = tx.find("a:bodyPr", NS)
        ins = {k: int(bp.get(k, dflt)) / self.EMU_MM for k, dflt in (("lIns", 91440), ("tIns", 45720),
                                                                       ("rIns", 91440), ("bIns", 45720))}
        wrap = bp.get("wrap", "square") != "none"
        defaults = {}
        for src in (base.find("p:txBody/a:lstStyle", NS) if base is not None else None, tx.find("a:lstStyle", NS)):
            if src is None:
                continue
            l1 = src.find("a:lvl1pPr", NS)
            if l1 is None:
                continue
            dr = l1.find("a:defRPr", NS)
            if dr is not None:
                if dr.get("sz"):
                    defaults["sz"] = int(dr.get("sz")) / 100
                if dr.get("b"):
                    defaults["b"] = dr.get("b") == "1"
                c = self.color(dr.find("a:solidFill", NS))
                if c:
                    defaults["color"] = c
            ls = l1.find("a:lnSpc/a:spcPts", NS)
            if ls is not None:
                defaults["lnpt"] = int(ls.get("val")) / 100
            if l1.get("algn"):
                defaults["algn"] = l1.get("algn")
        box_w = w - ins["lIns"] - ins["rIns"]
        cy = y + ins["tIns"]
        first = True
        for p in tx.findall("a:p", NS):
            ppr = p.find("a:pPr", NS)
            algn = (ppr.get("algn") if ppr is not None and ppr.get("algn") else defaults.get("algn", "l"))
            runs = []
            for el in p:
                nm = self.etree.QName(el).localname
                if nm in ("r", "fld"):
                    rp = el.find("a:rPr", NS)
                    t = el.findtext("a:t", default="", namespaces=NS)
                    if nm == "fld" and el.get("type") == "slidenum":
                        t = str(idx + 1)
                    size = int(rp.get("sz")) / 100 if rp is not None and rp.get("sz") else defaults.get("sz", 18)
                    bold = (rp.get("b") == "1") if rp is not None and rp.get("b") else defaults.get("b", False)
                    ital = rp is not None and rp.get("i") == "1"
                    colr = self.color(rp.find("a:solidFill", NS) if rp is not None else None, defaults.get("color", (0, 0, 0, 255)))
                    sup = rp is not None and rp.get("baseline") not in (None, "0")
                    runs.append([t, size, bold, ital, colr, sup])
                elif nm == "br":
                    runs.append(["\n", None, None, None, None, None])
            sizes = [r[1] for r in runs if r[1]] or [defaults.get("sz", 18)]
            size0 = max(sizes)
            lnpt = None
            if ppr is not None and ppr.find("a:lnSpc/a:spcPts", NS) is not None:
                lnpt = int(ppr.find("a:lnSpc/a:spcPts", NS).get("val")) / 100
            elif ppr is not None and ppr.find("a:lnSpc/a:spcPct", NS) is not None:
                lnpt = int(ppr.find("a:lnSpc/a:spcPct", NS).get("val")) / 100000 * 1.22 * size0
            else:
                lnpt = defaults.get("lnpt", 1.22 * size0)
            L = lnpt * 25.4 / 72
            if not first and ppr is not None and ppr.find("a:spcBef/a:spcPts", NS) is not None:
                cy += int(ppr.find("a:spcBef/a:spcPts", NS).get("val")) / 100 * 25.4 / 72
            # lay out words greedily within the box width (this is where overflow shows up as extra lines)
            lines, cur, curw = [], [], 0.0
            for t, size, bold, ital, colr, sup in runs:
                if t == "\n":
                    lines.append(cur)
                    cur, curw = [], 0.0
                    continue
                f = self.font(size * (0.65 if sup else 1), bold, ital)
                for tok in re.split(r"(\s+)", t):
                    if not tok:
                        continue
                    tw = f.getlength(tok) / self.s
                    if wrap and cur and not tok.isspace() and curw + tw > box_w + 0.05:
                        lines.append(cur)
                        cur, curw = [], 0.0
                    if tok.isspace() and not cur:
                        continue
                    cur.append((tok, f, colr, sup, tw))
                    curw += tw
            lines.append(cur)
            breaks = sum(1 for r in runs if r[0] == "\n")
            if len(lines) > breaks + 1:
                words = "".join(r[0] for r in runs if r[0] != "\n")[:50]
                self.wrapped.append(f"slide {idx + 1}: text wraps in its box: {words!r}")
            for ln in lines:
                base_y = cy + L - 0.269 * size0 * 25.4 / 72
                lw = sum(t[4] for t in ln)
                if algn == "r":
                    xx = x + w - ins["rIns"] - lw
                elif algn == "ctr":
                    xx = x + ins["lIns"] + (box_w - lw) / 2
                else:
                    xx = x + ins["lIns"]
                for tok, f, colr, sup, tw in ln:
                    dy = -0.33 * f.size if sup else 0
                    for part in re.split(r"([\u2700-\u27bf]+)", tok):
                        if not part:
                            continue
                        ff = f
                        if re.match(r"[\u2700-\u27bf]", part):  # symbols Calibri lacks; PowerPoint uses a fallback font
                            ff = self.ImageFont.truetype("DejaVuSans.ttf", max(1, int(f.size * 0.85)))
                        d.text((xx * self.s, base_y * self.s + dy), part, font=ff, fill=colr, anchor="ls")
                        xx += ff.getlength(part) / self.s
                cy += L
            first = False


def builds_pdf_pages(pdf: Path):
    import fitz

    doc = fitz.open(pdf)
    labels = []
    for p in doc:
        lab = p.get_label() or ""
        m = re.fullmatch(r"<FEFF([0-9A-Fa-f]*)>", lab)
        if m:
            h = m.group(1)
            lab = "".join(chr(int(h[i:i + 4], 16)) for i in range(0, len(h), 4))
        labels.append(lab)
    groups, prev = [], None
    for i, lab in enumerate(labels):
        if lab != prev or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
        prev = lab
    return groups


def preview(pptx: Path, beamer: Path, out_dir: Path, dpi: int, fontdir: Path, pages=None, builds=None) -> int:
    from lxml import etree

    pv = PptxPreview(pptx, dpi, fontdir)
    left = render_pdf(beamer, dpi)
    sel = list(range(len(left))) if not pages else [p - 1 for p in pages]
    right = [pv.render(i) for i in sel]
    files = sheets([left[i] for i in sel], right, out_dir,
                   titles=("Beamer PDF", "preview from the .pptx XML (not a PowerPoint render)"),
                   labels=None, numbers=[i + 1 for i in sel])
    for i, im in zip(sel, right):
        im.save(out_dir / f"preview-{i + 1:02d}.png")
    print("\n".join(str(f) for f in files))
    for w in sorted(set(pv.wrapped)):
        print("WRAP:", w)
    if builds:
        # every click state of every animated slide against the matching overlay page of the step-by-step PDF
        import fitz
        groups = builds_pdf_pages(builds)
        bl = render_pdf(builds, dpi)
        L, R, N = [], [], []
        for i in sel:
            x = etree.fromstring(pv.z.read(pv.slides[i]))
            t = x.find("p:timing", NS)
            nclicks = len(t.findall(".//p:seq/p:cTn/p:childTnLst/p:par", NS)) if t is not None else 0
            if nclicks == 0 and len(groups[i]) == 1:
                continue
            for k, page in enumerate(groups[i]):
                L.append(bl[page])
                R.append(pv.render(i, step=k))
                N.append(f"{i + 1}.{k + 1}")
            if nclicks + 1 != len(groups[i]):
                print(f"slide {i + 1}: {nclicks} clicks but {len(groups[i])} overlay pages")
        if L:
            bfiles = sheets(L, R, out_dir / "builds", titles=("Beamer overlay", "pptx after this click (preview)"),
                            numbers=N)
            print("\n".join(str(f) for f in bfiles))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect")
    a.add_argument("pptx")
    b = sub.add_parser("compare")
    b.add_argument("beamer_pdf")
    b.add_argument("pptx_pdf")
    b.add_argument("--out", default=str(HERE / "out" / "qa"))
    b.add_argument("--dpi", type=int, default=110)
    c = sub.add_parser("preview")
    c.add_argument("pptx")
    c.add_argument("beamer_pdf")
    c.add_argument("--builds-pdf", default="", help="step-by-step PDF: also compare every click state")
    c.add_argument("--out", default=str(HERE / "out" / "preview"))
    c.add_argument("--dpi", type=int, default=110)
    c.add_argument("--fonts", default="", help="folder with Carlito-*.ttf (Calibri metrics)")
    c.add_argument("--pages", type=int, nargs="*")
    args = ap.parse_args()
    if args.cmd == "inspect":
        sys.exit(inspect(Path(args.pptx)))
    if args.cmd == "compare":
        sys.exit(compare(Path(args.beamer_pdf), Path(args.pptx_pdf), Path(args.out), args.dpi))
    sys.exit(preview(Path(args.pptx), Path(args.beamer_pdf), Path(args.out), args.dpi,
                     Path(args.fonts) if args.fonts else Path("/usr/share/fonts/truetype/crosextra"), args.pages,
                     Path(args.builds_pdf) if args.builds_pdf else None))


if __name__ == "__main__":
    main()
