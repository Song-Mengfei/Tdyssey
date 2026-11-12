#!/usr/bin/env node
/*
 * Builds the editable PowerPoint deck from out/deck.json (made by pdf_to_deck.py from the Beamer PDFs).
 *
 *   node build_deck.js [--deck out/deck.json] [--out out/tdyssey-slides.pptx] [--videos embed|drive]
 *
 * Structure: one theme (deck colours, one font family), three layouts (title, content, plain), the section
 * kicker and the frame title as placeholders, footer text, slide number and progress-bar track on the content
 * layout. Every text is a text box, every picture its own picture object, videos are embedded MP4 files with the
 * Beamer poster as cover (or, with --videos drive, posters linked to the public Drive copies for Google Slides).
 * Beamer overlay builds become "Appear on click" animations (added to the slide XML after pptxgenjs has written
 * the file, since pptxgenjs has no animation API).
 */
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");

// ------------------------------------------------------------------------------------------------ arguments
const HERE = __dirname;
const PRES = path.resolve(HERE, "..");
function arg(name, dflt) {
  const i = process.argv.indexOf("--" + name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}
const DECK = path.resolve(arg("deck", path.join(HERE, "out", "deck.json")));
const OUT = path.resolve(arg("out", path.join(HERE, "out", "tdyssey-slides.pptx")));
const VIDEOS = arg("videos", "embed");
if (!["embed", "drive"].includes(VIDEOS)) throw new Error("--videos must be embed or drive");
const deck = JSON.parse(fs.readFileSync(DECK, "utf8"));
const ROOT = path.dirname(DECK);

// ------------------------------------------------------------------------------------------------ theme
// Colours of slides/main.tex. The theme slots let the deck be recoloured in one place.
const THEME = {
  name: "Tdyssey",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2328", // tdink
    lt1: "FFFFFF",
    dk2: "5F6670", // tdgrey
    lt2: "F3F5F8", // tdpanel
    accent1: "2A78D6", // tdblue
    accent2: "EDA100", // tdgold
    accent3: "8A5A00", // tdgolddark
    accent4: "8F8D86", // tdorig
    accent5: "1F8A4C", // tdgood
    accent6: "B3261E", // tdbad
    hlink: "2A78D6",
    folHlink: "9E2A8A",
  },
};
const SCHEME = {
  "1F2328": "tx1", FFFFFF: "bg1", "5F6670": "tx2", F3F5F8: "bg2", "2A78D6": "accent1", EDA100: "accent2",
  "8A5A00": "accent3", "8F8D86": "accent4", "1F8A4C": "accent5", B3261E: "accent6",
};
const col = (hex) => (hex && SCHEME[hex.toUpperCase()]) || hex;

// Size scale (pt), taken from the Beamer preamble; the teammates' frames keep their own explicit sizes.
const SIZE = { kicker: 8, title: 16, footer: 7, body: 10, caption: 9, takeaway: 12, stat: 15, hero: 30 };

const MM = (v) => v / 25.4; // mm -> inches
const PT2MM = 25.4 / 72;
// Text box top for a first baseline (mm), font size and exact line spacing (pt): baseline = top + line - 0.26 size.
const boxTop = (baseline, size, line) => baseline - (line - 0.26 * size) * PT2MM;

// ------------------------------------------------------------------------------------------------ deck
const pres = new pptxgen();
pres.defineLayout({ name: "BEAMER_16_9", width: MM(deck.page_mm[0]), height: MM(deck.page_mm[1]) });
pres.layout = "BEAMER_16_9";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Tdyssey: one camera policy for every letter";
pres.subject = "Team 28 talk deck, generated from slides/main.tex";
pres.author = "Team 28: Liang Wenzhong, Song Mengfei, Su Shenyi";

const lay = deck.layout;
const W = deck.page_mm[0];

function textPlaceholder(name, type, spec, x, w) {
  const line = 1.2 * spec.size;
  return {
    placeholder: {
      options: {
        name, type, x: MM(x), y: MM(boxTop(spec.baseline, spec.size, line)), w: MM(w), h: MM(line * PT2MM + 0.6),
        fontSize: spec.size, bold: !!spec.bold, color: col(spec.color), margin: 0, valign: "top", align: "left",
        lineSpacing: line, fontFace: type === "title" ? "+mj-lt" : "+mn-lt",
      },
      text: "",
    },
  };
}

// content frame: section kicker, title, footer, slide number, progress-bar track
const footLine = 1.2 * lay.footer.size;
pres.defineSlideMaster({
  title: "Content",
  background: { color: "FFFFFF" },
  margin: 0,
  objects: [
    { rect: { x: 0, y: MM(lay.track.y), w: MM(W), h: MM(lay.track.h), fill: { color: lay.track.color }, line: { type: "none" } } },
    {
      text: {
        text: lay.footer.text,
        options: {
          x: MM(lay.footer.x), y: MM(boxTop(lay.footer.baseline, lay.footer.size, footLine)), w: MM(60),
          h: MM(footLine * PT2MM + 0.4), fontSize: lay.footer.size, color: lay.footer.color, margin: 0,
          valign: "top", lineSpacing: footLine, fontFace: "+mn-lt", isTextBox: true,
        },
      },
    },
    textPlaceholder("kicker", "body", lay.kicker, lay.kicker.x, 144),
    textPlaceholder("title", "title", lay.title, lay.title.x, 144),
  ],
  slideNumber: {
    x: MM(lay.number.x1 - 30), y: MM(boxTop(lay.number.baseline, lay.number.size, footLine)), w: MM(30),
    h: MM(footLine * PT2MM + 0.4), fontSize: lay.number.size, color: lay.number.color, align: "right", margin: 0,
    valign: "top", fontFace: "+mn-lt",
  },
});

// title and plain frames: the big title is a placeholder at its Beamer position
function plainLayout(name, slide) {
  const objects = [];
  const t = slide && slide.elements.find((e) => e.role === "title");
  if (t) {
    const size = t.paragraphs[0].runs[0].size;
    objects.push({
      placeholder: {
        options: {
          name: "title", type: "title", x: MM(t.x), y: MM(t.y), w: MM(t.w), h: MM(t.h), fontSize: size, bold: true,
          color: col(t.paragraphs[0].runs[0].color), margin: 0, valign: "top", align: t.align, lineSpacing: t.line_pt,
          fontFace: "+mj-lt",
        },
        text: "",
      },
    });
  }
  pres.defineSlideMaster({ title: name, background: { color: "FFFFFF" }, margin: 0, objects });
}
plainLayout("Title slide", deck.slides.find((s) => s.kind === "title"));
plainLayout("Plain", deck.slides.find((s) => s.kind === "plain"));

// ------------------------------------------------------------------------------------------------ slides
const post = { crops: {}, dashes: {}, anim: {}, numbers: {}, lineEnds: {} };
let section = null;
const usedSections = new Set();
const report = [];

function runsOf(el, opts) {
  const out = [];
  el.paragraphs.forEach((p, pi) => {
    p.runs.forEach((r, ri) => {
      const o = { fontSize: r.size, bold: !!r.bold, italic: !!r.italic, color: col(r.color) };
      if (r.script === "sup") o.superscript = true;
      if (r.script === "sub") o.subscript = true;
      if (ri === 0 && p.space_before_pt) o.paraSpaceBefore = p.space_before_pt;
      if (ri === p.runs.length - 1 && pi < el.paragraphs.length - 1) o.breakLine = true;
      if (opts && opts.link && ri === 0) o.hyperlink = opts.link;
      out.push({ text: r.text, options: o });
    });
  });
  return out;
}

function linkOf(el) {
  if (!el.link) return undefined;
  if (el.link.url) return { url: el.link.url };
  if (el.link.file) {
    // local clip next to the deck: relative to the output folder
    const abs = path.join(PRES, el.link.file);
    return { url: path.relative(path.dirname(OUT), abs).split(path.sep).join("/"), tooltip: el.link.file };
  }
  return undefined;
}

function dataUri(file) {
  const ext = path.extname(file).slice(1).toLowerCase();
  const mime = ext === "jpg" || ext === "jpeg" ? "image/jpeg" : "image/png";
  return `${mime};base64,` + fs.readFileSync(file).toString("base64");
}

for (const sl of deck.slides) {
  // sections follow the Beamer section kicker; a name used before gets a suffix (sections must be contiguous)
  let secTitle = sl.section || (sl.kind === "title" ? "Title" : section ? section.title : "Title");
  if (!section || secTitle !== section.base) {
    let name = secTitle;
    let k = 2;
    while (usedSections.has(name)) name = `${secTitle} (${k++})`;
    usedSections.add(name);
    pres.addSection({ title: name });
    section = { base: secTitle, title: name };
  }
  const master = sl.kind === "content" ? "Content" : sl.kind === "title" ? "Title slide" : "Plain";
  const slide = pres.addSlide({ masterName: master, sectionTitle: section.title });
  const names = [];
  const steps = {};
  const counters = {};
  const nameFor = (base, el) => {
    counters[base] = (counters[base] || 0) + 1;
    const n = `${base} ${counters[base]}`;
    if (el && el.step > 1) (steps[el.step] = steps[el.step] || []).push(n);
    names.push(n);
    return n;
  };

  if (sl.kind === "content") {
    if (sl.progress_w) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0, y: MM(lay.track.y), w: MM(sl.progress_w), h: MM(lay.track.h), fill: { color: col(lay.progress_color) },
        line: { type: "none" }, objectName: "Progress bar",
      });
    }
    if (sl.section) slide.addText(sl.section, { placeholder: "kicker" });
    if (sl.title) slide.addText(sl.title, { placeholder: "title" });
    post.numbers[sl.page] = sl.appendix ? { static: sl.number_text } : { total: deck.main_total };
  }

  for (const el of sl.elements) {
    if (el.type === "text") {
      if (el.role === "title" && sl.kind !== "content") {
        slide.addText(runsOf(el), { placeholder: "title" });
        continue;
      }
      const name = nameFor("Text", el);
      slide.addText(runsOf(el, { link: linkOf(el) }), {
        x: MM(el.x), y: MM(el.y), w: MM(el.w), h: MM(el.h), margin: 0, valign: "top", align: el.align,
        lineSpacing: el.line_pt, fit: "none", wrap: true, isTextBox: true, objectName: name,
      });
    } else if (el.type === "image" || el.type === "picture") {
      const base = el.type === "picture" ? (el.role === "graph" ? "Graph" : "Diagram") : "Picture";
      const name = nameFor(el.type === "picture" && el.name ? `${base} step` : base, el);
      const opts = { path: path.join(ROOT, el.path), x: MM(el.x), y: MM(el.y), w: MM(el.w), h: MM(el.h), objectName: name };
      opts.altText = el.type === "picture" ? `${base} from the Beamer deck, page ${sl.page}` : "";
      const ln = linkOf(el);
      if (ln) opts.hyperlink = ln;
      slide.addImage(opts);
      if (el.crop) post.crops[`${sl.page}/${name}`] = el.crop;
    } else if (el.type === "video") {
      const poster = path.join(ROOT, el.cover);
      if (VIDEOS === "embed") {
        const name = nameFor(`Video ${el.video_id}`, el);
        slide.addMedia({
          type: "video", path: path.join(PRES, el.mp4), cover: dataUri(poster), x: MM(el.x), y: MM(el.y), w: MM(el.w),
          h: MM(el.h), objectName: name,
        });
        if (el.crop) post.crops[`${sl.page}/${name}`] = el.crop;
      } else {
        // Google Slides drops embedded MP4 files: link the poster to the public copy, or to the local clip
        const name = nameFor(`Video poster ${el.video_id}`, el);
        const link = el.url
          ? { url: el.url, tooltip: "Play the video (Google Drive)" }
          : { url: path.relative(path.dirname(OUT), path.join(PRES, el.mp4)).split(path.sep).join("/"), tooltip: el.mp4 };
        slide.addImage({
          path: poster, x: MM(el.x), y: MM(el.y), w: MM(el.w), h: MM(el.h), objectName: name,
          hyperlink: link, altText: `Video ${el.mp4}`,
        });
        if (el.crop) post.crops[`${sl.page}/${name}`] = el.crop;
      }
    } else if (el.type === "shape") {
      addShape(slide, el, nameFor("Shape", el), sl.page);
    }
  }
  if (Object.keys(steps).length) post.anim[sl.page] = steps;
  report.push(`${sl.page}`);
}

function lineOpts(el) {
  if (!el.line) return { type: "none" };
  const o = { color: col(el.line), width: el.line_w };
  if (el.line_alpha !== undefined) o.transparency = Math.round((1 - el.line_alpha) * 100);
  if (el.dash) o.dashType = "dash";
  return o;
}
function fillOpts(el) {
  if (!el.fill) return { type: "none" };
  const o = { color: col(el.fill) };
  if (el.fill_alpha !== undefined) o.transparency = Math.round((1 - el.fill_alpha) * 100);
  return o;
}

function addShape(slide, el, name, page) {
  const base = { x: MM(el.x), y: MM(el.y), w: MM(Math.max(el.w, 0.01)), h: MM(Math.max(el.h, 0.01)), objectName: name };
  if (el.dash) post.dashes[`${page}/${name}`] = { dash: el.dash, width: el.line_w };
  if (el.geom === "rect") {
    slide.addShape(pres.shapes.RECTANGLE, { ...base, fill: fillOpts(el), line: lineOpts(el) });
  } else if (el.geom === "roundRect") {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { ...base, fill: fillOpts(el), line: lineOpts(el), rectRadius: MM(el.radius) });
  } else if (el.geom === "ellipse") {
    slide.addShape(pres.shapes.OVAL, { ...base, fill: fillOpts(el), line: lineOpts(el) });
  } else if (el.geom === "line") {
    const x = Math.min(el.x1, el.x2), y = Math.min(el.y1, el.y2);
    slide.addShape(pres.shapes.LINE, {
      x: MM(x), y: MM(y), w: MM(Math.abs(el.x2 - el.x1)), h: MM(Math.abs(el.y2 - el.y1)), objectName: name,
      line: lineOpts(el), flipH: el.x2 < el.x1 !== el.y2 < el.y1 && el.x2 !== el.x1 && el.y2 !== el.y1 ? true : false,
    });
  } else if (el.geom === "path") {
    const pts = [];
    for (const seg of el.path) {
      const [op, ps] = Object.entries(seg)[0];
      if (op === "m") pts.push({ x: MM(ps[0][0]), y: MM(ps[0][1]), moveTo: true });
      else if (op === "l") pts.push({ x: MM(ps[0][0]), y: MM(ps[0][1]) });
      else if (op === "c")
        pts.push({ x: MM(ps[2][0]), y: MM(ps[2][1]), curve: { type: "cubic", x1: MM(ps[0][0]), y1: MM(ps[0][1]), x2: MM(ps[1][0]), y2: MM(ps[1][1]) } });
      else if (op === "z") pts.push({ close: true });
    }
    slide.addShape(pres.shapes.CUSTOM_GEOMETRY, { ...base, points: pts, fill: fillOpts(el), line: lineOpts(el) });
  }
}

// ------------------------------------------------------------------------------------------------ write + post-process
(async () => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  const buf = await pres.write({ outputType: "nodebuffer" });
  const zip = await JSZip.loadAsync(buf);
  await applyTheme(zip);
  const slideFiles = Object.keys(zip.files).filter((f) => /^ppt\/slides\/slide\d+\.xml$/.test(f));
  const counts = { animated: 0, effects: 0 };
  for (const f of slideFiles) {
    const page = Number(f.match(/slide(\d+)\.xml/)[1]);
    let xml = await zip.file(f).async("string");
    xml = onePPr(xml);
    xml = placeholderStyle(xml, false);
    xml = renumberIds(xml);
    xml = applyCrops(xml, page);
    xml = applyDashes(xml, page);
    xml = slideNumber(xml, page);
    if (post.anim[page]) {
      const r = addTiming(xml, post.anim[page]);
      xml = r.xml;
      counts.animated += 1;
      counts.effects += r.effects;
    }
    zip.file(f, xml);
  }
  // the same number suffix on the layout, so a new slide shows "n / total" too
  for (const f of Object.keys(zip.files).filter((f) => /^ppt\/slideLayouts\/slideLayout\d+\.xml$/.test(f))) {
    let xml = await zip.file(f).async("string");
    xml = onePPr(xml);
    xml = placeholderStyle(xml, true);
    xml = xml.replace(/(<a:fld [^>]*type="slidenum"[^>]*>[\s\S]*?<\/a:fld>)/, `$1${numberSuffixRun(deck.main_total)}`);
    zip.file(f, xml);
  }
  // copy files only: folder entries in the ZIP upset some cloud services
  const clean = new JSZip();
  for (const [name, entry] of Object.entries(zip.files)) {
    if (!entry.dir) clean.file(name, await entry.async("nodebuffer"), { date: new Date(Date.UTC(2026, 0, 1)), createFolders: false });
  }
  const out = await clean.generateAsync({ type: "nodebuffer", compression: "DEFLATE", compressionOptions: { level: 6 } });
  fs.writeFileSync(OUT, out);
  console.log(`wrote ${OUT}: ${deck.slides.length} slides, ${counts.animated} with click builds (${counts.effects} appear effects), videos: ${VIDEOS}`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

async function applyTheme(zip) {
  const f = "ppt/theme/theme1.xml";
  let xml = await zip.file(f).async("string");
  const c = THEME.colors;
  const slot = (n, v) => `<a:${n}><a:srgbClr val="${v}"/></a:${n}>`;
  const scheme =
    `<a:clrScheme name="${THEME.name}">` +
    slot("dk1", c.dk1) + slot("lt1", c.lt1) + slot("dk2", c.dk2) + slot("lt2", c.lt2) +
    ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"].map((k) => slot(k, c[k])).join("") +
    `</a:clrScheme>`;
  xml = xml.replace(/<a:clrScheme [\s\S]*?<\/a:clrScheme>/, scheme);
  xml = xml.replace(/<a:theme ([^>]*)name="[^"]*"/, `<a:theme $1name="${THEME.name}"`);
  xml = xml.replace(/(<a:majorFont>\s*<a:latin typeface=")[^"]*"/, `$1${THEME.headFontFace}"`);
  xml = xml.replace(/(<a:minorFont>\s*<a:latin typeface=")[^"]*"/, `$1${THEME.bodyFontFace}"`);
  zip.file(f, xml);
}

// pptxgenjs 4.0.1 writes a paragraph-properties element before every run of a paragraph; the schema allows one,
// first. Keep the first and drop the repeats (they carry the same paragraph settings).
function onePPr(xml) {
  return xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, body) => {
    let seen = false;
    const out = body.replace(/<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g, (pp) => {
      if (seen) return "";
      seen = true;
      return pp;
    });
    return `<a:p>${out}</a:p>`;
  });
}

// Placeholders: readable names, no insets on the layout, and on slides no run formatting, so that the kicker and
// the title take size, weight and colour from the layout only (restyle all titles by editing the layout).
function placeholderStyle(xml, isLayout) {
  return replaceBlocks(xml, (b) => {
    const ph = b.xml.match(/<p:ph\b([^>]*)\/>/);
    if (!ph) return b.xml;
    const attrs = ph[1];
    const type = (attrs.match(/type="([^"]+)"/) || [])[1];
    let name = null;
    if (type === "title") name = "Title";
    else if (type === "body") name = "Section kicker";
    else if (type === "sldNum") name = "Slide number";
    let x = b.xml;
    if (name) x = x.replace(/(<p:cNvPr id="\d+" name=")[^"]*"/, `$1${name}"`);
    x = x.replace(/<a:bodyPr wrap="square" rtlCol="0"><\/a:bodyPr>/, '<a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="t"></a:bodyPr>');
    if (!isLayout && (type === "title" || type === "body")) {
      x = x.replace(/<a:xfrm>[\s\S]*?<\/a:xfrm>/, ""); // position from the layout
      x = x.replace(/<a:rPr ([^>]*?)>[\s\S]*?<\/a:rPr>/g, (m, a) => `<a:rPr ${a.replace(/\s(sz|b)="[^"]*"/g, "")}/>`);
      x = x.replace(/<a:endParaRPr ([^>]*?)\s*\/>/g, (m, a) => `<a:endParaRPr ${a.replace(/\s(sz|b)="[^"]*"/g, "")}/>`);
    }
    return x;
  });
}

// pptxgenjs can give a video the same id as another shape; ids must be unique per slide for animations
// (video shapes keep the id pptxgenjs gave them; every other shape gets a fresh id that collides with none)
function renumberIds(xml) {
  const keep = new Set();
  const re = /<p:pic>[\s\S]*?<\/p:pic>/g;
  let m;
  while ((m = re.exec(xml))) {
    if (/<a:videoFile /.test(m[0])) keep.add(Number(m[0].match(/<p:cNvPr id="(\d+)"/)[1]));
  }
  let n = 1;
  return xml.replace(/<p:pic>[\s\S]*?<\/p:pic>|<p:cNvPr id="\d+"/g, (t) => {
    if (t.startsWith("<p:pic>")) {
      if (/<a:videoFile /.test(t)) return t;
      return t.replace(/<p:cNvPr id="\d+"/, () => {
        do n++; while (keep.has(n));
        return `<p:cNvPr id="${n}"`;
      });
    }
    do n++; while (keep.has(n));
    return `<p:cNvPr id="${n}"`;
  });
}

function shapeBlocks(xml) {
  // every top-level shape element with its name and id
  const out = [];
  const re = /<(p:sp|p:pic|p:cxnSp|p:graphicFrame)>[\s\S]*?<\/\1>/g;
  let m;
  while ((m = re.exec(xml))) {
    const nm = m[0].match(/<p:cNvPr id="(\d+)" name="([^"]*)"/);
    if (nm) out.push({ start: m.index, end: m.index + m[0].length, id: Number(nm[1]), name: unxml(nm[2]), xml: m[0], tag: m[1] });
  }
  return out;
}
const unxml = (s) => s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

function replaceBlocks(xml, fn) {
  const blocks = shapeBlocks(xml);
  let out = "", last = 0;
  for (const b of blocks) {
    out += xml.slice(last, b.start) + fn(b);
    last = b.end;
  }
  return out + xml.slice(last);
}

function applyCrops(xml, page) {
  return replaceBlocks(xml, (b) => {
    const c = post.crops[`${page}/${b.name}`];
    if (!c) return b.xml;
    const v = (x) => Math.round(x * 100000);
    const src = `<a:srcRect l="${v(c.l)}" t="${v(c.t)}" r="${v(c.r)}" b="${v(c.b)}"/>`;
    // the shape keeps the full poster box; the crop shows only the clipped part, like the Beamer clip
    let x = b.xml.replace(/(<p:blipFill>\s*<a:blip [^>]*\/>)/, `$1${src}`);
    // position: shrink the frame to the visible part
    const off = x.match(/<a:off x="(\d+)" y="(\d+)"\/>\s*<a:ext cx="(\d+)" cy="(\d+)"\/>/);
    if (off) {
      const [X, Y, CX, CY] = off.slice(1).map(Number);
      const nx = Math.round(X + c.l * CX), ny = Math.round(Y + c.t * CY);
      const ncx = Math.round(CX * (1 - c.l - c.r)), ncy = Math.round(CY * (1 - c.t - c.b));
      x = x.replace(off[0], `<a:off x="${nx}" y="${ny}"/><a:ext cx="${ncx}" cy="${ncy}"/>`);
    }
    return x;
  });
}

function applyDashes(xml, page) {
  return replaceBlocks(xml, (b) => {
    const d = post.dashes[`${page}/${b.name}`];
    if (!d) return b.xml;
    const w = d.width || 1;
    let ds = "";
    for (let i = 0; i < d.dash.length; i += 2) {
      const on = d.dash[i], off = d.dash[i + 1] !== undefined ? d.dash[i + 1] : d.dash[i];
      ds += `<a:ds d="${Math.max(1, Math.round((on / w) * 100000))}" sp="${Math.max(1, Math.round((off / w) * 100000))}"/>`;
    }
    return b.xml.replace(/<a:prstDash val="[^"]*"\/>/, `<a:custDash>${ds}</a:custDash>`);
  });
}

function numberSuffixRun(total) {
  return `<a:r><a:rPr lang="en-US" dirty="0"/><a:t> / ${total}</a:t></a:r>`;
}

// main slides: "<slide number field> / 42"; backup slides restart at 1 in Beamer, so they carry static text
function slideNumber(xml, page) {
  const spec = post.numbers[page];
  if (!spec) return xml;
  return replaceBlocks(xml, (b) => {
    if (!/type="sldNum"/.test(b.xml)) return b.xml;
    if (spec.static) {
      const txt = spec.static.replace(/\s*\/\s*/, " / ");
      return b.xml.replace(/<a:fld [^>]*type="slidenum"[^>]*>[\s\S]*?<\/a:fld>/, `<a:r><a:rPr lang="en-US" dirty="0"/><a:t>${txt}</a:t></a:r>`);
    }
    return b.xml.replace(/(<a:fld [^>]*type="slidenum"[^>]*>[\s\S]*?<\/a:fld>)/, `$1${numberSuffixRun(spec.total)}`);
  });
}

// ------------------------------------------------------------------------------------------------ animations
// One click per Beamer overlay step after the first; everything that appears at that step appears together.
function addTiming(xml, steps) {
  const blocks = shapeBlocks(xml);
  const idOf = {};
  for (const b of blocks) idOf[b.name] = b;
  let id = 2;
  const clicks = [];
  const bld = [];
  let effects = 0;
  for (const k of Object.keys(steps).map(Number).sort((a, b) => a - b)) {
    const targets = steps[k].map((n) => idOf[n]).filter(Boolean);
    if (!targets.length) continue;
    const clickId = ++id, groupId = ++id;
    let inner = "";
    targets.forEach((t, i) => {
      const effId = ++id, setId = ++id;
      effects += 1;
      inner +=
        `<p:par><p:cTn id="${effId}" presetID="1" presetClass="entr" presetSubtype="0" fill="hold"` +
        `${t.tag === "p:sp" ? ' grpId="0"' : ""} nodeType="${i === 0 ? "clickEffect" : "withEffect"}">` +
        `<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>` +
        `<p:set><p:cBhvr><p:cTn id="${setId}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>` +
        `<p:tgtEl><p:spTgt spid="${t.id}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>` +
        `</p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set></p:childTnLst></p:cTn></p:par>`;
      if (t.tag === "p:sp") bld.push(`<p:bldP spid="${t.id}" grpId="0"${/<p:txBody>/.test(t.xml) ? "" : ' animBg="1"'}/>`);
    });
    clicks.push(
      `<p:par><p:cTn id="${clickId}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst>` +
        `<p:par><p:cTn id="${groupId}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>` +
        inner +
        `</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>`
    );
  }
  if (!clicks.length) return { xml, effects: 0 };
  const timing =
    `<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>` +
    `<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>` +
    clicks.join("") +
    `</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>` +
    `<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>` +
    `</p:childTnLst></p:cTn></p:par></p:tnLst>` +
    (bld.length ? `<p:bldLst>${bld.join("")}</p:bldLst>` : "") +
    `</p:timing>`;
  // <p:timing> follows <p:clrMapOvr> and precedes <p:extLst> in <p:sld>
  if (/<p:timing>/.test(xml)) throw new Error("slide already has timing");
  if (/<\/p:clrMapOvr>/.test(xml)) xml = xml.replace(/<\/p:clrMapOvr>/, `</p:clrMapOvr>${timing}`);
  else xml = xml.replace(/<\/p:cSld>/, `</p:cSld>${timing}`);
  return { xml, effects };
}
