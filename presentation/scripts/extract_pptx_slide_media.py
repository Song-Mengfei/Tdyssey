"""Extract the reviewed 17 embedded videos and original posters from the deck.

Uses only the Python standard library. The source PPTX hash and fixed metadata
come from data/pptx_slide_media.json. No decoding, recompression, simulation or
model calls occur. Run from any directory:

    python -B presentation/scripts/extract_pptx_slide_media.py --pptx /path/deck.pptx

Add --verify-only to check the source and existing extracted files without writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def poster_dimensions(data: bytes) -> tuple[int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            while i < len(data) and data[i] == 0xFF:
                i += 1
            marker = data[i]
            i += 1
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            length = int.from_bytes(data[i:i + 2], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height, width = struct.unpack(">HH", data[i + 3:i + 7])
                return width, height
            i += length
    raise ValueError("Unsupported original poster format; expected PNG or JPEG")


def safe_output(relative: str, prefix: str) -> Path:
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not relative.startswith(prefix):
        raise ValueError(f"Unexpected output path: {relative}")
    target = ROOT.joinpath(*p.parts).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output escapes presentation/: {relative}")
    return target


def binding_text(manifest: dict) -> str:
    def number(v):
        return f"{v:.8f}".rstrip("0").rstrip(".") or "0"
    lines = ["% Generated from data/pptx_slide_media.json; coordinates in160x90mm.",
             "% Root defines \\tdplacevideo{x}{y}{width}{height}{poster}{mp4}{DriveURL}.",
             "% Video bytes and original PPTX posters are preserved without re-encoding."]
    for v in manifest["videos"]:
        b = v["bbox_160x90_mm"]
        args = [number(b[k]) for k in ("x", "y", "width", "height")]
        args += [v["poster"]["path"], v["media"]["path"], v["drive_url"]]
        lines.append("\\expandafter\\def\\csname tdclip@" + v["page_id"] + "\\endcsname{\\tdplacevideo" + "".join("{" + a + "}" for a in args) + "}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "data/pptx_slide_media.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_bytes = args.pptx.read_bytes()
    if sha256(source_bytes) != manifest["source_pptx"]["sha256"]:
        raise ValueError("PPTX SHA256 does not match the reviewed source manifest")
    planned = []
    with zipfile.ZipFile(args.pptx) as archive:
        media_parts = [n for n in archive.namelist() if n.startswith("ppt/media/") and n.lower().endswith(".mp4")]
        assert len(media_parts) == manifest["video_count"] == 17
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        size = presentation.find("p:sldSz", NS)
        page_width, page_height = int(size.get("cx")), int(size.get("cy"))
        assert [page_width, page_height] == manifest["source_pptx"]["page_size_emu"]
        assert len(presentation.findall("p:sldIdLst/p:sldId", NS)) == manifest["slide_count"] == 62
        for video in manifest["videos"]:
            slide = ET.fromstring(archive.read(video["pptx_slide_part"]))
            matches = [pic for pic in slide.findall(".//p:pic", NS) if pic.find("p:nvPicPr/p:cNvPr", NS).get("id") == video["pptx_shape_id"]]
            assert len(matches) == 1
            transform = matches[0].find("p:spPr/a:xfrm", NS)
            off, ext = transform.find("a:off", NS), transform.find("a:ext", NS)
            bbox = {"x": int(off.get("x")) / page_width * 160, "y": int(off.get("y")) / page_height * 90,
                    "width": int(ext.get("cx")) / page_width * 160, "height": int(ext.get("cy")) / page_height * 90}
            assert all(abs(bbox[k] - video["bbox_160x90_mm"][k]) < 1e-7 for k in bbox)
            for kind, prefix in (("media", "videos/slides/"), ("poster", "videos/posters/slides/")):
                item = video[kind]
                data = archive.read(item["pptx_part"])
                assert sha256(data) == item["sha256"] and len(data) == item["bytes"], video["page_id"]
                if kind == "poster":
                    assert list(poster_dimensions(data)) == item["dimensions_px"]
                target = safe_output(item["path"], prefix)
                if target.exists() and sha256(target.read_bytes()) != item["sha256"]:
                    raise ValueError(f"Existing file differs; preserve and inspect it first: {item['path']}")
                if args.verify_only:
                    assert target.exists(), item["path"]
                planned.append((target, data))
    expected_bindings = binding_text(manifest)
    bindings = ROOT / "data/slide_video_bindings.tex"
    if args.verify_only:
        assert bindings.read_text(encoding="utf-8") == expected_bindings
    else:
        for target, data in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
        bindings.write_text(expected_bindings, encoding="utf-8", newline="\n")
    print(json.dumps({"verified_videos": len(manifest["videos"]), "mp4_bytes": sum(v["media"]["bytes"] for v in manifest["videos"]),
                      "poster_bytes": sum(v["poster"]["bytes"] for v in manifest["videos"]), "verify_only": args.verify_only,
                      "source_sha256": manifest["source_pptx"]["sha256"]}))


if __name__ == "__main__":
    main()
