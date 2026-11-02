"""RGB-only diagnostics and bounded display-only compatibility case plates.

Preview clipping/nearest-neighbor thumbnails/RGB332 quantization are never
returned to the predictor, measurement, or task-score paths.
"""

import json
from pathlib import Path
import struct
import zlib

import numpy as np

from .wm_compat_transform import transform_for_arm


CATEGORIES = ("rigid_rejection", "pusher_rejection", "B_valid_A_invalid", "TF_valid_AR_invalid")
PNG_LIMIT, MANIFEST_LIMIT = 96 * 1024, 32 * 1024
ANCHORS = tuple(f"T{i}" for i in range(12, 16)) + tuple(f"L{i:02d}" for i in range(12))
CANDIDATES, SEEDS = ("hold", "toward", "side", "away"), (5478, 5479, 5480)


def _frame(value, *, real=False):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32
            or value.shape != (3, 96, 96) or not np.isfinite(value).all()
            or (real and (value.min() < 0 or value.max() > 1))):
        raise ValueError("Expected finite float32 CHW 96x96 RGB")
    return value


def appearance_metrics(raw, branch, arm, horizon):
    """Within-arm raw RGB MSE and last-real-frame persistence; no true pose."""
    _frame(raw)
    if arm not in ("A", "B") or type(horizon) is not int or not 1 <= horizon <= 4:
        raise ValueError("Undeclared arm or horizon")
    actual = transform_for_arm(_frame(branch.actual[horizon-1], real=True), arm)[0]
    latest = transform_for_arm(_frame(branch.history[-1], real=True), arm)[0]
    # Float64 differences keep numeric diagnostics finite without altering raw.
    reference = actual.astype(np.float64)
    return dict(forecast_rgb_mse=float(np.square(raw.astype(np.float64)-reference).mean()),
                persistence_rgb_mse=float(np.square(latest.astype(np.float64)-reference).mean()))


def _validity(row):
    measurement = row.get("measurement") or {}
    state, shape, pusher = (measurement.get(name) or {} for name in ("state", "shape", "pusher"))
    return dict(object_shape=state.get("valid") is True and shape.get("valid") is True,
                shape=shape.get("valid") is True, pusher=pusher.get("valid") is True,
                state_reason=state.get("reason"), shape_reason=shape.get("reason"),
                pusher_reason=pusher.get("reason"))


def _key(anchor, candidate, seed, mode, arm, horizon):
    return f"{anchor}/{candidate}/{seed}/{mode}/{arm}/H{horizon}"


def _read_raw(row, root):
    expected = "predictions/" + row["key"].replace("/", "-") + ".npy"
    if row.get("raw_path") != expected:
        raise ValueError("Raw case path differs from its fixed key")
    path = root / expected
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Case paths cannot use links or junctions")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Raw case is outside the budget output root")
    with path.open("rb") as stream:
        value = np.load(stream, allow_pickle=False)
    return _frame(value)


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)


FONT = {
    "0": (14,17,19,21,25,17,14), "1": (4,12,4,4,4,4,14),
    "2": (14,17,1,2,4,8,31), "3": (30,1,1,14,1,1,30),
    "4": (2,6,10,18,31,2,2), "5": (31,16,16,30,1,1,30),
    "6": (14,16,16,30,17,17,14), "7": (31,1,2,4,8,8,8),
    "8": (14,17,17,14,17,17,14), "9": (14,17,17,15,1,1,14),
    "A": (14,17,17,31,17,17,17), "B": (30,17,17,30,17,17,30),
    "O": (14,17,17,17,17,17,14), "T": (31,4,4,4,4,4,4),
    "R": (30,17,17,30,20,18,17), "F": (31,16,16,30,16,16,16),
}


def _label(canvas, text, x, y):
    for character in text:
        for dy, bits in enumerate(FONT[character]):
            for dx in range(5):
                if bits & (1 << (4-dx)):
                    canvas[y+dy, x+dx] = 255
        x += 6


def _png(panels):
    # Even with incompressible indices, 24 rows occupy <85KiB including PNG
    # palette/scanline/zlib overhead; the explicit serialized bound is checked.
    canvas = np.zeros((36*len(panels), 96), dtype=np.uint8)
    for index, (frames, labels) in enumerate(panels):
        top = index*36
        for panel, (frame, label) in enumerate(zip(frames, labels)):
            _label(canvas, f"{index+1:02d}O" if panel == 0 else label, panel*24+2, top+2)
            rgb = np.rint(np.clip(frame[:, ::4, ::4], 0, 1)*255).astype(np.uint8)
            canvas[top+12:top+36, panel*24:panel*24+24] = (
                ((rgb[0] >> 5) << 5) | ((rgb[1] >> 5) << 2) | (rgb[2] >> 6))
    palette = bytes(channel for index in range(256) for channel in
                    (((index >> 5) & 7)*255//7, ((index >> 2) & 7)*255//7, (index & 3)*255//3))
    scanlines = b"".join(b"\x00"+row.tobytes() for row in canvas)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 96, len(canvas), 8, 3, 0, 0, 0))
            + _chunk(b"PLTE", palette) + _chunk(b"IDAT", zlib.compress(scanlines, 9)) + _chunk(b"IEND", b""))


def export_case_plates(rows, branches, sink):
    """Four fixed categories, first six canonical matched cases per category."""
    root = Path(sink.budget.output_dir).resolve()
    lookup = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        identity = (row.get("anchor"), row.get("candidate"), row.get("seed"),
                    row.get("mode"), row.get("arm"), row.get("horizon"))
        anchor, candidate, seed, mode, arm, horizon = identity
        if (anchor not in ANCHORS or candidate not in CANDIDATES or type(seed) is not int or seed not in SEEDS
                or arm not in ("A", "B") or mode not in ("AR", "TF") or type(horizon) is not int
                or horizon not in ((1,2,3,4) if mode == "AR" else (2,4))
                or row.get("key") != _key(anchor,candidate,seed,mode,arm,horizon)):
            raise ValueError("Malformed completed case identity")
        if identity in lookup:
            raise ValueError("Duplicate completed case identity")
        lookup[identity] = row
    branch_lookup = {(branch.anchor, branch.candidate): branch for branch in branches}
    selected = {category: [] for category in CATEGORIES}
    for anchor in ANCHORS:
        for candidate in CANDIDATES:
            if (anchor, candidate) not in branch_lookup:
                continue
            for seed in SEEDS:
                for horizon in (1,2,3,4):
                    a, b = (lookup.get((anchor,candidate,seed,"AR",arm,horizon)) for arm in ("A","B"))
                    if a is not None and b is not None:
                        av, bv = _validity(a), _validity(b)
                        conditions = (not av["shape"] or not bv["shape"], not av["pusher"] or not bv["pusher"],
                                      bv["object_shape"] and not av["object_shape"])
                        for category, condition in zip(CATEGORIES[:3], conditions):
                            if condition and len(selected[category]) < 6:
                                selected[category].append((a,b,"A AR","B AR"))
                    if horizon in (2,4):
                        for arm in ("A","B"):
                            ar, tf = (lookup.get((anchor,candidate,seed,mode,arm,horizon)) for mode in ("AR","TF"))
                            if (ar is not None and tf is not None and not _validity(ar)["object_shape"]
                                    and _validity(tf)["object_shape"] and len(selected[CATEGORIES[3]]) < 6):
                                selected[CATEGORIES[3]].append((ar,tf,arm+" AR",arm+" TF"))
    cases, panels = [], []
    for category in CATEGORIES:
        for first, second, first_label, second_label in selected[category]:
            branch = branch_lookup[first["anchor"],first["candidate"]]
            actual = _frame(branch.actual[first["horizon"]-1], real=True)
            transformed = transform_for_arm(actual,"B")[0]
            frames = (actual, transformed, _read_raw(first,root), _read_raw(second,root))
            panels.append((frames, ("O","T","AR" if category==CATEGORIES[3] else "A",
                                     "TF" if category==CATEGORIES[3] else "B")))
            cases.append(dict(row=len(cases)+1,category=category,keys=[first["key"],second["key"]],
                anchor=first["anchor"],candidate=first["candidate"],seed=first["seed"],horizon=first["horizon"],
                panels=["original actual","T1 actual",first_label,second_label],
                model_validity=[_validity(first),_validity(second)],
                physical_truth_separate=dict(truth=first.get("truth"),contact_by_h4=first.get("contact_by_h4"))))
    manifest = dict(kind="wm_compat_display_only_cases.v1",categories=list(CATEGORIES),per_category_limit=6,
        ordering="T12..T15,L00..L11; hold,toward,side,away; seed; horizon; arm A,B for TF matches",
        selected_cases=len(cases),cases=cases,display_only=True,feedback_or_measurement_use=False,
        preview=dict(thumbnail=[24,24],method="fixed stride4 nearest pixel; display clip[0,1]; round uint8 RGB332",
                     panel_width=24,label_band_height=12,row_height=36,physical_labels_are_separate=True))
    encoded = (json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    if len(encoded) > MANIFEST_LIMIT:
        raise ValueError("Case manifest exceeds its fixed 32KiB bound")
    image_name = None
    if panels:
        payload = _png(panels)
        if len(payload) > PNG_LIMIT:
            raise ValueError("Display-only PNG exceeds its fixed 96KiB bound")
        image_name = "case-plates.png"
        sink.budget.write_chunk(image_name,payload,max_bytes=PNG_LIMIT,final=True)
    sink.record("case-plates.json",manifest,maximum=MANIFEST_LIMIT,final=True)
    return dict(image=image_name,manifest="case-plates.json",selected_cases=len(cases),display_only=True)
