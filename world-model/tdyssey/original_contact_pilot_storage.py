"""Lossless saved-telemetry storage, independent of any simulator or model.

V2 extracts only complete, finite, 15-float microstep state triples into exact
little-endian float64 vectors. JSON types and all other evidence are retained.
Reconstruction must reproduce the original canonical JSON SHA, not just the
compact skeleton SHA. Legacy archives remain readable with allow_pickle=False.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np

FORMAT = "original-contact-pilot-telemetry.float64-ref.v2"
STATE_FIELDS = ("pre_state", "integrator_pre_state", "post_state")
TELEMETRY_STORAGE_KEYS = (
    "telemetry_zlib", "telemetry_sha256", "telemetry_format",
    "compact_telemetry_zlib", "compact_telemetry_sha256",
    "telemetry_microstates_float64", "telemetry_microstate_refs",
)


def canonical(value):
    """Byte-identical to the existing collector's canonical JSON convention."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def safe_json(value):
    """Keep the legacy sanitizer's types, including its nonfinite strings."""
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _finite_float_state(value):
    # Integers/bools must stay JSON: 0 and 0.0 have different canonical bytes.
    return (isinstance(value, list) and len(value) == 15
            and all(type(item) is float and math.isfinite(item) for item in value))


def pack_telemetry(telemetry, decisions):
    """Return storage-only NPZ arrays; never mutate the caller's evidence.

    A complete microstep triple is extracted together. Partial, malformed, mixed
    integer/float and nonfinite rows remain entirely in the legacy JSON form.
    Exact vector-byte deduplication preserves the sign of zero and never rounds.
    """
    skeleton = safe_json(dict(telemetry=telemetry, decisions=decisions))
    original_digest = hashlib.sha256(canonical(skeleton)).digest()
    vectors, references, known = [], [], {}
    controls = skeleton["telemetry"]
    for control_index, control in enumerate(controls if isinstance(controls, list) else []):
        rows = control.get("substeps") if isinstance(control, dict) else None
        if not isinstance(rows, list):
            continue
        for micro_index, row in enumerate(rows):
            if not isinstance(row, dict) or not all(_finite_float_state(row.get(key)) for key in STATE_FIELDS):
                continue
            for field_index, field in enumerate(STATE_FIELDS):
                state = np.asarray(row.pop(field), dtype="<f8")
                bits = state.tobytes()
                state_index = known.get(bits)
                if state_index is None:
                    state_index = len(vectors)
                    known[bits] = state_index
                    vectors.append(state)
                references.append((control_index, micro_index, field_index, state_index))
    compact = canonical(skeleton)
    return dict(
        telemetry_format=np.asarray(FORMAT),
        compact_telemetry_zlib=np.frombuffer(zlib.compress(compact, 6), np.uint8),
        compact_telemetry_sha256=np.frombuffer(hashlib.sha256(compact).digest(), np.uint8),
        telemetry_sha256=np.frombuffer(original_digest, np.uint8),
        # Column-major .npy order groups like components for deflate while the
        # logical shapes, dtype and exact float64 values remain unchanged.
        telemetry_microstates_float64=np.asfortranarray(np.asarray(vectors, dtype="<f8").reshape(-1, 15)),
        telemetry_microstate_refs=np.asfortranarray(np.asarray(references, dtype="<i8").reshape(-1, 4)),
    )


def _bytes_array(arrays, key, expected_length=None):
    value = np.asarray(arrays[key])
    if value.dtype != np.dtype("uint8") or value.ndim != 1:
        raise ValueError("Invalid byte-array storage field: " + key)
    if expected_length is not None and len(value) != expected_length:
        raise ValueError("Invalid byte-array length: " + key)
    return value.tobytes()


def _verified_json(arrays, data_key, digest_key):
    payload = zlib.decompress(_bytes_array(arrays, data_key))
    if hashlib.sha256(payload).digest() != _bytes_array(arrays, digest_key, 32):
        raise ValueError("Saved telemetry hash mismatch: " + data_key)
    return json.loads(payload)


def unpack_telemetry(arrays):
    """Rebuild original telemetry/decisions from a legacy or V2 array mapping."""
    if "telemetry_format" not in arrays:
        if "compact_telemetry_zlib" in arrays:
            raise ValueError("Compact telemetry lacks an explicit format identity")
        return _verified_json(arrays, "telemetry_zlib", "telemetry_sha256")
    marker = np.asarray(arrays["telemetry_format"])
    if marker.ndim != 0 or marker.dtype.kind != "U" or marker.item() != FORMAT:
        raise ValueError("Unsupported telemetry storage format")
    if "telemetry_zlib" in arrays:
        raise ValueError("Ambiguous legacy and compact telemetry payloads")
    result = _verified_json(arrays, "compact_telemetry_zlib", "compact_telemetry_sha256")
    if not isinstance(result, dict) or set(result) != {"telemetry", "decisions"}:
        raise ValueError("Unexpected compact telemetry payload keys")
    states = np.asarray(arrays["telemetry_microstates_float64"])
    refs = np.asarray(arrays["telemetry_microstate_refs"])
    if states.dtype != np.dtype("<f8") or states.ndim != 2 or states.shape[1] != 15 or not np.isfinite(states).all():
        raise ValueError("Invalid exact-float64 microstate table")
    if refs.dtype != np.dtype("<i8") or refs.ndim != 2 or refs.shape[1] != 4:
        raise ValueError("Invalid microstate reference table")
    controls = result["telemetry"]
    if len(refs) and not isinstance(controls, list):
        raise ValueError("Microstate references need a control list")
    seen = set()
    used_states = set()
    for c, m, f, s in refs:
        c, m, f, s = int(c), int(m), int(f), int(s)
        if c < 0 or c >= len(controls) or f < 0 or f >= len(STATE_FIELDS) or s < 0 or s >= len(states):
            raise ValueError("Microstate reference index outside saved evidence")
        rows = controls[c].get("substeps") if isinstance(controls[c], dict) else None
        if not isinstance(rows, list) or m < 0 or m >= len(rows) or not isinstance(rows[m], dict):
            raise ValueError("Microstate reference target is not a saved microstep")
        key = STATE_FIELDS[f]
        location = (c, m, f)
        if location in seen or key in rows[m]:
            raise ValueError("Duplicate or overwriting microstate reference")
        seen.add(location)
        used_states.add(s)
        # Each reference receives an independent list, including aliases.
        rows[m][key] = states[s].tolist()
    if len(used_states) != len(states):
        raise ValueError("Unreferenced microstate vector in compact evidence")
    if hashlib.sha256(canonical(result)).digest() != _bytes_array(arrays, "telemetry_sha256", 32):
        raise ValueError("Reconstructed original telemetry hash mismatch")
    return result


def load_attempt_archive(path):
    """Legacy-compatible public result, excluding all storage-only arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    payload = unpack_telemetry(arrays)
    result = {key: value for key, value in arrays.items() if key not in TELEMETRY_STORAGE_KEYS}
    result.update(payload)
    return result
