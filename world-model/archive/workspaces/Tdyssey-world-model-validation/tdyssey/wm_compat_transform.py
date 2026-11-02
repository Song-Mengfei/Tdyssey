"""The fixed T1 RGB-only palette hypothesis; NumPy array operations only.

This is a pointwise color-line approximation, not an inverse renderer or a
geometry repair. It accepts neither state, mask, goal, action nor contact.
The future harness must apply it only to real observation inputs, once; raw
autoregressive predictions must bypass this module.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


_SWATCHES = {
    "W": (255, 255, 255), "Q": (211, 211, 211),
    "P0": (65, 105, 225), "P1": (78, 126, 255),
    "G0": (119, 136, 153), "G1": (143, 163, 184),
    "R0": (165, 28, 48), "R1": (198, 34, 58),
    "V": (144, 238, 144), "K": (0, 0, 0),
}
_SEGMENTS = (
    ("W", "Q", "W", "Q"),
    ("W", "P0", "W", "P0"),
    ("P0", "P1", "P0", "P1"),
    ("W", "P1", "W", "P1"),
    ("W", "G0", "W", "R0"),
    ("G0", "G1", "R0", "R1"),
    ("W", "G1", "W", "R1"),
    ("W", "V", "W", "K"),
    ("V", "G0", "K", "R0"),
    ("V", "G1", "K", "R1"),
)

T1_SPEC = {
    "version": "wm-input-compatibility-t1-v1",
    "proposal": "docs/GPC_INPUT_COMPATIBILITY_PROPOSAL.md",
    "input": {"shape": [3, 96, 96], "dtype": "float32", "finite": True,
              "range": [0, 1], "malformed": "reject; never cast or clip"},
    "swatches_rgb_255": {name: list(color) for name, color in _SWATCHES.items()},
    "segments": [dict(order=index+1, source=[u, v], output=[a, b],
                       identity=(u, v)==(a, b))
                 for index, (u, v, a, b) in enumerate(_SEGMENTS)],
    "arithmetic": {
        "pixel_conversion": "float64(input) * float64(255)",
        "projection": "alpha=clip(dot(p-u,v-u)/dot(v-u,v-u),0,1), float64",
        "distance": "squared Euclidean RGB distance to the source segment, float64",
        "tie_squared_distance": 1e-9,
        "tie_rule": "earliest ordered segment whose distance <= global minimum + 1e-9",
        "acceptance": "sqrt(selected squared RGB distance / 3) <= 8",
        "rms_channel_residual_max_rgb_255": 8,
        "conversion_output": "float32(output_u+alpha*(output_v-output_u)) * float32(1/255)",
        "requantization": "none",
        "identity_and_unmatched": "bitwise preserve original float32 pixel in a new array",
    },
    "spatial_operation": "none; independent mapping at each original pixel address",
    "runtime_boundary": "once per real history/TF observation; never transform raw AR feedback",
    "forbidden_inputs": ["state", "pose", "mask", "goal", "action", "contact", "future truth"],
}


def _canonical_spec():
    return json.dumps(T1_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


T1_SHA256 = hashlib.sha256(_canonical_spec()).hexdigest()


def _validate(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.shape != (3, 96, 96) or rgb.dtype != np.dtype("float32"):
        raise ValueError("T1 requires a (3,96,96) float32 NumPy RGB array")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError("T1 requires finite real-observation RGB in [0,1]; no clipping")
    if hashlib.sha256(_canonical_spec()).hexdigest() != T1_SHA256:
        raise RuntimeError("The exported T1 specification was modified after initialization")


def _winner_indices(squared_distances):
    """Tie against the global minimum, rather than chained pairwise ties."""
    minimum = np.min(squared_distances, axis=0)
    tied = squared_distances <= minimum[None, :] + T1_SPEC["arithmetic"]["tie_squared_distance"]
    return np.argmax(tied, axis=0)


def _base_diagnostics(rgb, output, arm):
    return dict(arm=arm, transform="identity" if arm=="A" else "T1",
        t1_spec_sha256=T1_SHA256, pixel_count=96*96, shape=[3, 96, 96], dtype="float32",
        input_sha256=hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
        output_sha256=hashlib.sha256(output.tobytes(order="C")).hexdigest(),
        input_range=[float(rgb.min()), float(rgb.max())],
        output_range=[float(output.min()), float(output.max())],
        output_finite=bool(np.isfinite(output).all()),
        output_in_range=bool(np.all(output>=0) and np.all(output<=1)),
        changed_pixels=int(np.any(output != rgb, axis=0).sum()),
        input_mutated=False, spatial_operation="none", observations_only=True)


def transform_t1(rgb):
    """Return a new float32 CHW image and JSON-safe fixed-rule diagnostics.

    No spatial filtering, threshold adaptation, random draw or image decoder is
    involved. Identity/unmatched pixels retain their exact original bit values.
    Winner counts include unmatched pixels; accepted counts distinguish them.
    """
    _validate(rgb)
    pixels = np.moveaxis(rgb, 0, -1).astype(np.float64).reshape(-1, 3)*np.float64(255)
    count = len(pixels)
    alphas = np.empty((len(_SEGMENTS), count), dtype=np.float64)
    distances = np.empty_like(alphas)
    for index, (u, v, _, _) in enumerate(_SEGMENTS):
        start = np.asarray(_SWATCHES[u], dtype=np.float64)
        direction = np.asarray(_SWATCHES[v], dtype=np.float64)-start
        alpha = np.clip(np.sum((pixels-start)*direction, axis=1)/np.dot(direction, direction), 0, 1)
        alphas[index] = alpha
        distances[index] = np.square(pixels-(start+alpha[:, None]*direction)).sum(axis=1)
    winners = _winner_indices(distances)
    offsets = np.arange(count)
    residuals = np.sqrt(distances[winners, offsets]/np.float64(3))
    accepted = residuals <= T1_SPEC["arithmetic"]["rms_channel_residual_max_rgb_255"]
    output = rgb.copy()
    output_pixels = np.moveaxis(output, 0, -1).copy().reshape(-1, 3)
    winner_counts, accepted_counts, per_segment = {}, {}, {}
    converted = np.zeros(count, dtype=bool)
    identity = np.zeros(count, dtype=bool)
    for index, (u, v, a, b) in enumerate(_SEGMENTS):
        name = f"{index+1:02d}:{u}--{v}"
        won = winners==index
        selected = won & accepted
        winner_counts[name] = int(won.sum())
        accepted_counts[name] = int(selected.sum())
        segment_residuals = residuals[won]
        per_segment[name] = dict(winner_pixels=int(won.sum()), accepted_pixels=int(selected.sum()),
            mean_rms_rgb_255=float(segment_residuals.mean()) if len(segment_residuals) else None,
            max_rms_rgb_255=float(segment_residuals.max()) if len(segment_residuals) else None)
        if (u, v)==(a, b):
            identity[selected] = True
            continue
        if np.any(selected):
            start = np.asarray(_SWATCHES[a], dtype=np.float64)
            direction = np.asarray(_SWATCHES[b], dtype=np.float64)-start
            mapped = (start+alphas[index, selected, None]*direction).astype(np.float32)
            output_pixels[selected] = mapped*np.float32(1/255)
            converted[selected] = True
    output = np.moveaxis(output_pixels.reshape(96, 96, 3), -1, 0).copy()
    if not np.isfinite(output).all() or np.any(output < 0) or np.any(output > 1):
        raise RuntimeError("Fixed T1 produced an invalid output; no output clipping is allowed")
    diagnostics = _base_diagnostics(rgb, output, "B")
    diagnostics.update(classification_performed=True, winning_segment_counts=winner_counts,
        accepted_segment_counts=accepted_counts, residual_by_winning_segment=per_segment,
        accepted_pixels=int(accepted.sum()), unmatched_pixels=int((~accepted).sum()),
        converted_pixels=int(converted.sum()), identity_preserved_pixels=int(identity.sum()),
        unchanged_pixels=count-diagnostics["changed_pixels"],
        residual_rms_rgb_255=dict(min=float(residuals.min()), max=float(residuals.max()),
            mean=float(residuals.mean()), p50=float(np.percentile(residuals,50)),
            p90=float(np.percentile(residuals,90)), p99=float(np.percentile(residuals,99))),
        residual_rms_limit_rgb_255=8, tie_squared_distance=1e-9,
        unmatched_policy="original float32 pixel preserved exactly")
    return output, diagnostics


def transform_for_arm(rgb, arm):
    """A is a strict validated identity copy; B applies the sole fixed T1."""
    if arm not in ("A", "B"):
        raise ValueError("Compatibility arm must be exactly 'A' or 'B'")
    if arm=="B":
        return transform_t1(rgb)
    _validate(rgb)
    output=rgb.copy()
    diagnostics=_base_diagnostics(rgb,output,"A")
    diagnostics.update(classification_performed=False, accepted_pixels=None,
        unmatched_pixels=None, winning_segment_counts={}, accepted_segment_counts={},
        converted_pixels=0, identity_preserved_pixels=96*96, unchanged_pixels=96*96,
        residual_rms_rgb_255=None, unmatched_policy="not applicable to identity arm")
    return output, diagnostics
