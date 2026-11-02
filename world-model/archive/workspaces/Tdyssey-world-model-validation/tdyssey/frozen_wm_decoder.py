"""Frozen image-only diagnostic measurement, with explicit abstention.

The inputs are RGB, a disclosed known shape and its fixed goal. No simulator,
future state, action target or truth mask enters registration. Colors and fit
thresholds were fixed before real calibration/forecast images. This is not a
learned state estimator and its calibration gate must pass before score use.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize
from scipy.special import expit

from .envs.official import official_shape
from .evaluation import goal_metrics
from .geometry import Pose


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "frozen_wm_decoder_v1.json"
CONFIG_BYTES = CONFIG_PATH.read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
CONFIG_SHA256 = hashlib.sha256(CONFIG_BYTES).hexdigest()
VERSION = CONFIG["version"]
SCALE = CONFIG["input"]["world_extent"] / CONFIG["input"]["rgb_width"]
_YY, _XX = np.mgrid[:96, :96]
_WORLD_X = (_XX + 0.5) * SCALE - 0.5
_WORLD_Y = (_YY + 0.5) * SCALE - 0.5


def _pose(value):
    if isinstance(value, Pose):
        return value
    if isinstance(value, dict):
        return Pose(tuple(value["position"]), value["angle"])
    if len(value) == 3:
        return Pose(tuple(value[:2]), value[2])
    raise ValueError("Goal must be a Pose, position/angle mapping, or x,y,angle")


def _between(values, limits):
    return (values >= limits[0]) & (values <= limits[1])


def _masks(rgb):
    r, g, b = (rgb[..., index] for index in range(3))
    gray = CONFIG["object_gray"]
    object_gray = (_between(r, gray["r"]) & _between(g, gray["g"])
                   & _between(b, gray["b"]) & _between(b-r, gray["b_minus_r"])
                   & _between(b-g, gray["b_minus_g"]))
    red = CONFIG["object_red"]
    object_red = ((r >= red["r_min"]) & (r-g >= red["r_minus_g_min"])
                  & (r-b >= red["r_minus_b_min"]))
    blue = CONFIG["pusher_blue"]
    pusher = ((b >= blue["b_min"]) & (b-r >= blue["b_minus_r_min"])
              & (b-g >= blue["b_minus_g_min"]))
    return object_gray | object_red, pusher


def _component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return np.zeros_like(mask), 0.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labels == largest, float(sizes[largest] / mask.sum())


def _signed_distance(shape, x, y, position, angle):
    """Distance to the union (negative inside), using known local rectangles."""
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = x-position[0], y-position[1]
    lx, ly = c*dx+s*dy, -s*dx+c*dy
    distance = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), np.inf)
    for rect in shape.components:
        qx = np.abs(lx-rect.center[0]) - (rect.x1-rect.x0)/2
        qy = np.abs(ly-rect.center[1]) - (rect.y1-rect.y0)/2
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        signed = outside + np.minimum(np.maximum(qx, qy), 0)
        distance = np.minimum(distance, signed)
    return distance


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/union) if union else 0.0


@lru_cache(maxsize=2)
def _coarse_geometry(shape_name):
    shape = official_shape(shape_name)
    step = CONFIG["fit"]["angle_coarse_degrees"]
    return shape, np.deg2rad(np.arange(-180, 180, step))


def _fit_object(mask, shape_name):
    shape, angles = _coarse_geometry(shape_name)
    outline = CONFIG["fit"]["template_outline_world_pixels"]
    mx, my = float(_WORLD_X[mask].mean()), float(_WORLD_Y[mask].mean())
    centroid = shape.centroid
    coarse = []
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        position = (mx-c*centroid[0]+s*centroid[1], my-s*centroid[0]-c*centroid[1])
        predicted = _signed_distance(shape, _WORLD_X, _WORLD_Y, position, angle) <= outline
        coarse.append((_iou(mask, predicted), position, float(angle)))
    coarse.sort(key=lambda row: -row[0])
    survivors = coarse[:CONFIG["fit"]["coarse_survivors"]]
    settings = CONFIG["fit"]
    softness = settings["soft_edge_world_pixels"]
    target = mask.astype(float)

    def loss(values):
        distance = _signed_distance(shape, _WORLD_X, _WORLD_Y, values[:2], values[2])
        # A continuous silhouette loss permits subpixel fitting, without any
        # true state or renderer invocation. Hard IoU is independently checked.
        return float(np.square(expit((outline-distance)/softness)-target).sum())

    refined = []
    for _, position, angle in survivors:
        span = settings["position_bound_world_pixels"]
        angle_span = math.radians(settings["angle_refine_bound_degrees"])
        bounds = [(position[0]-span, position[0]+span),
                  (position[1]-span, position[1]+span),
                  (angle-angle_span, angle+angle_span)]
        fit = optimize.minimize(loss, (*position, angle), method="Powell", bounds=bounds,
                                options={"maxiter": settings["powell_max_iterations"],
                                         "xtol": settings["x_tolerance"],
                                         "ftol": settings["function_tolerance"]})
        refined.append((float(fit.fun), fit.x))
    _, best = min(refined, key=lambda row: row[0])
    pose = Pose((float(best[0]), float(best[1])), math.atan2(math.sin(best[2]), math.cos(best[2])))
    predicted = _signed_distance(shape, _WORLD_X, _WORLD_Y, pose.position, pose.angle) <= outline
    return pose, _iou(mask, predicted)


def _empty(reason, **diagnostics):
    return {"decoder_version": VERSION, "decoder_config_sha256": CONFIG_SHA256,
            "valid": False, "pose": None, "pusher": None, "coverage": None,
            "contact_proxy": None, "reason": reason, "fit_iou": None, **diagnostics}


def decode(image, shape_name, goal):
    """Return JSON-safe image estimates; invalid rows never receive a score.

    Accept HWC uint8 or HWC/CHW float [0,1]. Raw forecasts outside [0,1] are
    clipped solely for this measurement; the clipped fraction is reported.
    The caller retains raw predictions for feedback and separate RGB metrics.
    Geometry proximity is explicitly a proxy, never a contact-head output.
    """
    if shape_name not in ("L", "T"):
        raise ValueError("Frozen decoder supports only official L and T")
    goal = _pose(goal)
    array = np.asarray(image)
    if array.shape == (3, 96, 96):
        array = np.moveaxis(array, 0, -1)
    if array.shape != (96, 96, 3) or not np.isfinite(array).all():
        return _empty("invalid_rgb")
    if array.dtype == np.uint8:
        rgb, clipped_fraction = array.astype(float), 0.0
    elif np.issubdtype(array.dtype, np.floating):
        clipped_fraction = float(((array < 0) | (array > 1)).mean())
        rgb = np.clip(array.astype(float), 0, 1)*255
    else:
        return _empty("unsupported_rgb_dtype")
    object_mask, pusher_mask = _masks(rgb)
    object_component, object_fraction = _component(object_mask)
    pusher_component, pusher_fraction = _component(pusher_mask)
    shape = official_shape(shape_name)
    mask_settings = CONFIG["mask"]
    object_area_ratio = float(object_mask.sum()*SCALE**2/shape.area)
    pusher_radius = CONFIG["contact_proxy"]["pusher_radius_world_pixels"]
    pusher_area_ratio = float(pusher_mask.sum()*SCALE**2/(math.pi*pusher_radius**2))
    diagnostics = {"clipped_fraction": clipped_fraction,
                   "object_area_ratio": object_area_ratio,
                   "pusher_area_ratio": pusher_area_ratio,
                   "object_component_fraction": object_fraction,
                   "pusher_component_fraction": pusher_fraction}
    if not _between(object_area_ratio, mask_settings["object_area_ratio"]):
        return _empty("object_missing_or_area_mismatch", **diagnostics)
    if object_fraction < mask_settings["largest_component_fraction_min"]:
        return _empty("fragmented_object", **diagnostics)
    if not _between(pusher_area_ratio, mask_settings["pusher_area_ratio"]):
        return _empty("pusher_missing_or_area_mismatch", **diagnostics)
    if pusher_fraction < mask_settings["largest_component_fraction_min"]:
        return _empty("fragmented_pusher", **diagnostics)
    pusher = (float(_WORLD_X[pusher_component].mean()), float(_WORLD_Y[pusher_component].mean()))
    circle = np.hypot(_WORLD_X-pusher[0], _WORLD_Y-pusher[1]) <= pusher_radius
    pusher_iou = _iou(circle, pusher_mask)
    diagnostics["pusher_circle_iou"] = pusher_iou
    if pusher_iou < mask_settings["pusher_circle_iou_min"]:
        return _empty("pusher_circle_mismatch", **diagnostics)
    pose, fit_iou = _fit_object(object_component, shape_name)
    diagnostics["fit_iou"] = fit_iou
    if fit_iou < mask_settings["fit_iou_min"]:
        return _empty("object_shape_mismatch", **diagnostics)
    vertices = [v for polygon in shape.polygons(pose) for v in polygon]
    if (any(not (0 <= coord <= 511) for v in vertices for coord in v)
            or any(coord-pusher_radius < 0 or coord+pusher_radius > 511 for coord in pusher)):
        return _empty("scene_clipped", **diagnostics)
    separation = float(_signed_distance(shape, pusher[0], pusher[1], pose.position, pose.angle))
    proxy = separation <= pusher_radius + CONFIG["contact_proxy"]["extra_proximity_world_pixels"]
    return {"decoder_version": VERSION, "decoder_config_sha256": CONFIG_SHA256,
            "valid": True, "pose": {"position": list(pose.position), "angle": pose.angle},
            "pusher": list(pusher), "coverage": goal_metrics(shape, pose, goal).coverage,
            "contact_proxy": bool(proxy), "reason": "ok", **diagnostics}


def calibration_summary(rows):
    """Gate per shape; rows carry prediction and truth as explicit labels.

    Expected row keys: shape_name, prediction (decode result), truth with pose
    (position/angle) and coverage. Optional truth pusher is only an error label.
    Invalid predictions remain in measurable denominators and missing counts.
    This function does not call ``decode`` or tune any setting.
    """
    summaries = {}
    gate = CONFIG["calibration_gate"]
    for name in ("L", "T"):
        selected = [r for r in rows if r.get("shape_name", r.get("shape")) == name]
        valid = [r for r in selected if r["prediction"]["valid"]]
        positions, angles, coverages, pushers = [], [], [], []
        near_goal = []
        for row in valid:
            prediction, truth = row["prediction"], row["truth"]
            predicted_pose, true_pose = _pose(prediction["pose"]), _pose(truth["pose"])
            positions.append(math.dist(predicted_pose.position, true_pose.position))
            angles.append(math.degrees(official_shape(name).angle_error(predicted_pose.angle, true_pose.angle)))
            coverages.append(abs(prediction["coverage"]-truth["coverage"]))
            if truth.get("pusher") is not None:
                pushers.append(math.dist(prediction["pusher"], truth["pusher"]))
            if abs(truth["coverage"]-0.95) <= 0.05:
                near_goal.append(coverages[-1])
        fraction = len(valid)/len(selected) if selected else 0.0
        position_p90 = float(np.percentile(positions, 90)) if positions else None
        angle_p90 = float(np.percentile(angles, 90)) if angles else None
        coverage_mae = float(np.mean(coverages)) if coverages else None
        pusher_p90 = float(np.percentile(pushers, 90)) if pushers else None
        passed = bool(selected and fraction >= gate["measurable_fraction_min"]
                      and position_p90 is not None and position_p90 <= gate["position_p90_world_pixels_max"]
                      and len(pushers) == len(valid) and pusher_p90 is not None
                      and pusher_p90 <= gate["pusher_position_p90_world_pixels_max"]
                      and angle_p90 <= gate["angle_p90_degrees_max"]
                      and coverage_mae <= gate["coverage_mae_max"])
        summaries[name] = {"total": len(selected), "valid": len(valid),
                           "missing": len(selected)-len(valid), "measurable_fraction": fraction,
                           "position_p90_pixels": position_p90, "angle_p90_degrees": angle_p90,
                           "coverage_mae": coverage_mae,
                           "pusher_position_p90_pixels": pusher_p90,
                           "near_goal_valid": len(near_goal),
                           "near_goal_coverage_mae": float(np.mean(near_goal)) if near_goal else None,
                           "passed": passed, "ranking_available": passed,
                           "success_precision_established": False}
    return summaries
