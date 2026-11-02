"""Saved-RGB measurement with continuous chromatic evidence and local validity.

No environment, learned model or truth label is imported or accepted. Known
rigid geometry is a diagnostic prior, never a license to score a malformed image.
Development empirical intervals are sensitivity safeguards, not probabilities.
"""

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
from .wm_measurement_contract import coverage_measurement, termination_from_coverage

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/wm_image_measurement_v2.json"
CONFIG_BYTES = CONFIG_PATH.read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
CONFIG_SHA256 = hashlib.sha256(CONFIG_BYTES).hexdigest()
VERSION = CONFIG["version"]
SCALE = 512 / 96
_Y, _X = np.mgrid[:96, :96]
WORLD_X, WORLD_Y = (_X + .5) * SCALE - .5, (_Y + .5) * SCALE - .5


def _goal(value):
    if isinstance(value, Pose):
        return value
    if isinstance(value, dict):
        return Pose(tuple(value["position"]), value["angle"])
    return Pose(tuple(value[:2]), value[2])


def _chroma_map(rgb, channel_pair, direction, angular_sigma=None):
    """Fractional object evidence: neutral backgrounds have zero chroma.

    This is a deterministic color-unmixing heuristic, not an occupancy probability.
    The channel direction retains antialiased RGB values rather than binarizing.
    """
    a, b = channel_pair
    vector = np.stack((rgb[..., a[0]]-rgb[..., a[1]], rgb[..., b[0]]-rgb[..., b[1]]), axis=-1)
    prototype = np.asarray(direction, dtype=float)
    amplitude = (vector * prototype).sum(-1) / (prototype @ prototype)
    residual = np.sqrt(np.square(vector-amplitude[..., None]*prototype).sum(-1))
    result = np.clip(amplitude, 0, 1) * np.exp(-.5 * np.square(residual / CONFIG["chromatic"]["perpendicular_sigma"]))
    if angular_sigma is not None:
        result *= np.exp(-.5*np.square(residual/np.maximum(np.linalg.norm(vector, axis=-1), 1)/angular_sigma))
    result[np.linalg.norm(vector, axis=-1) < CONFIG["chromatic"]["noise_chroma_floor"]] = 0
    return result


def _green_edge_unmix(rgb, palette):
    """Retain antialias blends against visible green, without any goal ROI.

    The observed RGB alone supplies color evidence; fixed-goal coordinates never
    enter object fitting. Pure green maps to zero foreground alpha.
    """
    background = np.asarray(CONFIG["chromatic"]["green_background_rgb"], dtype=float)
    colors = ((143, 163, 184), (119, 136, 153)) if palette == "gray" else ((198, 34, 58), (165, 28, 48))
    estimates = []
    for foreground in colors:
        direction = np.asarray(foreground)-background
        amplitude = np.sum((rgb-background)*direction, axis=-1)/(direction@direction)
        residual = np.linalg.norm(rgb-(background+amplitude[..., None]*direction), axis=-1)
        estimates.append(np.clip(amplitude, 0, 1)*np.exp(-.5*np.square(residual/CONFIG["chromatic"]["green_unmix_residual_sigma"])))
    alpha = np.maximum.reduce(estimates)
    greenish = (rgb[..., 1] > rgb[..., 0]+10) & (rgb[..., 2] < rgb[..., 1]+5)
    return np.where(greenish, alpha, 0)


def continuous_maps(rgb):
    settings = CONFIG["chromatic"]
    gray = _chroma_map(rgb, ((2, 0), (2, 1)), settings["gray_direction"])
    red_direction = settings["red_direction"]
    r, g, b = (rgb[..., i] for i in range(3))
    seeds = (r >= 100) & (r-g >= 50) & (r-b >= 35)
    if seeds.sum() >= 12:
        vectors = np.stack((r[seeds]-g[seeds], r[seeds]-b[seeds]), axis=-1)
        strength = np.linalg.norm(vectors, axis=-1)
        red_direction = np.median(vectors[strength >= np.percentile(strength, 75)], axis=0)
    red = _chroma_map(rgb, ((0, 1), (0, 2)), red_direction)
    gray = np.maximum(gray, _green_edge_unmix(rgb, "gray"))
    red = np.maximum(red, _green_edge_unmix(rgb, "red"))
    blue = _chroma_map(rgb, ((2, 0), (2, 1)), settings["blue_direction"], settings["blue_angular_sigma"])
    palette = "gray" if float(gray.sum()) >= float(red.sum()) else "red"
    return (gray if palette == "gray" else red), blue, palette


def _region(alpha, minimum_seed):
    seed = alpha >= CONFIG["chromatic"]["component_seed_alpha"]
    labels, count = ndimage.label(seed, np.ones((3, 3), dtype=bool))
    if not count or seed.sum() < minimum_seed:
        return None, 0.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == int(np.argmax(sizes))
    y, x = np.nonzero(largest)
    pad = CONFIG["chromatic"]["roi_pad_image_pixels"]
    bounds = (slice(max(0, y.min()-pad), min(96, y.max()+pad+1)),
              slice(max(0, x.min()-pad), min(96, x.max()+pad+1)))
    # Keep fractional boundary pixels and all components in the image-derived ROI.
    component_mass = float(alpha[ndimage.binary_dilation(largest, iterations=1)].sum())
    total_mass = float(alpha.sum())
    return bounds, min(1.0, component_mass/max(total_mass, 1e-12))


def signed_distance(shape, x, y, position, angle):
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = x-position[0], y-position[1]
    lx, ly = c*dx+s*dy, -s*dx+c*dy
    distance = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), np.inf)
    for rect in shape.components:
        qx = np.abs(lx-rect.center[0])-(rect.x1-rect.x0)/2
        qy = np.abs(ly-rect.center[1])-(rect.y1-rect.y0)/2
        distance = np.minimum(distance, np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0))
    return distance


def soft_iou(first, second):
    denominator = np.maximum(first, second).sum()
    return float(np.minimum(first, second).sum()/denominator) if denominator else 0.0


def _template(shape, x, y, values, palette):
    outline = CONFIG["fit"]["official_outline_world_pixels"] if palette == "gray" else values[3]
    blur = 0 if palette == "gray" else values[4]
    distance = signed_distance(shape, x, y, values[:2], values[2])
    softness = CONFIG["fit"]["edge_softness_world_pixels"]
    border = CONFIG["fit"]["border_chroma_ratio"]
    result = border*expit((outline-distance)/softness) + (1-border)*expit((-2-distance)/softness)
    if blur > 1e-6:
        result = ndimage.gaussian_filter(result, blur, mode="constant")
    return result


@lru_cache(maxsize=2)
def _coarse(shape_name):
    return official_shape(shape_name), np.deg2rad(np.arange(-180, 180, CONFIG["fit"]["coarse_degrees"]))


def fit_object(alpha, shape_name, palette, bounds):
    shape, angles = _coarse(shape_name)
    x, y, target = WORLD_X[bounds], WORLD_Y[bounds], alpha[bounds]
    mass = target.sum()
    mx, my = float((x*target).sum()/mass), float((y*target).sum()/mass)
    cx, cy = shape.centroid
    settings = CONFIG["fit"]
    starts = []
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        position = (mx-c*cx+s*cy, my-s*cx-c*cy)
        values = (*position, angle, 1.0, .25)
        estimate = _template(shape, x, y, values, palette)
        starts.append((float(np.square(estimate-target).sum()), values))
    starts.sort(key=lambda item: item[0])
    refined = []
    for _, values in starts[:settings["survivors"]]:
        span = settings["position_span_pixels"]
        rotation = math.radians(settings["angle_span_degrees"])
        fit_bounds = [(values[0]-span, values[0]+span), (values[1]-span, values[1]+span), (values[2]-rotation, values[2]+rotation)]
        initial = values[:3]
        if palette == "red":
            fit_bounds += [settings["red_outline_bounds_world_pixels"], settings["generated_blur_sigma_image_pixels"]]
            initial = values
        def loss(candidate):
            return float(np.square(_template(shape, x, y, candidate, palette)-target).sum())
        fitted = optimize.minimize(loss, initial, method="Powell", bounds=fit_bounds,
                                  options={"maxiter": settings["max_iterations"], "xtol": .01, "ftol": 1e-7})
        refined.append((float(fitted.fun), fitted.x))
    refined.sort(key=lambda item: item[0])
    loss, values = refined[0]
    pose = Pose((float(values[0]), float(values[1])), math.atan2(math.sin(values[2]), math.cos(values[2])))
    fitted = _template(shape, x, y, values, palette)
    roi_consistency = soft_iou(target, fitted)
    full_template = _template(shape, WORLD_X, WORLD_Y, values, palette)
    consistency = soft_iou(alpha, full_template)
    alternatives = [dict(position=[float(v[0]), float(v[1])], angle=float(v[2]), loss=float(cost)) for cost, v in refined[1:]]
    return pose, dict(soft_iou=consistency, fit_iou=consistency, roi_soft_iou=roi_consistency,
                      full_soft_iou=consistency, outside_roi_template_mass=float(full_template.sum()-full_template[bounds].sum()),
                      squared_residual=float(np.square(full_template-alpha).sum()), optimization_roi_residual=loss,
                      residual_is_probability=False, palette=palette, alternative_fits=alternatives,
                      fitted_outline=2.0 if palette == "gray" else float(values[3]),
                      fitted_blur_sigma=0.0 if palette == "gray" else float(values[4]))


def measure_pusher(alpha):
    settings = CONFIG["pusher_gate"]
    bounds, component_fraction = _region(alpha, settings["minimum_seed_pixels"])
    record = dict(valid=False, position=None, reason="pusher_missing", component_mass_fraction=component_fraction)
    if bounds is None:
        return record
    labels, count = ndimage.label(alpha >= CONFIG["chromatic"]["component_seed_alpha"], np.ones((3, 3), dtype=bool))
    nominal_mass = math.pi*settings["radius"]**2/SCALE**2
    components = []
    for label in range(1, count+1):
        selected = labels == label
        if selected.sum() < settings["minimum_seed_pixels"]:
            continue
        mass = float(alpha[selected].sum())
        components.append(dict(mass=mass, nominal_mass_ratio=mass/nominal_mass,
            position=[float((WORLD_X[selected]*alpha[selected]).sum()/mass), float((WORLD_Y[selected]*alpha[selected]).sum()/mass)]))
    strongest = max((item["mass"] for item in components), default=0)
    meaningful = [item for item in components if item["nominal_mass_ratio"] >= settings["meaningful_component_nominal_mass_min"]
                  and item["mass"] >= settings["meaningful_component_relative_mass_min"]*strongest]
    record["meaningful_components"] = meaningful
    if len(meaningful) > 1:
        record.update(reason="ambiguous_multiple_pushers", candidate_positions=[item["position"] for item in meaningful])
        return record
    target, x, y = alpha[bounds], WORLD_X[bounds], WORLD_Y[bounds]
    mass = float(target.sum())
    start = (float((target*x).sum()/mass), float((target*y).sum()/mass))
    def template(values):
        distance = np.hypot(x-values[0], y-values[1])-settings["radius"]
        return expit(-distance/.65)
    fit = optimize.minimize(lambda v: float(np.square(template(v)-target).sum()), start,
                            method="Powell", bounds=[(start[0]-6, start[0]+6), (start[1]-6, start[1]+6)],
                            options={"maxiter": 30, "xtol": .01})
    ratio = mass*SCALE**2/(math.pi*settings["radius"]**2)
    consistency = soft_iou(target, template(fit.x))
    valid = settings["area_ratio"][0] <= ratio <= settings["area_ratio"][1] and consistency >= settings["soft_iou_min"]
    record.update(valid=bool(valid), position=[float(v) for v in fit.x] if valid else None,
                  candidate_position=[float(v) for v in fit.x], area_ratio=ratio, fit_iou=consistency,
                  reason="ok" if valid else "pusher_size_or_shape_mismatch")
    return record


def _empty(reason):
    coverage = coverage_measurement(None, valid=False, reason=reason)
    return dict(decoder_version=VERSION, decoder_config_sha256=CONFIG_SHA256, valid=False, pose=None,
                reason=reason, state=dict(valid=False, pose=None, reason=reason),
                shape=dict(valid=False, reason=reason, fit_iou=None),
                pusher=dict(valid=False, position=None, reason=reason), coverage=coverage,
                termination=termination_from_coverage(coverage))


def decode(image, shape_name, goal):
    """RGB+shape+fixed goal only; object and pusher failures stay local."""
    if shape_name not in ("L", "T"):
        raise ValueError("Only official L/T geometry is supported")
    array = np.asarray(image)
    if array.shape == (3, 96, 96):
        array = np.moveaxis(array, 0, -1)
    if array.shape != (96, 96, 3) or not np.isfinite(array).all():
        return _empty("invalid_rgb")
    if array.dtype == np.uint8:
        rgb, fraction = array.astype(float), 0.0
    elif np.issubdtype(array.dtype, np.floating):
        fraction = float(((array < 0) | (array > 1)).mean())
        rgb = np.clip(array, 0, 1).astype(float)*255
    else:
        return _empty("unsupported_rgb_dtype")
    object_alpha, pusher_alpha, palette = continuous_maps(rgb)
    pusher = measure_pusher(pusher_alpha)
    shape = official_shape(shape_name)
    gate = CONFIG["shape_gate"]
    bounds, component_fraction = _region(object_alpha, gate["minimum_object_seed_pixels"])
    output = _empty("object_missing")
    output.update(pusher=pusher, clipped_fraction=fraction, palette=palette)
    if bounds is None:
        return output
    pose, fit = fit_object(object_alpha, shape_name, palette, bounds)
    ratio = float(object_alpha[bounds].sum()*SCALE**2/shape.area)
    geometry_valid = bool(gate["area_ratio"][0] <= ratio <= gate["area_ratio"][1]
                          and component_fraction >= gate["component_mass_fraction_min"]
                          and fit["soft_iou"] >= gate["soft_iou_min"])
    vertices = [v for polygon in shape.polygons(pose) for v in polygon]
    clipped = any(not 0 <= coordinate <= 511 for vertex in vertices for coordinate in vertex)
    reason = ("scene_clipped" if clipped else "object_area_mismatch" if not gate["area_ratio"][0] <= ratio <= gate["area_ratio"][1]
              else "fragmented_object" if component_fraction < gate["component_mass_fraction_min"]
              else "rigid_shape_mismatch" if fit["soft_iou"] < gate["soft_iou_min"] else "ok")
    valid = geometry_valid and not clipped
    pose_dict = dict(position=list(pose.position), angle=pose.angle)
    output["shape"] = dict(valid=valid, reason=reason, area_ratio=ratio, component_mass_fraction=component_fraction, **fit)
    output["state"] = dict(valid=valid, pose=pose_dict if valid else None, candidate_pose=pose_dict, reason=reason,
                           ambiguity_assessed=False, ambiguity=None,
                           ambiguity_note="three local fit alternatives and residual do not assess full pose ambiguity or calibrated confidence")
    if valid:
        estimate = goal_metrics(shape, pose, _goal(goal)).coverage
        output["coverage"] = coverage_measurement(estimate, half_width=CONFIG["termination"]["coverage_half_width"],
            uncertainty_kind=CONFIG["termination"]["uncertainty_kind"], reason="development_empirical_error_floor_not_certified")
    else:
        output["coverage"] = coverage_measurement(None, valid=False, reason=reason)
    output["termination"] = termination_from_coverage(output["coverage"])
    output.update(valid=bool(valid and pusher["valid"]), pose=pose_dict if valid else None,
                  reason=reason if not valid else "ok" if pusher["valid"] else "pusher_unavailable_object_valid",
                  pusher_position=pusher["position"], fit_iou=fit["soft_iou"],
                  contact_proxy=None)
    if valid and pusher["valid"]:
        distance = float(signed_distance(shape, pusher["position"][0], pusher["position"][1], pose.position, pose.angle))
        output["contact_proxy"] = bool(distance <= 17)
    return output
