"""Offline evaluation of unchanged GPC RGB predictions.

Registration in this module is a *scoring operation*. Its masks, transformations,
simulator labels and calibration never enter the sampler or replace its output.
All generated RGB frames, including malformed shapes, feed back unchanged.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


TRAIN_LETTERS = tuple("ABCE FHIJLMOPSTUVXY".replace(" ", ""))
UNSEEN_LETTERS = tuple("DGKNQRWZ")
METRICS_VERSION = "native-gpc-rgb-evaluation.v1"
IMAGE_SIZE = 96
WORLD_SCALE = 512 / IMAGE_SIZE


def _json_value(value):
    """JSON is finite; unavailable measurements remain explicit nulls."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    Path(path).write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _renderer_source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("tdyssey/native_rgb_env.py", "tdyssey/envs/official.py", "tdyssey/letters.py", "tdyssey/geometry.py")
    return {name: _file_sha256(root/name) for name in names}


def validate_calibration(calibration):
    """Reject altered tolerances or a calibration from another renderer source."""
    if isinstance(calibration, (str, Path)):
        calibration = json.loads(Path(calibration).read_text(encoding="utf-8"))
    raw = {key: value for key, value in calibration.items() if key != "sha256"}
    if (calibration.get("model_outputs_used") is not False
            or calibration.get("sha256") != hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            or set(calibration.get("letters", [])) != set(TRAIN_LETTERS)
            or calibration.get("renderer_source_sha256") != _renderer_source_identity()):
        raise ValueError("Expected intact frozen seen-renderer calibration bound to current renderer sources")
    return calibration


def final_protocol_identity(config, checkpoint, data_dir, calibration):
    """Return the identity a server must save before either final split is read.

    This hashes metadata and a trusted complete checkpoint; it performs no model
    inference, final-image rendering, or optimizer update.
    """
    from .native_rgb_training import load_checkpoint_payload
    from .gpc_world_model import world_model_spec
    calibration = validate_calibration(calibration)
    payload = load_checkpoint_payload(checkpoint)
    checkpoint_identity = payload.get("identity", {})
    if checkpoint_identity.get("initialization") != world_model_spec()["checkpoint"]:
        raise ValueError("Final checkpoint must descend from the pinned original GPC release")
    manifest_hash = _file_sha256(Path(data_dir)/"manifest.json")
    if checkpoint_identity.get("data", {}).get("manifest_sha256") != manifest_hash:
        raise ValueError("Final checkpoint and evaluation dataset identities differ")
    return dict(kind="native-gpc-rgb-final-protocol.v1",
        checkpoint_sha256=_file_sha256(checkpoint), data_manifest_sha256=manifest_hash,
        data_audit_sha256=_file_sha256(Path(data_dir)/"physics-audit.json"),
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        calibration_sha256=calibration["sha256"],
        evaluation_source_sha256=_file_sha256(__file__),
        renderer_source_sha256=_renderer_source_identity(),
        original_initialization=world_model_spec()["checkpoint"],
        allowed_final_splits=["seen_test", "unseen_test"])


def _rgb(image):
    import numpy as np
    image = np.asarray(image)
    if image.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Expected raw CHW RGB96")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    else:
        image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Raw RGB must be finite and in [0,1]; no clipping repair")
    return image


def palette_masks(image, calibration):
    """Separate the rendered object and pusher; never fill holes/fragments."""
    import numpy as np
    image = _rgb(image)
    pixels = image.transpose(1, 2, 0)
    prototype = np.asarray(calibration["object_rgb"], dtype=np.float64)
    direction = 1 - prototype
    alpha = ((1 - pixels) * direction).sum(-1) / float(direction @ direction)
    residual = np.linalg.norm(pixels - (1 - alpha[..., None] * direction), axis=-1)
    object_mask = ((alpha >= calibration["palette"]["minimum_alpha"])
                   & (alpha <= calibration["palette"]["maximum_alpha"])
                   & (residual <= calibration["palette"]["colour_residual"]))
    r, g, b = image
    pusher_mask = (b > .60) & (r < .45) & (g < .65) & (b - np.maximum(r, g) > .25)
    return object_mask, pusher_mask


def _centroid(mask):
    import numpy as np
    y, x = np.nonzero(mask)
    return None if not len(x) else np.asarray([x.mean(), y.mean()], dtype=np.float64)


def _iou(first, second):
    import numpy as np
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else None


def _topology(mask, minimum_component_pixels=2):
    import numpy as np
    from scipy import ndimage
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    foreground_components = int((sizes >= minimum_component_pixels).sum())
    # Dual 8-connected foreground / 4-connected background is intentional.
    inverse, count = ndimage.label(~np.pad(mask, 1), structure=ndimage.generate_binary_structure(2, 1))
    exterior = int(inverse[0, 0])
    hole_sizes = np.bincount(inverse.ravel())
    holes = sum(1 for i in range(1, count + 1)
                if i != exterior and hole_sizes[i] >= minimum_component_pixels)
    return dict(components=foreground_components, holes=int(holes),
                euler=foreground_components - int(holes),
                tiny_fragment_pixels=int(sizes[sizes < minimum_component_pixels].sum()))


def _boundary(mask):
    import numpy as np
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)))


def _contour_distance(first, second):
    import numpy as np
    from scipy import ndimage
    a, b = _boundary(first), _boundary(second)
    if not a.any() or not b.any():
        return None
    da, db = ndimage.distance_transform_edt(~a), ndimage.distance_transform_edt(~b)
    return float((da[b].mean() + db[a].mean()) / 2 * WORLD_SCALE)


def _warp_for_score(mask, angle_degrees, target_center, output_shape):
    """Nearest-neighbour affine, used only for offline shape decomposition."""
    import numpy as np
    import cv2
    center = _centroid(mask)
    if center is None:
        return np.zeros(output_shape, dtype=bool)
    angle = math.radians(angle_degrees)
    matrix = np.asarray([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    offset = np.asarray(target_center) - matrix @ center
    affine = np.column_stack((matrix, offset))
    return cv2.warpAffine(mask.astype(np.uint8), affine,
                          (output_shape[1], output_shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def align_masks_for_score(predicted, truth, *, angle_step=10):
    """Return pose/shape measurements, never an image for prediction feedback.

    Translation is centroid alignment; a coarse angular grid plus a one-degree
    local grid maximizes mask IoU. Equivalent mask orientations choose the smallest
    absolute angle. This is an apparent visible-mask angle, not a pose estimator.
    """
    import numpy as np
    center = _centroid(truth)
    if center is None or _centroid(predicted) is None:
        raise ValueError("Cannot score an empty object mask")
    # Padding prevents the scoring transform itself from clipping components.
    padding = IMAGE_SIZE
    first, second = np.pad(predicted, padding), np.pad(truth, padding)
    target = center + padding
    candidates = []
    def measure(angle):
        mask = _warp_for_score(first, angle, target, second.shape)
        candidates.append((float(_iou(mask, second)), abs(angle), angle, mask))
    for angle in range(-180, 180, angle_step):
        measure(angle)
    best = max(candidates, key=lambda row: (row[0], -row[1]))
    for angle in range(int(best[2] - angle_step), int(best[2] + angle_step + 1)):
        measure((angle + 180) % 360 - 180)
    best = max(candidates, key=lambda row: (row[0], -row[1]))
    # A subpixel centroid can change after binary rasterization. A fixed +/-one
    # image-pixel translation grid prevents attributing that aliasing to shape
    # damage. The same grid is included in renderer-only calibration below.
    for angle in (best[2]-1, best[2], best[2]+1):
        angle = (angle + 180) % 360 - 180
        for dx in (-1, -.5, 0, .5, 1):
            for dy in (-1, -.5, 0, .5, 1):
                mask = _warp_for_score(first, angle, target + (dx, dy), second.shape)
                candidates.append((float(_iou(mask, second)), abs(angle), angle, mask))
    best = max(candidates, key=lambda row: (row[0], -row[1]))
    aligned = best[3]
    missing = float((second & ~aligned).sum() / max(1, second.sum()))
    excess = float((aligned & ~second).sum() / max(1, second.sum()))
    return dict(angle_error_deg=float(best[1]), aligned_mask_iou=float(best[0]),
                missing_area_fraction=missing, extra_area_fraction=excess,
                aligned_contour_error_px=_contour_distance(aligned, second))


def calibrate_renderer(dataset, *, letters=TRAIN_LETTERS, angles=None):
    """Freeze pixel/topology tolerances using real renderer outputs alone.

    Canonical poses and subpixel offsets are fixed independently of predictions.
    This calls rendering, never steps physics or collects interaction. Do not fit
    this calibration to a trained model or expand it after inspecting errors.
    """
    import numpy as np
    angles = tuple(angles if angles is not None else (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5))
    if set(letters) - set(TRAIN_LETTERS):
        raise ValueError("Renderer calibration is restricted to the 18 seen letters")
    entries = {entry["shape"]: entry for entry in dataset.entries}
    if set(letters) - set(entries):
        raise ValueError("Calibration requires all requested seen-letter renderers")
    images = []
    for letter in letters:
        state = np.zeros(15, dtype=np.float64)
        state[:5] = (40, 40, 256, 256, 0)
        images.append(_rgb(dataset.render_state(entries[letter], state)))
    colours = np.concatenate([image.transpose(1, 2, 0).reshape(-1, 3) for image in images])
    candidates = colours[(colours.max(1) - colours.min(1) < .22)
                        & (colours.mean(1) > .2) & (colours.mean(1) < .8)]
    if not len(candidates):
        raise ValueError("Canonical renderer did not expose an object palette")
    values, counts = np.unique(np.rint(candidates * 255).astype(np.uint8), axis=0, return_counts=True)
    prototype = values[counts.argmax()].astype(np.float64) / 255
    calibration = dict(version="renderer-antialias-calibration.v1", object_rgb=prototype.tolist(),
                       palette=dict(minimum_alpha=.40, maximum_alpha=1.25, colour_residual=.12),
                       minimum_component_pixels=2, angle_grid_degrees=10,
                       score_registration_only=True, model_outputs_used=False,
                       letters=list(letters), angles_degrees=list(angles),
                       offsets_world_pixels=[[0, 0], [.5, .5]], per_letter={},
                       renderer_source_sha256=_renderer_source_identity())
    hasher = hashlib.sha256()
    baseline_iou, baseline_contour, baseline_missing, baseline_extra = [], [], [], []
    for letter in letters:
        areas, components, holes, tiny = [], [], [], []
        reference_mask = None
        for angle in angles:
            for dx, dy in ((0, 0), (.5, .5)):
                state = np.zeros(15, dtype=np.float64)
                state[:5] = (40, 40, 256 + dx, 256 + dy, math.radians(angle))
                image = dataset.render_state(entries[letter], state)
                hasher.update(np.ascontiguousarray(image).tobytes())
                mask, _ = palette_masks(image, calibration)
                if not mask.any():
                    raise ValueError("Renderer calibration produced an empty object: " + letter)
                topology = _topology(mask, 2)
                areas.append(int(mask.sum()))
                components.append(topology["components"])
                holes.append(topology["holes"])
                tiny.append(topology["tiny_fragment_pixels"])
                if reference_mask is None:
                    reference_mask = mask.copy()
                reference = np.pad(reference_mask, IMAGE_SIZE)
                current = np.pad(mask, IMAGE_SIZE)
                target_center = _centroid(reference)
                # Rotation sign follows the renderer's screen convention; take
                # the better of the two known-angle transforms, not a learned fit.
                alternatives = [_warp_for_score(current, sign*angle+da, target_center+(dx,dy), reference.shape)
                                for sign in (-1, 1) for da in (-1, 0, 1)
                                for dx in (-1, -.5, 0, .5, 1) for dy in (-1, -.5, 0, .5, 1)]
                aligned = max(alternatives, key=lambda value: _iou(value, reference))
                baseline_iou.append(float(_iou(aligned, reference)))
                baseline_contour.append(_contour_distance(aligned, reference))
                baseline_missing.append(float((reference & ~aligned).sum()/reference.sum()))
                baseline_extra.append(float((aligned & ~reference).sum()/reference.sum()))
        relative_area_range = (max(areas) - min(areas)) / max(1, min(areas))
        calibration["per_letter"][letter] = dict(area_min=min(areas), area_max=max(areas),
            max_area_fraction_tolerance=float(relative_area_range + .02),
            components_min=min(components), components_max=max(components),
            holes_min=min(holes), holes_max=max(holes), tiny_fragment_pixels_max=max(tiny))
    # A one-image-pixel discrepancy is reported in world pixels. The following
    # limits are declared before any model output, rather than tuned to qualify it.
    calibration.update(renderer_pixels_sha256=hasher.hexdigest(),
                       contour_tolerance_px=float(max(baseline_contour) + WORLD_SCALE/2),
                       missing_extra_fraction_tolerance=float(max(baseline_missing+baseline_extra)+.02),
                       aligned_iou_floor=float(max(0, min(baseline_iou)-.02)),
                       rasterization_baseline=dict(min_iou=min(baseline_iou),
                            max_contour_px=max(baseline_contour), max_missing=max(baseline_missing),
                            max_extra=max(baseline_extra)),
                       topology_exclusion="pusher occlusion or viewport clipping -> unavailable, never a pass")
    calibration["registration_translation_grid_pixels"] = [-1, -.5, 0, .5, 1]
    calibration["sha256"] = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    return calibration


def score_rgb(prediction, truth, calibration, *, letter, truth_full_mask=None):
    """Measure raw outputs. Occlusion/clipping ambiguity stays visible in rows."""
    import numpy as np
    from scipy import ndimage
    prediction, truth = _rgb(prediction), _rgb(truth)
    predicted, blue = palette_masks(prediction, calibration)
    target, target_blue = palette_masks(truth, calibration)
    if not target.any():
        raise ValueError("Invalid truth: object mask empty")
    if not predicted.any():
        raise ValueError("Prediction object mask empty")
    c1, c2 = _centroid(predicted), _centroid(target)
    p1, p2 = _centroid(blue), _centroid(target_blue)
    if p2 is None:
        raise ValueError("Invalid truth: pusher mask empty")
    if p1 is None:
        raise ValueError("Prediction pusher mask empty")
    topology = _topology(predicted, calibration["minimum_component_pixels"])
    true_topology = _topology(target, calibration["minimum_component_pixels"])
    clipping = bool(predicted[0].any() or predicted[-1].any() or predicted[:, 0].any()
                    or predicted[:, -1].any() or target[0].any() or target[-1].any()
                    or target[:, 0].any() or target[:, -1].any())
    occluded = bool((ndimage.binary_dilation(predicted, iterations=1) & blue).any()
                    or (ndimage.binary_dilation(target, iterations=1) & target_blue).any())
    alignment = align_masks_for_score(predicted, target,
                                      angle_step=calibration.get("angle_grid_degrees", 10))
    radius = float(np.sqrt(((np.argwhere(target) - c2[::-1]) ** 2).sum(1).mean()) * WORLD_SCALE)
    position = float(np.linalg.norm(c1 - c2) * WORLD_SCALE)
    row = dict(rgb_mse=float(((prediction - truth) ** 2).mean()),
               object_position_error_px=position,
               pusher_position_error_px=float(np.linalg.norm(p1-p2)*WORLD_SCALE),
               raw_mask_iou=_iou(predicted, target), raw_contour_error_px=_contour_distance(predicted, target),
               area_fraction_error=float(abs(predicted.sum()-target.sum())/max(1, target.sum())),
               predicted_components=topology["components"], truth_components=true_topology["components"],
               extra_fragments=max(0, topology["components"]-true_topology["components"]),
               missing_components=max(0, true_topology["components"]-topology["components"]),
               predicted_holes=topology["holes"], truth_holes=true_topology["holes"],
               holes_difference=abs(topology["holes"]-true_topology["holes"]),
               tiny_fragment_pixels=topology["tiny_fragment_pixels"],
               occlusion_ambiguous=occluded, viewport_clipped=clipping,
               shape_measurement_available=not (occluded or clipping),
               angle_scope="visible-mask angular registration; symmetric/occluded shapes can be ambiguous",
               **alignment)
    row["motion_error_px"] = position + radius * math.radians(alignment["angle_error_deg"])
    if truth_full_mask is not None:
        full = np.asarray(truth_full_mask) >= calibration["palette"]["minimum_alpha"]
        if full.shape != (96, 96) or not full.any():
            raise ValueError("Invalid privileged full truth object mask")
        row["full_object_mask_iou"] = _iou(predicted, full)
        row["truth_visible_fraction"] = float((target & full).sum()/full.sum())
        row["full_mask_scope"] = "raw predicted visible mask versus privileged unoccluded truth; occlusion is penalized, separately disclosed"
    reference = calibration["per_letter"].get(letter)
    if reference is None:
        # Unseen shapes must not get tolerances fitted on their predictions.
        reference = dict(max_area_fraction_tolerance=max(v["max_area_fraction_tolerance"]
                                                        for v in calibration["per_letter"].values()),
                         components_min=0, components_max=max(v["components_max"]-v["components_min"]
                                                              for v in calibration["per_letter"].values()),
                         holes_min=0, holes_max=max(v["holes_max"]-v["holes_min"]
                                                    for v in calibration["per_letter"].values()))
    component_tolerance = reference["components_max"] - reference["components_min"]
    hole_tolerance = reference["holes_max"] - reference["holes_min"]
    row["shape_deformation_failure"] = None if occluded or clipping else bool(
        row["area_fraction_error"] > reference["max_area_fraction_tolerance"]
        or alignment["aligned_mask_iou"] < calibration["aligned_iou_floor"]
        or alignment["missing_area_fraction"] > calibration["missing_extra_fraction_tolerance"]
        or alignment["extra_area_fraction"] > calibration["missing_extra_fraction_tolerance"]
        or alignment["aligned_contour_error_px"] > calibration["contour_tolerance_px"]
        or row["extra_fragments"] > component_tolerance
        or row["missing_components"] > component_tolerance or row["holes_difference"] > hole_tolerance)
    return row


def _stable_seed(key, base):
    return (int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) + int(base)) % 2**32


def native_sampler_predictor(model, source, *, sampler_config=None):
    """One unchanged upstream sampler call. No postprocessing is permitted."""
    from .gpc_world_model import world_model_spec
    import numpy as np
    import torch
    sampler = source.DiffusionSampler(model, source.DiffusionSamplerConfig(
        **(sampler_config or world_model_spec()["sampler"])))
    device = next(model.parameters()).device
    def predict(history, actions, seed):
        history = np.asarray(history, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if history.shape != (4, 3, 96, 96) or actions.shape != (4, 2):
            raise ValueError("Native sampler requires four RGB frames and four absolute actions")
        if not np.isfinite(actions).all() or actions.min() < 0 or actions.max() > 511:
            raise ValueError("Absolute actions outside [0,511]")
        devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            image = torch.as_tensor(history.copy(), device=device)[None]
            action = torch.as_tensor(actions.copy(), device=device)[None] / 511 * 2 - 1
            result, _ = sampler.sample(image, action)
        return result[0].detach().float().cpu().numpy()
    return predict


def validate_native_batch_inputs(histories, actions, *, allow_generated_history=False):
    """Real observations stay [0,1]; finite generated RGB keeps its raw values."""
    import numpy as np
    histories, actions = np.asarray(histories), np.asarray(actions)
    if type(allow_generated_history) is not bool:
        raise ValueError("Generated-history provenance must be an explicit bool")
    if (histories.ndim != 5 or histories.shape[1:] != (4,3,96,96)
            or actions.shape != (len(histories),4,2) or not len(histories)):
        raise ValueError("Expected nonempty batch of native RGB histories/actions")
    if (not np.isfinite(histories).all() or not np.isfinite(actions).all()
            or actions.min() < 0 or actions.max() > 511):
        raise ValueError("Nonfinite history or invalid absolute actions")
    if not allow_generated_history and (histories.min() < 0 or histories.max() > 1):
        raise ValueError("Real observed RGB must remain in [0,1]")
    return histories, actions


def native_sampler_batch_predictor(model, source, *, sampler_config=None):
    """Actual batched upstream sampling with a declared shared batch RNG seed."""
    from .gpc_world_model import world_model_spec
    import numpy as np
    import torch
    sampler = source.DiffusionSampler(model, source.DiffusionSamplerConfig(
        **(sampler_config or world_model_spec()["sampler"])))
    device = next(model.parameters()).device
    def predict(histories, actions, *, batch_seed=5478, allow_generated_history=False):
        histories = np.asarray(histories, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        validate_native_batch_inputs(histories, actions, allow_generated_history=allow_generated_history)
        if type(batch_seed) is not int or not 0 <= batch_seed < 2**32:
            raise ValueError("Batch seed must be uint32")
        devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(batch_seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(batch_seed)
            image = torch.as_tensor(histories.copy(), device=device)
            action = torch.as_tensor(actions.copy(), device=device)/511*2-1
            result, _ = sampler.sample(image, action)
        # Return unmodified sampler RGB. Per-example finite/range failure handling
        # belongs to the evaluation orchestrator, not a clipping/repair operation.
        return result.detach().float().cpu().numpy()
    return predict


class NativePredictor:
    """Raw native-GPC callable shared with the viewer; it never knows a letter."""

    def __init__(self, source_root, init_checkpoint, checkpoint=None, device="cuda"):
        from .native_rgb_training import construct_model, load_checkpoint_payload
        self.model, self.source = construct_model(source_root, init_checkpoint,
                                                 device=device, trainable=False)
        if checkpoint is not None:
            payload = load_checkpoint_payload(checkpoint)
            from .gpc_world_model import world_model_spec
            if (payload.get("kind") != "native_gpc_rgb_training.v1"
                    or payload.get("identity", {}).get("initialization") != world_model_spec()["checkpoint"]):
                raise ValueError("Expected a native RGB checkpoint descended from the pinned original release")
            self.model.load_state_dict(payload["model"], strict=True)
            del payload
        self.model.eval().requires_grad_(False)
        self._predict = native_sampler_predictor(self.model, self.source)
        self._batch_predict = native_sampler_batch_predictor(self.model, self.source)

    def __call__(self, history, absolute_actions, seed=5478):
        if self._predict is None:
            raise RuntimeError("NativePredictor has been closed")
        return self._predict(history, absolute_actions, seed)

    def batch_predict(self, histories, absolute_actions, *, batch_seed=5478, allow_generated_history=False):
        """True sampler batch; frozen/fine-tuned comparisons share shape/seed.

        Noise is paired by fixed batch membership and shared batch seed, not by
        claiming equality with older per-window sequential draws. Freeze batch
        size before evaluation; resource checks must establish that it fits.
        """
        if self._batch_predict is None:
            raise RuntimeError("NativePredictor has been closed")
        return self._batch_predict(histories, absolute_actions, batch_seed=batch_seed,
                                   allow_generated_history=allow_generated_history)

    def close(self):
        self._predict = None
        self._batch_predict = None
        self.model = self.source = None


def rollout_rgb(predictor, history, action_sequence, *, horizon, seed):
    """Actions contain three executed controls followed by proposed controls.

    No future images/labels are accepted by this function. Every sampler RGB is
    inserted directly; segmentation/alignment functions are deliberately absent.
    """
    import numpy as np
    history = np.stack([_rgb(frame) for frame in np.asarray(history)]).copy()
    actions = np.asarray(action_sequence, dtype=np.float32)
    if history.shape != (4, 3, 96, 96) or actions.shape != (horizon+3, 2):
        raise ValueError("Misaligned history/actions for RGB rollout")
    for step in range(horizon):
        prediction = _rgb(predictor(history.copy(), actions[step:step+4].copy(),
                                    (int(seed)+step) % 2**32))
        yield prediction.copy()
        history = np.concatenate((history[1:], prediction[None]), axis=0)


def select_windows(dataset, *, windows_per_letter=4, horizons=(1, 4), allow_unseen=False):
    """Deterministic uniform windows; inspect metadata before any array access."""
    import numpy as np
    split = dataset.split
    if split in ("unseen_test", "seen_test") and not allow_unseen:
        raise ValueError("Local final/unseen-letter evaluation is prohibited")
    permitted = set(UNSEEN_LETTERS if split == "unseen_test" else TRAIN_LETTERS)
    if split not in ("seen_val", "seen_test", "unseen_test"):
        raise ValueError("Evaluation requires seen_val or an explicitly enabled final test")
    by_letter = defaultdict(list)
    maximum = max(horizons)
    for index, anchor in dataset.windows:
        entry = dataset.entries[index]
        if entry["shape"] not in permitted:
            raise ValueError("Unexpected letter in evaluation split")
        if entry.get("split", split) != split:
            raise ValueError("Trajectory split differs from dataset split")
        if anchor >= 3 and anchor+maximum <= int(entry["controls"]):
            by_letter[entry["shape"]].append((index, anchor))
    selected = []
    for letter in sorted(by_letter):
        pool = sorted(by_letter[letter])
        count = min(windows_per_letter, len(pool))
        selected.extend(pool[int(i)] for i in np.linspace(0, len(pool)-1, count, dtype=int))
    if not selected:
        raise ValueError("No evaluation windows have a complete requested horizon")
    return selected


def _groups(arrays, anchor, horizon, truth_mask):
    import numpy as np
    states = arrays["states"][anchor:anchor+horizon+1]
    translation = float(np.linalg.norm(states[-1, 2:4] - states[0, 2:4]))
    angle = abs(math.atan2(math.sin(states[-1, 4]-states[0, 4]), math.cos(states[-1, 4]-states[0, 4])))
    near_edge = bool(truth_mask[:8].any() or truth_mask[-8:].any()
                     or truth_mask[:, :8].any() or truth_mask[:, -8:].any())
    return dict(contact=bool(np.asarray(arrays["contact"])[anchor:anchor+horizon].any()),
                rotation=angle >= .025, boundary=near_edge,
                stationary=translation <= .5 and angle <= .005,
                truth_translation_px=translation, truth_rotation_deg=math.degrees(angle))


def _fatal(error):
    message = str(error).lower()
    return isinstance(error, MemoryError) or any(s in message for s in
        ("out of memory", "device-side assert", "illegal memory access", "cuda error"))


def _summary(rows):
    import numpy as np
    metrics = ("rgb_mse", "object_position_error_px", "pusher_position_error_px", "angle_error_deg",
               "motion_error_px", "raw_mask_iou", "raw_contour_error_px", "area_fraction_error",
               "aligned_mask_iou", "full_object_mask_iou", "missing_area_fraction", "extra_area_fraction",
               "aligned_contour_error_px", "extra_fragments", "missing_components", "holes_difference")
    valid = [r for r in rows if r["status"] == "ok"]
    result = dict(total=len(rows), valid=len(valid), failed=len(rows)-len(valid),
                  failure_rate=(len(rows)-len(valid))/len(rows) if rows else None,
                  failure_reasons=dict(Counter(r["reason"] for r in rows if r["status"] != "ok")),
                  shape_available=sum(r.get("shape_measurement_available", False) for r in valid),
                  shape_deformation_failures=sum(r.get("shape_deformation_failure") is True for r in valid),
                  shape_unavailable=sum(r.get("shape_deformation_failure") is None for r in valid),
                  numeric_scope="conditional on valid measurement; all failures remain in total and qualification")
    for metric in metrics:
        values = [r[metric] for r in valid if r.get(metric) is not None]
        result[metric] = dict(count=len(values), mean=float(np.mean(values)) if values else None,
                              p95=float(np.percentile(values, 95)) if values else None)
    return result


def summarize_rows(rows):
    """Per-letter/subset counts never discard failed predictions."""
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        summary[method] = {}
        for horizon in sorted({r["horizon"] for r in rows}):
            chosen = [r for r in rows if r["method"] == method and r["horizon"] == horizon]
            per_letter = {letter: _summary([r for r in chosen if r["letter"] == letter])
                          for letter in sorted({r["letter"] for r in chosen})}
            macro = {}
            for metric in ("object_position_error_px", "angle_error_deg", "motion_error_px", "raw_mask_iou"):
                values = [stats[metric]["mean"] for stats in per_letter.values()
                          if stats[metric]["mean"] is not None]
                macro[metric] = dict(letters_with_valid_measurement=len(values),
                    mean_of_letter_means=sum(values)/len(values) if values else None)
            summary[method][str(horizon)] = dict(overall=_summary(chosen),
                per_letter=per_letter, macro=macro,
                subsets={name: _summary([r for r in chosen if r.get(name)])
                         for name in ("contact", "rotation", "boundary", "stationary")})
    return summary


def qualification(rows, *, improvement=.20):
    """Paired finite rows plus an explicit all-denominator failure requirement."""
    import numpy as np
    results = {}
    for method in sorted({r["method"] for r in rows} - {"hold"}):
        results[method] = {}
        for horizon in sorted({r["horizon"] for r in rows}):
            model = [r for r in rows if r["method"] == method and r["horizon"] == horizon]
            hold = {r["window"]: r for r in rows if r["method"] == "hold" and r["horizon"] == horizon}
            checks = {}
            for subset in ("contact", "rotation", "boundary"):
                chosen = [r for r in model if r.get(subset)]
                pairs = [(r, hold[r["window"]]) for r in chosen if r["status"] == "ok"
                         and r["window"] in hold and hold[r["window"]]["status"] == "ok"]
                baseline = float(np.mean([b["motion_error_px"] for _, b in pairs])) if pairs else None
                error = float(np.mean([a["motion_error_px"] for a, _ in pairs])) if pairs else None
                gain = (baseline-error)/baseline if baseline is not None and baseline > 1e-8 else None
                checks[subset] = dict(total=len(chosen), paired_valid=len(pairs),
                    hold_motion_error_px=baseline, model_motion_error_px=error,
                    relative_improvement=gain, required_relative_improvement=improvement,
                    passed=bool(chosen and len(pairs) == len(chosen) and gain is not None and gain >= improvement))
            shape = [r for r in model if r["status"] == "ok" and r.get("shape_measurement_available")]
            stationary = [r for r in model if r.get("stationary")]
            stationary_valid = [r for r in stationary if r["status"] == "ok"]
            drift = float(np.mean([r["object_position_error_px"] for r in stationary_valid])) if stationary_valid else None
            results[method][str(horizon)] = dict(subsets=checks,
                motion_passed=all(item["passed"] for item in checks.values()),
                prediction_failures=sum(r["status"] != "ok" for r in model),
                shape_assessable=len(shape), shape_total=len(model),
                shape_failures=sum(r.get("shape_deformation_failure") is True for r in shape),
                stationary=dict(total=len(stationary), valid=len(stationary_valid),
                    mean_position_error_px=drift, tolerance_px=WORLD_SCALE,
                    passed=bool(stationary and len(stationary_valid)==len(stationary)
                                and drift <= WORLD_SCALE)),
                rigidity_guaranteed=False)
    return results


def _coverage_metadata(report, rows, split):
    expected = set(UNSEEN_LETTERS if split == "unseen_test" else TRAIN_LETTERS)
    present = {row["letter"] for row in rows}
    report.update(expected_letters=sorted(expected), evaluated_letters=sorted(present),
                  missing_letters=sorted(expected-present), all_letter_coverage_complete=present == expected,
                  unclassified_failed_rows=sum(row["status"] != "ok" and "contact" not in row for row in rows))
    if expected-present:
        for method in report["qualification"].values():
            for horizon in method.values():
                horizon["motion_passed"] = False
                horizon["missing_letter_coverage"] = sorted(expected-present)


def _gallery(output, candidates, limit):
    from PIL import Image, ImageDraw
    galleries = []
    root = Path(output)/"gallery"
    if not candidates or limit <= 0:
        return galleries
    root.mkdir(exist_ok=True)
    for index, candidate in enumerate(sorted(candidates, key=lambda x: x[0], reverse=True)[:limit]):
        _, row, previous, prediction, truth = candidate
        canvas = Image.new("RGB", (96*3, 118), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (label, image) in enumerate((("observed", previous), ("raw prediction", prediction), ("truth", truth))):
            draw.text((column*96+2, 1), label, fill="black")
            if image is not None:
                import numpy as np
                pixels = np.rint(_rgb(image).transpose(1, 2, 0)*255).astype(np.uint8)
                canvas.paste(Image.fromarray(pixels), (column*96, 20))
            else:
                draw.text((column*96+2, 48), "missing/failed", fill="red")
        name = f"{index:02d}-{row['method']}-{row['letter']}-h{row['horizon']}.png"
        canvas.save(root/name)
        galleries.append(dict(file="gallery/"+name, source_row=row))
    _write_json(root/"sources.json", galleries)
    return galleries


def evaluate_windows(dataset, predictors, *, calibration, horizons=(1, 4),
                     windows_per_letter=4, seed=5478, output=None,
                     allow_unseen=False, gallery_limit=8, batch_size=1,
                     _return_gallery=False):
    """Bounded paired evaluation; callbacks receive images/actions only."""
    import numpy as np
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not horizons or min(horizons) < 1 or max(horizons) > 16:
        raise ValueError("Evaluation horizons must be in [1,16]")
    if "hold" in predictors:
        raise ValueError("hold is constructed internally")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation batch size must be a positive frozen integer")
    if output is not None and (Path(output)/"report.json").exists():
        raise FileExistsError("Evaluation output already exists; do not rerun an inspected split")
    if batch_size > 1:
        return _evaluate_batched(dataset, predictors, calibration=calibration,
            horizons=horizons, windows_per_letter=windows_per_letter, seed=seed,
            output=output, allow_unseen=allow_unseen, gallery_limit=gallery_limit,
            batch_size=batch_size)
    selected = select_windows(dataset, windows_per_letter=windows_per_letter,
                              horizons=horizons, allow_unseen=allow_unseen)
    methods = ["hold", *predictors]
    rows, gallery, fatal_methods = [], [], {}
    for entry_index, anchor in selected:
        entry = dataset.entries[entry_index]
        identifier = entry.get("trajectory_id", entry.get("file", str(entry_index)))
        key = f"{identifier}:{anchor}"
        letter = entry["shape"]
        # Every selected window exists in every method/horizon denominator even
        # when loading, rendering, scoring, or a device operation fails.
        records = {(method, h): dict(window=key, letter=letter, trajectory_id=identifier,
                    anchor=anchor, method=method, horizon=h, status="failed", reason="not_evaluated")
                   for method in methods for h in horizons}
        try:
            loaded_entry, arrays = dataset.trajectory(entry_index)
            if loaded_entry["shape"] != letter:
                raise ValueError("Trajectory identity changed")
            history = np.stack([_rgb(dataset.render_state(entry, state))
                                for state in arrays["states"][anchor-3:anchor+1]])
            truth, full_masks = {}, {}
            for h in horizons:
                if hasattr(dataset, "render_layers"):
                    layers = dataset.render_layers(entry, arrays["states"][anchor+h])
                    truth[h] = _rgb(layers["rgb"])
                    full_masks[h] = layers["full_object_mask"]
                else:
                    truth[h] = _rgb(dataset.render_state(entry, arrays["states"][anchor+h]))
            for h in horizons:
                mask, _ = palette_masks(truth[h], calibration)
                groups = _groups(arrays, anchor, h, mask)
                valid_truth = bool(np.asarray(arrays["valid_transition"])[anchor:anchor+h].all())
                for method in methods:
                    records[method, h].update(groups)
                    if not valid_truth:
                        records[method, h]["reason"] = "invalid_truth_transition"
            actions = np.asarray(arrays["actions"])[anchor-3:anchor+max(horizons)]
            for method in methods:
                predicted = {}
                error = None
                if method in fatal_methods:
                    error = "device_unavailable_after_fatal: " + fatal_methods[method]
                elif method == "hold":
                    predicted = {h: history[-1].copy() for h in horizons}
                else:
                    try:
                        for step, image in enumerate(rollout_rgb(predictors[method], history, actions,
                                horizon=max(horizons), seed=_stable_seed(key, seed)), start=1):
                            if step in horizons:
                                predicted[step] = image
                    except Exception as exc:
                        error = type(exc).__name__ + ": " + str(exc)[:300]
                        if _fatal(exc):
                            fatal_methods[method] = error
                for h in horizons:
                    row = records[method, h]
                    if row["reason"] == "invalid_truth_transition":
                        continue
                    if h not in predicted:
                        row["reason"] = "prediction_error: " + (error or "missing")
                    else:
                        try:
                            measurement = score_rgb(predicted[h], truth[h], calibration,
                                                    letter=letter, truth_full_mask=full_masks.get(h))
                            row.update(measurement, status="ok", reason="")
                        except Exception as exc:
                            row["reason"] = "score_error: " + type(exc).__name__ + ": " + str(exc)[:300]
                    priority = 1e9 if row["status"] != "ok" else row["motion_error_px"]
                    gallery.append((priority, dict(row), history[-1].copy(), predicted.get(h), truth[h]))
                    # A gallery stores at most a small fixed number of RGBs.
                    gallery.sort(key=lambda item: item[0], reverse=True)
                    del gallery[max(0, gallery_limit):]
        except Exception as exc:
            for row in records.values():
                row.update(status="failed", reason="window_error: " + type(exc).__name__ + ": " + str(exc)[:300])
        rows.extend(records.values())
    report = dict(metrics_version=METRICS_VERSION, split=dataset.split,
                  windows=len(selected), horizons=list(horizons), methods=methods,
                  calibration_sha256=calibration.get("sha256"),
                  inference="unchanged native sampler RGB feedback; no labels, registration or repair",
                  metrics_scope="visible palette masks in 512 coordinates; offline alignment separates apparent motion and deformation",
                  unseen_evaluated=dataset.split == "unseen_test", future_truth_in_feedback=False,
                  inference_batch_size=1, noise_pairing="same per-window seed and action-aligned step",
                  fatal_methods=fatal_methods, summary=summarize_rows(rows), qualification=qualification(rows),
                  rows=rows)
    _coverage_metadata(report, rows, dataset.split)
    if _return_gallery:
        report["_gallery_candidates"] = gallery
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if (output/"report.json").exists():
            raise FileExistsError("Evaluation output already exists; do not overwrite an inspected split")
        _write_json(output/"calibration.json", calibration)
        fields = sorted({key for row in rows for key in row})
        with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report["gallery"] = _gallery(output, gallery, gallery_limit)
        _write_json(output/"report.json", report)
    return report


def _evaluate_batched(dataset, predictors, *, calibration, horizons,
                      windows_per_letter, seed, output, allow_unseen,
                      gallery_limit, batch_size):
    """Bounded batches: original RGB feedback on GPU, scoring separately on CPU."""
    import numpy as np
    for name, predictor in predictors.items():
        if not callable(getattr(predictor, "batch_predict", None)):
            raise ValueError("True batched evaluation requires batch_predict: " + name)
    selected = select_windows(dataset, windows_per_letter=windows_per_letter,
                              horizons=horizons, allow_unseen=allow_unseen)
    rows, gallery, fatal_methods, batches = [], [], {}, []
    maximum = max(horizons)
    class WindowSubset:
        entries = dataset.entries
        split = dataset.split
        def __init__(self, windows):
            self.windows = windows
        def trajectory(self, index):
            return dataset.trajectory(index)
        def render_state(self, entry, state):
            return dataset.render_state(entry, state)
    if hasattr(dataset, "render_layers"):
        WindowSubset.render_layers = lambda self, entry, state: dataset.render_layers(entry, state)
    for start in range(0, len(selected), batch_size):
        chunk = selected[start:start+batch_size]
        loaded, failures = [], {}
        for index, anchor in chunk:
            entry = dataset.entries[index]
            identifier = entry.get("trajectory_id", entry.get("file", str(index)))
            key = f"{identifier}:{anchor}"
            base_seed = _stable_seed(key, seed)
            try:
                _, arrays = dataset.trajectory(index)
                history = np.stack([_rgb(dataset.render_state(entry, state))
                                    for state in arrays["states"][anchor-3:anchor+1]])
                actions = np.asarray(arrays["actions"])[anchor-3:anchor+maximum].astype(np.float32)
                if history.shape != (4,3,96,96) or actions.shape != (maximum+3,2):
                    raise ValueError("Invalid RGB rollout window")
                loaded.append(dict(key=key, seed=base_seed, history=history, actions=actions))
            except Exception as exc:
                failures[base_seed] = exc
        shared_seed = _stable_seed("batch:"+"|".join(row["key"] for row in loaded), seed)
        batches.append(dict(start=start, windows=[row["key"] for row in loaded],
                            batch_seed=shared_seed, requested_batch_size=batch_size))
        lookups = {}
        for method, predictor in predictors.items():
            values = {}
            active = list(range(len(loaded)))
            histories = [row["history"].copy() for row in loaded]
            if method in fatal_methods:
                active = []
            for step in range(maximum):
                if not active:
                    break
                try:
                    rgb_batch = predictor.batch_predict(
                        np.stack([histories[i] for i in active]),
                        np.stack([loaded[i]["actions"][step:step+4] for i in active]),
                        batch_seed=(shared_seed+step) % 2**32)
                    rgb_batch = np.asarray(rgb_batch)
                    if rgb_batch.shape != (len(active),3,96,96):
                        raise ValueError("Native batched sampler returned the wrong shape")
                except Exception as exc:
                    for i in active:
                        for later in range(step, maximum):
                            values[(loaded[i]["seed"]+later) % 2**32] = exc
                    if _fatal(exc):
                        fatal_methods[method] = type(exc).__name__+": "+str(exc)[:300]
                    break
                next_active = []
                for i, raw in zip(active, rgb_batch):
                    try:
                        raw = _rgb(raw)
                        address = (loaded[i]["seed"]+step) % 2**32
                        if address in values:
                            raise ValueError("Evaluation seed collision; no ambiguous lookup allowed")
                        values[address] = raw.copy()
                        histories[i] = np.concatenate((histories[i][1:], raw[None]), axis=0)
                        next_active.append(i)
                    except Exception as exc:
                        for later in range(step, maximum):
                            values[(loaded[i]["seed"]+later) % 2**32] = exc
                active = next_active
            def lookup(history, actions, call_seed, *, cache=values, name=method):
                value = cache.get(call_seed)
                if value is None:
                    if name in fatal_methods:
                        raise RuntimeError("device unavailable after fatal: "+fatal_methods[name])
                    raise RuntimeError("Window unavailable in native RGB batch")
                if isinstance(value, Exception):
                    raise value
                return value
            lookups[method] = lookup
        report = evaluate_windows(WindowSubset(chunk), lookups, calibration=calibration,
            horizons=horizons, windows_per_letter=max(windows_per_letter, len(chunk)),
            seed=seed, allow_unseen=allow_unseen, gallery_limit=gallery_limit,
            _return_gallery=True)
        rows.extend(report["rows"])
        gallery.extend(report.pop("_gallery_candidates"))
        gallery.sort(key=lambda candidate: candidate[0], reverse=True)
        del gallery[max(0, gallery_limit):]
    report = dict(metrics_version=METRICS_VERSION, split=dataset.split,
        windows=len(selected), horizons=list(horizons), methods=["hold", *predictors],
        calibration_sha256=calibration.get("sha256"),
        inference="true batched unchanged native sampler, generated RGB fed back directly",
        inference_batch_size=batch_size,
        noise_pairing="fixed ordered batch membership/shared batch seed and shape for both models; not sequential per-window noise parity",
        batches=batches, unseen_evaluated=dataset.split == "unseen_test",
        future_truth_in_feedback=False, fatal_methods=fatal_methods,
        summary=summarize_rows(rows), qualification=qualification(rows), rows=rows)
    _coverage_metadata(report, rows, dataset.split)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output/"calibration.json", calibration)
        fields = sorted({key for row in rows for key in row})
        with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report["gallery"] = _gallery(output, gallery, gallery_limit)
        _write_json(output/"report.json", report)
    return report


def action_response_probe(predictor, image, pusher_xy, calibration, *, seed=5478):
    """Static image/action counterfactual, explicitly not a contact-success test."""
    import numpy as np
    image = _rgb(image)
    center = np.asarray(pusher_xy, dtype=np.float32)
    history = np.repeat(image[None], 4, axis=0)
    past = np.repeat(center[None], 3, axis=0)
    cases, outputs = {}, []
    for name, offset in (("left", (-32, 0)), ("right", (32, 0)), ("up", (0, -32)), ("down", (0, 32))):
        target = np.clip(center + offset, 0, 511)
        try:
            result = _rgb(predictor(history.copy(), np.concatenate((past, target[None])), seed))
            obj, pusher = palette_masks(result, calibration)
            cases[name] = dict(status="ok", target=target.tolist(),
                               object_centroid_world=None if _centroid(obj) is None else (_centroid(obj)*WORLD_SCALE).tolist(),
                               pusher_centroid_world=None if _centroid(pusher) is None else (_centroid(pusher)*WORLD_SCALE).tolist())
            outputs.append(result)
        except Exception as exc:
            cases[name] = dict(status="failed", reason=type(exc).__name__+": "+str(exc)[:300])
    pairwise = [float(((a-b)**2).mean()) for i, a in enumerate(outputs) for b in outputs[i+1:]]
    return dict(kind="static_rgb_action_response", cases=cases,
                mean_pairwise_rgb_mse=float(np.mean(pairwise)) if pairwise else None,
                observed_action_response=bool(pairwise and max(pairwise) > 1e-7),
                physical_contact_success_test=False,
                interpretation="different RGB responses are a conditioning sanity check, not verified object dynamics")


def evaluate(config, output, data_dir, *, source_root, init_checkpoint, checkpoint=None,
             split="seen_val", horizons=None, allow_unseen=False, predictors=None,
             calibration=None, physics_source_root=None, frozen_identity=None):
    """Load audited data and original/fine-tuned native GPC without any training."""
    from .native_rgb_data import NativeRGBDataset, verify_dataset
    if split in ("unseen_test", "seen_test") and not (allow_unseen and config.get("server", False)):
        raise ValueError("Final seen/unseen letters may be evaluated only in an explicitly enabled server run")
    if output is not None and (Path(output)/"report.json").exists():
        raise FileExistsError("Do not repeat an already inspected evaluation split")
    if split in ("seen_test", "unseen_test"):
        if checkpoint is None or calibration is None or frozen_identity is None:
            raise ValueError("Final tests require a frozen checkpoint/data/config/source/calibration identity")
        if isinstance(frozen_identity, (str, Path)):
            frozen_identity = json.loads(Path(frozen_identity).read_text(encoding="utf-8"))
        expected = final_protocol_identity(config, checkpoint, data_dir, calibration)
        if frozen_identity != expected:
            raise ValueError("Final-test identity differs from the predeclared frozen protocol")
    manifest, audit = verify_dataset(data_dir, require_gate=True)
    options = config.get("evaluation", {})
    dataset = NativeRGBDataset(data_dir, split=split,
        cache_bytes=int(options.get("cache_bytes", 32*1024**2)),
        source_root=physics_source_root or config.get("physics_source_root"))
    if isinstance(calibration, (str, Path)):
        calibration = json.loads(Path(calibration).read_text(encoding="utf-8"))
    if calibration is None:
        if split != "seen_val":
            raise ValueError("Final evaluation requires the previously frozen seen-renderer calibration")
        calibration = calibrate_renderer(dataset)
    calibration = validate_calibration(calibration)
    supplied_hash = calibration["sha256"]
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        _write_json(Path(output)/"calibration.json", calibration)
    models = []
    if predictors is None:
        frozen = NativePredictor(source_root, init_checkpoint, device=config.get("device", "cuda"))
        models.append(frozen)
        predictors = {"frozen_gpc": frozen}
        if checkpoint is not None:
            trained = NativePredictor(source_root, init_checkpoint, checkpoint,
                                      device=config.get("device", "cuda"))
            models.append(trained)
            predictors["fine_tuned"] = trained
    try:
        report = evaluate_windows(dataset, predictors, calibration=calibration,
            horizons=tuple(horizons or options.get("horizons", [1, 4])),
            windows_per_letter=int(options.get("windows_per_letter", 4)),
            seed=int(config.get("seed", 5478)), output=output,
            allow_unseen=allow_unseen, gallery_limit=int(options.get("gallery_limit", 8)),
            batch_size=int(options.get("batch_size", 1)))
        if options.get("action_probe", True):
            import numpy as np
            first_entry, first_arrays = dataset.trajectory(0)
            state = first_arrays["states"][3]
            static_image = dataset.render_state(first_entry, state)
            report["action_response"] = {}
            for name, predictor in predictors.items():
                if name in report["fatal_methods"]:
                    report["action_response"][name] = dict(status="not_retried_after_fatal",
                                                           physical_contact_success_test=False)
                else:
                    report["action_response"][name] = action_response_probe(
                        predictor, static_image, np.asarray(state[:2]), calibration,
                        seed=int(config.get("seed", 5478)))
    finally:
        for model in models:
            model.close()
    report["data_manifest_sha256"] = audit.get("manifest_sha256", audit.get("dataset_manifest_sha256"))
    report["data_letters"] = sorted({entry["shape"] for entry in dataset.entries})
    report["checkpoint"] = None if checkpoint is None else str(Path(checkpoint))
    report["checkpoint_sha256"] = None if checkpoint is None else _file_sha256(checkpoint)
    report["evaluation_source_sha256"] = _file_sha256(__file__)
    report["frozen_final_identity"] = frozen_identity
    report["evaluation_protocol"] = dict(options=options, seed=int(config.get("seed",5478)),
        requested_horizons=list(horizons or options.get("horizons",[1,4])),
        calibration_sha256=supplied_hash, split=split, original_initialization_only=True)
    report["evaluation_protocol_sha256"] = hashlib.sha256(json.dumps(
        report["evaluation_protocol"], sort_keys=True).encode()).hexdigest()
    if output is not None:
        _write_json(Path(output)/"report.json", report)
    return report
