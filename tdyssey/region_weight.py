"""Privileged T/pusher occupancy and frame-diff weights. Not model inputs.

These maps supervise or score pixels. The denoiser still sees only RGB and
actions. Geometry occupancy is simulator-derived and must be disclosed.
"""

from .collection import integer
from .envs.official import official_shape
from .geometry import finite
from .world_model_data import object_masks

DEFAULT_REGION_WEIGHT = 8
MOTION_THRESHOLD = 0.02
PUSHER_RADIUS = 15.0
IMAGE_SIZE = 96
WORLD = 512.0


def motion_mask(next_image, last_image, threshold=MOTION_THRESHOLD):
    """True where the next official frame differs from the last conditioning frame."""
    import numpy as np

    cutoff = finite(threshold, "motion threshold")
    if cutoff < 0:
        raise ValueError("motion threshold must be nonnegative")
    delta = np.abs(np.asarray(next_image, dtype=np.float64) - np.asarray(last_image, dtype=np.float64))
    if delta.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Motion masks expect 3x96x96 official frames")
    return np.mean(delta, axis=0) > cutoff


def pusher_mask(position, radius=PUSHER_RADIUS):
    """Official pusher disk at policy-image pixel centers, dilated one pixel."""
    import numpy as np

    reach = finite(radius, "pusher radius")
    if reach <= 0:
        raise ValueError("pusher radius must be positive")
    y, x = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE].astype(np.float64)
    x = (x + 0.5) * WORLD / IMAGE_SIZE
    y = (y + 0.5) * WORLD / IMAGE_SIZE
    px, py = position
    disk = (x - finite(px, "pusher x")) ** 2 + (y - finite(py, "pusher y")) ** 2 <= reach ** 2
    return _dilate(disk)


def object_occupancy(snapshot):
    """Dilated official object occupancy. Label only; not a network input."""
    pose = snapshot.object_body.pose
    masks = object_masks(official_shape(snapshot.shape),
                         [(pose.position[0], pose.position[1], pose.angle)])
    return masks[0, 0]


def window_region(before, after, next_image=None, last_image=None):
    """T and pusher at both step ends, plus optional RGB change."""
    region = object_occupancy(before) | object_occupancy(after)
    region = region | pusher_mask(before.pusher_body.pose.position)
    region = region | pusher_mask(after.pusher_body.pose.position)
    if next_image is not None and last_image is not None:
        region = region | motion_mask(next_image, last_image)
    return region


def window_useful(moved, n_contacts=0):
    """True when the predicted step moved the object or reported contact."""
    return bool(moved) or integer(n_contacts, "contacts", 0, 100000) > 0


def window_case_region(before, after, next_image=None, last_image=None, moved=False,
                       n_contacts=0):
    """Hit windows use motion and the next T; misses keep the still T. Table is out."""
    pusher = pusher_mask(before.pusher_body.pose.position) | pusher_mask(after.pusher_body.pose.position)
    if window_useful(moved, n_contacts):
        region = object_occupancy(after) | pusher
        if next_image is not None and last_image is not None:
            region = region | motion_mask(next_image, last_image)
        return region
    return object_occupancy(before) | object_occupancy(after) | pusher


def region_weights(region, weight=DEFAULT_REGION_WEIGHT):
    """Background stays 1. Region pixels use ``weight``. Empty regions stay uniform."""
    import numpy as np

    scale = integer(weight, "region weight", 1, 64)
    mask = np.asarray(region, dtype=bool)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Region mask must be 96x96")
    if not mask.any():
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return np.asarray(1.0 + (scale - 1.0) * mask, dtype=np.float32)


def foreground_weights(region):
    """Foreground is 1, table is 0. Empty masks stay uniform so EDM stays defined."""
    import numpy as np

    mask = np.asarray(region, dtype=bool)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Region mask must be 96x96")
    if not mask.any():
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return mask.astype(np.float32)


def _dilate(mask):
    import numpy as np

    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return np.logical_or.reduce([padded[row:row + IMAGE_SIZE, col:col + IMAGE_SIZE]
                                 for row in range(3) for col in range(3)])
