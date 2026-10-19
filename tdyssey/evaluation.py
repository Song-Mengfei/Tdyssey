"""Explicit geometric scores, separate from the inherited GPC reward."""

from dataclasses import dataclass
import math

from .geometry import Pose, Shape, convex_intersection, finite, polygon_area


@dataclass(frozen=True)
class GoalMetrics:
    coverage: float
    iou: float
    position_error: float
    angle_error: float


def goal_metrics(shape: Shape, actual: Pose, goal: Pose) -> GoalMetrics:
    intersection = sum(polygon_area(convex_intersection(a, b))
                       for a in shape.polygons(actual) for b in shape.polygons(goal))
    intersection = min(shape.area, max(0.0, intersection))
    return GoalMetrics(intersection / shape.area, intersection / (2 * shape.area - intersection),
                       math.dist(actual.position, goal.position),
                       shape.angle_error(actual.angle, goal.angle))


class SuccessTracker:
    """Count post-action control steps; require consecutive overlap before success.

    This is an additional project metric, not a replacement for GPC's score.
    """

    def __init__(self, threshold: float = 0.90, hold_steps: int = 5):
        threshold = finite(threshold, "threshold")
        if not 0 < threshold <= 1:
            raise ValueError("Threshold must be in (0, 1]")
        if isinstance(hold_steps, bool) or not isinstance(hold_steps, int) or hold_steps < 1:
            raise ValueError("Hold steps must be a positive integer")
        self.threshold, self.hold_steps = threshold, hold_steps
        self.control_steps = self.consecutive_steps = 0
        self.final_iou = self.max_iou = 0.0
        self.first_success_step = None

    def observe(self, iou: float) -> bool:
        iou = finite(iou, "IoU")
        if not 0 <= iou <= 1:
            raise ValueError("IoU must be in [0, 1]")
        self.control_steps += 1
        self.final_iou, self.max_iou = iou, max(self.max_iou, iou)
        self.consecutive_steps = self.consecutive_steps + 1 if iou >= self.threshold else 0
        if self.first_success_step is None and self.consecutive_steps >= self.hold_steps:
            self.first_success_step = self.control_steps
        return self.first_success_step is not None
