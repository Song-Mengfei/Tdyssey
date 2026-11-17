"""Slice labels for the official T 16-direction one-step probe.

Contact counts and object translation come from the simulator. They are
evaluation labels, not model inputs.
"""


def world_side(target, object_position):
    """World left/right of the object CoG. Equal x counts as right."""
    return "left" if target[0] < object_position[0] else "right"


def worse_object_iou(scores):
    return min(scores)


def not_better_than_still(worse_iou, still_iou):
    return worse_iou <= still_iou


def slice_stats(rows):
    if not rows:
        return {"count": 0, "min_worse_iou": None, "mean_worse_iou": None,
                "stable_missing": 0, "not_better_than_still": 0}
    worse = [row["worse_object_iou"] for row in rows]
    return {
        "count": len(rows),
        "min_worse_iou": min(worse),
        "mean_worse_iou": sum(worse) / len(worse),
        "stable_missing": sum(1 for row in rows if row["stable_missing"]),
        "not_better_than_still": sum(
            1 for row in rows if not_better_than_still(row["worse_object_iou"], row["still_object_iou"])),
    }


def summarize_cases(cases):
    """Overall and privileged left/right, contact/miss, and far/near slices."""
    return {
        "all": slice_stats(cases),
        "left": slice_stats([row for row in cases if row["world_side"] == "left"]),
        "right": slice_stats([row for row in cases if row["world_side"] == "right"]),
        "contact": slice_stats([row for row in cases if row["n_contacts"] > 0]),
        "no_contact": slice_stats([row for row in cases if row["n_contacts"] == 0]),
        "distance_20": slice_stats([row for row in cases if row["distance"] == 20.0]),
        "distance_150": slice_stats([row for row in cases if row["distance"] == 150.0]),
    }
