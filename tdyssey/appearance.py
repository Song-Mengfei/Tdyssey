"""Renderer-only palette control and explicit image-proxy metrics for T checks."""

SOURCE_OBJECT_RGB = (165, 28, 48)  # Pymunk draw options brighten this at rendering.
DISPLAY_OBJECT_RGB = ((119, 136, 153), (143, 163, 184), (165, 28, 48), (198, 34, 58))
DISPLAY_TARGET_RGB = ((144, 238, 144), (0, 0, 0))


def source_palette_image(env):
    """Render an auxiliary view without stepping/changing the policy observation.

    Simulator rendering access is required. This is an experimental control,
    not an image-only adapter or access to future object geometry for the model.
    """
    import numpy as np
    import pygame

    shapes = list(env.block.shapes)
    colors, goal, screen = [shape.color for shape in shapes], env.goal_color, env.screen
    try:
        for shape in shapes:
            shape.color = pygame.Color(*SOURCE_OBJECT_RGB)
        env.goal_color = pygame.Color("Black")
        image = env._render_frame(mode="rgb_array")
        # Match torchvision v2.ToDtype(scale=True), including float32 rounding.
        return np.moveaxis(image.astype(np.float32) * np.float32(1/255), -1, 0)
    finally:
        for shape, color in zip(shapes, colors):
            shape.color = color
        env.goal_color = goal
        env.screen = screen


def palette_mask(image, colors, tolerance=.15):
    import numpy as np
    return np.logical_or.reduce([
        np.linalg.norm(image - np.asarray(color)[:, None, None] / 255, axis=0) < tolerance
        for color in colors])


def mask_iou(first, second):
    union = (first | second).sum()
    return None if not union else float((first & second).sum() / union)


def centroid(mask):
    import numpy as np
    y, x = np.nonzero(mask)
    return None if len(x) < 12 else np.array([x.mean() + .5, y.mean() + .5]) * 512 / 96


def prediction_metrics(predicted, actual, latest, object_roi, tolerance=.15):
    """Score RGB error, palette-independent object masks, and target visibility.

    These masks use known renderer colors, never future geometry to extract a
    prediction. The geometry-derived ROI is an evaluation label only.
    """
    import numpy as np

    truth = palette_mask(actual, DISPLAY_OBJECT_RGB, tolerance)
    predicted_mask = palette_mask(predicted, DISPLAY_OBJECT_RGB, tolerance)
    target = palette_mask(actual, DISPLAY_TARGET_RGB, tolerance)
    predicted_target = palette_mask(predicted, DISPLAY_TARGET_RGB, tolerance)
    start = palette_mask(latest, DISPLAY_OBJECT_RGB, tolerance)
    start_center, actual_center, predicted_center = (centroid(mask) for mask in (start, truth, predicted_mask))
    error = (predicted - actual) ** 2
    return {
        "image_mse": float(error.mean()),
        "object_roi_mse": float(error[:, object_roi].mean()),
        "object_mask_iou": mask_iou(predicted_mask, truth),
        "object_centroid_error_pixels": None if actual_center is None or predicted_center is None
            else float(np.linalg.norm(predicted_center - actual_center)),
        "predicted_centroid_shift_pixels": None if predicted_center is None or start_center is None
            else float(np.linalg.norm(predicted_center - start_center)),
        "actual_centroid_shift_pixels": None if actual_center is None or start_center is None
            else float(np.linalg.norm(actual_center - start_center)),
        "predicted_object_pixels": int(predicted_mask.sum()),
        "target_mask_iou": mask_iou(predicted_target, target),
        "visible_target_pixels": int(target.sum()),
        "visible_target_recall": None if not target.any() else float((predicted_target & target).sum() / target.sum()),
    }
