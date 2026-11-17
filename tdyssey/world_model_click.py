"""Mouse-target conversion and the four-frame history used by the denoiser viewer."""


def click_to_target(x, y, width, height, workspace=512):
    """Map a click inside the displayed scene to a clipped workspace target."""
    if width <= 0 or height <= 0:
        raise ValueError("Display size must be positive")
    limit = workspace - 1
    horizontal = min(limit, max(0, x / width * workspace))
    vertical = min(limit, max(0, y / height * workspace))
    return (float(horizontal), float(vertical))


def conditioning_window(frames, past_actions, click, hold):
    """Four images and four targets. Missing history repeats the current scene and pusher position."""
    if not frames:
        raise ValueError("Prediction needs the current frame")
    images = list(frames)
    while len(images) < 4:
        images.insert(0, images[0])
    past = list(past_actions)
    while len(past) < 3:
        past.insert(0, hold)
    return images[-4:], past[-3:] + [click]
