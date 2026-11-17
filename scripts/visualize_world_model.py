"""Click a pusher target and show the trained world model's next frame.

The left image is the live official T simulator. A click chooses a target but
does not move the object. Space executes that target, so the next prediction
uses the new real frame. Escape closes the window.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.gpc_denoiser_training import load_denoiser_sampler, predict_next_frame
from tdyssey.world_model_click import click_to_target, conditioning_window


def observation_frame(observation):
    import numpy as np

    image = np.asarray(observation["image"], dtype=np.float32)
    if image.shape != (3, 96, 96):
        raise ValueError("Expected an official 96x96 frame")
    return image


def display_frame(image):
    import numpy as np

    return np.rint(np.clip(image, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def panel_surface(pygame, image, size):
    surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
    return pygame.transform.smoothscale(surface, (size, size))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts/official-wm-T-hour-2026-10-23/model-30m/denoiser.pt")
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--panel", type=int, default=512)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"Missing trained denoiser: {args.checkpoint}")
    import numpy as np
    import pygame
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source

    sampler, device = load_denoiser_sampler(args.gpc_root, args.checkpoint)
    env = make_official_shape_env_class(load_official_source(args.source_root))(shape="T", legacy=True)
    pygame.display.init()
    pygame.font.init()
    size = args.panel
    window = pygame.display.set_mode((size * 2, size))
    font = pygame.font.SysFont(None, 28)
    env.seed(args.seed)
    observation = env.reset()
    frames = [observation_frame(observation)]
    executed = []
    target = None
    predicted = None
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and event.pos[0] < size:
                    target = click_to_target(event.pos[0], event.pos[1], size, size)
                    hold = tuple(float(value) for value in observation["agent_pos"])
                    images, actions = conditioning_window(frames, executed, target, hold)
                    predicted = predict_next_frame(sampler, np.stack(images), np.stack(actions), device, seed=args.seed)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and target is not None:
                    observation, _reward, _done, _info = env.step(np.asarray(target, dtype=np.float64))
                    frames.append(observation_frame(observation))
                    executed.append(target)
                    target = None
                    predicted = None
            current = display_frame(frames[-1])
            window.fill((255, 255, 255))
            window.blit(panel_surface(pygame, current, size), (0, 0))
            if predicted is not None:
                window.blit(panel_surface(pygame, display_frame(predicted), size), (size, 0))
            if target is not None:
                point = (round(target[0] / 512 * size), round(target[1] / 512 * size))
                pygame.draw.circle(window, (220, 40, 40), point, 8, 2)
                label = font.render(f"target {target[0]:.0f}, {target[1]:.0f}    Space executes    Esc closes", True, (20, 20, 20))
            else:
                label = font.render("Click the left image to choose a push target", True, (20, 20, 20))
            window.blit(label, (12, 12))
            pygame.display.set_caption("Official T | left real, right predicted next frame")
            pygame.display.flip()
            pygame.time.wait(20)
    finally:
        env.close()
        pygame.display.quit()


if __name__ == "__main__":
    main()
