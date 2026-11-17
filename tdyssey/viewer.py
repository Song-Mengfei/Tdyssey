"""Optional desktop viewer for PushT RGB frames; uses existing pygame, not OpenCV GUI."""

import time


def presentation_frame(env, size=512):
    """Native physics rendering for people; preserve 96px policy observations."""
    previous_size, previous_screen = env.render_size, env.screen
    try:
        env.render_size = size
        return env._render_frame(mode="rgb_array")
    finally:
        env.render_size, env.screen = previous_size, previous_screen


class PushTViewer:
    def __init__(self, title="Tdyssey — PushT", hold_seconds=0, window_size=768):
        import pygame

        self.pygame, self.window = pygame, None
        self.closed, self.hold_seconds = False, hold_seconds
        self.window_size = window_size
        self.last_frame_time = None
        pygame.display.init()
        pygame.display.set_caption(title)

    def poll(self):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.closed = True
            elif event.type == pygame.VIDEORESIZE and self.window is not None:
                self.window = pygame.display.set_mode(event.size, pygame.RESIZABLE)
        return not self.closed

    def draw(self, image, caption, control_hz=None):
        pygame = self.pygame
        if self.closed:
            return False
        if self.window is None:
            height, width = image.shape[:2]
            scale = self.window_size / max(height, width)
            self.window = pygame.display.set_mode((round(width * scale), round(height * scale)), pygame.RESIZABLE)
        if control_hz and self.last_frame_time is not None:
            # Keep event handling responsive while pacing displayed control steps.
            end = self.last_frame_time + 1 / control_hz
            while time.monotonic() < end:
                if not self.poll():
                    return False
                time.sleep(min(0.01, max(0, end - time.monotonic())))
        if not self.poll():
            return False
        pygame.display.set_caption(caption)
        surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
        self.window.blit(pygame.transform.smoothscale(surface, self.window.get_size()), (0, 0))
        pygame.display.flip()
        self.last_frame_time = time.monotonic()
        return True

    def finish(self):
        # Bounded hold lets a completed short run remain visible. Closing the
        # window or pressing Escape ends it immediately; never wait indefinitely.
        until = time.monotonic() + self.hold_seconds
        while self.window is not None and not self.closed and time.monotonic() < until:
            self.poll()
            time.sleep(0.02)

    def close(self):
        self.pygame.display.quit()
