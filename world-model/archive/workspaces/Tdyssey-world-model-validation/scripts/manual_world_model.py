"""Open the manual T/L world-model desktop demo; no training or automatic collection."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
from tdyssey.manual_world_model import ManualSession, PinnedPredictor, event, display_observation, OFFICIAL_DOMAIN, LEGACY_DOMAIN

WIDTH, HEIGHT, SIDE = 1000, 538, 472
PANELS = [pygame.Rect(16 + i * 496, 50, SIDE, SIDE) for i in range(2)]


def target_from_click(position):
    if not PANELS[0].collidepoint(position):
        return None
    return np.array([(position[0] - PANELS[0].x) * 512 / SIDE,
                     (position[1] - PANELS[0].y) * 512 / SIDE]).clip(0, 511)


class App:
    def __init__(self, output, seconds=1800, *, session=None, predictor=None,
                 role_labels=None, monitor_host=True, available_roles=None):
        self.output, self.started, self.seconds = output, time.monotonic(), seconds
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (output / "events.jsonl").open("x", encoding="utf-8")
        event(self.log, "opened", kind_of_session="manual_exploration_not_training_or_test_set",
              pid=os.getpid(), model_updates=0, maximum_seconds=seconds,
              image_domain=session.image_domain if session is not None else LEGACY_DOMAIN)
        self.session = session if session is not None else ManualSession()
        self.monitor_host = monitor_host
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tdyssey")
        self.font = pygame.font.SysFont("segoeui", 19)
        self.small = pygame.font.SysFont("segoeui", 16)
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.predictor = predictor if predictor is not None else PinnedPredictor()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self.busy = self.failed = self.closed = False
        self.status = "Click left to push"
        self.elapsed_prediction = None
        self.result = None
        self.video = None
        self.video_path = None
        self.video_frames = 0
        self.video_last = 0
        self.last_memory = 0
        self.peak_private = self.peak_ws = 0
        self.buttons = []
        role_labels = role_labels or {"frozen": "Initial", "adapted": "Adapted 500"}
        self.role_labels = dict(role_labels)
        self.available_roles = tuple(available_roles) if available_roles is not None else ("frozen","adapted")
        labels = [("T", "T"), ("L", "L"), (role_labels["frozen"], "frozen"),
                  (role_labels["adapted"], "adapted"), ("4 steps", "horizon"),
                  ("Reset", "reset"), ("Stop recording", "record")]
        x = 16
        for label, action in labels:
            if action in ("frozen","adapted") and action not in self.available_roles:
                continue
            width = max(48, self.small.size(label)[0] + 22)
            self.buttons.append((pygame.Rect(x, 10, width, 30), label, action))
            x += width + 8

    def _work(self):
        try:
            while True:
                request = self.requests.get()
                if request is None:
                    return
                try:
                    self.results.put((request, self.predictor.predict(request), None))
                except BaseException:
                    self.results.put((request, None, traceback.format_exc()))
        finally:
            self.predictor.close()

    def text(self, message, xy, color=(215, 223, 235), small=False):
        font = self.small if small else self.font
        self.window.blit(font.render(str(message), True, color), xy)

    def action(self, action):
        if action == "save":
            self.save()
            return
        if action == "record":
            self.record()
            return
        if self.busy or self.failed:
            return
        try:
            if action in ("T", "L", "reset"):
                self.session.reset(action if action in ("T", "L") else None)
                self.result = None
                self.status = ("T-trained weights; L is exploratory" if self.session.image_domain==OFFICIAL_DOMAIN
                               and self.session.shape=="L" else "Click left to push")
                if (self.session.role=="adapted"
                        and getattr(self,"role_labels",{}).get("adapted")=="Raw T+L"):
                    self.status = "Raw T+L; prediction quality under review"
                event(self.log, "reset", shape=self.session.shape, warmup_controls=3)
            elif action in ("frozen", "adapted"):
                if action not in getattr(self,"available_roles",("frozen","adapted")):
                    self.status = "Single model view"
                    return
                role_error = getattr(self.predictor, "role_error", lambda role: None)(action)
                if role_error:
                    self.status = role_error
                    return
                self.session.configure(role=action)
                self.result = None
                self.status = "Model selected"
                event(self.log, "checkpoint_selected", role=action)
            elif action == "horizon":
                if self.session.image_domain == OFFICIAL_DOMAIN:
                    self.status = "Official baseline: 1 step"
                    return
                self.session.configure(horizon=4 if self.session.horizon == 1 else 1)
                self.result = None
                self.status = f"{self.session.horizon} step(s) per click"
            elif action == "predict":
                request = self.session.request()
                self.busy = True
                self.status = "Predicting..."
                self.requests.put(request)
                event(self.log, "prediction_requested", role=request["role"], shape=request["shape"],
                      horizon=request["horizon"], target=request["target"].tolist(), seed=request["noise_seed"],
                      request_key=request["request_key"])
            elif action == "compare":
                role = "adapted" if self.session.role == "frozen" else "frozen"
                if role not in getattr(self,"available_roles",("frozen","adapted")):
                    self.status = "Single model view"
                    return
                role_error = getattr(self.predictor, "role_error", lambda role: None)(role)
                if role_error:
                    self.status = role_error
                    return
                request = self.session.request_for_comparison(role)
                self.busy = True
                self.status = "Comparing the same saved input..."
                self.requests.put(request)
                event(self.log, "comparison_requested", role=role, horizon=request["horizon"],
                      request_key=request["request_key"], physical_execution=False)
            elif action == "execute":
                self.result = self.session.execute()
                self.status = "Click left to push"
                if self.session.done:
                    self.status = "Success! Reset to continue"
                event(self.log, "executed", **self.result)
        except ValueError as exc:
            self.status = str(exc)

    def handle(self, item):
        if item.type == pygame.QUIT or (item.type == pygame.KEYDOWN and item.key == pygame.K_ESCAPE):
            self.closed = True
        elif item.type == pygame.KEYDOWN:
            actions = {pygame.K_t: "T", pygame.K_l: "L", pygame.K_1: "frozen", pygame.K_2: "adapted",
                       pygame.K_h: "horizon", pygame.K_p: "predict",
                       pygame.K_r: "reset", pygame.K_s: "save", pygame.K_v: "record",
                       pygame.K_c: "compare"}
            if item.key in actions:
                self.action(actions[item.key])
        elif item.type == pygame.MOUSEBUTTONDOWN and item.button == 1:
            for rect, _, action in self.buttons:
                if rect.collidepoint(item.pos):
                    self.action(action)
                    return
            target = target_from_click(item.pos)
            if target is not None and not self.busy and not self.failed:
                self.session.select(target)
                self.result = None
                self.action("predict")

    def poll(self):
        for item in pygame.event.get():
            self.handle(item)
            if self.closed:
                return
        if self.closed:
            return
        if time.monotonic() - self.started >= self.seconds:
            self.status, self.closed = "Session time limit reached", True
            return
        if not self.results.empty():
            request, result, error = self.results.get_nowait()
            self.busy = False
            if error:
                self.fail_prediction(request, error)
            else:
                raw, measurement, self.elapsed_prediction = result
                comparison = request.get("comparison_only", False)
                try:
                    if comparison:
                        self.session.accept_comparison(request, raw, measurement)
                    else:
                        self.session.accept(request, raw, measurement)
                except BaseException:
                    self.fail_prediction(request, traceback.format_exc(), raw)
                    return
                event(self.log, "predicted", role=request["role"], horizon=len(raw),
                      seconds=self.elapsed_prediction, measurement=measurement,
                      raw_min=float(raw.min()), raw_max=float(raw.max()), loads=self.predictor.loads,
                      checkpoint=self.predictor.assets["weights"][request["role"]],
                      request_key=request["request_key"], requested_horizon=request["horizon"],
                      trace=request.get("prediction_trace"), usage=self.predictor.usage(),
                      comparison_only=comparison)
                # One click = one prediction followed by its exact real action.
                # Failed/obsolete predictions never authorize physical stepping.
                if comparison:
                    self.status = "Same input compared; no extra push"
                    bundle = self.session.capture_case()
                    execution = bundle["metadata"]["execution"]
                    self.result = execution["result"] if execution is not None else None
                else:
                    self.action("execute")
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            self.status, self.closed = "Session time limit reached", True
        if getattr(self, "monitor_host", True) and os.name == "nt" and time.monotonic() - self.last_memory >= 1:
            from tdyssey.policy_memory import process_memory, system_memory
            self.last_memory = time.monotonic()
            process, system = process_memory(os.getpid()), system_memory()
            if not process["ok"] or not system["ok"]:
                raise RuntimeError("Cannot read manual-demo host resource usage")
            self.peak_private = max(self.peak_private, process["private_bytes"])
            self.peak_ws = max(self.peak_ws, process["working_set_bytes"])
            if self.peak_private > 3 * 1024**3 or self.peak_ws > 2 * 1024**3:
                raise MemoryError("Manual demo host-memory limit exceeded")
            if (system["physical"]["available_bytes"] < 1.5 * 1024**3
                    or system["commit"]["headroom_bytes"] < 3 * 1024**3):
                raise MemoryError("Manual demo stopped to preserve host RAM/commit headroom")

    def panel(self, index, image):
        rect = PANELS[index]
        pygame.draw.rect(self.window, (30, 40, 56), rect, border_radius=8)
        if image is not None:
            # Clip for display only. Stored predictions and AR feedback stay raw.
            pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(pixels.transpose(2, 1, 0))
            self.window.blit(pygame.transform.scale(surface, rect.size), rect)
        else:
            self.text("Click left to begin", (rect.x + 175, rect.y + 220), small=True)

    def draw(self):
        self.window.fill((15, 23, 36))
        for rect, label, action in self.buttons:
            active = action == self.session.shape or action == self.session.role
            pygame.draw.rect(self.window, (42, 107, 166) if active else (40, 53, 73), rect, border_radius=5)
            if action == "horizon":
                label = f"{self.session.horizon} step" + ("s" if self.session.horizon != 1 else "")
            elif action == "record":
                label = "Stop recording" if self.video is not None else "Record"
            self.text(label, (rect.x + 9, rect.y + 5), small=True)
        index = len(self.session.actual) - 1 if self.session.actual is not None else self.session.horizon - 1
        predicted = None if self.session.prediction is None else self.session.prediction[index]
        # Display both sides in the same palette; real physics/history stay raw.
        actual = (self.session.actual[-1] if self.session.actual is not None
                  else display_observation(self.session.observation["image"], self.session.image_domain))
        self.panel(0, actual)
        self.panel(1, predicted)
        if self.session.target is not None:
            point = (round(PANELS[0].x + self.session.target[0] * SIDE / 512),
                     round(PANELS[0].y + self.session.target[1] * SIDE / 512))
            pygame.draw.circle(self.window, (230, 70, 50), point, 7, 2)
        status_x = self.buttons[-1][0].right + 16 if self.buttons else 650
        status = self.status
        while status and self.small.size(status)[0] > WIDTH - status_x - 16:
            status = status[:-4] + "..." if len(status) > 4 else ""
        self.text(status, (status_x, 16), (255, 201, 104) if self.failed else (151, 168, 190), small=True)
        pygame.display.flip()
        if self.video is not None and time.monotonic() - self.video_last >= .1:
            import cv2
            frame = pygame.surfarray.array3d(self.window).transpose(1, 0, 2)
            self.video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.video_frames += 1
            self.video_last = time.monotonic()
            if self.video_path.stat().st_size >= 128 * 1024**2:
                self.record()
                self.status = "Recording saved (size limit)"

    def save(self):
        name = datetime.now().strftime("view-%H%M%S-%f")
        pygame.image.save(self.window, str(self.output / (name + ".png")))
        if self.session.prediction is not None:
            np.save(self.output / (name + "-prediction.npy"), self.session.prediction, allow_pickle=False)
        case = (self.session.save_case(self.output / (name + "-case"))
                if self.session.last_case_key is not None else None)
        event(self.log, "screenshot", path=name + ".png", role=self.session.role, shape=self.session.shape,
              result=self.result, measurement=self.session.measurement, captured_case=case)
        self.status = "Case and screenshot saved" if case is not None else "Screenshot saved"

    def fail_prediction(self, request, error, raw=None):
        self.failed = True
        (self.output / "error.txt").write_text(error, encoding="utf-8")
        self.save_failed_request(request, raw)
        event(self.log, "first_error", traceback=error, usage=self.predictor.usage(),
              trace=request.get("prediction_trace"))
        self.status = "Prediction failed. See error.txt"

    def save_failed_request(self, request, raw=None):
        """Keep the exact failed input and any returned prefix; never retries."""
        import json
        folder = self.output / datetime.now().strftime("failed-%H%M%S-%f")
        folder.mkdir(exist_ok=False)
        metadata = {key: value for key, value in request.items()
                    if key not in ("images", "past_actions", "target", "goal")}
        metadata.update(target=request["target"].tolist(), training_data=False,
                        accepted=False, usage=self.predictor.usage())
        partial = self.predictor.last_partial_predictions if raw is None else list(raw)
        np.savez_compressed(folder / "arrays.npz", history_raw=request["images"],
            past_actions=request["past_actions"],
            partial_prediction=(np.stack(partial) if partial else np.empty((0, 3, 96, 96), np.float32)))
        (folder / "request.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    def record(self):
        if self.video is not None:
            self.video.release()
            self.video = None
            event(self.log, "video_saved", path=self.video_path.name, frames=self.video_frames, fps=10)
            self.status = "Recording saved"
            return
        import cv2
        self.video_path = self.output / (datetime.now().strftime("manual-%H%M%S-%f") + ".mp4")
        self.video = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (WIDTH, HEIGHT))
        if not self.video.isOpened():
            self.video.release()
            self.video = None
            raise RuntimeError("Installed OpenCV cannot encode MP4; no install is attempted")
        self.video_frames, self.video_last = 0, 0
        self.status = "Recording..."

    def close(self):
        if self.video is not None:
            self.record()
        self.requests.put(None)
        self.worker.join(timeout=5)
        event(self.log, "closed", controls=self.session.controls, forecasts=self.session.forecasts,
              resets=self.session.resets, loads=self.predictor.loads, failed=self.failed,
              peak_private_bytes=self.peak_private if self.monitor_host else None,
              peak_working_set_bytes=self.peak_ws if self.monitor_host else None,
              host_memory_monitor="legacy Windows PSAPI" if self.monitor_host and os.name == "nt" else "not verified on this path",
              peak_cuda_allocated_bytes=self.predictor.peak_allocated,
              peak_cuda_reserved_bytes=self.predictor.peak_reserved,
              elapsed_seconds=time.monotonic() - self.started, worker_stopped=not self.worker.is_alive(),
              prediction_usage=self.predictor.usage(), training_updates=0)
        self.log.close()
        self.session.close()
        pygame.quit()
        if self.worker.is_alive():
            # A native call cannot be cancelled safely from another Python thread.
            # End this demo process only; the OS releases its CUDA context.
            os._exit(2)


def smoke(app):
    """Bounded real UI/controller validation: 2 serial models, T/L, H1/H4, 20 forecasts."""
    clock = pygame.time.Clock()
    initial_drawn = False
    for role in ("frozen", "adapted"):
        app.action(role)
        for shape in ("T", "L"):
            app.action(shape)
            for horizon in (1, 4):
                if app.session.horizon != horizon:
                    app.action("horizon")
                # Route through the same mouse and key handlers as manual use.
                point = (PANELS[0].x + SIDE // 2, PANELS[0].y + round(320 * SIDE / 512))
                app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point))
                app.draw()
                if not initial_drawn:
                    app.record()
                    initial_drawn = True
                before = app.session.controls
                app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
                while app.busy:
                    app.poll(); app.draw(); clock.tick(30)
                    if app.closed or time.monotonic() - app.started > 120:
                        raise TimeoutError("Manual smoke stopped or exceeded 120 seconds")
                if app.failed:
                    raise RuntimeError("Model preview failed; see error.txt")
                assert app.session.controls == before + horizon, "Click did not execute exactly once"
                assert app.result and app.result["executed_controls"] == horizon
                app.draw()
                app.save()
                event(app.log, "smoke_case_passed", role=role, shape=shape, horizon=horizon)
    app.record()
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert app.closed
    event(app.log, "smoke_passed", cases=8, forecasts=20, training_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--smoke", action="store_true", help="Run the explicit bounded real verification and exit")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error("Choose a session duration of 1..1800 seconds")
    output = args.output or ROOT / "artifacts" / "manual-world-model" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # Existing pinned assets only: no hub access or global cache writes.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(output.resolve() / "runtime-cache")
    os.environ["HF_HUB_CACHE"] = str(output.resolve() / "runtime-cache" / "hub")
    app = None
    try:
        app = App(output.resolve(), args.seconds)
        if args.smoke:
            smoke(app)
        else:
            clock = pygame.time.Clock()
            while not app.closed:
                app.poll(); app.draw(); clock.tick(30)
    except BaseException:
        if app is not None:
            app.failed = True
            error = traceback.format_exc()
            event(app.log, "fatal_error", traceback=error)
            (app.output / "error.txt").write_text(error, encoding="utf-8")
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
