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
from tdyssey.manual_world_model import ManualSession, PinnedPredictor, event
from tdyssey.wm_compat_transform import transform_t1

WIDTH, HEIGHT, SIDE = 1248, 730, 384
PANELS = [pygame.Rect(24 + i * 408, 178, SIDE, SIDE) for i in range(3)]


def target_from_click(position):
    if not PANELS[0].collidepoint(position):
        return None
    return np.array([(position[0] - PANELS[0].x) * 512 / SIDE,
                     (position[1] - PANELS[0].y) * 512 / SIDE]).clip(0, 511)


class App:
    def __init__(self, output, seconds=1800):
        self.output, self.started, self.seconds = output, time.monotonic(), seconds
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (output / "events.jsonl").open("x", encoding="utf-8")
        event(self.log, "opened", kind_of_session="manual_exploration_not_training_or_test_set",
              pid=os.getpid(), model_updates=0, maximum_seconds=seconds)
        self.session = ManualSession()
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tdyssey | T/L 手动预测与真实执行")
        self.font = pygame.font.SysFont("microsoftyahei,simhei,segoeui", 19)
        self.small = pygame.font.SysFont("microsoftyahei,simhei,segoeui", 16)
        self.title_font = pygame.font.SysFont("microsoftyahei,simhei,segoeui", 25, bold=True)
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.predictor = PinnedPredictor()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self.busy = self.failed = self.closed = False
        self.status = "点击左图选择目标并预测；按空格执行。首次预测会加载模型。"
        self.elapsed_prediction = None
        self.result = None
        self.video = None
        self.video_path = None
        self.video_frames = 0
        self.video_last = 0
        self.last_memory = 0
        self.peak_private = self.peak_ws = 0
        self.buttons = []
        labels = [("T", "T"), ("L", "L"), ("初始权重 [1]", "frozen"),
                  ("适配500步 [2]", "adapted"), ("1步 [H]", "horizon"),
                  ("预测 [P]", "predict"), ("执行 [空格]", "execute"),
                  ("重置 [R]", "reset"), ("截图 [S]", "save"), ("录屏 [V]", "record")]
        x = 24
        for label, action in labels:
            width = max(48, self.small.size(label)[0] + 22)
            self.buttons.append((pygame.Rect(x, 77, width, 38), label, action))
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
                self.status = "已重置并用3步真实保持动作建立历史；点击左图选目标。"
                event(self.log, "reset", shape=self.session.shape, warmup_controls=3)
            elif action in ("frozen", "adapted"):
                self.session.configure(role=action)
                self.result = None
                self.status = "已切换权重；点击预测时加载，旧模型先释放。"
                event(self.log, "checkpoint_selected", role=action)
            elif action == "horizon":
                self.session.configure(horizon=4 if self.session.horizon == 1 else 1)
                self.result = None
                self.status = "4步模式连续预测4次，执行时连续4次使用同一目标。"
            elif action == "predict":
                request = self.session.request()
                self.busy = True
                self.status = "正在加载 / 预测……窗口仍可关闭，未推进真实仿真。"
                self.requests.put(request)
                event(self.log, "prediction_requested", role=request["role"], shape=request["shape"],
                      horizon=request["horizon"], target=request["target"].tolist(), seed=request["noise_seed"])
            elif action == "execute":
                self.result = self.session.execute()
                self.status = "执行完成：中、右是相同动作和时刻的预测/真实对照。再次选目标可继续。"
                if self.session.done:
                    self.status = "真实仿真达到成功阈值；本轮已停止。按 R 重置。"
                event(self.log, "executed", **self.result)
        except ValueError as exc:
            self.status = str(exc)

    def handle(self, item):
        if item.type == pygame.QUIT or (item.type == pygame.KEYDOWN and item.key == pygame.K_ESCAPE):
            self.closed = True
        elif item.type == pygame.KEYDOWN:
            actions = {pygame.K_t: "T", pygame.K_l: "L", pygame.K_1: "frozen", pygame.K_2: "adapted",
                       pygame.K_h: "horizon", pygame.K_p: "predict", pygame.K_SPACE: "execute",
                       pygame.K_r: "reset", pygame.K_s: "save", pygame.K_v: "record"}
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
        if not self.results.empty():
            request, result, error = self.results.get_nowait()
            self.busy = False
            if error:
                self.failed = True
                (self.output / "error.txt").write_text(error, encoding="utf-8")
                event(self.log, "first_error", traceback=error)
                self.status = "预测失败，已停止模型操作。错误保存在本次输出目录；可截图或退出。"
            else:
                raw, measurement, self.elapsed_prediction = result
                self.session.accept(request, raw, measurement)
                self.status = "预测完成，真实仿真尚未移动。按空格执行同一动作。"
                event(self.log, "predicted", role=request["role"], horizon=len(raw),
                      seconds=self.elapsed_prediction, measurement=measurement,
                      raw_min=float(raw.min()), raw_max=float(raw.max()), loads=self.predictor.loads,
                      checkpoint=self.predictor.assets["weights"][request["role"]])
        elapsed = time.monotonic() - self.started
        if elapsed >= self.seconds:
            self.status, self.closed = "本次手操时限已到，正在关闭。", True
        if os.name == "nt" and time.monotonic() - self.last_memory >= 1:
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

    def panel(self, index, image, title, subtitle):
        rect = PANELS[index]
        pygame.draw.rect(self.window, (30, 40, 56), rect, border_radius=8)
        self.text(title, (rect.x, rect.y - 50))
        self.text(subtitle, (rect.x, rect.y - 25), (151, 168, 190), small=True)
        if image is not None:
            # Clip for display only. Stored predictions and AR feedback stay raw.
            pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(pixels.transpose(2, 1, 0))
            self.window.blit(pygame.transform.scale(surface, rect.size), rect)
        else:
            self.text("等待预测" if index == 1 else "等待真实执行", (rect.x + 125, rect.y + 180), small=True)

    def draw(self):
        self.window.fill((15, 23, 36))
        self.window.blit(self.title_font.render("Tdyssey · 亲手推一下，再对照模型预测", True, (239, 244, 251)), (24, 20))
        role = "初始 GPC" if self.session.role == "frozen" else "适配500步：已知退化，不代表通过验证"
        self.text(f"{self.session.shape}  |  {role}  |  不训练、不自动收集数据", (560, 27), small=True)
        for rect, label, action in self.buttons:
            active = action == self.session.shape or action == self.session.role
            pygame.draw.rect(self.window, (42, 107, 166) if active else (40, 53, 73), rect, border_radius=5)
            if action == "horizon":
                label = f"{self.session.horizon}步 [H]"
            elif action == "record" and self.video is not None:
                label = "停止录屏 [V]"
            self.text(label, (rect.x + 9, rect.y + 9), small=True)
        index = len(self.session.actual) - 1 if self.session.actual is not None else self.session.horizon - 1
        predicted = None if self.session.prediction is None else self.session.prediction[index]
        actual = None if self.session.actual is None else self.session.actual[-1]
        self.panel(0, self.session.observation["image"], "真实仿真 · 点击这里选目标", "原始灰色物体 / 绿色目标；选择不会移动")
        self.panel(1, predicted, f"模型预测 · 第 {index + 1} 步", "T1输入配色：红色物体 / 黑色目标")
        self.panel(2, actual, "真实执行结果 · 与中图对照", "仅将真实结果转为相同配色，不修改物理")
        if self.session.target is not None:
            point = (round(PANELS[0].x + self.session.target[0] * SIDE / 512),
                     round(PANELS[0].y + self.session.target[1] * SIDE / 512))
            pygame.draw.circle(self.window, (230, 70, 50), point, 7, 2)
        self.text(self.status, (24, 579), (255, 201, 104) if self.failed else (210, 230, 245), small=True)
        if self.session.measurement is not None:
            m = self.session.measurement
            cov = m["coverage"].get("estimate") if m["coverage"].get("valid") else None
            reason = m["state"].get("reason") or "valid"
            self.text(f"预测判读：{reason}；coverage = {cov if cov is not None else '未知'}", (24, 612), small=True)
        if self.result is not None:
            self.text(f"真实 coverage：{self.result['truth_coverage']:.3f}；匹配到第 {self.result['matched_horizon']} 步；"
                      f"像素 MAE：{self.result['pixel_mae']:.4f}（仅描述）", (24, 640), small=True)
        self.text(f"真实步数 {self.session.controls}（含初始化） | 预测帧 {self.session.forecasts} | "
                  f"ESC 退出 | S 截图 | V 录屏 | 本次输出：{self.output.name}", (24, 686), (151, 168, 190), small=True)
        pygame.display.flip()
        if self.video is not None and time.monotonic() - self.video_last >= .1:
            import cv2
            frame = pygame.surfarray.array3d(self.window).transpose(1, 0, 2)
            self.video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.video_frames += 1
            self.video_last = time.monotonic()
            if self.video_path.stat().st_size >= 128 * 1024**2:
                self.record()
                self.status = "录像达到128MiB上限，已保存；手操仍可继续。"

    def save(self):
        name = datetime.now().strftime("view-%H%M%S-%f")
        pygame.image.save(self.window, str(self.output / (name + ".png")))
        if self.session.prediction is not None:
            np.save(self.output / (name + "-prediction.npy"), self.session.prediction, allow_pickle=False)
        event(self.log, "screenshot", path=name + ".png", role=self.session.role, shape=self.session.shape,
              result=self.result, measurement=self.session.measurement)
        self.status = f"已保存截图：{name}.png"

    def record(self):
        if self.video is not None:
            self.video.release()
            self.video = None
            event(self.log, "video_saved", path=self.video_path.name, frames=self.video_frames, fps=10)
            self.status = f"录像已保存：{self.video_path.name}"
            return
        import cv2
        self.video_path = self.output / (datetime.now().strftime("manual-%H%M%S-%f") + ".mp4")
        self.video = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (WIDTH, HEIGHT))
        if not self.video.isOpened():
            self.video.release()
            self.video = None
            raise RuntimeError("Installed OpenCV cannot encode MP4; no install is attempted")
        self.video_frames, self.video_last = 0, 0
        self.status = "正在录屏；再次按 V 保存。"

    def close(self):
        if self.video is not None:
            self.record()
        self.requests.put(None)
        self.worker.join(timeout=5)
        event(self.log, "closed", controls=self.session.controls, forecasts=self.session.forecasts,
              resets=self.session.resets, loads=self.predictor.loads, failed=self.failed,
              peak_private_bytes=self.peak_private, peak_working_set_bytes=self.peak_ws,
              peak_cuda_allocated_bytes=self.predictor.peak_allocated,
              peak_cuda_reserved_bytes=self.predictor.peak_reserved,
              elapsed_seconds=time.monotonic() - self.started, worker_stopped=not self.worker.is_alive(),
              training_updates=0)
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
                assert app.session.controls == before, "Prediction advanced the simulator"
                app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
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
