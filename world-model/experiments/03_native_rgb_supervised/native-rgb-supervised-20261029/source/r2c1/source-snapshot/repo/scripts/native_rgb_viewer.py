"""Bounded click viewer for the native RGB model and shared physical controller.

Opening this viewer executes three real history holds. It is not run by automated
checks. The default command is a plan; live use requires --execute and an explicit
control allowance. Model inputs never contain the UI target cross or a letter ID.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdyssey.native_rgb_evaluation import TRAIN_LETTERS, UNSEEN_LETTERS

SIDE = 384
LEFT = (12, 110, SIDE, SIDE)
RIGHT = (412, 110, SIDE, SIDE)
WIDTH, HEIGHT = 808, 540


def click_target(position, panel=LEFT):
    """Convert only clicks inside the actual-image panel to absolute targets."""
    x, y = position
    px, py, width, height = panel
    if not px <= x < px+width or not py <= y < py+height:
        return None
    return (min(511., max(0., (x-px)*512/width)),
            min(511., max(0., (y-py)*512/height)))


def allowed_letters(*, server=False, allow_unseen=False, frozen_validated=False):
    if allow_unseen and not (server and frozen_validated):
        raise ValueError("Unseen-letter UI requires a server and a validated frozen final protocol")
    return tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if allow_unseen else TRAIN_LETTERS


def validate_frozen_viewer_protocol(config, checkpoint, data, calibration, frozen):
    """Accept the CLI's full seal or its direct core identity, before any UI.

    The raw viewer performs no physical scoring, so it does not use a physical
    instrument identity to unlock letters. Explicit server authorization and the
    intact core checkpoint/data/config/calibration identity remain mandatory.
    """
    from tdyssey.native_rgb_evaluation import final_protocol_identity
    if not config.get("server"):
        raise ValueError("A local config cannot enable frozen final viewing")
    if isinstance(frozen,(str,Path)):
        frozen=json.loads(Path(frozen).read_text(encoding="utf-8"))
    if "evaluation_identity" in frozen:
        if frozen.get("kind") != "native-gpc-rgb-frozen.v1":
            raise ValueError("Expected the complete native RGB frozen-model seal")
        digest=hashlib.sha256()
        with Path(checkpoint).open("rb") as stream:
            for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
        if frozen.get("checkpoint_sha256") != digest.hexdigest():
            raise ValueError("Frozen seal checkpoint SHA differs from the selected model file")
        frozen=frozen["evaluation_identity"]
    if frozen != final_protocol_identity(config,checkpoint,data,calibration):
        raise ValueError("Viewer core frozen identity differs")
    return True


class NativeRGBSession:
    """Four *actual* frames and three actual controls; one click executes one H1.

    The environment factory and predictor are injected so tests can verify control
    timing without stepping real physics or initializing CUDA.
    """

    def __init__(self, env_factory, predictor, *, letter="T", letters=TRAIN_LETTERS,
                 max_controls=600, seed=5478):
        if type(max_controls) is not int or not 3 <= max_controls <= 18000:
            raise ValueError("Viewer control allowance must be in [3,18000]")
        self.env_factory, self.predictor = env_factory, predictor
        self.letters, self.max_controls = tuple(letters), max_controls
        self.seed, self.controls, self.resets = int(seed), 0, 0
        self.env = None
        self.closed = False
        self.events = []
        self.reset(letter)

    def _room(self, controls):
        if self.controls+controls > self.max_controls:
            raise RuntimeError("Viewer control allowance exhausted; no control was executed")

    def _step(self, target):
        self._room(1)
        # Charge before executing, including a step interrupted by an exception.
        self.controls += 1
        return self.env.step(target)

    def reset(self, letter=None):
        import numpy as np
        from tdyssey.native_rgb_env import CONTROLLER
        from tdyssey.native_rgb_evaluation import _rgb
        letter = letter or self.letter
        if letter not in self.letters:
            raise ValueError("This letter is sealed from the local viewer")
        self._room(3)
        env = self.env_factory(letter)
        if env.controller != CONTROLLER:
            raise ValueError("Viewer must use the same approved finite-force controller")
        # The custom environment renderer is offscreen; no display/goal markers.
        initial = _rgb(env.reset_scene()).copy()
        self.env, self.letter = env, letter
        self.history = [initial]
        self.previous_actions = []
        self.target = self.last_prediction = self.last_actual = None
        self.blocked, self.last_error = False, None
        try:
            for _ in range(3):
                target = np.asarray(self.env.state()[:2], dtype=np.float32).copy()
                frame, info = self._step(target)
                self.history.append(_rgb(frame).copy())
                self.previous_actions.append(target)
                if not info.get("valid", False):
                    raise RuntimeError("Invalid physics during history hold; viewer stopped")
        except Exception as exc:
            self.blocked, self.last_error = True, str(exc)
            raise
        self.resets += 1
        self.events.append(dict(kind="reset", letter=letter, executed_history_holds=3,
                                total_controls=self.controls))

    def click_and_step(self, target):
        import numpy as np
        from tdyssey.native_rgb_evaluation import _rgb
        if self.blocked:
            raise RuntimeError("Viewer stopped after an error: "+str(self.last_error))
        self._room(1)
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (2,) or not np.isfinite(target).all() or target.min() < 0 or target.max() > 511:
            raise ValueError("Click target must be finite absolute xy in [0,511]")
        history = np.stack(self.history).copy()
        actions = np.concatenate((np.stack(self.previous_actions), target[None]), axis=0)
        self.target = target.copy()
        try:
            # Predict BEFORE advancing the real scene. No truth labels/letter enter.
            raw = _rgb(self.predictor(history, actions, (self.seed+self.controls) % 2**32)).copy()
            self.last_prediction = raw
            frame, info = self._step(target.copy())
            actual = _rgb(frame).copy()
            self.last_actual = actual
            self.history = [*self.history[1:], actual]
            self.previous_actions = [*self.previous_actions[1:], target.copy()]
            result = dict(kind="click", letter=self.letter, target=target.tolist(),
                          total_controls=self.controls, physics_valid=bool(info.get("valid", False)),
                          contact=int(info.get("contact", 0)),
                          clean_model_input=True, output_repair=False)
            self.events.append(result)
            if not result["physics_valid"]:
                self.blocked, self.last_error = True, "Invalid physical transition; no further click is allowed"
            return result
        except Exception as exc:
            self.blocked, self.last_error = True, type(exc).__name__+": "+str(exc)
            self.events.append(dict(kind="error", message=self.last_error,
                                    total_controls=self.controls))
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.env is not None and hasattr(self.env, "close"):
                self.env.close()
        finally:
            self.env = None
            self.predictor.close()


def run_ui(session, *, output, seconds=600):
    """Render the target cross on the display surface only."""
    import numpy as np
    import pygame
    if type(seconds) is not int or not 1 <= seconds <= 1800:
        raise ValueError("Viewer duration must be in [1,1800] seconds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    log = (output/"events.jsonl").open("x", encoding="utf-8")
    pygame.display.init()
    pygame.font.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Native GPC RGB: actual / raw H1 prediction")
    font = pygame.font.SysFont("sans", 16)
    buttons = [(pygame.Rect(12+i*30, 12, 27, 28), letter)
               for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")]
    cursor, started, done, status = 0, time.monotonic(), False, "Click actual image to predict, then execute one control"
    clock = pygame.time.Clock()
    def flush_events():
        nonlocal cursor
        for row in session.events[cursor:]:
            log.write(json.dumps(row, sort_keys=True)+"\n")
        cursor = len(session.events)
        log.flush()
    def panel(rect, image):
        pygame.draw.rect(window, (230,230,230), rect)
        if image is not None:
            # A display-only uint8 conversion; no overlay or resize feeds back.
            pixels = np.rint(image*255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(pixels.transpose(2,1,0))
            window.blit(pygame.transform.scale(surface, (rect[2],rect[3])), rect[:2])
    try:
        while not done and time.monotonic()-started < seconds:
            for item in pygame.event.get():
                if item.type == pygame.QUIT or (item.type == pygame.KEYDOWN and item.key == pygame.K_ESCAPE):
                    done = True
                elif item.type == pygame.KEYDOWN and item.key == pygame.K_r:
                    try:
                        session.reset()
                        status = "Reset; three real history holds executed"
                    except Exception as exc:
                        status = str(exc)
                elif item.type == pygame.MOUSEBUTTONDOWN and item.button == 1:
                    try:
                        chosen = next((letter for rect, letter in buttons if rect.collidepoint(item.pos)), None)
                        if chosen is not None:
                            session.reset(chosen)
                            status = "Reset "+chosen+"; three real history holds executed"
                        else:
                            target = click_target(item.pos)
                            if target is not None:
                                result = session.click_and_step(target)
                                status = "Predicted raw RGB; executed one control" if result["physics_valid"] else session.last_error
                    except Exception as exc:
                        status = str(exc)
            flush_events()
            window.fill((246,246,246))
            for rect, letter in buttons:
                colour = (145,190,220) if letter == session.letter else (210,215,220)
                if letter not in session.letters:
                    colour = (238,238,238)
                pygame.draw.rect(window, colour, rect)
                window.blit(font.render(letter, True, (30,30,30)), (rect.x+7,rect.y+4))
            window.blit(font.render("R reset | Esc close | controls "+str(session.controls)+"/"+str(session.max_controls), True, (30,30,30)), (12,48))
            window.blit(font.render(str(status)[:98], True, (30,30,30)), (12,72))
            window.blit(font.render("Actual clean RGB", True, (20,20,20)), (LEFT[0],91))
            window.blit(font.render("Unchanged sampler H1 RGB", True, (20,20,20)), (RIGHT[0],91))
            panel(LEFT, session.history[-1])
            panel(RIGHT, session.last_prediction)
            if session.target is not None:
                x = round(LEFT[0]+float(session.target[0])*SIDE/512)
                y = round(LEFT[1]+float(session.target[1])*SIDE/512)
                pygame.draw.line(window, (225,35,35), (x-7,y), (x+7,y), 2)
                pygame.draw.line(window, (225,35,35), (x,y-7), (x,y+7), 2)
            pygame.display.flip()
            clock.tick(30)
    finally:
        flush_events()
        log.close()
        session.close()
        pygame.display.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--gpc-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Complete native RGB training checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/native-rgb-viewer")
    parser.add_argument("--letter", choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), default="T")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--max-controls", type=int, default=600, help="Includes three real holds per reset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Explicit live UI; starts real controls")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--allow-unseen", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--frozen-protocol", type=Path)
    args = parser.parse_args(argv)
    validated = False
    if args.allow_unseen or args.frozen_protocol:
        if not (args.server and args.config and args.data and args.calibration and args.frozen_protocol):
            parser.error("Unseen viewer needs server/config/data/calibration/frozen-protocol")
        from scripts.native_rgb_world_model import load_config
        config = load_config(args.config, args.artifact_root or args.checkpoint.parent.parent)
        if not config.get("server"):
            parser.error("A local config cannot enable unseen viewing")
        try:
            validate_frozen_viewer_protocol(config,args.checkpoint,args.data,args.calibration,args.frozen_protocol)
        except ValueError as error:
            parser.error(str(error))
        validated = True
    letters = allowed_letters(server=args.server, allow_unseen=args.allow_unseen, frozen_validated=validated)
    if args.letter not in letters:
        parser.error("This letter is sealed from local predictions")
    if not 1 <= args.seconds <= 1800 or not 3 <= args.max_controls <= 18000:
        parser.error("seconds1..1800 and max-controls3..18000 are required")
    if not args.execute:
        result = dict(status="plan_only", real_controls=0, allowed_letters=list(letters),
                      live_command_requires="--execute; three real controls per reset are included in max-controls")
        print(json.dumps(result, indent=2))
        return result
    from tdyssey.native_rgb_env import CONTROLLER, make_env
    from tdyssey.native_rgb_evaluation import NativePredictor
    if args.output.exists():
        raise FileExistsError("Choose a new viewer output before any controls are executed")
    predictor = NativePredictor(args.gpc_root, args.init_checkpoint, args.checkpoint, args.device)
    session = None
    try:
        session = NativeRGBSession(lambda letter: make_env(args.physics_root, letter, CONTROLLER),
            predictor, letter=args.letter, letters=letters, max_controls=args.max_controls)
        run_ui(session, output=args.output, seconds=args.seconds)
    finally:
        if session is None:
            predictor.close()
        else:
            session.close()


if __name__ == "__main__":
    main()
