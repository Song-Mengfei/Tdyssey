"""Injected portable H1 UI. The visualize entry owns assets and explicit paths."""
from datetime import datetime
import io
from pathlib import Path
import queue
import threading
import time
import traceback

from tdyssey.manual_world_model import event

SIDE=480
WIDTH,HEIGHT=1008,560
LEFT=(16,64,SIDE,SIDE)
RIGHT=(512,64,SIDE,SIDE)


def target_from_click(position):
    x,y=position
    if not LEFT[0]<=x<LEFT[0]+SIDE or not LEFT[1]<=y<LEFT[1]+SIDE:return None
    return (min(511.,(x-LEFT[0])*512/SIDE),min(511.,(y-LEFT[1])*512/SIDE))


class App:
    """One worker owns serial inference/model cleanup; only UI executes physics."""
    def __init__(self,output,seconds=1800, *, session,predictor):
        if type(seconds) is not int or not 1<=seconds<=1800:raise ValueError("Choose1..1800 UI seconds")
        import pygame
        self.pg=pygame
        self.output,self.seconds=Path(output),seconds
        self.output.mkdir(parents=True,exist_ok=False)
        self.log=(self.output/"events.jsonl").open("x",encoding="utf-8")
        self.session,self.predictor=session,predictor
        self.started=time.monotonic()
        self.busy=self.failed=self.closed=self._cleanup_done=False
        self._worker_cleanup_error=None
        self.requests,self.results=queue.Queue(),queue.Queue()
        self.video=None;self.video_frames=0;self.video_last=0
        self.status="Click left to predict and push once"
        try:
            pygame.display.init();pygame.font.init()
            self.window=pygame.display.set_mode((WIDTH,HEIGHT))
            pygame.display.set_caption("Raw T+L | actual and predicted H1")
            self.font=pygame.font.SysFont("sans",17)
            self.buttons=[]
            for index,(label,action) in enumerate((("T","T"),("L","L"),("Reset (R)","reset"),
                    ("Save (S)","save"),("Record (V)","record"))):
                self.buttons.append((pygame.Rect(16+index*112,12,104,30),label,action))
            event(self.log,"opened",image_domain="official-original-color",horizon=1,training_updates=0,
                  controls_including_history_holds=session.controls,resets=session.resets)
            self.worker=threading.Thread(target=self._work,name="raw-h1-model",daemon=False)
            self.worker.start()
        except BaseException:
            predictor.close();session.close();self.log.close();pygame.display.quit()
            raise

    def _work(self):
        try:
            while True:
                request=self.requests.get()
                if request is None:break
                if self.closed:continue
                try:self.results.put((request,self.predictor.predict(request),None))
                except BaseException:self.results.put((request,None,traceback.format_exc()))
        finally:
            try:self.predictor.close()
            except BaseException:self._worker_cleanup_error=traceback.format_exc()

    def _fail(self,error):
        self.failed=True;self.busy=False
        self.status="Stopped after an error; save or close"
        with (self.output/"first-error.txt").open("x",encoding="utf-8") as stream:stream.write(error)
        event(self.log,"first_error",error=error)

    def action(self,action):
        if action=="record":
            self.record();return
        if self.busy:return
        if action=="save":
            try:self.save()
            except ValueError as error:self.status=str(error)
            return
        if self.failed or self.closed:return
        try:
            if action in ("T","L","reset"):
                self.session.reset(None if action=="reset" else action)
                self.status="Click left to predict and push once"
                event(self.log,"reset",shape=self.session.shape,history_holds=3)
            elif action=="predict":
                request=self.session.request()
                self.busy=True;self.status="Predicting H1..."
                self.requests.put(request)
                event(self.log,"prediction_requested",request_id=request["request_id"],shape=request["shape"],
                    horizon=1,target=request["target"].tolist(),seed=request["noise_seed"])
        except ValueError as error:self.status=str(error)
        except BaseException:self._fail(traceback.format_exc())

    def handle(self,item):
        pg=self.pg
        if item.type==pg.QUIT or (item.type==pg.KEYDOWN and item.key==pg.K_ESCAPE):
            self.closed=True;return
        if item.type==pg.KEYDOWN:
            actions={pg.K_t:"T",pg.K_l:"L",pg.K_r:"reset",pg.K_s:"save",pg.K_v:"record"}
            if item.key in actions:self.action(actions[item.key])
        elif item.type==pg.MOUSEBUTTONDOWN and item.button==1:
            for rect,_,action in self.buttons:
                if rect.collidepoint(item.pos):self.action(action);return
            target=target_from_click(item.pos)
            if target is not None and not self.busy and not self.failed and not self.closed:
                self.session.select(target)
                self.action("predict")

    def poll(self):
        if self.closed:return
        for item in self.pg.event.get():
            self.handle(item)
            if self.closed:return
        if time.monotonic()-self.started>=self.seconds:
            self.closed=True;return
        try:request,prediction,error=self.results.get_nowait()
        except queue.Empty:return
        self.busy=False
        if error:self._fail(error);return
        try:
            self.session.accept(request,prediction)
            result=self.session.execute()
            event(self.log,"executed",**result,asset_identity=request.get("asset_identity"))
            self.status="Episode ended; reset to continue" if self.session.done else "Click left to predict and push once"
        except BaseException:self._fail(traceback.format_exc())

    def _panel(self,rect,image):
        import numpy as np
        self.pg.draw.rect(self.window,(230,230,230),rect)
        if image is None:return
        # Display conversion only: the captured/passed RGB remains untouched.
        pixels=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
        surface=self.pg.surfarray.make_surface(pixels.transpose(2,1,0))
        self.window.blit(self.pg.transform.scale(surface,(SIDE,SIDE)),rect[:2])

    def draw(self):
        self.window.fill((245,245,245))
        for rect,label,action in self.buttons:
            color=(150,190,220) if action==self.session.shape else (210,215,220)
            self.pg.draw.rect(self.window,color,rect)
            self.window.blit(self.font.render(label,True,(20,20,20)),(rect.x+5,rect.y+6))
        actual=self.session.actual[0] if self.session.actual is not None else self.session.observation["image"]
        prediction=self.session.prediction[0] if self.session.prediction is not None else None
        self._panel(LEFT,actual);self._panel(RIGHT,prediction)
        self.window.blit(self.font.render("Actual",True,(20,20,20)),(LEFT[0],44))
        self.window.blit(self.font.render("Predicted H1",True,(20,20,20)),(RIGHT[0],44))
        self.window.blit(self.font.render(self.status[:70],True,(30,30,30)),(580,20))
        if self.session.target is not None:
            point=(round(LEFT[0]+self.session.target[0]*SIDE/512),round(LEFT[1]+self.session.target[1]*SIDE/512))
            self.pg.draw.circle(self.window,(220,50,40),point,6,2)
        self.pg.display.flip()
        if self.video is not None and time.monotonic()-self.video_last>=.1:
            import cv2
            frame=self.pg.surfarray.array3d(self.window).transpose(1,0,2)
            self.video.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            self.video_frames+=1;self.video_last=time.monotonic()
            if self.video_path.stat().st_size>=128*1024**2:self.record()

    def save(self):
        import numpy as np
        if self.session.case is None:raise ValueError("No accepted click is available to save")
        name=datetime.now().strftime("view-%H%M%S-%f")
        # All captures have new names and exclusive files; no old case is reused.
        buffer=io.BytesIO();self.pg.image.save(self.window,buffer,".png")
        with (self.output/(name+".png")).open("xb") as stream:stream.write(buffer.getvalue())
        with (self.output/(name+"-prediction.npy")).open("xb") as stream:
            np.save(stream,self.session.prediction,allow_pickle=False)
        saved=self.session.save_case(self.output/(name+"-case"))
        event(self.log,"saved",**saved,png=name+".png")
        self.status="Case and screenshot saved"

    def record(self):
        if self.video is not None:
            video,self.video=self.video,None
            video.release()
            event(self.log,"video_saved",path=self.video_path.name,frames=self.video_frames,fps=10)
            self.status="Video saved";return
        import cv2
        self.video_path=self.output/datetime.now().strftime("manual-%H%M%S-%f.mp4")
        if self.video_path.exists():raise FileExistsError(self.video_path)
        video=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*"mp4v"),10,(WIDTH,HEIGHT))
        if not video.isOpened():
            video.release();raise RuntimeError("Existing OpenCV cannot encode MP4")
        self.video=video;self.video_frames=0;self.video_last=0
        self.status="Recording; press V to stop"

    def close(self):
        if self._cleanup_done:return
        self._cleanup_done=True;self.closed=True
        errors=[]
        try:
            if self.video is not None:self.record()
        except BaseException:errors.append(traceback.format_exc())
        self.requests.put(None)
        self.worker.join(timeout=30)
        if self.worker.is_alive():errors.append("Inference worker has not stopped; model cleanup is still pending")
        if self._worker_cleanup_error:errors.append(self._worker_cleanup_error)
        try:
            self.session.close()
            event(self.log,"closed",controls_including_history_holds=self.session.controls,
                resets=self.session.resets,accepted_forecasts=self.session.forecasts,
                worker_stopped=not self.worker.is_alive(),cleanup_errors=errors,training_updates=0)
        finally:
            self.log.close();self.pg.display.quit()
        if errors:raise RuntimeError("; ".join(errors))
