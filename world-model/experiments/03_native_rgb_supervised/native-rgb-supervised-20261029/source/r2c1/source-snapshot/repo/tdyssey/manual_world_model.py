"""Portable original-RGB H1 session. No training, image adapter or measurement.

Source paths and the model are supplied by the caller. Module import performs
no NumPy, Torch, upstream, simulator, file or GUI operation.
"""
from collections import deque
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time

from .state import Snapshot


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _state(value):
    if hasattr(value,"to_json"):value=json.loads(value.to_json())
    return json.loads(Snapshot.from_dict(value).to_json())


def _frame(value):
    import numpy as np
    if (not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=(3,96,96)
            or not np.isfinite(value).all() or (value<0).any() or (value>1).any()):
        raise ValueError("Expected original float32 CHW96 real RGB in [0,1]")
    return value.copy()


def _key(request):
    fields={name:request[name] for name in ("revision","shape","horizon","noise_seed",
        "initial_state","history_states","history_steps")}
    fields["target"]=request["target"].tolist()
    digest=hashlib.sha256(_json(fields).encode())
    for name in ("images","past_actions"):
        value=request[name]
        digest.update(str((value.shape,value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def event(stream,kind,**details):
    stream.write(_json(dict(time=time.time(),kind=kind,**details))+"\n")
    stream.flush()


def make_environment(shape, *, source_root, scene_seed=5478):
    """Execution-only factory; no source discovery or sibling paths."""
    from .official import load_official_source
    from .envs.official import make_official_shape_env_class
    if shape not in ("T","L") or source_root is None:
        raise ValueError("Explicit official source and T/L shape are required")
    cls=make_official_shape_env_class(load_official_source(source_root))
    env=cls(shape=shape,legacy=True)
    try:
        env.seed(scene_seed)
        return env,env.reset()
    except BaseException:
        env.close()
        raise


class ModelPredictor:
    """One supplied OfficialWorldModel; its helper normalizes absolute actions."""
    def __init__(self,model):
        self.model=model
        self.asset_identity=_copy(model.asset_identity)

    def predict(self,request):
        import numpy as np
        if request.get("horizon")!=1 or request.get("image_domain")!="official-original-color":
            raise ValueError("Only original-RGB H1 requests are supported")
        if self.model is None:raise RuntimeError("Model is closed")
        actions=np.concatenate((request["past_actions"].astype(np.float64),request["target"][None]),axis=0)
        frame=self.model.predict_next(request["images"],actions,seed=request["noise_seed"])
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.float32 or frame.shape!=(3,96,96) or not np.isfinite(frame).all():
            raise ValueError("Model returned malformed H1 RGB")
        request["asset_identity"]=deepcopy(self.asset_identity)
        return frame[None].copy()

    def close(self):
        if self.model is not None:
            model,self.model=self.model,None
            model.close()


class ManualSession:
    """Only execute advances a pending click; reset records its three holds."""
    def __init__(self,env_factory, *, noise_seed=5478):
        if type(noise_seed) is not int or not 0<=noise_seed<2**32:
            raise ValueError("Sampling seed must be uint32")
        self.factory,self.noise_seed=env_factory,noise_seed
        self.env=None
        self.revision=self.controls=self.resets=self.forecasts=0
        self.reset("T")

    def reset(self,shape=None):
        import numpy as np
        shape=shape or self.shape
        if shape not in ("T","L"):raise ValueError("Choose T or L")
        if self.env is not None:self.env.close()
        self.env=None
        try:
            self.env,self.observation=self.factory(shape)
            self.shape=shape;self.resets+=1;self.revision+=1
            self.images=deque([_frame(self.observation["image"])],maxlen=4)
            self.actions=deque(maxlen=3)
            self.states=deque([_state(self.env.capture_state())],maxlen=4)
            self.steps=deque(maxlen=3)
            self.done=False
            self.target=self.pending=self.prediction=self.actual=self.case=None
            self.issued=None
            hold=np.asarray(self.observation["agent_pos"],dtype=np.float64)
            for _ in range(3):
                self._step(hold)
                if self.done:break
        except BaseException:
            self.close()
            raise

    def select(self,target):
        import numpy as np
        value=np.asarray(target,dtype=np.float64)
        if value.shape!=(2,) or not np.isfinite(value).all() or (value<0).any() or (value>511).any():
            raise ValueError("Target must be an absolute point in [0,511]")
        self.target=value.copy();self.revision+=1
        self.pending=self.prediction=self.actual=self.case=None
        self.issued=None

    def request(self):
        import numpy as np
        if self.done:raise ValueError("Episode ended; reset before another push")
        if self.target is None or len(self.images)!=4 or len(self.actions)!=3:
            raise ValueError("Select a target after four recorded real frames")
        current=_state(self.env.capture_state())
        if current!=self.states[-1]:raise ValueError("Physical state changed outside this session")
        request=dict(revision=self.revision,shape=self.shape,horizon=1,noise_seed=self.noise_seed,
            image_domain="official-original-color",target=self.target.copy(),images=np.stack(self.images),
            past_actions=np.asarray(self.actions,dtype=np.float32),initial_state=current,
            history_states=deepcopy(list(self.states)),history_steps=deepcopy(list(self.steps)))
        request["request_id"]=_key(request)
        self.issued=request["request_id"]
        return request

    def accept(self,request,prediction):
        import numpy as np
        if (request["revision"]!=self.revision or request["shape"]!=self.shape or request["horizon"]!=1
                or request["noise_seed"]!=self.noise_seed or request.get("request_id")!=self.issued
                or request.get("image_domain")!="official-original-color" or not np.array_equal(request["target"],self.target)
                or _key(request)!=self.issued):
            raise ValueError("Prediction belongs to an obsolete or changed click")
        value=np.asarray(prediction)
        if value.shape==(3,96,96):value=value[None]
        if value.shape!=(1,3,96,96) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError("Expected one finite float32 H1 prediction")
        self.pending=deepcopy(request);self.prediction=value.copy();self.actual=None
        self.case=dict(request=deepcopy(request),prediction=value.copy(),execution=None,actual=None)
        self.forecasts+=1

    def _step(self,action):
        import numpy as np
        action=np.asarray(action,dtype=np.float64)
        before=deepcopy(self.states[-1])
        self.controls+=1  # A failed physical call is still an attempted control.
        observation,reward,done,info=self.env.step(action)
        after=_state(self.env.capture_state())
        if after["control_step"]!=before["control_step"]+1 or after["pusher_target"]!=action.tolist():
            raise ValueError("Executed control/state boundary differs")
        self.observation=observation
        self.images.append(_frame(observation["image"]));self.actions.append(action.astype(np.float32))
        row=dict(before_state=before,after_state=after,action=action.tolist(),reward=float(reward),done=bool(done),
            contacts=int(info["n_contacts"]))
        self.states.append(after);self.steps.append(deepcopy(row))
        self.done=bool(done)
        return row

    def execute(self):
        if self.pending is None or self.done or self.pending["revision"]!=self.revision:
            raise ValueError("A fresh accepted click is required")
        request,self.pending=self.pending,None
        if _state(self.env.capture_state())!=request["initial_state"]:
            raise ValueError("Physical state changed before execution")
        row=self._step(request["target"])
        self.actual=self.observation["image"][None].copy()
        self.case["execution"]=deepcopy(row);self.case["actual"]=self.actual.copy()
        self.revision+=1
        return dict(request_id=request["request_id"],executed_controls=1,step=row)

    def save_case(self,path):
        """New immutable JSON+NPZ capture; never predicts, restores, or steps."""
        import numpy as np
        if self.case is None:raise ValueError("No accepted click is available to save")
        request=self.case["request"]
        arrays=dict(history_raw=request["images"].copy(),past_actions=request["past_actions"].copy(),
            prediction=self.case["prediction"].copy(),actual_raw=self.case["actual"].copy()
                if self.case["actual"] is not None else np.empty((0,3,96,96),np.float32))
        metadata=dict(kind="manual_raw_h1_case.v1",training_data=False,image_domain="official-original-color",t1=False,
            prediction_scope="exact returned helper RGB; not claimed unclipped native Euler",
            request={key:deepcopy(value) for key,value in request.items() if key not in ("images","past_actions","target")},
            target=request["target"].tolist(),execution=deepcopy(self.case["execution"]),
            array_identities={key:dict(shape=list(value.shape),dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in arrays.items()})
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);payload=buffer.getvalue()
        metadata["payload"]=dict(path="arrays.npz",bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        encoded=(_json(metadata)+"\n").encode()
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        for name,value in (("arrays.npz",payload),("case.json",encoded)):
            with (path/name).open("xb") as stream:
                if stream.write(value)!=len(value):raise OSError("Short case write")
                stream.flush();os.fsync(stream.fileno())
        return dict(path=str(path),request_id=request["request_id"],bytes=len(payload)+len(encoded))

    def close(self):
        if self.env is not None:
            env,self.env=self.env,None
            env.close()
