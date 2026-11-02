"""Explicit R2C0 saved-array/data-order bridges; never construct/train a model.

This does not adapt or invoke the old hard-coded18/36 training entry. A future
authorized runner must separately own model/budget/optimizer/checkpoint state.
Only image/action are returned as network inputs; identities remain audit data.
"""
from copy import deepcopy
import hashlib
import json

KIND = "native-rgb-r2c0-prepared-array-interface.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


class PreparedInputs:
    def __init__(self,dataset,order,*,dataset_identity,sampling_identity,evaluation_identity,
                 seed=5478,effective_batch=16,microbatch=4):
        if (seed,effective_batch,microbatch)!=(5478,16,4):
            raise ValueError("This proposal fixes seed5478/effective16/micro4")
        if getattr(dataset,"split",None)!="train" or len(dataset)!=861:
            raise ValueError("Only the frozen41-trajectory861-window TRAIN may be sampled")
        if not callable(getattr(dataset,"__getitem__",None)):
            raise ValueError("Saved-array dataset requires explicit getitem")
        if any(not callable(getattr(order,n,None)) for n in ("identity","take","state_dict","load_state_dict")):
            raise ValueError("Order requires identity/take/state_dict/load_state_dict")
        if order.identity()!=sampling_identity:
            raise ValueError("Passed sampler identity differs from actual order")
        if getattr(order,"cursor",0)!=0:
            raise ValueError("Construct with a fresh order; restore only the complete prepared-input snapshot")
        if any(not isinstance(x,dict) or not x for x in (dataset_identity,sampling_identity,evaluation_identity)):
            raise ValueError("Explicit nonempty frozen data/sampling/evaluation identities required")
        self.dataset,self.order=dataset,order
        self.identity_payload=dict(kind=KIND,seed=seed,effective_batch=effective_batch,microbatch=microbatch,
            dataset=deepcopy(dataset_identity),sampling=deepcopy(sampling_identity),evaluation=deepcopy(evaluation_identity))
        self.identity_sha256=_digest(self.identity_payload)
        self.effective_batch,self.microbatch=effective_batch,microbatch
        self.pending=None;self.micro_cursor=0;self.delivered_draws=0

    def identity(self):
        return dict(**deepcopy(self.identity_payload),identity_sha256=self.identity_sha256)

    def begin_update(self):
        """Select16 CPU indices; caller reserves any future optimizer attempt first."""
        if self.pending is not None:raise RuntimeError("A prepared update still has unread microbatches")
        selected=list(self.order.take(self.effective_batch))
        if len(selected)!=self.effective_batch or any(type(i) is not int or not 0<=i<len(self.dataset) for i in selected):
            raise ValueError("Sampler returned invalid TRAIN indices")
        self.pending=selected;self.micro_cursor=0;self.delivered_draws+=len(selected)
        return tuple(selected)

    def next_microbatch(self):
        """CPU NumPy arrays only. No device transfer/Torch/model operation exists."""
        import numpy as np
        if self.pending is None:raise StopIteration("No prepared update")
        indices=self.pending[self.micro_cursor:self.micro_cursor+self.microbatch]
        images=[];actions=[]
        for index in indices:
            item=self.dataset[index]
            image,action=np.asarray(item["image"]),np.asarray(item["action"])
            if (image.shape!=(5,3,96,96) or action.shape!=(4,2) or image.dtype!=np.float32 or action.dtype!=np.float32
                    or not np.isfinite(image).all() or not np.isfinite(action).all()
                    or image.min()<0 or image.max()>1 or action.min()< -1 or action.max()>1):
                raise ValueError("Raw float32[0,1] five RGB frames and normalizedfloat32[-1,1] actions required")
            images.append(image);actions.append(action)
        payload=dict(image=np.stack(images),action=np.stack(actions))
        self.micro_cursor+=len(indices)
        if self.micro_cursor==self.effective_batch:self.pending=None;self.micro_cursor=0
        return dict(indices=tuple(indices),model_inputs=payload)

    def state_dict(self):
        """CPU data state only; this never saves in-flight gradients/model state.

        Future real training checkpoints require a completed optimizer boundary
        and the separate full model/optimizer/scaler/scheduler/RNG payload.
        """
        state=dict(kind=KIND,identity_sha256=self.identity_sha256,order=deepcopy(self.order.state_dict()),
            pending_indices=deepcopy(self.pending),micro_cursor=self.micro_cursor,delivered_draws=self.delivered_draws)
        state["state_sha256"]=_digest(state)
        return state

    def load_state_dict(self,state):
        if set(state)!={"kind","identity_sha256","order","pending_indices","micro_cursor","delivered_draws","state_sha256"}:
            raise ValueError("Incomplete or unknown prepared-input state")
        value=deepcopy(state);checksum=value.pop("state_sha256")
        if checksum!=_digest(value):raise ValueError("Prepared-input identity/order/pending/cursor checksum differs")
        if state["kind"]!=KIND or state["identity_sha256"]!=self.identity_sha256:
            raise ValueError("Frozen split/pools/evaluation/settings identity changed")
        pending=state["pending_indices"];cursor=state["micro_cursor"];draws=state["delivered_draws"]
        if type(draws) is not int or draws<0 or type(cursor) is not int or cursor<0 or cursor%self.microbatch:
            raise ValueError("Invalid CPU draw/microbatch cursor")
        if not isinstance(state["order"],dict) or state["order"].get("cursor")!=draws:
            raise ValueError("Sampler and prepared-input delivered draw counters differ")
        if pending is None:
            if cursor!=0:raise ValueError("No pending update but nonzero microbatch cursor")
        elif (not isinstance(pending,list) or len(pending)!=self.effective_batch or cursor>=len(pending)
                or any(type(i) is not int or not 0<=i<len(self.dataset) for i in pending)):
            raise ValueError("Invalid saved pending TRAIN indices")
        self.order.load_state_dict(deepcopy(state["order"]))
        self.pending=deepcopy(pending);self.micro_cursor=cursor;self.delivered_draws=draws


def evaluate_prepared_cohort(rollout_callback,predict_callback,cohort_arrays,*,cohort,expected_identity,execution_guard,
                             seed=5478,batch_size=4):
    """Explicit pure-array handoff; future real predictor requires caller authorization.

    Truth arrays may exist for later scoring in cohort_arrays, but are not passed
    to the rollout or predictor. Predictor action domain is normalized[-1,1]
    under the new R2C0 evaluation API, unlike the old absolute-action wrapper.
    """
    if not callable(execution_guard) or not callable(rollout_callback) or not callable(predict_callback):
        raise ValueError("Explicit guard/rollout/predictor callbacks required")
    if seed!=5478 or batch_size!=4:raise ValueError("Fixed evaluation seed5478/batch4 required")
    if cohort_arrays["identity"]!=expected_identity:
        raise ValueError("Prepared evaluation manifest/index/protocol identity differs")
    if cohort not in ("train_visual","val_natural","val_motion"):
        raise ValueError("Only the three frozen R2C0 diagnostic cohorts")
    if cohort_arrays.get("cohort",cohort)!=cohort:raise ValueError("Explicit cohort differs from loaded arrays")
    allowed=execution_guard(dict(cohort=cohort,identity=deepcopy(expected_identity),seed=seed,batch_size=batch_size))
    if allowed is False:raise PermissionError("Execution guard rejected predictor callback")
    return rollout_callback(predict_callback,cohort_arrays["histories"],cohort_arrays["absolute_actions"],
        window_ids=list(cohort_arrays["window_ids"]),seed=seed,cohort=cohort,batch_size=batch_size)
