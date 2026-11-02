"""Manual H1 bridge to the shared official-colour model, without another sampler."""
from copy import deepcopy
import hashlib
import time

import numpy as np

from .manual_world_model import PinnedPredictor, OFFICIAL_DOMAIN
from .official_world_model import OfficialWorldModel, inspect_profile, normalize_actions


class OfficialPredictor(PinnedPredictor):
    def __init__(self, profiles, gpc_root, *, inspections=None):
        super().__init__()
        self.profiles, self.gpc_root = deepcopy(profiles), gpc_root
        self.inspections = inspections or {role:inspect_profile(profile) for role,profile in profiles.items()}
        self.assets = dict(weights={role:deepcopy(info['checkpoint']) for role,info in self.inspections.items()})

    def role_error(self, role):
        info = self.inspections.get(role)
        return None if info and info['status']=='available' else (info['reason'] if info else
            'Initial checkpoint missing' if role=='frozen' else 'Official T checkpoint missing')

    def load(self, role):
        if self.role == role:
            return
        error = self.role_error(role)
        if error:
            raise FileNotFoundError(error)
        self.close()
        if self.model_load_cap is not None and self.load_reservations >= self.model_load_cap:
            raise RuntimeError('Official model-load cap reached')
        self.load_reservations += 1
        self.model = OfficialWorldModel(self.profiles[role], self.gpc_root, execute=True)
        self.role, self.loads = role, self.loads+1
        self.assets['weights'][role] = deepcopy(self.model.asset_identity)
        self.check_cuda()

    def predict(self, request):
        if request.get('image_domain') != OFFICIAL_DOMAIN or request['horizon'] != 1:
            raise ValueError('Official predictor requires its separate original-colour H1 request')
        started = time.monotonic()
        self.last_trace, self.last_partial_predictions = [], []
        record = None
        try:
            self.load(request['role'])
            if self.forecast_cap is not None and self.forecast_reservations >= self.forecast_cap:
                raise RuntimeError('Official forecast cap reached')
            actions = np.asarray(list(request['past_actions'])+[request['target']],dtype=np.float64)
            record = dict(horizon=1,status='reserved',image_domain=OFFICIAL_DOMAIN,
                input_sha256=hashlib.sha256(request['images'].tobytes()).hexdigest(),
                action_sha256=hashlib.sha256(normalize_actions(actions).tobytes()).hexdigest(),
                actions_absolute=actions.tolist(),euler_reserved=3,noise_seed=request['noise_seed'],
                sigma_cond=None,output_transform='remote np.clip(native,0,1)',t1_applied=False,
                rng_scope='remote CUDA fork_rng/seed; capsule not observed by this bridge')
            self.last_trace.append(record)
            self.forecast_reservations += 1
            self.euler_reservations += 3
            frame = self.model.predict_next(request['images'],actions,seed=request['noise_seed'])
            self.forecast_returns += 1
            if frame.shape!=(3,96,96) or frame.dtype!=np.float32 or not np.isfinite(frame).all():
                raise ValueError('Official helper returned malformed RGB')
            self.last_partial_predictions.append(frame.copy())
            record.update(status='validated',raw_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                          raw_min=float(frame.min()),raw_max=float(frame.max()))
            self.check_cuda()
            measured = dict(status='not_measured',reason='official raw viewer parity; legacy palette decoder is not applied')
            return frame[None],measured,time.monotonic()-started
        except BaseException as error:
            if record is not None:record.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:
            request['prediction_trace'] = deepcopy(self.last_trace)
            request['prediction_usage'] = self.usage()
            request['checkpoint_identity'] = deepcopy(self.assets['weights'].get(request['role']))

    def close(self):
        if self.model is not None:
            self.model.close()
        super().close()
