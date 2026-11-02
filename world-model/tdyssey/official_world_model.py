"""Explicit raw-RGB H1 assets and the pinned native GPC predictor.

Import/help/asset inspection needs no Torch or simulator. Callers supply trained
weights and sidecar; there is no released-model fallback for the viewer.
"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import weakref
from .gpc_world_model import load_world_model_source,world_model_spec

RAW_CONTRACT = dict(version='official_raw_rgb_h1.v1',history_frames=4,image_shape=[3,96,96],
    image_dtype='float32',image_range=[0,1],action_shape=[4,2],
    action_semantics='absolute world-frame pixel targets in [0,511]',
    action_normalization='clip[0,511], then a/511*2-1',horizon=1,euler_steps=3,
    sigma_cond=None,t1=False,returned_image='clip(native_prediction,0,1)')
MAX_WEIGHT_BYTES=64*1024**2
_MODEL_LOCK=threading.Lock()
_ACTIVE_MODEL=None


def _sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _reject_constant(value):raise ValueError('Nonfinite JSON: '+value)


def _file(path,maximum):
    path=Path(path).expanduser().resolve();before=path.stat()
    if not path.is_file() or not 0<before.st_size<=maximum:raise ValueError('Asset size/type is invalid: '+str(path))
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Asset changed during inspection')
    return dict(path=str(path),bytes=after.st_size,sha256=digest.hexdigest())


def model_profile(checkpoint,expected_sha256,*,sidecar,expected_sidecar_sha256=None,label='Raw T+L'):
    if not _sha(expected_sha256) or (expected_sidecar_sha256 is not None and not _sha(expected_sidecar_sha256)):
        raise ValueError('Explicit lowercase SHA256 identities are required')
    return dict(kind='trained',checkpoint=str(Path(checkpoint).expanduser().resolve()),
        expected_sha256=expected_sha256,sidecar=str(Path(sidecar).expanduser().resolve()),
        expected_sidecar_sha256=expected_sidecar_sha256,label=label,contract=deepcopy(RAW_CONTRACT))


def initial_profile(checkpoint,expected_sha256=None):
    """Released initialization is training-only, never a viewer fallback."""
    expected=world_model_spec()['checkpoint']
    if expected_sha256 is not None and expected_sha256!=expected['sha256']:
        raise ValueError('Training initialization must match the pinned released asset')
    return dict(kind='initial',checkpoint=str(Path(checkpoint).expanduser().resolve()),
        expected_sha256=expected['sha256'],expected_bytes=expected['bytes'],sidecar=None,
        label='Training initialization',contract=deepcopy(RAW_CONTRACT))


def inspect_profile(profile):
    result=dict(ready=False,status='invalid',reason=None,profile=deepcopy(profile))
    try:
        if profile.get('kind') not in ('trained','initial') or not _sha(profile.get('expected_sha256')):
            raise ValueError('Invalid explicit model profile')
        weight=_file(profile['checkpoint'],MAX_WEIGHT_BYTES)
        if weight['sha256']!=profile['expected_sha256'] or (profile.get('expected_bytes') is not None
                and weight['bytes']!=profile['expected_bytes']):raise ValueError('Checkpoint identity mismatch')
        result['checkpoint']=weight
        if profile['kind']=='trained':
            if weight['sha256']==world_model_spec()['checkpoint']['sha256']:
                raise ValueError('Released initialization is not a trained raw checkpoint')
            side=_file(profile['sidecar'],65536)
            if profile.get('expected_sidecar_sha256') is not None and side['sha256']!=profile['expected_sidecar_sha256']:
                raise ValueError('Sidecar identity mismatch')
            metadata=json.loads(Path(side['path']).read_bytes(),parse_constant=_reject_constant)
            if (not isinstance(metadata,dict) or type(metadata.get('updates')) is not int or metadata['updates']<1
                    or not _sha(metadata.get('init_sha256')) or metadata.get('t1') is True
                    or metadata.get('optimizer_state') is True):raise ValueError('Expected trained model-only raw sidecar')
            if metadata.get('input_contract',RAW_CONTRACT['version'])!=RAW_CONTRACT['version']:
                raise ValueError('Sidecar input contract is not raw H1 RGB')
            raw=metadata.get('raw_contract',{})
            if not isinstance(raw,dict) or any(raw.get(key,RAW_CONTRACT[key])!=RAW_CONTRACT[key] for key in
                    ('version','history_frames','image_shape','action_shape','horizon','euler_steps','sigma_cond','t1')):
                raise ValueError('Sidecar raw frame/action/sampler contract differs')
            declared=metadata.get('checkpoint',{})
            if declared and (declared.get('sha256')!=weight['sha256'] or declared.get('bytes',weight['bytes'])!=weight['bytes']):
                raise ValueError('Sidecar checkpoint identity mismatch')
            # Embedded old provenance paths are never followed or required.
            side['metadata']=metadata;result['sidecar']=side
        else:
            pinned=world_model_spec()['checkpoint']
            if weight['sha256']!=pinned['sha256'] or weight['bytes']!=pinned['bytes']:
                raise ValueError('Initialization identity mismatch')
        result.update(ready=True,status='available')
    except FileNotFoundError as error:result.update(status='missing',reason=str(error))
    except (OSError,ValueError,KeyError,TypeError) as error:result['reason']=str(error)
    return result


def _construct_denoiser(profile,source_root,*,training=False,execute=False):
    if execute is not True:raise PermissionError('Explicit execution is required for model construction')
    inspected=inspect_profile(profile)
    if not inspected['ready']:raise ValueError(inspected['reason'])
    if not training and profile['kind']!='trained':raise ValueError('Viewer requires explicitly trained raw weights')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no CPU model fallback')
    source=load_world_model_source(source_root);spec=world_model_spec();config=dict(spec['model'])
    denoiser={key:config.pop(key) for key in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
    with torch.random.fork_rng(devices=[]):
        network=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**config),**denoiser)).float()
    payload=torch.load(profile['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    if not isinstance(payload,dict) or not payload or any(not isinstance(v,torch.Tensor) or not bool(torch.isfinite(v).all()) for v in payload.values()):
        raise ValueError('Expected finite model-only state_dict')
    network.load_state_dict(payload,strict=True);del payload
    device=torch.device('cuda:0')
    try:
        network.to(device)
        if training:
            if any(not name.startswith('inner_model.') for name,_ in network.named_parameters()):
                raise ValueError('Unexpected trainable parameter outside inner_model')
            network.requires_grad_(True).train()
            network.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
        else:network.requires_grad_(False).eval()
        network.asset_identity=inspected
    except BaseException as error:
        _discard_partial(network,error)
        raise
    return network,device,source


def load_denoiser(profile,source_root,*,training=False,execute=False):
    """One live network per process; caller owns the run's external limits."""
    global _ACTIVE_MODEL
    with _MODEL_LOCK:
        if _ACTIVE_MODEL is not None and _ACTIVE_MODEL() is not None:
            raise RuntimeError('Close the active world model before another load')
        result=_construct_denoiser(profile,source_root,training=training,execute=execute)
        _ACTIVE_MODEL=weakref.ref(result[0])
        return result


def _discard_partial(network,error):
    """Drop GPU storage even when an exception traceback retains a native frame."""
    global _ACTIVE_MODEL
    if network is None:return
    if _ACTIVE_MODEL is not None and _ACTIVE_MODEL() is network:_ACTIVE_MODEL=None
    try:
        network.cpu()
        gc.collect()
        import torch
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    except BaseException as cleanup:error.add_note('Partial model cleanup failed: '+str(cleanup))


def normalize_actions(actions):
    import numpy as np
    values=np.asarray(actions,dtype=np.float64)
    return (np.clip(values,0,511)/511*2-1).astype(np.float32)


class OfficialWorldModel:
    def __init__(self,profile,source_root,*,execute=False):
        if profile.get('kind')!='trained':raise ValueError('A trained raw model is required')
        self.sampler=self.device=None;network=None
        try:
            network,self.device,source=load_denoiser(profile,source_root,execute=execute)
            self.asset_identity=deepcopy(network.asset_identity)
            sampler_spec=world_model_spec()['sampler']
            if sampler_spec['num_steps_denoising']!=3 or sampler_spec['s_cond']!=0:
                raise ValueError('Raw H1 requires the native three-Euler/default-conditioning sampler')
            self.sampler=source.DiffusionSampler(network,source.DiffusionSamplerConfig(**sampler_spec))
            self.check_limits()
        except BaseException as error:
            self.sampler=self.device=None;_discard_partial(network,error);network=None
            raise

    def check_limits(self):
        import torch
        torch.cuda.synchronize(self.device)
        if torch.cuda.max_memory_allocated(self.device)>512*1024**2 or torch.cuda.max_memory_reserved(self.device)>1024**3:
            raise MemoryError('World-model CUDA peak ceiling exceeded')

    def predict_next(self,images,absolute_actions,seed=5478):
        import numpy as np
        image=np.asarray(images);action=np.asarray(absolute_actions)
        if (image.shape!=(4,3,96,96) or image.dtype!=np.float32 or not np.isfinite(image).all()
                or (image<0).any() or (image>1).any() or action.shape!=(4,2)
                or not np.isfinite(action).all() or (action<0).any() or (action>511).any()):
            raise ValueError('Expected four raw RGB frames and four absolute pixel targets')
        if self.sampler is None or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Closed model or invalid seed')
        import torch
        self.check_limits();device=self.device
        index=device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index),torch.random.fork_rng(devices=[index]),torch.inference_mode():
            torch.cuda.manual_seed(seed)
            prediction,_=self.sampler.sample(torch.as_tensor(image,device=device)[None],
                torch.as_tensor(normalize_actions(action),device=device)[None])
        frame=prediction[0].detach().cpu().numpy()
        if not np.isfinite(frame).all():raise FloatingPointError('Nonfinite native forecast')
        self.check_limits()
        return np.clip(frame,0,1)

    def close(self):
        self.sampler=self.device=None;gc.collect()
        import torch
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
