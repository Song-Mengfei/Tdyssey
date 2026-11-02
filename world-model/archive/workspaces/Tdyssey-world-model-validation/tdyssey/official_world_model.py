"""Official-colour H1 profiles and the shared remote GPC viewer helpers.

Module import is standard-library-only. Profiles never infer a trained asset
from a filename, fall back to initial weights, or convert real inputs through
T1. The returned image follows df99f8's clipping to [0,1]; it is not advertised
as the unclipped native Euler state. No alternative diffusion sampler exists
here. A sidecar/hash binds an explicitly declared asset, not a quality claim.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat

from .gpc_world_model import load_world_model_source, world_model_spec

ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA256 = '4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892'
LEGACY_T1_SHA256 = '0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3'
RAW_CONTRACT = dict(version='official_raw_rgb_h1.v1', image_shape=[3,96,96],
    image_dtype='float32', image_range=[0,1], history_frames=4, action_shape=[4,2],
    action_semantics='absolute world-frame pixel targets in [0,511]',
    action_normalization='remote model_action: clip[0,511], then a/511*2-1',
    horizon=1, t1=False, sigma_cond=None, euler_steps=3,
    returned_image='df99f8 predict_next_frame: np.clip(native_prediction,0,1)')
_NATIVE_SAMPLER_SPEC = dict(num_steps_denoising=3, sigma_min=.002, sigma_max=5,
                          rho=7, order=1, s_churn=0, s_cond=0)


def _sha(value):return hashlib.sha256(value).hexdigest()


def _digest_string(value):
    return isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _reject_constant(value):
    raise ValueError('Nonfinite sidecar JSON: '+value)


def model_profile(kind, checkpoint=None, *, expected_sha256=None, sidecar=None,
                  expected_sidecar_sha256=None, label=None):
    """Declare initial or explicitly supplied official-trained weights.

    Missing official weights remain missing. Official sidecars retain the
    remote format (updates/init_sha256); an optional sidecar SHA tightens that
    binding. Their format alone does not prove training-data lineage.
    """
    if kind not in ('initial', 'official'):
        raise ValueError('Profile kind must be initial or official')
    if label is not None and not isinstance(label,str):
        raise ValueError('Profile label must be JSON text')
    for value in (expected_sha256, expected_sidecar_sha256):
        if value is not None and not _digest_string(value):
            raise ValueError('Expected asset hashes must be lowercase SHA256')
    expected_bytes = None
    if kind == 'initial':
        spec = world_model_spec()['checkpoint']
        if expected_sha256 is not None and expected_sha256 != spec['sha256']:
            raise ValueError('Initial profile must use the released checkpoint identity')
        expected_sha256, expected_bytes = spec['sha256'], spec['bytes']
        checkpoint = ROOT/'artifacts/frozen-wm-assets/denoiser.pth' if checkpoint is None else checkpoint
        if sidecar is not None or expected_sidecar_sha256 is not None:
            raise ValueError('Released initial weights do not use a trained sidecar')
    elif checkpoint is not None and sidecar is None:
        sidecar = Path(checkpoint).with_name('denoiser.json')
    if kind=='initial':
        label = 'Released initial GPC (not official-trained)'+(' / '+label if label else '')
    return dict(kind=kind, label=label or 'Explicit official-colour checkpoint',
        checkpoint=None if checkpoint is None else str(Path(checkpoint).absolute()),
        expected_sha256=expected_sha256, expected_bytes=expected_bytes,
        sidecar=None if sidecar is None else str(Path(sidecar).absolute()),
        expected_sidecar_sha256=expected_sidecar_sha256, contract=deepcopy(RAW_CONTRACT),
        origin='released_pretrained' if kind=='initial' else 'explicit_official_asset_declaration')


def _asset_path(value):
    path = Path(value).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, 'is_junction', lambda:False)():
            raise ValueError('Asset aliases are not accepted')
    return path


def _file_identity(path):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
        raise ValueError('Asset must be a nonempty ordinary single-link file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if any(getattr(before,key)!=getattr(opened,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns')):
            raise ValueError('Opened asset differs from its path identity')
        for chunk in iter(lambda:stream.read(1024**2), b''):digest.update(chunk)
    after = path.stat()
    fields = ('st_dev','st_ino','st_size','st_mtime_ns')
    if any(getattr(before,key)!=getattr(after,key) for key in fields):
        raise ValueError('Asset changed while its identity was read')
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def inspect_profile(profile):
    """Check only this declared profile; no tensor, fallback, or source import."""
    result = dict(status='invalid', ready=False, reason=None, profile=deepcopy(profile), checkpoint=None, sidecar=None)
    try:
        if not isinstance(profile,dict) or profile.get('kind') not in ('initial','official') or profile.get('contract')!=RAW_CONTRACT:
            raise ValueError('Unknown profile or official raw H1 contract')
        kind, expected = profile['kind'], profile.get('expected_sha256')
        if kind=='official' and expected==LEGACY_T1_SHA256:
            raise ValueError('Known T1 checkpoint500 is not an official-colour checkpoint')
        if kind=='official' and expected==INITIAL_SHA256:
            raise ValueError('Released initial weights must retain the initial profile kind')
        if profile.get('checkpoint') is None:
            result.update(status='missing', reason='No '+('initial' if kind=='initial' else 'explicit official')+' checkpoint was supplied');return result
        path = _asset_path(profile['checkpoint'])
        if not path.exists():
            result.update(status='missing', reason='Checkpoint is missing: '+str(path));return result
        if not _digest_string(expected):
            raise ValueError('An explicit matching checkpoint SHA256 is required')
        if kind=='initial':
            pinned = world_model_spec()['checkpoint']
            if expected!=pinned['sha256'] or profile.get('expected_bytes')!=pinned['bytes']:
                raise ValueError('Initial profile identity differs from the pinned release')
        else:
            if profile.get('sidecar') is None:
                raise ValueError('Explicit official checkpoint needs its denoiser.json sidecar')
            sidecar = _asset_path(profile['sidecar'])
            if not sidecar.exists():
                result.update(status='missing', reason='Official sidecar is missing: '+str(sidecar));return result
            if sidecar.stat().st_size>65536:raise ValueError('Official sidecar exceeds64KiB')
            identity = _file_identity(sidecar); raw = sidecar.read_bytes()
            if _sha(raw)!=identity['sha256']:raise ValueError('Sidecar changed during inspection')
            declared = profile.get('expected_sidecar_sha256')
            if declared is not None and identity['sha256']!=declared:raise ValueError('Sidecar SHA256 differs')
            metadata = json.loads(raw,parse_constant=_reject_constant)
            if (not isinstance(metadata,dict) or type(metadata.get('updates')) is not int or metadata['updates']<0
                    or not _digest_string(metadata.get('init_sha256'))):
                raise ValueError('Official sidecar lacks valid updates/init_sha256')
            if metadata.get('input_transform')=='T1' or metadata.get('input_contract', RAW_CONTRACT['version'])!=RAW_CONTRACT['version']:
                raise ValueError('Sidecar does not declare the official raw input contract')
            result['sidecar'] = dict(identity, metadata=metadata,
                provenance_scope='explicit profile and sidecar/hash; not independent training-data verification')
        identity = _file_identity(path)
        if identity['sha256']!=expected or (profile.get('expected_bytes') is not None and identity['bytes']!=profile['expected_bytes']):
            raise ValueError('Checkpoint bytes/SHA256 differ from the declared profile')
        result.update(status='available', ready=True, reason='Explicit profile assets verified', checkpoint=identity)
    except (OSError,ValueError,TypeError,KeyError) as error:
        result.update(status='invalid', reason=type(error).__name__+': '+str(error))
    return result


def model_action(value):
    """df99f8 helper: clip one pixel target, then map it to [-1,1]."""
    value = float(value)
    if value<0 or value>511:value = min(511.0,max(0.0,value))
    return value/511*2-1


def normalize_actions(actions):
    """df99f8 helper; stored absolute targets are never modified."""
    import numpy as np
    values = np.asarray(actions,dtype=np.float64)
    return np.array([model_action(value) for value in values.reshape(-1)],dtype=np.float32).reshape(values.shape)


def _inputs(images, actions):
    import numpy as np
    image, action = np.asarray(images), np.asarray(actions)
    if (image.shape!=(4,3,96,96) or image.dtype!=np.float32 or not np.isfinite(image).all()
            or (image<0).any() or (image>1).any()):
        raise ValueError('Official H1 needs four float32 CHW96 original-colour frames in [0,1]')
    if (action.shape!=(4,2) or action.dtype.kind not in 'fiu' or not np.isfinite(action).all()
            or (action<0).any() or (action>511).any()):
        raise ValueError('Official H1 needs four finite absolute pixel targets in [0,511]')


def load_denoiser_sampler(source_root, checkpoint, *, profile=None, execute=False):
    """Remote native loader with explicit profile validation before Torch.

    This guard is not training/run authorization; its caller must already own
    that authorization and the process/resource budget. No optimizer or RNG is
    restored. Native architecture, strict state load and sampler are unchanged.
    """
    if execute is not True or profile is None:
        raise PermissionError('Explicit execution and a matching model profile are required')
    inspected = inspect_profile(profile)
    if inspected['status']=='missing':raise FileNotFoundError(inspected['reason'])
    if inspected['status']!='available':raise ValueError(inspected['reason'])
    if _asset_path(checkpoint)!=_asset_path(profile['checkpoint']):
        raise ValueError('Requested checkpoint differs from the verified profile')
    if source_root is None:raise ValueError('Explicit pinned GPC source root is required')
    spec = world_model_spec()
    if (spec['sampler']!=_NATIVE_SAMPLER_SPEC or spec['model']['num_steps_conditioning']!=4
            or spec['model']['img_channels']!=3):
        raise ValueError('Official profile requires the fixed four-frame/three-Euler native contract')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('World-model prediction requires CUDA; no CPU fallback')
    source = load_world_model_source(source_root,spec)
    config = dict(spec['model'])
    denoiser_config = {name:config.pop(name) for name in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
    network = source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**config),**denoiser_config)).float()
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if not all(isinstance(value,torch.Tensor) and torch.isfinite(value).all() for value in payload.values()):
        raise ValueError('Non-tensor or non-finite world-model weights')
    network.load_state_dict(payload,strict=True)
    device = torch.device('cuda')
    network.to(device).eval()
    sampler = source.DiffusionSampler(network,source.DiffusionSamplerConfig(**spec['sampler']))
    sampler.official_profile_inspection = deepcopy(inspected)
    return sampler,device


def predict_next_frame(sampler, images, actions, device, seed=0):
    """Official raw H1; df99f8's output clipping is intentionally preserved."""
    import numpy as np
    _inputs(images,actions)
    import torch
    image = torch.as_tensor(np.asarray(images,dtype=np.float32),device=device)[None]
    action = torch.as_tensor(normalize_actions(actions),device=device)[None]
    if image.shape!=(1,4,3,96,96) or action.shape!=(1,4,2):
        raise ValueError('Prediction expects four96x96 frames and four actions')
    gpu = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(gpu),torch.random.fork_rng(devices=[gpu]),torch.inference_mode():
        torch.cuda.manual_seed(seed)
        prediction,_steps = sampler.sample(image,action)
    return np.clip(prediction[0].detach().cpu().numpy(),0,1)


class OfficialWorldModel:
    """One H1-only native sampler for one explicitly declared asset profile."""
    def __init__(self, profile, source_root, *, execute=False):
        if execute is not True:
            raise PermissionError('Actual model loading requires explicit execution')
        self.profile = deepcopy(profile)
        self.sampler,self.device = load_denoiser_sampler(source_root,self.profile['checkpoint'],
            profile=self.profile,execute=True)
        inspected = self.sampler.official_profile_inspection
        if inspected.get('profile')!=self.profile or inspected.get('ready') is not True:
            raise ValueError('Loader did not retain this verified profile identity')
        self.asset_identity = dict(kind=self.profile['kind'],label=self.profile['label'],
            origin=self.profile['origin'],checkpoint=inspected['checkpoint'],sidecar=inspected['sidecar'],
            contract=deepcopy(RAW_CONTRACT))

    def predict_next(self, images, absolute_actions, seed=5478):
        if self.sampler is None:raise RuntimeError('Official model is closed')
        return predict_next_frame(self.sampler,images,absolute_actions,self.device,seed=seed)

    def close(self):
        self.sampler = self.device = None
