"""Original native GPC inference sampler; no training entrypoint."""

def torch_native_sample(model,source,sampler_settings):
    """Called only inside the authorized clock. Upstream sampler/math unchanged."""
    import numpy as np
    import torch
    sampler=source.DiffusionSampler(model,source.DiffusionSamplerConfig(**sampler_settings))
    device=next(model.parameters()).device
    def sample(histories,actions,*,seed):
        devices=[device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices),torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
            image=torch.as_tensor(np.asarray(histories,dtype=np.float32).copy(),device=device)
            action=torch.as_tensor(np.asarray(actions,dtype=np.float32).copy(),device=device)
            result,_=sampler.sample(image,action)
        return result.detach().float().cpu().numpy()
    return sample
