"""A fixed, non-training denoising diagnostic from baseline development scenes."""

import torch

from .forward_policy import forward_noise


class DenoisingProbe:
    def __init__(self, state=None):
        self.rows = list(state or [])

    def capture(self, shape, seed, condition, plan):
        # First plan of the first baseline scene per shape. No extra interaction
        # or sampling; never replace it with the updated actor's plans.
        if any(row["shape"] == shape for row in self.rows):
            return
        generator = torch.Generator().manual_seed(20261023+sum(map(ord, shape)))
        clean = plan.detach().cpu().repeat(4, 1, 1)
        self.rows.append({"shape": shape, "scene_seed": seed,
            "condition": condition.detach().cpu().repeat(4, 1), "clean": clean,
            "noise": torch.randn(clean.shape, generator=generator),
            "timesteps": torch.tensor([0, 25, 50, 99], dtype=torch.long)})

    @torch.no_grad()
    def evaluate(self, agent):
        values = {}
        for row in self.rows:
            clean, noise, t, condition = (row[k].to(agent.device)
                for k in ("clean", "noise", "timesteps", "condition"))
            x = forward_noise(agent.network.noise_scheduler, clean, noise, t)
            prediction = agent.predict(x, t, condition)
            reference = agent.predict(x, t, condition, "reference")
            mse = (prediction-noise).square().flatten(1).mean(1)
            drift = (prediction-reference).square().mean()
            prefix = f"probe/{row['shape']}"
            values[prefix+"/fixed_noise_mse"] = float(mse.mean())
            values[prefix+"/initial_policy_drift_mse"] = float(drift)
            for timestep, loss in zip(row["timesteps"].tolist(), mse.tolist()):
                values[f"{prefix}/noise_mse_t{timestep}"] = loss
        return values
