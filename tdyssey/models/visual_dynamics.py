"""Small image/action-conditioned predictor, separate from Diffusion Policy."""

import torch
from torch import nn
from torch.nn import functional as F


class VisualDynamics(nn.Module):
    """Autoregressive two-frame image/proprioception history; no physical-state input."""

    def __init__(self, stats):
        super().__init__()
        self.register_buffer("motion_mean", torch.tensor(stats["motion_mean"], dtype=torch.float32))
        self.register_buffer("motion_scale", torch.tensor(stats["motion_scale"], dtype=torch.float32))
        def block(inputs, outputs, stride=1):
            return nn.Sequential(nn.Conv2d(inputs,outputs,3,stride,padding=1), nn.GroupNorm(4,outputs), nn.SiLU())
        self.enc1, self.enc2, self.enc3 = block(6,16,2), block(16,32,2), block(32,64,2)
        self.condition = nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.middle = block(96,64)
        self.dec2, self.dec1, self.dec0 = block(96,32), block(48,16), block(16,16)
        self.image_head = nn.Conv2d(16,3,3,padding=1)
        self.motion_head = nn.Sequential(nn.Flatten(),nn.Linear(64*12*12,128),nn.SiLU(),nn.Linear(128,5))
        self.scalar_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.image_head.weight); nn.init.zeros_(self.image_head.bias)
        nn.init.zeros_(self.motion_head[-1].weight); nn.init.zeros_(self.motion_head[-1].bias)
        nn.init.zeros_(self.scalar_head[-1].weight)
        rates = torch.tensor([stats["mean_reward"],stats["contact_frequency"]]).clamp(.001,.999)
        with torch.no_grad():
            self.scalar_head[-1].bias.copy_(torch.logit(rates))

    def step(self, images, agent_pos, goal_pose, action):
        if images.shape[1:] != (2,3,96,96) or agent_pos.shape[1:] != (2,2) or action.shape[1:] != (2,):
            raise ValueError("Expected two official image/proprioception frames and one pixel action")
        first = self.enc1(images.flatten(1,2))
        second, third = self.enc2(first), None
        third = self.enc3(second)
        goal = goal_pose[:, -1]
        condition = torch.cat((agent_pos.flatten(1)/256-1,
            (action-agent_pos[:, -1])/256, goal[:, :2]/256-1,
            torch.sin(goal[:, 2:3]), torch.cos(goal[:, 2:3])), dim=1)
        condition = self.condition(condition)[:, :, None, None].expand(-1,-1,12,12)
        hidden = self.middle(torch.cat((third,condition),dim=1))
        decoded = self.dec2(torch.cat((F.interpolate(hidden,scale_factor=2,mode="bilinear",align_corners=False),second),dim=1))
        decoded = self.dec1(torch.cat((F.interpolate(decoded,scale_factor=2,mode="bilinear",align_corners=False),first),dim=1))
        decoded = self.dec0(F.interpolate(decoded,scale_factor=2,mode="bilinear",align_corners=False))
        image = (images[:, -1]+self.image_head(decoded)).clamp(0,1)
        normalized_motion = self.motion_head(hidden)
        motion = normalized_motion*self.motion_scale+self.motion_mean
        scalars = self.scalar_head(hidden)
        return {"image":image,"agent_pos":agent_pos[:, -1]+motion[:, :2],
                "motion":motion,"normalized_motion":normalized_motion,
                "reward":scalars[:, 0].sigmoid(),"contact_logit":scalars[:, 1]}

    def forward(self, observations, actions):
        images, agent, goals = (observations[key] for key in ("image","agent_pos","goal_pose"))
        outputs = []
        for action in actions.unbind(1):
            prediction = self.step(images,agent,goals,action)
            outputs.append(prediction)
            # Predictions, never future ground truth, populate the next history.
            images = torch.cat((images[:, 1:],prediction["image"][:, None]),dim=1)
            agent = torch.cat((agent[:, 1:],prediction["agent_pos"][:, None]),dim=1)
        return {key:torch.stack([row[key] for row in outputs],dim=1) for key in outputs[0]}
