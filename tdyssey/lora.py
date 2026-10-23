"""Small native-PyTorch low-rank updates for ordinary Conv1d kernels."""

import torch
from torch import nn
from torch.nn import functional as F


class LowRankKernel(nn.Module):
    def __init__(self, convolution, rank, alpha):
        super().__init__()
        width = convolution.in_channels*convolution.kernel_size[0]
        self.rank = min(rank, width, convolution.out_channels)
        self.scale = alpha/self.rank
        self.a = nn.Parameter(convolution.weight.new_empty(self.rank, width))
        self.b = nn.Parameter(convolution.weight.new_zeros(convolution.out_channels, self.rank))
        nn.init.normal_(self.a, std=.02)


class LoRAConv1d(nn.Module):
    def __init__(self, base, rank=8, alpha=8.):
        super().__init__()
        if base.groups != 1 or base.padding_mode != "zeros":
            raise ValueError("LoRA supports ungrouped zero-padded Conv1d only")
        self.base = base.requires_grad_(False)
        self.delta = LowRankKernel(base, rank, alpha)
        self.enabled = True

    def forward(self, x):
        output = self.base(x)
        if not self.enabled:
            return output
        # Two convolutions equal a flattened-kernel update B@A. Base bias stays
        # frozen; no full dense delta kernel/optimizer state is saved.
        low = F.conv1d(x, self.delta.a.reshape(self.delta.rank, self.base.in_channels, -1),
                       stride=self.base.stride, padding=self.base.padding, dilation=self.base.dilation)
        update = F.conv1d(low, self.delta.b[:, :, None])
        return output+self.delta.scale*update


def inject_lora(module, rank, alpha):
    """Deterministic traversal, replacing Conv1d only (not ConvTranspose1d)."""
    layers = []

    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv1d):
                layer = LoRAConv1d(child, rank, alpha)
                setattr(parent, name, layer)
                layers.append(layer)
            else:
                visit(child)

    visit(module)
    if not layers:
        raise ValueError("No compatible Conv1d layers found")
    return layers
