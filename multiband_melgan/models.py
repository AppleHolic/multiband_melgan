import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from pytorch_sound.models import register_model, register_model_architecture
from typing import List


#
# Make blocks
#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


#
# Build Generator
#
@register_model('generator')
class Generator(nn.Module):

    def __init__(self, mel_dim: int = 80, dim: int = 384, out_dim: int = 4, res_kernels: List[int] = [4, 4, 4]):
        super().__init__()
        # make in conv
        self.in_conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            WNConv1d(mel_dim, dim, kernel_size=7, padding=0)
        )

        # body
        self.res_stack = nn.ModuleList()
        self.res_params = res_kernels
        res_dilations = [3 ** i for i in range(4)]

        for idx, ratio in enumerate(self.res_params):
            stack = nn.Sequential(
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    dim,
                    dim // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    padding=ratio // 2 + ratio % 2,
                    output_padding=ratio % 2,
                ),
                *[ResnetBlock(dim // 2, 3, dilation=res_dilations[i]) for i in range(4)]
            )
            self.res_stack.append(stack)

            dim //= 2

        # out
        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(dim, out_dim, kernel_size=7, padding=0),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(mel)
        for stack in self.res_stack:
            x = stack(x)
        return self.out(x)


class DiscriminatorBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(
                WNConv1d(1, 16, 15, padding=7),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                WNConv1d(16, 64, 41, stride=4, groups=4, padding=4 * 5),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                WNConv1d(64, 256, 41, stride=4, groups=16, padding=4 * 5),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                WNConv1d(256, 512, 41, stride=4, groups=64, padding=4 * 5),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                WNConv1d(512, 512, 5, padding=2),
                nn.LeakyReLU(0.2)
            ),
            WNConv1d(512, 1, 3, padding=1)
        ])

    def forward(self, x):
        results = []
        for module in self.module_list:
            x = module(x)
            results.append(x)
        return results


@register_model('discriminator')
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([DiscriminatorBlock()] * 3)
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        results = []
        for idx, block in enumerate(self.blocks):
            results.extend(block(x))
            if idx < len(self.blocks) - 1:
                x = self.downsample(x)
        return results


@register_model_architecture('generator', 'generator_mb_16k')
def generator_mb():
    return {
        'mel_dim': 80,
        'dim': 384,
        'out_dim': 4,
        'res_kernels': [2, 5, 5]
    }


@register_model_architecture('generator', 'generator_mb')
def generator_mb():
    return {
        'mel_dim': 80,
        'dim': 384,
        'out_dim': 4,
        'res_kernels': [2, 4, 8]
    }


@register_model_architecture('discriminator', 'discriminator_base')
def discriminator_base():
    return {}
