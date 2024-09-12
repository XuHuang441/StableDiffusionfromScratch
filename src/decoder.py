import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将x的形状从(Batch_size, features, height, width)变为(Batch_size, sequence_len, features)
        # 来做attention

        residue = x

        n, c, h, w = x.shape
        x = x.view(n, c, h*w)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # Restore the shape
        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return x + residue



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 当kernel_size=3, padding=1时，可以保证输入和输出的图像大小不变

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_decoder(nn.Sequential):
    """
    将图片的恢复成原样，即channel数量减少到3（rgb），图片大小恢复到512*512
    """
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height/8, width/8) -> (Batch_size, 512, Height/4, width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height/4, width/4) -> (Batch_size, 512, Height/2, width/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height/2, width/2) -> (Batch_size, 256, Height, width)
            nn.Upsample(scale_factor=2),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (Batch_size, 256, Height, width) -> (Batch_size, 3, Height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:(Batch_size, 4, height/ 8, width/8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # x:(Batch_size, 3, height, width)
        return x