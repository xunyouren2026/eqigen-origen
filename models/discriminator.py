import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoDiscriminator(nn.Module):
    """3D CNN判别器，输入视频 (B, C, T, H, W)，输出真实性概率"""

    def __init__(self, in_channels=3, base_channels=64, num_layers=4):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(
                nn.Conv3d(channels, out_channels, kernel_size=4,
                          stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = out_channels

        self.features = nn.Sequential(*layers)

        # 输出层
        self.final_conv = nn.Conv3d(
            channels, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x: (B, C, T, H, W)
        features = self.features(x)
        out = self.final_conv(features)
        # 全局平均池化得到标量
        out = out.mean(dim=(2, 3, 4))
        return out
