import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock3D(nn.Module):
    """3D残差块，带时间步嵌入"""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 注入时间步嵌入
        time_emb = self.time_emb_proj(F.silu(time_emb))
        # (B, C, 1,1,1)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers, downsample=True):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock3D(in_channels if i == 0 else out_channels,
                       out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                                    stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, time_emb):
        for block in self.resblocks:
            x = block(x, time_emb)
        x = self.downsample(x)
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers, upsample=True):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock3D(in_channels if i == 0 else out_channels,
                       out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3,
                                           stride=2, padding=1, output_padding=1) if upsample else nn.Identity()

    def forward(self, x, time_emb):
        for block in self.resblocks:
            x = block(x, time_emb)
        x = self.upsample(x)
        return x


class UNetModel(nn.Module):
    """
    基于3D卷积的UNet，支持时间步和条件嵌入。
    输出与SpatialTemporalUNet接口一致。
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.vae_latent_channels
        self.model_channels = config.unet_model_channels
        self.channel_mult = config.unet_channel_mult
        self.num_res_blocks = config.unet_num_res_blocks
        self.num_heads = config.unet_num_heads  # 未使用，保留
        self.time_emb_dim = self.model_channels * 4

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # 条件嵌入（文本/多模态）
        self.cond_proj = nn.Linear(config.dit_context_dim, self.time_emb_dim)

        # 输入卷积
        self.conv_in = nn.Conv3d(
            self.in_channels, self.model_channels, kernel_size=3, padding=1)

        # 下采样
        self.down_blocks = nn.ModuleList()
        ch = self.model_channels
        for i, mult in enumerate(self.channel_mult):
            out_ch = self.model_channels * mult
            self.down_blocks.append(
                DownBlock3D(ch, out_ch, self.time_emb_dim,
                            num_layers=self.num_res_blocks,
                            downsample=(i < len(self.channel_mult)-1))
            )
            ch = out_ch

        # 中间块
        self.mid_block = ResBlock3D(ch, ch, self.time_emb_dim)

        # 上采样
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(self.channel_mult)):
            out_ch = self.model_channels * mult
            self.up_blocks.append(
                UpBlock3D(ch + out_ch, out_ch, self.time_emb_dim,
                          num_layers=self.num_res_blocks,
                          upsample=(i < len(self.channel_mult)-1))
            )
            ch = out_ch

        # 输出卷积
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, self.model_channels),
            nn.SiLU(),
            nn.Conv3d(self.model_channels, self.in_channels,
                      kernel_size=3, padding=1)
        )

    def forward(self, x, t, cond, prev_state=None, entity_mask=None, return_state=False):
        """
        x: (B, C, T, H, W)
        t: (B,) 时间步索引
        cond: (B, L, D) 条件嵌入，取均值后与时间步融合
        """
        # 时间步嵌入
        t_emb = self._timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # 条件嵌入
        cond_emb = cond.mean(dim=1)  # (B, D)
        cond_emb = self.cond_proj(cond_emb)
        t_emb = t_emb + cond_emb

        # 输入卷积
        h = self.conv_in(x)

        # 下采样，保存跳跃连接
        skips = []
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        # 中间
        h = self.mid_block(h, t_emb)

        # 上采样，拼接跳跃连接
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        # 输出
        out = self.conv_out(h)

        # 返回与SpatialTemporalUNet兼容的格式
        if return_state:
            return out, None, None
        else:
            return out, None

    def _timestep_embedding(self, timesteps, dim):
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
