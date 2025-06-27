import torch
from torch import nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_emb_dim=256, dropout=0.1):
        super(ResidualBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels, affine=True),
        )
        self.block12 = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.residual_conv1 = (
            nn.Conv2d(channels, channels, kernel_size=1)
            if channels != channels else nn.Identity()
        )
        self.time_embed1 = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x, t_emb, x_cond):
        residual = self.residual_conv1(x)

        out = self.block1(x)
        out = out + self.time_embed1(t_emb)[:, :, None, None]
        out = self.block12(out)

        return residual + out


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return self.mlp(emb)


class ClassEmbeddingBlock(nn.Module):
    def __init__(self, num_classes, class_emb_size):
        super().__init__()
        self.embed = nn.Embedding(num_classes, class_emb_size)

    def forward(self, y):
        return self.embed(y)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm1(x).view(b, c, h * w).transpose(1, 2)

        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out

        x = x + self.ff(self.norm2(x))

        x = x.transpose(1, 2).view(b, c, h, w)
        return x + residual

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, context_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        context_dim = context_dim or embed_dim

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)

        self.kv_proj = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ff_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(embed_dim, embed_dim)
        )



    def forward(self, x, t_emb=None):
        residual = x
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.layernorm(x_seq)

        if t_emb is not None:
            context = t_emb.unsqueeze(1)
            context = self.kv_proj(context)
        else:
            context = x_seq

        x_attn, _ = self.attn(
            query=x_seq,
            key=context,
            value=context
        )
        x_attn = self.ff_proj(x_attn) + x_attn
        x_attn = x_attn.transpose(1, 2).view(b, c, h, w)
        x_attn = x_attn + residual
        return x_attn

class EncodeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim=256):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # downsample first
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),  # then transform
        )
        self.residual = nn.ModuleList([
            ResidualBlock(output_channels, time_emb_dim) for _ in range(2)
        ])
        self.self_attention = nn.ModuleList([
            AttentionBlock(output_channels) for _ in range(2)
        ])
        self.emb_layer1 = nn.Sequential(
            nn.Linear(time_emb_dim, input_channels),
            nn.SiLU(),
            nn.Linear(input_channels, input_channels),
        )
        self.emb_layer2 = nn.Sequential(
            nn.Linear(time_emb_dim, output_channels),
            nn.SiLU(),
            nn.Linear(output_channels, output_channels),
        )

    def forward(self, x, t_emb, x_cond):
        t_emb1 = self.emb_layer1(t_emb)
        t_emb1 = t_emb1[:, :, None, None]

        x = self.block1(x)

        for res, self_attn in zip(self.residual, self.self_attention):
            x = res(x, t_emb, x_cond)
            if x.shape[-1] == 8 and x.shape[-2] == 8:
                x = self_attn(x)

        t_emb2 = self.emb_layer2(t_emb)
        t_emb2 = t_emb2[:, :, None, None]

        return x + t_emb2


class DecodeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim=128):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # upsample first
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels * 3, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(output_channels, output_channels // 2, kernel_size=3, padding=1)
        )
        self.residual = nn.ModuleList([
            ResidualBlock(output_channels // 2, time_emb_dim) for _ in range(2)
        ])
        self.self_attention = nn.ModuleList([
            AttentionBlock(output_channels // 2) for _ in range(2)
        ])
        self.emb_layer1 = nn.Sequential(
            nn.Linear(time_emb_dim, output_channels),
            nn.SiLU(),
            nn.Linear(output_channels, output_channels // 2),
        )


    def forward(self, x, skip, t_emb, x_cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        t_emb1 = self.emb_layer1(t_emb)
        t_emb1 = t_emb1[:, :, None, None]

        for res, self_attn in zip(self.residual, self.self_attention):
            x = res(x, t_emb, x_cond)
            if x.shape[-1] == 8 and x.shape[-2] == 8:
                x = self_attn(x)


        return x + t_emb1



class BottleneckLayer(nn.Module):
    def __init__(self, channel, time_emb_dim=256):
        super().__init__()
        self.residual = nn.ModuleList([
            ResidualBlock(channel, time_emb_dim) for _ in range(2)
        ])
        self.self_attention = nn.ModuleList([
            AttentionBlock(channel) for _ in range(2)
        ])

        self.emb_layer1 = nn.Sequential(
            nn.Linear(time_emb_dim, channel),
            nn.SiLU(),
            nn.Linear(channel, channel),
        )

    def forward(self, x, t_emb, x_cond):
        t_emb1 = self.emb_layer1(t_emb)
        t_emb1 = t_emb1[:, :, None, None]

        for res, self_attn in zip(self.residual, self.self_attention):
            x = res(x, t_emb, x_cond)
            x = self_attn(x)

        return x + t_emb1



def compute_downsample_layers(h, w, target=4):
    ds_h = int(math.log2(h // target))
    ds_w = int(math.log2(w // target))
    assert ds_h == ds_w, "Width and height must downsample to 4x4 symmetrically"
    return ds_h



class Unet_Diffusion_Model_1(nn.Module):
    def __init__(self, input_resolution, time_emb_dim, num_classes):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)


        self.class_emb = ClassEmbeddingBlock(num_classes, time_emb_dim)
        self.null_conditioning = nn.Parameter(torch.zeros(1, time_emb_dim))

        self.input_resolution = input_resolution
        self.num_downsamples = compute_downsample_layers(*input_resolution)

        base_channel = 64

        channels = [base_channel] + [(base_channel * 2) * (2 ** i) for i in range(self.num_downsamples)]
        print(f'Channels for layers: {channels}')
        self.conv_in = nn.Sequential(
            nn.Conv2d(3 + time_emb_dim * 2, base_channel, kernel_size=3, padding=(1, 1)),
            nn.SiLU(inplace=False),
        )
        self.conv_in_res = nn.ModuleList([
            ResidualBlock(base_channel, time_emb_dim) for _ in range(2)
        ])

        self.encode_blocks = nn.ModuleList([
            EncodeBlock(channels[i], channels[i + 1], time_emb_dim)
            for i in range(self.num_downsamples)
        ])

        self.bottleneck_block = BottleneckLayer(channels[-1], time_emb_dim )

        self.decode_blocks = nn.ModuleList([
            DecodeBlock(channels[i], channels[i + 1], time_emb_dim)
            for i in reversed(range(self.num_downsamples))
        ])

        self.conv_out_res = nn.ModuleList([
            ResidualBlock(channels[0], time_emb_dim) for _ in range(2)
        ])
        self.conv_out = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Conv2d(channels[0], 3, kernel_size=1),
        )

        self.emb_layer1 = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, 3),
        )
        self.emb_layer2 = nn.Sequential(
            nn.Linear(time_emb_dim , time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, base_channel),
        )


    def forward(self, x, t, y=None):

        if y is None:
            x_cond = self.null_conditioning.expand(x.size(0), -1)
        else:
            x_cond = self.class_emb(y)

        t_emb = self.time_embedding(t)

        bs, ch, w, h = x.shape
        t_emb_channel = x_cond.view(bs, t_emb.shape[1], 1, 1).expand(bs, t_emb.shape[1], w, h)
        x_cond_channel = x_cond.view(bs, x_cond.shape[1], 1, 1).expand(bs, x_cond.shape[1], w, h)
        x = torch.cat((x, t_emb_channel, x_cond_channel), 1)

        t_emb = t_emb + x_cond

        enc_feats = []
        x = self.conv_in(x)
        for res in self.conv_in_res:
            x = res(x, t_emb, x_cond)
        t_emb2 = self.emb_layer2(t_emb)
        t_emb2 = t_emb2[:, :, None, None]
        x = x + t_emb2

        enc_feats.append(x)
        for encode in self.encode_blocks:
            x = encode(x, t_emb, x_cond)
            enc_feats.append(x)

        for _ in range(2):
            x = self.bottleneck_block(x, t_emb, x_cond)

        assert x.shape[-1] == 4 and x.shape[-2] == 4, f"Expected 4x4 bottleneck, got {x.shape[-2:]}"

        enc_feats.pop()

        for decode, skip_x in zip(self.decode_blocks, reversed(enc_feats)):
            x = decode(x, skip_x, t_emb, x_cond)

        for res in self.conv_out_res:
            x = res(x, t_emb, x_cond)
        x = self.conv_out(x)

        return x

