"""
SeisCodec: Descript Audio Codec adapted for 3-component seismic waveforms.

Input:  [B, 3, 6000]  (E/N/Z components, 100Hz, 60s)
Output: [B, 3, 6000]  (reconstructed waveforms)

encoder_rates = [2, 2, 4, 4] → hop_length = 128
latent frames  = 6000 / 128  = 46 (with padding to 6016 → 47)
"""

import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


# ============================================================================
# Layers
# ============================================================================

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


# ============================================================================
# Vector Quantization
# ============================================================================

class VectorQuantize(nn.Module):
    """
    VQ with factorized codes and L2-normalized lookup (from Improved VQGAN).
    """
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = z_e + (z_q - z_e).detach()  # straight-through estimator
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """Residual Vector Quantization (from SoundStream)."""
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim] * n_codebooks
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList([
            VectorQuantize(input_dim, codebook_size, codebook_dim[i])
            for i in range(n_codebooks)
        ])
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes


# ============================================================================
# Encoder / Decoder
# ============================================================================

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2, dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_in: int = 3,
        d_model: int = 64,
        strides: list = [2, 4, 4, 4],
        d_latent: int = 64,
    ):
        super().__init__()
        self.block = [WNConv1d(d_in, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim, output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out: int = 3):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Discriminator
# ============================================================================

def WNConv1d_D(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d_D(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    """Multi-Period Discriminator."""
    def __init__(self, period, in_channels=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            WNConv2d_D(in_channels, 32, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d_D(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d_D(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d_D(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d_D(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = WNConv2d_D(1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False)

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []
        x = self.pad_to_period(x)
        b, c, t = x.shape
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class MSD(nn.Module):
    """Multi-Scale Discriminator."""
    def __init__(self, rate: int = 1, in_channels=3):
        super().__init__()
        self.rate = rate
        self.convs = nn.ModuleList([
            WNConv1d_D(in_channels, 16, 15, 1, padding=7),
            WNConv1d_D(16, 64, 41, 4, groups=4, padding=20),
            WNConv1d_D(64, 256, 41, 4, groups=16, padding=20),
            WNConv1d_D(256, 1024, 41, 4, groups=64, padding=20),
            WNConv1d_D(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = WNConv1d_D(1024, 1, 3, 1, padding=1, act=False)

    def forward(self, x):
        if self.rate > 1:
            x = F.avg_pool1d(x, self.rate, self.rate)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        periods: list = [2, 3, 5, 7, 11],
        rates: list = [1, 2, 4],
    ):
        super().__init__()
        discs = []
        discs += [MPD(p, in_channels) for p in periods]
        discs += [MSD(r, in_channels) for r in rates]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, x):
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


# ============================================================================
# SeisCodec
# ============================================================================

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class SeisCodec(nn.Module):
    """
    DAC adapted for 3-component seismic waveforms.

    Default config for 100 Hz seismic data:
        encoder_rates = [2, 4, 4, 4]  → hop_length = 128
        Input [B, 3, 6000] → latent [B, 1024, 47] → codes [B, 9, 47]
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 2, 4, 4],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [4, 4, 2, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.5,
        sample_rate: int = 100,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = int(np.prod(encoder_rates))

        self.encoder = Encoder(in_channels, encoder_dim, encoder_rates, latent_dim)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.decoder = Decoder(latent_dim, decoder_dim, decoder_rates, d_out=out_channels)
        self.apply(init_weights)

    def preprocess(self, x):
        length = x.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        x = F.pad(x, (0, right_pad))
        return x

    def encode(self, x, n_quantizers=None):
        z = self.encoder(x)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, n_quantizers=None):
        length = x.shape[-1]
        x = self.preprocess(x)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(x, n_quantizers)
        x_hat = self.decode(z)
        return {
            "audio": x_hat[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


if __name__ == "__main__":
    model = SeisCodec()
    x = torch.randn(2, 3, 6000)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out['audio'].shape}")
    print(f"Codes shape:  {out['codes'].shape}")
    print(f"Z shape:      {out['z'].shape}")
    print(f"Latent dim:   {model.latent_dim}")
    print(f"Hop length:   {model.hop_length}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters:   {n_params:.2f}M")
