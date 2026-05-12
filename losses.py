"""
Loss functions for SeisCodec.
- Multi-scale STFT loss (replaces Mel loss, more suitable for seismic signals)
- GAN loss with feature matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for seismic waveforms.
    Computes L1 distance on log-magnitude STFT at multiple resolutions.

    Window lengths are chosen for seismic data at 100 Hz:
    - Small windows (32, 64): capture high-frequency transients (P/S arrivals)
    - Large windows (256, 512): capture low-frequency envelope
    """
    def __init__(
        self,
        window_lengths: list = [32, 64, 128, 256, 512],
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        clamp_eps: float = 1e-5,
    ):
        super().__init__()
        self.window_lengths = window_lengths
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.clamp_eps = clamp_eps

    def _stft(self, x, window_length):
        hop_length = window_length // 4
        # x: [B, C, T] → process each channel
        b, c, t = x.shape
        x = x.reshape(b * c, t)
        window = torch.hann_window(window_length, device=x.device)
        stft = torch.stft(
            x, n_fft=window_length, hop_length=hop_length,
            win_length=window_length, window=window,
            return_complex=True,
        )
        mag = stft.abs()  # [B*C, F, T]
        mag = mag.reshape(b, c, mag.shape[1], mag.shape[2])
        return mag

    def forward(self, x_hat, x):
        loss = 0.0
        for wl in self.window_lengths:
            mag_x = self._stft(x, wl)
            mag_xhat = self._stft(x_hat, wl)

            if self.log_weight > 0:
                loss += self.log_weight * F.l1_loss(
                    mag_xhat.clamp(self.clamp_eps).log10(),
                    mag_x.clamp(self.clamp_eps).log10(),
                )
            if self.mag_weight > 0:
                loss += self.mag_weight * F.l1_loss(mag_xhat, mag_x)
        return loss


class WaveformLoss(nn.Module):
    """Simple multi-scale waveform L1 loss."""
    def __init__(self, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales

    def forward(self, x_hat, x):
        loss = 0.0
        for s in self.scales:
            if s > 1:
                x_ds = F.avg_pool1d(x, s, s)
                xhat_ds = F.avg_pool1d(x_hat, s, s)
            else:
                x_ds = x
                xhat_ds = x_hat
            loss += F.l1_loss(xhat_ds, x_ds)
        return loss / len(self.scales)


class HighFrequencyLoss(nn.Module):
    """
    高频重建损失，强调波形突变处（P/S波到达等）的重建质量。

    三个组件：
    1. 多阶时域差分 L1：一阶捕捉快速变化，二阶捕捉尖锐脉冲
    2. 频域高频加权：对 STFT 高频 bin 施加递增权重
    3. 变化率加权 L1：用真实波形的局部变化率作为连续权重，
       变化剧烈处（到达、突变）自动获得更大惩罚，无需显式检测

    参数说明：
        diff_orders: 差分阶数列表
        diff_weight: 时域差分损失权重
        spectral_weight: 频域高频加权损失权重
        grad_weight: 变化率加权损失权重
        grad_smooth: 计算局部变化率的滑动窗口大小(采样点)
        stft_window: 频域损失的STFT窗口大小
        freq_power: 频率加权的幂次, 1=线性, 2=二次
    """
    def __init__(
        self,
        diff_orders: list = [1, 2],
        diff_weight: float = 1.0,
        spectral_weight: float = 1.0,
        grad_weight: float = 1.0,
        grad_smooth: int = 5,
        stft_window: int = 64,
        freq_power: float = 1.0,
    ):
        super().__init__()
        self.diff_orders = diff_orders
        self.diff_weight = diff_weight
        self.spectral_weight = spectral_weight
        self.grad_weight = grad_weight
        self.grad_smooth = grad_smooth
        self.stft_window = stft_window
        self.freq_power = freq_power

    def _diff(self, x, order):
        """多阶差分: order=1 即一阶差分, order=2 即二阶差分"""
        for _ in range(order):
            x = x[..., 1:] - x[..., :-1]
        return x

    def _temporal_diff_loss(self, x_hat, x):
        """多阶差分 L1 损失"""
        loss = 0.0
        for order in self.diff_orders:
            diff_x = self._diff(x, order)
            diff_xhat = self._diff(x_hat, order)
            loss += F.l1_loss(diff_xhat, diff_x)
        return loss / len(self.diff_orders)

    def _spectral_highfreq_loss(self, x_hat, x):
        """频域高频加权损失：高频 bin 权重更大"""
        b, c, t = x.shape
        hop = self.stft_window // 4
        window = torch.hann_window(self.stft_window, device=x.device)

        x_flat = x.reshape(b * c, t)
        xhat_flat = x_hat.reshape(b * c, t)

        mag_x = torch.stft(
            x_flat, n_fft=self.stft_window, hop_length=hop,
            win_length=self.stft_window, window=window,
            return_complex=True,
        ).abs()
        mag_xhat = torch.stft(
            xhat_flat, n_fft=self.stft_window, hop_length=hop,
            win_length=self.stft_window, window=window,
            return_complex=True,
        ).abs()

        n_freq = mag_x.shape[1]
        # 线性递增权重: 低频权重小, 高频权重大
        freq_weights = torch.arange(1, n_freq + 1, device=x.device, dtype=x.dtype)
        freq_weights = freq_weights.pow(self.freq_power)
        freq_weights = freq_weights / freq_weights.mean()  # 归一化使均值为1
        freq_weights = freq_weights[None, :, None]  # [1, F, 1]

        weighted_diff = freq_weights * (mag_xhat - mag_x).abs()
        return weighted_diff.mean()

    def _gradient_weighted_loss(self, x_hat, x):
        """
        变化率加权 L1 损失：
        用真实波形的局部变化率作为连续权重。
        变化剧烈的地方（P/S波到达、突变）权重自然大，
        平缓的地方权重自然小，无需阈值或显式检测。
        """
        # 一阶差分绝对值 → 瞬时变化率 [B, C, T-1]
        dx = (x[..., 1:] - x[..., :-1]).abs()

        # 局部平滑，避免单点噪声主导权重
        if self.grad_smooth > 1:
            dx_smooth = F.avg_pool1d(
                dx.reshape(-1, 1, dx.shape[-1]),
                kernel_size=self.grad_smooth,
                stride=1,
                padding=self.grad_smooth // 2,
            ).reshape(dx.shape)[..., :dx.shape[-1]]
        else:
            dx_smooth = dx

        # 归一化到均值=1，使权重不改变损失的整体量级
        weights = dx_smooth / (dx_smooth.mean(dim=-1, keepdim=True) + 1e-8)

        # 补回长度到 T (差分少一个点，首位补1.0)
        weights = F.pad(weights, (1, 0), value=1.0)  # [B, C, T]

        # 加权 L1
        weighted_error = weights * (x_hat - x).abs()
        return weighted_error.mean()

    def forward(self, x_hat, x):
        loss = 0.0
        if self.diff_weight > 0:
            loss += self.diff_weight * self._temporal_diff_loss(x_hat, x)
        if self.spectral_weight > 0:
            loss += self.spectral_weight * self._spectral_highfreq_loss(x_hat, x)
        if self.grad_weight > 0:
            loss += self.grad_weight * self._gradient_weighted_loss(x_hat, x)
        return loss


class GANLoss(nn.Module):
    """
    GAN losses with Relativistic average Hinge (RaHinge) adversarial loss
    + feature matching.

    RaHinge discriminator:
        L_D = [ ReLU(1 - (D(real) - mean(D(fake)))) + ReLU(1 + (D(fake) - mean(D(real)))) ] / 2
    RaHinge generator:
        L_G = [ ReLU(1 + (D(real) - mean(D(fake)))) + ReLU(1 - (D(fake) - mean(D(real)))) ] / 2
    """
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def discriminator_loss(self, fake, real):
        d_fake = self.discriminator(fake.detach())
        d_real = self.discriminator(real)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            d_real_out = x_real[-1]
            d_fake_out = x_fake[-1]
            loss_d += (F.relu(1.0 - (d_real_out - d_fake_out.mean())).mean() +
                       F.relu(1.0 + (d_fake_out - d_real_out.mean())).mean()) / 2
        return loss_d

    def generator_loss(self, fake, real):
        d_fake = self.discriminator(fake)
        with torch.no_grad():
            d_real = self.discriminator(real)

        loss_g = 0
        for x_fake, x_real in zip(d_fake, d_real):
            d_real_out = x_real[-1]  # already detached via no_grad
            d_fake_out = x_fake[-1]
            loss_g += (F.relu(1.0 + (d_real_out - d_fake_out.mean())).mean() +
                       F.relu(1.0 - (d_fake_out - d_real_out.mean())).mean()) / 2

        loss_feat = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feat += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

        return loss_g, loss_feat