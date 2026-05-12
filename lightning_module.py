"""
PyTorch Lightning module for training SeisCodec.
Two optimizers: generator (encoder + quantizer + decoder) and discriminator.
Includes EMA (Exponential Moving Average) of generator weights.
"""

import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import SeisCodec, Discriminator, init_weights
from losses import MultiScaleSTFTLoss, WaveformLoss, HighFrequencyLoss, GANLoss


# ============================================================================
# EMA
# ============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer step:
        ema.update()
        # to get EMA weights for inference/saving:
        ema_state = ema.state_dict()
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        self.model = model

        # Initialize shadow params as copies
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow params with current model params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Replace model params with EMA shadow params (for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original params after apply_shadow."""
        for name, param in self.model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name])
        self.original = {}

    def state_dict(self):
        """Return EMA shadow weights as state dict."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        """Load EMA shadow weights."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ============================================================================
# Lightning Module
# ============================================================================

class SeisCodecLightning(pl.LightningModule):
    """
    Lightning wrapper for SeisCodec training.

    Loss composition:
        L_total = λ_stft * L_stft
                + λ_wave * L_waveform
                + λ_hf  * L_high_frequency
                + λ_adv  * L_gen
                + λ_feat * L_feature_matching
                + λ_commit * L_commitment
                + λ_codebook * L_codebook
    """
    def __init__(
        self,
        # Model
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 2, 4, 4],
        decoder_dim: int = 1536,
        decoder_rates: list = [4, 4, 2, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        sample_rate: int = 100,
        # Discriminator
        disc_periods: list = [2, 3, 5, 7, 11],
        disc_rates: list = [1, 2, 4],
        # Loss weights
        lambda_stft: float = 15.0,
        lambda_wave: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat: float = 2.0,
        lambda_commit: float = 0.25,
        lambda_codebook: float = 1.0,
        lambda_hf: float = 5.0,
        # Optimizer
        lr: float = 1e-4,
        betas: tuple = (0.8, 0.99),
        # Training
        disc_start_step: int = 10000,
        # EMA
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # manual optimization for 2 optimizers

        # Generator
        self.model = SeisCodec(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rate=sample_rate,
        )

        # Discriminator
        self.discriminator = Discriminator(
            in_channels=in_channels,
            periods=disc_periods,
            rates=disc_rates,
        )

        # Losses
        self.stft_loss = MultiScaleSTFTLoss(
            window_lengths=[32, 64, 128, 256, 512],
        )
        self.wave_loss = WaveformLoss(scales=[1, 2, 4])
        self.hf_loss = HighFrequencyLoss(
            diff_orders=[1, 2],
            diff_weight=1.0,
            spectral_weight=1.0,
            grad_weight=1.0,
            grad_smooth=5,        # 局部平滑窗口, 0.05s @ 100Hz
            stft_window=64,
            freq_power=1.0,
        )
        self.gan_loss = GANLoss(self.discriminator)

        # EMA (initialized in on_fit_start after model is on device)
        self.ema_decay = ema_decay
        self.ema = None

        # Loss accumulator for averaging print every N steps
        self._loss_acc = {}
        self._loss_acc_count = 0

    def on_fit_start(self):
        """Initialize EMA after model is moved to device."""
        #self.discriminator.apply(init_weights)
        self.ema = EMA(self.model, decay=self.ema_decay)
        print(f"✅ EMA initialized with decay={self.ema_decay}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        x = batch  # [B, 3, 6000]

        # ---- Generator step ----
        out = self.model(x)
        x_hat = out["audio"]
        commit_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

        # Reconstruction losses
        loss_stft = self.stft_loss(x_hat, x)
        loss_wave = self.wave_loss(x_hat, x)
        loss_hf = self.hf_loss(x_hat, x)

        loss_g = (
            self.hparams.lambda_stft * loss_stft
            + self.hparams.lambda_wave * loss_wave
            + self.hparams.lambda_hf * loss_hf
            + self.hparams.lambda_commit * commit_loss
            + self.hparams.lambda_codebook * codebook_loss
        )

        # Adversarial losses (after warmup)
        use_disc = self.global_step >= self.hparams.disc_start_step
        if use_disc:
            loss_gen, loss_feat = self.gan_loss.generator_loss(x_hat, x)
            loss_g = loss_g + (
                self.hparams.lambda_adv * loss_gen
                + self.hparams.lambda_feat * loss_feat
            )

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt_g.step()

        # ---- EMA update ----
        if self.ema is not None:
            self.ema.update()

        # ---- Discriminator step ----
        if use_disc:
            loss_d = self.gan_loss.discriminator_loss(x_hat.detach(), x)
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            opt_d.step()
        else:
            loss_d = torch.tensor(0.0)

        # ---- LR schedulers ----
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        if use_disc:
            sch_d.step()

        # Logging
        self.log("train/loss_g", loss_g, prog_bar=True)
        self.log("train/loss_d", loss_d, prog_bar=True)
        self.log("train/loss_stft", loss_stft)
        self.log("train/loss_wave", loss_wave)
        self.log("train/loss_hf", loss_hf)
        self.log("train/commit_loss", commit_loss)
        self.log("train/codebook_loss", codebook_loss)
        if use_disc:
            self.log("train/loss_gen_adv", loss_gen)
            self.log("train/loss_feat", loss_feat)

        # Codebook usage metrics
        codes = out["codes"]
        for i in range(codes.shape[1]):
            usage = codes[:, i].unique().numel() / self.model.codebook_size
            self.log(f"train/codebook_{i}_usage", usage)

        # Accumulate losses for averaged print
        loss_gen_adv_val = loss_gen.item() if use_disc else 0.0
        loss_feat_val = loss_feat.item() if use_disc else 0.0

        cur = {
            "loss_sum": loss_g.item(),
            "adv_d": loss_d.item(),
            "adv_g": loss_gen_adv_val,
            "stft": loss_stft.item(),
            "wave": loss_wave.item(),
            "hf": loss_hf.item(),
            "commit": commit_loss.item(),
            "codebook": codebook_loss.item(),
            "feat": loss_feat_val,
        }
        for k, v in cur.items():
            self._loss_acc[k] = self._loss_acc.get(k, 0.0) + v
        self._loss_acc_count += 1

        if self._loss_acc_count >= 50:
            avg = {k: v / self._loss_acc_count for k, v in self._loss_acc.items()}
            codebook_usage_str = " | ".join(
                [f"cb{i}:{codes[:, i].unique().numel() / self.model.codebook_size:.2%}"
                 for i in range(codes.shape[1])]
            )
            print(
                f"[Step {self.global_step} avg50] "
                f"loss_sum: {avg['loss_sum']:.4f} | "
                f"adv_d: {avg['adv_d']:.4f} | "
                f"adv_g: {avg['adv_g']:.4f} | "
                f"stft: {avg['stft']:.4f} | "
                f"wave: {avg['wave']:.4f} | "
                f"hf: {avg['hf']:.4f} | "
                f"commit: {avg['commit']:.4f} | "
                f"codebook: {avg['codebook']:.4f} | "
                f"feat: {avg['feat']:.4f} | "
                f"usage: [{codebook_usage_str}]"
            )
            self._loss_acc = {}
            self._loss_acc_count = 0

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
        )
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=200000)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=200000)
        return [opt_g, opt_d], [sched_g, sched_d]

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state dict alongside the regular checkpoint."""
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Restore EMA state dict when loading a checkpoint."""
        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def get_ema_model_state_dict(self):
        """Return the EMA-averaged model weights for export/inference."""
        if self.ema is not None:
            return self.ema.state_dict()
        return self.model.state_dict()