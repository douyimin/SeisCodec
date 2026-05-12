"""
Callback for visualizing SeisCodec reconstruction results.
Saves matplotlib figures showing original, reconstructed, and residual waveforms
for all 3 components (Z/N/E) every k steps.
"""

import os
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pytorch_lightning as pl


class ReconstructionVisualizationCallback(pl.Callback):
    """
    Every `every_k_steps` global steps, take the current training batch,
    run the model, and save a 3×3 figure:

        Row 0: Original   Z / N / E
        Row 1: Reconstructed Z / N / E
        Row 2: Residual   Z / N / E

    Files are saved to `save_dir/recon_step{global_step}_sample{idx}.png`.
    """

    def __init__(
        self,
        every_k_steps: int = 5000,
        save_dir: str = "logs/reconstructions",
        num_samples: int = 4,
        sample_rate: int = 100,
        component_names: list = None,
        dpi: int = 150,
    ):
        """
        Parameters
        ----------
        every_k_steps : int
            Save a figure every this many global training steps.
        save_dir : str
            Directory to save PNG figures.
        num_samples : int
            Number of samples from the batch to visualize (each as a separate figure).
        sample_rate : int
            Sampling rate in Hz, used for time axis.
        component_names : list
            Names for the 3 channels, default ["Z", "N", "E"].
        dpi : int
            Resolution of saved figures.
        """
        super().__init__()
        self.every_k_steps = every_k_steps
        self.save_dir = Path(save_dir)
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.component_names = component_names or ["Z", "N", "E"]
        self.dpi = dpi
        self._last_plotted_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0:
            return
        if step == self._last_plotted_step:
            return
        if step % self.every_k_steps != 0:
            return
        self._last_plotted_step = step

        # 直接使用当前训练 batch，不再依赖 val_dataloaders
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        device = pl_module.device
        x = batch.to(device)

        # Run model in eval mode
        pl_module.eval()
        with torch.no_grad():
            out = pl_module.model(x)
            x_hat = out["audio"]
            codes = out["codes"]
        pl_module.train()

        # Move to CPU numpy
        x_np = x.cpu().numpy()
        x_hat_np = x_hat.cpu().numpy()
        residual_np = x_np - x_hat_np

        # Save figures
        self.save_dir.mkdir(parents=True, exist_ok=True)
        n_plot = min(self.num_samples, x_np.shape[0])

        for idx in range(n_plot):
            fig = self._plot_single_sample(
                x_np[idx],
                x_hat_np[idx],
                residual_np[idx],
                step=step,
                sample_idx=idx,
                codes=codes[idx].cpu().numpy() if codes is not None else None,
            )
            save_path = self.save_dir / f"recon_step{step:07d}_sample{idx}.png"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            # Log to TensorBoard if available
            if trainer.logger and hasattr(trainer.logger.experiment, "add_figure"):
                trainer.logger.experiment.add_figure(
                    f"reconstruction/sample_{idx}", fig, global_step=step
                )

    def _plot_single_sample(self, x, x_hat, residual, step, sample_idx, codes=None):
        """
        Create a 3-row × 3-col figure for one sample.

        Parameters
        ----------
        x : np.ndarray, shape [3, T]
        x_hat : np.ndarray, shape [3, T]
        residual : np.ndarray, shape [3, T]
        step : int
        sample_idx : int
        codes : np.ndarray or None, shape [n_codebooks, T_latent]
        """
        n_channels = x.shape[0]
        T = x.shape[1]
        time = np.arange(T) / self.sample_rate  # time in seconds

        fig, axes = plt.subplots(
            3, n_channels,
            figsize=(5 * n_channels, 8),
            sharex=True,
            gridspec_kw={"hspace": 0.3, "wspace": 0.25},
        )

        # Ensure axes is 2D
        if n_channels == 1:
            axes = axes[:, np.newaxis]

        row_labels = ["Original", "Reconstructed", "Residual"]
        row_data = [x, x_hat, residual]
        row_colors = ["#2563eb", "#16a34a", "#dc2626"]

        # Compute global ylim across original and reconstructed for consistent scale
        global_max = max(np.abs(x).max(), np.abs(x_hat).max()) * 1.1

        for row in range(3):
            for col in range(n_channels):
                ax = axes[row, col]
                data = row_data[row][col]
                color = row_colors[row]

                ax.plot(time, data, color=color, linewidth=0.5, alpha=0.9)
                ax.fill_between(time, data, alpha=0.15, color=color)

                # Y limits: same scale for original/reconstructed, auto for residual
                if row < 2:
                    ax.set_ylim(-global_max, global_max)
                else:
                    res_max = np.abs(residual).max() * 1.1
                    if res_max > 0:
                        ax.set_ylim(-res_max, res_max)

                # Labels
                if row == 0:
                    ax.set_title(
                        f"{self.component_names[col]}",
                        fontsize=12,
                        fontweight="bold",
                    )
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=11, fontweight="bold")
                if row == 2:
                    ax.set_xlabel("Time (s)", fontsize=10)

                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.axhline(y=0, color="gray", linewidth=0.3)

        # Compute per-component SNR
        snr_per_ch = []
        for c in range(n_channels):
            sig_power = np.sum(x[c] ** 2)
            noise_power = np.sum(residual[c] ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(sig_power / noise_power)
            else:
                snr = float("inf")
            snr_per_ch.append(snr)

        snr_str = "  ".join(
            [f"{self.component_names[c]}: {snr_per_ch[c]:.1f} dB" for c in range(n_channels)]
        )

        # Compute overall correlation coefficient
        cc_per_ch = []
        for c in range(n_channels):
            cc = np.corrcoef(x[c], x_hat[c])[0, 1]
            cc_per_ch.append(cc)
        cc_str = "  ".join(
            [f"{self.component_names[c]}: {cc_per_ch[c]:.4f}" for c in range(n_channels)]
        )

        fig.suptitle(
            f"Step {step}  |  Sample {sample_idx}\n"
            f"SNR:  {snr_str}\n"
            f"CC:   {cc_str}",
            fontsize=11,
            y=1.02,
        )

        return fig