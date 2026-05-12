"""
Validate all checkpoints in logs/seis_codec/checkpoints/ on a fixed subset
of seismic data and rank them.

Key design decisions:
  - Builds the dev-split dataset ONCE, with augmentation DISABLED, and pre-fetches
    a fixed batch list to disk (or cached in memory). This guarantees every
    checkpoint sees the *exact same* waveforms in the *exact same order*.
  - Scores each checkpoint on:
      stft  : MultiScaleSTFTLoss (lower = better, primary metric)
      wave  : multi-scale waveform L1
      hf    : HighFrequencyLoss
      snr   : 10*log10(signal_power / residual_power)  -- per-channel mean
      cc    : Pearson correlation coefficient between x and x_hat
      cbk   : codebook usage averaged over all codebooks
  - Evaluates BOTH the EMA weights (saved as `ema_state_dict` in your ckpt)
    and the raw model weights, since you save both.
  - Writes a ranked CSV and prints the top-5.

Usage
-----
    # default: scans logs/seis_codec/checkpoints, uses dev split,
    # 500 samples (~7 batches at bs=70), evaluates every ckpt
    python validate_checkpoints.py

    # custom
    python validate_checkpoints.py \
        --ckpt_dir   logs/seis_codec/checkpoints \
        --data_path  ./seisBenchDatasets \
        --num_samples 1000 \
        --batch_size 32 \
        --device cuda:0 \
        --weights both          # 'ema' | 'raw' | 'both'
        --stride 1              # evaluate every Nth ckpt; useful when there are 40+ ckpts

Outputs
-------
    logs/seis_codec/validation_results.csv
    Console: top-5 by stft loss, top-5 by composite score.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Set CUDA_VISIBLE_DEVICES via the environment (e.g. CUDA_VISIBLE_DEVICES=0 python validate_checkpoints.py).
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import csv
#import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- import project modules ---
# The validation script lives next to train.py, lightning_module.py etc.
# If you run it from a different cwd, set PYTHONPATH to your project dir.
from model import SeisCodec
from losses import MultiScaleSTFTLoss, WaveformLoss, HighFrequencyLoss

# data.py uses seisbench heavily; we re-use its loader but override the
# generator pipeline to disable augmentation.
import seisbench.data as sbd
import seisbench.generate as sbg
from data import load_dataset_from_path, TARGET_LENGTH, TARGET_SR


# ---------------------------------------------------------------------------
# Deterministic dev dataset (no augmentation, fixed sample order)
# ---------------------------------------------------------------------------

class FixedDevDataset(Dataset):
    """
    Builds a deterministic dev-split dataset: the SAME sample indices are
    drawn for every checkpoint, so the comparison is apples-to-apples.

    No random augmentation. Only:
      - peak normalization (matches training preprocessing)
      - center crop / pad to sample_length
      - clamp NaN/Inf -> 0
    """
    def __init__(
        self,
        data_path: str,
        dataset_names=None,
        sample_length: int = TARGET_LENGTH,
        num_samples: int = 500,
        seed: int = 1234,
    ):
        super().__init__()
        self.sample_length = sample_length

        data_path = Path(data_path)
        if dataset_names is None:
            dataset_names = [d.name for d in sorted(data_path.iterdir()) if d.is_dir()]

        loaded = []
        for name in dataset_names:
            folder = data_path / name
            if not folder.is_dir():
                continue
            ds = load_dataset_from_path(folder, sampling_rate=TARGET_SR, component_order="ZNE")
            if ds is None:
                continue
            try:
                split = ds.dev()
                if len(split) == 0:
                    split = ds
            except Exception:
                split = ds
            loaded.append(split)

        if not loaded:
            raise RuntimeError(f"No datasets loaded from {data_path}")

        # Build deterministic generators (no augmentation)
        self.generators = []
        for ds in loaded:
            gen = sbg.GenericGenerator(ds)
            gen.add_augmentations([
                # Deterministic windowing: take from the start (low=0, high=0
                # forces strategy='pad' deterministic). We use a centered crop
                # by setting selection to "first" via RandomWindow with low/high
                # collapsed -- but RandomWindow always randomizes, so we just
                # use a non-augmented FixedWindow if available; otherwise use
                # WindowAroundSample with fixed offset.
                sbg.FixedWindow(p0=0, windowlen=sample_length, strategy="pad"),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
                sbg.ChangeDtype(np.float32),
            ])
            self.generators.append(gen)

        # Pre-pick (dataset_idx, sample_idx) pairs ONCE, deterministically
        rng = random.Random(seed)
        self.index_list = []
        n_each = max(num_samples // len(self.generators), 1)
        for di, gen in enumerate(self.generators):
            n_take = min(n_each, len(gen))
            picks = rng.sample(range(len(gen)), n_take)
            for pi in picks:
                self.index_list.append((di, pi))
        # truncate or extend to exactly num_samples
        rng.shuffle(self.index_list)
        if len(self.index_list) > num_samples:
            self.index_list = self.index_list[:num_samples]

        print(f"Built FixedDevDataset: {len(self.index_list)} samples "
              f"from {len(self.generators)} datasets (seed={seed})")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.index_list[idx]
        sample = self.generators[ds_idx][sample_idx]
        x = sample["X"]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # length normalize
        if x.shape[-1] < self.sample_length:
            x = F.pad(x, (0, self.sample_length - x.shape[-1]))
        elif x.shape[-1] > self.sample_length:
            x = x[..., :self.sample_length]

        # channel normalize to 3
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] < 3:
            pad = torch.zeros(3 - x.shape[0], self.sample_length)
            x = torch.cat([x, pad], dim=0)
        elif x.shape[0] > 3:
            x = x[:3]
        return x


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(ckpt_dir: Path, stride: int = 1):
    """Find all *.ckpt files, sort by step in filename. stride=N picks every Nth."""
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    # Try to parse step from name like 'seis_codec-stepstep=0700000.ckpt'
    def parse_step(p: Path):
        name = p.stem
        for tok in name.replace("-", "=").split("="):
            tok = tok.strip()
            if tok.isdigit():
                return int(tok)
        return 0
    ckpts = sorted(ckpts, key=parse_step)
    if stride > 1:
        ckpts = ckpts[::stride]
    return [(parse_step(p), p) for p in ckpts]


# ---------------------------------------------------------------------------
# Loading model from a Lightning ckpt (manual, avoids re-instantiating
# the full LightningModule with discriminator etc.)
# ---------------------------------------------------------------------------

def load_seis_codec_from_ckpt(ckpt_path: Path, device: str, use_ema: bool):
    """
    Load only the SeisCodec generator weights from a Lightning checkpoint.

    Lightning saves keys with prefix 'model.' (the generator inside
    SeisCodecLightning). EMA shadow is in `ema_state_dict` with no prefix
    (parameter names directly).
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = blob.get("hyper_parameters", {})

    model = SeisCodec(
        in_channels   = hp.get("in_channels", 3),
        out_channels  = hp.get("out_channels", 3),
        encoder_dim   = hp.get("encoder_dim", 64),
        encoder_rates = hp.get("encoder_rates", [2, 2, 4, 4]),
        decoder_dim   = hp.get("decoder_dim", 1536),
        decoder_rates = hp.get("decoder_rates", [4, 4, 2, 2]),
        n_codebooks   = hp.get("n_codebooks", 12),
        codebook_size = hp.get("codebook_size", 2048),
        codebook_dim  = hp.get("codebook_dim", 16),
        quantizer_dropout = 0.0,  # eval-time: no quantizer dropout
        sample_rate   = hp.get("sample_rate", 100),
    )

    if use_ema and "ema_state_dict" in blob:
        # EMA shadow keys are bare parameter names (no 'model.' prefix)
        ema_sd = blob["ema_state_dict"]
        # Some EMAs only store trainable params; copy on top of model state
        target = model.state_dict()
        loaded = 0
        for k, v in ema_sd.items():
            if k in target and target[k].shape == v.shape:
                target[k].copy_(v)
                loaded += 1
        model.load_state_dict(target, strict=False)
        weight_tag = f"ema({loaded}/{len(ema_sd)})"
    else:
        # Raw weights: strip 'model.' prefix
        sd = blob["state_dict"]
        gen_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(gen_sd, strict=False)
        weight_tag = f"raw(missing={len(missing)},unexpected={len(unexpected)})"

    model.eval().to(device)
    return model, weight_tag, hp


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one(model, loader, stft_loss, wave_loss, hf_loss, device,
                 n_codebooks, codebook_size, desc="eval"):
    """Average metrics across the entire fixed dev loader."""
    sums = dict(stft=0.0, wave=0.0, hf=0.0, snr=0.0, cc=0.0, n=0)
    code_unique = [set() for _ in range(n_codebooks)]

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for batch in pbar:
        x = batch.to(device, non_blocking=True)
        out = model(x)
        x_hat = out["audio"]

        sums["stft"] += stft_loss(x_hat, x).item() * x.size(0)
        sums["wave"] += wave_loss(x_hat, x).item() * x.size(0)
        sums["hf"]   += hf_loss(x_hat, x).item()   * x.size(0)

        # SNR (per-sample, per-channel, mean)
        residual = x - x_hat
        sig_pwr = (x ** 2).sum(dim=-1)            # [B, C]
        noi_pwr = (residual ** 2).sum(dim=-1).clamp_min(1e-12)
        snr = 10.0 * torch.log10(sig_pwr.clamp_min(1e-12) / noi_pwr)
        sums["snr"] += snr.mean(dim=1).sum().item()  # mean over channels, sum over batch

        # Pearson CC per channel, per sample
        x_c = x - x.mean(dim=-1, keepdim=True)
        y_c = x_hat - x_hat.mean(dim=-1, keepdim=True)
        num = (x_c * y_c).sum(dim=-1)
        den = (x_c.pow(2).sum(dim=-1).sqrt() * y_c.pow(2).sum(dim=-1).sqrt()).clamp_min(1e-12)
        cc = (num / den).mean(dim=1)              # mean over channels
        sums["cc"] += cc.sum().item()

        # codebook usage: collect unique codes
        codes = out["codes"]  # [B, K, T]
        for k in range(min(codes.shape[1], n_codebooks)):
            code_unique[k].update(codes[:, k].unique().cpu().tolist())

        sums["n"] += x.size(0)

        # live metrics in the bar
        running_n = max(sums["n"], 1)
        pbar.set_postfix(stft=f"{sums['stft']/running_n:.3f}",
                         snr=f"{sums['snr']/running_n:+.1f}",
                         n=sums["n"])
    pbar.close()

    n = max(sums["n"], 1)
    avg = {k: sums[k] / n for k in ("stft", "wave", "hf", "snr", "cc")}
    avg["cbk_usage"] = float(np.mean([len(s) / codebook_size for s in code_unique]))
    avg["n_samples"] = sums["n"]
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",  type=str, default="logs/seis_codec/checkpoints")
    p.add_argument("--data_path", type=str, default="./seisBenchDatasets")
    p.add_argument("--dataset_names", type=str, nargs="+", default=None)
    p.add_argument("--num_samples", type=int, default=500,
                   help="Total samples used for evaluation (split across datasets)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_length", type=int, default=6000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--weights", type=str, default="both", choices=["ema", "raw", "both"])
    p.add_argument("--stride", type=int, default=1,
                   help="Evaluate every Nth checkpoint (sorted by step)")
    p.add_argument("--out_csv", type=str, default="logs/seis_codec/validation_results.csv")
    p.add_argument("--limit", type=int, default=-1,
                   help="Only run this many checkpoints (after stride). -1 = all")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Find checkpoints ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = discover_checkpoints(ckpt_dir, stride=args.stride)
    if args.limit > 0:
        ckpts = ckpts[:args.limit]
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(1)
    print(f"Found {len(ckpts)} checkpoints (stride={args.stride})")
    print(f"Steps: {[s for s, _ in ckpts]}")

    # ---- Build fixed eval set ----
    ds = FixedDevDataset(
        data_path=args.data_path,
        dataset_names=args.dataset_names,
        sample_length=args.sample_length,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,           # critical: same order for every ckpt
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Loss modules (reused across ckpts) ----
    stft_loss = MultiScaleSTFTLoss(window_lengths=[32, 64, 128, 256, 512]).to(device)
    wave_loss = WaveformLoss(scales=[1, 2, 4]).to(device)
    hf_loss = HighFrequencyLoss(
        diff_orders=[1, 2], diff_weight=1.0,
        spectral_weight=1.0, grad_weight=1.0,
        grad_smooth=5, stft_window=64, freq_power=1.0,
    ).to(device)

    weight_modes = ["ema", "raw"] if args.weights == "both" else [args.weights]

    rows = []
    total_runs = len(ckpts) * len(weight_modes)
    outer_bar = tqdm(total=total_runs, desc="checkpoints", unit="ckpt")
    for step, ckpt_path in ckpts:
        for wm in weight_modes:
            t0 = time.time()
            try:
                model, tag, hp = load_seis_codec_from_ckpt(
                    ckpt_path, device=device, use_ema=(wm == "ema"))
            except Exception as e:
                outer_bar.write(f"  ❌ failed to load {ckpt_path.name} ({wm}): {e}")
                outer_bar.update(1)
                continue
            n_cb = hp.get("n_codebooks", 12)
            cb_size = hp.get("codebook_size", 2048)

            metrics = evaluate_one(
                model, loader, stft_loss, wave_loss, hf_loss,
                device, n_codebooks=n_cb, codebook_size=cb_size,
                desc=f"step={step} [{wm}]")
            dt = time.time() - t0

            row = {
                "step": step,
                "weights": wm,
                "ckpt": ckpt_path.name,
                "tag": tag,
                **metrics,
                "elapsed_sec": round(dt, 1),
            }
            rows.append(row)
            outer_bar.write(
                f"step={step:>7d}  [{wm:3s}]  "
                f"stft={metrics['stft']:.4f}  wave={metrics['wave']:.4f}  "
                f"hf={metrics['hf']:.4f}  snr={metrics['snr']:+.2f}dB  "
                f"cc={metrics['cc']:.4f}  cbk={metrics['cbk_usage']:.3f}  "
                f"({dt:.1f}s)")
            outer_bar.set_postfix(last_step=step, last_stft=f"{metrics['stft']:.3f}")
            outer_bar.update(1)

            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    outer_bar.close()

    # ---- Save CSV ----
    if rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n💾 Wrote {len(rows)} rows to {out_path}")

    # ---- Rankings ----
    print("\n" + "=" * 78)
    print("Top 5 by STFT loss (lower = better — primary metric)")
    print("=" * 78)
    for r in sorted(rows, key=lambda r: r["stft"])[:5]:
        print(f"  step={r['step']:>7d}  [{r['weights']:3s}]  "
              f"stft={r['stft']:.4f}  snr={r['snr']:+.2f}dB  cc={r['cc']:.4f}")

    print("\nTop 5 by composite z-score of stft+wave+hf (lower = better)")
    print("=" * 78)
    if len(rows) >= 3:
        keys = ("stft", "wave", "hf")
        zs = {}
        for k in keys:
            v = np.array([r[k] for r in rows])
            zs[k] = (v - v.mean()) / (v.std() + 1e-12)
        composite = np.mean(np.stack([zs[k] for k in keys]), axis=0)
        for r, c in sorted(zip(rows, composite), key=lambda rc: rc[1])[:5]:
            print(f"  step={r['step']:>7d}  [{r['weights']:3s}]  "
                  f"composite_z={c:+.3f}  "
                  f"stft={r['stft']:.4f}  snr={r['snr']:+.2f}dB  cc={r['cc']:.4f}")

    print("\nTop 5 by SNR (higher = better)")
    print("=" * 78)
    for r in sorted(rows, key=lambda r: -r["snr"])[:5]:
        print(f"  step={r['step']:>7d}  [{r['weights']:3s}]  "
              f"snr={r['snr']:+.2f}dB  stft={r['stft']:.4f}  cc={r['cc']:.4f}")


if __name__ == "__main__":
    main()
