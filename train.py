"""
Training script for SeisCodec.

Usage:
    python train.py --data_path "I:\\seisBenchDatasets"
    python train.py --data_path "I:\\seisBenchDatasets" --dataset_names STEAD lenDB
    python train.py --save_every_k_steps 10000 --ema_decay 0.999
    python train.py --fast_dev_run  # quick debug run
"""
#import time
#time.sleep(1800)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Set the visible GPUs via environment variable before running, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 python train.py ...
# Uncomment the line below to hard-code GPUs.
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning_module import SeisCodecLightning
from data import SeismicDataModule
from callbacks import ReconstructionVisualizationCallback
from load_dac_weights import load_dac_weights_into_seis_codec

def parse_args():
    parser = argparse.ArgumentParser(description="Train SeisCodec")

    # Data
    parser.add_argument("--data_path", type=str, default="./seisBenchDatasets",
                        help="Root dir containing dataset subfolders")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=None,
                        help="Subfolder names to use (default: all)")
    parser.add_argument("--batch_size", type=int, default=70)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_length", type=int, default=6000)

    # Model
    parser.add_argument("--encoder_dim", type=int, default=64)
    parser.add_argument("--decoder_dim", type=int, default=1536)
    parser.add_argument("--n_codebooks", type=int, default=12)
    parser.add_argument("--codebook_size", type=int, default=2048)
    parser.add_argument("--codebook_dim", type=int, default=16)
    parser.add_argument("--quantizer_dropout", type=float, default=0.25)

    # Training
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--max_epochs", type=int, default=-1,
                        help="-1 means no epoch limit, train by steps only")
    parser.add_argument("--max_steps", type=int, default=5000000)
    parser.add_argument("--disc_start_step", type=int, default=10000)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # Loss weights
    parser.add_argument("--lambda_stft", type=float, default=15.0)
    parser.add_argument("--lambda_wave", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_feat", type=float, default=2.0)
    parser.add_argument("--lambda_commit", type=float, default=0.25)
    parser.add_argument("--lambda_codebook", type=float, default=1.0)
    parser.add_argument("--lambda_hf", type=float, default=5.0)

    # Hardware
    #parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["16", "bf16", "32"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Logging & Checkpointing
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="seis_codec")
    parser.add_argument("--save_every_k_steps", type=int, default=10000,
                        help="Save checkpoint every k training steps")

    # Debug
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a Lightning .ckpt to resume training from. Default None (train from scratch).")

    # Visualization
    parser.add_argument("--plot_every_k_steps", type=int, default=5000)
    parser.add_argument("--plot_num_samples", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Data (train only, no validation) ----
    data_module = SeismicDataModule(
        data_path=args.data_path,
        dataset_names=args.dataset_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_length=args.sample_length,
    )

    # ---- Model ----
    model = SeisCodecLightning(
        in_channels=3,
        out_channels=3,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        quantizer_dropout=args.quantizer_dropout,
        lr=args.lr,
        lambda_stft=args.lambda_stft,
        lambda_wave=args.lambda_wave,
        lambda_adv=args.lambda_adv,
        lambda_feat=args.lambda_feat,
        lambda_commit=args.lambda_commit,
        lambda_codebook=args.lambda_codebook,
        lambda_hf=args.lambda_hf,
        disc_start_step=args.disc_start_step,
        ema_decay=args.ema_decay,
    )
    #load_dac_weights_into_seis_codec(model.model,'logs/seis_codec/checkpoints/seis_codec-stepstep=0320000.ckpt')
    # ---- Callbacks ----
    callbacks = [
        # Save every k steps (no monitor needed, no last)
        ModelCheckpoint(
            dirpath=f"{args.log_dir}/{args.exp_name}/checkpoints",
            filename="seis_codec-step{step:07d}",
            every_n_train_steps=args.save_every_k_steps,
            save_top_k=-1,       # keep all step checkpoints
            save_last=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        ReconstructionVisualizationCallback(
            every_k_steps=args.plot_every_k_steps,
            save_dir=f"{args.log_dir}/{args.exp_name}/reconstructions",
            num_samples=args.plot_num_samples,
            sample_rate=100,
        ),
    ]

    # ---- Logger ----
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name,
    )
    strategy = DDPStrategy(
            find_unused_parameters=True,
           # static_graph=True
        )

    # ---- Trainer (no validation) ----
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        #devices=args.gpus if args.gpus > 0 else "auto",
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
       # gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=50,
        # No validation
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # ---- Train ----
    trainer.fit(model, data_module, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
