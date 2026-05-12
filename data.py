"""
Data module using SeisBench to load seismic datasets from a custom path.
Each sample is randomly cropped to 6000 samples (60s @ 100Hz).
All datasets are resampled to 100Hz and output 3-component (ZNE) waveforms.

Usage:
    python train.py --data_path "I:\\seisBenchDatasets"
    python train.py --data_path "I:\\seisBenchDatasets" --dataset_names STEAD lenDB instancecounts
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import seisbench.data as sbd
import seisbench.generate as sbg


TARGET_LENGTH = 6000   # 60s @ 100Hz
TARGET_SR = 100


# ============================================================================
# Custom Augmentations
# ============================================================================

def _to_ndarray(x):
    """安全地将 state_dict 中的值转为 ndarray，失败返回 None。"""
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x, dtype=np.float32)
    except (ValueError, TypeError):
        return None


class ChannelDropout:
    """
    随机将某个分量置零，模拟台站单分量故障。
    以概率 p 触发，触发时随机选 1 个通道置零。
    """
    def __init__(self, p: float = 0.2, key: str = "X"):
        self.p = p
        self.key = key

    def __call__(self, state_dict):
        x = _to_ndarray(state_dict[self.key])
        if x is None or x.ndim != 2:
            return
        if np.random.rand() < self.p:
            ch = np.random.randint(x.shape[0])
            x[ch] = 0.0
        state_dict[self.key] = x


class RandomGap:
    """
    随机在波形中插入一段数据间断（置零），模拟数据缺失/传输中断。
    gap 长度为 min_len ~ max_len 采样点，以概率 p 触发。
    """
    def __init__(self, p: float = 0.15, min_len: int = 50, max_len: int = 500, key: str = "X"):
        self.p = p
        self.min_len = min_len
        self.max_len = max_len
        self.key = key

    def __call__(self, state_dict):
        x = _to_ndarray(state_dict[self.key])
        if x is None or x.ndim != 2:
            return
        if np.random.rand() < self.p:
            T = x.shape[-1]
            gap_len = np.random.randint(self.min_len, min(self.max_len, T // 2) + 1)
            start = np.random.randint(0, T - gap_len)
            x[..., start:start + gap_len] = 0.0
        state_dict[self.key] = x


class TimeStretch:
    """
    时间拉伸/压缩，模拟不同震源距离导致的波形展宽/压缩。
    在 [rate_min, rate_max] 范围内随机选取拉伸因子，
    使用线性插值重采样后裁剪/补零到原长度。以概率 p 触发。
    """
    def __init__(self, p: float = 0.3, rate_min: float = 0.85, rate_max: float = 1.15, key: str = "X"):
        self.p = p
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.key = key

    def __call__(self, state_dict):
        x = _to_ndarray(state_dict[self.key])
        if x is None or x.ndim != 2:
            return
        if np.random.rand() < self.p:
            rate = np.random.uniform(self.rate_min, self.rate_max)
            C, T = x.shape
            new_T = int(T * rate)
            if new_T < 2:
                state_dict[self.key] = x
                return
            # 线性插值重采样
            old_idx = np.linspace(0, T - 1, new_T)
            x_stretched = np.zeros((C, new_T), dtype=x.dtype)
            for c in range(C):
                x_stretched[c] = np.interp(old_idx, np.arange(T), x[c])
            # 裁剪或补零到原长度
            if new_T >= T:
                x = x_stretched[..., :T]
            else:
                x = np.zeros((C, T), dtype=x.dtype)
                x[..., :new_T] = x_stretched
        state_dict[self.key] = x


class PolarityFlip:
    """
    随机翻转极性（乘以 -1），地震信号的极性取决于震源机制和台站方位，
    翻转不改变物理含义但增加数据多样性。以概率 p 触发。
    """
    def __init__(self, p: float = 0.5, key: str = "X"):
        self.p = p
        self.key = key

    def __call__(self, state_dict):
        x = _to_ndarray(state_dict[self.key])
        if x is None:
            return
        if np.random.rand() < self.p:
            x = -x
        state_dict[self.key] = x


class RandomGaussianNoise:
    """
    以概率 p 添加高斯噪声，避免每条样本都加噪。
    """
    def __init__(self, p: float = 0.5, scale_min: float = 0.0, scale_max: float = 0.075, key: str = "X"):
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.key = key

    def __call__(self, state_dict):
        x = _to_ndarray(state_dict[self.key])
        if x is None:
            return
        if np.random.rand() < self.p:
            scale = np.random.uniform(self.scale_min, self.scale_max)
            x = x + np.random.randn(*x.shape).astype(x.dtype) * scale
        state_dict[self.key] = x


def _detect_chunks(folder_path: Path):
    """
    检测 SeisBench chunked 数据集。
    Chunk 文件命名约定: metadata{chunk_name}.csv + waveforms{chunk_name}.hdf5
    例如: metadatanc2018.csv + waveformsnc2018.hdf5
    返回 chunk 名称列表，如 ['nc2018', 'nc2017', ...]；无 chunk 则返回空列表。
    """
    meta_files = sorted(folder_path.glob("metadata*.csv"))
    chunks = []
    for mf in meta_files:
        # 跳过标准的 metadata.csv
        if mf.name == "metadata.csv":
            continue
        # 提取 chunk name: metadata{chunk_name}.csv → chunk_name
        chunk_name = mf.stem.replace("metadata", "", 1)
        wf = folder_path / f"waveforms{chunk_name}.hdf5"
        if wf.exists():
            chunks.append(chunk_name)
    return chunks


def load_dataset_from_path(folder_path: Path, **kwargs):
    """
    Load a SeisBench WaveformDataset from a local folder.
    支持两种格式:
    1. 标准: metadata.csv + waveforms.hdf5
    2. Chunked: metadata{chunk}.csv + waveforms{chunk}.hdf5 (如 CEED, MLAAPDE)
    """
    folder_path = Path(folder_path)
    has_metadata = (folder_path / "metadata.csv").exists()
    has_waveforms = (folder_path / "waveforms.hdf5").exists()

    # 标准格式
    if has_metadata and has_waveforms:
        ds = sbd.WaveformDataset(folder_path, **kwargs)
        print(f"  ✅ {folder_path.name}: {len(ds)} traces")
        return ds

    # 尝试 chunked 格式
    chunks = _detect_chunks(folder_path)
    if chunks:
        print(f"  📦 {folder_path.name}: detected {len(chunks)} chunks: {chunks[:5]}{'...' if len(chunks) > 5 else ''}")
        try:
            ds = sbd.WaveformDataset(folder_path, **kwargs)
            print(f"  ✅ {folder_path.name}: {len(ds)} traces (chunked)")
            return ds
        except Exception as e:
            print(f"  ❌ Failed to load chunked dataset {folder_path.name}: {e}")
            return None

    # 都不匹配
    missing = []
    if not has_metadata:
        missing.append("metadata.csv")
    if not has_waveforms:
        missing.append("waveforms.hdf5")
    print(f"  ❌ Skipped {folder_path.name}: missing {', '.join(missing)}, no chunks detected")
    return None


class SeismicWaveformDataset(Dataset):
    """
    Wraps SeisBench datasets to yield [3, 6000] tensors.
    Uses SeisBench's augmentation pipeline for random windowing and normalization.
    """
    def __init__(
        self,
        data_path: str,
        dataset_names: list = None,
        sample_length: int = TARGET_LENGTH,
        split: str = "train",
    ):
        """
        Parameters
        ----------
        data_path : str
            Root directory containing dataset subfolders
            (e.g., I:\\seisBenchDatasets with STEAD/, lenDB/, etc. inside)
        dataset_names : list, optional
            List of subfolder names to load. If None, load all subfolders
            that contain metadata.csv + waveforms.hdf5.
        sample_length : int
            Number of samples per waveform (default: 6000)
        split : str
            One of "train", "dev", "test"
        """
        super().__init__()
        data_path = Path(data_path)
        self.sample_length = sample_length

        # Discover datasets
        if dataset_names is None:
            dataset_names = [
                d.name for d in sorted(data_path.iterdir()) if d.is_dir()
            ]

        print(f"\n{'='*60}")
        print(f"Loading datasets from: {data_path}")
        print(f"Split: {split} | Folders: {dataset_names}")
        print(f"{'='*60}")

        # Load each dataset
        loaded = []
        for name in dataset_names:
            folder = data_path / name
            if not folder.is_dir():
                print(f"  ⚠️  Not found: {folder}")
                continue

            ds = load_dataset_from_path(
                folder,
                sampling_rate=TARGET_SR,
                component_order="ZNE",
            )
            if ds is None:
                continue

            # Apply split
            try:
                if split == "train":
                    ds_split = ds.train()
                elif split == "dev":
                    ds_split = ds.dev()
                elif split == "test":
                    ds_split = ds.test()
                else:
                    ds_split = ds

                if len(ds_split) > 0:
                    print(f"         → {split}: {len(ds_split)} traces")
                    loaded.append(ds_split)
                else:
                    print(f"  ⚠️  {name} {split} split is empty, using full dataset")
                    loaded.append(ds)
            except Exception as e:
                print(f"  ⚠️  No split for {name} ({e}), using full dataset")
                loaded.append(ds)

        if not loaded:
            raise RuntimeError(
                f"No datasets could be loaded from {data_path}!\n"
                f"Each subfolder must contain metadata.csv + waveforms.hdf5"
            )

        # Keep datasets separate for equal-probability sampling
        self.datasets = loaded
        self.n_datasets = len(loaded)
        self.dataset_lengths = [len(ds) for ds in loaded]
        # Virtual epoch length = n_datasets * max single dataset length
        # so that each dataset gets fully covered on average
        self.virtual_length = self.n_datasets * max(self.dataset_lengths)

        print(f"\n📊 {self.n_datasets} datasets, sizes: {self.dataset_lengths}")
        print(f"📊 Virtual epoch length: {self.virtual_length}")
        print(f"{'='*60}\n")

        # Build one SeisBench generator per dataset
        self.generators = []
        for ds in self.datasets:
            gen = sbg.GenericGenerator(ds)
            gen.add_augmentations([
                # 1. 随机窗口裁剪
                sbg.RandomWindow(windowlen=sample_length, strategy="pad"),
                # 2. 时间拉伸/压缩 (在裁剪后、归一化前)
                TimeStretch(p=0.3, rate_min=0.85, rate_max=1.15),
                # 3. 随机旋转三分量 (E/N混合，模拟不同方位角)
                sbg.RandomArrayRotation(keys=["X"]),
                # 4. 极性翻转
                PolarityFlip(p=0.5),
                # 5. 随机通道丢失 (模拟单分量故障)
                ChannelDropout(p=0.2),
                # 6. 随机数据间断 (模拟传输中断)
                RandomGap(p=0.15, min_len=50, max_len=500),
                # 7. 加高斯噪声 (50%概率)
                RandomGaussianNoise(p=0.5, scale_min=0.0, scale_max=0.075),
                # 8. 去均值 + peak归一化 (放在最后)
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
                sbg.ChangeDtype(np.float32),
            ])
            self.generators.append(gen)

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        # Equal probability: randomly pick a dataset, then randomly pick a sample
        ds_idx = np.random.randint(self.n_datasets)
        sample_idx = np.random.randint(len(self.generators[ds_idx]))
        sample = self.generators[ds_idx][sample_idx]
        waveform = sample["X"]  # [3, 6000]

        waveform = torch.from_numpy(waveform)
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure correct length
        if waveform.shape[-1] < self.sample_length:
            pad = self.sample_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[-1] > self.sample_length:
            waveform = waveform[..., :self.sample_length]

        # Ensure 3 channels
        if waveform.shape[0] < 3:
            pad_ch = torch.zeros(3 - waveform.shape[0], self.sample_length)
            waveform = torch.cat([waveform, pad_ch], dim=0)
        elif waveform.shape[0] > 3:
            waveform = waveform[:3]

        return waveform  # [3, 6000]


class SeismicDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for seismic waveforms.
    Loads datasets from a local directory.
    """
    def __init__(
        self,
        data_path: str = r"I:\seisBenchDatasets",
        dataset_names: list = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_length: int = TARGET_LENGTH,
    ):
        """
        Parameters
        ----------
        data_path : str
            Root directory containing dataset subfolders.
        dataset_names : list, optional
            Subfolder names to load. None = auto-discover all.
            Example: ["STEAD", "lenDB", "instancecounts"]
        batch_size : int
        num_workers : int
        sample_length : int
        """
        super().__init__()
        self.data_path = data_path
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_length = sample_length

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = SeismicWaveformDataset(
                data_path=self.data_path,
                dataset_names=self.dataset_names,
                sample_length=self.sample_length,
                split="train",
            )
            self.val_ds = SeismicWaveformDataset(
                data_path=self.data_path,
                dataset_names=self.dataset_names,
                sample_length=self.sample_length,
                split="dev",
            )
        if stage == "test":
            self.test_ds = SeismicWaveformDataset(
                data_path=self.data_path,
                dataset_names=self.dataset_names,
                sample_length=self.sample_length,
                split="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )