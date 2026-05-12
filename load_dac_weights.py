"""
Utility to load pretrained original DAC weights into SeisCodec.

Original DAC:  in_channels=1,  encoder_rates=[2,4,8,8],  decoder_rates=[8,8,4,2], d_out=1
SeisCodec:    in_channels=3,  encoder_rates=[2,4,4,4],  decoder_rates=[4,4,4,2], d_out=3

Transferable weights:
  - RVQ quantizer (all weights, identical architecture)
  - Encoder/Decoder blocks that share the same internal channel dims
  - Snake1d alpha parameters (channel-dim dependent, same where dims match)

Non-transferable (shape mismatch):
  - Encoder first conv: in_channels 1 vs 3
  - Decoder last conv: out_channels 1 vs 3
  - Stride convolutions where kernel_size = 2*stride differs (stride 8 vs 4)

Strategy:
  For matching shapes  → copy directly
  For channel mismatch → partial copy + random init for extra channels
  For stride mismatch  → skip (random init)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from model import SeisCodec


def load_dac_weights_into_seis_codec(
    seismic_model: SeisCodec,
    dac_checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load original DAC pretrained weights into a SeisCodec model.

    Parameters
    ----------
    seismic_model : SeisCodec
        Target model to load weights into.
    dac_checkpoint_path : str
        Path to original DAC checkpoint. Supports:
        - Full DAC checkpoint (.pth with 'state_dict' or 'model' key)
        - Raw state_dict (.pth)
        - DAC .pkl files (via dac library)
    strict : bool
        If True, raise error on any mismatch. Default False.
    verbose : bool
        Print detailed loading info.

    Returns
    -------
    dict with keys:
        'loaded': list of loaded parameter names
        'skipped_shape': list of skipped params (shape mismatch)
        'skipped_missing': list of params in DAC but not in SeisCodec
        'partial': list of params with partial copy (channel expansion)
        'remaining': list of SeisCodec params not found in DAC
    """

    # ---- Step 1: Load DAC state dict ----
    raw = torch.load(dac_checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(raw, dict):
        if "state_dict" in raw:
            dac_sd = raw["state_dict"]
        elif "model" in raw:
            dac_sd = raw["model"]
        elif "generator" in raw:
            dac_sd = raw["generator"]
        else:
            # Assume it's a raw state_dict
            dac_sd = raw
    else:
        # Try if it's a DAC model object
        try:
            dac_sd = raw.state_dict()
        except AttributeError:
            raise ValueError(f"Cannot extract state_dict from {type(raw)}")

    # ---- Step 2: Normalize key prefixes ----
    # Original DAC keys may have prefixes like "model.", "module.", etc.
    # SeisCodec keys are: encoder.*, quantizer.*, decoder.*
    dac_sd = _strip_prefix(dac_sd)

    seismic_sd = seismic_model.state_dict()

    # ---- Step 3: Match and load ----
    loaded = []
    skipped_shape = []
    skipped_missing = []
    partial = []

    for name, param in dac_sd.items():
        if name not in seismic_sd:
            skipped_missing.append(name)
            continue

        target_shape = seismic_sd[name].shape
        source_shape = param.shape

        if source_shape == target_shape:
            # Exact match → copy
            seismic_sd[name].copy_(param)
            loaded.append(name)

        elif _is_input_conv(name) and source_shape[1] != target_shape[1]:
            # Encoder first conv: [64, 1, 7] → [64, 3, 7]
            # Copy channel 0, init channels 1,2 by repeating
            _partial_copy_input_channels(seismic_sd[name], param)
            partial.append(f"{name}: {list(source_shape)} → {list(target_shape)} (channel expansion)")

        elif _is_output_conv(name) and source_shape[0] != target_shape[0]:
            # Decoder last conv: [1, dim, 7] → [3, dim, 7]
            # Copy channel 0, init channels 1,2 by repeating
            _partial_copy_output_channels(seismic_sd[name], param)
            partial.append(f"{name}: {list(source_shape)} → {list(target_shape)} (channel expansion)")

        elif _is_output_bias(name, dac_sd) and source_shape[0] != target_shape[0]:
            # Decoder last conv bias: [1] → [3]
            _partial_copy_bias(seismic_sd[name], param)
            partial.append(f"{name}: {list(source_shape)} → {list(target_shape)} (bias expansion)")

        else:
            skipped_shape.append(
                f"{name}: DAC {list(source_shape)} vs SeisCodec {list(target_shape)}"
            )

    # Find remaining (SeisCodec params not in DAC)
    remaining = [n for n in seismic_sd if n not in dac_sd]

    # Apply the modified state dict
    seismic_model.load_state_dict(seismic_sd, strict=False)

    # ---- Step 4: Report ----
    result = {
        "loaded": loaded,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "partial": partial,
        "remaining": remaining,
    }

    if verbose:
        _print_report(result, len(seismic_sd))

    if strict and (skipped_shape or skipped_missing or remaining):
        raise RuntimeError(
            f"Strict loading failed: {len(skipped_shape)} shape mismatches, "
            f"{len(skipped_missing)} missing in target, "
            f"{len(remaining)} remaining uninitialized"
        )

    return result


# ============================================================================
# Helpers
# ============================================================================

def _strip_prefix(sd: dict) -> dict:
    """Remove common prefixes to align with SeisCodec naming."""
    new_sd = OrderedDict()
    for k, v in sd.items():
        # Strip "model." or "module." prefix
        new_key = k
        for prefix in ["model.", "module.", "generator."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_sd[new_key] = v
    return new_sd


def _is_input_conv(name: str) -> bool:
    """Check if this is the encoder's first convolution (in_channels differs)."""
    return "encoder.block.0.weight" in name


def _is_output_conv(name: str) -> bool:
    """Check if this is the decoder's last convolution (out_channels differs)."""
    # Decoder structure: model.0 (first conv), model.1..N-3 (blocks),
    # model.N-2 (Snake), model.N-1 (final conv), model.N (Tanh)
    # The final conv weight will have pattern like "decoder.model.X.weight"
    # and its shape will be [out_ch, dim, kernel]
    # We detect by checking it's a decoder weight with small out_channels
    return ("decoder.model" in name and name.endswith(".weight")
            and "block" not in name  # not inside DecoderBlock
            and name != "decoder.model.0.weight")  # not first conv


def _is_output_bias(name: str, sd: dict) -> bool:
    """Check if this is the bias corresponding to the output conv."""
    weight_name = name.replace(".bias", ".weight")
    if weight_name in sd and name.endswith(".bias"):
        return _is_output_conv(weight_name)
    return False


def _partial_copy_input_channels(target: torch.Tensor, source: torch.Tensor):
    """
    Copy weights for input channel expansion.
    source shape: [out_ch, 1, kernel]
    target shape: [out_ch, 3, kernel]
    Strategy: repeat source across new channels, scale by 1/n_new for energy preservation.
    """
    n_src = source.shape[1]
    n_tgt = target.shape[1]
    # Copy source to first channel(s)
    target[:, :n_src].copy_(source)
    # Initialize remaining channels as copies scaled down
    for i in range(n_src, n_tgt):
        target[:, i].copy_(source[:, i % n_src])
    # Scale all channels so total energy is preserved
    target.mul_(n_src / n_tgt)


def _partial_copy_output_channels(target: torch.Tensor, source: torch.Tensor):
    """
    Copy weights for output channel expansion.
    source shape: [1, in_ch, kernel]
    target shape: [3, in_ch, kernel]
    Strategy: repeat source across new output channels.
    """
    n_src = source.shape[0]
    for i in range(target.shape[0]):
        target[i].copy_(source[i % n_src])


def _partial_copy_bias(target: torch.Tensor, source: torch.Tensor):
    """Expand bias from [1] to [3] by repeating."""
    n_src = source.shape[0]
    for i in range(target.shape[0]):
        target[i].copy_(source[i % n_src])


def _print_report(result: dict, total_params: int):
    """Print a human-readable loading report."""
    n_loaded = len(result["loaded"])
    n_partial = len(result["partial"])
    n_skip_shape = len(result["skipped_shape"])
    n_skip_miss = len(result["skipped_missing"])
    n_remaining = len(result["remaining"])

    print("=" * 70)
    print("DAC → SeisCodec Weight Loading Report")
    print("=" * 70)
    print(f"  Total SeisCodec params:    {total_params}")
    print(f"  ✅ Loaded (exact match):     {n_loaded}")
    print(f"  🔶 Partial copy (expanded):  {n_partial}")
    print(f"  ❌ Skipped (shape mismatch): {n_skip_shape}")
    print(f"  ⚠️  DAC-only (not in target): {n_skip_miss}")
    print(f"  🆕 Uninitialized (new):      {n_remaining}")
    print("-" * 70)

    if result["partial"]:
        print("\n🔶 Partially copied (channel expansion):")
        for p in result["partial"]:
            print(f"    {p}")

    if result["skipped_shape"]:
        print("\n❌ Skipped due to shape mismatch (stride differences):")
        for s in result["skipped_shape"]:
            print(f"    {s}")

    if result["remaining"][:10]:
        print(f"\n🆕 Uninitialized params (first 10 of {n_remaining}):")
        for r in result["remaining"][:10]:
            print(f"    {r}")
        if n_remaining > 10:
            print(f"    ... and {n_remaining - 10} more")

    coverage = (n_loaded + n_partial) / total_params * 100
    print(f"\n📊 Coverage: {n_loaded + n_partial}/{total_params} = {coverage:.1f}%")
    print("=" * 70)


# ============================================================================
# Convenience: download and load official DAC weights
# ============================================================================

def load_official_dac_44khz(seismic_model: SeisCodec, verbose=True) -> dict:
    """
    Download official DAC 44kHz weights and load into SeisCodec.
    Requires the `dac` package: pip install descript-audio-codec

    Usage:
        model = SeisCodec()
        result = load_official_dac_44khz(model)
    """
    try:
        import dac
        model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(model_path)
        # Save as temp state dict
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        torch.save(dac_model.state_dict(), tmp.name)
        tmp.close()
        result = load_dac_weights_into_seis_codec(
            seismic_model, tmp.name, verbose=verbose
        )
        os.unlink(tmp.name)
        return result
    except ImportError:
        raise ImportError(
            "Please install descript-audio-codec: pip install descript-audio-codec"
        )


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    import sys

    model = SeisCodec()
    print(f"SeisCodec created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    if len(sys.argv) > 1:
        # Load from provided checkpoint path
        ckpt_path = sys.argv[1]
        print(f"\nLoading from: {ckpt_path}")
        result = load_dac_weights_into_seis_codec(model, ckpt_path, verbose=True)
    else:
        print("\nUsage:")
        print("  python load_dac_weights.py <path_to_dac_checkpoint.pth>")
        print()
        print("Or in Python:")
        print("  from load_dac_weights import load_dac_weights_into_seis_codec")
        print("  model = SeisCodec()")
        print("  result = load_dac_weights_into_seis_codec(model, 'dac_44khz.pth')")
        print()
        print("Or to download official weights:")
        print("  from load_dac_weights import load_official_dac_44khz")
        print("  result = load_official_dac_44khz(model)")

        # Dry-run: show what WOULD be transferred with a dummy DAC
        print("\n" + "=" * 70)
        print("Dry-run: simulating weight transfer mapping")
        print("=" * 70)

        # Build a dummy DAC-like state dict to show the mapping
        from model import Encoder, Decoder, ResidualVectorQuantize

        class DummyDAC(nn.Module):
            def __init__(self):
                super().__init__()
                d = 64
                latent = d * 16  # 1024
                self.encoder = Encoder(d_in=1, d_model=d, strides=[2,4,8,8], d_latent=latent)
                self.quantizer = ResidualVectorQuantize(
                    input_dim=latent, n_codebooks=9, codebook_size=1024, codebook_dim=8)
                self.decoder = Decoder(latent, 1536, [8,8,4,2], d_out=1)

        dummy = DummyDAC()
        dac_sd = dummy.state_dict()
        seismic_sd = model.state_dict()

        match = sum(1 for k in dac_sd if k in seismic_sd and dac_sd[k].shape == seismic_sd[k].shape)
        mismatch = sum(1 for k in dac_sd if k in seismic_sd and dac_sd[k].shape != seismic_sd[k].shape)
        missing = sum(1 for k in dac_sd if k not in seismic_sd)

        print(f"  DAC params:              {len(dac_sd)}")
        print(f"  SeisCodec params:       {len(seismic_sd)}")
        print(f"  Exact shape match:       {match}")
        print(f"  Shape mismatch:          {mismatch}")
        print(f"  In DAC but not SeisCodec:  {missing}")

        print("\n  Shape mismatches:")
        for k in sorted(dac_sd):
            if k in seismic_sd and dac_sd[k].shape != seismic_sd[k].shape:
                print(f"    {k}: DAC {list(dac_sd[k].shape)} → SeisCodec {list(seismic_sd[k].shape)}")
