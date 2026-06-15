"""Shared constants for the Wan2.1-VACE-1.3B reconstruction deck.

Every figure is grounded in the real DiffSynth-Studio implementation:
  base DiT  : diffsynth/models/wan_video_dit.py        (WanModel, DiTBlock)
  VACE      : diffsynth/models/wan_video_vace.py        (VaceWanModel, VaceWanAttentionBlock)
  injection : diffsynth/pipelines/wan_video.py:1525-1573 (model_fn)
  config    : diffsynth/configs/model_configs.py:144     (Wan2.1-T2V-1.3B)

Wan2.1-T2V-1.3B numbers used throughout:
  dim=1536  ffn_dim=8960  num_heads=12  head_dim=128  num_layers=30
  in_dim=out_dim=16 (VAE latents)  text_dim=4096 (UMT5)  freq_dim=256 (timestep)
  patch_size=(1,2,2)  eps=1e-6
  VACE: vace_in_dim=96  vace_layers=(0,2,…,28) → 15 blocks → 15 hints
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "out" / "wan_vace_deep"

# the config, in one place so every figure cites the same numbers
DIM = 1536
FFN = 8960
HEADS = 12
HEAD_DIM = 128
LAYERS = 30
VACE_LAYERS = 15
