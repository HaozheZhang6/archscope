# Wan2.1-VACE-1.3B, module by module

A fine-grained reconstruction deck for the **Wan2.1-VACE-1.3B** video diffusion model.
Every figure is transcribed from the real implementation — no hand-waving — so that a
reader who knows only *transformers* and *flow matching* can rebuild the architecture.

Source of truth (DiffSynth-Studio):
- base DiT — `diffsynth/models/wan_video_dit.py` (`WanModel`, `DiTBlock`)
- VACE branch — `diffsynth/models/wan_video_vace.py` (`VaceWanModel`, `VaceWanAttentionBlock`)
- injection / flow — `diffsynth/pipelines/wan_video.py:1525-1573` (`model_fn`)
- config — `diffsynth/configs/model_configs.py:144` (Wan2.1-T2V-1.3B)

The numbers, everywhere: `dim=1536, ffn_dim=8960, heads=12, head_dim=128, layers=30,
in=out=16 (VAE latents), text_dim=4096 (UMT5), freq_dim=256, patch=(1,2,2)`.
VACE adds `vace_in_dim=96` and **15** blocks at layers `0,2,…,28`.

## Reading order

| # | figure | what it builds |
|---|--------|----------------|
| 1 | `fig01_system_flowmatch` | the model as a velocity field `v=f(x_σ,σ,text,control)`; the flow-matching sampling loop; VAE bookends — **the map** |
| 2 | `fig02_patchify_rope` | latent → tokens: strided `Conv3d` patchify, exact shapes, 3D RoPE (head split 44/42/42) |
| 3 | `fig03_conditioning` | timestep → AdaLN params `t_mod (6·1536)`; UMT5 text → cross-attn context |
| 4 | `fig04_ditblock` | **the DiTBlock, line by line** — self-attn / cross-attn / FFN, modulate + gate, residuals, the 6-param bus |
| 5 | `fig05_self_attention` | QK-RMSNorm + 3D RoPE, 12 heads, flash attention |
| 6 | `fig06_cross_attention` | video tokens query the text context (no RoPE) |
| 7 | `fig07_adaln` | AdaLN-single: `(table + t_mod).chunk(6)`, `modulate`, `gate`, where each applies |
| 8 | `fig08_vace_context` | the control signal: `vace_context = 96 = 16 (keep) + 16 (edit) + 64 (mask)` |
| 9 | `fig09_vace_branch` | the 15-block VACE side-network that emits 15 hints |
| 10 | `fig10_vace_block` | `VaceWanAttentionBlock = DiTBlock + before_proj + after_proj` (the zero-conv / ControlNet trick) |
| 11 | `fig11_injection` | how the hints meet the main DiT: `x += hint·vace_scale` after every even block |

## The whole thing in one paragraph

A video is VAE-encoded to a 16-channel latent and noised to level σ (flow matching).
A strided `Conv3d` cuts it into 1536-d tokens (fig 2). 30 `DiTBlock`s (fig 4) refine the
tokens: each does self-attention with QK-norm + 3D RoPE (fig 5), cross-attention to the
UMT5 text context (fig 6), and an FFN — all three modulated by the timestep through
AdaLN-single (figs 3, 7). A `Head` projects back and unpatchifies to a velocity of the
same shape; the sampler Euler-steps σ from 1 to 0 (fig 1). **VACE** adds control: a masked
control video becomes a 96-channel `vace_context` (fig 8); a separate 15-block branch
(fig 9), built from DiT blocks wrapped in zero-conv projections (fig 10), turns it into 15
hints that are added into the main DiT after layers 0,2,…,28 (fig 11). Train only the VACE
branch; the zero-conv init means it starts as a no-op.

## Build

```bash
cd examples/wan_vace_deep
python fig04_ditblock.py        # → out/wan_vace_deep/fig04_ditblock.{svg,png}
```
