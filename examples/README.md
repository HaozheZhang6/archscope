# Examples

Every example is a standalone script; outputs land in [`out/`](../out/) as
self-contained SVG (+ 4× PNG if `resvg` is installed).

| script | what it teaches | output |
|---|---|---|
| [`quickstart.py`](quickstart.py) | the API in ~40 lines: blocks, stacks, chains, a residual rail, vector math | [svg](../out/examples/quickstart.svg) |
| [`transformer_block.py`](transformer_block.py) | GPT-2 pre-LN block: `RepeatStack` card-stack ×12, residual rails inside the card, known-concept chip | [svg](../out/examples/transformer_block.svg) |
| [`qwen3_block.py`](qwen3_block.py) | a REAL implementation drawn with `file:line` receipts (nano-vllm): fused GQA shapes, per-head QK-RMSNorm, two-branch SwiGLU | [svg](../out/examples/qwen3_block.svg) |
| [`training_methods.py`](training_methods.py) | training objectives as storyboards: GPT next-token vs BERT masked-LM, mask grids computed from the rule | [svg](../out/examples/training_methods.svg) |
| [`flow_matching.py`](flow_matching.py) | the rectified-flow objective (SD3/FLUX/Wan): path → training → sampling panels | [svg](../out/examples/flow_matching.svg) |
| [`dit_latent_diffusion.py`](dit_latent_diffusion.py) | a DiT denoiser drawn so the prediction is unmistakable: noised latent + t + condition IN → predicted noise (same shape) OUT, with the adaLN-Zero block detail | [svg](../out/examples/dit_latent_diffusion.svg) |
| [`autoregressive_flower.py`](autoregressive_flower.py) | the illustrated style with a REAL photo: a tulip → patch tokens → a multimodal sequence; AR = input sequence in, the same sequence shifted by one out (predict the next token). Uses `RasterImage` | [svg](../out/examples/autoregressive_flower.svg) |
| [`diffusion_flower.py`](diffusion_flower.py) | the diffusion counterpart on the same photo: the noising trajectory x_0→x_1, then one denoising step made explicit (noised image + t IN → predicted noise, same shape, OUT), σ_t→σ_{t-1}, looped ×T. Contrasts AR | [svg](../out/examples/diffusion_flower.svg) |
| [`prediction_targets.py`](prediction_targets.py) | what the denoiser actually predicts: x_σ on the line between clean x_0 and noise ε, and the three equivalent targets — ε-pred, x_0-pred, v-pred — with their losses and conversions | [svg](../out/examples/prediction_targets.svg) |
| [`lingbot_va/`](lingbot_va/) | full case study: 8 figures drawn by an agent reading Ant Group's LingBot-VA code + paper, incl. the paper-vs-code MoT discrepancy | [gallery](../out/lingbot_va/README.md) |

```bash
pip install -e ..        # from this directory, or `pip install -e .` from the root
python quickstart.py
```
