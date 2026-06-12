"""Fig 4 — inside WanAttention: QK-RMSNorm, 3D RoPE, and the KV-cache pool."""
from archscope import (Block, Diagram, HStack, IOLabel, KnownChip, Spacer,
                       Swatches, TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 4 · WanAttention — what is non-standard inside the attention (code version)",
    subtitle="Standard scaled-dot-product attention at the core (collapsed: known concept); the "
             "Wan-specific parts are post-projection QK-RMSNorm, 3D axial RoPE shared by video "
             "AND action tokens, and a slot-pool KV cache for autoregressive inference. "
             "wan_va/modules/model.py:289-465")

X = 380

main = VStack([
    IOLabel("attn out  (1, L, 3072)", id="out"),
    Block("to_out", kind="linear", sub="Linear 3072 → 3072 · Dropout", src="model.py:321",
          id="o"),
    Block("attention op", kind="attention",
          sub="train: FlexAttention + mask (Fig 5) · infer: torch SDPA over cached KV",
          src="model.py:302-307, 455", id="attn"),
    Block("3D RoPE on q, k", kind="cond",
          sub="complex rotation per head · 128 dims = 44 (frame) + 42 (y) + 42 (x)",
          src="model.py:258-260, 432-441", id="rope"),
    Block("split heads", kind="op", sub="(1, L, 3072) → (1, L, 24, 128)", id="heads"),
    Block("RMSNorm_q · RMSNorm_k", kind="norm",
          sub="affine, over the full 3072 (before head split) · v is NOT normed",
          src="model.py:325-330, 427-429", id="qknorm"),
    Block("to_q · to_k · to_v", kind="linear", sub="3 × Linear 3072 → 3072 (bias)",
          src="model.py:318-320", id="qkv"),
    IOLabel("normed hidden states  (1, L, 3072)", id="inp"),
], gap=20, id="main")
d.place(main, X, 60)
d.chain(["inp", "qkv", "qknorm", "heads", "rope", "attn", "o", "out"])

# the standard core, collapsed as a known concept beside the attention op
ab = d.box("attn")
chip = KnownChip("softmax(qk/√d)·v — standard SDPA core", ref="known", id="sdpa")
chip.measure()
d.place(chip, ab.x - chip.w - 64, ab.cy - chip.h / 2)
d.edge("sdpa.r", (ab.x, ab.cy), style_name="faint", arrow=False, dash="3 3")

# ---- left: RoPE grids ---------------------------------------------------------------
rb = d.box("rope")
grids = VStack([
    TextLabel("one shared position grid (f, y, x):", size=style.T_SUB + 1,
              color=style.INK, weight="600", anchor="start"),
    TextLabel("video token  = (frame, y, x) after 2×2 patchify",
              size=style.T_SUB + 0.5, color=style.MUTED, mono=True, anchor="start"),
    TextLabel("action token = (frame, 1..16, 1)",
              size=style.T_SUB + 0.5, color=style.MUTED, mono=True, anchor="start"),
    TextLabel("actions are 16 extra “pixels” of their frame —",
              size=style.T_SUB + 0.5, color=style.FAINT, anchor="start"),
    TextLabel("temporal alignment with video comes for free",
              size=style.T_SUB + 0.5, color=style.FAINT, anchor="start"),
], gap=7, align="start")
grids.measure()
gx = rb.x - grids.w - 70
d.place(grids, gx, rb.cy - grids.h / 2)
d.edge((gx + grids.w + 8, rb.cy), (rb.x, rb.cy), style_name="cond", arrow=True)

# ---- right: KV cache pool -------------------------------------------------------------
mb = d.box("main")
cache = VStack([
    Block("KV cache pool  (inference only)", kind="mask",
          sub="pre-alloc (window/2)·(tokens per chunk) slots per layer",
          src="model.py:331-409", id="cache", min_w=280),
    TextLabel("update_cache semantics:", size=style.T_SUB + 1, color=style.INK,
              weight="600", anchor="start"),
    TextLabel("0 — attend, then free the slots (denoising", size=style.T_SUB + 0.5,
              color=style.MUTED, anchor="start"),
    TextLabel("      iterations leave no trace in the cache)", size=style.T_SUB + 0.5,
              color=style.MUTED, anchor="start"),
    TextLabel("1 — keep, marked is_pred (imagined video,", size=style.T_SUB + 0.5,
              color=style.MUTED, anchor="start"),
    TextLabel("      evicted when the real observation arrives)", size=style.T_SUB + 0.5,
              color=style.MUTED, anchor="start"),
    TextLabel("2 — keep as committed history (real obs,", size=style.T_SUB + 0.5,
              color=style.MUTED, anchor="start"),
    TextLabel("      executed actions)", size=style.T_SUB + 0.5,
              color=style.MUTED, anchor="start"),
    TextLabel("pool full → evict oldest insertion ids first", size=style.T_SUB + 0.5,
              color=style.FAINT, anchor="start"),
], gap=7, align="start")
cache.measure()
d.place(cache, mb.x2 + 130, ab.cy - 24)
cb = d.box("cache")
d.edge((cb.x, ab.cy), (ab.x2, ab.cy), style_name="cache", route="straight",
       label="k, v of history", label_at=0.5, label_dy=-2, label_anchor="middle")

leg = Swatches([("attention", "attention"), ("norm", "normalization"),
                ("linear", "linear"), ("cond", "position/conditioning"),
                ("mask", "cache"), ("known", "known concept", "dash")],
               max_w=170, id="leg")
leg.measure()
d.place(leg, gx, 70)

d.save(OUT / "fig04_attention.svg")
print("ok")
