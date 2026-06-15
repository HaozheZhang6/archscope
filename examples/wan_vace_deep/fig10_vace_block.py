"""Wan2.1-VACE-1.3B · 10 · VaceWanAttentionBlock = DiTBlock + before_proj + after_proj.
diffsynth models/wan_video_vace.py:5-24.

This is the ControlNet trick. A VACE block IS a full DiT block (fig 4), wrapped with two
Linear layers: before_proj (block 0 only) injects the main latent x into the control stream;
after_proj (a zero-conv) reads out the hint. Initialized to zero so VACE starts as a no-op.
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, Swatches, TextLabel, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 10 · VaceWanAttentionBlock  ·  a DiT block wrapped with before/after_proj",
    subtitle="block 0: c ← before_proj(c) + x  (seed the control stream with the main latent). Every block: run "
             "the full DiTBlock on c (self-attn + cross-attn(text) + FFN + AdaLN, fig 4), then hint = after_proj(c). "
             "after_proj is the zero-conv — weights init 0, so an untrained VACE adds nothing.")

X = 360

spine = VStack([
    IOLabel("hint  (B, L, 1536)   → injected into main DiT (fig 11)", id="hint", modality="none"),
    Block("after_proj  ·  Linear 1536→1536  (zero-conv)", kind="trainable",
          sub="init 0 → no effect until trained", src="vace.py:11", id="after", min_w=340),
    Block("DiTBlock  (the FULL block, fig 4)", kind="model",
          sub="self-attn · cross-attn(text) · FFN · AdaLN(t)  ·  super().forward", src="vace.py:20",
          id="dit", min_w=340),
    OpDot("+", id="add"),
    Block("before_proj  ·  Linear 1536→1536   (block 0 only)", kind="trainable",
          sub="seeds the control stream", src="vace.py:10", id="before", min_w=340),
    IOLabel("control tokens  c   (from patch-embed, fig 9)", id="c", modality="state"),
], gap=20)
d.place(spine, X, 130)
d.chain(["c", "before", "add", "dit", "after", "hint"])

# the main latent x is injected at the '+' (block 0)
xin = IOLabel("main latent  x   (block 0 only)", id="x", modality="video", hatched=True)
d.place(xin, X - 280, d.box("add").cy - 11)
d.edge("x.r@0.5", "add.l@0.5", a_side="r", b_side="l", style_name="main", label="+ x", label_bg=True)

# c also passes forward to the NEXT vace block (the running stream)
d.note(d.box("dit").x2 + 26, d.box("dit").cy - 6,
       "the block's output c also\ncontinues to the next VACE\nblock (the running stream);\n"
       "after_proj only TAPS a copy", size=style.T_SUB, color=style.FAINT, max_w=190)
d.edge("dit.r@0.5", (d.box("dit").x2 + 200, d.box("dit").cy), a_side="r", arrow=True,
       style_name="faint", label="c → next block", label_side="right", label_bg=True)

# text / timestep / rope shared
sh = IOLabel("shared: text, t_mod, RoPE", id="sh", modality="text")
sh.measure()
d.place(sh, d.box("dit").x - sh.w - 36, d.box("dit").cy - 11)
d.edge("sh.r@0.5", "dit.l@0.5", a_side="r", b_side="l", style_name="cond", color="#DB2777")

d.note(X - 280, d.box("c").y2 if False else d.box("x").y2 + 30,
       "Why a zero-conv? With after_proj (and before_proj) initialized to zero, the very first "
       "training step injects exactly nothing, so the frozen main DiT keeps its pretrained behaviour "
       "and VACE learns its control signal as a residual on top — the ControlNet recipe.",
       size=style.T_SUB, color=style.MUTED, max_w=560)

leg = Swatches([("state", "control tokens"), ("video", "main latent"), ("model", "DiT block (fig 4)"),
                ("trainable", "VACE Linear (trained)"), ("none", "hint"),
                ("main", "data flow", "edge"), ("cond", "text/t/RoPE", "edge"),
                ("faint", "running stream", "edge")], max_w=900, id="leg")
d.place(leg, X - 280, 96)

d.save(OUT / "fig10_vace_block.svg")
print("ok")
