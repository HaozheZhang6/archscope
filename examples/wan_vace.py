"""Wan-VACE — a ControlNet-style control branch for a video diffusion DiT (1.3B).

VACE adds a parallel, trainable branch of DiT blocks that read a control signal
(a masked video + its mask, optionally a reference image) and inject residual
"hints" back into the frozen main DiT at every other layer:  x = x + hint·scale.
The after-projection plays the role of ControlNet's zero-conv.

Grounded in DiffSynth-Studio: diffsynth/models/wan_video_vace.py and the injection
loop in diffsynth/pipelines/wan_video.py:1525-1573.
"""
from pathlib import Path

from archscope import (Block, Diagram, IOLabel, RepeatStack, Swatches,
                       TextLabel, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Wan-VACE (1.3B) — a ControlNet-style control branch for video diffusion",
    subtitle="A parallel branch of 15 trainable DiT blocks reads the control signal and adds residual hints "
             "into the FROZEN main DiT at every other layer (0,2,…,28): x = x + hint·vace_scale. The "
             "after-projection is the zero-conv. diffsynth/models/wan_video_vace.py + pipelines/wan_video.py")

# ============ MAIN DiT (left, frozen) ============
MX = 150
main = VStack([
    IOLabel("denoised video latent", id="m_out", modality="video"),
    RepeatStack(Block("DiT block", kind="model",
                      sub="self-attn · cross-attn(text) · FFN · AdaLN(t)", id="m_blk", min_w=240),
                times="×30", id="m_dit"),
    IOLabel("noised video latent  (16, F, h, w)", id="m_in", modality="video", hatched=True),
], gap=24)
d.place(main, MX, 110)
d.chain(["m_in", "m_dit", "m_out"])
d.note(MX, 92, "main Wan DiT — FROZEN", size=style.T_SUB + 1, color=style.INK, weight="600")

# ============ VACE branch (right, trainable) ============
VX = 620
vace = VStack([
    IOLabel("15 hints  ->  injected into the main DiT", id="v_out", modality="state"),
    RepeatStack(Block("VaceWanAttentionBlock", kind="model",
                      sub="= DiT block + before_proj (block0) + after_proj (zero-conv)",
                      src="wan_video_vace.py:5-24", id="v_blk", min_w=300),
                times="×15", id="v_dit"),
    Block("vace_patch_embedding", kind="linear", sub="Conv3d 96 -> 1536, kernel (1,2,2)",
          src="wan_video_vace.py:51", id="v_patch", min_w=300),
    IOLabel("vace_context  (96, F, h, w)", id="v_in", modality="state"),
], gap=22)
d.place(vace, VX, 110)
d.chain(["v_in", "v_patch", "v_dit", "v_out"])
d.note(VX, 92, "VACE branch — TRAINABLE  (15 blocks at layers 0,2,…,28)",
       size=style.T_SUB + 1, color="#B45309", weight="600")

# the hint injection arrow VACE -> main (upper)
d.edge("v_dit.l@0.30", "m_dit.r@0.30", style_name="cache", color="#B45309", width=1.8,
       label="x = x + hint · vace_scale", label_side="right")
# block-0 injects the main hidden state (lower, clearly separated)
d.edge("m_dit.r@0.78", "v_dit.l@0.80", style_name="faint", dash="4 3",
       color="#94A3B8", label="block0: c = before_proj(c) + x", label_side="left")

# ============ control-signal prep (bottom) ============
prep = VStack([
    Block("vace_context = [ inactive | reactive | mask ]  = 96 ch", kind="cond",
          sub="inactive=VAE(video×(1-mask)) 16 · reactive=VAE(video×mask) 16 · mask 64",
          src="wan_video.py:676-706", id="prep", min_w=420),
    Block("control video  +  binary mask  (+ optional reference image)", kind="io",
          sub="keep region vs edit/generate region", id="ctrl", min_w=420),
], gap=14)
prep.measure()
d.place(prep, VX + 20, d.box("v_in").y2 + 50)
d.edge("ctrl", "prep")
d.edge("prep", "v_in", b_side="b")

# ============ the every-other-layer injection map (inset) ============
ix, iy = MX, d.box("m_in").y2 + 60
d.note(ix, iy - 14, "injection map: hints land at 15 of 30 layers (every other)",
       size=style.T_SUB, color=style.MUTED)
CW = 22
for i in range(30):
    even = (i % 2 == 0)
    fill, stroke = (("#FEF3C7", "#B45309") if even else ("#F1F5F9", "#CBD5E1"))
    d.doc.rect("nodes", ix + (i % 15) * (CW + 2), iy + (i // 15) * (CW + 2), CW, CW,
               fill, stroke, 1.0, rx=3)
    d.doc.text("nodes", ix + (i % 15) * (CW + 2) + CW / 2,
               iy + (i // 15) * (CW + 2) + CW / 2 + 3, str(i), style.T_TINY, style.MUTED)
d.note(ix, iy + 2 * (CW + 2) + 14,
       "amber = a VACE layer (hint added) · grey = plain DiT layer. Only the VACE branch "
       "and its embeddings are trained; the main DiT stays frozen.", size=style.T_SUB,
       color=style.FAINT, max_w=560)

leg = Swatches([("video", "main video latent"), ("state", "control / hints"),
                ("model", "DiT block"), ("linear", "patch embed"),
                ("cond", "control composition"), (("#FEF3C7", "#B45309"), "VACE layer")],
               max_w=640, id="leg")
d.place(leg, MX, 70)

d.save(OUT / "wan_vace.svg")
print("wrote", OUT / "wan_vace.svg")
