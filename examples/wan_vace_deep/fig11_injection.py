"""Wan2.1-VACE-1.3B · 11 · how VACE meets the main DiT — the injection loop.
diffsynth pipelines/wan_video.py:1525-1573 (model_fn).

The VACE branch is run ONCE on the initial tokens to produce all 15 hints. Then the main DiT
runs its 30 blocks; after each EVEN block (0,2,…,28) the matching hint is added, scaled by
vace_scale. Odd blocks get nothing. That is the whole coupling.
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, Swatches, TextLabel, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 11 · injection — vace_hints added after every other main block",
    subtitle="vace_hints = vace(x, vace_context, …) is computed once (fig 9). Then for block_id in 0..29: "
             "x = block(x); if block_id in {0,2,…,28}: x = x + vace_hints[map[block_id]] · vace_scale. So 15 of 30 "
             "main blocks receive a hint; the others are untouched. vace_scale (default 1.0) tunes control strength.")

X = 330

# the main DiT block stack with injection points
spine = VStack([
    IOLabel("→ head → unpatchify → velocity v  (fig 1)", id="out", modality="video"),
    Block("main DiT block 29  (odd → no hint)", kind="frozen", id="b29", min_w=300),
    OpDot("+", id="i28"),
    Block("main DiT block 28  (even → hint)", kind="model", id="b28", min_w=300),
    Block("⋮   blocks 3–27   (even ones get a hint)", kind="model", sub="x += hint·scale after each even block",
          id="mid", min_w=300),
    OpDot("+", id="i2"),
    Block("main DiT block 2  (even → hint)", kind="model", id="b2", min_w=300),
    Block("main DiT block 1  (odd → no hint)", kind="frozen", id="b1", min_w=300),
    OpDot("+", id="i0"),
    Block("main DiT block 0  (even → hint)", kind="model", id="b0", min_w=300),
    IOLabel("tokens  x  (B, L, 1536)  (after patchify, fig 2)", id="in", modality="video", hatched=True),
], gap=13)
d.place(spine, X, 96)
d.chain(["in", "b0", "i0", "b1", "b2", "i2", "mid", "b28", "i28", "b29", "out"])

# the VACE branch (left) supplies the hints to the + nodes
vace = Block("VACE branch\n(fig 9)", kind="trainable", sub="15 hints", id="vace", min_w=130, h=130)
vace.measure()
d.place(vace, X - 250, d.box("i2").cy - vace.h / 2)
for inj, lab in [("i0", "hint@0 · scale"), ("i2", "hint@2 · scale"), ("i28", "hint@28 · scale")]:
    d.edge("vace.r@0.5", f"{inj}.l@0.5", a_side="r", b_side="l", style_name="cache",
           color="#B45309", label=lab, label_side="right", label_bg=True)

# layer map inset (bottom): which of the 30 layers get a hint
mx, my = X, d.box("in").y2 + 56
d.note(mx, my - 16, "the 30 main layers — amber = receives a VACE hint (even), grey = plain:",
       size=style.T_SUB, color=style.MUTED)
CW = 20
for i in range(30):
    even = (i % 2 == 0)
    fill, stroke = (("#FEF3C7", "#D97706") if even else ("#F1F5F9", "#CBD5E1"))
    d.doc.rect("nodes", mx + (i % 15) * (CW + 2), my + (i // 15) * (CW + 2), CW, CW, fill, stroke, 1.0, rx=3)
    d.doc.text("nodes", mx + (i % 15) * (CW + 2) + CW / 2, my + (i // 15) * (CW + 2) + CW / 2 + 3,
               str(i), style.T_TINY, style.MUTED)

d.note(mx, my + 2 * (CW + 2) + 16,
       "Note the timing (DiffSynth): all 15 hints are computed up front from the INITIAL tokens, then "
       "added as the main blocks run — the VACE branch does not re-read the main stream between blocks. "
       "The hint is a residual; vace_scale=0 recovers the plain Wan2.1 DiT.",
       size=style.T_SUB, color=style.FAINT, max_w=640)

leg = Swatches([("video", "tokens / latent"), ("model", "main block (+ hint here)"),
                ("frozen", "odd block (no hint)"), ("trainable", "VACE branch"),
                ("op", "add hint", "glyph:+"), ("cache", "hint · vace_scale", "edge")], max_w=820, id="leg")
d.place(leg, X - 250, 70)

d.save(OUT / "fig11_injection.svg")
print("ok")
