"""Wan2.1-VACE-1.3B · 4 · the DiTBlock — one of 30 identical blocks, transcribed line-by-line
from diffsynth/models/wan_video_dit.py:229-245 (DiTBlock.forward).

Three sub-layers, each a residual branch: (1) self-attention, AdaLN-modulated and GATED;
(2) cross-attention to text, plain residual (no gate, its own affine norm); (3) FFN,
AdaLN-modulated and gated. modulate(y,sh,sc)=y·(1+sc)+sh; gate adds g·branch.
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, Swatches, TextLabel, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 4 · DiTBlock  ·  ×30 identical blocks  (dim=1536, heads=12, ffn=8960)",
    subtitle="x ← x + gate_msa · SelfAttn(modulate(norm1 x)) ; x ← x + CrossAttn(norm3 x, text) ; "
             "x ← x + gate_mlp · FFN(modulate(norm2 x)). The 6 modulation params come from a per-block learned "
             "table + the timestep (fig 7). norm1/norm2 are LayerNorm WITHOUT affine; norm3 LayerNorm WITH affine.")

X = 360

# ---------------- the main spine, bottom -> top ----------------------------------
spine = VStack([
    IOLabel("hidden states  x  (B, L, 1536)", id="out"),
    OpDot("+", id="add3"),
    OpDot("o", id="g3"),
    Block("FFN", kind="ffn", sub="Linear 1536→8960 · GELU(tanh) · Linear 8960→1536",
          src="dit.py:224", id="ffn", min_w=300),
    Block("modulate  (shift_mlp, scale_mlp)", kind="cond", sub="y·(1+scale)+shift", id="ss_mlp"),
    Block("norm2  ·  LayerNorm (no affine)", kind="norm", id="norm2"),
    OpDot("+", id="add2"),
    Block("Cross-Attention  ← text context", kind="attention",
          sub="Q←x · K,V←text(1536)  ·  fig 6", src="dit.py:242", id="xattn", min_w=300),
    Block("norm3  ·  LayerNorm (affine)", kind="norm", id="norm3"),
    OpDot("+", id="add1"),
    OpDot("o", id="g1"),
    Block("Self-Attention  (3D RoPE on q,k)", kind="attention",
          sub="RMSNorm(q), RMSNorm(k) · flash-attn  ·  fig 5", src="dit.py:241", id="sattn", min_w=300),
    Block("modulate  (shift_msa, scale_msa)", kind="cond", sub="y·(1+scale)+shift", id="ss_msa"),
    Block("norm1  ·  LayerNorm (no affine)", kind="norm", id="norm1"),
    IOLabel("hidden states  x  (B, L, 1536)", id="in"),
], gap=12)
d.place(spine, X, 96)
d.chain(["in", "norm1", "ss_msa", "sattn", "g1", "add1", "norm3", "xattn", "add2",
         "norm2", "ss_mlp", "ffn", "g3", "add3", "out"])

# ---------------- text context enters cross-attention (left) ----------------------
txt = IOLabel("text context  (B, 512, 1536)", id="txt", modality="text")
xb = d.box("xattn")
d.place(txt, X - 320, xb.cy - 11)   # left of the residual rail (X-92) so it isn't pierced
d.edge("txt.r", (xb.x, xb.cy), style_name="cond", color="#DB2777")

# ---------------- residual rails (left) -------------------------------------------
rail = X - 92
for src, dst in [("in", "add1"), ("add1", "add2"), ("add2", "add3")]:
    sb, db = d.box(src), d.box(dst)
    d.edge((sb.x, sb.cy), (db.x, db.cy), style_name="residual",
           via=[(rail, sb.cy), (rail, db.cy)], arrow=True)
d.note(rail - 8, d.box("add1").cy + 30, "residual", anchor="end", size=style.T_SUB,
       color=style.FAINT)

# ---------------- modulation bus (right): 6 params tap the gates/modulates ---------
mods = [("ss_msa", "shift_msa, scale_msa"), ("g1", "gate_msa"),
        ("ss_mlp", "shift_mlp, scale_mlp"), ("g3", "gate_mlp")]
rail2 = max(d.box(i).x2 for i in ("ffn", "xattn", "sattn")) + 44
ys = [d.box(m).cy for m, _ in mods]
ada = Block("modulation table + t_mod  →  chunk 6", kind="cond",
            sub="Param(1,6,1536) + timestep(6·1536)  ·  fig 7", src="dit.py:233", id="ada", min_w=240)
ada.measure()
d.place(ada, rail2 + 28, (min(ys) + max(ys)) / 2 - ada.h / 2)
d.edge((d.box("ada").x, d.box("ada").cy), (rail2, d.box("ada").cy), style_name="cond",
       arrow=False, route="straight")
d.doc.line("edges", rail2, min(ys), rail2, max(ys), "#DB2777", 1.2, dash="5 3")
for m, lab in mods:
    bb = d.box(m)
    d.edge((rail2, bb.cy), (bb.x2, bb.cy), style_name="cond", route="straight")
    d.note(bb.x2 + 10, bb.cy - 6, lab, size=style.T_TINY + 0.5, color="#BE185D", mono=True)
d.note(d.box("ada").x, d.box("ada").y2 + 12,
       "6 params = shift/scale/gate for the self-attn branch + shift/scale/gate for the FFN "
       "branch. gates init small → block starts near identity (fig 7).", size=style.T_SUB,
       color=style.FAINT, max_w=250)

# the three op glyphs explained inline
d.note(d.box("g1").x2 + 16, d.box("g1").cy + 1, "⊙ = ×gate", size=style.T_TINY + 0.5,
       color=style.FAINT)

leg = Swatches([("attention", "attention"), ("ffn", "FFN"), ("norm", "LayerNorm"),
                ("cond", "AdaLN modulation"), ("io", "tensor"),
                ("op", "add (residual)", "glyph:+"), ("op", "gate (×g)", "glyph:o"),
                ("residual", "residual", "edge"), ("cond", "modulation / text", "edge")],
               max_w=640, id="leg")
d.place(leg, X - 92, d.box("out").y - 44)

d.save(OUT / "fig04_ditblock.svg")
print("ok")
