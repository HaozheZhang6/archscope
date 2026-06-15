"""Wan2.1-VACE-1.3B · 7 · AdaLN-single — how the timestep modulates each block.
diffsynth wan_video_dit.py:226 (modulation table), :66-67 (modulate), :204-209 (gate),
:233-244 (use inside DiTBlock).

Instead of a per-block MLP (DiT's adaLN-Zero), Wan keeps ONE learned table per block and adds
the shared timestep vector; the sum is chunked into 6 vectors that scale/shift/gate the two
residual branches. This is why timestep conditioning costs almost no parameters.
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, HStack, Formula,
                       Swatches, TextLabel, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 7 · AdaLN-single  ·  6 modulation params per block from table + timestep",
    subtitle="six = (modulation_table[1,6,1536] + t_mod[B,6,1536]).chunk(6) → "
             "shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp. modulate(y,sh,sc)=y·(1+sc)+sh is "
             "applied after norm1 / norm2; gate(x,g,branch)=x+g·branch wraps the self-attn and FFN residuals.")

# ---------------- left: producing the 6 params ------------------------------------
prod = VStack([
    HStack([
        Block("shift_msa", kind="cond", id="p0", min_w=92),
        Block("scale_msa", kind="cond", id="p1", min_w=92),
        Block("gate_msa", kind="cond", id="p2", min_w=92),
    ], gap=8),
    HStack([
        Block("shift_mlp", kind="cond", id="p3", min_w=92),
        Block("scale_mlp", kind="cond", id="p4", min_w=92),
        Block("gate_mlp", kind="cond", id="p5", min_w=92),
    ], gap=8),
    Block("chunk into 6  (along dim 1)", kind="linear", sub="each (B, 1536)", id="chunk", min_w=300),
    OpDot("+", id="sum"),
    HStack([
        Block("modulation table", kind="cond", sub="Parameter(1, 6, 1536)\nlearned, per block",
              id="tab", min_w=145),
        Block("t_mod", kind="cond", sub="(B, 6, 1536)\nfrom timestep, fig 3", id="tm", min_w=145),
    ], gap=10),
], gap=16)
d.place(prod, 70, 150)
d.edge("tab.t@0.5", "sum.l@0.5", b_side="l", style_name="cond")
d.edge("tm.t@0.5", "sum.r@0.5", b_side="r", style_name="cond")
d.chain(["sum", "chunk"])
d.edge("chunk.t@0.3", "p3.b@0.5", b_side="b", style_name="cond")
d.edge("chunk.t@0.7", "p0.b@0.5", b_side="b", style_name="cond")
d.note(70, 136, "the 6 params (each a 1536-vector):", size=style.T_SUB, color=style.INK, weight="600")

# ---------------- right: the two operations + where they sit ----------------------
RX = d.box("chunk").x2 + 120
d.note(RX, 150, "the two operations", size=style.T_SECTION, color=style.INK, weight="600")
ops = VStack([
    Formula(r"$\mathrm{modulate}(y,\,\mathrm{sh},\,\mathrm{sc})=y\cdot(1+\mathrm{sc})+\mathrm{sh}$", size=12.5),
    Formula(r"$\mathrm{gate}(x,\,g,\,\mathrm{branch})=x+g\cdot\mathrm{branch}$", size=12.5),
], gap=12, align="start")
d.place(ops, RX, 178)

# a mini one-branch diagram (the self-attn branch as the example)
mini = VStack([
    OpDot("+", id="m_add"),
    OpDot("o", id="m_gate"),
    Block("Self-Attention branch", kind="attention", sub="(fig 5)", id="m_attn", min_w=230),
    Block("modulate( · , shift_msa, scale_msa )", kind="cond", id="m_mod", min_w=230),
    Block("norm1  (LayerNorm, no affine)", kind="norm", id="m_norm", min_w=230),
    IOLabel("x", id="m_in"),
], gap=12)
d.place(mini, RX + 6, d.box("ops").y2 + 30 if False else 300)
d.chain(["m_in", "m_norm", "m_mod", "m_attn", "m_gate", "m_add"])
# residual rail — left of the widest mini box
rr = d.box("m_norm").x - 22
sb, db = d.box("m_in"), d.box("m_add")
d.edge((sb.x, sb.cy), (db.x, db.cy), style_name="residual", via=[(rr, sb.cy), (rr, db.cy)])
d.note(d.box("m_add").x2 + 14, d.box("m_add").cy + 1, "x + gate_msa · attn", size=style.T_SUB,
       color="#BE185D")
d.note(d.box("m_gate").x2 + 14, d.box("m_gate").cy + 1, "× gate_msa", size=style.T_SUB,
       color="#BE185D")
d.note(RX + 6, d.box("m_add").y - 22, "where they apply (self-attn branch; FFN branch is identical "
       "with the _mlp params):", size=style.T_SUB, color=style.MUTED, max_w=300)

d.note(70, d.box("tab").y2 + 22,
       "gate_* are initialized near zero, so a fresh block starts as the identity (x passes "
       "through) and learns to add its contribution — like adaLN-Zero, but the modulation is a "
       "cheap learned table plus the timestep, not a per-block MLP.",
       size=style.T_SUB, color=style.MUTED, max_w=440)

leg = Swatches([("cond", "modulation param / op"), ("linear", "chunk"), ("norm", "LayerNorm"),
                ("attention", "the gated branch"), ("io", "tensor"),
                ("op", "add", "glyph:+"), ("op", "gate", "glyph:o")], max_w=640, id="leg")
d.place(leg, 70, 96)

d.save(OUT / "fig07_adaln.svg")
print("ok")
