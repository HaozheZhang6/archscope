"""Fig 3 — WanTransformerBlock (code version): DiT block with per-token AdaLN."""
from archscope import (Block, Diagram, IOLabel, OpDot, Spacer, Swatches,
                       TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 3 · WanTransformerBlock — one of 30 identical layers (code version)",
    subtitle="Pre-norm DiT block: self-attention over the interleaved video-action sequence, "
             "cross-attention to text, gated by per-token AdaLN. All weights are shared by "
             "both modalities — wan_va/modules/model.py:468-566")

X = 330

main = VStack([
    IOLabel("hidden states  (1, L, 3072)", id="out"),
    OpDot("+", id="add3"),
    OpDot("x", id="gate3"),
    Block("FFN", kind="ffn", sub="Linear 3072→14336 · GELU(approx) · Linear 14336→3072",
          src="model.py:507", id="ffn"),
    OpDot("o", id="mod3"),
    Block("LayerNorm", kind="norm", sub="no affine", id="norm3"),
    OpDot("+", id="add2"),
    Block("Cross-Attention", kind="attention",
          sub="Q ← sequence · KV ← text (1, B·512, 3072)", src="model.py:494  attn2",
          id="xattn"),
    Block("LayerNorm", kind="norm", sub="affine", id="norm2"),
    OpDot("+", id="add1"),
    OpDot("x", id="gate1"),
    Block("Self-Attention", kind="attention",
          sub="WanAttention: RMSNorm(Q,K) · 3D RoPE · FlexAttention mask",
          src="model.py:484  attn1   → Fig 4 / Fig 5", id="attn"),
    OpDot("o", id="mod1"),
    Block("LayerNorm", kind="norm", sub="no affine", id="norm1"),
    IOLabel("hidden states  (1, L, 3072)", id="inp"),
], gap=17)
d.place(main, X, 60)

# main chain, bottom-up
chain = ["inp", "norm1", "mod1", "attn", "gate1", "add1", "norm2", "xattn",
         "add2", "norm3", "mod3", "ffn", "gate3", "add3", "out"]
d.chain(chain)

# residual rails (left)
rail = X - 86
for src, dst in [("inp", "add1"), ("add1", "add2"), ("add2", "add3")]:
    sb, db = d.box(src), d.box(dst)
    d.edge((sb.x, sb.cy), (db.x, db.cy), style_name="residual",
           via=[(rail, sb.cy), (rail, db.cy)], arrow=True)
d.note(rail - 8, d.box("add2").cy - 40, "residual", anchor="end",
       size=style.T_SUB, color=style.FAINT)

# AdaLN conditioning (right): one bus with four branches
mods = [("mod1", "shift_msa, scale_msa"), ("gate1", "gate_msa"),
        ("mod3", "c_shift, c_scale"), ("gate3", "c_gate")]
rail = max(d.box(i).x2 for i in ("attn", "xattn", "ffn")) + 40
ys = [d.box(m).cy for m, _ in mods]

ada = Block("per-token AdaLN", kind="cond",
            sub="t-emb (1, L, 6·3072) + scale_shift_table (1, 6, 3072)",
            src="model.py:512, 524-532", id="ada")
ada.measure()
d.place(ada, rail + 30, (min(ys) + max(ys)) / 2 - ada.h / 2)
ab = d.box("ada")
d.edge((ab.x, ab.cy), (rail, ab.cy), style_name="cond", arrow=False,
       route="straight")
d.doc.line("edges", rail, min(ys), rail, max(ys), "#DB2777", 1.2, dash="5 3")
for m, lab in mods:
    db = d.box(m)
    d.edge((rail, db.cy), (db.x2, db.cy), style_name="cond", route="straight")
    d.note(db.x2 + 12, db.cy - 6, lab, size=style.T_TINY + 0.5,
           color="#BE185D", mono=True)
expl = TextLabel("each token carries its own timestep — video, clean-history "
                 "and action segments are conditioned on different noise "
                 "levels within the same forward pass",
                 size=style.T_SUB + 0.5, color=style.MUTED, max_w=230)
expl.measure()
d.place(expl, ab.x, ab.y2 + 12)

# text input (left of cross-attn) — sits LEFT of the residual rail (x=X-86) so the
# rail's vertical run never pierces it; the short hop to xattn simply crosses the rail.
txt = IOLabel("text emb  (1, B·512, 3072)", id="txt", modality="text")
xb = d.box("xattn")
d.place(txt, X - 285, xb.cy - 11)
d.edge("txt.r", (xb.x, xb.cy), style_name="cond", color="#DB2777")

leg = Swatches([("attention", "attention"), ("ffn", "MLP / FFN"),
                ("norm", "normalization"), ("cond", "conditioning"),
                ("io", "tensor")], id="leg")
d.place(leg, X - 86, d.box("out").y - 46)

d.save(OUT / "fig03_block.svg")
print("ok")
