"""The classic figure: a pre-LN transformer block (GPT-2 style), stacked ×12.

Shows: RepeatStack (card-stack with dashed ghosts), residual rails, KnownChip
(collapse what the reader already knows), legend.
"""
from pathlib import Path

from archscope import (Block, Diagram, GroupFrame, IOLabel, KnownChip, OpDot,
                       RepeatStack, Swatches, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Pre-LN Transformer block (GPT-2 style)",
    subtitle="Each sublayer is a residual branch x + f(LayerNorm(x)): the norm sits "
             "INSIDE the branch so the skip path stays an unnormalized identity. "
             "Attention then MLP, stacked ×12 identical layers.")

inner = VStack([
    OpDot("+", id="add2"),
    Block("MLP", kind="ffn", sub="Linear 768→3072 · GELU · Linear 3072→768", id="mlp"),
    Block("LayerNorm", kind="norm", id="ln2"),
    OpDot("+", id="add1"),
    Block("Multi-Head Attention", kind="attention",
          sub="12 heads × 64 · causal mask", id="attn"),
    Block("LayerNorm", kind="norm", id="ln1"),
], gap=22)

card = GroupFrame(inner, dashed=False, tint="#FFFFFF", stroke="#94A3B8", pad=20)
layer = RepeatStack(card, times="×12", id="layer")
col = VStack([
    IOLabel("hidden states  (B, T, 768)", id="out"),
    layer,
    IOLabel("hidden states  (B, T, 768)", id="inp"),
], gap=30)
d.place(col, 240, 60)

d.edge("inp", "layer")
d.chain(["ln1", "attn", "add1", "ln2", "mlp", "add2"])
d.edge("layer", "out")

# residual rails: tap the stream inside the card, feed the two add junctions
rail = d.box("layer").x + 9
cx = d.box("attn").cx
for tap_y, dst in [(d.box("ln1").y2 + 10, "add1"),
                   ((d.box("add1").y + d.box("ln2").y2) / 2, "add2")]:
    db = d.box(dst)
    d.edge((cx, tap_y), (db.x, db.cy), style_name="residual",
           via=[(rail, tap_y), (rail, db.cy)])

# the reader already knows attention? collapse it into a chip:
chip = KnownChip("scaled dot-product attention", ref="known", id="kc")
chip.measure()
ab = d.box("attn")
d.place(chip, d.box("layer").x - chip.w - 56, ab.cy - chip.h / 2)
d.edge("kc.r", (d.box("layer").x, ab.cy), style_name="faint", arrow=False,
       dash="3 3")

# the green-dashed KnownChip self-labels on the figure, so it needs no legend swatch
# (which would also collide with FFN green). Key the two edge weights instead.
leg = Swatches([("attention", "attention"), ("ffn", "MLP / FFN"),
                ("norm", "normalization"), ("io", "tensor"),
                ("main", "data flow", "edge"), ("residual", "residual", "edge")],
               max_w=240, id="leg")
d.place(leg, d.box("kc").x, 70)

d.save(OUT / "transformer_block.svg")
print("wrote", OUT / "transformer_block.svg")
