"""archscope quickstart — a residual MLP block in a handful of lines.

`d.flow([...])` stacks the boxes, auto-assigns ids from their labels, and connects
consecutive ones with arrows — no per-box coordinates and no manual chain. The
legend is generated from what was used; d.check() reports overlaps without rendering.

Run:  python examples/quickstart.py   →  out/examples/quickstart.{svg,png}
"""
from pathlib import Path

from archscope import Block, Diagram, Formula, IOLabel, OpDot

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Quickstart — a residual MLP block",
    subtitle="d.flow() stacks + auto-connects; d.auto_legend() builds the key; "
             "d.check() catches overlaps. Edges meet boxes perpendicular by default.")

# one call: stack bottom-up, auto-ids from labels, auto-arrows between neighbours
f = d.flow([
    IOLabel("x  (B, T, 768)"),
    Block("LayerNorm", kind="norm"),
    Block("MLP", kind="ffn", sub="768 → 3072 · GELU · 3072 → 768", src="model.py:42"),
    OpDot("+", id="add"),
    IOLabel("y  (B, T, 768)"),
], dir="up", x=240, y=120)

# a residual rail on the left (tap the input, feed the add)
inp = d.box("x_b_t_768")
d.edge((inp.x, inp.cy), "add.l", via=[(inp.x - 80, inp.cy), (inp.x - 80, d.box("add").cy)],
       style_name="residual", label="residual", label_side="left", label_bg=True)

# vector math, placed RELATIVE to the MLP box (no magic numbers)
d.place(Formula(r"$y = x + \mathrm{MLP}(\mathrm{LN}(x))$", size=14), right_of="mlp", gap=40)

d.auto_legend(160, 86)
d.save(OUT / "quickstart.svg")
print("wrote", OUT / "quickstart.svg", "· overlaps:", d.check())
