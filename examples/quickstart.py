"""archscope in ~40 lines: blocks, a stack, edges, a residual rail, a formula.

Run:  python examples/quickstart.py   →  out/examples/quickstart.{svg,png}
"""
from pathlib import Path

from archscope import Block, Diagram, Formula, IOLabel, OpDot, VStack

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Quickstart — a residual MLP block",
    subtitle="Blocks carry a semantic `kind` (color), a sublabel, and an optional "
             "source pointer; edges route perpendicular to box sides by default.")

# 1. build a column (measured & centered automatically), bottom-up flow
col = VStack([
    IOLabel("y  (B, T, 768)", id="out"),
    OpDot("+", id="add"),                                    # drawn glyph, not a font
    Block("MLP", kind="ffn", sub="Linear 768→3072 · GELU · Linear 3072→768",
          src="model.py:42", id="mlp"),
    Block("LayerNorm", kind="norm", id="ln"),
    IOLabel("x  (B, T, 768)", id="inp"),
], gap=22)
d.place(col, 200, 60)

# 2. connect by id — arrows enter/leave boxes perpendicular, with shape labels
d.chain(["inp", "ln", "mlp", "add", "out"], labels=[None, "(B, T, 768)", None, None])

# 3. a residual rail on the left
src, dst = d.box("inp"), d.box("add")
d.edge((src.x, src.cy), (dst.x, dst.cy), style_name="residual",
       via=[(120, src.cy), (120, dst.cy)])

# 4. vector math (matplotlib mathtext → outlined SVG paths)
f = Formula(r"$y = x + \mathrm{MLP}(\mathrm{LN}(x))$", size=14)
f.measure()
d.place(f, d.box("mlp").x2 + 40, d.box("mlp").cy - f.h / 2)

d.save(OUT / "quickstart.svg")
print("wrote", OUT / "quickstart.svg")
