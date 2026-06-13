"""Autoregressive generation, illustrated with a concrete example (a flower).

The point: an autoregressive model is fed a SEQUENCE of tokens and outputs a
SEQUENCE of tokens — the same sequence shifted by one, i.e. at every position it
predicts the NEXT token. To make it concrete we tokenize a little flower image
into patch-tokens and interleave them with the text prompt (a multimodal token
stream, in the spirit of Mixture-of-Transformers), then show the next-token shift.

Drawing approach borrowed from multimodal-AR papers (tokens as colored chips, the
shifted input/output rows); the content and style here are our own.
"""
from pathlib import Path

from archscope import (Block, Diagram, IOLabel, PatchGrid, Swatches, TextLabel,
                       TokenRow, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

# ---- a tiny flower, as a 5x5 grid of patch colors (raster order = token order) ----
P, Y, G, B = "#F472B6", "#FCD34D", "#22C55E", "#EFF6FF"   # petal, center, stem, bg
FLOWER = [
    [B, P, P, P, B],
    [P, Y, Y, Y, P],
    [P, Y, Y, Y, P],
    [B, P, P, P, B],
    [B, B, G, B, B],
]
patches = [c for row in FLOWER for c in row]   # 25 patch tokens, raster order

d = Diagram(
    title="Autoregressive generation, illustrated — in: a token sequence, out: the next token at every position",
    subtitle="An image is a sequence of patch-tokens; prepend the text prompt and you have one multimodal "
             "token stream. The AR model maps the input sequence to the same sequence shifted by one — each "
             "position predicts the token that should come next. Generation = append the predicted token and repeat.")

# ============ PANEL A: an image is a sequence of patch tokens ============
flow = PatchGrid(FLOWER, cell=22, id="flower", outline="#94A3B8")
d.place(flow, 60, 110)
d.note(60, 96, "a 5×5 image", size=style.T_SUB, color=style.MUTED)
d.note(60, d.box("flower").y2 + 16, "= 25 patch tokens", size=style.T_SUB, color=style.MUTED)

# the flattened patch-token strip (first 8 + ellipsis)
strip = TokenRow(
    [dict(fill=patches[i], stroke="#64748B", w=24) for i in range(8)] +
    [dict(label="...", modality="none", w=30)],
    cell_w=24, cell_h=24, gap=3, id="strip")
d.place(strip, d.box("flower").x2 + 70, 132)
d.note(d.box("flower").x2 + 70, 118, "flatten in raster order ->", size=style.T_SUB,
       color=style.MUTED)
d.edge("flower", "strip", a_side="r", b_side="l")
d.note(d.box("strip").x, d.box("strip").y2 + 16,
       "p1  p2  p3  p4  p5  p6  p7  p8 ...", size=style.T_TINY + 1, color=style.FAINT,
       mono=True)

# ============ PANEL B: AR = predict the next token ============
BY = 290
# a window of the multimodal sequence: 3 text tokens then 5 image-patch tokens
TXT = ("#FEF3C7", "#D97706", "#78350F")   # text-token color (amber)
in_cells = [
    dict(label="a", fill=TXT[0], stroke=TXT[1], tcolor=TXT[2], w=46),
    dict(label="pink", fill=TXT[0], stroke=TXT[1], tcolor=TXT[2], w=46),
    dict(label="flower", fill=TXT[0], stroke=TXT[1], tcolor=TXT[2], w=52, gap_after=10),
    dict(fill=patches[1], stroke="#64748B", w=34, sub="p1"),
    dict(fill=patches[2], stroke="#64748B", w=34, sub="p2"),
    dict(fill=patches[3], stroke="#64748B", w=34, sub="p3"),
    dict(fill=patches[5], stroke="#64748B", w=34, sub="p4"),
    dict(fill=patches[6], stroke="#64748B", w=34, sub="p5"),
]
out_cells = [
    dict(label="pink", fill=TXT[0], stroke=TXT[1], tcolor=TXT[2], w=46),
    dict(label="flower", fill=TXT[0], stroke=TXT[1], tcolor=TXT[2], w=52),
    dict(fill=patches[1], stroke="#64748B", w=34, gap_after=10),
    dict(fill=patches[2], stroke="#64748B", w=34),
    dict(fill=patches[3], stroke="#64748B", w=34),
    dict(fill=patches[5], stroke="#64748B", w=34),
    dict(fill=patches[6], stroke="#64748B", w=34),
    dict(fill=patches[7], stroke="#64748B", w=34, bold_border=True, sub="p6  NEW"),
]

out_row = TokenRow(out_cells, cell_h=30, gap=3, id="out", label="output:", label_w=66)
d.place(out_row, 90, BY)
d.note(90, BY - 22, "OUT: the next token at every position  (input shifted by one)",
       size=style.T_SUB + 1, color="#166534", weight="600")

model = Block("Autoregressive Transformer  ·  causal self-attention",
              kind="model", sub="position i attends to tokens <= i, predicts token i+1",
              id="model", min_w=560)
d.place(model, 100, BY + 78)

in_row = TokenRow(in_cells, cell_h=30, gap=3, id="in", label="input:", label_w=66)
d.place(in_row, 90, BY + 170)
d.note(90, BY + 170 + 52,
       "IN: text prompt tokens (amber) + image patch tokens (the flower, in raster order)",
       size=style.T_SUB + 1, color=style.INK, weight="600")

# arrows: input -> model -> output (the shift). Draw a few representative ones.
ib, ob, mb = d.box("in"), d.box("out"), d.box("model")
d.edge((ib.cx, ib.y), (mb.cx, mb.y2), a_side="t", b_side="b", label="whole sequence")
d.edge((mb.cx, mb.y), (ob.cx, ob.y2), a_side="t", b_side="b")
# highlight the generation step: last input position -> predicted p6
d.note(ob.x2 + 18, ob.cy - 4, "<- predicted: append p6,", size=style.T_SUB + 0.5,
       color="#B91C1C")
d.note(ob.x2 + 18, ob.cy + 9, "   then feed back & repeat", size=style.T_SUB + 0.5,
       color="#B91C1C")

# the "image grows" callout: p6 fills the next patch of the flower
GROW = [row[:] for row in FLOWER]
# blank out everything after the 8th raster patch to show "so far", keep p<=8
flat_idx = 0
for r in range(5):
    for c in range(5):
        if flat_idx > 7:
            GROW[r][c] = "#F1F5F9"
        flat_idx += 1
grow = PatchGrid(GROW, cell=22, id="grow", outline="#B91C1C")
d.place(grow, mb.x2 + 70, BY + 70)
d.note(mb.x2 + 70, BY + 56, "the image so far", size=style.T_SUB, color="#B91C1C")
d.note(mb.x2 + 70, d.box("grow").y2 + 16, "each step fills one more patch",
       size=style.T_SUB, color=style.FAINT, max_w=150)

leg = Swatches([((TXT[0], TXT[1]), "text token"),
                (("#F472B6", "#64748B"), "image patch token"),
                ("model", "AR transformer"),
                (("#FFFFFF", "#B91C1C"), "predicted / new")], max_w=520, id="leg")
d.place(leg, 60, 70)

d.save(OUT / "autoregressive_flower.svg")
print("wrote", OUT / "autoregressive_flower.svg")
