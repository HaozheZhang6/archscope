"""Autoregressive generation, illustrated with a REAL image (a tulip photo, CC0).

The point: an autoregressive model is fed a SEQUENCE of tokens and outputs a
SEQUENCE of tokens — the input shifted by one, i.e. at every position it predicts
the NEXT token. We tokenize a real photo into patch-tokens and interleave them
with the text prompt (one multimodal stream, in the spirit of Mixture-of-
Transformers), then show the next-token shift and the image growing patch by patch.

Image: CC0 1.0 (public domain) "flower" by imuzio — see assets/flower/NOTICE.md.
Drawing approach borrowed from multimodal-AR papers; content and style are our own.
"""
from pathlib import Path

from archscope import (Block, Diagram, RasterImage, Swatches, TextLabel,
                       TokenRow, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"
ASSETS = Path(__file__).resolve().parent / "assets" / "flower"
def P(r, c):
    return str(ASSETS / f"p_{r}_{c}.png")

N = 6
raster = [(r, c) for r in range(N) for c in range(N)]   # patch token order

d = Diagram(
    title="Autoregressive generation, illustrated — in: a token sequence, out: the next token at every position",
    subtitle="A real image is a sequence of patch-tokens; prepend the text prompt and you have one multimodal "
             "token stream. The AR model maps the input sequence to the same sequence shifted by one — each "
             "position predicts the token that comes next. Generation = append the predicted token and repeat.")

# ============ PANEL A: a real image -> patch tokens ============
AY = 110
img = RasterImage(str(ASSETS / "full.png"), 132, id="img", rx=6, outline="#94A3B8")
d.place(img, 60, AY)
d.note(60, AY - 14, "a real photo", size=style.T_SUB, color=style.MUTED)
d.note(60, d.box("img").y2 + 15, "(tulip, CC0)", size=style.T_TINY + 1, color=style.FAINT)

# exploded 6x6 patch grid (gaps make the patchification visible)
PC, PG = 24, 3
gx, gy = d.box("img").x2 + 90, AY
for r in range(N):
    for c in range(N):
        RasterImage.emit(d.doc, P(r, c), gx + c * (PC + PG), gy + r * (PC + PG),
                         PC, PC, rx=3)
        d.doc.rect("nodes", gx + c * (PC + PG), gy + r * (PC + PG), PC, PC,
                   "none", "#FFFFFF", 0.8, rx=3)
gw = N * (PC + PG) - PG
d.note(gx, gy - 14, "split into 6×6 patches", size=style.T_SUB, color=style.MUTED)
d.note(gx, gy + gw + 16, "= 36 patch tokens (raster order)", size=style.T_SUB,
       color=style.MUTED)
# arrow image -> grid
d.edge("img", (gx - 12, gy + gw / 2), a_side="r", b_side="l", label="tokenize")

# flattened strip (first 8 patches + ellipsis)
sx = gx + gw + 80
strip = TokenRow(
    [dict(img=P(*raster[i]), stroke="#64748B", w=26) for i in range(8)] +
    [dict(label="...", modality="none", w=30)], cell_w=26, cell_h=26, gap=3, id="strip")
d.place(strip, sx, AY + gw / 2 - 13)
d.note(sx, AY + gw / 2 - 30, "flatten ->", size=style.T_SUB, color=style.MUTED)
d.edge((gx + gw + 12, gy + gw / 2), "strip.l", b_side="l")
d.note(sx, d.box("strip").y2 + 15, "p1  p2  p3  p4  p5  p6  p7  p8 ...",
       size=style.T_TINY + 1, color=style.FAINT, mono=True)

# ============ PANEL B: AR = predict the next token ============
BY = 330
TXT = ("#FEF3C7", "#D97706", "#78350F")
def txt(w, s):
    return dict(label=s, fill=TXT[0], stroke=TXT[1], tcolor=TXT[2], w=w)
def pim(i, **kw):
    return dict(img=P(*raster[i]), stroke="#64748B", w=34, **kw)

in_cells = [txt(40, "a"), txt(46, "red"), txt(54, "tulip"),
            dict(**pim(0), gap_after=10), pim(1, sub="p2"), pim(2, sub="p3"),
            pim(3, sub="p4"), pim(4, sub="p5")]
# wire the leading p1's sub separately (gap_after consumed the dict)
in_cells[3]["sub"] = "p1"
out_cells = [txt(46, "red"), txt(54, "tulip"),
             dict(img=P(*raster[0]), stroke="#64748B", w=34, gap_after=10),
             pim(1), pim(2), pim(3), pim(4),
             pim(5, bold_border=True, sub="p6  NEW")]

out_row = TokenRow(out_cells, cell_h=34, gap=3, id="out", label="output:", label_w=66)
d.place(out_row, 90, BY)
d.note(90, BY - 22, "OUT: the next token at every position  (input shifted by one)",
       size=style.T_SUB + 1, color="#166534", weight="600")

model = Block("Autoregressive Transformer  ·  causal self-attention",
              kind="model", sub="position i attends to tokens <= i, predicts token i+1",
              id="model", min_w=560)
d.place(model, 100, BY + 86)

in_row = TokenRow(in_cells, cell_h=34, gap=3, id="in", label="input:", label_w=66)
d.place(in_row, 90, BY + 184)
d.note(90, BY + 184 + 56,
       "IN: text prompt tokens (amber) + image patch tokens (the photo, in raster order)",
       size=style.T_SUB + 1, color=style.INK, weight="600")

ib, ob, mb = d.box("in"), d.box("out"), d.box("model")
d.edge((ib.cx, ib.y), (mb.cx, mb.y2), a_side="t", b_side="b", label="whole sequence")
d.edge((mb.cx, mb.y), (ob.cx, ob.y2), a_side="t", b_side="b")
d.note(ob.x2 + 18, ob.cy - 4, "<- predicted: append p6,", size=style.T_SUB + 0.5,
       color="#B91C1C")
d.note(ob.x2 + 18, ob.cy + 9, "   then feed back & repeat", size=style.T_SUB + 0.5,
       color="#B91C1C")

# "the image so far": reveal the first K patches, grey the rest
K = 21
gx2, gy2 = mb.x2 + 70, BY + 70
SC, SGAP = 22, 2
for idx, (r, c) in enumerate(raster):
    px, py = gx2 + c * (SC + SGAP), gy2 + r * (SC + SGAP)
    if idx < K:
        RasterImage.emit(d.doc, P(r, c), px, py, SC, SC, rx=2)
        d.doc.rect("nodes", px, py, SC, SC, "none", "#FFFFFF", 0.7, rx=2)
    else:
        d.doc.rect("nodes", px, py, SC, SC, "#F1F5F9", "#E2E8F0", 0.8, rx=2)
sw = N * (SC + SGAP) - SGAP
d.doc.rect("frames", gx2 - 3, gy2 - 3, sw + 6, sw + 6, "none", "#B91C1C", 1.2, rx=4)
d.note(gx2, gy2 - 14, "the image so far", size=style.T_SUB, color="#B91C1C")
d.note(gx2, gy2 + sw + 16, "each step fills one more patch", size=style.T_SUB,
       color=style.FAINT, max_w=160)

leg = Swatches([((TXT[0], TXT[1]), "text token"),
                (("#E2E8F0", "#64748B"), "image patch token"),
                ("model", "AR transformer"),
                (("#FFFFFF", "#B91C1C"), "predicted / new")], max_w=520, id="leg")
d.place(leg, 60, 70)

d.save(OUT / "autoregressive_flower.svg")
print("wrote", OUT / "autoregressive_flower.svg")
