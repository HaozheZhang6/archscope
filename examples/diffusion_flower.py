"""Diffusion generation, illustrated with a REAL image — the counterpart to
`autoregressive_flower.py`. Where AR fills one patch token at a time, diffusion
denoises the WHOLE image at once, over T steps.

Top: the noising trajectory x_0 (clean) -> x_1 (pure noise) on a real tulip.
Bottom: one denoising step made explicit — IN: a noised image x_t + timestep t;
OUT: the predicted noise (same shape); update x_{t-1} = x_t - step . eps; repeat.

Image: CC0 1.0 (public domain) "flower" by imuzio — see assets/flower/NOTICE.md.
Noised frames are a flow-matching interpolation x_sigma = (1-sigma) x_0 + sigma . eps.
"""
from pathlib import Path

from archscope import (Block, Diagram, Formula, RasterImage, Swatches,
                       TextLabel, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"
DIFF = Path(__file__).resolve().parent / "assets" / "flower" / "diff"
def F(i):
    return str(DIFF / f"x_{i}.png")

d = Diagram(
    title="Diffusion generation, illustrated — the whole image is denoised at once",
    subtitle="A diffusion model turns pure noise into the image by removing a little noise at a time. "
             "Forward (training): add noise to a real image. Reverse (generation): a network predicts the "
             "noise in a noised image and subtracts a step of it, repeating T times. Contrast the previous "
             "figure: autoregressive fills ONE patch token at a time; diffusion refines ALL pixels together.")

# ============ TOP: the noising / denoising trajectory ============
TY = 150
IM = 92
GAP = 26
sig = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
xs = []
for i in range(6):
    x = 80 + i * (IM + GAP)
    xs.append(x)
    edge_col = "#16A34A" if i == 0 else ("#B91C1C" if i == 5 else "#CBD5E1")
    d.place(RasterImage(F(i), IM, id=f"f{i}", rx=6, outline=edge_col), x, TY)
    d.note(x + IM / 2, TY + IM + 16, f"sigma={sig[i]}", size=style.T_SUB, color=style.MUTED,
           anchor="middle", mono=True)
d.note(xs[0], TY - 14, "x_0  (clean image)", size=style.T_SUB, color="#166534")
d.note(xs[5] + IM, TY - 14, "x_1  (pure noise)", size=style.T_SUB, color="#B91C1C",
       anchor="end")

# forward arrow (add noise) above the strip, reverse (denoise) below
fb, lb = d.box("f0"), d.box("f5")
d.edge((fb.x, TY - 30), (lb.x2, TY - 30), color="#64748B", width=1.4,
       label="forward: add noise   x_0 -> x_1   (defines training)", label_side="right")
d.edge((lb.x2, TY + IM + 34), (fb.x, TY + IM + 34), color="#16A34A", width=1.6,
       label="reverse: denoise = GENERATE   x_1 -> x_0   (T steps)", label_side="right")

# ============ BOTTOM: one denoising step, prediction made explicit ============
BY = TY + IM + 110
xt = RasterImage(F(4), 96, id="xt", rx=6, outline="#B91C1C")
d.place(xt, 110, BY)
d.note(d.box("xt").cx, BY - 14, "IN: noised x_t", size=style.T_SUB + 0.5, color=style.INK,
       anchor="middle", weight="600")
d.note(d.box("xt").cx, d.box("xt").y2 + 15, "+ timestep t", size=style.T_SUB,
       color=style.MUTED, anchor="middle")

model = Block("denoiser  e_theta  (DiT / U-Net)", kind="model",
              sub="same network at every step · t drives AdaLN", id="model", min_w=200, h=96)
d.place(model, d.box("xt").x2 + 56, BY)
d.edge("xt", "model", label="x_t + t")

eps = RasterImage(str(DIFF / "noise.png"), 96, id="eps", rx=6, outline="#7C3AED")
d.place(eps, d.box("model").x2 + 56, BY)
d.edge("model", "eps")
d.note(d.box("eps").cx, BY - 14, "OUT: predicted noise  e_theta", size=style.T_SUB + 0.5,
       color="#6D28D9", anchor="middle", weight="600")
d.note(d.box("eps").cx, d.box("eps").y2 + 15, "same shape as x_t", size=style.T_SUB,
       color="#6D28D9", anchor="middle")

xtm = RasterImage(F(3), 96, id="xtm", rx=6, outline="#16A34A")
d.place(xtm, d.box("eps").x2 + 70, BY)
d.edge("eps", "xtm", label="subtract a step")
d.note(d.box("xtm").cx, BY - 14, "x_{t-1}  (a bit cleaner)", size=style.T_SUB + 0.5,
       color="#166534", anchor="middle", weight="600")
upd = Formula(r"$x_{t-1}=x_t-\mathrm{step}\cdot e_\theta(x_t,t)$", size=12.5)
upd.measure()
d.place(upd, d.box("xtm").cx - upd.w / 2, d.box("xtm").y2 + 14)

# the loop: x_{t-1} feeds back as the next x_t, x T
mb, xb = d.box("model"), d.box("xt")
ly = BY + 96 + 50
d.edge((d.box("xtm").cx, d.box("xtm").y2 + 30), (xb.cx, xb.y2),
       a_side="b", b_side="b", via=[(d.box("xtm").cx, ly), (xb.cx, ly)],
       color="#16A34A", width=1.5, label="x T:  feed x_{t-1} back as the next x_t")

# contrast with AR
d.note(110, ly + 40,
       "vs autoregressive (previous figure):  AR predicts the NEXT token given past tokens "
       "(one patch at a time, in order);", size=style.T_SUB + 0.5, color=style.MUTED, max_w=820)
d.note(110, ly + 55,
       "diffusion predicts the NOISE in the whole image (all pixels at once) and peels it off over "
       "T steps. Both are trained by regression to a known target (next token / the added noise).",
       size=style.T_SUB + 0.5, color=style.MUTED, max_w=820)

leg = Swatches([(("#FFFFFF", "#16A34A"), "clean image x_0 / cleaner"),
                (("#FFFFFF", "#B91C1C"), "noised x_t / pure noise"),
                (("#FFFFFF", "#7C3AED"), "predicted noise"),
                ("model", "denoiser network")], max_w=620, id="leg")
d.place(leg, 80, 80)

d.save(OUT / "diffusion_flower.svg")
print("wrote", OUT / "diffusion_flower.svg")
