"""What does a diffusion denoiser actually predict? — epsilon, x_0, or v.

The relationship is the whole story: at noise level sigma, the noised image
    x_sigma = (1 - sigma) * x_0 + sigma * eps
lies on the straight line between the clean image x_0 and the noise eps. The
network can be trained to output ANY of three quantities; given (x_sigma, sigma)
each one determines the other two — they are equivalent reparameterizations that
differ only in the loss weighting and in what is easy to learn at high vs low noise.

Uses the same CC0 tulip (assets/flower/NOTICE.md). Flow-matching convention
(x_sigma linear in sigma), as used by SD3 / Wan / the cad_video_gen model.
"""
from pathlib import Path

from archscope import (Diagram, Formula, GroupFrame, RasterImage, Swatches,
                       TextLabel, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"
DIFF = Path(__file__).resolve().parent / "assets" / "flower" / "diff"

d = Diagram(
    title="What does the denoiser predict? — ε, x₀, or v (three equivalent targets)",
    subtitle="At noise level σ, the noised image is a blend of the clean image x₀ and the noise ε. "
             "The network can output the noise (ε-pred), the clean image (x₀-pred), or the velocity "
             "v = ε − x₀ (v-pred). Given the noised input and σ, each determines the other two — so they are "
             "interchangeable; they differ in loss weighting and stability across noise levels.")

# ============ the relationship: x_0 --- x_sigma --- eps on one line ============
AX, AY, IM = 150, 150, 96
x0 = RasterImage(str(DIFF / "x_0.png"), IM, id="x0", rx=6, outline="#16A34A")
xs = RasterImage(str(DIFF / "x_2.png"), IM, id="xs", rx=6, outline="#334155", ow=2.0)
ep = RasterImage(str(DIFF / "noise.png"), IM, id="ep", rx=6, outline="#7C3AED")
d.place(x0, AX, AY)
d.place(xs, AX + 300, AY)
d.place(ep, AX + 620, AY)
d.note(d.box("x0").cx, AY - 14, "x₀  (clean image)", size=style.T_SUB + 0.5,
       color="#166534", anchor="middle", weight="600")
d.note(d.box("xs").cx, AY - 14, "the network's input  (σ = 0.4)", size=style.T_SUB + 0.5,
       color=style.INK, anchor="middle", weight="600")
d.note(d.box("ep").cx, AY - 14, "ε  (pure noise)", size=style.T_SUB + 0.5,
       color="#6D28D9", anchor="middle", weight="600")

# the interpolation line through the three thumbnails
ly = AY + IM / 2
d.doc.line("edges", d.box("x0").x2, ly, d.box("ep").x, ly, "#94A3B8", 1.4)
interp = Formula(r"$x_\sigma=(1-\sigma)\,x_0+\sigma\,\epsilon\qquad(\sigma=0.4)$", size=12.5)
interp.measure()
d.place(interp, d.box("xs").cx - interp.w / 2, AY + IM + 12)
d.note(d.box("xs").cx, AY + IM + 40,
       "small σ → mostly clean,   large σ → mostly noise",
       size=style.T_SUB, color=style.FAINT, anchor="middle")

# the three prediction targets, as arrows from x_sigma
xb = d.box("xs")
# x_0-pred: arrow left toward the clean image
d.edge((xb.x, ly), (d.box("x0").x2, ly), a_side="l", b_side="r", route="straight",
       color="#16A34A", width=1.8, label="x₀-pred: output the clean image",
       label_side="left", label_at=0.5, label_bg=True)
# eps-pred: arrow right toward the noise
d.edge((xb.x2, ly), (d.box("ep").x, ly), a_side="r", b_side="l", route="straight",
       color="#7C3AED", width=1.8, label="ε-pred: output the noise", label_at=0.5,
       label_bg=True)
# v-pred: the velocity vector v = ε − x₀ (direction of the line, data → noise)
vy = AY + IM + 64
d.edge((d.box("x0").cx, vy), (d.box("ep").cx, vy), color="#D97706", width=2.0,
       label="v-pred: the velocity  v = ε − x₀  (line direction, data → noise)",
       label_side="right", label_at=0.5, label_bg=True)

# ============ three cards: what each outputs + how to recover the rest ============
CY = vy + 70
def card(title, color, out_line, loss, recover_tex, note):
    return GroupFrame(VStack([
        TextLabel(title, size=style.T_SECTION, color=color, weight="700", anchor="start"),
        TextLabel(out_line, size=style.T_SUB + 1, color=style.MUTED, anchor="start", max_w=300),
        Formula(loss, size=12.5),
        Formula(recover_tex, size=11),
        TextLabel(note, size=style.T_SUB + 0.5, color=style.FAINT, anchor="start", max_w=300),
    ], gap=9, align="start"), dashed=False, stroke=color,
        tint="rgba(148,163,184,0.04)", pad=16)

cards = [
    card("ε-pred  (DDPM)", "#7C3AED", "the network outputs the noise  ε",
         r"$\mathcal{L}=\Vert \hat{\epsilon}-\epsilon\Vert^2$",
         r"$x_0=(x_\sigma-\sigma\,\hat{\epsilon})\,/\,(1-\sigma)$",
         "weak signal near σ=1 (target ~ pure noise)"),
    card("x₀-pred", "#16A34A", "the network outputs the clean image  x₀",
         r"$\mathcal{L}=\Vert \hat{x}_0-x_0\Vert^2$",
         r"$\epsilon=(x_\sigma-(1-\sigma)\,\hat{x}_0)\,/\,\sigma$",
         "weak signal near σ=0 (target ~ the input)"),
    card("v-pred / flow", "#D97706", "the network outputs the velocity  v = ε − x₀",
         r"$\mathcal{L}=\Vert \hat{v}-(\epsilon-x_0)\Vert^2$",
         r"$x_0=x_\sigma-\sigma\hat{v},\quad \epsilon=x_\sigma+(1-\sigma)\hat{v}$",
         "balanced across all σ → SD3 / Wan / DF"),
]
cx = AX
for i, c in enumerate(cards):
    c.measure()
    d.place(c, cx, CY)
    cx += c.w + 26

# the unifying takeaway
d.note(AX, CY + 165,
       "Same x_sigma, same network — only the OUTPUT TARGET (and its loss weight) changes. "
       "Convert freely with the formulas above. v-pred is the modern default because its target "
       "stays informative at every noise level.", size=style.T_SUB + 1, color=style.INK,
       max_w=900)

leg = Swatches([(("#FFFFFF", "#16A34A"), "x₀ (clean)"),
                (("#FFFFFF", "#7C3AED"), "ε (noise)"),
                (("#FFFFFF", "#D97706"), "v (velocity)"),
                (("#FFFFFF", "#334155"), "noised input")], max_w=620, id="leg")
d.place(leg, AX, 80)

d.save(OUT / "prediction_targets.svg")
print("wrote", OUT / "prediction_targets.svg")
