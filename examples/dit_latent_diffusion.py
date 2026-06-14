"""DiT — a latent diffusion transformer, drawn so the PREDICTION is unmistakable.

The lesson this example teaches: a diffusion network is a function. Show its inputs
(a noised latent at level t, the timestep, the condition) and its output (the predicted
noise / velocity) with the SAME tensor shape in and out — that is what "what does it
predict" means. Structure after Peebles & Xie, "Scalable Diffusion Models with
Transformers" (DiT), with the signature adaLN-Zero block detail on the right.
"""
from pathlib import Path

from archscope import (Block, Diagram, Formula, GroupFrame, HStack, IOLabel,
                       OpDot, RepeatStack, Swatches, TextLabel, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="DiT — what a diffusion transformer actually predicts",
    subtitle="A denoiser is a function f(x_t, t, c). IN: a noised latent x_t at level t + the "
             "timestep + the condition. OUT: the predicted noise (same shape as x_t). The "
             "sampler subtracts a bit of it and repeats; training regresses it to the true noise.")

# ============ LEFT: the in -> network -> out pipeline ============
X = 150

pipe = VStack([
    # OUTPUT (top): same shape as the input latent
    IOLabel("predicted noise  e_theta  (32, 32, 4)", id="out", modality="video"),
    Block("Linear & Reshape", kind="head", sub="tokens (T, d) -> (32, 32, 4)", id="head"),
    Block("Layer Norm", kind="norm", id="lnf"),
    RepeatStack(Block("DiT block", kind="model",
                      sub="adaLN-Zero self-attn + MLP  (detail ->)", id="blk", min_w=210),
                times="×N", id="dit"),
    Block("Patchify -> tokens (T, d)", kind="linear", sub="T = (32/p)^2 patches", id="patch"),
    IOLabel("noised latent  x_t  (32, 32, 4)", id="xt", modality="video", hatched=True),
], gap=22)
d.place(pipe, X, 110)
d.chain(["xt", "patch", "dit", "lnf", "head", "out"])

# the two side inputs that make it a *conditional denoiser*
tstep = Block("timestep  t", kind="cond", sub="sinusoidal -> MLP", id="t")
cond = Block("condition  c", kind="cond", sub="class / text / pose", id="c")
ins = HStack([tstep, cond], gap=24)
ins.measure()
d.place(ins, d.box("patch").cx - ins.w / 2, d.box("xt").y + 120)
# they don't flow into patchify; they drive every block's modulation
d.edge("t", "dit.l@0.62", style_name="cond", b_side="b")
d.edge("c", "dit.l@0.38", style_name="cond", b_side="b")
d.note(d.box("t").x, d.box("t").y2 + 16,
       "t + c are SUMMED, drive adaLN in every block (they are not tokens)",
       size=style.T_SUB, color=style.FAINT, max_w=260)

# the prediction, stated as a formula beside the network
d.note(d.box("out").x2 + 30, d.box("out").cy - 4,
       "same shape in -> out", size=style.T_SUB + 1, color="#166534")
loss = Formula(r"$\mathcal{L}=\mathbb{E}_{x_0,\epsilon,t}\,\Vert \epsilon_\theta(x_t,t,c)-\epsilon\Vert^2$",
               size=13)
loss.measure()
d.place(loss, d.box("head").x2 + 30, d.box("head").cy - loss.h / 2)
d.note(d.box("dit").x2 + 30, d.box("dit").cy - 6,
       "x_t = sqrt(a_t) x_0 + sqrt(1-a_t) eps", size=style.T_SUB, color=style.MUTED,
       mono=True)
d.note(d.box("dit").x2 + 30, d.box("dit").cy + 8,
       "the net sees x_t, must recover eps", size=style.T_SUB, color=style.FAINT)

# the sampling loop: subtract predicted noise, repeat
nb, hb = d.box("xt"), d.box("out")
xr = nb.x - 40
d.edge((hb.x, hb.cy), (nb.x, nb.cy), a_side="l", b_side="l",
       via=[(xr, hb.cy), (xr, nb.cy)], color="#16A34A", width=1.5,
       label="x T  (x_t -> x_{t-1})", label_side="left")

# ============ RIGHT: the adaLN-Zero block detail ============
BX = d.box("out").x2 + 230
blk = VStack([
    OpDot("+", id="b_add2"),
    OpDot("o", id="b_g2"),
    Block("Pointwise Feedforward", kind="ffn", id="b_ff"),
    Block("scale, shift  (gamma2, beta2)", kind="cond", id="b_ss2", text_size=style.T_SUB + 1),
    Block("Layer Norm", kind="norm", id="b_ln2"),
    OpDot("+", id="b_add1"),
    OpDot("o", id="b_g1"),
    Block("Multi-Head Self-Attention", kind="attention", id="b_attn"),
    Block("scale, shift  (gamma1, beta1)", kind="cond", id="b_ss1", text_size=style.T_SUB + 1),
    Block("Layer Norm", kind="norm", id="b_ln1"),
    IOLabel("input tokens", id="b_in"),
], gap=13)
detail = GroupFrame(blk, title="DiT block · adaLN-Zero", title_pos="tag", dashed=False,
                    stroke="#94A3B8", tint="rgba(148,163,184,0.05)", id="detail", pad=18)
d.place(detail, BX, 96)
d.chain(["b_in", "b_ln1", "b_ss1", "b_attn", "b_g1", "b_add1", "b_ln2", "b_ss2",
         "b_ff", "b_g2", "b_add2"])

# the conditioning MLP rail: t+c -> 6 params, tapping into the bands/gates
mlp = Block("MLP", kind="cond", sub="t + c -> 6 params", id="b_mlp")
mlp.measure()
db = d.box("detail")
d.place(mlp, db.x2 + 36, d.box("b_attn").cy - mlp.h / 2)
rail = db.x2 + 18
ys = [d.box(i).cy for i in ("b_ss1", "b_g1", "b_ss2", "b_g2")]
d.edge("b_mlp.l", (rail, d.box("b_mlp").cy), a_side="l", arrow=False, style_name="cond")
d.doc.line("edges", rail, min(ys), rail, max(ys), "#DB2777", 1.2, dash="5 3")
for i, lab in [("b_ss1", "g1,b1"), ("b_g1", "a1 (gate)"), ("b_ss2", "g2,b2"),
               ("b_g2", "a2 (gate)")]:
    bb = d.box(i)
    d.edge((rail, bb.cy), (bb.x2, bb.cy), style_name="cond", route="straight")
d.note(d.box("b_mlp").x, d.box("b_mlp").y2 + 14,
       "adaLN-Zero: scale/shift AFTER each norm, gate BEFORE each residual; "
       "gates init 0 -> block starts as identity.", size=style.T_SUB, color=style.FAINT,
       max_w=250)

leg = Swatches([("video", "latent / output"), ("model", "DiT"), ("attention", "attention"),
                ("ffn", "FFN"), ("norm", "norm"), ("cond", "conditioning / modulation"),
                ("head", "output head"),
                (("#E0F2FE", "#0284C7"), "hatched = noised", "hatch")], max_w=620, id="leg")
d.place(leg, X - 40, max(d.box("t").y2, d.box("c").y2) + 44)

d.save(OUT / "dit_latent_diffusion.svg")
print("wrote", OUT / "dit_latent_diffusion.svg")
