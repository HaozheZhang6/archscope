"""Flow matching (rectified flow) as a training-method storyboard.

The objective behind SD3 / FLUX / Wan — drawn as: the noising path, the training
step, the sampling loop, and how the diffusion parameterizations relate.
"""
from pathlib import Path

from archscope import (Block, Diagram, Formula, GroupFrame, HStack, IOLabel,
                       OpDot, Swatches, TextLabel, TokenRow, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Flow matching (rectified flow) — train & sample",
    subtitle="One straight path between data and noise; the network learns its "
             "constant velocity. The recipe behind SD3, FLUX and Wan.")

P_W = 460

# ---------------- P1: the path -------------------------------------------------------
p1 = GroupFrame(VStack([
    TokenRow([
        dict(label="x₀", modality="video", w=44, sub="data"),
        dict(label="x.₂₅", modality="video", hatch=True, w=44, sub="σ=.25"),
        dict(label="x.₅", modality="video", hatch=True, w=44, sub="σ=.5"),
        dict(label="x.₇₅", modality="none", hatch=True, w=44, sub="σ=.75"),
        dict(label="ε", modality="none", w=44, sub="noise"),
    ], cell_h=26),
    Formula(r"$x_\sigma=(1-\sigma)\,x_0+\sigma\,\varepsilon$", size=13.5),
    TextLabel("linear interpolation — so the true velocity is CONSTANT along the "
              "path and independent of σ:", size=style.T_SUB + 1,
              color=style.MUTED, max_w=P_W - 60, anchor="start"),
    Formula(r"$v^{\ast}=\frac{dx_\sigma}{d\sigma}=\varepsilon-x_0$", size=13.5),
], gap=12), title="1 · one straight path, data → noise", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", pad=18)

# ---------------- P2: training -------------------------------------------------------
p2 = GroupFrame(VStack([
    TextLabel("per step: draw x₀ from data, ε from N(0,I), σ from a schedule —",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60, anchor="start"),
    Formula(r"$\mathcal{L}=\mathbb{E}_{\,x_0,\varepsilon,\sigma}\;"
            r"\Vert v_\theta(x_\sigma,\sigma)-(\varepsilon-x_0)\Vert^2$", size=13.5),
    TextLabel("σ sampling is the main tuning knob: uniform is the baseline; "
              "logit-normal (SD3) concentrates training where the task is hardest "
              "— mid noise. The σ also conditions the net (AdaLN / time embedding).",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60, anchor="start"),
    TextLabel("relation to diffusion: DDPM's ε-prediction, x₀-prediction and "
              "v-prediction are the same regression with different output "
              "parameterizations and σ-weightings — rectified flow picks the "
              "straight path + velocity target.",
              size=style.T_SUB + 0.5, color=style.FAINT, max_w=P_W - 60, anchor="start"),
], gap=12), title="2 · training = regress the velocity", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", pad=18)

# ---------------- P3: sampling -------------------------------------------------------
steps = HStack([
    IOLabel("ε", modality="none", id="s0"),
    Block("v_θ(x, σ)", kind="model", id="net", min_w=110),
    OpDot("+", id="step"),
    IOLabel("x₀ (generated)", modality="video", id="s1"),
], gap=34)
p3 = GroupFrame(VStack([
    steps,
    Formula(r"$x \leftarrow x + v_\theta(x,\sigma)\,\Delta\sigma,"
            r"\qquad \sigma: 1 \rightarrow 0$", size=13),
    TextLabel("Euler steps along the learned field. Because the target path is "
              "straight, few steps go far (4-30 typical; distilled models: 1-4). "
              "Same loop powers image, video and action-chunk generation.",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60, anchor="start"),
], gap=14), title="3 · sampling = integrate it back", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", pad=18)

row = VStack([HStack([p1, p2], gap=26, align="start"),
              HStack([p3], gap=26, align="start")], gap=26)
d.place(row, 40, 56)

d.edge("s0", "net", label="σ=1")
d.edge("net", "step", label="v·Δσ")
d.edge("step", "s1", label="… ×N steps", label_at=0.5)

leg = Swatches([("video", "data"), (("#F1F5F9", "#94A3B8"), "noised (hatched)", "hatch"),
                ("model", "network")], max_w=420, id="leg")
d.place(leg, 40, 20)

d.save(OUT / "flow_matching.svg")
print("wrote", OUT / "flow_matching.svg")
