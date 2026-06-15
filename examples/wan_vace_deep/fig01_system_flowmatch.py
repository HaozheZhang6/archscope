"""Wan2.1-VACE-1.3B · 1 · the denoiser as a function f(x_σ, σ, text, control), wrapped in
a flow-matching sampling loop. This is the map; later figures open each box.

If you know flow matching: the network regresses the velocity v = ε − x_0 of the straight
path x_σ = (1−σ)x_0 + σε. Sampling integrates that velocity from noise (σ=1) to data (σ=0).
Grounded in diffsynth WanModel.forward + the Wan2.1 flow-matching scheduler.
"""
from archscope import (Block, Diagram, Formula, IOLabel, VStack, Swatches, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 1 · the model is a velocity field; sampling is a flow-matching loop",
    subtitle="The DiT is a function v = f(x_σ, σ, text, control): IN a noised latent x_σ at level σ, OUT a "
             "velocity of the SAME shape. Training regresses v to ε − x_0 on the straight path "
             "x_σ = (1−σ)x_0 + σε. Sampling starts at noise (σ=1) and Euler-integrates v down to σ=0; the VAE "
             "maps pixels ↔ latents at the two ends.")

CX = 480   # spine centre

# ---------------- the denoiser spine (centre, bottom -> top) ----------------------
spine = VStack([
    IOLabel("predicted velocity  v   (16, F', H/16, W/16)", id="v", modality="video"),
    Block("WanModel  ·  the DiT denoiser", kind="model", min_w=370,
          sub="patchify → 30 DiT blocks (+ VACE hints) → head → unpatchify   (figs 2–11)", id="dit"),
    IOLabel("noised latent  x_σ   (16, F', H/16, W/16)", id="xs", modality="video", hatched=True),
], gap=30)
d.place(spine, CX - 185, 200)
d.chain(["xs", "dit", "v"])

# ---------------- left: the three conditions feeding the DiT ----------------------
cond = VStack([
    Block("text prompt → UMT5 enc", kind="cond", sub="(B, 512, 4096) → 1536   (fig 3)",
          id="txt", modality="text", min_w=215),
    Block("noise level σ → timestep", kind="cond", sub="256 → 1536 → 6·1536   (figs 3, 7)",
          id="t", min_w=215),
    Block("control video + mask → VACE", kind="trainable", sub="→ 15 hints   (figs 8–11)",
          id="ctrl", min_w=215),
], gap=26, align="end")
d.place(cond, 40, d.box("dit").cy - 78)
d.edge("txt", "dit.l@0.22", style_name="cond", color="#DB2777")
d.edge("t", "dit.l@0.5", style_name="cond")
d.edge("ctrl", "dit.l@0.8", style_name="cache", color="#B45309", label="× scale", label_bg=True)

# ---------------- right: the flow-matching sampling loop --------------------------
# v drives an Euler step that updates the running latent, which re-enters as the next x_σ
step = Block("Euler step  ×T", kind="op", sub="x ← x − (σ_t − σ_{t-1}) · v\nσ : 1 → 0   (~25 steps)",
             id="step", min_w=210)
step.measure()
d.place(step, d.box("dit").x2 + 95, d.box("dit").cy - step.h / 2)
ch = (d.box("dit").x2 + d.box("step").x) / 2     # clear channel between DiT and sampler
# v drives the step (enters its LEFT-upper); the updated latent loops back to x_σ (LEFT-lower)
d.edge("v.r@0.5", "step.l@0.3", a_side="r", b_side="l",
       via=[(ch + 12, d.box("v").cy), (ch + 12, d.box("step").y + 0.3 * step.h)],
       style_name="main", label="v", label_bg=True)
d.edge("step.l@0.75", "xs.r@0.5", a_side="l", b_side="r",
       via=[(ch - 12, d.box("step").y + 0.75 * step.h), (ch - 12, d.box("xs").cy)],
       style_name="cache", color="#0D9488", label="next x_σ", label_side="left", label_bg=True)

# noise in (start, top) and clean out (end, bottom) of the loop
noise = IOLabel("start: x_1 ~ N(0, I)", id="x1", modality="video", hatched=True)
d.place(noise, d.box("step").x2 - 30, d.box("step").y - 52)
d.edge("x1.b@0.5", "step.t@0.6", b_side="t", style_name="faint")
x0 = IOLabel("end (σ=0): clean latent x_0", id="x0", modality="video")
d.place(x0, d.box("step").x, d.box("step").y2 + 40)
d.edge("step.b@0.6", "x0.t@0.5", b_side="t", style_name="faint", label="when σ→0", label_bg=True)

# ---------------- flow-matching definition (top-left band) ------------------------
fm = VStack([
    Formula(r"$x_\sigma=(1-\sigma)\,x_0+\sigma\,\varepsilon$", size=13),
    Formula(r"$v^{*}=\varepsilon-x_0,\qquad \mathcal{L}=\Vert v-v^{*}\Vert^2$", size=13),
], gap=9, align="start")
d.place(fm, 40, 150)
d.note(40, 136, "the straight path & training target:", size=style.T_SUB, color=style.INK,
       weight="600")

# ---------------- VAE bookends (bottom) -------------------------------------------
enc = Block("Wan2.1 VAE encode", kind="vae", sub="video → 16-ch latent · 8× spatial, 4× temporal",
            id="enc", min_w=260)
enc.measure()
d.place(enc, 40, d.box("ctrl").y2 + 56)
d.edge("enc.t@0.5", "ctrl.b@0.5", b_side="b", style_name="faint", label="x_0 (training)", label_bg=True)
dec = Block("Wan2.1 VAE decode", kind="vae", sub="16-ch latent → video", id="dec", min_w=230)
dec.measure()
d.place(dec, d.box("x0").x, d.box("x0").y2 + 30)
d.edge("x0.b@0.5", "dec.t@0.5", b_side="t", style_name="main")
d.note(d.box("dec").x, d.box("dec").y2 + 14, "→ generated video  (3, F, H, W)",
       size=style.T_SUB, color=style.MUTED)
d.note(d.box("enc").x, d.box("enc").y2 + 14,
       "Wan2.1 VAE: a video (3, F, H, W) becomes a latent (16, F'=(F−1)/4+1, H/8, W/8).",
       size=style.T_SUB, color=style.FAINT, max_w=300)

leg = Swatches([("video", "latent (16-ch)"), ("model", "DiT denoiser"),
                ("cond", "conditioning"), ("trainable", "VACE branch"), ("vae", "VAE"),
                ("op", "sampler"),
                (("#E0F2FE", "#0284C7"), "hatched = noised", "hatch"),
                ("cache", "sampling loop", "edge")], max_w=900, id="leg")
d.place(leg, 40, 96)

d.save(OUT / "fig01_system_flowmatch.svg")
print("ok")
