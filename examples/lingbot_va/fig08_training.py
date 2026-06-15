"""Fig 8 — how LingBot-VA is trained (code version, with paper mapping notes)."""
from archscope import (Diagram, Formula, GroupFrame, HStack, Spacer, Swatches,
                       TextLabel, TokenRow, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 8 · Training method — flow matching on an interleaved video-action sequence",
    subtitle="Everything below is one forward pass per step: teacher-forced clean history plus "
             "noisy targets, per-frame timesteps, two masked MSE losses. "
             "wan_va/train.py · wan_va/utils/scheduler.py")

P_W = 470

# ---------- P1: flow matching objective -------------------------------------------
p1 = GroupFrame(VStack([
    Formula(r"$x_\sigma=(1-\sigma)\,x_0+\sigma\,\varepsilon,\qquad v^{\ast}=\varepsilon-x_0$", size=13.5),
    TextLabel("the model predicts the velocity v(x_σ, σ); one Euler step is "
              "x ← x + v·Δσ (scheduler.py:88-113).",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60),
    Formula(r"$\mathcal{L}=\Vert v_\theta - v^{\ast}\Vert^2_{\,\mathrm{frame-weighted}}$", size=13.5),
    TextLabel("paper convention maps as s = 1 − σ (s=1 is data): "
              "x⁽ˢ⁾ = (1−s)ε + s·x₁, target x₁ − ε. Same objective, opposite sign.",
              size=style.T_SUB + 0.5, color=style.FAINT, max_w=P_W - 60),
], gap=12), title="1 · Flow-matching objective (both modalities)", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", min_w=P_W)

# ---------- P2: per-frame timesteps -------------------------------------------------
p2 = GroupFrame(VStack([
    TokenRow([
        dict(label="z₀", modality="video", hatch=True, sub="σ=.92"),
        dict(label="z₁", modality="video", hatch=True, sub="σ=.31"),
        dict(label="z₂", modality="video", hatch=True, sub="σ=.77", gap_after=12),
        dict(label="a₀", modality="action", hatch=True, sub="σ=.55"),
        dict(label="a₁", modality="action", hatch=True, sub="σ=.08"),
        dict(label="a₂", modality="action", hatch=True, sub="σ=.63"),
    ], cell_w=34),
    TextLabel("every frame draws its own σ (train.py:167-217) — diffusion-forcing style. "
              "The per-token AdaLN (Fig 3) is what makes mixed noise levels in one "
              "sequence possible.",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60),
    TextLabel("timestep sampling is SNR-shifted per modality: video shift 5.0 (biased "
              "toward high noise), action shift 1.0 (uniform); loss weighted by a "
              "mid-σ Gaussian bell (scheduler.py:56-73).",
              size=style.T_SUB + 0.5, color=style.FAINT, max_w=P_W - 60),
], gap=12), title="2 · Independent per-frame timesteps", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", min_w=P_W)

# ---------- P3: teacher forcing + noisy history -------------------------------------
p3 = GroupFrame(VStack([
    TokenRow([
        dict(label="~z", modality="video", hatch=True, sub="target"),
        dict(label="~z", modality="video", hatch=True, gap_after=14),
        dict(label="~z", modality="video", hatch=True, sub="hist .5"),
        dict(label="z", modality="video", sub="clean", gap_after=14),
        dict(label="~a", modality="action", hatch=True, sub="target"),
        dict(label="~a", modality="action", hatch=True, gap_after=14),
        dict(label="a", modality="action", sub="clean"),
        dict(label="a", modality="action"),
    ], cell_w=50),
    TextLabel("teacher forcing: ground-truth history rides along as the clean segments "
              "(Fig 2), so all chunks train in parallel under the causal mask (Fig 5).",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60),
    TextLabel("noisy-history augmentation: with p=0.5 the clean VIDEO history is re-noised "
              "with σ_aug ∈ [0.5, 1.0] (train.py:229) — so inverse dynamics learns to read "
              "half-denoised frames, and inference can stop video at σ≈0.5 (3 Euler steps). "
              "Action history is never noised (train.py:236).",
              size=style.T_SUB + 0.5, color=style.FAINT, max_w=P_W - 60),
], gap=12), title="3 · Teacher forcing + noisy-history augmentation", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", min_w=P_W)

# ---------- P4: losses ----------------------------------------------------------------
p4 = GroupFrame(VStack([
    Formula(r"$\mathcal{L}=\mathcal{L}_{video}+\mathcal{L}_{action}$", size=13.5),
    TextLabel("both are per-frame-averaged MSE on v, with the Gaussian σ-weight; the action "
              "loss is additionally masked to the 16 real dims of the padded 30-dim dual-arm "
              "action space (train.py:256-295).",
              size=style.T_SUB + 1, color=style.MUTED, max_w=P_W - 60),
    TextLabel("structural augmentation: chunk size K ~ U{1..4} and attention window ~ U{4..64} "
              "are re-rolled EVERY step, rebuilding the mask (train.py:245-247) — the model "
              "must work at any chunking/context length.",
              size=style.T_SUB + 0.5, color=style.FAINT, max_w=P_W - 60),
    TextLabel("paper adds, in post-training only, an FDM loss: re-imagine the next latent from "
              "real feedback and the executing action — used to ground asynchronous inference "
              "(paper Eq. 13).",
              size=style.T_SUB + 0.5, color=style.FAINT, max_w=P_W - 60),
], gap=12), title="4 · Losses + structural augmentation", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", min_w=P_W)

for p in (p1, p2, p3, p4):
    p.pad = 18

grid = VStack([
    HStack([p1, p2], gap=26, align="start"),
    HStack([p3, p4], gap=26, align="start"),
], gap=26)
d.place(grid, 40, 60)

d.save(OUT / "fig08_training.svg")
print("ok")
