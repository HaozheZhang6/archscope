"""Fig 10b — AR vs diffusion, illustrated on a concrete moving clip (a synthesized sunrise).
The companion to fig10a (structure). Two axes, made tangible:
  AR        — along TIME: generate the next frame given the past frames.
  diffusion — within the CLIP: denoise ALL frames together, noise → clean, over T steps.
Frames are a procedural CC0 'sunrise' (assets/make_motion.py); content is illustrative."""
from archscope import Diagram, RasterImage, Swatches, TextLabel, style
from common import OUT
from pathlib import Path

M = Path(__file__).resolve().parent / "assets" / "motion"

d = Diagram(
    title="Fig 10b · AR vs diffusion, illustrated — the SAME moving clip, two generation axes",
    subtitle="AUTOREGRESSIVE runs along the time axis: each frame is produced after the ones before it (predict "
             "the next frame). DIFFUSION runs within the clip: every frame starts as noise and is denoised "
             "together, noise → clean, over T steps. (Frames: a procedural sunrise — clear motion, illustrative.)")

FW = 52      # frame size
GAP = 5


def strip(x, y, tag, n=5, border=None, bw=1.2, dashed=False):
    """Draw frames f0..f{n-1}_{tag} left→right; return right edge x."""
    for i in range(n):
        fx = x + i * (FW + GAP)
        RasterImage.emit(d.doc, str(M / f"f{i}_{tag}.png"), fx, y, FW, FW, rx=4)
        col = border or "#94A3B8"
        d.doc.rect("nodes", fx, y, FW, FW, "none", col, bw, rx=4,
                   dash="3 2" if dashed else None)
    return x + n * (FW + GAP) - GAP


# ============ AR axis (top): generate the next frame ============
AY = 150
d.note(60, AY - 24, "AUTOREGRESSIVE — along the TIME axis: predict the next frame",
       size=style.T_SECTION, color="#0C4A6E", weight="700")
# frames 0..3 are already generated (solid), frame 4 is being predicted (dashed red)
x = 60
for i in range(4):
    RasterImage.emit(d.doc, str(M / f"f{i}_clean.png"), x, AY, FW, FW, rx=4)
    d.doc.rect("nodes", x, AY, FW, FW, "none", "#0284C7", 1.4, rx=4)
    d.doc.text("nodes", x + FW / 2, AY + FW + 13, f"frame {i+1}", style.T_TINY + 1,
               style.MUTED, "500")
    x += FW + GAP
# the predicted next frame
RasterImage.emit(d.doc, str(M / "f4_clean.png"), x, AY, FW, FW, rx=4)
d.doc.rect("nodes", x, AY, FW, FW, "none", "#B91C1C", 2.0, rx=4, dash="4 3")
d.doc.text("nodes", x + FW / 2, AY + FW + 13, "frame 5", style.T_TINY + 1, "#B91C1C", "600")
d.doc.text("nodes", x + FW / 2, AY + FW + 25, "predicted next", style.T_TINY, "#B91C1C", "500")
ar_x2 = x + FW
# causal arrow under the row
d.edge((60, AY + FW + 40), (ar_x2, AY + FW + 40), color="#0284C7", width=1.5,
       label="time →   each frame attends only the frames before it (causal)", label_side="right")
# the "given so far" bracket
d.note(60, AY - 8, "given so far", size=style.T_TINY + 1, color=style.FAINT)
d.note(x - 4 * (FW + GAP), AY - 8, "", size=style.T_TINY, color=style.FAINT)

# ============ diffusion axis (bottom): denoise the whole clip ============
DY = AY + FW + 120
d.note(60, DY - 24, "DIFFUSION — within the CLIP: denoise ALL frames together, noise → clean",
       size=style.T_SECTION, color="#6D28D9", weight="700")
labels = [("noise", "all frames = noise  (σ=1)"), ("half", "partly denoised  (σ≈0.5)"),
          ("clean", "clean clip  (σ=0)")]
gx = 60
ends = []
for tag, lab in labels:
    x2 = strip(gx, DY, tag, border="#7C3AED" if tag != "clean" else "#16A34A",
               bw=1.4, dashed=(tag != "clean"))
    d.doc.text("nodes", (gx + x2) / 2, DY + FW + 14, lab, style.T_TINY + 1,
               style.MUTED, "500")
    ends.append((gx, x2))
    gx = x2 + 70
# denoise arrows between the three clips
for (a_l, a_r), (b_l, b_r) in zip(ends, ends[1:]):
    d.edge((a_r + 6, DY + FW / 2), (b_l - 6, DY + FW / 2), color="#7C3AED", width=1.6,
           route="straight", label="denoise", label_bg=True)
d.note(ends[0][0], DY + FW + 30,
       "one denoiser (the DiT, fig 10a) predicts the noise/velocity for EVERY frame at once; "
       "repeat ×T. There is no left-to-right order — the whole clip refines in parallel.",
       size=style.T_SUB, color=style.MUTED, max_w=900)

# ============ the one-line contrast ============
by = DY + FW + 76
d.note(60, by, "Same clip, two axes:  AR picks the next FRAME (sequence grows in time);  "
       "diffusion picks the next NOISE LEVEL (the whole clip sharpens).  LingBot-VA does BOTH — "
       "AR across chunks, diffusion within each chunk (fig 10a).",
       size=style.T_SUB + 1, color=style.INK, max_w=940)

leg = Swatches([(("#FFFFFF", "#0284C7"), "generated frame (AR)"),
                (("#FFFFFF", "#B91C1C"), "predicted next frame", "dash"),
                (("#FFFFFF", "#7C3AED"), "noised clip (diffusion)", "dash"),
                (("#FFFFFF", "#16A34A"), "clean clip"),
                (("#7C3AED", None), "denoise (diffusion)", "edge")], max_w=900, id="leg")
d.place(leg, 60, 96)

d.save(OUT / "fig10b_illustration.svg")
print("ok")
