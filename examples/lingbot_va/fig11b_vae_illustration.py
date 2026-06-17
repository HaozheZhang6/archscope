"""Fig 11b — the VAE, illustrated on the moving clip (companion to fig11a's structure).
A few frames of video are squeezed into a tiny 48-channel latent (/16 spatial, /4 temporal),
then decoded back. The latent is what the DiT actually denoises (figs 1-10). Because the VAE
is causal, lingbot encodes the stream chunk-by-chunk in real time. Frames: procedural CC0."""
from archscope import Diagram, RasterImage, Swatches, TextLabel, style
from common import OUT
from pathlib import Path

M = Path(__file__).resolve().parent / "assets" / "motion"
d = Diagram(
    title="Fig 11b · the VAE, illustrated — squeeze the clip into a tiny latent, then decode it back",
    subtitle="The pixels (3, F, H, W) are compressed to a 48-channel latent (48, F/4, H/16, W/16) — far fewer "
             "numbers — and decoded back to ≈ the same video. The DiT works entirely in this latent. The VAE is "
             "CAUSAL, so the latent is built chunk-by-chunk as frames stream in (no need for the whole clip).")

FW = 56
GAP = 6
N = 5


def strip(y, tag, label, border):
    x = 220
    for i in range(N):
        RasterImage.emit(d.doc, str(M / f"f{i}_{tag}.png"), x, y, FW, FW, rx=4)
        d.doc.rect("nodes", x, y, FW, FW, "none", border, 1.4, rx=4)
        x += FW + GAP
    d.note(220, y + FW / 2 - 6, label, size=style.T_SUB, color=style.MUTED, anchor="end")
    return x - GAP


# ---------------- input clip (top) ----------------
IY = 150
x2 = strip(IY, "clean", "input video\n(3, F, H, W)", "#0284C7")
d.note(220, IY - 18, "a few frames of video (the clip)", size=style.T_SUB + 1, color="#0C4A6E",
       weight="700")

# ---------------- latent (middle, small) ----------------
LY = IY + FW + 82
# draw the latent as a compact stack of tiny abstract tiles: F'≈2 latent frames, 48ch each
lat_x = 220 + (x2 - 220) / 2 - 70
for j in range(2):           # ~2 latent frames (5 frames /4, causal)
    bx = lat_x + j * 64
    for k in range(3):       # a few channel tiles to suggest 48 channels stacked
        off = k * 3
        d.doc.rect("nodes", bx + off, LY + off, 30, 30,
                   ["#C7D2FE", "#A5B4FC", "#818CF8"][k], "#4F46E5", 1.0, rx=3)
    d.doc.text("nodes", bx + 18, LY + 48, f"z[:, {j}]", style.T_TINY, style.MUTED)
d.note(lat_x + 64, LY - 16, "latent  z  (48, F'≈2, H/16, W/16)", size=style.T_SUB + 1,
       color="#312E81", weight="700", anchor="middle")
d.note(lat_x + 64, LY + 64, "48 channels · ~1/4 the frames · 1/16 the height & width",
       size=style.T_SUB, color=style.FAINT, anchor="middle")

# encode / decode arrows
midx = 220 + (x2 - 220) / 2
d.edge((midx, IY + FW + 6), (midx, LY - 8), color="#EA580C", width=1.8,
       label="VAE encode  →  /16 spatial · /4 temporal · 48 ch", label_side="right", label_bg=True)
OY = LY + 120
d.edge((midx, LY + 78), (midx, OY - 8), color="#EA580C", width=1.8,
       label="VAE decode  →  back to pixels", label_side="right", label_bg=True)

# ---------------- reconstructed clip (bottom) ----------------
x2b = strip(OY, "clean", "reconstructed\n≈ input", "#16A34A")
d.note(220, OY - 18, "decoded video (≈ the input)", size=style.T_SUB + 1, color="#166534",
       weight="700")

# ---------------- streaming note ----------------
d.note(220, OY + FW + 26,
       "Why it matters here: the DiT never touches pixels — it denoises this small latent (figs 1–10), which is "
       "why video diffusion is affordable. And because the VAE is CAUSAL, lingbot feeds frames in as they arrive "
       "and grows the latent chunk-by-chunk (streaming), instead of waiting for the whole clip — essential for a "
       "real-time closed-loop world model.", size=style.T_SUB, color=style.MUTED, max_w=900)

leg = Swatches([(("#FFFFFF", "#0284C7"), "input frames"),
                (("#C7D2FE", "#4F46E5"), "latent (48-ch, abstract)"),
                (("#FFFFFF", "#16A34A"), "reconstructed frames"),
                (("#EA580C", None), "VAE encode / decode", "edge")], max_w=900, id="leg")
d.place(leg, 60, 96)

d.save(OUT / "fig11b_vae_illustration.svg")
print("ok")
