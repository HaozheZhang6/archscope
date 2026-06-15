"""Wan2.1-VACE-1.3B · 8 · building the VACE control signal: a 96-channel vace_context.
diffsynth pipelines/wan_video.py:664-707 (WanVideoUnit_Vace.process).

VACE conditions on a control video + a binary mask that says which region to KEEP vs which to
GENERATE. The masked/unmasked halves are VAE-encoded separately and stacked with a patchified
copy of the mask itself → 96 channels: 16 (keep) + 16 (edit) + 64 (mask).
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, HStack, Swatches,
                       TextLabel, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 8 · the control signal — vace_context = 96 channels (16 + 16 + 64)",
    subtitle="inactive = video·(1−mask) [keep region], reactive = video·mask [edit region]; each is VAE-encoded "
             "to 16 latent channels. The mask is patchified 8×8 into 64 channels and temporally matched to the "
             "latent frames. Concatenate → vace_context (96, f, h, w), the input to the VACE branch (fig 9).")

# inputs
ctrl = IOLabel("control video  (3, F, H, W)", id="ctrl", modality="none")
mask = IOLabel("binary mask  (1, F, H, W)   keep=0 / edit=1", id="mask", modality="none")
d.place(ctrl, 70, 150)
d.place(mask, 70, 210)

# the two masked halves
inact = Block("inactive = video · (1 − mask)", kind="io", sub="the KEEP region (content preserved)",
              id="inact", min_w=300)
react = Block("reactive = video · mask", kind="io", sub="the EDIT / generate region", id="react",
              min_w=300)
d.place(inact, 430, 138)
d.place(react, 430, 198)
d.edge("ctrl.r@0.5", "inact.l@0.5", a_side="r", style_name="main")
d.edge("ctrl.r@0.5", "react.l@0.5", a_side="r", style_name="main")
d.edge("mask.r@0.5", "inact.l@0.8", a_side="r", style_name="faint", color="#0D9488")
d.edge("mask.r@0.5", "react.l@0.8", a_side="r", style_name="faint", color="#0D9488")

# VAE-encode each half -> 16ch
enci = Block("VAE encode → 16", kind="vae", id="enci", min_w=150)
encr = Block("VAE encode → 16", kind="vae", id="encr", min_w=150)
d.place(enci, d.box("inact").x2 + 60, d.box("inact").y - 4)
d.place(encr, d.box("react").x2 + 60, d.box("react").y - 4)
d.edge("inact.r@0.5", "enci.l@0.5", a_side="r", style_name="main")
d.edge("react.r@0.5", "encr.l@0.5", a_side="r", style_name="main")

# the mask -> 64ch patchified
mlat = Block("patchify 8×8 + temporal match", kind="linear",
             sub="T (H·8) (W·8) → (8·8=64), interp to f", src="wan_video.py:682", id="mlat", min_w=300)
d.place(mlat, 430, 270)
d.edge("mask.r@0.5", "mlat.l@0.3", a_side="r", style_name="main", color="#0D9488")

# concat into 96
cat = OpDot("c", id="cat")   # 'c' op as a concat marker
cat2 = Block("concat channels", kind="op", sub="16 + 16 + 64 = 96", id="catb", min_w=170)
d.place(cat2, d.box("enci").x2 + 70, d.box("react").cy - 14)
d.edge("enci.r@0.5", "catb.l@0.2", a_side="r", style_name="main", label="16  keep", label_bg=True)
d.edge("encr.r@0.5", "catb.l@0.5", a_side="r", style_name="main", label="16  edit", label_bg=True)
d.edge("mlat.r@0.5", "catb.l@0.85", a_side="r", style_name="main", color="#0D9488",
       via=[(d.box("catb").x - 26, d.box("mlat").cy)], label="64  mask", label_bg=True)

out = IOLabel("vace_context  (96, f, h, w)   → fig 9", id="out", modality="state")
out.measure()
d.place(out, d.box("catb").x2 + 50, d.box("catb").cy - 11)
d.edge("catb.r@0.5", "out.l@0.5", a_side="r", style_name="main")

d.note(70, d.box("mlat").y2 + 40,
       "Channel accounting: the two VAE encodings give 16+16=32 latent channels carrying the kept "
       "and the to-be-generated pixels; the 64-channel mask tells the model, per latent patch, how "
       "much is fixed vs free. A reference image (optional) is prepended along the frame axis. "
       "(With no control, vace_context is all zeros and VACE is a no-op.)",
       size=style.T_SUB, color=style.MUTED, max_w=900)

leg = Swatches([("none", "pixels / mask"), ("io", "masked halves"), ("vae", "VAE encode"),
                ("linear", "mask patchify"), ("op", "concat"), ("state", "control latent"),
                (("#0D9488", "5 3"), "mask path", "edge")], max_w=820, id="leg")
d.place(leg, 70, 96)

d.save(OUT / "fig08_vace_context.svg")
print("ok")
