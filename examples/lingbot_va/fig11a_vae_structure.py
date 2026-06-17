"""Fig 11a — lingbot's video VAE (Wan2.2), the STRUCTURE. A 3D CAUSAL conv VAE that
compresses a video to a 48-channel latent: /16 spatial, /4 temporal. Companion to fig11b's
illustration. Grounded in the Wan VAE family (diffsynth wan_video_vae.py: Encoder3d /
Decoder3d / CausalConv3d / ResidualBlock / Resample); lingbot config z_dim=48, /16 spatial."""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, Swatches, TextLabel, style)
from common import OUT

d = Diagram(
    title="Fig 11a · lingbot's VAE (Wan2.2) — a 3D CAUSAL conv VAE: video ↔ 48-ch latent (/16 spatial, /4 temporal)",
    subtitle="Encoder: CausalConv3d stem → 4 down-stages (2 residual blocks each, spatial /2 per stage → /16; two "
             "stages also halve time → /4) → mid (resblock+attn+resblock) → head → (mu, log_var). Sample z, "
             "normalize by per-channel mean/std. Decoder mirrors it. CAUSAL in time ⇒ encode chunk-by-chunk (fig 11b).")

# ---------------- ENCODER (left, bottom -> top) -----------------------------------
EX = 70
enc = VStack([
    Block("head:  RMSNorm · SiLU · CausalConv3d → 2·48", kind="vae", sub="→ mu, log_var", id="ehead", min_w=300),
    Block("middle:  ResBlock · self-Attention · ResBlock", kind="attention", id="emid", min_w=300),
    Block("down-stage 4:  ResBlock×2 · downsample2d (spatial /2)", kind="vae", sub="C=512", id="e4", min_w=300),
    Block("down-stage 3:  ResBlock×2 · downsample2d (spatial /2)", kind="vae", sub="C=512", id="e3", min_w=300),
    Block("down-stage 2:  ResBlock×2 · downsample3d (space+TIME /2)", kind="vae", sub="C=256", id="e2", min_w=300),
    Block("down-stage 1:  ResBlock×2 · downsample3d (space+TIME /2)", kind="vae", sub="C=128", id="e1", min_w=300),
    Block("stem:  CausalConv3d(3 → 128, k=3)", kind="vae", id="estem", min_w=300),
    IOLabel("input video  (3, F, H, W)", id="vin", modality="none"),
], gap=12)
d.place(enc, EX, 150)
d.chain(["vin", "estem", "e1", "e2", "e3", "e4", "emid", "ehead"])
d.note(EX, 136, "ENCODER", size=style.T_SUB + 1, color="#7C2D12", weight="700")

# ---------------- LATENT (top centre) ---------------------------------------------
samp = OpDot("o", id="samp")     # reparameterize / sample
d.place(samp, d.box("ehead").x2 + 70, d.box("ehead").cy - 9)
norm = Block("normalize  (z − mean)/std", kind="linear", sub="per-channel · vae.config", id="norm", min_w=180)
d.place(norm, d.box("samp").x2 + 30, d.box("ehead").cy - 16)
lat = IOLabel("latent  z  (48, F', H/16, W/16)   F'=(F−1)/4+1", id="z", modality="video")
lat.measure()
d.place(lat, d.box("norm").x2 + 36, d.box("norm").cy - 11)
d.edge("ehead.r@0.5", "samp.l@0.5", a_side="r", b_side="l", label="mu, log_var", label_bg=True)
d.edge("samp.r@0.5", "norm.l@0.5", a_side="r", b_side="l", label="z", label_bg=True)
d.edge("norm.r@0.5", "z.l@0.5", a_side="r", b_side="l")

# ---------------- DECODER (right, top -> bottom) ----------------------------------
DX = d.box("z").x2 + 60
dec = VStack([
    Block("in:  Conv3d(48 → 512)", kind="vae", id="din", min_w=300),
    Block("middle:  ResBlock · self-Attention · ResBlock", kind="attention", id="dmid", min_w=300),
    Block("up-stage 1:  ResBlock×2 · upsample (space+TIME ×2)", kind="vae", sub="C=512", id="u1", min_w=300),
    Block("up-stage 2:  ResBlock×2 · upsample (space+TIME ×2)", kind="vae", sub="C=256", id="u2", min_w=300),
    Block("up-stage 3:  ResBlock×2 · upsample2d (spatial ×2)", kind="vae", sub="C=128", id="u3", min_w=300),
    Block("up-stage 4:  ResBlock×2 · upsample2d (spatial ×2)", kind="vae", sub="C=128", id="u4", min_w=300),
    Block("head:  RMSNorm · SiLU · CausalConv3d → 3", kind="vae", id="dhead", min_w=300),
    IOLabel("reconstructed video  (3, F, H, W)", id="vout", modality="none"),
], gap=12)
d.place(dec, DX, 150)
d.chain(["din", "dmid", "u1", "u2", "u3", "u4", "dhead", "vout"])
d.note(DX, 136, "DECODER  (mirror)", size=style.T_SUB + 1, color="#7C2D12", weight="700")
d.edge("z.r@0.5", "din.t@0.5", a_side="r", b_side="t",
       via=[(d.box("din").cx, d.box("z").cy)], label="z", label_bg=True)

d.note(EX, d.box("vin").y2 + 22,
       "CAUSAL: every CausalConv3d pads only the PAST in time, so latent frame f depends only on video "
       "frames ≤ its time — that is why F frames → (F−1)/4+1 latent frames, and why lingbot can encode the "
       "stream CHUNK BY CHUNK in real time (feat_cache), instead of needing the whole clip (fig 11b).",
       size=style.T_SUB, color=style.MUTED, max_w=620)
d.note(DX, d.box("vout").y2 + 22,
       "Compression (lingbot / Wan2.2): (3, F, H, W) → (48, F/4, H/16, W/16) ≈ a few-fold fewer numbers, so "
       "the DiT works in this compact latent. Wan2.1-VACE's VAE is the same family but 16-ch, /8 spatial.",
       size=style.T_SUB, color=style.FAINT, max_w=520)

leg = Swatches([("vae", "conv / VAE stage"), ("attention", "self-attention"), ("linear", "normalize"),
                ("video", "latent"), ("none", "pixels"),
                ("op", "sample (reparameterize)", "glyph:o")], max_w=900, id="leg")
d.place(leg, EX, 96)

d.save(OUT / "fig11a_vae_structure.svg")
print("ok")
