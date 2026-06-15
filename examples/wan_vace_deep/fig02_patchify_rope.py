"""Wan2.1-VACE-1.3B · 2 · from latent to a token sequence: Conv3d patchify + 3D RoPE.
diffsynth wan_video_dit.py:382-383 (patch_embedding), :492-501 (patchify), :77-100 (RoPE),
:530-536 (freqs assembly).

A latent video is cut into 1×2×2 patches by a strided Conv3d, flattened to a sequence of
1536-d tokens. Each token carries a 3D position (frame f, row h, col w); 3D RoPE rotates
q,k using head_dim=128 split 44/42/42 across the three axes.
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, HStack, Swatches,
                       TextLabel, Formula, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 2 · latent → tokens: strided Conv3d patchify, then 3D RoPE positions",
    subtitle="patch_embedding = Conv3d(16→1536, kernel=stride=(1,2,2)): no temporal patching, 2×2 spatial. "
             "Flatten (f,h,w) → a sequence of L = f·(h/2)·(w/2) tokens of width 1536. 3D RoPE gives each token a "
             "(frame,row,col) position and rotates q,k; the 128-d head is split 44/42/42 across the three axes.")

# ---------------- left: the shape pipeline (bottom -> top) ------------------------
pipe = VStack([
    IOLabel("token sequence  (B, L, 1536)   L = f · h/2 · w/2", id="tok", modality="video"),
    Block("flatten  (f, h/2, w/2) → L", kind="linear", sub="dit.py:530  x.flatten(2).transpose(1,2)",
          id="flat", min_w=300),
    Block("patch_embedding  ·  Conv3d", kind="vae",
          sub="16 → 1536, kernel = stride = (1, 2, 2)", src="dit.py:382", id="conv", min_w=300),
    IOLabel("noised latent  x_σ  (B, 16, f, h, w)", id="lat", modality="video", hatched=True),
], gap=26)
d.place(pipe, 70, 150)
d.chain(["lat", "conv", "flat", "tok"])
d.note(d.box("conv").x2 + 18, d.box("conv").cy - 6,
       "(1,2,2) stride → spatial /2,\nchannels 16 → 1536", size=style.T_SUB, color=style.FAINT,
       max_w=170)

# concrete worked example
ex = TextLabel("worked example (81-frame 480×832 video):\n"
               "VAE → latent (16, 21, 60, 104)\n"
               "patchify (1,2,2) → (1536, 21, 30, 52)\n"
               "flatten → L = 21·30·52 = 32 760 tokens",
               size=style.T_SUB, color=style.MUTED, mono=True, max_w=300)
ex.measure()
d.place(ex, 70, d.box("lat").y2 + 26)
d.note(70, d.box("lat").y2 + 12, "shapes, concretely:", size=style.T_SUB, color=style.INK,
       weight="600")

# ---------------- right: 3D RoPE ---------------------------------------------------
RX = d.box("tok").x2 + 90
d.note(RX, 150, "3D RoPE — position of each token", size=style.T_SECTION, color=style.INK,
       weight="600")
rope = VStack([
    Block("rotate q, k by the per-axis angles", kind="cond",
          sub="rope_apply: view_as_complex · ×freqs · dit.py:94", id="rot", min_w=320),
    HStack([
        Block("frame f\n→ 44 dims", kind="cond", sub="freqs[0]", id="rf", min_w=98),
        Block("row h\n→ 42 dims", kind="cond", sub="freqs[1]", id="rh", min_w=98),
        Block("col w\n→ 42 dims", kind="cond", sub="freqs[2]", id="rw", min_w=98),
    ], gap=10),
    Block("head_dim = 128  split per axis", kind="norm",
          sub="44 + 42 + 42 = 128  ·  precompute_freqs_cis_3d", src="dit.py:77", id="split", min_w=320),
], gap=18)
d.place(rope, RX, 188)
# split fans up to the three per-axis bands, which then feed the rotate box (distinct ports)
d.edge("split.t@0.2", "rf.b@0.5", b_side="b", style_name="cond")
d.edge("split.t@0.5", "rh.b@0.5", b_side="b", style_name="cond")
d.edge("split.t@0.8", "rw.b@0.5", b_side="b", style_name="cond")
d.edge("rf.t@0.5", "rot.b@0.2", b_side="b", style_name="cond")
d.edge("rh.t@0.5", "rot.b@0.5", b_side="b", style_name="cond")
d.edge("rw.t@0.5", "rot.b@0.8", b_side="b", style_name="cond")
d.note(RX, d.box("split").y2 + 14,
       "Each token's q and k are rotated by an angle that depends on its (f,h,w) grid index — so "
       "attention sees relative 3D position. Applied to q,k only, every block, before attention "
       "(fig 5). v is not rotated.", size=style.T_SUB, color=style.MUTED, max_w=340)

leg = Swatches([("video", "latent / tokens"), ("vae", "Conv patchify"), ("linear", "reshape"),
                ("cond", "RoPE / position"), ("norm", "split"),
                (("#E0F2FE", "#0284C7"), "hatched = noised", "hatch")], max_w=620, id="leg")
d.place(leg, 70, 96)

d.save(OUT / "fig02_patchify_rope.svg")
print("ok")
