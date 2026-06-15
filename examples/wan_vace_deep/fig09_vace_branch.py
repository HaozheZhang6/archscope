"""Wan2.1-VACE-1.3B · 9 · the VACE branch (VaceWanModel) — a 15-block side-network.
diffsynth models/wan_video_vace.py:27-74 (VaceWanModel.forward).

The control tokens flow through 15 VaceWanAttentionBlocks (separate weights from the main DiT,
same architecture). Each block emits ONE hint via its after_proj (a zero-conv). The 15 hints
are injected into the main DiT at layers 0,2,…,28 (fig 11).
"""
from archscope import (Block, Diagram, IOLabel, VStack, GroupFrame, Swatches,
                       TextLabel, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 9 · the VACE branch — 15 blocks process the control, emit 15 hints",
    subtitle="vace_patch_embedding (Conv3d 96→1536) turns vace_context into control tokens c, zero-padded to the "
             "main sequence length L. c then flows through 15 VaceWanAttentionBlocks (fig 10), each producing one "
             "hint = after_proj(block output). Separate weights from the main DiT; only this branch is trained.")

X = 250

# the control-token spine through the 15 blocks
spine = VStack([
    IOLabel("(final c discarded)", id="cend", modality="none"),
    Block("VaceWanAttentionBlock  ·  layer 28", kind="trainable", sub="block_id = 28", id="b28", min_w=320),
    Block("⋮   layers 4, 6, …, 26   (12 more blocks)", kind="trainable", sub="each emits a hint", id="dots",
          min_w=320),
    Block("VaceWanAttentionBlock  ·  layer 2", kind="trainable", sub="block_id = 2", id="b2", min_w=320),
    Block("VaceWanAttentionBlock  ·  layer 0", kind="trainable",
          sub="block_id = 0 — also injects the main latent x (fig 10)", id="b0", min_w=320),
    Block("zero-pad to length L", kind="linear", sub="c → (1, L, 1536)  ·  vace.py:60-63", id="pad", min_w=320),
    Block("vace_patch_embedding  ·  Conv3d", kind="trainable", sub="96 → 1536, kernel=stride=(1,2,2)",
          src="vace.py:51", id="vconv", min_w=320),
    IOLabel("vace_context  (96, f, h, w)   ← fig 8", id="vc", modality="state"),
], gap=15)
d.place(spine, X, 120)
d.chain(["vc", "vconv", "pad", "b0", "b2", "dots", "b28", "cend"])

# the hints fan out to the right (each block -> one hint)
hints = VStack([
    IOLabel("hint @ 28", id="h28", modality="none"),
    IOLabel("hints @ 4…26  (×12)", id="hdots", modality="none"),
    IOLabel("hint @ 2", id="h2", modality="none"),
    IOLabel("hint @ 0", id="h0", modality="none"),
], gap=33, align="start")
d.place(hints, d.box("b28").x2 + 70, d.box("b0").y - 6)
for b, h, lab in [("b0", "h0", "after_proj"), ("b2", "h2", ""), ("dots", "hdots", ""),
                  ("b28", "h28", "")]:
    d.edge(f"{b}.r@0.5", f"{h}.l@0.5", a_side="r", style_name="cache", color="#B45309",
           label=lab, label_bg=bool(lab))
d.note(d.box("h0").x, d.box("h28").y - 26, "15 hints  →  injected into the main DiT (fig 11)",
       size=style.T_SUB + 1, color="#B45309", weight="600")

d.note(X, d.box("vc").y2 + 18,
       "x (the main noised latent, after patchify) is fed in ONCE at block 0 (before_proj(c)+x). "
       "The text context, timestep modulation and RoPE are the SAME tensors the main DiT uses — the "
       "VACE blocks are ordinary DiT blocks (fig 4) with two extra Linear layers (fig 10).",
       size=style.T_SUB, color=style.MUTED, max_w=640)

leg = Swatches([("state", "control latent"), ("trainable", "VACE block (trained)"),
                ("linear", "pad"), ("none", "hint / tensor"),
                ("cache", "hint (after_proj)", "edge")], max_w=720, id="leg")
d.place(leg, X, 96)

d.save(OUT / "fig09_vace_branch.svg")
print("ok")
