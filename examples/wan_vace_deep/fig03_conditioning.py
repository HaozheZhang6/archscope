"""Wan2.1-VACE-1.3B · 3 · the two conditioning embedders: timestep and text.
diffsynth wan_video_dit.py:384-395 (text/time embeddings), :520-523 (forward).

The noise level σ becomes a timestep, embedded sinusoidally, projected to 1536, then
EXPANDED to 6·1536 — the per-block AdaLN parameters (fig 7). The text prompt is encoded by a
frozen UMT5 (4096-d) and projected to 1536 — the cross-attention context (fig 6).
"""
from archscope import (Block, Diagram, IOLabel, VStack, Swatches, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 3 · conditioning: timestep → AdaLN params, and text → cross-attn context",
    subtitle="Left: σ → sinusoidal(256) → MLP → t(1536) → time_projection → t_mod(6·1536), the modulation fed to "
             "every block (fig 7). The head gets its own 2·1536 from t. Right: UMT5 text(4096) → Linear·GELU·Linear "
             "→ context(1536), the K/V source for cross-attention (fig 6).")

# ---------------- left: timestep -> AdaLN -----------------------------------------
tcol = VStack([
    IOLabel("t_mod  (B, 6, 1536)   → every block's AdaLN", id="tmod", modality="none"),
    Block("time_projection", kind="cond", sub="SiLU · Linear 1536 → 6·1536", src="dit.py:394",
          id="tproj", min_w=260),
    IOLabel("t  (B, 1536)   (also → head: 2·1536)", id="temb", modality="none"),
    Block("time_embedding", kind="cond", sub="Linear 256→1536 · SiLU · Linear 1536→1536",
          src="dit.py:389", id="tmlp", min_w=260),
    Block("sinusoidal_embedding_1d", kind="linear", sub="σ·1000 → 256-d  ·  cos ⊕ sin",
          src="dit.py:70", id="sin", min_w=260),
    IOLabel("noise level  σ  →  timestep  (scalar)", id="sig", modality="none"),
], gap=18)
d.place(tcol, 80, 150)
d.chain(["sig", "sin", "tmlp", "temb", "tproj", "tmod"])
d.note(d.box("tproj").x2 + 18, d.box("tproj").cy - 6,
       "the 6 = shift/scale/gate\nfor self-attn + for FFN", size=style.T_SUB, color=style.FAINT,
       max_w=170)

# ---------------- right: text -> context ------------------------------------------
ccol = VStack([
    IOLabel("context  (B, 512, 1536)   → cross-attn K,V", id="ctx", modality="text"),
    Block("text_embedding", kind="cond", sub="Linear 4096→1536 · GELU(tanh) · Linear 1536→1536",
          src="dit.py:384", id="temb2", min_w=300, modality="text"),
    IOLabel("UMT5 text features  (B, 512, 4096)", id="umt5", modality="text"),
    Block("UMT5 text encoder  (frozen)", kind="cond", badge="frozen", sub="prompt → 4096-d tokens",
          id="enc", min_w=300, modality="text"),
    IOLabel("text prompt  (string)", id="prompt", modality="text"),
], gap=18)
d.place(ccol, d.box("tmod").x2 + 150, 168)
d.chain(["prompt", "enc", "umt5", "temb2", "ctx"])

d.note(80, d.box("sig").y2 + 26,
       "Why expand to 6·1536? AdaLN-single: instead of a per-block MLP, Wan keeps ONE timestep "
       "vector and a small learned table per block; their sum is chunked into the 6 modulation "
       "params used inside the block (fig 4, fig 7).", size=style.T_SUB, color=style.MUTED, max_w=460)

leg = Swatches([("cond", "embedder / conditioning"), ("linear", "fixed transform"),
                ("text", "text modality"), ("none", "scalar / timestep")], max_w=620, id="leg")
d.place(leg, 80, 96)

d.save(OUT / "fig03_conditioning.svg")
print("ok")
