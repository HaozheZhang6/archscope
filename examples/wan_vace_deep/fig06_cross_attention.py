"""Wan2.1-VACE-1.3B · 6 · CrossAttention internals (text conditioning).
diffsynth wan_video_dit.py:165-201 (CrossAttention.forward, has_image_input=False for T2V).

Same shape as self-attention, but the query comes from the video tokens while the key/value
come from the TEXT context (fig 3). No RoPE (text has no 3D position). This is how the prompt
steers every block.
"""
from archscope import (Block, Diagram, IOLabel, VStack, HStack, Swatches, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 6 · CrossAttention  ·  video tokens query the text context",
    subtitle="q = RMSNorm(W_q · video_tokens), k = RMSNorm(W_k · text), v = W_v · text. Attention(q,k,v), then "
             "W_o. The query length is L (video tokens); the key/value length is 512 (UMT5 text). No RoPE here — "
             "text carries no 3D position. (The I2V variant adds a parallel image K/V; VACE T2V does not use it.)")

X = 360

top = VStack([
    IOLabel("cross-attention out  (B, L, 1536)", id="out"),
    Block("output projection  W_o", kind="linear", sub="Linear 1536 → 1536", src="dit.py:201", id="o"),
    Block("scaled-dot-product attention  (flash)", kind="attention",
          sub="softmax(qkᵀ/√128) v  ·  12 heads", id="attn", min_w=380),
], gap=16)
d.place(top, X, 110)
d.chain(["attn", "o", "out"])

# left: query from video tokens; right: key/value from text
qcol = VStack([
    Block("RMSNorm (q)", kind="norm", id="nq", min_w=170),
    Block("W_q", kind="linear", sub="1536→1536", id="wq", min_w=170),
    IOLabel("video tokens  (B, L, 1536)", id="vid", modality="video"),
], gap=14)
d.place(qcol, X - 30, d.box("attn").y2 + 44)

kvcol = VStack([
    HStack([
        Block("RMSNorm (k)", kind="norm", id="nk", min_w=150),
        Block("(v: no norm)", kind="io", id="vp", min_w=120),
    ], gap=16),
    HStack([
        Block("W_k", kind="linear", sub="1536→1536", id="wk", min_w=150),
        Block("W_v", kind="linear", sub="1536→1536", id="wv", min_w=120),
    ], gap=16),
    IOLabel("text context  (B, 512, 1536)   ← fig 3", id="txt", modality="text"),
], gap=14)
kvcol.measure()
d.place(kvcol, d.box("wq").x2 + 90, d.box("attn").y2 + 44)

d.chain(["vid", "wq", "nq"]); d.chain(["wk", "nk"]); d.chain(["wv", "vp"])
d.edge("txt.t@0.3", "wk.b@0.5", b_side="b", style_name="cond", color="#DB2777")
d.edge("txt.t@0.7", "wv.b@0.5", b_side="b", style_name="cond", color="#DB2777")
# q, k, v into attention
d.edge("nq.t@0.5", "attn.b@0.25", b_side="b", style_name="main", label="q (from video)", label_bg=True)
d.edge("nk.t@0.5", "attn.b@0.6", b_side="b", style_name="cond", color="#DB2777", label="k (text)", label_bg=True)
d.edge("vp.t@0.5", "attn.b@0.8", b_side="b", style_name="cond", color="#DB2777", label="v (text)", label_bg=True)

d.note(X - 30, d.box("vid").y2 + 24,
       "Result added back to x as a PLAIN residual (no gate, with its own affine norm3) — see the "
       "middle of the DiTBlock, fig 4.", size=style.T_SUB, color=style.MUTED, max_w=300)

leg = Swatches([("video", "video tokens (query)"), ("text", "text context (key/value)"),
                ("linear", "Linear"), ("norm", "RMSNorm"), ("attention", "attention"),
                ("cond", "text path", "edge")], max_w=720, id="leg")
d.place(leg, X - 30, d.box("out").y - 44)

d.save(OUT / "fig06_cross_attention.svg")
print("ok")
