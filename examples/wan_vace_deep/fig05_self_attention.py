"""Wan2.1-VACE-1.3B · 5 · SelfAttention internals.
diffsynth wan_video_dit.py:139-162 (SelfAttention.forward).

Standard multi-head attention with two Wan specifics: (1) q and k are RMSNorm-ed (QK-norm)
right after their projections; (2) q and k are rotated by 3D RoPE (fig 2). v is neither
normed nor rotated. 12 heads × 128 dims = 1536.
"""
from archscope import (Block, Diagram, IOLabel, OpDot, VStack, HStack, Swatches, style)
from common import OUT

d = Diagram(
    title="Wan2.1-VACE-1.3B · 5 · SelfAttention  ·  QK-RMSNorm + 3D RoPE, 12 heads × 128",
    subtitle="q = RMSNorm(W_q x), k = RMSNorm(W_k x), v = W_v x. RoPE rotates q,k by their 3D position (fig 2). "
             "Reshape to 12 heads, scaled-dot-product (flash) attention, then output projection W_o. Self-attention "
             "is over the FULL token sequence (all frames attend each other — bidirectional in space-time).")

X = 330

top = VStack([
    IOLabel("attention out  (B, L, 1536)", id="out"),
    Block("output projection  W_o", kind="linear", sub="Linear 1536 → 1536", src="dit.py:149", id="o"),
    Block("scaled-dot-product attention  (flash)", kind="attention",
          sub="softmax(qkᵀ/√128) v  ·  12 heads  ·  dit.py:30", id="attn", min_w=360),
    Block("reshape → 12 heads × 128", kind="norm", sub="b s (n d) → b n s d", id="heads", min_w=360),
], gap=16)
d.place(top, X, 96)
d.chain(["heads", "attn", "o", "out"])

# q / k / v columns feeding the reshape+attention
qcol = VStack([
    Block("RoPE(q)", kind="cond", sub="rotate · fig 2", id="ropeq", min_w=150),
    Block("RMSNorm  (q)", kind="norm", sub="QK-norm", id="nq", min_w=150),
    Block("W_q", kind="linear", sub="1536→1536", id="wq", min_w=150),
], gap=12)
kcol = VStack([
    Block("RoPE(k)", kind="cond", sub="rotate · fig 2", id="ropek", min_w=150),
    Block("RMSNorm  (k)", kind="norm", sub="QK-norm", id="nk", min_w=150),
    Block("W_k", kind="linear", sub="1536→1536", id="wk", min_w=150),
], gap=12)
vcol = VStack([
    Block("(no norm, no RoPE)", kind="io", sub="v passes straight", id="vpass", min_w=150),
    Block("W_v", kind="linear", sub="1536→1536", id="wv", min_w=150),
], gap=12)
row = HStack([qcol, kcol, vcol], gap=40)
row.measure()
d.place(row, X + (360 - row.w) / 2, d.box("heads").y2 + 40)

inp = IOLabel("hidden states  x  (modulated, fig 4)", id="in")
inp.measure()
d.place(inp, X + (360 - inp.w) / 2, d.box("wq").y2 + 30)
# x fans into W_q, W_k, W_v
for b in ("wq", "wk", "wv"):
    d.edge("in.t@0.5", f"{b}.b@0.5", b_side="b", style_name="main")
d.chain(["wq", "nq", "ropeq"]); d.chain(["wk", "nk", "ropek"]); d.chain(["wv", "vpass"])
# q,k,v converge into the reshape/attention
d.edge("ropeq.t@0.5", "heads.b@0.2", b_side="b", style_name="main", label="q", label_bg=True)
d.edge("ropek.t@0.5", "heads.b@0.5", b_side="b", style_name="main", label="k", label_bg=True)
d.edge("vpass.t@0.5", "heads.b@0.8", b_side="b", style_name="main", label="v", label_bg=True)

d.note(d.box("attn").x2 + 22, d.box("attn").cy - 6,
       "no causal mask: every token attends every token (video is generated all-frames-at-once, "
       "not autoregressively).", size=style.T_SUB, color=style.FAINT, max_w=200)

leg = Swatches([("io", "tensor"), ("linear", "Linear / projection"), ("norm", "RMSNorm / reshape"),
                ("cond", "RoPE"), ("attention", "attention")], max_w=620, id="leg")
d.place(leg, X, d.box("out").y - 44)

d.save(OUT / "fig05_self_attention.svg")
print("ok")
