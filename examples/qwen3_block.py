"""A modern LLM decoder layer — Qwen3, drawn from a real implementation.

Structure grounded in nano-vllm (nanovllm/models/qwen3.py); dims = Qwen3-8B.
Shows: asymmetric GQA shapes on edges, fused projections, per-head QK-RMSNorm,
SwiGLU as a two-branch Free layout, RepeatStack ×36.
"""
from pathlib import Path

from archscope import (Block, Diagram, Free, GroupFrame, IOLabel, KnownChip,
                       OpDot, RepeatStack, Swatches, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Qwen3 decoder layer (8B config) — as implemented",
    subtitle="Grounded in nano-vllm: fused QKV with grouped-query KV heads, per-head "
             "QK-RMSNorm, RoPE, and a fused-gate SwiGLU MLP. "
             "nanovllm/models/qwen3.py:14-159")

# ---------------- attention half ----------------------------------------------------
attn_inner = VStack([
    Block("o_proj", kind="linear", sub="Linear 4096 → 4096 (no bias)",
          src="qwen3.py:49", id="o"),
    Block("attention", kind="attention", sub="GQA: 8 KV heads shared by 32 Q heads",
          src="qwen3.py:62", id="attn"),
    Block("RoPE", kind="cond", sub="rotary on q, k · θ=1e6", src="qwen3.py:56", id="rope"),
    Block("q_norm · k_norm", kind="norm", sub="RMSNorm per HEAD (dim 128) — Qwen3 signature",
          src="qwen3.py:68-70", id="qknorm"),
    Block("qkv_proj  (fused)", kind="linear",
          sub="Linear 4096 → 32·128 + 8·128 + 8·128", src="qwen3.py:42", id="qkv"),
], gap=16)

# ---------------- SwiGLU half (two-branch) -------------------------------------------
gate = Block("gate", kind="linear", sub="4096 → 12288", id="gate", min_w=130)
up = Block("up", kind="linear", sub="4096 → 12288", id="up", min_w=130)
silu = Block("SiLU", kind="ffn", id="silu", min_w=130)
mul = OpDot("x", id="mul")
down = Block("down_proj", kind="linear", sub="12288 → 4096", src="qwen3.py:105", id="down")
for el in (gate, up, silu, mul, down):
    el.measure()
LW = 150
mlp_inner = Free([
    (down, (2 * LW + 16 - down.w) / 2, 0),
    (mul, (2 * LW + 16 - mul.w) / 2, down.h + 18),
    (silu, (LW - silu.w) / 2, down.h + 18 + mul.h + 16),
    (gate, (LW - gate.w) / 2, down.h + 18 + mul.h + 16 + silu.h + 16),
    (up, LW + 16 + (LW - up.w) / 2, down.h + 18 + mul.h + 16 + (silu.h + 16 + gate.h) / 2 - up.h / 2),
], pad_r=0, pad_b=0)
mlp_frame = GroupFrame(mlp_inner, title="SwiGLU MLP   (fused gate_up_proj · qwen3.py:100-111)",
                       dashed=False, tint="rgba(34,197,94,0.04)", stroke="#16A34A",
                       pad=14, id="mlp")

# ---------------- the layer ------------------------------------------------------------
inner = VStack([
    OpDot("+", id="add2"),
    mlp_frame,
    Block("RMSNorm", kind="norm", sub="post_attention_layernorm", src="qwen3.py:144", id="ln2"),
    OpDot("+", id="add1"),
    GroupFrame(attn_inner, title="self-attention", dashed=False,
               tint="rgba(37,99,235,0.04)", stroke="#2563EB", pad=14, id="sa"),
    Block("RMSNorm", kind="norm", sub="input_layernorm", src="qwen3.py:143", id="ln1"),
], gap=18)
card = GroupFrame(inner, dashed=False, tint="#FFFFFF", stroke="#94A3B8", pad=20)
layer = RepeatStack(card, times="×36", id="layer")

col = VStack([
    IOLabel("hidden states  (T, 4096)", id="out"),
    layer,
    IOLabel("hidden states  (T, 4096)", id="inp"),
], gap=28)
d.place(col, 300, 60)

# main chain
d.edge("inp", "layer")
d.chain(["ln1", "qkv", "qknorm", "rope", "attn"])
d.edge("attn", "o", label="(T, 32, 128)", label_at=0.45)
d.edge("sa", "add1")
d.chain(["add1", "ln2"])
d.edge("ln2", "mlp")
# SwiGLU internals
d.edge("mlp.b@0.30", "gate", arrow=True, stub=8)
d.edge("mlp.b@0.72", "up", arrow=True, stub=8)
d.edge("gate", "silu")
d.edge("silu", "mul")
d.edge("up", "mul.r@0.5", route="vh")
d.edge("mul", "down")
d.edge("mlp", "add2")

# GQA shape labels on the q/k/v path (left margin, clear of the card)
qb = d.box("qkv")
lx = d.box("layer").x - 12
d.note(lx, qb.cy - 8, "q: (T, 32, 128)", size=style.T_TINY + 0.5,
       color=style.MUTED, mono=True, anchor="end")
d.note(lx, qb.cy + 4, "k,v: (T, 8, 128)", size=style.T_TINY + 0.5,
       color=style.MUTED, mono=True, anchor="end")
d.note(lx, qb.cy + 16, "= 4x smaller KV cache", size=style.T_TINY + 0.5,
       color=style.FAINT, anchor="end")

# residual rails
rail = d.box("layer").x + 9
cx = d.box("sa").cx
for tap_y, dst in [(d.box("ln1").y2 + 9, "add1"),
                   ((d.box("add1").y + d.box("ln2").y2) / 2, "add2")]:
    db = d.box(dst)
    d.edge((cx, tap_y), (db.x, db.cy), style_name="residual",
           via=[(rail, tap_y), (rail, db.cy)])
d.note(rail - 6, d.box("add1").cy - 26, "fused residual: RMSNorm(x, res) "
       "returns both (qwen3.py:152-157)", size=style.T_TINY, color=style.FAINT,
       anchor="end")

chip = KnownChip("scaled dot-product attention", ref="known", id="kc")
chip.measure()
ab = d.box("attn")
d.place(chip, d.box("layer").x - chip.w - 54, ab.cy - chip.h / 2)
d.edge("kc.r", (d.box("layer").x, ab.cy), style_name="faint", arrow=False, dash="3 3")

leg = Swatches([("attention", "attention"), ("ffn", "activation"),
                ("linear", "linear"), ("norm", "RMSNorm"), ("cond", "position"),
                ("known", "known concept", "dash")], max_w=240, id="leg")
d.place(leg, d.box("kc").x, 70)

d.save(OUT / "qwen3_block.svg")
print("wrote", OUT / "qwen3_block.svg")
