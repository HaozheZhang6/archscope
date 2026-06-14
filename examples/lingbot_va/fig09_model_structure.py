"""Fig 9 — LingBot-VA model structure, paper vs code, rebuilt for legibility. Instead of
two full architectures side by side, draw ONE shared spine (the modules are the same in
both) and mark, per module, what the paper splits vs what the code shares. The ONLY real
structural difference is the 30 blocks: per-modality weights (paper) vs one shared set
(code). A small table carries the specifics. paper §3.3 + wan_va/modules/model.py."""
from archscope import (Block, Diagram, IOLabel, Swatches, TextLabel, VStack,
                       style)
from archscope.text import measure
from common import OUT

d = Diagram(
    title="Fig 9 · Same DiT backbone — the paper SPLITS the block weights per modality, the code SHARES them",
    subtitle="One spine, drawn once: video+action embed -> 30 blocks -> heads. The I/O modules are split by "
             "modality in BOTH. The only structural difference is the 30 blocks (paper: two per-modality streams "
             "3072/768; code: one shared 3072 stream, modalities kept apart by the attention mask).")

X = 360  # spine center

# ----- the shared spine (bottom -> top), drawn ONCE --------------------------------
spine = VStack([
    IOLabel("velocity v_z  ·  velocity v_a", id="out", modality="video"),
    Block("output heads:  proj_out (3072->192)  |  action_proj_out (3072->30)",
          kind="head", sub="split by modality", id="heads", min_w=380),
    Block("30 x  DiT block", kind="model",
          sub="self-attn · cross-attn(text) · FFN · per-token AdaLN", id="blk", min_w=380),
    Block("condition embedders:  condition_embedder  |  ..._action (deepcopy)",
          kind="cond", sub="time+text, split by modality", id="cond", min_w=380),
    Block("input embedders:  patch_embedding_mlp (192->3072)  |  action_embedder (30->3072)",
          kind="linear", sub="split by modality", id="emb", min_w=380),
    IOLabel("noised video latent  +  action tokens", id="in", modality="video", hatched=True),
], gap=20)
d.place(spine, X - 190, 130)
d.chain(["in", "emb", "cond", "blk", "heads", "out"])

# ----- per-module paper|code annotations (right of the spine) ----------------------
def same(mid, text):
    b = d.box(mid)
    d.note(b.x2 + 26, b.cy + 3.5, "paper = code:  " + text, size=style.T_SUB,
           color="#166534")
same("emb", "both split by modality")
same("cond", "both split by modality")
same("heads", "both split by modality")

# the block is THE difference — a highlighted callout
bb = d.box("blk")
dx = bb.x2 + 26
d.doc.rect("nodes", dx, bb.y - 6, 320, bb.h + 12, "rgba(220,38,38,0.05)", "#DC2626", 1.3, rx=8)
d.note(dx + 12, bb.cy - 14, "THE difference:", size=style.T_SUB + 1, color="#B91C1C", weight="600")
d.note(dx + 12, bb.cy + 1, "paper: 2 per-modality streams (d 3072 / 768)", size=style.T_SUB, color="#9F1239")
d.note(dx + 12, bb.cy + 15, "code: 1 SHARED stream (3072), mask routes", size=style.T_SUB, color="#9F1239")
d.edge((dx, bb.cy), (bb.x2, bb.cy), a_side="l", b_side="r", style_name="main", color="#DC2626")

# ----- compact table (bottom) ------------------------------------------------------
y0 = d.box("in").y2 + 56
rows = [
    ("block weights (QKV/O/FFN/norms)", "per modality", "ONE shared set"),
    ("width", "video 3072 / action 768", "all 3072"),
    ("kept apart by", "separate weights", "attention mask (fig5)"),
    ("params", "5.3B", "~5.0B"),
]
xq, xp, xc = 60, 380, 600
d.note(xq, y0, "question", size=style.T_SUB + 1, weight="600", color=style.INK)
d.note(xp, y0, "paper §3.3", size=style.T_SUB + 1, weight="600", color=style.INK)
d.note(xc, y0, "shipped code", size=style.T_SUB + 1, weight="600", color=style.INK)
d.doc.line("labels", xq, y0 + 8, xc + 220, y0 + 8, "#CBD5E1", 1.1)
for i, (q, p, c) in enumerate(rows):
    yy = y0 + 26 + i * 20
    d.note(xq, yy, q, size=style.T_SUB + 1, color=style.MUTED)
    d.note(xp, yy, p, size=style.T_SUB + 1, color="#9F1239")
    d.note(xc, yy, c, size=style.T_SUB + 1, color="#9F1239")
d.note(xq, y0 + 26 + len(rows) * 20 + 12,
       "Fossil: _keep_in_fp32_modules (model.py:581-593) still lists action_norm1/2/3, "
       "scale_shift_table_action — modules absent from the shipped block. The release matches "
       "the paper's own \"Share Weights\" ablation, not the headline dual-stream MoT.",
       size=style.T_SUB + 1, color="#9F1239", max_w=900)

leg = Swatches([("video", "video"), ("action", "action"), ("model", "block"),
                ("linear", "embed"), ("head", "head"), ("cond", "condition"),
                (("#FEE2E2", "#DC2626"), "the difference"),
                ("main", "data flow", "edge")], max_w=760, id="leg")
d.place(leg, 60, 90)

d.save(OUT / "fig09_model_structure.svg")
print("ok")
