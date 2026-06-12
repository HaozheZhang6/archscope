"""Fig 7 — paper's MoT vs released code: same name, different architecture."""
from archscope import (Block, Diagram, Free, GroupFrame, HStack, IOLabel,
                       OpDot, RepeatStack, Spacer, Swatches, TextLabel, VStack,
                       style)
from common import OUT

d = Diagram(
    title="Fig 7 · “MoT” in the paper vs. in the released code",
    subtitle="Both are one joint-attention transformer over interleaved video+action tokens, "
             "but the paper describes a dual-stream block with modality-specific weights, while "
             "the released code ships a single shared-weight backbone — modality separation "
             "lives only at the embedders/heads and in the attention mask.")

# ---------------- left: paper ----------------------------------------------------
def pb(label, dim, mod, id, kind="linear"):
    return Block(label, kind=kind, sub=dim, modality=mod, id=id, min_w=140)

Y_QKV, Y_JA, Y_O, Y_FFN = 22, 90, 158, 226
paper_inner = Free([
    (TextLabel("video stream", size=style.T_SUB + 1, color="#0C4A6E", weight="600"), 30, 0),
    (TextLabel("action stream", size=style.T_SUB + 1, color="#064E3B", weight="600"), 222, 0),
    (pb("Norm + QKV", "d = 3072", "video", "pv_qkv"), 10, Y_QKV),
    (pb("Norm + QKV", "d = 768 ↑3072", "action", "pa_qkv"), 196, Y_QKV),
    (Block("Joint Self-Attention", kind="attention",
           sub="shared computation, separate weights", id="p_ja", min_w=326), 10, Y_JA),
    (pb("O proj", "d = 3072", "video", "pv_o"), 10, Y_O),
    (pb("down-proj + res", "3072 ↓768", "action", "pa_o"), 196, Y_O),
    (pb("FFN + norms", "d = 3072", "video", "pv_f", kind="ffn"), 10, Y_FFN),
    (pb("FFN + norms", "d = 768", "action", "pa_f", kind="ffn"), 196, Y_FFN),
], pad_r=10)
paper = RepeatStack(GroupFrame(paper_inner, dashed=False, tint="#FFFFFF",
                               stroke="#CBD5E1", id="paper_frame", pad=14),
                    times="×30", id="paper_stack")
ptitle = VStack([
    TextLabel("Paper §3.3 — dual-stream MoT (5.3B)", size=style.T_SECTION,
              color=style.INK, weight="600"),
    TextLabel("every non-attention parameter ×2 (per modality); action stream "
              "up/down-projects 768↔3072 around the joint attention",
              size=style.T_SUB + 1, color=style.MUTED, max_w=330),
    Spacer(0, 2),
    paper,
], gap=9, align="start")
d.place(ptitle, 40, 56)

pj = d.box("p_ja")
for src in ["pv_qkv", "pa_qkv"]:
    sb = d.box(src)
    d.edge((sb.cx, sb.y2), (sb.cx, pj.y), route="straight", arrow=True)
for dst in ["pv_o", "pa_o"]:
    db = d.box(dst)
    d.edge((db.cx, pj.y2), (db.cx, db.y), route="straight", arrow=True)
d.edge("pv_o", "pv_f", route="straight")
d.edge("pa_o", "pa_f", route="straight")

# ---------------- right: code -----------------------------------------------------
code_inner = Free([
    (VStack([
        Block("Norm + QKV + O", kind="linear", sub="d = 3072 · one set of weights",
              id="c_qkv", min_w=290),
        Block("Self-Attention", kind="attention",
              sub="FlexAttention: causal interleave mask", id="c_ja", min_w=290),
        Block("Cross-Attn (text) · FFN · norms", kind="ffn",
              sub="d = 3072 · shared by video & action tokens", id="c_f", min_w=290),
    ], gap=30), 0, 14),
    (TextLabel("one stream — video and action tokens flow through identical weights",
               size=style.T_SUB + 0.5, color=style.MUTED), 18, 0),
], pad_r=0)
code = RepeatStack(GroupFrame(code_inner, dashed=False, tint="#FFFFFF",
                              stroke="#CBD5E1", id="code_frame", pad=14),
                   times="×30", id="code_stack")
ctitle = VStack([
    TextLabel("Released code — shared single stream (≈5.0B)", size=style.T_SECTION,
              color=style.INK, weight="600"),
    TextLabel("modality separation only at the boundary: two input embedders, two "
              "condition embedders, two output heads; routing done by the attention "
              "mask, not by weights  (model.py:624-651)",
              size=style.T_SUB + 1, color=style.MUTED, max_w=360),
    Spacer(0, 56),          # room for the input chips above the frame
    code,
], gap=9, align="start")
d.place(ctitle, 480, 56)

d.chain(["c_qkv", "c_ja", "c_f"], route="straight")

# IO chips: inputs enter at the top, heads read out at the bottom
cb = d.box("code_stack")
io_in = HStack([
    IOLabel("patch_embedding_mlp (192→3072)", modality="video", id="ci_v"),
    IOLabel("action_embedder (30→3072)", modality="action", id="ci_a"),
], gap=12)
io_in.measure()
d.place(io_in, cb.cx - io_in.w / 2, cb.y - 50)
d.edge("ci_v.b", (cb.cx - 60, cb.y), b_side="t")
d.edge("ci_a.b", (cb.cx + 60, cb.y), b_side="t")
io_out = HStack([
    IOLabel("proj_out (3072→192)", modality="video", id="co_v"),
    IOLabel("action_proj_out (3072→30)", modality="action", id="co_a"),
], gap=12)
io_out.measure()
d.place(io_out, cb.cx - io_out.w / 2, cb.y2 + 28)
d.edge((cb.cx - 60, cb.y2), "co_v.t", a_side="b")
d.edge((cb.cx + 60, cb.y2), "co_a.t", a_side="b")

# ---------------- difference table ------------------------------------------------
y0 = max(d.box("paper_stack").y2, d.box("co_v").y2) + 48
rows = [
    ("joint attention over video+action in one sequence", "yes", "yes (mask-structured)"),
    ("modality-specific QKV / O / FFN / norms", "yes — the point of MoT", "no — all 30 blocks shared"),
    ("width asymmetry (video 3072 / action 768)", "yes (+350M action stream)", "no — actions live at 3072"),
    ("modality-specific time/text embedders & heads", "yes", "yes (only place weights split)"),
    ("params", "5.3B", "≈5.0B (Wan2.2-5B-sized)"),
]
x_q, x_p, x_c = 60, 460, 740
d.note(x_q, y0, "design question", size=style.T_SUB + 1, weight="600", color=style.INK)
d.note(x_p, y0, "paper", size=style.T_SUB + 1, weight="600", color=style.INK)
d.note(x_c, y0, "released code", size=style.T_SUB + 1, weight="600", color=style.INK)
d.doc.line("labels", x_q, y0 + 8, x_c + 240, y0 + 8, "#CBD5E1", 1.1)
for i, (q, p, c) in enumerate(rows):
    yy = y0 + 26 + i * 21
    d.note(x_q, yy, q, size=style.T_SUB + 1, color=style.MUTED)
    d.note(x_p, yy, p, size=style.T_SUB + 1,
           color="#166534" if p.startswith("yes") else "#9F1239")
    d.note(x_c, yy, c, size=style.T_SUB + 1,
           color="#166534" if c.startswith("yes") else "#9F1239")

d.note(x_q, y0 + 26 + len(rows) * 21 + 12,
       "Fossil evidence in the release: _keep_in_fp32_modules still lists action_norm1/2/3, "
       "text_norm1/2/3, scale_shift_table_action — modules that do not exist in the shipped "
       "WanTransformerBlock (model.py:581-593). The released code matches the “Share Weights” "
       "variant from the paper's initialization ablation (paper Fig. 7), not the headline MoT.",
       size=style.T_SUB + 1, color="#9F1239", max_w=900)

d.save(OUT / "fig07_paper_vs_code.svg")
print("ok")
