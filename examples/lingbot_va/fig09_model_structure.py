"""Fig 9 — LingBot-VA model structure: the paper's description (§3.3) vs the actual
module tree shipped in the code (wan_va/modules/model.py). Same model name, two
different architectures: the paper describes a dual-stream MoT with per-modality
weights and asymmetric width; the released code is ONE shared backbone whose
modality split lives only at the I/O boundary, routed by an attention mask."""
from archscope import (Block, Diagram, Free, GroupFrame, IOLabel, OpDot,
                       RepeatStack, Swatches, TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 9 · LingBot-VA model structure — paper §3.3  vs  shipped code (wan_va/modules/model.py)",
    subtitle="Left: as written — a dual-stream Mixture-of-Transformers, every non-attention weight per modality, "
             "video d=3072 / action d=768. Right: as shipped — one shared 3072-d backbone (30 identical blocks); "
             "modality split only at 2 input embedders + 2 condition embedders + 2 output heads; routing by the "
             "FlexAttention mask, not by weights. The module names below are the real ones in model.py.")

# ============ LEFT: paper §3.3 — dual-stream MoT ============
def vb(label, sub, kind="linear", id=None):
    return Block(label, kind=kind, sub=sub, modality="video", id=id, min_w=150)
def ab(label, sub, kind="linear", id=None):
    return Block(label, kind=kind, sub=sub, modality="action", id=id, min_w=130)

Y_QKV, Y_JA, Y_O, Y_FFN = 22, 92, 162, 232
paper_inner = Free([
    (TextLabel("video stream  d=3072", size=style.T_SUB + 1, color="#0C4A6E", weight="600"), 24, 0),
    (TextLabel("action stream  d=768", size=style.T_SUB + 1, color="#064E3B", weight="600"), 232, 0),
    (vb("Norm + QKV", "W_qkv^video", id="pv_qkv"), 10, Y_QKV),
    (ab("Norm + QKV", "W_qkv^action", id="pa_qkv"), 210, Y_QKV),
    (ab("up-proj", "768 -> 3072", id="pa_up"), 210, Y_QKV + 40),
    (Block("Joint Self-Attention", kind="attention",
           sub="shared computation, per-modality weights", id="p_ja", min_w=340), 10, Y_JA),
    (vb("O proj", "W_o^video", id="pv_o"), 10, Y_O),
    (ab("down-proj + res", "3072 -> 768", id="pa_o"), 210, Y_O),
    (vb("FFN + norms", "per-modality", kind="ffn", id="pv_f"), 10, Y_FFN),
    (ab("FFN + norms", "per-modality", kind="ffn", id="pa_f"), 210, Y_FFN),
], pad_r=8)
paper = RepeatStack(GroupFrame(paper_inner, dashed=False, tint="#FFFFFF",
                               stroke="#CBD5E1", id="p_frame", pad=14),
                    times="×30", id="p_stack")
ptitle = VStack([
    TextLabel("PAPER §3.3 — dual-stream MoT  (5.3B)", size=style.T_SECTION,
              color=style.INK, weight="600"),
    TextLabel("two parallel transformers; action up/down-projects 768<->3072 around "
              "the joint attention; init W_action = interp(W_video)·sqrt(d_v/d_a). "
              "Language = frozen UMT5 via cross-attn (not a stream).",
              size=style.T_SUB + 1, color=style.MUTED, max_w=370),
    paper,
], gap=10, align="start")
d.place(ptitle, 40, 60)
pj = d.box("p_ja")
for s in ("pv_qkv", "pa_up"):
    sb = d.box(s)
    d.edge((sb.cx, sb.y2), (sb.cx, pj.y), route="straight")
for t in ("pv_o", "pa_o"):
    tb = d.box(t)
    d.edge((tb.cx, pj.y2), (tb.cx, tb.y), route="straight")
d.edge("pa_qkv", "pa_up", route="straight")
d.edge("pv_o", "pv_f", route="straight")
d.edge("pa_o", "pa_f", route="straight")

# ============ RIGHT: shipped code — one shared backbone (the real module tree) ============
PX = 760
# the 30x shared block, with its real sublayers
block = RepeatStack(GroupFrame(VStack([
    Block("norm3 -> ffn  (FeedForward)", kind="ffn", sub="model.py:507,510", id="c_ffn", min_w=300),
    Block("norm2 -> attn2  WanAttention (cross -> text)", kind="attention",
          sub="model.py:494,502", id="c_x", min_w=300),
    Block("norm1 -> attn1  WanAttention (self)", kind="attention",
          sub="to_q/k/v · norm_q/k RMSNorm · to_out · KV-cache · model.py:484", id="c_self",
          min_w=300),
], gap=12), title="WanTransformerBlock  (model.py:468)", title_pos="tag", dashed=False,
    tint="rgba(71,85,105,0.04)", stroke="#475569", id="c_block", pad=15),
    times="×30", id="c_stack")

repo_inner = VStack([
    # split output heads
    Free([(Block("proj_out  3072->192", kind="head", sub="video · model.py:647",
                 modality="video", id="c_hv"), 0, 0),
          (Block("action_proj_out  3072->30", kind="head", sub="action · model.py:649",
                 modality="action", id="c_ha"), 230, 0)], pad_r=0),
    block,
    # split input + condition embedders
    Free([(Block("patch_embedding_mlp  192->3072", kind="linear", sub="video · model.py:624",
                 modality="video", id="c_ev"), 0, 0),
          (Block("action_embedder  30->3072", kind="linear", sub="action · model.py:627",
                 modality="action", id="c_ea"), 230, 0)], pad_r=0),
    Free([(Block("condition_embedder", kind="cond", sub="time+text, video · :628",
                 modality="video", id="c_cv"), 0, 0),
          (Block("condition_embedder_action", kind="cond", sub="deepcopy · :635",
                 modality="action", id="c_ca"), 230, 0)], pad_r=0),
], gap=18, align="center")
rtitle = VStack([
    TextLabel("SHIPPED CODE — one shared backbone  (≈5.0B)", size=style.T_SECTION,
              color=style.INK, weight="600"),
    TextLabel("WanTransformer3DModel (model.py:569): 30 identical blocks at d=3072. "
              "Modality split ONLY at the 6 boundary modules below; the blocks are one "
              "shared set of weights. Per-token AdaLN via scale_shift_table.",
              size=style.T_SUB + 1, color=style.MUTED, max_w=380),
    repo_inner,
], gap=10, align="start")
d.place(rtitle, PX, 60)
d.chain(["c_self", "c_x", "c_ffn"], route="straight")
cb = d.box("c_stack")
# inputs flow up into the stack, heads read out of the top
d.edge("c_ev.t", (cb.cx - 60, cb.y2), a_side="t", b_side="b")
d.edge("c_ea.t", (cb.cx + 60, cb.y2), a_side="t", b_side="b")
d.edge((cb.cx - 60, cb.y), "c_hv.b", a_side="t", b_side="b")
d.edge((cb.cx + 60, cb.y), "c_ha.b", a_side="t", b_side="b")
# the mask routes modalities (not weights)
mask = Block("FlexAttnFunc mask", kind="mask",
             sub="interleave + causal routing\nmodel.py:42-201 · → fig5", id="c_mask")
mask.measure()
d.place(mask, cb.x2 + 40, d.box("c_self").cy - mask.h / 2)
d.edge("c_mask.l", "c_self.r@0.5", style_name="cond", color="#0D9488")

# ============ BOTTOM: the discrepancy ============
y0 = max(d.box("p_stack").y2, d.box("c_cv").y2) + 44
rows = [
    ("non-attention weights (QKV/O/FFN/norms)", "per modality (the point of MoT)", "ONE shared set, all 30 blocks"),
    ("width", "video 3072 / action 768 (asymmetric)", "everything at 3072"),
    ("how modalities are kept apart", "separate weights + streams", "I/O embedders/heads + attention mask"),
    ("params", "5.3B", "≈5.0B (Wan2.2-5B-sized)"),
]
xq, xpp, xc = 60, 470, 800
d.note(xq, y0, "question", size=style.T_SUB + 1, weight="600", color=style.INK)
d.note(xpp, y0, "paper §3.3", size=style.T_SUB + 1, weight="600", color=style.INK)
d.note(xc, y0, "shipped code", size=style.T_SUB + 1, weight="600", color=style.INK)
d.doc.line("labels", xq, y0 + 8, xc + 300, y0 + 8, "#CBD5E1", 1.1)
for i, (q, p, c) in enumerate(rows):
    yy = y0 + 26 + i * 20
    d.note(xq, yy, q, size=style.T_SUB + 1, color=style.MUTED)
    d.note(xpp, yy, p, size=style.T_SUB + 1, color="#166534" if "per" in p or "3072 /" in p else "#9F1239")
    d.note(xc, yy, c, size=style.T_SUB + 1, color="#9F1239" if ("ONE" in c or "everything" in c) else "#166534")
d.note(xq, y0 + 26 + len(rows) * 20 + 10,
       "Fossil in the release: _keep_in_fp32_modules (model.py:581-593) still lists action_norm1/2/3, "
       "text_norm1/2/3, scale_shift_table_action — modules that do NOT exist in the shipped block. The "
       "released code matches the paper's own \"Share Weights\" ablation, not the headline dual-stream MoT.",
       size=style.T_SUB + 1, color="#9F1239", max_w=1080)
d.note(xq, y0 + 26 + len(rows) * 20 + 40,
       "NOTE: this figure compares WEIGHT-SHARING only. Both sides are DiT blocks in an AR-diffusion world "
       "model (noised latent in, flow-matching velocity out, timestep -> AdaLN, interleaved causal chunks). "
       "fig 10 shows that generative mechanism.", size=style.T_SUB + 1, color=style.INK, max_w=1080)

leg = Swatches([("video", "video weights"), ("action", "action weights"),
                ("attention", "attention"), ("ffn", "FFN"), ("linear", "embed"),
                ("cond", "condition"), ("head", "head"), ("mask", "mask routing")],
               max_w=760, id="leg")
d.place(leg, 40, 20)

d.save(OUT / "fig09_model_structure.svg")
print("ok")
