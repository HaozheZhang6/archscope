"""Fig 2 — WanTransformer3DModel.forward_train: the whole-model dataflow (code version)."""
from archscope import (Block, Diagram, GroupFrame, HStack, IOLabel, OpDot,
                       RepeatStack, Spacer, Swatches, TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 2 · WanTransformer3DModel — training forward pass (code version)",
    subtitle="Four segments — noisy/clean video latents and noisy/clean actions — are embedded, "
             "concatenated into ONE sequence, and pushed through 30 shared blocks; the mask "
             "(Fig 5) supplies all structure. wan_va/modules/model.py:702-798")

X = 430

# ---- inputs row -------------------------------------------------------------------
ins = HStack([
    IOLabel("noisy z  (B,48,F,h,w)", modality="video", hatched=True, id="nz"),
    IOLabel("clean z  (B,48,F,h,w)", modality="video", id="cz"),
    IOLabel("noisy a  (B,30,F,16,1)", modality="action", hatched=True, id="na"),
    IOLabel("clean a  (B,30,F,16,1)", modality="action", id="ca"),
], gap=22)
ins.measure()
d.place(ins, X - ins.w / 2, 640)

# ---- embedders --------------------------------------------------------------------
emb = HStack([
    Block("patch_embedding_mlp", kind="linear", sub="Linear 48·1·2·2=192 → 3072",
          src="model.py:624 (one set of weights, applied to noisy & clean)", id="pe"),
    Block("action_embedder", kind="linear", sub="Linear 30 → 3072",
          src="model.py:627 (one set of weights, applied to noisy & clean)", id="ae"),
], gap=46)
emb.measure()
d.place(emb, X - emb.w / 2, 560)

for s in ("nz", "cz"):
    d.edge(s, "pe")
for s in ("na", "ca"):
    d.edge(s, "ae")

# ---- concat -----------------------------------------------------------------------
cc = Block("concat  [noisy z | clean z | noisy a | clean a]  + pad to ×128", kind="op",
           id="cat", min_w=420)
cc.measure()
d.place(cc, X - cc.w / 2, 500)
d.edge("pe", "cat", label="×2")
d.edge("ae", "cat", label="×2", label_side="left")

seq = IOLabel("(1, B·(2·F·h·w/4 + 2·F·16) + pad, 3072)", id="seq")
seq.measure()
d.place(seq, X - seq.w / 2, 448)
d.edge("cat", "seq", route="straight")

# ---- transformer body ---------------------------------------------------------------
body = RepeatStack(
    Block("WanTransformerBlock", kind="model",
          sub="self-attn (FlexAttention mask) · cross-attn (text) · FFN · per-token AdaLN",
          src="→ Fig 3", id="blk", min_w=380),
    times="×30", id="body")
body.measure()
d.place(body, X - body.w / 2, 330)
d.edge("seq", "body")

# side conditioners
rope = Block("3D RoPE", kind="cond", sub="grid (f, y, x) — actions use (f, 1..16, 1)",
             src="model.py:622, 727-731", id="rope")
rope.measure()
bb = d.box("body")
d.place(rope, bb.x - rope.w - 64, bb.cy - 36 - rope.h / 2)
d.edge("rope.r", "body.l@0.28", style_name="cond")

mask = Block("FlexAttn mask", kind="mask", sub="rebuilt each step (random chunk 1-4, window 4-64)",
             src="→ Fig 5", id="mask")
mask.measure()
d.place(mask, bb.x - mask.w - 64, bb.cy + 36 - mask.h / 2)
d.edge("mask.r", "body.l@0.75", style_name="cond", color="#0D9488")

txt = Block("text emb", kind="cond", sub="UMT5 (B,512,4096) → Linear → (B·512, 3072)",
            src="model.py:714  · cross-attn KV", id="txt", modality="text")
txt.measure()
d.place(txt, bb.x2 + 64, bb.cy - 36 - txt.h / 2)
d.edge("txt.l", "body.r@0.28", style_name="cond", color="#DB2777")

tstep = Block("per-token timesteps", kind="cond",
              sub="condition_embedder (video) | condition_embedder_action (action)",
              src="model.py:628-635 · separate weights per modality", id="ts")
tstep.measure()
d.place(tstep, bb.x2 + 64, bb.cy + 36 - tstep.h / 2)
d.edge("ts.l", "body.r@0.75", style_name="cond")
tsb = d.box("ts")
d.note(tsb.x, tsb.y2 + 14,
       "noisy & clean segments get their own t — clean history may be re-noised "
       "(augmentation, Fig 8)", size=style.T_SUB, color=style.FAINT, max_w=250)

# ---- output ---------------------------------------------------------------------------
no = Block("norm_out + AdaLN(shift, scale)", kind="norm",
           sub="scale_shift_table (1,2,3072) + per-token temb", src="model.py:646-651, 780-787",
           id="no", min_w=320)
no.measure()
d.place(no, X - no.w / 2, 252)
d.edge("body", "no")

sp = Block("split  [L_z | L_z | L_a | L_a | pad]", kind="op", id="sp", min_w=300)
sp.measure()
d.place(sp, X - sp.w / 2, 196)
d.edge("no", "sp", route="straight")

heads = HStack([
    Block("proj_out", kind="head", sub="3072 → 192 → unpatchify", src="model.py:647", id="ho",
          modality="video"),
    Block("(clean z, clean a, pad — discarded)", kind="io", sub="prediction is only read "
          "from the noisy segments", id="disc"),
    Block("action_proj_out", kind="head", sub="3072 → 30", src="model.py:649", id="ha",
          modality="action"),
], gap=40)
heads.measure()
d.place(heads, X - heads.w / 2, 110)
d.edge("sp", "ho")
d.edge("sp", "ha")
d.edge("sp", "disc", route="straight", style_name="faint", arrow=False)

outs = HStack([
    IOLabel("v_z  (B, 48, F, h, w)", modality="video", id="vo"),
    Spacer(150, 1),
    IOLabel("v_a  (B, 30, F, 16, 1)", modality="action", id="va"),
], gap=30)
outs.measure()
d.place(outs, X - outs.w / 2, 44)
d.edge("ho", "vo", label="velocity")
d.edge("ha", "va", label="velocity", label_side="left")

leg = Swatches([("video", "video"), ("action", "action"), ("text", "text"),
                (("#E0F2FE", "#0284C7"), "hatched = noisy", "hatch"),
                ("linear", "linear/embed"), ("cond", "conditioning"),
                ("mask", "mask"), ("head", "output head")], max_w=460, id="leg")
ib = d.box("nz")
d.place(leg, ib.x, 712)

d.save(OUT / "fig02_forward.svg")
print("ok")
