"""Fig 2 — WanTransformer3DModel training forward pass. Rebuilt for legibility:
one bottom->top spine (inputs -> embed -> concat -> DiT x30 -> split -> heads ->
velocity outputs), conditioning gathered into ONE right-side bus that taps the
block at named ports, the block inlined as a 3-row stub, and an edge/symbol legend.
wan_va/modules/model.py:702-798."""
from archscope import (Block, Diagram, GroupFrame, HStack, IOLabel, Swatches,
                       TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 2 · WanTransformer3DModel — training forward pass:  4 noised/clean segments -> 2 velocities",
    subtitle="Noisy & clean video latents and noisy & clean actions are embedded to 3072, concatenated into ONE "
             "sequence, pushed through 30 shared DiT blocks, split back, and read out as per-modality velocities. "
             "Conditioning (RoPE, mask, text, per-frame timestep) taps every block.")

X = 430  # spine center

# ---- bottom: INPUT pills ----------------------------------------------------------
ins = HStack([
    IOLabel("noisy z (B,48,F,h,w)", modality="video", hatched=True, id="nz"),
    IOLabel("clean z (B,48,F,h,w)", modality="video", id="cz"),
    IOLabel("noisy a (B,30,F,16,1)", modality="action", hatched=True, id="na"),
    IOLabel("clean a (B,30,F,16,1)", modality="action", id="ca"),
], gap=20)
ins.measure()
d.place(ins, X - ins.w / 2, 720)
d.note(X - ins.w / 2, 720 + 30, "INPUT: noised targets + clean conditioning (teacher forcing)",
       size=style.T_SUB + 0.5, color=style.INK, weight="600")

# ---- embedders --------------------------------------------------------------------
emb = HStack([
    Block("patch_embedding_mlp", kind="linear", sub="Linear 192 -> 3072",
          src="model.py:624 · both z's", id="pe", min_w=210),
    Block("action_embedder", kind="linear", sub="Linear 30 -> 3072",
          src="model.py:627 · both a's", id="ae", min_w=210),
], gap=70)
emb.measure()
d.place(emb, X - emb.w / 2, 636)
d.edge("nz", "pe"); d.edge("cz", "pe")
d.edge("na", "ae"); d.edge("ca", "ae")

# ---- concat → one sequence --------------------------------------------------------
seq = IOLabel("concat -> one sequence  (1, L, 3072)   L = [noisy z | clean z | noisy a | clean a] + pad",
              id="seq")
seq.measure()
d.place(seq, X - seq.w / 2, 576)
d.edge("pe", "seq"); d.edge("ae", "seq")

# ---- the DiT block stub, x30 (inlined, not a pointer) -----------------------------
stub = GroupFrame(VStack([
    Block("FFN", kind="ffn", id="s_ffn", min_w=300),
    Block("cross-attention  <- text", kind="attention", id="s_x", min_w=300),
    Block("self-attention  (FlexAttention mask, RoPE)", kind="attention", id="s_self", min_w=300),
], gap=10), title="WanTransformerBlock  x30  ·  per-token AdaLN  (-> fig3 for the full block)",
    title_pos="tag", dashed=False, tint="rgba(71,85,105,0.04)", stroke="#475569",
    id="stub", pad=14)
d.place(stub, X - 165, 400)
d.chain(["s_self", "s_x", "s_ffn"])
d.edge("seq", "stub.b@0.5", b_side="b", label="(1, L, 3072)")

# ---- conditioning BUS (right), named-port taps ------------------------------------
sb = d.box("stub")
bus_x = sb.x2 + 70
cond = VStack([
    Block("per-frame timestep", kind="cond", sub="-> 6-param AdaLN", id="c_t", min_w=170),
    Block("text emb (UMT5)", kind="cond", sub="(1,B·512,3072)", id="c_txt", modality="text", min_w=170),
    Block("FlexAttention mask", kind="mask", sub="-> fig5", id="c_m", min_w=170),
    Block("3D RoPE", kind="cond", sub="grid (f,y,x)", id="c_r", min_w=170),
], gap=12)
cond.measure()
d.place(cond, bus_x, sb.cy - cond.h / 2)
rail = sb.x2 + 34
for cid in ("c_t", "c_txt", "c_m", "c_r"):
    cb = d.box(cid)
    d.edge("%s.l" % cid, (rail, cb.cy), a_side="l", arrow=False, style_name="cond",
           color="#DB2777" if cid == "c_txt" else None)
d.doc.line("edges", rail, d.box("c_t").cy, rail, d.box("c_r").cy, "#94A3B8", 1.2, dash="5 3")
d.edge((rail, d.box("s_self").cy + 6), "s_self.r@0.55", style_name="cond", label="-> AdaLN / RoPE / mask",
       label_side="right")
d.edge((rail, d.box("s_x").cy), "s_x.r@0.5", style_name="cond", color="#DB2777",
       label="-> cross-attn K/V", label_side="right")

# ---- norm_out + split → heads -----------------------------------------------------
no = Block("norm_out + AdaLN(shift, scale)", kind="norm", src="model.py:646-651", id="no", min_w=300)
no.measure()
d.place(no, X - no.w / 2, 322)
d.edge("stub", "no")

heads = HStack([
    Block("proj_out", kind="head", sub="3072 -> 192 -> unpatchify", modality="video", id="ho", min_w=160),
    Block("action_proj_out", kind="head", sub="3072 -> 30", modality="action", id="ha", min_w=160),
], gap=40)
heads.measure()
d.place(heads, X - heads.w / 2, 250)
d.edge("no.t@0.4", "ho.b@0.5", b_side="b", label="noisy z half")
d.edge("no.t@0.6", "ha.b@0.5", b_side="b", label="noisy a half")

# discarded clean+pad half as a greyed branch
disc = Block("clean z, clean a, pad", kind="io", sub="discarded — prediction read only from the noisy half",
             id="disc", min_w=240)
disc.measure()
d.place(disc, d.box("no").x2 + 50, d.box("no").cy - disc.h / 2)
d.edge("no.r@0.5", "disc.l@0.5", a_side="r", style_name="faint", arrow=True)

# ---- top: OUTPUT velocity pills ---------------------------------------------------
outs = HStack([
    IOLabel("OUT: velocity v_z (B,48,F,h,w)", modality="video", id="vo"),
    IOLabel("OUT: velocity v_a (B,30,F,16,1)", modality="action", id="va"),
], gap=60)
outs.measure()
d.place(outs, X - outs.w / 2, 180)
d.edge("ho", "vo"); d.edge("ha", "va")

leg = Swatches([("video", "video"), ("action", "action"), ("text", "text"),
                ("model", "DiT block"), ("cond", "conditioning"), ("head", "head"),
                (("#E0F2FE", "#0284C7"), "hatched = noised", "hatch"),
                ("main", "data flow", "edge"), ("cond", "conditioning tap", "edge"),
                ("faint", "discarded", "edge")], max_w=720, id="leg")
d.place(leg, X - 360, 130)

d.save(OUT / "fig02_forward.svg")
print("ok")
