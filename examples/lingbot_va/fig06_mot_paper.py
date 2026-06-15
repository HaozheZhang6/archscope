"""Fig 6 — the MoT block as the PAPER describes it (dual-stream, asymmetric width)."""
from archscope import (Block, Diagram, Formula, IOLabel, OpDot, Swatches,
                       TextLabel, TokenRow, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 6 · Mixture-of-Transformers block — as described in the paper (§3.3)",
    subtitle="Dual-stream MoT: every non-attention parameter (QKV/O, FFN, norms) is "
             "modality-specific; only the self-attention computation is shared. Video stream "
             "d=3072 (init from Wan2.2-5B), action stream d=768 (~350M params, same depth 30). "
             "NOTE: this is NOT what the released code implements — see Fig 7.")

V_W, A_W = 250, 210

# blocks are coloured by KIND (norm/linear/ffn/attention). The stream is carried by
# column position + the "video/action-specific" sublabels + the token pills, so the
# modality accent bar is dropped — it was unkeyed and duplicated column position, and
# its blue/green collided in the legend with the attention/ffn fills.
def vblock(label, sub=None, kind="norm", id=None):
    return Block(label, kind=kind, sub=sub, id=id, min_w=V_W)

def ablock(label, sub=None, kind="norm", id=None):
    return Block(label, kind=kind, sub=sub, id=id, min_w=A_W)

# --- lane stacks -------------------------------------------------------------------
v_top = VStack([
    IOLabel("video tokens  (L_z, 3072)", id="v_in", modality="video"),
    vblock("LayerNorm", "video-specific", id="v_ln1"),
    vblock("QKV proj", "W_qkv video  (3072 → 3·3072)", kind="linear", id="v_qkv"),
], gap=15)
a_top = VStack([
    IOLabel("action tokens  (L_a, 768)", id="a_in", modality="action"),
    ablock("LayerNorm", "action-specific", id="a_ln1"),
    ablock("QKV proj", "W_qkv action  (768 → …)", kind="linear", id="a_qkv"),
    ablock("up-proj", "Linear 768 → 3072", kind="linear", id="a_up"),
], gap=15)
v_bot = VStack([
    vblock("O proj", "W_o video  (3072 → 3072)", kind="linear", id="v_o"),
    OpDot("+", id="v_add1"),
    vblock("LayerNorm", "video-specific", id="v_ln2"),
    vblock("FFN", "video-specific  (3072 → …)", kind="ffn", id="v_ffn"),
    OpDot("+", id="v_add2"),
    IOLabel("video tokens  (L_z, 3072)", id="v_out", modality="video"),
], gap=15)
a_bot = VStack([
    ablock("down-proj", "Linear 3072 → 768", kind="linear", id="a_dn"),
    OpDot("+", id="a_add1"),
    ablock("LayerNorm", "action-specific", id="a_ln2"),
    ablock("FFN", "action-specific  (768 → …)", kind="ffn", id="a_ffn"),
    OpDot("+", id="a_add2"),
    IOLabel("action tokens  (L_a, 768)", id="a_out", modality="action"),
], gap=15)

for s in (v_top, a_top, v_bot, a_bot):
    s.measure()

VXL = 150                                   # video lane left edge
AXL = VXL + max(v_top.w, v_bot.w) + 110     # action lane left edge
vcx = VXL + max(v_top.w, v_bot.w) / 2
acx = AXL + max(a_top.w, a_bot.w) / 2

# --- interleaved input sequence ----------------------------------------------------
seq = TokenRow([
    dict(label="z₁", modality="video", w=34),
    dict(label="a₁", modality="action"), dict(label="a₂", modality="action"),
    dict(label="a₃", modality="action"), dict(label="a₄", modality="action", gap_after=8),
    dict(label="z₂", modality="video", w=34),
    dict(label="a₅", modality="action"), dict(label="a₆", modality="action"),
    dict(label="a₇", modality="action"), dict(label="a₈", modality="action", gap_after=8),
    dict(label="…", modality="none", w=30),
], cell_w=26, cell_h=24, id="seq")
seq.measure()
d.place(seq, VXL + 40, 40)
d.note(VXL + 40, 30, "interleaved sequence: one video frame z_t, then its τ=4 actions "
       "(video runs at 1/4 of the action rate)",
       size=style.T_SUB + 0.5, color=style.FAINT)

# --- place lanes -------------------------------------------------------------------
d.place(v_top, VXL + (max(v_top.w, v_bot.w) - v_top.w) / 2, 120)
d.place(a_top, AXL + (max(a_top.w, a_bot.w) - a_top.w) / 2, 120)

ja_y = max(d.box("v_qkv").y2, d.box("a_up").y2) + 40
ja = Block("Joint Self-Attention", kind="attention",
           sub="one attention over the concatenated [video ; action] sequence — "
               "the ONLY shared computation",
           id="ja", min_w=(acx - vcx) + 240, h=52)
ja.measure()
d.place(ja, (vcx + acx) / 2 - ja.w / 2, ja_y)

boty = ja_y + 52 + 40
d.place(v_bot, VXL + (max(v_top.w, v_bot.w) - v_bot.w) / 2, boty)
d.place(a_bot, AXL + (max(a_top.w, a_bot.w) - a_bot.w) / 2, boty)

# --- edges ---------------------------------------------------------------------------
sq = d.box("seq")
d.edge((sq.x + 17, sq.y2), "v_in.t", style_name="faint", color="#0284C7",
       route="zv", frac=0.55)          # under z₁ (a video token)
d.edge((sq.x + 64, sq.y2), "a_in.t", style_name="faint", color="#059669",
       route="zv", frac=0.55)          # under a₁ (an action token), not the gap by z₂
d.note(sq.x2 + 16, sq.cy + 4, "route by modality", size=style.T_SUB,
       color=style.FAINT)

d.chain(["v_in", "v_ln1", "v_qkv"])
d.chain(["a_in", "a_ln1", "a_qkv", "a_up"])
jb = d.box("ja")
d.edge("v_qkv.b", (vcx, jb.y), route="straight", label="q,k,v (3072)",
       label_side="left")
d.edge("a_up.b", (acx, jb.y), route="straight", label="q,k,v (3072)")
d.edge((vcx, jb.y2), "v_o.t", route="straight")
d.edge((acx, jb.y2), "a_dn.t", route="straight")
d.chain(["v_o", "v_add1", "v_ln2", "v_ffn", "v_add2", "v_out"])
d.chain(["a_dn", "a_add1", "a_ln2", "a_ffn", "a_add2", "a_out"])

# residual rails (video: left, action: right)
v_rail = d.box("v_ffn").x - 34
a_rail = d.box("a_ffn").x2 + 34
for src, dst, rx, left in [("v_in", "v_add1", v_rail, True),
                           ("v_add1", "v_add2", v_rail, True),
                           ("a_in", "a_add1", a_rail, False),
                           ("a_add1", "a_add2", a_rail, False)]:
    sb, db = d.box(src), d.box(dst)
    sp = (sb.x, sb.cy) if left else (sb.x2, sb.cy)
    dp = (db.x, db.cy) if left else (db.x2, db.cy)
    d.edge(sp, dp, style_name="residual", via=[(rx, sb.cy), (rx, db.cy)])
note_x = a_rail + 18
d.note(note_x, d.box("a_ln2").cy - 6, "residual keeps the action", size=style.T_SUB,
       color=style.FAINT)
d.note(note_x, d.box("a_ln2").cy + 7, "stream in its 768-d space", size=style.T_SUB,
       color=style.FAINT)

# --- right column: legend + repeat -------------------------------------------------
# key the KINDS (the fills) only. The blue/green token pills self-label ("video
# tokens" / "action tokens"), so a modality swatch is unneeded — and would duplicate
# the attention-blue / ffn-green it sits next to.
leg = Swatches([("attention", "shared attention"), ("linear", "linear / projection"),
                ("ffn", "FFN"), ("norm", "norm")], max_w=210, id="leg")
leg.measure()
leg_x = note_x + 160
d.place(leg, leg_x, 130)
d.note(leg_x, 260, "× 30 layers", size=style.T_SUB + 2, color=style.INK,
       weight="600")
d.note(leg_x, 276, "same depth in both streams", size=style.T_SUB,
       color=style.MUTED)

# --- bottom annotations --------------------------------------------------------------
notes_y = max(d.box("v_out").y2, d.box("a_out").y2) + 46
frame_note = VStack([
    TextLabel("Action-stream initialization (paper §3.3, ablated in paper Fig. 7):",
              size=style.T_SUB + 1, color=style.MUTED, weight="600"),
    Formula(r"$W_{action} = \mathrm{interp}(W_{video})\cdot\alpha,\qquad \alpha=\sqrt{d_v/d_a}=2$",
            size=12.5),
    TextLabel("random init → unstable (high grad norms); sharing video weights → stable but "
              "suboptimal; interpolated copy + variance-preserving scale works best.",
              size=style.T_SUB + 0.5, color=style.MUTED, max_w=560),
], gap=8, align="start")
frame_note.measure()
d.place(frame_note, VXL - 40, notes_y)

d.note(VXL - 40, notes_y + frame_note.h + 22,
       "Language is NOT a third stream: a frozen UMT5 encoder is injected via "
       "cross-attention. MoT citation: Liang et al., TMLR 2025 (arXiv:2411.04996).",
       size=style.T_SUB + 0.5, color=style.FAINT, max_w=620)

d.save(OUT / "fig06_mot_paper.svg")
print("ok")
