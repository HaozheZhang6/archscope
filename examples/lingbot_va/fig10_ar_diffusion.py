"""Fig 10 — how LingBot-VA's architecture is AR x diffusion. fig9 only compared which
weights are shared; this figure shows the generative mechanism the model actually is:
an AUTOREGRESSIVE interleaved video-action sequence (causal, KV-cache, chunk by chunk),
where each chunk is produced by FLOW-MATCHING DIFFUSION through the DiT (the timestep
drives per-token AdaLN). The blocks in fig9 are DiT blocks; here is what they do."""
from archscope import (Block, Diagram, IOLabel, RepeatStack, Swatches,
                       TextLabel, TokenRow, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 10 · The architecture is AR x diffusion — interleaved causal chunks, each denoised by the DiT",
    subtitle="Two axes. SEQUENCE axis (autoregressive): video and action tokens are interleaved into one causal "
             "sequence and generated chunk by chunk; the KV-cache holds the clean past (causal mask -> fig5). "
             "WITHIN-CHUNK axis (diffusion): each chunk starts as noise and is denoised by the DiT via flow "
             "matching; the diffusion timestep drives the per-token AdaLN that conditions every block (-> fig3).")

# ============ TOP: the AR axis — interleaved causal sequence ============
SY = 110
seq = TokenRow([
    dict(label="z1", modality="video", w=40, sub="frame"),
    dict(label="a1", modality="action", w=40, sub="actions", gap_after=14),
    dict(label="z2", modality="video", w=40, hatch=True, bold_border=True, sub="GENERATING"),
    dict(label="a2", modality="action", w=40, gap_after=14),
    dict(label="z3", modality="video", w=40), dict(label="a3", modality="action", w=40),
    dict(label="...", modality="none", w=34),
], cell_w=40, cell_h=30, id="seq")
seq.measure()
d.place(seq, 150, SY)
d.note(150, SY - 26, "autoregressive: one interleaved video+action sequence, generated chunk by chunk "
       "(z = video frame, a = its tau=4 actions)", size=style.T_SUB + 0.5, color=style.INK,
       weight="600")
sb = d.box("seq")
# causal generation arrow + history/future labels, below the row's sub-labels
ay = sb.y2 + 22
d.edge((sb.x, ay), (sb.x2 - 30, ay), color="#334155", width=1.4,
       label="generate left -> right · chunk t+1 attends <= t", label_side="right")
d.note(150, ay + 18, "<- clean history (KV-cache)", size=style.T_SUB, color="#0D9488")
d.note(sb.x2, ay + 18, "future ->", size=style.T_SUB, color=style.FAINT, anchor="end")

# ============ ZOOM: how the GENERATING chunk z2 is produced ============
# z2 is the 3rd cell: after z1,a1 (2 cells + gap) + the 14px gap_after.
z2x = sb.x + 2 * (40 + 3.5) + 14 + 20
d.note(z2x, ay + 40, "zoom: how one chunk is generated", size=style.T_SUB, color=style.MUTED,
       anchor="middle")

# ============ BODY: the diffusion axis — flow-matching denoise of one chunk ============
DX = 420
body = VStack([
    IOLabel("clean chunk  z2  (the generated frame)", id="clean", modality="video"),
    Block("velocity head  v_theta", kind="head",
          sub="OUT: predicted velocity, same shape as z2 · v = (data - noise) direction", id="vel",
          min_w=300),
    RepeatStack(Block("DiT block", kind="model",
                      sub="self-attn (causal mask, KV-cache) · cross-attn(text) · FFN",
                      src="-> fig3 / fig9 (same shared block)", id="blk", min_w=300),
                times="×30", id="dit"),
    IOLabel("IN: noised chunk  z2^(s)  (pure noise at s=0)", id="noised",
            modality="video", hatched=True),
], gap=22)
d.place(body, DX, SY + 128)
d.chain(["noised", "dit", "vel", "clean"])

# zoom lines from the z2 cell down to the diffusion stack top
nb = d.box("noised")
d.edge((z2x, sb.y2 + 28), (nb.cx, nb.y), style_name="faint", dash="4 3", a_side="b", b_side="t")

# the denoising loop (flow matching, s: 0 -> 1) — on the LEFT
vb = d.box("vel")
xr = nb.x - 38
d.edge((vb.x, vb.cy), (nb.cx, nb.y), a_side="l", b_side="t",
       via=[(xr, vb.cy), (xr, nb.y - 16)], color="#16A34A", width=1.5,
       label="x T:  integrate s 0 -> 1  (flow matching)", label_side="left")

# the diffusion conditioning: timestep s -> AdaLN (the thing that makes it a DiT) — RIGHT, mid
ditb = d.box("dit")
ada = Block("diffusion timestep  s  ->  AdaLN", kind="cond",
            sub="temb + scale_shift_table -> 6 params (shift,scale,gate)x2 · model.py:524 · "
                "conditions EVERY block", id="ada", min_w=240)
ada.measure()
d.place(ada, ditb.x2 + 56, ditb.cy - ada.h / 2 - 18)
d.edge("ada.l", "dit.r@0.42", style_name="cond")
txt = Block("umT5 text", kind="cond", sub="prompt -> cross-attn K/V", id="txt", modality="text")
txt.measure()
d.place(txt, ditb.x2 + 56, ditb.cy + 24)
d.edge("txt.l", "dit.r@0.72", style_name="cond", color="#DB2777")

# the action chunk: a compact note at the TOP-RIGHT (clean's row), well above the AdaLN rail
cb = d.box("clean")
a_note = Block("then: action chunk a2", kind="io", modality="action",
               sub="same DiT denoises the actions, conditioned on the predicted z2\n"
                   "(inverse dynamics) · s: 0 -> 1", id="a_note", min_w=250)
a_note.measure()
d.place(a_note, cb.x2 + 70, cb.cy - a_note.h / 2)
d.edge("clean.r@0.5", "a_note.l@0.5", style_name="cache", color="#0D9488", label="z2")

# ============ BOTTOM: the two-axis summary ============
by = max(d.box("noised").y2, d.box("txt").y2) + 46
d.note(150, by,
       "So: AR is the SEQUENCE axis (interleave video+action, causal, KV-cache, chunk by chunk); "
       "diffusion is the WITHIN-CHUNK axis (flow-matching denoise through the DiT, timestep -> AdaLN).",
       size=style.T_SUB + 1, color=style.INK, max_w=1000)
d.note(150, by + 16,
       "The shared DiT backbone of fig9 is exactly this DiT — one network both predicts video (forward "
       "dynamics) and decodes actions (inverse dynamics), at every autoregressive step.",
       size=style.T_SUB + 1, color=style.MUTED, max_w=1000)

leg = Swatches([("video", "video chunk / latent"), ("action", "action chunk"),
                ("model", "DiT (denoiser)"), ("head", "velocity head"),
                ("cond", "diffusion timestep / text"),
                (("#E0F2FE", "#0284C7"), "hatched = noised", "hatch")], max_w=720, id="leg")
d.place(leg, 150, SY - 60)

d.save(OUT / "fig10_ar_diffusion.svg")
print("ok")
