"""Fig 10 — LingBot-VA is AR x diffusion. Rebuilt for legibility (reviewer playbook):
one input pill -> one output pill spine, the AR sequence drawn with CONCRETE tokens
(video frames as image thumbnails, actions as value chips, like the paper's Fig 2),
the diffusion zoom nested directly under the generating chunk (no whitespace gap), the
DiT block inlined as a 3-row stub (not a pointer), conditioning gathered into one bus
with named ports, and the two loops (x30 layers vs xT denoise) drawn differently.

Video-frame thumbnails use a CC0 image as an illustrative stand-in (assets/flower)."""
from pathlib import Path

from archscope import (Block, Diagram, GroupFrame, IOLabel, OpDot, Swatches,
                       TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 10 · LingBot-VA = AR x diffusion — generate one interleaved video+action sequence, each chunk denoised by the DiT",
    subtitle="SEQUENCE axis (autoregressive): video frames (z) and action chunks (a) interleave into one causal "
             "sequence, generated chunk by chunk; KV-cache holds the clean past. WITHIN-CHUNK axis (diffusion): "
             "each chunk starts as noise and is denoised by the DiT (timestep s -> AdaLN). Video frames drawn as "
             "frame icons, actions as value chips (illustrative).")

VS = 46  # video thumbnail size

# ============ LAYER 1: the AR sequence, concrete tokens, input->output spine ============
SY = 120
d.place(IOLabel("given: prompt + first obs o_0", id="g_in", modality="text"), 60, SY + VS/2 - 11)

# interleaved chunks: z1 a1 | z2(gen) a2 | z3 a3.
# a video token is drawn as a neutral "video frame" glyph (rounded frame + play
# triangle), tinted blue; hatched when it is the chunk being denoised.
def vtok(x, label, grey=False, noised=False):
    stroke = "#CBD5E1" if grey else ("#B91C1C" if noised else "#0284C7")
    fill = ("#F1F5F9" if grey else
            (f'url(#{d.doc.hatch("#E0F2FE", "#0284C7")})' if noised else "#E0F2FE"))
    d.doc.rect("nodes", x, SY, VS, VS, fill, stroke, 2.0 if noised else 1.3, rx=6)
    cx, cy = x + VS / 2, SY + VS / 2
    tri = "#94A3B8" if grey else ("#B91C1C" if noised else "#0284C7")
    d.doc.path("nodes", f"M {cx-6},{cy-9} L {cx-6},{cy+9} L {cx+9},{cy} Z",
               fill=tri, stroke=tri, sw=1)
    d.doc.text("nodes", x + VS / 2, SY + VS + 12, label, style.T_TINY + 1,
               style.MUTED if not noised else "#B91C1C", "600" if noised else "500")
def atok(x, vals, label):
    w = 50
    d.doc.rect("nodes", x, SY + 8, w, VS - 16, "#D1FAE5", "#059669", 1.1, rx=5)
    d.doc.text("nodes", x + w/2, SY + VS/2 + 1, vals, style.T_TINY + 1, "#064E3B", "600")
    d.doc.text("nodes", x + w/2, SY + VS + 12, label, style.T_TINY + 1, style.MUTED, "500")
    return w

x = 250
vtok(x, "z1 frame"); x += VS + 6
aw = atok(x, "1.7 1.2", "a1"); x += aw + 20
vtok(x, "z2  GENERATING", noised=True); z2cx = x + VS/2; x += VS + 6
atok(x, "? ?", "a2"); x += 50 + 20
vtok(x, "z3", grey=True); x += VS + 6
atok(x, "...", "a3"); x += 50 + 16
d.doc.text("nodes", x + 8, SY + VS/2 + 4, "...", style.T_LABEL, style.FAINT)

d.place(IOLabel("out: video frames + action chunks", id="g_out", modality="video"), x + 40, SY + VS/2 - 11)
d.note(250, SY - 26, "AUTOREGRESSIVE sequence — generate chunk by chunk:", size=style.T_SUB + 1,
       color=style.INK, weight="600")
# causal arrow under the row
ay = SY + VS + 26
d.edge((250, ay), (x - 10, ay), color="#334155", width=1.4,
       label="chunk t+1 attends <= t · KV-cache holds the clean past (z1,a1)", label_side="right")
d.edge("g_in.r", (244, SY + VS/2), b_side="l", style_name="faint")

# ============ LAYER 2 + 3: zoom into z2 -> the diffusion denoise (no gap) ============
DY = SY + VS + 70
stub = GroupFrame(VStack([
    Block("FFN", kind="ffn", id="s_ffn", min_w=230),
    Block("cross-attention  <- text", kind="attention", id="s_x", min_w=230),
    Block("self-attention  (causal mask, KV-cache)", kind="attention", id="s_self", min_w=230),
], gap=9), title="DiT block  x30  (the same shared block)", title_pos="tag", dashed=False,
    tint="rgba(71,85,105,0.04)", stroke="#475569", id="stub", pad=13)

body = VStack([
    IOLabel("clean chunk  z2   (-> goes back into the sequence above)", id="clean", modality="video"),
    Block("velocity head -> predicted v_theta   (OUT: same shape as z2)", kind="head", id="vel", min_w=300),
    stub,
    IOLabel("IN: noised chunk  z2^(s)   (pure noise at s=0)", id="noised", modality="video", hatched=True),
], gap=18)
d.place(body, z2cx - 150, DY + 26)
d.chain(["noised", "stub", "vel", "clean"])
d.note(z2cx - 150, DY + 8, "DIFFUSION within the chunk — flow-matching denoise:", size=style.T_SUB + 1,
       color=style.INK, weight="600")

# magnifier: connect z2 cell to the diffusion IN, no whitespace canyon
nb = d.box("noised")
d.edge((z2cx, SY + VS), (nb.cx, nb.y), style_name="faint", dash="3 3", a_side="b", b_side="t",
       color="#B91C1C", label="zoom: how z2 is made")

# the xT denoise loop (distinct: green, back-edge on the left into noised's LEFT
# side — NOT the top, which the zoom arrow already uses)
vb = d.box("vel")
xr = nb.x - 36
d.edge((vb.x, vb.cy), (nb.x, nb.cy), a_side="l", b_side="l",
       via=[(xr, vb.cy), (xr, nb.cy)], color="#16A34A", width=1.6,
       label="x T: integrate s 0 -> 1", label_side="left")

# conditioning BUS with named ports (right of the stub)
sb = d.box("stub")
bus_x = sb.x2 + 80
cond = VStack([
    Block("diffusion timestep s", kind="cond", sub="model.py:524", id="c_s", min_w=150),
    Block("umT5 text", kind="cond", sub="prompt", id="c_t", modality="text", min_w=150),
], gap=14)
d.place(cond, bus_x, sb.cy - 30)
rail = sb.x2 + 36
d.edge("c_s.l", (rail, d.box("c_s").cy), a_side="l", arrow=False, style_name="cond")
d.edge("c_t.l", (rail, d.box("c_t").cy), a_side="l", arrow=False, style_name="cond", color="#DB2777")
d.doc.line("edges", rail, d.box("c_s").cy, rail, d.box("c_t").cy, "#94A3B8", 1.2, dash="5 3")
d.edge((rail, d.box("s_self").cy - 6), "s_self.r@0.4", style_name="cond",
       label="-> AdaLN", label_side="right")
d.edge((rail, d.box("s_x").cy), "s_x.r@0.5", style_name="cond", color="#DB2777",
       label="-> K/V", label_side="right")

# ============ LAYER 4: the action chunk (compact, conditioned on z2) ============
cb = d.box("clean")
a_note = Block("then a2: the SAME DiT denoises the action chunk", kind="io", modality="action",
               sub="conditioned on the just-made z2 (inverse dynamics) · s: 0 -> 1", id="a2", min_w=270)
a_note.measure()
d.place(a_note, cb.x2 + 60, cb.cy - a_note.h / 2)
d.edge("clean.r@0.5", "a2.l@0.5", style_name="cache", color="#0D9488", label="z2 conditions a2")

# ============ bottom one-liner ============
by = max(d.box("noised").y2, d.box("c_t").y2) + 34
d.note(60, by, "Read it as: AR is the SEQUENCE axis (interleave z & a, causal, KV-cache, chunk by chunk); "
       "DIFFUSION is the WITHIN-CHUNK axis (denoise each chunk through the DiT, s -> AdaLN). One DiT does both.",
       size=style.T_SUB + 1, color=style.MUTED, max_w=1000)

leg = Swatches([(("#E0F2FE", "#0284C7"), "video frame token z"), ("action", "action chunk a"),
                ("model", "DiT (denoiser)"), ("head", "velocity head"),
                ("cond", "diffusion timestep / text"),
                (("#E0F2FE", "#B91C1C"), "noised (being denoised)", "hatch")], max_w=820, id="leg")
d.place(leg, 60, 70)

d.save(OUT / "fig10_ar_diffusion.svg")
print("ok")
