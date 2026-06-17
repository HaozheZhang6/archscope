"""Fig 10a — LingBot-VA = AR × diffusion, the STRUCTURE (companion to fig10b's illustration).
The two axes as architecture: the AUTOREGRESSIVE sequence (interleaved video+action chunks,
causal, KV-cache) and the DIFFUSION denoise of one chunk (the shared DiT, s→AdaLN). For the
concrete 'what does each axis DO' picture on a moving clip, see fig10b."""
from pathlib import Path

from archscope import (Block, Diagram, GroupFrame, IOLabel, OpDot, Swatches,
                       TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 10a · LingBot-VA = AR × diffusion — the STRUCTURE  (see fig 10b for the moving-clip picture)",
    subtitle="SEQUENCE axis (autoregressive): video chunks (z) and action chunks (a) interleave into one causal "
             "sequence, generated chunk by chunk; the KV-cache holds the clean past. WITHIN-CHUNK axis (diffusion): "
             "each chunk starts as noise and is denoised by the shared DiT (timestep s → AdaLN). One DiT does both.")

VS = 46  # token glyph size

# ============ LAYER 1: the AR sequence (abstract structural tokens) ============
SY = 120
d.place(IOLabel("given: prompt + first obs o_0", id="g_in", modality="text"), 60, SY + VS / 2 - 11)

def vtok(x, label, grey=False, noised=False):
    stroke = "#CBD5E1" if grey else ("#B91C1C" if noised else "#0284C7")
    fill = ("#F1F5F9" if grey else
            (f'url(#{d.doc.hatch("#E0F2FE", "#0284C7")})' if noised else "#E0F2FE"))
    d.doc.rect("nodes", x, SY, VS, VS, fill, stroke, 2.0 if noised else 1.3, rx=6)
    cx, cy = x + VS / 2, SY + VS / 2
    tri = "#94A3B8" if grey else ("#B91C1C" if noised else "#0284C7")
    d.doc.path("nodes", f"M {cx-6},{cy-9} L {cx-6},{cy+9} L {cx+9},{cy} Z", fill=tri, stroke=tri, sw=1)
    d.doc.text("nodes", x + VS / 2, SY + VS + 12, label, style.T_TINY + 1,
               style.MUTED if not noised else "#B91C1C", "600" if noised else "500")

def atok(x, vals, label):
    w = 50
    d.doc.rect("nodes", x, SY + 8, w, VS - 16, "#D1FAE5", "#059669", 1.1, rx=5)
    d.doc.text("nodes", x + w / 2, SY + VS / 2 + 1, vals, style.T_TINY + 1, "#064E3B", "600")
    d.doc.text("nodes", x + w / 2, SY + VS + 12, label, style.T_TINY + 1, style.MUTED, "500")
    return w

x = 250
vtok(x, "z1"); x += VS + 6
aw = atok(x, "a1", "act"); x += aw + 20
vtok(x, "z2  GENERATING", noised=True); z2cx = x + VS / 2; x += VS + 6
atok(x, "a2  ?", "next"); x += 50 + 20
vtok(x, "z3", grey=True); x += VS + 6
atok(x, "...", "a3"); x += 50 + 16
d.doc.text("nodes", x + 8, SY + VS / 2 + 4, "...", style.T_LABEL, style.FAINT)

d.place(IOLabel("out: video + action chunks", id="g_out", modality="video"), x + 40, SY + VS / 2 - 11)
d.note(250, SY - 26, "AR axis — interleave & generate chunk by chunk:", size=style.T_SUB + 1,
       color="#0C4A6E", weight="700")
ay = SY + VS + 26
d.edge((250, ay), (x - 10, ay), color="#334155", width=1.4,
       label="chunk t+1 attends ≤ t  ·  KV-cache holds the clean past", label_side="right")
d.edge("g_in.r", (244, SY + VS / 2), b_side="l", style_name="faint")

# ============ LAYER 2 + 3: zoom into z2 → the diffusion denoise ============
DY = SY + VS + 70
stub = GroupFrame(VStack([
    Block("FFN", kind="ffn", id="s_ffn", min_w=230),
    Block("cross-attention  ← text", kind="attention", id="s_x", min_w=230),
    Block("self-attention  (causal mask, KV-cache)", kind="attention", id="s_self", min_w=230),
], gap=9), title="DiT block  ×30  (the same shared block · fig 3/4)", title_pos="tag", dashed=False,
    tint="rgba(71,85,105,0.04)", stroke="#475569", id="stub", pad=13)

body = VStack([
    IOLabel("clean chunk  z2   (→ goes back into the sequence above)", id="clean", modality="video"),
    Block("velocity head → predicted v_θ   (OUT: same shape as z2)", kind="head", id="vel", min_w=300),
    stub,
    IOLabel("IN: noised chunk  z2^(s)   (pure noise at s=0)", id="noised", modality="video", hatched=True),
], gap=18)
d.place(body, z2cx - 150, DY + 26)
d.chain(["noised", "stub", "vel", "clean"])
d.note(z2cx - 150, DY + 8, "diffusion axis — flow-matching denoise of ONE chunk:", size=style.T_SUB + 1,
       color="#6D28D9", weight="700")

nb = d.box("noised")
cleanb = d.box("clean")
d.edge((z2cx, SY + VS), (cleanb.cx, cleanb.y), style_name="faint", dash="3 3",
       a_side="b", b_side="t", color="#B91C1C", label="zoom: z2 is made by denoising (below)",
       label_bg=True)

vb = d.box("vel")
xr = min(nb.x, vb.x) - 36
d.edge((vb.x, vb.cy), (nb.x, nb.cy), a_side="l", b_side="l",
       via=[(xr, vb.cy), (xr, nb.cy)], color="#16A34A", width=1.6,
       label="×T: integrate s 0 → 1", label_side="left", label_bg=True)

# conditioning BUS
sb = d.box("stub")
bus_x = sb.x2 + 80
cond = VStack([
    Block("diffusion timestep s", kind="cond", sub="→ AdaLN", id="c_s", min_w=150),
    Block("UMT5 text", kind="cond", sub="prompt", id="c_t", modality="text", min_w=150),
], gap=14)
d.place(cond, bus_x, sb.cy - 30)
rail = sb.x2 + 36
d.edge("c_s.l", (rail, d.box("c_s").cy), a_side="l", arrow=False, style_name="cond")
d.edge("c_t.l", (rail, d.box("c_t").cy), a_side="l", arrow=False, style_name="cond", color="#DB2777")
d.doc.line("edges", rail, d.box("c_s").cy, rail, d.box("c_t").cy, "#94A3B8", 1.2, dash="5 3")
d.edge((rail, d.box("s_self").cy - 6), "s_self.r@0.4", style_name="cond",
       label="→ AdaLN", label_side="right", label_bg=True)
d.edge((rail, d.box("s_x").cy), "s_x.r@0.5", style_name="cond", color="#DB2777",
       label="→ K/V", label_side="right", label_bg=True)

# ============ LAYER 4: the action chunk ============
cb = d.box("clean")
a_note = Block("then a2: the SAME DiT denoises the action chunk", kind="io", modality="action",
               sub="conditioned on the just-made z2 (inverse dynamics) · s: 0 → 1", id="a2", min_w=270)
a_note.measure()
d.place(a_note, cb.x2 + 60, cb.cy - a_note.h / 2)
d.edge("clean.r@0.5", "a2.l@0.5", style_name="cache", color="#0D9488", label="z2 conditions a2",
       label_bg=True)

by = max(d.box("noised").y2, d.box("c_t").y2) + 34
d.note(60, by, "AR is the SEQUENCE axis (interleave z & a, causal, KV-cache, chunk by chunk); diffusion is the "
       "WITHIN-CHUNK axis (denoise each chunk through the DiT, s → AdaLN). fig 10b shows both on a moving clip.",
       size=style.T_SUB + 1, color=style.MUTED, max_w=1000)

leg = Swatches([(("#E0F2FE", "#0284C7"), "video chunk z"), ("action", "action chunk a"),
                ("model", "DiT (denoiser)"), ("head", "velocity head"),
                ("cond", "timestep / text"),
                (("#E0F2FE", "#B91C1C"), "noised (being denoised)", "hatch"),
                ("cache", "z conditions a", "edge")], max_w=900, id="leg")
d.place(leg, 60, 70)

d.save(OUT / "fig10a_structure.svg")
print("ok")
