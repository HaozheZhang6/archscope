"""Fig 1 — LingBot-VA system overview. Rebuilt for legibility: input pills (left) ->
encoders -> the AR world model with its TWO passes drawn as numbered ①→② arrows ->
decoders -> output pills (right), plus the closed-loop re-encode back-edge and an
edge/symbol legend. wan_va/wan_va_server.py."""
from archscope import (Block, Diagram, GroupFrame, HStack, IOLabel, Swatches,
                       TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 1 · LingBot-VA — system: prompt + observation  ->  next video chunk + action chunk  (closed loop)",
    subtitle="Per autoregressive step the world model runs TWO passes: ① denoise the next video chunk, then "
             "② denoise the action chunk that realizes it (attending to the just-predicted video). Actions are "
             "executed, the real observation is re-encoded into the KV-cache, and the loop repeats.")

# ----- INPUT pills (left) ----------------------------------------------------------
ins = VStack([
    IOLabel("task prompt (str)", modality="text", id="i_p"),
    IOLabel("multi-view images (V,H,W,3)", modality="video", id="i_o"),
    IOLabel("robot state (16,)", modality="action", id="i_s"),
], gap=40, align="end")
d.place(ins, 60, 150)

# ----- encoders --------------------------------------------------------------------
enc = VStack([
    Block("UMT5 text enc", kind="cond", badge="frozen", sub="-> (B,512,4096)", id="e_t",
          modality="text", min_w=190),
    Block("Wan2.2 VAE enc", kind="vae", badge="frozen", sub="-> video latents", id="e_v",
          modality="video", min_w=190),
    Block("quantile normalize", kind="linear", sub="state -> (1,30,F,16,1)", id="e_a",
          modality="action", min_w=190),
], gap=28)
d.place(enc, 320, 138)
# i_o enters e_v high-left; the closed-loop re-encode (below) enters low-left — distinct ports
d.edge("i_p", "e_t"); d.edge("i_o", "e_v.l@0.32"); d.edge("i_s", "e_a")

# ----- the AR world model core: two passes ①→② -------------------------------------
core = GroupFrame(VStack([
    Block("② denoise ACTION chunk -> a_t", kind="model", modality="action",
          sub="inverse dynamics: attends the just-predicted video", id="p2", min_w=300),
    Block("① denoise next VIDEO chunk -> z_hat", kind="model", modality="video",
          sub="forward dynamics: flow matching, 25 steps", id="p1", min_w=300),
    Block("WanTransformer3DModel  ·  shared DiT", kind="model",
          sub="one network, both passes  (-> fig2/fig10)", id="net", min_w=300),
], gap=16), title="autoregressive world model  ·  per chunk (K=4 frames)", title_pos="tag",
    dashed=False, stroke="#475569", tint="rgba(71,85,105,0.04)", id="core", pad=15)
d.place(core, 600, 120)
# all three encoders feed the world model. Land them on the tall core FRAME's lower-left,
# well separated — the short 'net' block alone can't hold 3 ports without a pile-up — and
# route through the clear gap between columns so none crosses an encoder box. text is pink.
gx = (d.box("e_v").x2 + d.box("core").x) / 2
ct = d.box("core")
d.edge("e_t", "core.l@0.55", a_side="r", b_side="l",
       via=[(gx - 7, d.box("e_t").cy), (gx - 7, ct.y + 0.55 * ct.h)],
       style_name="cond", color="#DB2777")
d.edge("e_v", "core.l@0.72", a_side="r", b_side="l",
       via=[(gx + 9, d.box("e_v").cy), (gx + 9, ct.y + 0.72 * ct.h)])
d.edge("e_a", "core.l@0.9", a_side="r", b_side="l")
# the ①→② numbered dependency (the crux): video pass feeds the action pass
p1b, p2b = d.box("p1"), d.box("p2")
d.edge((p1b.x, p1b.cy), (p2b.x, p2b.cy), a_side="l", b_side="l",
       via=[(p1b.x - 26, p1b.cy), (p2b.x - 26, p2b.cy)], color="#0D9488", width=1.6,
       label="① z_hat conditions ②", label_side="left")

# KV-cache below the core
cb = d.box("core")
cache = Block("KV-cache pool", kind="mask", sub="clean history · model.py:331-409", id="cache",
              min_w=300)
cache.measure()
d.place(cache, cb.x + (cb.w - cache.w) / 2, cb.y2 + 30)
d.edge("cache.t@0.5", (cb.cx, cb.y2), b_side="b", style_name="cache", label="history KV")

# ----- decoders --------------------------------------------------------------------
dec = VStack([
    Block("Wan VAE dec", kind="vae", badge="frozen", sub="latents -> frames", id="d_v",
          modality="video", min_w=180),
    Block("denormalize", kind="linear", sub="-> 16 dims, physical", id="d_a",
          modality="action", min_w=180),
], gap=44)
d.place(dec, cb.x2 + 70, cb.cy - 60)
d.edge("p1.r@0.5", "d_v.l@0.5", a_side="r", label="z_hat")
d.edge("p2.r@0.5", "d_a.l@0.5", a_side="r", label="a_t")

# ----- OUTPUT pills (right) ---------------------------------------------------------
outs = VStack([
    IOLabel("imagined future video", modality="video", id="o_v"),
    IOLabel("action chunk (16,K·16) @50Hz", modality="action", id="o_a"),
], gap=44)
d.place(outs, d.box("d_v").x2 + 50, d.box("d_v").y - 4)
d.edge("d_v", "o_v"); d.edge("d_a", "o_a")

# ----- robot + closed loop ----------------------------------------------------------
rob = Block("robot / environment", kind="io", sub="execute actions · render new obs", id="rob",
            min_w=240)
rob.measure()
roby = max(d.box("o_a").y2, d.box("cache").y2) + 40
d.place(rob, d.box("o_a").x, roby)
d.edge("o_a.b@0.5", "rob.t@0.5", b_side="t", style_name="main")
ev = d.box("e_v")
low = roby + rob.h + 26
lx = ev.x - 34          # up the clear channel LEFT of the encoder column (not through e_a)
d.edge((d.box("rob").x, d.box("rob").cy), "e_v.l@0.72", a_side="l", b_side="l",
       via=[(d.box("rob").x - 30, d.box("rob").cy), (d.box("rob").x - 30, low),
            (lx, low), (lx, ev.y + 0.72 * ev.h)], style_name="cache", color="#B45309",
       label="closed loop: re-encode the real observation", label_at=0.5, label_dy=-2)

leg = Swatches([("video", "video"), ("action", "action"), ("text", "text"),
                ("vae", "VAE (frozen)"), ("model", "DiT"), ("mask", "cache"),
                ("main", "data flow", "edge"), ("cond", "text cond", "edge"),
                ("cache", "KV-cache / loop", "edge")], max_w=820, id="leg")
d.place(leg, 60, 95)

d.save(OUT / "fig01_system.svg")
print("ok")
