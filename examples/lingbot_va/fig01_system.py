"""Fig 1 — LingBot-VA system overview (code version, inference server)."""
from archscope import (Block, Diagram, GroupFrame, HStack, IOLabel, Spacer,
                       Swatches, TextLabel, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 1 · LingBot-VA — system overview (released inference stack)",
    subtitle="An autoregressive video-action world model: predict the next video chunk, decode "
             "the actions that realize it, execute, re-encode real observations, repeat. "
             "wan_va/wan_va_server.py")

# ----- inputs ----------------------------------------------------------------------
ins = VStack([
    IOLabel("task prompt (str)", modality="text", id="i_p"),
    IOLabel("multi-view images (V, H, W, 3)", modality="video", id="i_o"),
    IOLabel("robot state (16,)", modality="action", id="i_s"),
], gap=46, align="end")
d.place(ins, 40, 120)

# ----- encoders ---------------------------------------------------------------------
encs = VStack([
    Block("UMT5 text encoder", kind="cond", sub="frozen · → (B, 512, 4096)",
          src="umt5-xxl", id="e_t", badge="frozen", modality="text"),
    Block("Wan2.2 causal VAE (enc)", kind="vae",
          sub="frozen · streaming · 16× spatial → (1, 48, F, H/16, V·W/16)",
          id="e_v", badge="frozen", modality="video"),
    Block("quantile normalize", kind="linear", sub="per-dim q01–q99 → [-1,1] · pad 16→30 dims",
          id="e_a", modality="action"),
], gap=32)
d.place(encs, 230, 100)
d.edge("i_p", "e_t")
d.edge("i_o", "e_v")
d.edge("i_s", "e_a")

# ----- core model -------------------------------------------------------------------
core = GroupFrame(VStack([
    Block("WanTransformer3DModel", kind="model",
          sub="≈5B · 30 layers · d=3072 · joint video-action sequence",
          src="→ Fig 2 (dataflow) · Fig 3 (block) · Fig 5 (mask)", id="m", min_w=330),
    HStack([
        Block("video denoise", kind="mask", sub="FlowMatch shift=5.0 · 25 steps (cfg 5.0)",
              id="s_v", modality="video"),
        Block("action denoise", kind="mask", sub="FlowMatch shift=1.0 · 50 steps (cfg 1.0)",
              id="s_a", modality="action"),
    ], gap=18),
    TextLabel("two passes per chunk: ① denoise next video latents (KV cache holds history) "
              "② denoise the action chunk attending to the fresh video prediction. "
              "Paper deployment: 3 video steps to s=0.6, 10 action steps.",
              size=style.T_SUB + 0.5, color=style.MUTED, max_w=480),
], gap=16), title="autoregressive world model (per chunk of K=4 frames)",
    dashed=False, stroke="#94A3B8", id="core", pad=18)
d.place(core, 660, 110)
d.edge("e_t", "core.l@0.2", style_name="cond", color="#DB2777",
       label="(B,512,4096)", label_at=0.5, label_dy=-2)
d.edge("e_v", "core.l@0.5", label="(1,48,F,h,w)", label_at=0.5, label_dy=-2)
d.edge("e_a", "core.l@0.8", label="(1,30,F,16,1)", label_at=0.5, label_dy=-2)

# ----- KV cache --------------------------------------------------------------------
cache = Block("KV cache pool", kind="mask",
              sub="per-layer slots · window×(tokens/chunk) · LRU · real obs replace predictions",
              src="model.py:331-409 · “closed-loop rollout”", id="cache", min_w=330)
cb = d.box("core")
cache.measure()
d.place(cache, cb.x + (cb.w - cache.w) / 2, cb.y2 + 34)
d.edge("cache.t", (cb.cx, cb.y2), b_side="b", style_name="cache", label="history KV (clean)")

# ----- outputs ----------------------------------------------------------------------
outs = VStack([
    Block("Wan VAE (dec)", kind="vae", sub="latents → video frames", id="o_v",
          badge="frozen", modality="video"),
    Block("denormalize + unpad", kind="linear", sub="30 → 16 dims · q01–q99 → physical",
          id="o_a", modality="action"),
], gap=44)
ob = d.box("core")
d.place(outs, ob.x2 + 95, 150)
d.edge((ob.x2, ob.y + 70), "o_v.l", a_side="r", label="next-chunk latents")
d.edge((ob.x2, ob.y + 130), "o_a.l", a_side="r", label="actions (30, K, 16)", label_at=0.6)

fin = VStack([
    IOLabel("imagined future video", modality="video", id="f_v"),
    Spacer(0, 24),
    IOLabel("action chunk (16, K·16) @50Hz", modality="action", id="f_a"),
], gap=20, align="start")
d.place(fin, d.box("o_v").x2 + 70, 160)
d.edge("o_v", "f_v")
d.edge("o_a", "f_a")

# ----- closed loop ------------------------------------------------------------------
rob = Block("robot / environment", kind="io", sub="execute actions · render new observations",
            id="rob", min_w=240)
rob.measure()
d.place(rob, ob.x2 + 95, d.box("cache").cy - rob.h / 2)
d.edge("f_a.b", "rob.r@0.5", style_name="main")
rb = d.box("rob")
ev = d.box("e_v")
low_y = max(d.box("cache").y2, rb.y2) + 34
d.edge((rb.cx, rb.y2), (ev.cx, ev.y2), style_name="cache", color="#B45309",
       via=[(rb.cx, low_y), (ev.cx, low_y)],
       label="real observations re-encoded every chunk (closed-loop rollout)",
       label_at=0.5, label_dy=-2)

leg = Swatches([("video", "video"), ("action", "action"), ("text", "text"),
                ("vae", "VAE/conv"), ("cond", "conditioning"), ("model", "transformer"),
                ("mask", "scheduler/cache")], max_w=300, id="leg")
d.place(leg, 40, 380)

d.save(OUT / "fig01_system.svg")
print("ok")
