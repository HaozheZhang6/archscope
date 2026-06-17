"""Wan2.1-VACE-1.3B · 12 · VACE, illustrated on a frame (companion to the structure, figs 8–11).
Concrete control: a video + a mask that says KEEP here / REGENERATE there. The two halves are
VAE-encoded and stacked with the mask into a 96-ch control, the VACE branch turns it into hints,
and the main DiT generates the masked region to match the kept region. Frames: procedural CC0."""
from archscope import Block, Diagram, IOLabel, OpDot, RasterImage, Swatches, TextLabel, style
from common import OUT
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "lingbot_va" / "assets" / "motion"
IM = 78

d = Diagram(
    title="Wan2.1-VACE-1.3B · 12 · VACE, illustrated — keep one region, regenerate another, guided by control",
    subtitle="The mask marks a region to REGENERATE (here, the sun) while the rest is KEPT. keep = video·(1−mask) "
             "and edit = video·mask are VAE-encoded (16+16) and stacked with the patchified mask (64) → a 96-ch "
             "control (fig 8). The VACE branch turns it into 15 hints (figs 9,10) that steer the main DiT (fig 11).")


def img(path, x, y, label, lcol=style.MUTED, border="#94A3B8"):
    RasterImage.emit(d.doc, str(M / path), x, y, IM, IM, rx=5)
    d.doc.rect("nodes", x, y, IM, IM, "none", border, 1.4, rx=5)
    d.doc.text("nodes", x + IM / 2, y + IM + 13, label, style.T_TINY + 1, lcol, "600")
    return x + IM


# ---------------- stage 1: the control inputs ----------------
Y = 210
img("vace_ctrl.png", 70, Y - 50, "control video", "#0C4A6E", "#0284C7")
img("vace_mask.png", 70, Y + 52, "mask (white = regenerate)", "#831843", "#DB2777")
d.note(70, Y - 70, "what you give VACE:", size=style.T_SUB + 1, color=style.INK, weight="600")

# ---------------- stage 2: the keep / edit split ----------------
SX = 250
img("vace_inactive.png", SX, Y - 50, "keep = video·(1−mask) → 16", "#166534", "#16A34A")
img("vace_reactive.png", SX, Y + 52, "edit = video·mask → 16", "#7C2D12", "#EA580C")
mbox = Block("mask → 64 ch", kind="linear", sub="8×8 patchify", id="m64", min_w=120)
d.place(mbox, SX, Y + 52 + IM + 26)   # below the edit frame so the fan-in to ctx never crosses it
d.edge((70 + IM, Y - 50 + IM / 2), (SX, Y - 50 + IM / 2), a_side="r", b_side="l", style_name="main")
d.edge((70 + IM, Y + 52 + IM / 2), (SX, Y + 52 + IM / 2), a_side="r", b_side="l", style_name="main")
d.edge((70 + IM, Y + 52 + IM - 6), "m64.l@0.5", b_side="l", style_name="faint", color="#DB2777")

# ---------------- stage 3: concat -> 96-ch control ----------------
cat = Block("vace_context\n96 ch", kind="io", modality="state", sub="16 + 16 + 64", id="ctx",
            min_w=130, h=92)
cat.measure()
d.place(cat, SX + IM + 200, Y - 6)
d.edge((SX + IM, Y - 50 + IM / 2), "ctx.l@0.25", a_side="r", b_side="l", style_name="main",
       label="16 keep", label_bg=True)
d.edge((SX + IM, Y + 52 + IM / 2), "ctx.l@0.6", a_side="r", b_side="l", style_name="main",
       label="16 edit", label_bg=True)
d.edge("m64.r@0.5", "ctx.l@0.85", a_side="r", b_side="l", style_name="faint", color="#DB2777",
       label="64 mask", label_bg=True)

# ---------------- stage 4: VACE branch -> hints -> main DiT ----------------
vb = Block("VACE branch\n→ 15 hints", kind="trainable", sub="figs 9, 10", id="vb", min_w=130, h=92)
d.place(vb, d.box("ctx").x2 + 56, d.box("ctx").y)
d.edge("ctx.r@0.5", "vb.l@0.5", a_side="r", style_name="main")
dit = Block("main Wan DiT\n+ hints (× scale)", kind="model", sub="fig 11", id="dit", min_w=140, h=92)
d.place(dit, d.box("vb").x2 + 56, d.box("vb").y)
d.edge("vb.r@0.5", "dit.l@0.5", a_side="r", style_name="cache", color="#B45309",
       label="hints", label_bg=True)

# ---------------- stage 5: the output ----------------
ox = d.box("dit").x2 + 56
RasterImage.emit(d.doc, str(M / "vace_ctrl.png"), ox, d.box("dit").cy - IM / 2, IM, IM, rx=5)
d.doc.rect("nodes", ox, d.box("dit").cy - IM / 2, IM, IM, "none", "#16A34A", 1.8, rx=5)
d.doc.text("nodes", ox + IM / 2, d.box("dit").cy + IM / 2 + 13, "generated video", style.T_TINY + 1,
           "#166534", "600")
d.doc.text("nodes", ox + IM / 2, d.box("dit").cy + IM / 2 + 25, "(edit region filled to match)",
           style.T_TINY + 1, style.FAINT, "500")
d.edge("dit.r@0.5", (ox, d.box("dit").cy), a_side="r", b_side="l", style_name="main")

d.note(70, d.box("m64").y2 + 22,
       "Read it as inpainting in motion: VACE preserves the unmasked pixels exactly and fills the masked region "
       "with new content that is consistent with what's kept (and with the text prompt). Swap the mask/control and "
       "the same machinery does pose-to-video, depth-to-video, outpainting, first-frame control, etc.",
       size=style.T_SUB, color=style.MUTED, max_w=1000)

leg = Swatches([(("#FFFFFF", "#0284C7"), "control / kept"), (("#FFFFFF", "#DB2777"), "mask"),
                (("#FFFFFF", "#EA580C"), "edit region"), ("state", "96-ch control"),
                ("trainable", "VACE branch"), ("model", "main DiT"),
                ("cache", "hints", "edge")], max_w=1000, id="leg")
d.place(leg, 70, 96)

d.save(OUT / "fig12_vace_illustration.svg")
print("ok")
