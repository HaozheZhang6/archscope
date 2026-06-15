"""Fig 5 — the FlexAttention mask that does all the modality/causality routing.

The grid cells are COMPUTED by re-implementing the three mask rules from
model.py:154-201 verbatim, so the figure cannot drift from the code.
"""
from archscope import (Diagram, Formula, GroupFrame, MaskGrid, Swatches,
                       TextLabel, TokenRow, VStack, style)
from common import OUT

d = Diagram(
    title="Fig 5 · One attention mask does the routing (code version)",
    subtitle="Sequence layout [noisy z | clean z | noisy a | clean a], frame ids interleaved "
             "as fid_z = 2·(f//K), fid_a = 2·(f//K)+1 — actions of chunk c sit strictly between "
             "video chunks c and c+1. Example: F=4 frames, chunk K=2, 1 token per frame. "
             "wan_va/modules/model.py:93-201")

F, K = 4, 2

# --- token bookkeeping, exactly as init_mask() builds it -------------------------
tokens = []   # (name, fid, nid, modality, hatch)
for f in range(F):
    tokens.append((f"~z{f}", (f // K) * 2, 0, "video", True))
for f in range(F):
    tokens.append((f"z{f}", (f // K) * 2, 1, "video", False))
for f in range(F):
    tokens.append((f"~a{f}", (f // K) * 2 + 1, 0, "action", True))
for f in range(F):
    tokens.append((f"a{f}", (f // K) * 2 + 1, 1, "action", False))

def rule(q, kv):
    """Returns which rule admits attention from query q to key kv (or None)."""
    _, fq, nq, _, _ = q
    _, fk, nk, _, _ = kv
    if nq == 1 and nk == 1 and fk <= fq:
        return "cc"        # clean -> clean, block-causal (incl. self chunk)
    if nq == 0 and nk == 1 and fk < fq:
        return "nc"        # noise -> clean, strictly past
    if nq == 0 and nk == 0 and fk == fq:
        return "nn"        # noise -> noise, same chunk only
    return None

cells = [[rule(q, kv) for kv in tokens] for q in tokens]
labels = [t[0] for t in tokens]

# rule-type palette chosen to AVOID the modality strip's blue (video) / green (action)
# below — otherwise a green rule cell reads as an action token. Amber / violet / rose
# are distinct from each other and from the modality colours.
colors = {
    "cc": ("#FDE68A", "clean → clean   (fid_kv ≤ fid_q): causal history"),
    "nn": ("#DDD6FE", "noise → noise   (fid_kv = fid_q): bidirectional within own chunk"),
    "nc": ("#FBCFE8", "noise → clean   (fid_kv < fid_q): denoise from strictly-past context"),
}

grid = MaskGrid(labels, labels, cells, colors, cell=21, seps=(F, 2 * F, 3 * F),
                id="grid")
d.place(grid, 60, 70)

# --- token strip with fid / nid annotations --------------------------------------
strip = TokenRow(
    [dict(label=t[0], modality=t[3], hatch=t[4], sub=f"fid {t[1]}",
          gap_after=10 if i in (F - 1, 2 * F - 1, 3 * F - 1) else 0)
     for i, t in enumerate(tokens)],
    cell_w=30, cell_h=26, id="strip")
gb = d.box("grid")
strip.measure()
d.place(strip, 60 + 36, gb.y2 + 36)
d.note(60 + 36, gb.y2 + 26, "the same 16 tokens as a sequence (hatched / ~ = noisy, being denoised):",
       size=style.T_SUB + 0.5, color=style.FAINT)

# --- right column: rules + consequences -------------------------------------------
right = VStack([
    TextLabel("Three OR-ed rules (model.py:195-197)", size=style.T_SUB + 2,
              color=style.INK, weight="600"),
    Swatches([(("#FDE68A", "#CA8A04"), colors["cc"][1]),
              (("#FBCFE8", "#DB2777"), colors["nc"][1]),
              (("#DDD6FE", "#7C3AED"), colors["nn"][1])],
             max_w=380, gap=12),
    TextLabel("then AND same-batch (packed sequences) and AND "
              "|fid_q − fid_kv| ≤ window (sliding window, randomized 4–64 at training).",
              size=style.T_SUB + 1, color=style.MUTED, max_w=380, anchor="start"),
    TextLabel("What the interleave buys (read off the grid):", size=style.T_SUB + 2,
              color=style.INK, weight="600"),
    TextLabel("· inverse dynamics — noisy ~a of chunk c attends clean z of chunks ≤ c "
              "(its own observations) but never future video;",
              size=style.T_SUB + 1, color=style.MUTED, max_w=380, anchor="start"),
    TextLabel("· forward dynamics — noisy ~z of chunk c+1 attends clean a of chunk c "
              "(the just-executed actions);",
              size=style.T_SUB + 1, color=style.MUTED, max_w=380, anchor="start"),
    TextLabel("· noisy video and noisy action never attend each other "
              "(fids differ by construction) — denoising targets stay independent given "
              "clean context;",
              size=style.T_SUB + 1, color=style.MUTED, max_w=380, anchor="start"),
    TextLabel("· within a chunk, noisy tokens are bidirectional (parallel generation), "
              "across chunks strictly causal — the paper's “causality without giving up "
              "chunk parallelism”.",
              size=style.T_SUB + 1, color=style.MUTED, max_w=380, anchor="start"),
], gap=13, align="start")
d.place(right, gb.x2 + 44, 90)

d.save(OUT / "fig05_mask.svg")
print("ok")
