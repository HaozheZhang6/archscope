"""Training-method storyboards: GPT next-token vs BERT masked-LM, side by side.

Shows: TokenRow (sequences, masking, highlights), MaskGrid with cells COMPUTED
from the actual attention rule, Formula, panel composition.
"""
from pathlib import Path

from archscope import (Diagram, Formula, GroupFrame, HStack, MaskGrid,
                       Swatches, TextLabel, TokenRow, VStack, style)

OUT = Path(__file__).resolve().parents[1] / "out" / "examples"

d = Diagram(
    title="Training methods as storyboards",
    subtitle="Two objectives on the same sentence. GPT predicts the next token under a "
             "causal mask (a position sees only the past); BERT masks 15% of tokens and "
             "predicts them from the whole bidirectional context.")

toks = ["the", "cat", "sat", "on", "the", "mat"]

# ---------------- GPT: next-token prediction ----------------------------------------
causal = MaskGrid(toks, toks,
                  [["a" if j <= i else None for j in range(6)] for i in range(6)],
                  colors={"a": ("#FDE68A", "visible")}, cell=19, id="cm",
                  q_title="query", k_title="key / value")

gpt = GroupFrame(VStack([
    TextLabel("input — every position predicts the NEXT token:",
              size=style.T_SUB + 1, color=style.MUTED, anchor="start"),
    TokenRow([dict(label=t, modality="video", w=44) for t in toks], cell_h=24),
    TokenRow([dict(label=t, modality="action", w=44,
                   sub="target" if i == 5 else None)
              for i, t in enumerate(toks[1:] + ["[?]"])], cell_h=24),
    TextLabel("causal attention (cells computed from the rule  j ≤ i):",
              size=style.T_SUB + 1, color=style.MUTED, anchor="start"),
    causal,
    Formula(r"$\mathcal{L}=-\sum_i \log p_\theta(x_{i+1}\mid x_{\leq i})$", size=12.5),
], gap=12), title="GPT — next-token prediction", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", pad=18)

# ---------------- BERT: masked LM ----------------------------------------------------
masked = {1, 4}
# bidirectional: every cell is allowed (mirror of the causal triangle, computed the
# same way) — also balances the two panels to equal height.
bidir = MaskGrid(toks, toks, [["a" for _ in range(6)] for _ in range(6)],
                 colors={"a": ("#FDE68A", "visible")}, cell=19, id="bm",
                 q_title="query", k_title="key / value")
bert = GroupFrame(VStack([
    TextLabel("input — 15% of tokens replaced by [M] = [MASK]:",
              size=style.T_SUB + 1, color=style.MUTED, anchor="start"),
    TokenRow([dict(label="[M]" if i in masked else t,
                   modality="none" if i in masked else "video",
                   hatch=i in masked, bold_border=i in masked, w=44)
              for i, t in enumerate(toks)], cell_h=24),
    TokenRow([dict(label=toks[i] if i in masked else "·",
                   modality="action" if i in masked else "none",
                   sub="predict" if i in masked else None, w=44)
              for i in range(6)], cell_h=24),
    TextLabel("bidirectional attention — every token sees every token (loss only at "
              "the masked positions):",
              size=style.T_SUB + 1, color=style.MUTED, max_w=300, anchor="start"),
    bidir,
    Formula(r"$\mathcal{L}=-\sum_{i\in M} \log p_\theta(x_i\mid x_{\setminus M})$",
            size=12.5),
], gap=12), title="BERT — masked language modeling", dashed=False,
    tint="#FFFFFF", stroke="#CBD5E1", pad=18)

row = HStack([gpt, bert], gap=30, align="start")
d.place(row, 40, 60)

leg = Swatches([("video", "given token"), ("action", "prediction target"),
                (("#F1F5F9", "#94A3B8"), "[MASK] (hatched)", "hatch"),
                (("#FDE68A", "#CA8A04"), "attention allowed")], max_w=620, id="leg")
d.place(leg, 40, 20)

d.save(OUT / "training_methods.svg")
print("wrote", OUT / "training_methods.svg")
