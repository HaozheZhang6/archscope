# Contributing

Contributions are welcome — new example figures are the most valuable kind.

## Setup

```bash
git clone https://github.com/HaozheZhang6/archscope && cd archscope
pip install -e .            # only dependency: matplotlib
brew install resvg          # optional, for PNG export (SVG always works)
python examples/quickstart.py
```

## Adding an example figure

1. Put the script in `examples/<name>.py` (or `examples/<model>/` for a multi-figure
   case study); outputs go to `out/` and are committed.
2. Follow the authoring rules (README): **verify before drawing** — every structural
   claim needs a `file:line` (code you read) or `§/page` (paper) annotation; if a
   figure encodes rules (masks, schedules), implement the rules in the script and
   generate the cells.
3. Render and **look at the output** before opening a PR. Overlapping boxes, labels
   on top of edges, and arrows that meet boxes at an angle are bugs.
4. Glyph safety: subscript digits (₀₁₂), Greek, arrows, ≤ ∈ ≈ are fine; subscript
   letters (ₜ) and combining accents (ẑ, z̃, v̂) are not — use `~z`, `z_t`, or a
   `Formula`. Check with `resvg fig.svg /tmp/x.png 2>&1 | grep -i lastresort`.

## Library changes

Keep the dependency footprint at "matplotlib only". New visual primitives should
earn their place by being needed in at least two figures. CI just renders every
example — make sure `python examples/*.py` passes.

## Conduct

Be kind; assume good faith. Disagreements about figures are settled by reading the
source being drawn.
