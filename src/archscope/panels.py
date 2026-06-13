"""Training-method panel primitives: token rows, mask grids, legends, captions."""
from . import style
from .node import Element
from .text import measure


class TokenRow(Element):
    """A row of token cells.

    cells: list of dicts:
      label, modality ('video'|'action'|'text'|'state'|'none'),
      hatch (noisy), bold_border, sub (small text under cell), gap_after (px)
    """

    def __init__(self, cells, cell_w=30, cell_h=26, gap=3.5, id=None,
                 label=None, label_w=0):
        self.cells, self.cw, self.ch, self.gap = cells, cell_w, cell_h, gap
        self.id, self.label, self.label_w = id, label, label_w

    def measure(self):
        self.w = self.label_w + sum(
            c.get("w", self.cw) + self.gap + c.get("gap_after", 0)
            for c in self.cells) - self.gap
        has_sub = any(c.get("sub") for c in self.cells)
        self.h = self.ch + (14 if has_sub else 0)
        return self.w, self.h

    def render(self, d, x, y):
        if self.label:
            d.doc.text("nodes", x + self.label_w - 10,
                       y + self.ch / 2 + style.T_SUB * 0.36, self.label,
                       style.T_SUB + 0.5, style.MUTED, anchor="end")
        cx = x + self.label_w
        for c in self.cells:
            w = c.get("w", self.cw)
            fill, stroke, tc = style.MODALITY[c.get("modality", "none")]
            # explicit per-cell color override (e.g. image-patch tokens)
            fill = c.get("fill", fill)
            stroke = c.get("stroke", stroke)
            tc = c.get("tcolor", tc)
            if c.get("img"):
                from .node import RasterImage
                RasterImage.emit(d.doc, c["img"], cx, y, w, self.ch, rx=4.5)
                d.doc.rect("nodes", cx, y, w, self.ch, "none", stroke,
                           1.8 if c.get("bold_border") else 1.1, rx=4.5)
            else:
                f = (f'url(#{d.doc.hatch(fill, stroke)})' if c.get("hatch")
                     else fill)
                d.doc.rect("nodes", cx, y, w, self.ch, f, stroke,
                           1.8 if c.get("bold_border") else 1.1, rx=4.5)
            if c.get("label"):
                d.doc.text("nodes", cx + w / 2, y + self.ch / 2 + 3.4,
                           c["label"], style.T_SUB + 0.5, tc, "500",
                           halo=bool(c.get("hatch")))
            if c.get("sub"):
                d.doc.text("nodes", cx + w / 2, y + self.ch + 11, c["sub"],
                           style.T_TINY, style.FAINT, mono=True)
            cx += w + self.gap + c.get("gap_after", 0)
        return self._reg(d, x, y)


class PatchGrid(Element):
    """An image drawn as a grid of colored patch-cells — the 'an image is a
    sequence of patch tokens' idiom. `cells` is a 2D list of fill colors (hex)
    or None (= background). Flatten in raster order to get the token sequence."""

    def __init__(self, cells, cell=18, gap=1.5, id=None, bg="#F8FAFC",
                 stroke="#FFFFFF", outline=None):
        self.cells, self.cell, self.gap = cells, cell, gap
        self.id, self.bg, self.stroke, self.outline = id, bg, stroke, outline

    def measure(self):
        rows = len(self.cells)
        cols = max(len(r) for r in self.cells)
        self.w = cols * (self.cell + self.gap) - self.gap
        self.h = rows * (self.cell + self.gap) - self.gap
        return self.w, self.h

    def render(self, d, x, y):
        p = self.cell + self.gap
        if self.outline:
            d.doc.rect("frames", x - 3, y - 3, self.w + 6, self.h + 6,
                       "none", self.outline, 1.2, rx=4)
        for r, row in enumerate(self.cells):
            for c, col in enumerate(row):
                d.doc.rect("nodes", x + c * p, y + r * p, self.cell, self.cell,
                           col or self.bg, self.stroke, 0.8, rx=2)
        return self._reg(d, x, y)


class MaskGrid(Element):
    """Attention-mask matrix. cells[i][j] = class key into `colors`;
    colors: {key: (fill, legend_label)}. Row/col labels are mono.
    seps: indices after which to draw a thicker separator line."""

    def __init__(self, row_labels, col_labels, cells, colors, cell=20,
                 id=None, seps=(), q_title="Query", k_title="Key / Value",
                 col_label_style=None):
        self.rl, self.cl, self.cells, self.colors = (row_labels, col_labels,
                                                     cells, colors)
        self.cell, self.id, self.seps = cell, id, set(seps)
        self.qt, self.kt = q_title, k_title

    def measure(self):
        self.lw = max(measure(s, style.T_TINY + 0.5, mono=True)[0]
                      for s in self.rl) + 12 + 18  # +18 for rotated Q title
        self.th = 30 + 14  # k-title + col labels (rotated? keep horizontal short)
        n, m = len(self.rl), len(self.cl)
        self.w = self.lw + m * self.cell + 4
        self.h = self.th + n * self.cell + 4
        return self.w, self.h

    def render(self, d, x, y):
        c = self.cell
        gx, gy = x + self.lw, y + self.th
        n, m = len(self.rl), len(self.cl)
        # titles
        d.doc.text("nodes", gx + m * c / 2, y + 10, self.kt,
                   style.T_TINY + 1, style.MUTED, "600")
        # column labels
        for j, s in enumerate(self.cl):
            d.doc.text("nodes", gx + j * c + c / 2, y + self.th - 5, s,
                       style.T_TINY + 0.5, style.MUTED, mono=True)
        # row labels + cells
        for i, s in enumerate(self.rl):
            d.doc.text("nodes", gx - 7, gy + i * c + c / 2 + 3.2, s,
                       style.T_TINY + 0.5, style.MUTED, anchor="end",
                       mono=True)
            for j in range(m):
                key = self.cells[i][j]
                fill = self.colors[key][0] if key else "#FFFFFF"
                d.doc.rect("nodes", gx + j * c, gy + i * c, c - 1.4, c - 1.4,
                           fill, "#CBD5E1", 0.7, rx=2.5)
        # separators
        for s in self.seps:
            d.doc.line("nodes", gx + s * c - 0.7, gy - 2,
                       gx + s * c - 0.7, gy + n * c, "#475569", 1.3)
            d.doc.line("nodes", gx - 2, gy + s * c - 0.7,
                       gx + m * c, gy + s * c - 0.7, "#475569", 1.3)
        # q title (left, rotated)
        qx, qy = x + 8, gy + n * c / 2
        d.doc.raw("nodes",
                  f'<text x="{qx:.1f}" y="{qy:.1f}" font-family="{style.FONT_SANS}" '
                  f'font-size="{style.T_TINY + 1}px" fill="{style.MUTED}" '
                  f'font-weight="600" text-anchor="middle" '
                  f'transform="rotate(-90 {qx:.1f} {qy:.1f})">{self.qt}</text>')
        return self._reg(d, x, y)


class Swatches(Element):
    """Inline legend: colored swatch + label pairs.

    entries: list of (kind_or_color, label[, opts]) where kind_or_color is a
    key of style.KIND / style.MODALITY, or a raw (fill, stroke) tuple.
    opts: 'hatch' to hatch the swatch, 'dash' for dashed border.
    """

    def __init__(self, entries, gap=16, id=None, swatch=13, size=None,
                 max_w=None, line_gap=8):
        self.entries, self.gap, self.id, self.sw = entries, gap, id, swatch
        self.size = size or style.T_SUB + 1
        self.max_w, self.line_gap = max_w, line_gap

    def _resolve(self, key):
        if isinstance(key, tuple):
            return key[0], key[1]
        if key in style.KIND:
            f, s, _ = style.KIND[key]
            return f, s
        f, s, _ = style.MODALITY[key]
        return f, s

    def measure(self):
        self.items = []
        for e in self.entries:
            key, label = e[0], e[1]
            opts = e[2] if len(e) > 2 else ""
            w = self.sw + 6 + measure(label, self.size)[0]
            self.items.append((key, label, opts, w))
        # simple wrap into lines
        self.lines, cur, cw = [], [], 0.0
        limit = self.max_w or float("inf")
        for it in self.items:
            if cur and cw + it[3] > limit:
                self.lines.append(cur)
                cur, cw = [], 0.0
            cur.append(it)
            cw += it[3] + self.gap
        if cur:
            self.lines.append(cur)
        self.w = max(sum(it[3] for it in ln) + self.gap * (len(ln) - 1)
                     for ln in self.lines)
        self.h = len(self.lines) * (self.sw + self.line_gap) - self.line_gap + 2
        return self.w, self.h

    def render(self, d, x, y):
        yy = y
        for ln in self.lines:
            xx = x
            for key, label, opts, w in ln:
                fill, stroke = self._resolve(key)
                if "hatch" in opts:
                    fill = f'url(#{d.doc.hatch(fill, stroke)})'
                d.doc.rect("nodes", xx, yy, self.sw, self.sw, fill, stroke,
                           1.1, rx=3.5, dash="3 2" if "dash" in opts else None)
                d.doc.text("nodes", xx + self.sw + 6,
                           yy + self.sw / 2 + self.size * 0.36, label,
                           self.size, style.MUTED, anchor="start")
                xx += w + self.gap
            yy += self.sw + self.line_gap
        return self._reg(d, x, y)
