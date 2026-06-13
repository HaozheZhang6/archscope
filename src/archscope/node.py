"""Diagram elements: blocks, op glyphs, IO stadiums, known-concept chips."""
import base64
from pathlib import Path

from . import style
from .geom import Box
from .text import measure

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".webp": "image/webp"}


class Element:
    """Measurable, renderable unit. Subclasses set self.w/self.h in measure()."""
    id: str | None = None
    w = 0.0
    h = 0.0

    def measure(self) -> tuple[float, float]:
        raise NotImplementedError

    def render(self, d, x, y):
        """d is the Diagram. Must register anchors for id'd elements."""
        raise NotImplementedError

    def _reg(self, d, x, y):
        box = Box(x, y, self.w, self.h)
        if self.id:
            d.anchors[self.id] = box
        return box


class RasterImage(Element):
    """Embed a raster image (PNG/JPG) as a self-contained base64 data URI — for
    figures that use real picture content (e.g. an image tokenized into patches).
    Reads file bytes only; no image library needed at render time."""

    def __init__(self, path, w, h=None, id=None, rx=0, outline=None, ow=1.4):
        self.path, self._w, self._h = Path(path), w, (h or w)
        self.id, self.rx, self.outline, self.ow = id or None, rx, outline, ow

    def measure(self):
        self.w, self.h = self._w, self._h
        return self.w, self.h

    @staticmethod
    def emit(doc, path, x, y, w, h, rx=0, layer="nodes"):
        """Write one base64 <image> into `layer`; shared by RasterImage and
        any element that wants real picture content (e.g. TokenRow img cells)."""
        path = Path(path)
        data = base64.b64encode(path.read_bytes()).decode()
        mime = _MIME.get(path.suffix.lower(), "image/png")
        clip = ""
        if rx:
            cid = f"clip-{abs(hash((x, y, w, h, rx))) % (10**8)}"
            doc.defs[cid] = (f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}"'
                             f' width="{w:.1f}" height="{h:.1f}" rx="{rx}"/></clipPath>')
            clip = f' clip-path="url(#{cid})"'
        doc.raw(layer,
                f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'preserveAspectRatio="xMidYMid slice"{clip} '
                f'href="data:{mime};base64,{data}"/>', bbox=(x, y, w, h))

    def render(self, d, x, y):
        RasterImage.emit(d.doc, self.path, x, y, self.w, self.h, self.rx)
        if self.outline:
            d.doc.rect("nodes", x, y, self.w, self.h, "none", self.outline,
                       self.ow, rx=self.rx)
        return self._reg(d, x, y)


class Element:
    """Measurable, renderable unit. Subclasses set self.w/self.h in measure()."""
    id: str | None = None
    w = 0.0
    h = 0.0

    def measure(self) -> tuple[float, float]:
        raise NotImplementedError

    def render(self, d, x, y):
        """d is the Diagram. Must register anchors for id'd elements."""
        raise NotImplementedError

    def _reg(self, d, x, y):
        box = Box(x, y, self.w, self.h)
        if self.id:
            d.anchors[self.id] = box
        return box


class Block(Element):
    """Rounded component box: label + optional sublabel + optional source line."""

    def __init__(self, label, kind="io", sub=None, src=None, id=None,
                 min_w=0, h=None, modality=None, badge=None, text_size=None):
        self.label, self.kind, self.sub, self.src = label, kind, sub, src
        self.id = id or None
        self.min_w, self.fixed_h = min_w, h
        self.modality, self.badge = modality, badge
        self.text_size = text_size or style.T_LABEL

    def measure(self):
        lw = measure(self.label, self.text_size, "500")[0]
        sw = measure(self.sub, style.T_SUB, mono=True)[0] if self.sub else 0
        cw = measure(self.src, style.T_TINY, mono=True)[0] if self.src else 0
        bw = measure(self.badge, style.T_TINY, "500")[0] + 14 if self.badge else 0
        self.w = max(lw + bw, sw, cw, self.min_w) + 26
        n = 1 + bool(self.sub) + bool(self.src)
        self.h = self.fixed_h or (30 if n == 1 else (44 if n == 2 else 55))
        return self.w, self.h

    def render(self, d, x, y):
        fill, stroke, tc = style.KIND[self.kind]
        d.doc.rect("nodes", x, y, self.w, self.h, fill, stroke, 1.3, rx=7)
        if self.modality:
            mfill, mstroke, _ = style.MODALITY[self.modality]
            d.doc.rect("nodes", x + 1.2, y + 4, 3.2, self.h - 8, mstroke, rx=1.6)
        n = 1 + bool(self.sub) + bool(self.src)
        cy = y + self.h / 2
        if n == 1:
            ly = cy + self.text_size * 0.36
        elif n == 2:
            ly = y + self.h * 0.42 + self.text_size * 0.36
        else:
            ly = y + 18
        d.doc.text("nodes", x + self.w / 2, ly, self.label,
                   self.text_size, tc, "500")
        yy = ly
        if self.sub:
            yy += 13.5
            d.doc.text("nodes", x + self.w / 2, yy, self.sub,
                       style.T_SUB, style.MUTED, mono=True)
        if self.src:
            yy += 12
            d.doc.text("nodes", x + self.w / 2, yy, self.src,
                       style.T_TINY, style.FAINT, mono=True)
        if self.badge:
            bw = measure(self.badge, style.T_TINY, "500")[0] + 10
            d.doc.rect("labels", x + self.w - bw - 4, y + 3.5, bw, 13,
                       "#FFFFFF", stroke, 0.9, rx=6.5, opacity=0.92)
            d.doc.text("labels", x + self.w - bw / 2 - 4, y + 13.2, self.badge,
                       style.T_TINY, tc, "500")
        return self._reg(d, x, y)


class OpDot(Element):
    """Circle operator glyph: '+' (add), 'x' (mul), 'o' (modulate/scale)."""

    R = 9.5

    def __init__(self, op="+", id=None):
        self.op, self.id = op, id or None

    def measure(self):
        self.w = self.h = 2 * self.R
        return self.w, self.h

    def render(self, d, x, y):
        cx, cy, r = x + self.R, y + self.R, self.R
        _, stroke, _ = style.KIND["op"]
        d.doc.circle("nodes", cx, cy, r, "#FFFFFF", stroke, 1.3)
        k = r * 0.48
        if self.op == "+":
            d.doc.line("nodes", cx - k, cy, cx + k, cy, stroke, 1.5)
            d.doc.line("nodes", cx, cy - k, cx, cy + k, stroke, 1.5)
        elif self.op == "x":
            k *= 0.85
            d.doc.line("nodes", cx - k, cy - k, cx + k, cy + k, stroke, 1.5)
            d.doc.line("nodes", cx - k, cy + k, cx + k, cy - k, stroke, 1.5)
        elif self.op == "o":
            d.doc.circle("nodes", cx, cy, r * 0.38, stroke)
        return self._reg(d, x, y)


class IOLabel(Element):
    """Stadium-shaped tensor label (mono), for diagram inputs/outputs."""

    def __init__(self, text, id=None, modality=None, size=None, hatched=False):
        self.text, self.id, self.modality = text, id or None, modality
        self.size = size or style.T_SUB + 0.5
        self.hatched = hatched

    def measure(self):
        self.w = measure(self.text, self.size, mono=True)[0] + 22
        self.h = 22
        return self.w, self.h

    def render(self, d, x, y):
        fill, stroke, tc = style.MODALITY.get(self.modality or "none",
                                              style.MODALITY["none"])
        if self.modality is None:
            fill, stroke, tc = style.KIND["io"]
        if self.hatched:
            fill = f'url(#{d.doc.hatch(fill, stroke)})'
        d.doc.rect("nodes", x, y, self.w, self.h, fill, stroke, 1.1,
                   rx=self.h / 2)
        d.doc.text("nodes", x + self.w / 2, y + self.h / 2 + self.size * 0.36,
                   self.text, self.size, tc, mono=True, halo=self.hatched)
        return self._reg(d, x, y)


class VAEShape(Element):
    """Trapezoid for an encoder/decoder (VAE, tokenizer). The signature LDM/SD
    idiom: the encoder narrows in the flow direction, the decoder widens.

    role: 'enc' (narrows toward flow) | 'dec' (widens toward flow).
    flow: 'r' (left→right) | 'l' | 'u' (bottom→top) | 'd'. The wide face is the
    input side; the narrow face the output (latent) side for an encoder.
    """

    def __init__(self, label, role="enc", flow="r", id=None, sub=None,
                 w=72, h=70, narrow=0.5, kind="vae", badge=None):
        self.label, self.role, self.flow = label, role, flow
        self.id, self.sub, self.badge = id or None, sub, badge
        self._w, self._h, self.narrow, self.kind = w, h, narrow, kind

    def measure(self):
        lw = measure(self.label, style.T_LABEL, "500")[0]
        if self.flow in "ud":
            self.w = max(self._w, lw + 22)
            self.h = self._h
        else:
            self.w = self._w
            self.h = max(self._h, lw + 22)
        return self.w, self.h

    def render(self, d, x, y):
        fill, stroke, tc = style.KIND[self.kind]
        w, h, n = self.w, self.h, self.narrow
        # which face is narrow: encoder narrows toward the flow direction
        narrow_face = self.flow if self.role == "enc" else \
            {"r": "l", "l": "r", "u": "d", "d": "u"}[self.flow]
        if narrow_face == "r":
            pts = [(x, y), (x + w, y + h * (1 - n) / 2), (x + w, y + h * (1 + n) / 2), (x, y + h)]
        elif narrow_face == "l":
            pts = [(x + w, y), (x + w, y + h), (x, y + h * (1 + n) / 2), (x, y + h * (1 - n) / 2)]
        elif narrow_face == "u":
            pts = [(x, y + h), (x + w * (1 - n) / 2, y), (x + w * (1 + n) / 2, y), (x + w, y + h)]
        else:  # narrow at bottom
            pts = [(x, y), (x + w, y), (x + w * (1 + n) / 2, y + h), (x + w * (1 - n) / 2, y + h)]
        dpath = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + " Z"
        d.doc.path("nodes", dpath, stroke=stroke, sw=1.4, fill=fill,
                   bbox=(x, y, w, h))
        rot = -90 if self.flow in "rl" else 0
        cx, cy = x + w / 2, y + h / 2
        if rot:
            d.doc.raw("nodes",
                      f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="{style.FONT_SANS}" '
                      f'font-size="{style.T_LABEL}px" fill="{tc}" font-weight="500" '
                      f'text-anchor="middle" transform="rotate({rot} {cx:.1f} {cy:.1f})">'
                      f'{self.label}</text>')
        else:
            d.doc.text("nodes", cx, cy + style.T_LABEL * 0.34, self.label,
                       style.T_LABEL, tc, "500")
        if self.badge:
            bw = measure(self.badge, style.T_TINY, "500")[0] + 10
            d.doc.rect("labels", x + w - bw - 2, y + 2, bw, 13, "#FFFFFF",
                       stroke, 0.9, rx=6.5, opacity=0.92)
            d.doc.text("labels", x + w - bw / 2 - 2, y + 11.4, self.badge,
                       style.T_TINY, tc, "500")
        return self._reg(d, x, y)


class KnownChip(Element):
    """Collapsed 'user already knows this' concept: check + name + pointer."""

    def __init__(self, label, ref=None, id=None):
        self.label, self.ref, self.id = label, ref, id or None

    def measure(self):
        lw = measure(self.label, style.T_SUB + 1, "500")[0]
        rw = measure(self.ref, style.T_TINY, mono=True)[0] + 8 if self.ref else 0
        self.w = 16 + lw + rw + 18
        self.h = 21
        return self.w, self.h

    def render(self, d, x, y):
        fill, stroke, tc = style.KIND["known"]
        d.doc.rect("nodes", x, y, self.w, self.h, fill, stroke, 1.1,
                   rx=self.h / 2, dash="3 2.2")
        cx, cy = x + 12, y + self.h / 2
        d.doc.path("nodes", f"M {cx-3.5},{cy} L {cx-1},{cy+2.8} L {cx+3.8},{cy-3}",
                   stroke=stroke, sw=1.7)
        tx = x + 20
        d.doc.text("nodes", tx, cy + (style.T_SUB + 1) * 0.36, self.label,
                   style.T_SUB + 1, tc, "500", anchor="start")
        if self.ref:
            d.doc.text("nodes", x + self.w - 9, cy + style.T_TINY * 0.36,
                       self.ref, style.T_TINY, style.MUTED, anchor="end",
                       mono=True)
        return self._reg(d, x, y)


class TextLabel(Element):
    def __init__(self, text, size=style.T_NOTE, color=style.MUTED,
                 weight="normal", mono=False, id=None, max_w=None,
                 anchor="middle", line_h=None):
        self.text, self.size, self.color = text, size, color
        self.weight, self.mono, self.id = weight, mono, id or None
        self.max_w, self.anchor = max_w, anchor
        self.line_h = line_h or size * 1.38

    def measure(self):
        from .text import wrap
        self.lines = (wrap(self.text, self.size, self.max_w, self.weight)
                      if self.max_w else [self.text])
        self.w = max(measure(ln, self.size, self.weight, self.mono)[0]
                     for ln in self.lines)
        self.h = len(self.lines) * self.line_h
        return self.w, self.h

    def render(self, d, x, y):
        ax = {"middle": x + self.w / 2, "start": x, "end": x + self.w}[self.anchor]
        for i, ln in enumerate(self.lines):
            d.doc.text("nodes", ax, y + i * self.line_h + self.size,
                       ln, self.size, self.color, self.weight,
                       anchor=self.anchor, mono=self.mono)
        return self._reg(d, x, y)


class Formula(Element):
    """Vector math (mathtext -> outlined paths)."""

    def __init__(self, s, size=13, color=style.INK, id=None):
        self.s, self.size, self.color, self.id = s, size, color, id or None

    def measure(self):
        from .text import tex
        d_, w, h, ymax, x0 = tex(self.s, self.size)
        self._t = (d_, w, h, ymax, x0)
        self.w, self.h = w, h
        return w, h

    def render(self, d, x, y):
        d_, w, h, ymax, x0 = self._t
        d.doc.raw("nodes",
                  f'<g transform="translate({x - x0:.2f},{y + ymax:.2f}) '
                  f'scale(1,-1)"><path d="{d_}" fill="{self.color}"/></g>',
                  bbox=(x, y, w, h))
        return self._reg(d, x, y)


class Spacer(Element):
    def __init__(self, w=0, h=0):
        self.w, self.h = w, h

    def measure(self):
        return self.w, self.h

    def render(self, d, x, y):
        return Box(x, y, self.w, self.h)
