"""Layout containers: stacks, frames, repeat/stack effects, free placement."""
from . import style
from .geom import Box
from .node import Element
from .text import measure


class RepeatStack(Element):
    """N identical blocks drawn as overlapping cards: the front card is the child,
    the occluded ones behind are dashed ghost outlines (offset up-right).

    Registers its id to the CLUSTER bbox, so edges attach to the stack envelope
    and never cross the ghost outlines.
    """

    def __init__(self, child, times="×N", id=None, n_ghost=2, ox=10, oy=9,
                 rx=10):
        self.child, self.times, self.id = child, times, id
        self.n_ghost, self.ox, self.oy, self.rx = n_ghost, ox, oy, rx

    def measure(self):
        cw, ch = self.child.measure()
        self.w = cw + self.n_ghost * self.ox
        self.h = ch + self.n_ghost * self.oy
        return self.w, self.h

    def render(self, d, x, y):
        cw, ch = self.child.w, self.child.h
        cx, cy = x, y + self.n_ghost * self.oy          # front card: bottom-left
        for k in range(self.n_ghost, 0, -1):            # back to front; frames layer
            d.doc.rect("frames", x + k * self.ox, cy - k * self.oy, cw, ch,
                       "#FFFFFF", style.FAINT, 1.1, rx=self.rx, dash="5 4")
        self.child.render(d, cx, cy)
        tw = measure(self.times, style.T_SUB + 2, "600")[0] + 16
        tx = x + self.w - tw + 8
        ty = y - 9
        d.doc.rect("labels", tx, ty, tw, 19, "#FFFFFF", "#64748B", 1.2, rx=9.5)
        d.doc.text("labels", tx + tw / 2, ty + 13.6, self.times,
                   style.T_SUB + 2, "#1E293B", "600")
        return self._reg(d, x, y)


class VStack(Element):
    """Vertical stack. connect="up"|"down" auto-draws an arrow between consecutive
    children (the dataflow), so you don't hand-write d.chain([...]). "up" = arrows
    point bottom→top (input at bottom); "down" = top→bottom."""

    def __init__(self, children, gap=26, align="center", id=None, connect=None,
                 edge_style="main"):
        self.children, self.gap, self.align, self.id = children, gap, align, id
        self.connect, self.edge_style = connect, edge_style

    def measure(self):
        sizes = [c.measure() for c in self.children]
        self.w = max((w for w, _ in sizes), default=0)
        self.h = sum(h for _, h in sizes) + self.gap * max(len(sizes) - 1, 0)
        return self.w, self.h

    def render(self, d, x, y):
        cy = y
        boxes = []
        for c in self.children:
            cx = {"center": x + (self.w - c.w) / 2, "start": x,
                  "end": x + self.w - c.w}[self.align]
            boxes.append(c.render(d, cx, cy))
            cy += c.h + self.gap
        if self.connect:
            for a, b in zip(boxes, boxes[1:]):
                up = self.connect == "up"
                lo, hi = (b, a) if up else (a, b)   # arrow lo(bottom) -> hi(top)
                src = (lo.cx, lo.y) if up else (lo.cx, lo.y2)
                dst = (hi.cx, hi.y2) if up else (hi.cx, hi.y)
                d.edge(src, dst, style_name=self.edge_style)
        return self._reg(d, x, y)


class HStack(Element):
    """Horizontal stack. connect="right"|"left" auto-draws arrows between children."""

    def __init__(self, children, gap=30, align="center", id=None, connect=None,
                 edge_style="main"):
        self.children, self.gap, self.align, self.id = children, gap, align, id
        self.connect, self.edge_style = connect, edge_style

    def measure(self):
        sizes = [c.measure() for c in self.children]
        self.h = max((h for _, h in sizes), default=0)
        self.w = sum(w for w, _ in sizes) + self.gap * max(len(sizes) - 1, 0)
        return self.w, self.h

    def render(self, d, x, y):
        cx = x
        boxes = []
        for c in self.children:
            cy = {"center": y + (self.h - c.h) / 2, "start": y,
                  "end": y + self.h - c.h}[self.align]
            boxes.append(c.render(d, cx, cy))
            cx += c.w + self.gap
        if self.connect:
            for a, b in zip(boxes, boxes[1:]):
                left = self.connect == "left"
                lo, hi = (b, a) if left else (a, b)
                src = (lo.x, lo.cy) if left else (lo.x2, lo.cy)
                dst = (hi.x2, hi.cy) if left else (hi.x, hi.cy)
                d.edge(src, dst, style_name=self.edge_style)
        return self._reg(d, x, y)


class Free(Element):
    """Absolute placement: [(element, dx, dy), ...]; size from content."""

    def __init__(self, items, id=None, pad_r=0, pad_b=0):
        self.items, self.id = items, id
        self.pad_r, self.pad_b = pad_r, pad_b

    def measure(self):
        self.w = self.h = 0
        for el, dx, dy in self.items:
            w, h = el.measure()
            self.w = max(self.w, dx + w)
            self.h = max(self.h, dy + h)
        self.w += self.pad_r
        self.h += self.pad_b
        return self.w, self.h

    def render(self, d, x, y):
        for el, dx, dy in self.items:
            el.render(d, x + dx, y + dy)
        return self._reg(d, x, y)


class GroupFrame(Element):
    """Titled rounded container around a child element."""

    def __init__(self, child, title=None, id=None, pad=16, dashed=True,
                 tint=None, stroke=None, title_color=None, pad_top=None,
                 title_pos="inline"):
        self.child, self.title, self.id = child, title, id
        self.pad, self.dashed = pad, dashed
        self.tint = tint if tint is not None else "rgba(148,163,184,0.055)"
        self.stroke = stroke or style.FAINT
        self.title_color = title_color or style.MUTED
        self.pad_top = pad_top
        # title_pos: "inline" (title eats vertical space at the top) | "tag"
        # (a pill straddling the top-left border — paper "zone" style, no
        # vertical cost, content fills the whole frame).
        self.title_pos = title_pos

    def measure(self):
        cw, ch = self.child.measure()
        self.th = 21 if (self.title and self.title_pos == "inline") else 0
        self.w = cw + 2 * self.pad
        self.h = ch + 2 * self.pad + self.th
        if self.pad_top is not None:
            self.h = ch + self.pad + self.pad_top + self.th
        return self.w, self.h

    def render(self, d, x, y):
        d.doc.rect("frames", x, y, self.w, self.h, self.tint, self.stroke,
                   1.2, rx=10, dash="6 4" if self.dashed else None)
        if self.title and self.title_pos == "inline":
            d.doc.text("frames", x + 13, y + 16.5, self.title,
                       style.T_SUB + 1.5, self.title_color, "600",
                       anchor="start")
        elif self.title and self.title_pos == "tag":
            tw = measure(self.title, style.T_SUB + 1, "600")[0] + 16
            d.doc.rect("labels", x + 14, y - 10, tw, 19, "#FFFFFF",
                       self.stroke, 1.1, rx=4)
            d.doc.text("labels", x + 14 + tw / 2, y + 3.4, self.title,
                       style.T_SUB + 1, self.title_color, "600")
        top = self.pad_top if self.pad_top is not None else self.pad
        self.child.render(d, x + (self.w - self.child.w) / 2, y + self.th + top)
        return self._reg(d, x, y)


class RepeatFrame(Element):
    """Solid light frame with an 'xN' tag — repeated blocks."""

    def __init__(self, child, times="×N", id=None, pad=18, stroke=None):
        self.child, self.times, self.id, self.pad = child, times, id, pad
        self.stroke = stroke or "#64748B"

    def measure(self):
        cw, ch = self.child.measure()
        self.w, self.h = cw + 2 * self.pad, ch + 2 * self.pad
        return self.w, self.h

    def render(self, d, x, y):
        d.doc.rect("frames", x, y, self.w, self.h, "rgba(100,116,139,0.04)",
                   self.stroke, 1.4, rx=11)
        tw = measure(self.times, style.T_SUB + 2, "600")[0] + 16
        d.doc.rect("labels", x + self.w - tw - 8, y - 9, tw, 19, "#FFFFFF",
                   self.stroke, 1.2, rx=9.5)
        d.doc.text("labels", x + self.w - tw / 2 - 8, y + 4.6, self.times,
                   style.T_SUB + 2, "#1E293B", "600")
        self.child.render(d, x + self.pad, y + self.pad)
        return self._reg(d, x, y)
