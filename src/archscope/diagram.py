"""Diagram: canvas, anchor registry, edges between placed elements."""
from pathlib import Path

from . import style
from .edges import draw_edge
from .geom import Box
from .svg import SvgDoc
from .text import measure, wrap

_NORMAL = {"t": (0, -1), "b": (0, 1), "l": (-1, 0), "r": (1, 0)}


def _perp_route(a, sa, b, sb, stub=14, frac=0.5):
    """Orthogonal polyline that exits `a` and enters `b` perpendicular to their
    box sides (sa/sb in t/b/l/r), with `stub`-length straight runs at each end."""
    na, nb = _NORMAL[sa], _NORMAL[sb]
    a2 = (a[0] + na[0] * stub, a[1] + na[1] * stub)
    b2 = (b[0] + nb[0] * stub, b[1] + nb[1] * stub)
    av, bv = na[0] == 0, nb[0] == 0
    if av and bv:
        ym = a2[1] + (b2[1] - a2[1]) * frac
        mid = [(a2[0], ym), (b2[0], ym)]
    elif av:
        mid = [(a2[0], b2[1])]
    elif bv:
        mid = [(b2[0], a2[1])]
    else:
        xm = a2[0] + (b2[0] - a2[0]) * frac
        mid = [(xm, a2[1]), (xm, b2[1])]
    pts = [a, a2, *mid, b2, b]
    out = [pts[0]]
    for p in pts[1:]:                       # drop duplicates
        if abs(p[0] - out[-1][0]) > .3 or abs(p[1] - out[-1][1]) > .3:
            out.append(p)
    res = [out[0]]
    for i in range(1, len(out) - 1):        # drop collinear midpoints
        x0, y0 = res[-1]
        x1, y1 = out[i]
        x2, y2 = out[i + 1]
        if (abs(x0 - x1) < .3 and abs(x1 - x2) < .3) or \
           (abs(y0 - y1) < .3 and abs(y1 - y2) < .3):
            continue
        res.append(out[i])
    res.append(out[-1])
    return res


class Diagram:
    def __init__(self, title=None, subtitle=None):
        self.doc = SvgDoc()
        self.anchors: dict[str, Box] = {}
        self.title, self.subtitle = title, subtitle

    # --- placement -----------------------------------------------------------
    def place(self, el, x, y):
        el.measure()
        return el.render(self, x, y)

    def box(self, id) -> Box:
        return self.anchors[id]

    def _pt(self, spec):
        """'id', 'id.t', 'id.b@0.25', or (x, y) tuple."""
        if isinstance(spec, (tuple, list)):
            return tuple(spec), None
        if "." in spec:
            name, a = spec.split(".", 1)
            return self.anchors[name].anchor(a), (name, a[0])
        return None, (spec, None)

    def edge(self, src, dst, a_side=None, b_side=None, stub=14, frac=0.5, **kw):
        a, ainfo = self._pt(src)
        b, binfo = self._pt(dst)
        sa = a_side or (ainfo[1] if ainfo and ainfo[1] and ainfo[1] in "tblr"
                        else None)
        sb = b_side or (binfo[1] if binfo and binfo[1] and binfo[1] in "tblr"
                        else None)
        # auto-pick facing sides when only ids were given
        if a is None or b is None:
            abox = self.anchors[ainfo[0]] if ainfo else None
            bbox = self.anchors[binfo[0]] if binfo else None
            if a is None and b is None:
                if bbox.y >= abox.y2 - 1:          # b below a
                    a, b, sa, sb = abox.bottom(), bbox.top(), "b", "t"
                elif abox.y >= bbox.y2 - 1:        # b above a
                    a, b, sa, sb = abox.top(), bbox.bottom(), "t", "b"
                elif bbox.x >= abox.x2 - 1:        # b right of a
                    a, b, sa, sb = abox.right(), bbox.left(), "r", "l"
                else:
                    a, b, sa, sb = abox.left(), bbox.right(), "l", "r"
            elif a is None:
                a, fs = self._face(abox, b)
                sa = a_side or fs
            else:
                b, fs = self._face(bbox, a)
                sb = b_side or fs
        # perpendicular entry/exit routing by default: leave the source box along
        # its side normal, arrive at the target box along its side normal.
        if kw.get("route", "auto") == "auto" and not kw.get("via") and sa and sb:
            pts = _perp_route(a, sa, b, sb, stub, frac)
            if len(pts) > 2:
                kw["via"] = pts[1:-1]
        draw_edge(self.doc, a, b, **kw)

    @staticmethod
    def _face(box, p):
        """(point, side) on `box` facing external point p."""
        x, y = p
        if y >= box.y2:
            return box.bottom(), "b"
        if y <= box.y:
            return box.top(), "t"
        return (box.left(), "l") if x < box.x else (box.right(), "r")

    def chain(self, ids, labels=None, **kw):
        labels = labels or [None] * (len(ids) - 1)
        for s, t, lab in zip(ids, ids[1:], labels):
            self.edge(s, t, label=lab, **kw)

    # --- annotations -----------------------------------------------------------
    def note(self, x, y, text_, size=style.T_NOTE, color=style.MUTED,
             anchor="start", max_w=None, mono=False, weight="normal"):
        lines = wrap(text_, size, max_w, weight) if max_w else [text_]
        for i, ln in enumerate(lines):
            self.doc.text("labels", x, y + i * size * 1.4, ln, size, color,
                          weight, anchor=anchor, mono=mono)

    # --- output ----------------------------------------------------------------
    def _title_block(self):
        if not (self.title or self.subtitle):
            return
        x = self.doc.x0
        y = self.doc.y0 - (16 if self.subtitle else 8)
        if self.subtitle:
            for ln in reversed(wrap(self.subtitle, style.T_SUBTITLE,
                                    max(self.doc.x1 - self.doc.x0, 480))):
                self.doc.text("labels", x, y, ln, style.T_SUBTITLE,
                              style.MUTED, anchor="start")
                y -= style.T_SUBTITLE * 1.45
        if self.title:
            self.doc.text("labels", x, y, self.title, style.T_TITLE,
                          style.INK, "600", anchor="start")

    def save(self, path, png=True, zoom=2.0):
        self._title_block()
        out = self.doc.save(path, png=png, zoom=zoom)
        return out
