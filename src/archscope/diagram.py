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
        # bookkeeping for the low-friction helpers (auto-legend / overlap warn)
        self.placements: list[tuple[str, Box]] = []
        self.used_kinds: list[str] = []      # order-preserving, de-duped on use
        self.used_modalities: list[str] = []
        self.used_edges: list[str] = []
        self.used_ops: list[str] = []

    def _seen(self, lst, v):
        if v and v not in lst:
            lst.append(v)

    # --- placement -----------------------------------------------------------
    def place(self, el, x=None, y=None, right_of=None, left_of=None, above=None,
              below=None, gap=44, ref_align=None):
        """Absolute place(el, x, y) OR relative: place(el, right_of="enc", gap=80,
        ref_align="cy"). Relative placement reads the reference box, so edits to
        content keep the gap instead of drifting into overlap. ref_align picks the
        edge to line up on: y/cy/y2 (for right_of/left_of) or x/cx/x2 (above/below)."""
        el.measure()
        if x is None or y is None:
            ref = right_of or left_of or above or below
            r = self.anchors[ref]
            if right_of:
                x = r.x2 + gap
                y = self._align_y(r, el, ref_align)
            elif left_of:
                x = r.x - gap - el.w
                y = self._align_y(r, el, ref_align)
            elif below:
                y = r.y2 + gap
                x = self._align_x(r, el, ref_align)
            else:  # above
                y = r.y - gap - el.h
                x = self._align_x(r, el, ref_align)
        box = el.render(self, x, y)
        self.placements.append((getattr(el, "id", None) or type(el).__name__, box))
        return box

    @staticmethod
    def _align_y(ref, el, a):
        a = a or "cy"
        return {"y": ref.y, "cy": ref.cy - el.h / 2, "y2": ref.y2 - el.h}[a]

    @staticmethod
    def _align_x(ref, el, a):
        a = a or "cx"
        return {"x": ref.x, "cx": ref.cx - el.w / 2, "x2": ref.x2 - el.w}[a]

    _slug_n: dict = None

    def _slug(self, s):
        import re
        base = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:18] or "n"
        self._slug_n = self._slug_n or {}
        self._slug_n[base] = self._slug_n.get(base, 0) + 1
        return base if self._slug_n[base] == 1 else f"{base}{self._slug_n[base]}"

    def flow(self, children, dir="up", connect=True, gap=22, align="center",
             edge_style="main", id=None, **place_kw):
        """One call = stack + place + auto-connect. ids are slugged from each
        child's label (so you can reference them), consecutive nodes are chained
        with perpendicular edges. dir: up/down (vertical) or right/left. The
        returned stack carries an id (given or auto) so you can place the next
        flow relative to it: d.flow([...], right_of=enc.id)."""
        from .containers import HStack, VStack
        for c in children:
            if not getattr(c, "id", None):
                c.id = self._slug(getattr(c, "label", None)
                                  or getattr(c, "text", None) or type(c).__name__)
        # children are written in DATAFLOW order (input first). For up/left the
        # first item belongs at the bottom/right, so reverse for the stack.
        kids = list(reversed(children)) if dir in ("up", "left") else list(children)
        Stack = HStack if dir in ("right", "left") else VStack
        stack = Stack(kids, gap=gap, align=align,
                      connect=(dir if connect else None), edge_style=edge_style,
                      id=id or self._slug("flow"))
        self.place(stack, **place_kw)
        return stack

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

    edge_ends: list = None

    def edge(self, src, dst, a_side=None, b_side=None, stub=14, frac=0.5, **kw):
        self._seen(self.used_edges, kw.get("style_name", "main"))
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
        if kw.get("arrow", True):                      # record arrowhead landing point
            self.edge_ends = self.edge_ends or []
            self.edge_ends.append(tuple(b))
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

    def loop_arc(self, box_id, side="r", label="×T", out=34, color=None,
                 frac=(0.18, 0.82)):
        """A feedback arc around `box_id` (the LDM 'denoising ×T' loop): exits one
        side near the top, runs along the side, re-enters near the bottom with an
        arrowhead. `frac` = (exit, enter) positions along the side."""
        color = color or style.EDGE["main"]["stroke"]
        b = self.anchors[box_id]
        if side in "rl":
            xedge = b.x2 if side == "r" else b.x
            rail = xedge + (out if side == "r" else -out)
            a = (xedge, b.y + b.h * frac[0])
            z = (xedge, b.y + b.h * frac[1])
            via = [(rail, a[1]), (rail, z[1])]
        else:
            yedge = b.y if side == "t" else b.y2
            rail = yedge + (-out if side == "t" else out)
            a = (b.x + b.w * frac[0], yedge)
            z = (b.x + b.w * frac[1], yedge)
            via = [(a[0], rail), (z[0], rail)]
        draw_edge(self.doc, a, z, via=via, color=color, width=1.4,
                  arrow=True, label=label, label_side="right",
                  label_size=style.T_SUB + 1)

    # --- annotations -----------------------------------------------------------
    def note(self, x, y, text_, size=style.T_NOTE, color=style.MUTED,
             anchor="start", max_w=None, mono=False, weight="normal"):
        lines = wrap(text_, size, max_w, weight) if max_w else [text_]
        for i, ln in enumerate(lines):
            self.doc.text("labels", x, y + i * size * 1.4, ln, size, color,
                          weight, anchor=anchor, mono=mono)

    # --- low-friction helpers --------------------------------------------------
    def auto_legend(self, x, y, max_w=720, edges=True, extra=None):
        """Build a legend from the kinds / modalities / edge-styles actually used,
        so you don't hand-list a Swatches every figure. `extra` appends manual
        entries (e.g. a hatched note)."""
        from .panels import Swatches
        ent = []
        for m in self.used_modalities:
            if m and m != "none":
                ent.append((m, style.MODALITY_NAMES.get(m, m) if hasattr(
                    style, "MODALITY_NAMES") else m))
        for k in self.used_kinds:
            if k in style.KIND_NAMES:
                ent.append((k, style.KIND_NAMES[k]))
        if edges:
            for e in self.used_edges:
                if e in style.EDGE_NAMES:
                    ent.append((e, style.EDGE_NAMES[e], "edge"))
        _opn = {"+": "add", "x": "gate (×)", "o": "modulate (⊙)"}
        for op in self.used_ops:
            ent.append((op, _opn.get(op, op), "glyph:" + op))
        ent += (extra or [])
        leg = Swatches(ent, max_w=max_w, id="leg")
        self.place(leg, x, y)
        return leg

    def check(self, pad=2.0, min_frac=0.08):
        """Return [(id_a, id_b, overlap_frac), ...] for id'd boxes that PARTIALLY
        overlap — so an agent can self-correct without ever opening the PNG.
        Containment (a box inside its container) is ignored."""
        items = list(self.anchors.items())
        hits = []
        for i in range(len(items)):
            na, A = items[i]
            for j in range(i + 1, len(items)):
                nb, B = items[j]
                ox = min(A.x2, B.x2) - max(A.x, B.x) - pad
                oy = min(A.y2, B.y2) - max(A.y, B.y) - pad
                if ox <= 0 or oy <= 0:
                    continue
                contained = ((A.x >= B.x and A.x2 <= B.x2 and A.y >= B.y and A.y2 <= B.y2)
                             or (B.x >= A.x and B.x2 <= A.x2 and B.y >= A.y and B.y2 <= A.y2))
                if contained:
                    continue
                frac = (ox * oy) / min(A.w * A.h, B.w * B.h)
                if frac >= min_frac:
                    hits.append((na, nb, round(frac, 2)))
        return hits

    def check_arrows(self, tol=7.0):
        """Return groups of arrowheads that land on (nearly) the same point —
        i.e. edges that converge and stack their heads. Route them to distinct
        ports (e.g. 'box.b@0.3' / '@0.7') to fix."""
        ends = self.edge_ends or []
        groups, used = [], [False] * len(ends)
        for i in range(len(ends)):
            if used[i]:
                continue
            grp = [i]
            for j in range(i + 1, len(ends)):
                if not used[j] and abs(ends[i][0] - ends[j][0]) <= tol \
                        and abs(ends[i][1] - ends[j][1]) <= tol:
                    grp.append(j); used[j] = True
            if len(grp) > 1:
                groups.append((tuple(round(v) for v in ends[i]), len(grp)))
        return groups

    def warn_overlaps(self, on_overlap="warn"):
        hits = self.check()
        arrows = self.check_arrows()
        if (hits or arrows) and on_overlap != "ignore":
            parts = []
            if hits:
                parts += ["%d overlapping box(es):" % len(hits)] + [
                    "  '%s' x '%s'  (%.0f%%)" % (a, b, f * 100) for a, b, f in hits]
            if arrows:
                parts += ["%d arrowhead pile-up(s) — route to distinct ports:" % len(arrows)] + [
                    "  %d arrows land at ~(%d,%d)" % (n, p[0], p[1]) for p, n in arrows]
            msg = "archscope: " + "\n".join(parts)
            if on_overlap == "raise":
                raise ValueError(msg)
            import sys
            print(msg, file=sys.stderr)
        return hits + arrows

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

    def save(self, path, png=True, zoom=2.0, on_overlap="warn"):
        self.warn_overlaps(on_overlap)
        self._title_block()
        out = self.doc.save(path, png=png, zoom=zoom)
        return out
