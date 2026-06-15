"""Edge routing: straight / orthogonal (rounded corners) / side rails."""
import math

from . import style
from .text import measure


def _unit(ax, ay, bx, by):
    d = math.hypot(bx - ax, by - ay) or 1.0
    return (bx - ax) / d, (by - ay) / d


def rounded(pts, r=7):
    """Polyline -> SVG path with rounded corners."""
    if len(pts) < 2:
        return ""
    if len(pts) == 2:
        return (f"M {pts[0][0]:.1f},{pts[0][1]:.1f} "
                f"L {pts[1][0]:.1f},{pts[1][1]:.1f}")
    d = [f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for i in range(1, len(pts) - 1):
        (px, py), (cx, cy), (nx, ny) = pts[i - 1], pts[i], pts[i + 1]
        d1 = math.hypot(cx - px, cy - py)
        d2 = math.hypot(nx - cx, ny - cy)
        rr = min(r, d1 / 2, d2 / 2)
        u1, v1 = _unit(px, py, cx, cy)
        u2, v2 = _unit(cx, cy, nx, ny)
        ax, ay = cx - u1 * rr, cy - v1 * rr
        bx, by = cx + u2 * rr, cy + v2 * rr
        d.append(f"L {ax:.1f},{ay:.1f} Q {cx:.1f},{cy:.1f} {bx:.1f},{by:.1f}")
    d.append(f"L {pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def route_points(a, b, route="auto", via=None, frac=0.5):
    """Compute polyline from point a to point b.

    route: 'straight' | 'vh' (vertical then horizontal) | 'hv' | 'zv'
    (vertical S: down, across at `frac` of the gap, down) | 'zh' | 'auto'.
    via: explicit intermediate points.
    """
    (x1, y1), (x2, y2) = a, b
    if via:
        return [a, *via, b]
    if route == "straight":
        return [a, b]
    if route == "vh":
        return [a, (x1, y2), b] if x1 != x2 else [a, b]
    if route == "hv":
        return [a, (x2, y1), b] if y1 != y2 else [a, b]
    if route == "zv":
        ym = y1 + (y2 - y1) * frac
        return [a, (x1, ym), (x2, ym), b]
    if route == "zh":
        xm = x1 + (x2 - x1) * frac
        return [a, (xm, y1), (xm, y2), b]
    # auto
    if abs(x1 - x2) < 0.6 or abs(y1 - y2) < 0.6:
        return [a, b]
    if abs(y2 - y1) >= abs(x2 - x1):
        return route_points(a, b, "zv", frac=frac)
    return route_points(a, b, "zh", frac=frac)


def draw_edge(doc, a, b, *, style_name="main", color=None, width=None,
              route="auto", via=None, frac=0.5, arrow=True, label=None,
              label_side="right", label_at=0.5, label_dx=0, label_dy=0,
              dash="unset", r=7, label_size=None, halo=True,
              label_anchor=None, label_bg=False):
    st = style.EDGE[style_name]
    color = color or st["stroke"]
    width = width or st["width"]
    dash = st["dash"] if dash == "unset" else dash
    pts = route_points(a, b, route, via, frac)
    d = rounded(pts, r)
    marker = doc.marker(color) if arrow else None
    doc.path("edges", d, stroke=color, sw=width, dash=dash,
             marker_end=marker, bbox=_bbox(pts))
    if label:
        # place along the longest segment, offset perpendicular
        segs = list(zip(pts, pts[1:]))
        (sx, sy), (ex, ey) = max(
            segs, key=lambda s: math.hypot(s[1][0] - s[0][0],
                                           s[1][1] - s[0][1]))
        mx, my = sx + (ex - sx) * label_at, sy + (ey - sy) * label_at
        vertical = abs(ey - sy) > abs(ex - sx)
        size = label_size or style.T_EDGE
        if vertical:
            anchor = label_anchor or ("start" if label_side == "right" else "end")
            off = 7 if label_side == "right" else -7
            tx, ty, ta = mx + off + label_dx, my + size * 0.35 + label_dy, anchor
        else:
            tx, ty, ta = mx + label_dx, my - 6 + label_dy, label_anchor or "middle"
        if label_bg:
            # an OPAQUE box behind the label — the per-glyph halo can't cover the
            # spaces between words, so a passing dashed line still reads as a
            # strike-through ("x = x" -> "x-=-x"). The box knocks the whole strip out.
            lw = measure(label, size, mono=True)[0]
            lx = {"start": tx, "middle": tx - lw / 2, "end": tx - lw}[ta]
            doc.rect("labels", lx - 4, ty - size, lw + 8, size + 6,
                     "#FFFFFF", None, 0, rx=3)
        doc.text("labels", tx, ty, label, size, style.MUTED, anchor=ta, mono=True,
                 halo=halo)
