"""Text measurement and TeX-to-SVG-path rendering via matplotlib.

We measure with DejaVu metrics (bundled with matplotlib) while rendering SVG
text with the Helvetica stack: DejaVu is slightly wider, so boxes never
underflow. Formulas are converted to pure <path> outlines (Computer Modern),
so exported SVGs are self-contained and survive any converter.
"""
from functools import lru_cache

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

matplotlib.rcParams["mathtext.fontset"] = "cm"


@lru_cache(maxsize=4096)
def measure(s: str, size: float = 12, weight: str = "normal",
            mono: bool = False) -> tuple[float, float]:
    """Approximate rendered width/height of a single-line string."""
    if not s:
        return (0.0, size * 1.25)
    fp = FontProperties(
        family="DejaVu Sans Mono" if mono else "DejaVu Sans",
        weight=weight, size=size)
    w = TextPath((0, 0), s, size=size, prop=fp).get_extents().width
    return (w, size * 1.25)


def wrap(s: str, size: float, max_w: float, weight="normal") -> list[str]:
    words, lines, cur = s.split(), [], ""
    for word in words:
        cand = (cur + " " + word).strip()
        if measure(cand, size, weight)[0] <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


_CODE = {Path.MOVETO: "M", Path.LINETO: "L", Path.CURVE3: "Q", Path.CURVE4: "C"}


def _path_to_d(tp: TextPath) -> str:
    parts = []
    for pts, code in tp.iter_segments(curves=True, simplify=False):
        if code == Path.CLOSEPOLY:
            parts.append("Z")
        else:
            cmd = _CODE.get(code, "L")
            coords = " ".join(f"{v:.2f}" for v in pts)
            parts.append(f"{cmd} {coords}")
    return " ".join(parts)


@lru_cache(maxsize=512)
def tex(s: str, size: float = 13) -> tuple[str, float, float, float, float]:
    """Render TeX (mathtext, e.g. '$x_t$') to an SVG path.

    Returns (d, width, height, ymax, x0). Draw inside
    <g transform="translate(x - x0, y + ymax) scale(1,-1)"> so that (x, y) is
    the top-left of the ink bbox; baseline sits at y + ymax.
    """
    tp = TextPath((0, 0), s, size=size, prop=FontProperties(size=size))
    ext = tp.get_extents()
    d = _path_to_d(tp)
    # shift so bbox x starts at 0
    return d, ext.width, ext.height, ext.ymax, ext.x0
