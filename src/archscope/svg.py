"""Minimal SVG document builder with layers, markers, patterns and bbox tracking."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import style
from .text import measure, normalize_glyphs


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _cid(color: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", color)


class SvgDoc:
    """Layered SVG builder. Layers paint in order: frames, edges, nodes, labels."""

    LAYERS = ("frames", "edges", "nodes", "labels")

    def __init__(self):
        self.layers = {k: [] for k in self.LAYERS}
        self.defs: dict[str, str] = {}
        self.x0 = self.y0 = float("inf")
        self.x1 = self.y1 = float("-inf")

    # --- bbox ----------------------------------------------------------------
    def grow(self, x, y, w=0.0, h=0.0):
        self.x0 = min(self.x0, x)
        self.y0 = min(self.y0, y)
        self.x1 = max(self.x1, x + w)
        self.y1 = max(self.y1, y + h)

    # --- defs ----------------------------------------------------------------
    def marker(self, color: str) -> str:
        mid = f"arr-{_cid(color)}"
        if mid not in self.defs:
            self.defs[mid] = (
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                f'<path d="M0.5,1 L9,5 L0.5,9 Z" fill="{color}"/></marker>')
        return mid

    def hatch(self, base: str, line: str) -> str:
        pid = f"hatch-{_cid(base)}-{_cid(line)}"
        if pid not in self.defs:
            self.defs[pid] = (
                f'<pattern id="{pid}" width="5.5" height="5.5" '
                f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="5.5" height="5.5" fill="{base}"/>'
                f'<line x1="0" y1="0" x2="0" y2="5.5" stroke="{line}" '
                f'stroke-width="1.4" stroke-opacity="0.55"/></pattern>')
        return pid

    # --- primitives ------------------------------------------------------------
    def rect(self, layer, x, y, w, h, fill, stroke=None, sw=1.2, rx=0.0,
             dash=None, opacity=None):
        a = [f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"',
             f'rx="{rx}"' if rx else "", f'fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        self.layers[layer].append(f'<rect {" ".join(filter(None, a))}/>')
        self.grow(x, y, w, h)

    def circle(self, layer, cx, cy, r, fill, stroke=None, sw=1.2):
        s = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.layers[layer].append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" {s}/>')
        self.grow(cx - r, cy - r, 2 * r, 2 * r)

    def line(self, layer, x1, y1, x2, y2, stroke, sw=1.2, dash=None, cap="butt"):
        d = f'stroke-dasharray="{dash}"' if dash else ""
        self.layers[layer].append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}" {d}/>')
        self.grow(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def path(self, layer, d, stroke=None, sw=1.4, fill="none", dash=None,
             marker_end=None, bbox=None, transform=None):
        a = [f'd="{d}"', f'fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{sw}" '
                     f'stroke-linejoin="round" stroke-linecap="round"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if marker_end:
            a.append(f'marker-end="url(#{marker_end})"')
        if transform:
            a.append(f'transform="{transform}"')
        self.layers[layer].append(f'<path {" ".join(a)}/>')
        if bbox:
            self.grow(*bbox)

    def text(self, layer, x, y, s, size=12, color=style.INK, weight="normal",
             anchor="middle", mono=False, halo=False, opacity=None):
        """(x, y) is the text anchor point at the BASELINE."""
        s = normalize_glyphs(s)
        fam = style.FONT_MONO if mono else style.FONT_SANS
        a = [f'x="{x:.1f}" y="{y:.1f}"',
             f'font-family="{fam}" font-size="{size}px" fill="{color}"',
             f'text-anchor="{anchor}"']
        if weight != "normal":
            a.append(f'font-weight="{weight}"')
        if halo:
            a.append('paint-order="stroke" stroke="#FFFFFF" stroke-width="3px"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        self.layers[layer].append(f'<text {" ".join(a)}>{_esc(s)}</text>')
        w, h = measure(s, size, weight, mono)
        x0 = {"middle": x - w / 2, "start": x, "end": x - w}[anchor]
        self.grow(x0, y - size, w, size * 1.3)

    def raw(self, layer, s, bbox=None):
        self.layers[layer].append(s)
        if bbox:
            self.grow(*bbox)

    # --- output ----------------------------------------------------------------
    def render(self, margin=26) -> str:
        x, y = self.x0 - margin, self.y0 - margin
        w, h = (self.x1 - self.x0) + 2 * margin, (self.y1 - self.y0) + 2 * margin
        defs = "".join(self.defs.values())
        body = "".join(
            f'<g id="{k}">{"".join(v)}</g>' for k, v in self.layers.items())
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}" '
            f'width="{w:.0f}" height="{h:.0f}">'
            f'<defs>{defs}</defs>'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{style.BG}"/>{body}</svg>')

    def save(self, path: str | Path, png: bool = True, zoom: float = 4.0,
             margin: float = 26):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(margin))
        if png:
            if shutil.which("resvg"):
                subprocess.run(
                    ["resvg", "--zoom", str(zoom), str(path),
                     str(path.with_suffix(".png"))],
                    check=True, capture_output=True)
            else:
                print("archscope: `resvg` not found — wrote SVG only "
                      "(install: brew/cargo install resvg)", file=sys.stderr)
        return path
