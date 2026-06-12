"""Boxes and anchor points."""
from dataclasses import dataclass


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    # --- corner / center points -------------------------------------------
    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    # side anchors; frac runs 0..1 along the side (left->right / top->bottom)
    def top(self, frac=0.5):
        return (self.x + self.w * frac, self.y)

    def bottom(self, frac=0.5):
        return (self.x + self.w * frac, self.y2)

    def left(self, frac=0.5):
        return (self.x, self.y + self.h * frac)

    def right(self, frac=0.5):
        return (self.x2, self.y + self.h * frac)

    def center(self):
        return (self.cx, self.cy)

    def anchor(self, spec: str):
        """spec: 't', 'b', 'l', 'r', 'c', optionally with fraction 't@0.25'."""
        frac = 0.5
        if "@" in spec:
            spec, f = spec.split("@")
            frac = float(f)
        return {"t": self.top, "b": self.bottom, "l": self.left,
                "r": self.right}[spec](frac) if spec != "c" else self.center()
