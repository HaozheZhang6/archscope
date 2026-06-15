"""Geometry guarantees the framework must keep: the auto-router avoids piercing
boxes, and the save()-time linter judges 'edge through a box' by the same rule the
router routed to. These lock in the 'generate correctly the first time' behavior —
an author writes the natural edge and does not hand-route around obstacles.

Run: python -m pytest tests/ -q   (or just `python tests/test_routing.py`).
"""
from archscope import Block, Diagram
from archscope.diagram import _perp_route


def _scene():
    """A sits top-left; OBST sits directly below it; TARGET is wide (so OBST counts
    as a leaf, not a container) and lower-right. The natural edge A -> target.l@0.25
    would, without obstacle-awareness, drop straight down through OBST."""
    d = Diagram(title="routing")
    d.place(Block("A", id="a", min_w=110), 60, 60)
    d.place(Block("OBST", id="obst", min_w=110), 60, 160)
    d.place(Block("TARGET wide enough that obst is a leaf", id="tgt", min_w=420), 360, 230)
    return d


def test_naive_route_would_pierce():
    """Guards the test itself: the un-routed perpendicular path really does pierce,
    so a passing test_router_detours is meaningful and not vacuous."""
    d = _scene()
    naive = _perp_route(d.box("a").bottom(), "b",
                        d.box("tgt").anchor("l@0.25"), "l", 14, 0.5)
    assert d._poly_pierces(naive, "a", "tgt"), "scene no longer blocks; fix the fixture"


def test_router_detours_around_obstacle():
    d = _scene()
    d.edge("a", "tgt.l@0.25")            # bare-id source: router may re-pick A's side
    poly = d.edge_paths[-1][0]
    assert not d._poly_pierces(poly, "a", "tgt"), "router failed to avoid OBST"
    assert d.check_edges_through_boxes() == [], "linter disagrees with router"


def test_default_clean_route_is_unchanged():
    """When the straight perpendicular route is already clean, the router must return
    it verbatim — obstacle-awareness must not perturb the common case."""
    d = Diagram(title="straight")
    d.place(Block("top", id="t", min_w=120), 100, 60)
    d.place(Block("bot", id="b", min_w=120), 100, 200)   # directly below, nothing between
    d.edge("t", "b")
    poly = d.edge_paths[-1][0]
    assert not d._poly_pierces(poly, "t", "b")
    # a clean vertical stack edge is a single straight segment (no detour bends)
    assert len(poly) == 2, "default-clean edge gained spurious bends: %r" % (poly,)


def test_pinned_ports_are_respected():
    """A pinned port (id.side@frac) must keep its landing point even while the router
    detours: only the FREE (bare-id) endpoint's side may change."""
    d = _scene()
    d.edge("a", "tgt.l@0.25")
    poly = d.edge_paths[-1][0]
    want = d.box("tgt").anchor("l@0.25")
    assert abs(poly[-1][0] - want[0]) < 0.5 and abs(poly[-1][1] - want[1]) < 0.5, \
        "router moved a pinned port"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all routing guarantees hold")
