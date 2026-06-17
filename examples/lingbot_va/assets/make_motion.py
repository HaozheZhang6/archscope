"""Synthesize a tiny 'sunrise' video — 5 clean frames + flow-matching-noised versions
at sigma in {0, 0.5, 1.0}. Used by fig10b to illustrate AR (next frame) vs diffusion
(denoise the whole clip) on concrete, obviously-moving content. CC0 (procedural)."""
import numpy as np
from PIL import Image
from pathlib import Path

OUT = Path(__file__).resolve().parent / "motion"
OUT.mkdir(exist_ok=True)
W = H = 80
NF = 5


def frame(t):
    """t in [0,1]: a sun rising over a horizon, sky warming."""
    img = np.zeros((H, W, 3), float)
    yy = np.linspace(0, 1, H)[:, None]
    # sky: night-blue (top) -> warm (bottom), brightening with t
    top = np.array([18, 22, 55]) + t * np.array([70, 95, 120])
    bot = np.array([150, 95, 70]) + t * np.array([95, 110, 90])
    sky = top[None, None] * (1 - yy[..., None]) + bot[None, None] * yy[..., None]
    img[:] = sky.reshape(H, 1, 3)
    # sun: rises from horizon to upper-third as t goes 0->1
    sx, sy, r = W * 0.5, (0.74 - 0.46 * t) * H, 10
    Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
    glow = np.clip(1 - d / (r * 2.6), 0, 1)[..., None]
    img = img * (1 - 0.55 * glow) + np.array([255, 225, 130]) * 0.55 * glow
    disk = (d <= r)[..., None]
    img = np.where(disk, np.array([255, 236, 150]) * (0.6 + 0.4 * t), img)
    # ground
    g = int(0.80 * H)
    img[g:] = np.array([34, 44, 32]) + t * np.array([18, 26, 18])
    return np.clip(img, 0, 255).astype(np.uint8)


cleans = [frame(i / (NF - 1)) for i in range(NF)]


def noised(clean, sigma, seed):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.5, 0.22, clean.shape) * 255
    out = (1 - sigma) * clean.astype(float) + sigma * eps
    return np.clip(out, 0, 255).astype(np.uint8)


for i, c in enumerate(cleans):
    for sig, tag in [(0.0, "clean"), (0.5, "half"), (1.0, "noise")]:
        Image.fromarray(noised(c, sig, seed=100 + i)).resize((W * 2, H * 2), Image.NEAREST)\
            .save(OUT / f"f{i}_{tag}.png")
print("wrote", NF * 3, "frames to", OUT)
