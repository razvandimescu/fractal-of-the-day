"""Fractal-of-the-day comparison: flame, Julia, Newton. All date-seeded.

Each type derives every parameter from the date so the image is deterministic
and reproducible: regenerate any date -> identical image.
"""
import sys
from datetime import date

import numpy as np
from PIL import Image

W = H = 1024


# ---------- shared ----------

def palette(rng, n=256):
    """Smooth cyclic palette from a few random control colors."""
    k = rng.integers(3, 6)
    ctrl = rng.random((k, 3)) ** 0.8
    xs = np.linspace(0, 1, k)
    t = np.linspace(0, 1, n)
    pal = np.stack([np.interp(t, xs, ctrl[:, c]) for c in range(3)], 1)
    return (pal * 255).astype(np.uint8)


def save(arr, name):
    Image.fromarray(arr, "RGB").save(name)
    print("wrote", name)


# ---------- 1. fractal flame (IFS chaos game) ----------

def variation(x, y, kind):
    r2 = x * x + y * y + 1e-9
    r = np.sqrt(r2)
    if kind == 0:                       # linear
        return x, y
    if kind == 1:                       # sinusoidal
        return np.sin(x), np.sin(y)
    if kind == 2:                       # spherical
        return x / r2, y / r2
    if kind == 3:                       # swirl
        return x * np.sin(r2) - y * np.cos(r2), x * np.cos(r2) + y * np.sin(r2)
    if kind == 4:                       # horseshoe
        return (x - y) * (x + y) / r, 2 * x * y / r
    # polar
    return np.arctan2(y, x) / np.pi, r - 1


def flame(seed):
    rng = np.random.default_rng(seed)
    nt = rng.integers(3, 6)
    A = rng.uniform(-1, 1, (nt, 6))          # affine coeffs per transform
    kinds = rng.integers(0, 6, nt)
    colors = rng.random((nt, 3)) ** 0.7
    weights = rng.random(nt); weights /= weights.sum()

    N = 200_000
    x = rng.uniform(-1, 1, N)
    y = rng.uniform(-1, 1, N)
    cr = np.zeros(N); cg = np.zeros(N); cb = np.zeros(N)

    hist = np.zeros((H, W))
    accum = np.zeros((H, W, 3))
    span = 2.4

    for it in range(120):
        t = rng.choice(nt, N, p=weights)
        nx = np.empty(N); ny = np.empty(N)
        for j in range(nt):
            m = t == j
            a, b, c, d, e, f = A[j]
            px = a * x[m] + b * y[m] + c
            py = d * x[m] + e * y[m] + f
            vx, vy = variation(px, py, kinds[j])
            nx[m], ny[m] = vx, vy
            cr[m] = 0.5 * (cr[m] + colors[j, 0])
            cg[m] = 0.5 * (cg[m] + colors[j, 1])
            cb[m] = 0.5 * (cb[m] + colors[j, 2])
        x, y = nx, ny
        if it < 20:
            continue
        ix = ((x / span + 0.5) * W).astype(int)
        iy = ((y / span + 0.5) * H).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy = ix[ok], iy[ok]
        np.add.at(hist, (iy, ix), 1)
        np.add.at(accum, (iy, ix, 0), cr[ok])
        np.add.at(accum, (iy, ix, 1), cg[ok])
        np.add.at(accum, (iy, ix, 2), cb[ok])

    mask = hist > 0
    col = np.zeros_like(accum)
    col[mask] = accum[mask] / hist[mask, None]
    bright = np.zeros((H, W))
    bright[mask] = np.log1p(hist[mask])
    bright /= bright.max() + 1e-9
    bright = bright ** 0.45                          # gamma
    img = (col * bright[..., None] * 255).clip(0, 255).astype(np.uint8)
    return img


# ---------- 2. Julia set ----------

def julia(seed):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi)
    rad = rng.uniform(0.6, 0.8)
    c = complex(rad * np.cos(ang), rad * np.sin(ang))
    zoom = rng.uniform(1.3, 1.7)

    y, x = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
    z = (x + 1j * y) * zoom
    itc = np.zeros((H, W))
    alive = np.ones((H, W), bool)
    maxit = 300
    for i in range(maxit):
        z[alive] = z[alive] ** 2 + c
        esc = alive & (np.abs(z) > 2)
        itc[esc] = i + 1 - np.log2(np.log2(np.abs(z[esc]) + 1e-9) + 1e-9)
        alive &= ~esc
    itc[alive] = 0
    norm = itc / (itc.max() + 1e-9)
    pal = palette(rng)
    idx = (norm * 255).astype(int).clip(0, 255)
    img = pal[idx]
    img[alive] = 0
    return img


# ---------- 3. Newton fractal ----------

def newton(seed):
    rng = np.random.default_rng(seed)
    deg = rng.integers(3, 7)
    roots = np.exp(2j * np.pi * np.arange(deg) / deg) * rng.uniform(0.8, 1.2)
    poly = np.poly(roots)
    dpoly = np.polyder(poly)
    zoom = rng.uniform(1.2, 1.8)

    y, x = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
    z = (x + 1j * y) * zoom
    counts = np.zeros((H, W))
    for i in range(60):
        z = z - np.polyval(poly, z) / (np.polyval(dpoly, z) + 1e-12)
        counts += 1
    # nearest root -> hue; iteration count -> shade
    dist = np.abs(z[..., None] - roots)
    which = dist.argmin(-1)
    pal = palette(rng, n=deg)
    shade = (counts / counts.max()) ** 0.3
    img = (pal[which] * (0.35 + 0.65 * shade[..., None])).clip(0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    seed = int(d.strftime("%Y%m%d"))
    print(f"date={d} seed={seed}")
    save(flame(seed), f"flame_{seed}.png")
    save(julia(seed), f"julia_{seed}.png")
    save(newton(seed), f"newton_{seed}.png")
