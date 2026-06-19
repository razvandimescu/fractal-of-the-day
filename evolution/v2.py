"""v2 'art of the day' — flow-field + domain-warp core with cosine palettes.

Pipeline (all CPU, deterministic per date):
  flow field from domain-warped value-noise fBm  (Quilez warp + Hobbs flow fields)
  -> integrate many particles into glowing ribbons, additive accumulation
  -> log-density tone map (flam3-style) at 2x supersample, LANCZOS downscale
  -> post: bloom + vignette + film grain
  -> cosine palette (Inigo Quilez), spatially clumped colour

Refs: iquilezles.org/articles/warp, /articles/palettes ; tylerxhobbs.com/words/flow-fields
"""
import sys
from datetime import date

import numpy as np
from PIL import Image, ImageFilter

OUT = 1024
SS = 2                      # supersample factor
W = H = OUT * SS


# ---------- deterministic value-noise fBm, samplable at arbitrary coords ----------

def make_fbm(seed, octaves=5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(256).astype(np.int32)
    perm = np.concatenate([perm, perm])          # length 512, avoids wrap math

    def vnoise(x, y):
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)
        xf = x - xi
        yf = y - yi
        u = xf * xf * (3 - 2 * xf)                # smoothstep
        v = yf * yf * (3 - 2 * yf)
        xi &= 255
        yi &= 255

        def corner(dx, dy):
            return perm[(perm[(xi + dx) & 255] + ((yi + dy) & 255)) & 255] / 255.0

        aa, ba = corner(0, 0), corner(1, 0)
        ab, bb = corner(0, 1), corner(1, 1)
        x1 = aa * (1 - u) + ba * u
        x2 = ab * (1 - u) + bb * u
        return x1 * (1 - v) + x2 * v

    def fbm(x, y):
        val = np.zeros_like(x)
        amp, freq, norm = 0.5, 1.0, 0.0
        for _ in range(octaves):
            val += amp * vnoise(x * freq, y * freq)
            norm += amp
            amp *= 0.5
            freq *= 2.0
        return val / norm

    return fbm


# ---------- cosine palette (Inigo Quilez) ----------

def make_palette(rng):
    a = rng.uniform(0.50, 0.62, 3)        # bright bias
    b = rng.uniform(0.35, 0.48, 3)        # strong contrast
    c = rng.uniform(0.7, 1.1, 3)          # gentle hue sweep (avoids muddy)
    d = rng.uniform(0.0, 1.0, 3)

    def pal(t):                                   # t: (...,) -> (...,3) in [0,1]
        t = t[..., None]
        return np.clip(a + b * np.cos(2 * np.pi * (c * t + d)), 0, 1)

    return pal


# ---------- domain-warped flow field ----------

def warped_angle(fbm, x, y, warp=0.35):
    """Angle from a once-warped fBm scalar field. x,y in noise space."""
    qx = fbm(x, y)
    qy = fbm(x + 5.2, y + 1.3)
    f = fbm(x + 4.0 * warp * qx, y + 4.0 * warp * qy)
    return f * 2.0 * np.pi * 2.0                  # a couple of turns of range


def render(seed):
    rng = np.random.default_rng(seed)
    fbm = make_fbm(seed)
    pal = make_palette(rng)

    NOISE_SCALE = 0.006 / SS * (OUT / 1024)       # ~Hobbs 0.005, feature size ~ const
    n_particles = 45000
    steps = 320
    step_len = 0.0022 * W                         # ~0.22% of width, cleaner curves
    margin = 0.18 * W
    brush = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])  # +-shaped thickness

    # start positions across canvas + margin
    px = rng.uniform(-margin, W + margin, n_particles)
    py = rng.uniform(-margin, H + margin, n_particles)

    # spatial colour clumping: base palette-t from a low-freq noise of start point
    base_t = fbm(px * NOISE_SCALE * 0.4 + 11.0, py * NOISE_SCALE * 0.4 + 7.0)
    base_t = (base_t - base_t.min()) / (np.ptp(base_t) + 1e-9)

    accum = np.zeros((H, W, 3), np.float32)
    dens = np.zeros((H, W), np.float32)
    alive = np.ones(n_particles, bool)

    for s in range(steps):
        ang = warped_angle(fbm, px * NOISE_SCALE, py * NOISE_SCALE)
        px = px + np.cos(ang) * step_len
        py = py + np.sin(ang) * step_len

        ix = px.astype(np.int32)
        iy = py.astype(np.int32)
        ok = alive & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        alive &= (ix >= -margin) & (ix < W + margin) & (iy >= -margin) & (iy < H + margin)

        # colour shimmers slightly along the curve
        t = (base_t[ok] + 0.0008 * s) % 1.0
        col = pal(t)
        bx, by = ix[ok], iy[ok]
        for ox, oy in brush:
            jx = np.clip(bx + ox, 0, W - 1)
            jy = np.clip(by + oy, 0, H - 1)
            w = 1.0 if (ox == 0 and oy == 0) else 0.4
            np.add.at(accum, (jy, jx), col * w)
            np.add.at(dens, (jy, jx), w)

    return tonemap(accum, dens)


# ---------- tone mapping + post ----------

def tonemap(accum, dens):
    mask = dens > 0
    colour = np.zeros_like(accum)
    colour[mask] = accum[mask] / dens[mask, None]          # mean hue per pixel
    bright = np.zeros((H, W), np.float32)
    bright[mask] = np.log1p(dens[mask])
    bright /= bright.max() + 1e-9
    bright = bright ** 0.45                                 # gamma
    img = colour * bright[..., None]

    pil = Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8), "RGB")
    pil = pil.resize((OUT, OUT), Image.LANCZOS)            # supersample downscale
    return post(pil)


def post(pil):
    arr = np.asarray(pil, np.float32) / 255.0

    # bloom: blurred bright areas added back
    bloom = pil.filter(ImageFilter.GaussianBlur(OUT * 0.012))
    arr = 1 - (1 - arr) * (1 - 0.45 * np.asarray(bloom, np.float32) / 255.0)  # screen

    # vignette
    yy, xx = np.mgrid[-1:1:OUT * 1j, -1:1:OUT * 1j]
    vig = 1 - 0.28 * (xx ** 2 + yy ** 2)
    arr *= vig[..., None]

    # film grain
    g = np.random.default_rng(0).normal(0, 0.012, arr.shape)
    arr = np.clip(arr + g, 0, 1)

    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


if __name__ == "__main__":
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    seed = int(d.strftime("%Y%m%d"))
    print(f"date={d} seed={seed}  render {W}x{H} -> {OUT}")
    img = render(seed)
    name = f"v2_{seed}.png"
    img.save(name)
    print("wrote", name)
