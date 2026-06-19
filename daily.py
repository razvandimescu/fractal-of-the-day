"""Daily generative art — pluggable palettes & styles + generate-N-pick-best.

Per date the seed picks a STYLE (marble / focal) and a PALETTE source
(cosine / sampled-from-reference), renders N cheap preview candidates, scores them
with a heuristic aesthetic metric, and re-renders the winner at full quality.

All CPU, deterministic per date.
Refs: iquilezles.org/articles/{warp,palettes} ; tylerxhobbs.com/words/flow-fields
"""
import colorsys
import json
import sys
from datetime import date

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REF = "reference.png"

PREVIEW = dict(OUT=384, SS=1, n=14000, steps=170)
FULL = dict(OUT=1024, SS=2, n=45000, steps=320)


# ---------- deterministic value-noise fBm ----------

def make_fbm(seed, octaves=5):
    rng = np.random.default_rng(seed)
    perm = np.concatenate([rng.permutation(256)] * 2).astype(np.int32)

    def vnoise(x, y):
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)
        xf, yf = x - xi, y - yi
        u = xf * xf * (3 - 2 * xf)
        v = yf * yf * (3 - 2 * yf)
        xi &= 255
        yi &= 255

        def corner(dx, dy):
            return perm[(perm[(xi + dx) & 255] + ((yi + dy) & 255)) & 255] / 255.0

        aa, ba, ab, bb = corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)
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


def warped_angle(fbm, x, y, warp=0.35):
    qx = fbm(x, y)
    qy = fbm(x + 5.2, y + 1.3)
    return fbm(x + 4.0 * warp * qx, y + 4.0 * warp * qy) * 4.0 * np.pi


# ---------- palettes (return pal(t)->(...,3) in [0,1]) ----------

def cosine_palette(rng):
    a = rng.uniform(0.50, 0.62, 3)
    b = rng.uniform(0.35, 0.48, 3)
    c = rng.uniform(0.7, 1.1, 3)
    d = rng.uniform(0.0, 1.0, 3)

    def pal(t):
        t = t[..., None]
        return np.clip(a + b * np.cos(2 * np.pi * (c * t + d)), 0, 1)

    pal.meta = {"type": "cosine", "a": a.tolist(), "b": b.tolist(),
                "c": c.tolist(), "d": d.tolist()}
    return pal


def reference_palette(rng, path=REF, k=12):
    """Sample vivid dominant colours from a reference image -> smooth gradient."""
    im = Image.open(path).convert("RGB").resize((128, 128))
    q = im.quantize(colors=k, method=Image.MEDIANCUT)
    cols = np.array(q.getpalette()[: k * 3]).reshape(-1, 3)[:k] / 255.0
    lum = cols @ [0.299, 0.587, 0.114]
    sat = cols.max(1) - cols.min(1)
    keep = cols[(lum > 0.12) & (sat > 0.10)]          # drop near-black / washed
    if len(keep) < 3:
        keep = cols[np.argsort(lum)[-4:]]
    keep = np.array([                                  # boost vividness (source is dark)
        colorsys.hsv_to_rgb(h, min(1, s * 1.7 + 0.15), min(1, v * 1.25 + 0.18))
        for h, s, v in (colorsys.rgb_to_hsv(*c) for c in keep)
    ])
    keep = keep[np.argsort(keep @ [0.299, 0.587, 0.114])]   # order by luminance
    keep = np.roll(keep, rng.integers(0, len(keep)), axis=0)  # vary entry hue
    stops = np.linspace(0, 1, len(keep))

    def pal(t):
        t = np.clip(t, 0, 1)
        return np.stack([np.interp(t, stops, keep[:, c]) for c in range(3)], -1)

    pal.meta = {"type": "reference", "stops": [[float(v) for v in c] for c in keep]}
    return pal


# ---------- core render (flow field; focal toggles composition) ----------

def integrate(seed, rng, pal, focal, OUT, SS, n, steps):
    W = H = OUT * SS
    fbm = make_fbm(seed)
    ns = 0.006 / SS * (OUT / 1024)
    step_len = 0.0022 * W
    cx, cy = W / 2, H / 2

    if focal:
        r = np.abs(rng.normal(0, 0.16 * W, n))
        th = rng.uniform(0, 2 * np.pi, n)
        px, py = cx + r * np.cos(th), cy + r * np.sin(th)
        margin = 0.05 * W
    else:
        margin = 0.18 * W
        px = rng.uniform(-margin, W + margin, n)
        py = rng.uniform(-margin, H + margin, n)

    base_t = fbm(px * ns * 0.4 + 11.0, py * ns * 0.4 + 7.0)
    base_t = (base_t - base_t.min()) / (np.ptp(base_t) + 1e-9)

    accum = np.zeros((H, W, 3), np.float32)
    dens = np.zeros((H, W), np.float32)
    alive = np.ones(n, bool)
    brush = [(0, 0, 1.0), (1, 0, 0.4), (-1, 0, 0.4), (0, 1, 0.4), (0, -1, 0.4)]

    for s in range(steps):
        ang = warped_angle(fbm, px * ns, py * ns)
        dx, dy = np.cos(ang), np.sin(ang)
        if focal:                                  # blend flow with outward radial
            ox, oy = px - cx, py - cy
            rr = np.hypot(ox, oy) + 1e-6
            dx, dy = 0.55 * dx + 0.45 * ox / rr, 0.55 * dy + 0.45 * oy / rr
            nn = np.hypot(dx, dy) + 1e-9
            dx, dy = dx / nn, dy / nn
        px, py = px + dx * step_len, py + dy * step_len

        ix, iy = px.astype(np.int32), py.astype(np.int32)
        ok = alive & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        alive &= (ix >= -margin) & (ix < W + margin) & (iy >= -margin) & (iy < H + margin)
        col = pal((base_t[ok] + 0.0008 * s) % 1.0)
        bx, by = ix[ok], iy[ok]
        for ox, oy, w in brush:
            np.add.at(accum, (np.clip(by + oy, 0, H - 1), np.clip(bx + ox, 0, W - 1)), col * w)
            np.add.at(dens, (np.clip(by + oy, 0, H - 1), np.clip(bx + ox, 0, W - 1)), w)

    mask = dens > 0
    colour = np.zeros_like(accum)
    colour[mask] = accum[mask] / dens[mask, None]
    bright = np.zeros((H, W), np.float32)
    bright[mask] = np.log1p(dens[mask])
    bright = (bright / (bright.max() + 1e-9)) ** 0.45
    return colour * bright[..., None]              # float HxWx3 in [0,1]


# ---------- aesthetic scoring (higher = better) ----------

def heuristic_score(a):
    """Free, instant: colourfulness + contrast + coverage + saturation."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    colourful = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    lit = lum > 0.06
    coverage = 1 - abs(lit.mean() - 0.45) / 0.45    # peak near 45% lit
    sat = (a.max(-1) - a.min(-1)) / (a.max(-1) + 1e-6)
    satscore = sat[lit].mean() if lit.any() else 0.0
    return 2.2 * colourful + 1.4 * lum.std() + 0.8 * coverage + 1.0 * satscore


class Scorer:
    """Pluggable aesthetic fitness. mode='heuristic' (default) or 'aesthetic'
    (LAION CLIP+MLP via simple-aesthetics-predictor, CPU, lazy-loaded). Falls back
    to the heuristic if the learned model or its deps are unavailable."""

    MODEL = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

    def __init__(self, mode="heuristic"):
        self.mode = mode
        self._net = None

    def _load(self):
        import torch
        from transformers import CLIPProcessor
        from aesthetics_predictor import AestheticsPredictorV1
        self._torch = torch
        self._proc = CLIPProcessor.from_pretrained(self.MODEL)
        self._net = AestheticsPredictorV1.from_pretrained(self.MODEL).eval()

    def score(self, arr):
        if self.mode != "aesthetic":
            return heuristic_score(arr)
        if self._net is None:
            try:
                self._load()
            except Exception as e:               # missing deps / weights -> degrade
                print(f"  [scorer] learned model unavailable ({e}); using heuristic")
                self.mode = "heuristic"
                return heuristic_score(arr)
        img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8), "RGB")
        inp = self._proc(images=img, return_tensors="pt")
        with self._torch.no_grad():
            return float(self._net(**inp).logits.squeeze())


# ---------- post ----------

def post(arr):
    OUT = arr.shape[0]
    pil = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8), "RGB")
    a = np.asarray(pil, np.float32) / 255.0
    bloom = np.asarray(pil.filter(ImageFilter.GaussianBlur(OUT * 0.012)), np.float32) / 255.0
    a = 1 - (1 - a) * (1 - 0.45 * bloom)            # screen bloom
    yy, xx = np.mgrid[-1:1:OUT * 1j, -1:1:OUT * 1j]
    a *= (1 - 0.30 * (xx ** 2 + yy ** 2))[..., None]
    a = np.clip(a + np.random.default_rng(0).normal(0, 0.012, a.shape), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8), "RGB")


# ---------- candidate + daily orchestration ----------

def candidate(cseed, focal, use_ref, q):
    prng = np.random.default_rng(cseed + 777)
    try:
        pal = reference_palette(prng) if use_ref else cosine_palette(prng)
    except FileNotFoundError:
        pal = cosine_palette(prng)
    return integrate(cseed, np.random.default_rng(cseed), pal, focal, **q), pal


# ---------- discrete-element overlay (rings / dots / diamonds / spirals) ----------

def decorate(base, rng, pal, focal):
    OUT = base.size[0]
    S = 2
    W = OUT * S

    def col(bmin, bmax):                           # palette colour, varied brightness
        c = np.asarray(pal(np.array([rng.random()]))).reshape(-1, 3)[0]
        s = rng.uniform(bmin, bmax)
        return tuple(int(min(1, float(x) * s) * 255) for x in c)

    def place():                                   # focal: cluster near centre
        if focal:
            return np.clip(rng.normal(0.5, 0.18, 2), 0.08, 0.92) * W
        return rng.uniform(0.1, 0.9, 2) * W

    def paint(dr, bmin, bmax, mul):
        for _ in range(int(rng.integers(2, 4) * mul)):        # concentric rings
            cx, cy = place()
            r0, gap = rng.uniform(0.03, 0.09) * W, rng.uniform(0.012, 0.03) * W
            for i in range(rng.integers(3, 8)):
                r = r0 + i * gap
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col(bmin, bmax), width=int(S * 1.4))
        for _ in range(int(rng.integers(3, 6) * mul)):        # halftone dot clusters
            cx, cy = place()
            sp = rng.uniform(0.014, 0.026) * W
            rad = sp * rng.uniform(0.14, 0.28)
            c = col(bmin, bmax)
            for ix in range(rng.integers(3, 7)):
                for iy in range(rng.integers(3, 7)):
                    x, y = cx + ix * sp, cy + iy * sp
                    dr.ellipse([x - rad, y - rad, x + rad, y + rad], fill=c)
        for _ in range(int(rng.integers(3, 6) * mul)):        # squares / diamonds
            cx, cy = place()
            s, ang = rng.uniform(0.008, 0.024) * W, rng.uniform(0, np.pi / 2)
            pts = [(cx + s * np.cos(ang + k * np.pi / 2), cy + s * np.sin(ang + k * np.pi / 2)) for k in range(4)]
            if rng.random() < 0.5:
                dr.polygon(pts, outline=col(bmin, bmax), width=int(S))
            else:
                dr.polygon(pts, fill=col(bmin, bmax))
        for _ in range(rng.integers(0, 3)):                   # spirals
            cx, cy = place()
            turns, rmax = rng.uniform(2, 4), rng.uniform(0.04, 0.09) * W
            pts = [(cx + (k / 140 * rmax) * np.cos(k / 140 * turns * 2 * np.pi),
                    cy + (k / 140 * rmax) * np.sin(k / 140 * turns * 2 * np.pi)) for k in range(140)]
            dr.line(pts, fill=col(bmin, bmax), width=int(S * 1.1), joint="curve")

    back = Image.new("RGB", (W, W), 0)             # dim, blurred -> recedes behind
    paint(ImageDraw.Draw(back), 0.20, 0.45, 1.0)
    back = back.filter(ImageFilter.GaussianBlur(W * 0.005))
    front = Image.new("RGB", (W, W), 0)            # bright, crisp -> pops forward
    paint(ImageDraw.Draw(front), 0.55, 1.0, 0.8)

    a = np.asarray(base, np.float32) / 255
    for layer, k in ((back, 0.7), (front, 0.92)):
        b = np.asarray(layer.resize((OUT, OUT), Image.LANCZOS), np.float32) / 255
        a = 1 - (1 - a) * (1 - b * k)              # screen
    return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8), "RGB")


def render_combo(seed, focal, use_ref, n_candidates=10, scorer=None, decorated=None):
    """Generate-N-pick-best for an explicit style/palette combo -> (img, params)."""
    scorer = scorer or Scorer()
    if decorated is None:
        decorated = focal
    best = (-1e9, None)
    for i in range(n_candidates):
        cseed = seed * 1000 + i
        sc = scorer.score(candidate(cseed, focal, use_ref, PREVIEW)[0])
        if sc > best[0]:
            best = (sc, cseed)

    full, pal = candidate(best[1], focal, use_ref, FULL)
    down = np.asarray(Image.fromarray((full.clip(0, 1) * 255).astype(np.uint8))
                      .resize((FULL["OUT"], FULL["OUT"]), Image.LANCZOS), np.float32) / 255.0
    img = post(down)
    if decorated:
        img = decorate(img, np.random.default_rng(best[1] + 999), pal, focal)

    style = f"{'focal' if focal else 'marble'}/{'reference' if use_ref else 'cosine'}"
    style += "+elements" if decorated else ""
    params = {
        "seed": seed, "style": style,
        "composition": "focal" if focal else "marble",
        "palette": pal.meta, "elements": decorated, "scorer": scorer.mode,
        "candidates": n_candidates, "score": round(float(best[0]), 4),
        "winning_candidate": best[1] - seed * 1000,
    }
    return img, params


def make_day(seed, n_candidates=10, scorer=None):
    drng = np.random.default_rng(seed)
    focal = bool(drng.random() < 0.45)
    use_ref = bool(drng.random() < 0.5)
    decorated = focal or drng.random() < 0.35
    img, params = render_combo(seed, focal, use_ref, n_candidates, scorer, decorated)
    print(f"  style={params['style']}  best candidate score={params['score']:.3f}")
    return img, params["style"], params


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode = "aesthetic" if "--aesthetic" in sys.argv else "heuristic"
    d = date.fromisoformat(argv[0]) if argv else date.today()
    n = int(argv[1]) if len(argv) > 1 else 10
    seed = int(d.strftime("%Y%m%d"))
    print(f"date={d} seed={seed}  candidates={n}  scorer={mode}")
    img, style, params = make_day(seed, n, Scorer(mode))
    params["date"] = d.isoformat()
    img.save(f"day_{seed}.png")
    with open(f"day_{seed}.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"wrote day_{seed}.png + .json  {style}")
