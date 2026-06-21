"""Companion artifacts from the SAME day-genome: an SVG mark and an ambient track.

Both consume the day's seed + cosine palette so they rhyme with the daily image.
Pure-numpy / stdlib, deterministic per date. No new deps.

  svg = render_icon(seed, palette)   -> compact streamline mark (favicon / card accent)
  render_song(seed, palette, path)   -> ~24s ambient WAV (pentatonic drone, forgiving)
"""
import struct
import wave

import numpy as np

from daily import make_fbm, warped_angle


# ---------- shared: palette as a callable from cosine meta ----------

def _pal_from_meta(meta):
    if not meta or meta.get("type") != "cosine":
        a, b = np.array([.5, .55, .5]), np.array([.45, .45, .45])
        c, d = np.array([1., .9, .9]), np.array([0., .33, .67])
    else:
        a, b, c, d = (np.array(meta[k]) for k in "abcd")

    def pal(t):
        t = np.asarray(t, float)[..., None]
        return np.clip(a + b * np.cos(2 * np.pi * (c * t + d)), 0, 1)
    return pal


# ---------- SVG mark: a few streamlines traced through the day's own flow field ----------

def _hex(c):
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(c, 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


def _rdp(pts, eps):                                        # Ramer–Douglas–Peucker: drop redundant vertices
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab = pts[i], pts[j] - pts[i]
        L = np.hypot(*ab) + 1e-9
        seg = pts[i + 1:j]
        d = np.abs((seg[:, 0] - a[0]) * ab[1] - (seg[:, 1] - a[1]) * ab[0]) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def _catmull(pts, m=12):                                   # smooth centreline through the kept points
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    P = np.vstack([pts[0], pts, pts[-1]])
    t = np.linspace(0, 1, m, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [pts[-1][None]])


def _ribbon_path(center, w_max):                           # outline a centreline as a tapered filled ribbon
    c = np.asarray(center, float)
    if len(c) < 3:
        return None
    s = np.linspace(0, 1, len(c))
    width = w_max * (0.08 + 0.92 * np.sin(np.pi * s) ** 0.85)  # calligraphic taper toward both ends
    tan = np.gradient(c, axis=0)
    tan /= np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-9
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    poly = np.vstack([c + nrm * (width / 2)[:, None],
                      (c - nrm * (width / 2)[:, None])[::-1]])
    return "M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in poly) + "Z"


def _streamline(fbm, sx, sy, ns, step, half, lo, hi):      # trace both ways from a seed -> crosses centre
    def walk(sgn):
        x, y, pts = sx, sy, []
        for _ in range(half):
            ang = float(warped_angle(fbm, np.array([x * ns]), np.array([y * ns]))[0])
            x += sgn * np.cos(ang) * step
            y += sgn * np.sin(ang) * step
            if not (lo < x < hi and lo < y < hi):
                break
            pts.append((x, y))
        return pts
    return walk(-1.0)[::-1] + [(sx, sy)] + walk(1.0)


def render_icon(seed, palette=None, mode="icon", W=64):
    """Compact brand mark from the day's flow field.
    mode='icon'   -> few bold tapered ribbons, centred, legible down to a favicon.
    mode='poster' -> the dense streamline version (decorative, not for small sizes)."""
    if mode == "poster":
        return _render_poster(seed, palette, W)

    fbm = make_fbm(seed)
    rng = np.random.default_rng(seed + 5)
    pal = _pal_from_meta(palette)
    cx, cy, pad = W / 2, W / 2, 0.10 * W
    ns, step, half = 0.03, 0.5, 64
    N = 5
    stops = np.linspace(0.16, 0.92, N)
    angles = rng.permutation(np.linspace(0, np.pi, N, endpoint=False)) + rng.uniform(0, np.pi)

    ribbons = []
    for i in range(N):
        rr = rng.uniform(0.02, 0.11) * W
        sx, sy = cx + rr * np.cos(angles[i]), cy + rr * np.sin(angles[i])
        raw = _streamline(fbm, sx, sy, ns, step, half, pad, W - pad)
        if len(raw) < 12:
            continue
        path = _ribbon_path(_catmull(_rdp(raw, 0.4)), w_max=rng.uniform(4.0, 6.5))
        if path:
            ribbons.append((path, _hex(np.asarray(pal(stops[i])) * 1.12 + 0.04)))

    c0 = _hex(np.asarray(pal(0.5)) * 0.16 + 0.04)          # palette-tinted dark backdrop
    body = "\n".join(
        f'  <path d="{d}" fill="{col}" stroke="#0a0a0d" stroke-width="0.5"/>'
        for d, col in ribbons)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {W}" width="{W}" height="{W}">\n'
        f'  <defs><radialGradient id="bg" cx="50%" cy="42%" r="75%">\n'
        f'    <stop offset="0" stop-color="{c0}"/><stop offset="1" stop-color="#08080b"/>\n'
        f'  </radialGradient></defs>\n'
        f'  <rect width="{W}" height="{W}" rx="12" fill="url(#bg)"/>\n'
        f'{body}\n</svg>\n'
    )


def _render_poster(seed, palette=None, W=64, n=16, steps=300):
    fbm = make_fbm(seed)
    rng = np.random.default_rng(seed + 5)
    pal = _pal_from_meta(palette)
    ns, step = 0.028, 0.5

    lines = []
    for _ in range(n):
        x = float(rng.uniform(0.12, 0.88) * W)
        y = float(rng.uniform(0.12, 0.88) * W)
        pts = [(x, y)]
        for _ in range(steps):
            ang = float(warped_angle(fbm, np.array([x * ns]), np.array([y * ns]))[0])
            x += np.cos(ang) * step
            y += np.sin(ang) * step
            if not (1.5 < x < W - 1.5 and 1.5 < y < W - 1.5):
                break
            pts.append((round(x, 2), round(y, 2)))
        if len(pts) > 18:
            lines.append((pts, _hex(pal(rng.random()))))

    paths = "\n".join(
        f'  <polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" '
        f'fill="none" stroke="{col}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>'
        for pts, col in lines
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {W}" width="{W}" height="{W}">\n'
        f'  <rect width="{W}" height="{W}" rx="12" fill="#0b0b0e"/>\n'
        f'{paths}\n</svg>\n'
    )


# ---------- ambient song: pentatonic drone, palette sets brightness ----------

_PENTA = [0, 2, 4, 7, 9]          # major pentatonic — forgiving under randomness


def _adsr(n, sr, a=0.4, r=0.6):
    env = np.ones(n)
    na = min(int(a * sr), n // 2)
    nr = min(int(r * sr), n - na)
    if na > 0:
        env[:na] = np.linspace(0, 1, na)
    if nr > 0:
        env[-nr:] = np.linspace(1, 0, nr)
    return env


def render_song(seed, palette=None, path="today.wav", dur=24, sr=22050):
    rng = np.random.default_rng(seed)
    pal = _pal_from_meta(palette)
    lum = float(np.mean(pal(np.linspace(0, 1, 16)) @ [0.299, 0.587, 0.114]))
    n_harm = 2 + int(round(lum * 5))                       # brighter palette -> richer timbre

    root = 110.0 * 2 ** (int(rng.integers(0, 7)) / 12.0)   # low root, A2-ish
    N = int(dur * sr)
    t = np.arange(N) / sr
    out = np.zeros(N)

    def tone(freq, amp, start, length):
        s = int(start * sr)
        n = int(length * sr)
        if s >= N:
            return
        n = min(n, N - s)
        tt = np.arange(n) / sr
        wave_ = sum(
            (1.0 / h) * np.sin(2 * np.pi * freq * h * tt)
            for h in range(1, n_harm + 1)
        )
        out[s:s + n] += amp * _adsr(n, sr) * wave_

    chords = [[0, 2, 4], [0, 2, 4], [-3, 0, 2], [2, 4, 7]]  # slow I–I–vi–IV-ish drift
    bar = dur / len(chords)
    for i, ch in enumerate(chords):                          # sustained pads
        for deg in ch:
            f = root * 2 ** (_PENTA[deg % len(_PENTA)] / 12.0 + deg // len(_PENTA))
            tone(f, 0.16, i * bar, bar + 0.5)

    for k in range(int(dur * 1.6)):                          # sparse high arpeggio
        deg = int(rng.integers(0, len(_PENTA)))
        f = root * 4 * 2 ** (_PENTA[deg] / 12.0)
        tone(f, 0.06, rng.uniform(0, dur - 1), rng.uniform(0.4, 1.1))

    # cheap reverb: a couple of decaying delay taps
    for delay, decay in ((0.07, 0.35), (0.11, 0.22)):
        d = int(delay * sr)
        out[d:] += decay * out[:-d]

    out *= _adsr(N, sr, a=1.5, r=2.5)                        # global fade in/out
    out /= (np.max(np.abs(out)) + 1e-9)
    pcm = (out * 0.9 * 32767).astype(np.int16)

    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", v) for v in pcm))
    return path


if __name__ == "__main__":
    import json
    import sys
    from datetime import date
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    seed = int(d.strftime("%Y%m%d"))
    meta = None
    try:
        meta = json.load(open(f"docs/data/today.json"))["palette"]
    except Exception:
        pass
    open(f"icon_{seed}.svg", "w").write(render_icon(seed, meta))
    render_song(seed, meta, f"song_{seed}.wav")
    print(f"wrote icon_{seed}.svg + song_{seed}.wav")
