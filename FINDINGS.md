# Fractal-of-the-Day — Research Findings & Build Plan

Goal: a **free, CPU-only, deterministic-per-date** "generative art of the day" pipeline
(one image/day, runnable in a GitHub Action) whose output approaches the look of
art-directed abstract diffusion art — rich palettes, layered swirls/ribbons/rings,
texture, intentional composition.

Reference aesthetic we're chasing: `https://dak.li/blog/datakraut/randomness.webp`
(a diffusion-model image — not actually a fractal).

Source: multi-source deep-research pass (27 sources, 117 claims, 22 confirmed via
3-vote adversarial verification, 3 refuted). Citations inline.

---

## Key reframing

The reference look is **not** best reproduced by IFS fractal flames. The dominant
driver of "designed chaos" (layered swirls/ribbons) is **domain warping + flow
fields**, which are more controllable, fully deterministic, and CPU-cheap. Flames
are one ingredient, not the foundation.

Fractals are self-similar by definition → monotone. The reference's appeal is
*deliberate variety*. We manufacture that variety with warping, flow, curated
palettes, and compositing — then select the best of many candidates.

---

## Levers, ranked (with concrete settings)

### 1. Cosine palettes (Inigo Quilez) — cheapest, highest-leverage color fix ✅
`color(t) = a + b·cos(2π·(c·t + d))`, all RGB 3-vectors.
Canonical balanced full-spectrum: `a=(.5,.5,.5)`, `b=(.5,.5,.5)`, `c=(1,1,1)`,
`d=(0,.33,.67)`. Deterministic, zero-cost, smooth — fixes muddy 2-tone palettes.
Vary all four params per date for daily variety.
- Source: https://iquilezles.org/articles/palettes/ (primary)
- ⚠️ Refuted: the claim that varying phase `d` *alone* controls hue — `a,b,c` also
  shape it. Vary all four.

### 2. Domain warping (Quilez) — the swirl/ribbon engine ✅
Replace `f(p)` with `f(p + h(p))`, `h` itself fBm. Nested form:
```
q = (fbm(p),         fbm(p + (5.2,1.3)))
pattern = fbm(p + 4·q)
r = (fbm(p+4q+(1.7,9.2)), fbm(p+4q+(8.3,2.8)))
pattern = fbm(p + 4·r)
```
THE technique for "layered designed chaos." CPU-cheap, deterministic.
- Source: https://iquilezles.org/articles/warp/ (primary)

### 3. Flow fields (Tyler Hobbs) — fluid directional ribbons ✅
- Angle grid = Perlin noise, **input coords ×0.005**, map [0,1] → [0,2π].
- Grid resolution **0.5% of image width**.
- Extend grid **50% beyond canvas** so curves can re-enter.
- Particle **step length 0.1–0.5% of width** (smaller = cleaner).
- Curve length sets character: short → "fur" texture; long → fluid flow. Use
  short-to-medium when blending multiple colors.
- Source: https://www.tylerxhobbs.com/words/flow-fields (primary)
- ⚠️ Doc's "10px for 1000px image" example is a factor-of-2 slip; trust the 0.5% prose.

### 4. Color the right way (Tyler Hobbs) ✅
- Work in **HSB, not RGB** (independent hue/sat/bright control).
- Define palettes as **weighted probability distributions** over a small curated set
  (e.g. 70% navy / 20% orange / 10% cream — matches the reference).
- **Clump similar colors** spatially to build perceptible shapes.
- **Sample palettes from a reference photo** (incl. `randomness.webp`) via k-means
  dominant-color clustering, then shift.
- For gradient *interpolation*, prefer perceptually-uniform **OKLCH/OKLab** over HSB
  to avoid muddy middles.
- Sources: https://www.tylerxhobbs.com/words/working-with-color-in-generative-art,
  https://www.tylerxhobbs.com/words/color-arrangment-in-generative-art (primary),
  OKLab: https://bottosson.github.io/posts/oklab/

### 5. Flame core: supersample + log-density tone mapping ✅
Render at **2–4× supersample with subpixel jitter, then box-downsample**;
`earlyclip=1`. This anti-aliasing-then-downscale is the documented polished-vs-amateur
differentiator. Log-density brightness + gamma (~0.45).
- Sources: https://github.com/scottdraves/flam3, https://www.ecsoft2.org/flam3-flame-algorithm
- ⚠️ Refuted/unverified: specific density-estimation kernel params
  (estimator_radius/curve/minimum, Epanechnikov→Gaussian) and the "98 variations 0–97"
  list. Don't rely on those internals.

### 6. Automated aesthetic selection — generate-N-pick-best ✅
- **LAION CLIP+MLP** aesthetic predictor: tiny MLP over 768-dim CLIP ViT-L/14
  embeddings. Hidden cost = the **CLIP encode at ~0.5–2s/image** (the MLP is free).
- **idealo NIMA-MobileNet** (SRCC 0.61 / LCC 0.626 with human ratings): **no CLIP
  needed**, ships `Dockerfile.cpu` → better fit for a GitHub Action time budget.
  Caveat: TF1.x-era Keras, needs a pinned env.
- **aesthetic-predictor-v2.5** (SigLIP-based) supersedes both — worth evaluating.
- SRCC 0.61 = "meaningfully better than random," not human taste. Pair with the
  GenerativeGI **noise-filter** trick (CNN feature-distance) to drop blurry/empty
  candidates first.
- Sources: https://github.com/christophschuhmann/improved-aesthetic-predictor,
  https://github.com/idealo/image-quality-assessment, https://arxiv.org/abs/1709.05424,
  GenerativeGI: https://arxiv.org/abs/2407.20095

### Bonus — harmonize toward a target palette ✅
RGBXY convex-hull palette decomposition recolors a 6 MP image in ~20 ms, solver-free —
a cheap pass to push any render toward a chosen harmony.
- Sources: https://arxiv.org/pdf/1804.01225, https://cragl.cs.gmu.edu/fastlayers/

---

## The honest gap

**Diffusion tier (fractal skeleton + img2img/ControlNet, Flux/SDXL low-denoise)
produced ZERO surviving evidence.** Cost/quality/feasibility unverified. Downgraded
from "nuclear option" to "untested — prototype before believing." Kept out of the
core path; revisit only if the free path plateaus.

---

## Target pipeline (v2)

```
Core:    flow-field + domain-warp render (swirl engine), date-seeded
Palette: cosine palette OR k-means-from-reference, weighted-probability assignment,
         spatial colour clumping
Render:  supersample 2–3× + box/LANCZOS downscale; additive accumulation + log-density
Post:    light bloom + film grain + vignette (+ optional halftone)
Select:  generate ~20–30 candidates → NIMA-MobileNet pick-best (+ noise prefilter)
```
All free, CPU, deterministic per date.

---

## Roadmap / status

- [x] v1 baseline: flame / Julia / Newton (`generate.py`) — established the gap.
- [x] **v2 core** (`v2.py`): flow-field + domain-warp + cosine palettes + supersample + post.
- [x] **Daily pipeline** (`daily.py`): pluggable palettes + styles + generate-N-pick-best.
  - [x] Palettes: cosine (Quilez) + sampled-from-reference (PIL MEDIANCUT + HSV vividness boost).
  - [x] Styles: `marble` (all-over flow) + `focal` (radial explosion w/ negative space).
        Per-date seed picks style+palette → days differ in character.
  - [x] Aesthetic selection: heuristic scorer (colorfulness + contrast + coverage + sat),
        N cheap previews → re-render winner full quality.
- [x] Composition polish v1: discrete-element overlay (`decorate`) — rings, halftone dot
      clusters, diamonds, spirals; palette-coloured, screen-blended; focal-biased placement.
  - [x] Taste pass: back (dim+blurred) vs front (crisp) depth layers, per-element
        brightness/scale variation, smaller dot clusters. Reads integrated, not stamped.
- [x] Aesthetic model upgrade: pluggable `Scorer` — heuristic (default, free) +
      `--aesthetic` LAION CLIP+MLP (simple-aesthetics-predictor, CPU, lazy-loaded,
      graceful fallback). Validated: on 2026-01-01 the two scorers DISAGREE — heuristic
      picks the high-contrast yellow/blue (#0), CLIP picks the richer multi-hue (#3,
      aesthetic 4.255), which is visibly more reference-like. Learned scorer adds real taste.
- [ ] Daily packaging: GitHub Action cron → commit → GitHub Pages archive.  ← LAST

### Aesthetic scorer trade-offs (measured)
- One-time ~1.7 GB CLIP ViT-L/14 download (cache in the Action via HF cache); per-candidate
  CPU encode ~0.5–2 s. Fine for one image/day; the cost is the N preview encodes.
- Extra deps (NOT in core path): `torch`, `transformers`, `simple-aesthetics-predictor`.
  Default pipeline stays heuristic-only and dependency-free; learned model is opt-in.
- LAION SRCC ~0.61 — better-than-random taste, not human-level. Worth it here because
  one good pick/day matters and the cost is amortised over a daily cadence.
- [x] Interactive site MVP (`build_site.py` + `docs/`): exploits determinism instead of a
      static gallery — daily image + its recipe, a live cosine-palette tuner (the actual
      formula, client-side), a same-seed style/palette comparator, permalink/export.
      Reviewed via the "Unreasonable Effectiveness of HTML" lens (show-don't-describe).
      Fonts: Space Grotesk / DM Sans / JetBrains Mono (shared with rinkt_neville).
- [ ] Packaging: GitHub Action cron → `build_site.py` → commit `docs/` → GitHub Pages. ← LAST
- [ ] (Optional, untested) diffusion-skin tier — prototype & measure before adopting.

### Status notes
- Reference-palette days score ~0.7 vs ~2.0 for cosine (source image is mostly black →
  muted). Kept as a deliberate "moody" minority style after HSV boost. Could instead let
  the scorer choose palette source per-day (would mostly pick cosine) if uniform vividness
  is preferred over variety.
- Heuristic scorer optimizes colorfulness/contrast — fast, no deps. Watch for garish bias;
  NIMA/CLIP would judge "taste" better but adds a heavy (TF/torch) dependency.

## Open questions (from research)

- Real end-to-end CPU wall-clock for generate-N-pick-best in an Action (N × CLIP
  encode); does NIMA-MobileNet or a smaller SigLIP fit the time limit better?
- Does aesthetic-predictor-v2.5 materially beat CLIP+MLP / NIMA as a fitness function,
  CPU-runnable?
- Post-processing ordering/params to emulate diffusion texture (no claim survived).
- Diffusion-skin: denoise strength / ControlNet weight that preserves composition at
  ~$0.003/day — unverified.
