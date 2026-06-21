"""Build the static assets the interactive site reads.

Renders, for a given date:
  docs/img/today.png        + docs/data/today.json   (the daily pick + its recipe)
  docs/img/<comp>_<pal>.png (4 forced style/palette corners for the comparator)

The page itself (docs/index.html) is static and reads these. Deterministic per date.

Usage: python build_site.py [YYYY-MM-DD] [N] [--aesthetic]
"""
import json
import sys
from datetime import date
from pathlib import Path

import daily
import extras

DOCS = Path("docs")
COMBOS = [("marble", "cosine"), ("marble", "reference"),
          ("focal", "cosine"), ("focal", "reference")]


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode = "aesthetic" if "--aesthetic" in sys.argv else "heuristic"
    d = date.fromisoformat(argv[0]) if argv else date.today()
    n = int(argv[1]) if len(argv) > 1 else 10
    seed = int(d.strftime("%Y%m%d"))
    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)

    (DOCS / "audio").mkdir(parents=True, exist_ok=True)

    print(f"[build] {d} seed={seed} scorer={mode}")
    img, _, params = daily.make_day(seed, n, daily.Scorer(mode))
    params["date"] = d.isoformat()
    img.save(DOCS / "img" / "today.png")

    pal = params["palette"]                               # companion artifacts: same genome
    (DOCS / "img" / "today.svg").write_text(extras.render_icon(seed, pal, mode="icon"))
    extras.render_song(seed, pal, str(DOCS / "audio" / "today.wav"))
    params["mark"], params["song"] = "img/today.svg", "audio/today.wav"
    (DOCS / "data" / "today.json").write_text(json.dumps(params, indent=2))
    print("  wrote today.png + today.svg + today.wav + today.json")

    corners = {}
    for comp, pal in COMBOS:                       # same seed, forced combos
        cimg, cparams = daily.render_combo(
            seed, focal=(comp == "focal"), use_ref=(pal == "reference"),
            n_candidates=max(4, n // 2), scorer=daily.Scorer(mode))
        fname = f"{comp}_{pal}.png"
        cimg.save(DOCS / "img" / fname)
        corners[f"{comp}/{pal}"] = {"img": f"img/{fname}", "palette": cparams["palette"]}
        print(f"  wrote {fname}")

    (DOCS / "data" / "corners.json").write_text(json.dumps(corners, indent=2))
    print("[build] done")


if __name__ == "__main__":
    main()
