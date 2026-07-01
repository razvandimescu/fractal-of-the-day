"""Release fractal + changelog from a commit range.

Reads the commits in a range (an explicit base..head for CI events, or the last tag /
last N commits locally), then:

  1. derives a deterministic seed from the range  -> unique & reproducible per release,
  2. maps repo signal -> visual params (the image is a fingerprint, not wallpaper),
  3. groups commit subjects into a changelog,
  4. renders the fractal with the engine (daily.py),
  5. writes a PR-comment-ready Markdown body + the PNG.

Public-repo only by design (see plans/PRODUCT_STRATEGY.md): with --private it emits a
waitlist no-op comment instead — that gate IS the demand signal for the paid tier.

Local:  python release_fractal.py [--max N] [--since REF]
CI:     python release_fractal.py --base $BASE --head $HEAD \
            --image-base-url https://raw.githubusercontent.com/$REPO/release-fractals \
            --private $IS_PRIVATE --out "$OUT"
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent          # engine home (daily.py, reference.png)
TARGET = Path(os.environ.get("GITHUB_WORKSPACE") or Path.cwd())   # repo being analyzed
os.chdir(REPO)                                  # so `import daily` + reference.png resolve
sys.path.insert(0, str(REPO))

from PIL import Image                            # noqa: E402
import daily                                     # noqa: E402
import changelog                                  # noqa: E402

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
US = "\x1f"
MARKER_START = "<!-- release-fractal:start -->"  # paired markers -> block is strippable
MARKER_END = "<!-- release-fractal:end -->"      # anywhere in the notes, not just the tail
RENDER_PX = 880                                  # ~3.6x the 240px display -> crisp on retina
DISPLAY_PX = 240                                 # small: notes text flows beside the float
WEBP_QUALITY = 82                                # ~150 KB for this art; ~12x smaller than PNG


def git(*args):
    return subprocess.run(["git", "-C", str(TARGET), *args],
                          capture_output=True, text=True).stdout.strip()


def _short(ref):
    r = ref or ""
    is_sha = len(r) == 40 and all(ch in "0123456789abcdef" for ch in r)
    return r[:7] if is_sha else r          # shorten SHAs, keep tag names intact


def resolve_range(base, head, since, max_n):
    if base:
        return base, f"{_short(base)}..{_short(head or 'HEAD')}", [f"{base}..{head or 'HEAD'}"]
    if head:                       # release with no previous tag -> everything up to the tag
        return None, _short(head), ["-n", str(max_n), head]
    if since:
        return since, since, [f"{since}..HEAD"]
    last_tag = git("describe", "--tags", "--abbrev=0")
    if last_tag:
        return last_tag, last_tag, [f"{last_tag}..HEAD"]
    return None, f"last {max_n} commits", ["-n", str(max_n)]


def gather(base, head, since, max_n):
    base_ref, label, log_args = resolve_range(base, head, since, max_n)
    fmt = US.join(["%H", "%h", "%an", "%s", "%ad"])
    raw = git("log", "--no-merges", f"--pretty=format:{fmt}", "--date=short", *log_args)
    commits = [dict(zip(("hash", "short", "author", "subject", "date"), ln.split(US)))
               for ln in raw.splitlines() if ln]

    if base_ref is None and commits:                  # no base: diff from root's parent
        oldest = commits[-1]["hash"]
        base_ref = git("rev-parse", "--verify", "--quiet", f"{oldest}^") or EMPTY_TREE
    ins = dele = files = 0
    if commits and base_ref:
        for ln in git("diff", "--numstat", f"{base_ref}..{head or 'HEAD'}").splitlines():
            a, d, *_ = (ln.split("\t") + ["", ""])[:3]
            ins += int(a) if a.isdigit() else 0
            dele += int(d) if d.isdigit() else 0
            files += 1

    dates = sorted(c["date"] for c in commits)
    return {
        "label": label, "commits": commits, "n_commits": len(commits),
        "n_authors": len({c["author"] for c in commits}), "files": files,
        "insertions": ins, "deletions": dele, "churn": ins + dele,
        "span": f"{dates[0]} → {dates[-1]}" if dates else "—",
    }


def derive_params(sig):
    """signal -> visual params; the mapping is the product."""
    seed_src = sig["label"] + "".join(c["hash"] for c in sig["commits"])
    seed = int(hashlib.sha256(seed_src.encode()).hexdigest()[:8], 16)
    churn_per_commit = sig["churn"] / max(sig["n_commits"], 1)
    return dict(
        seed=seed,
        focal=churn_per_commit >= 80,        # sweeping change -> radial burst
        use_ref=sig["n_authors"] >= 2,       # collaboration  -> sampled palette
        decorated=sig["n_commits"] >= 8,     # busy release   -> element overlay
    )


def comment_md(sig, image_url, changelog=None):
    # Image floats left (align= is GitHub-allowed; style= is stripped) so the heading, stats,
    # and whatever follows fill the row beside it. Leading with the image lifts it above the
    # feed card's "Read more" fold.
    head = f"""{MARKER_START}
<img align="left" width="{DISPLAY_PX}" height="{DISPLAY_PX}" src="{image_url}" alt="release fractal" />

### 🌀 Release fingerprint — {sig['label']}

*{sig['n_commits']} commits · {sig['n_authors']} contributor(s) · \
+{sig['insertions']}/-{sig['deletions']} across {sig['files']} files · {sig['span']}*
"""
    # keep mode: end the block early so the host's own notes flow beside the float.
    if not changelog:
        return head + f"{MARKER_END}\n"
    # generate mode: our changelog, then clear the float and close with the attribution.
    return head + f"""
{changelog}

<br clear="all" />

---
<sub>🎨 Generated from this change's commit signal — every release looks different.</sub>
{MARKER_END}
"""


def waitlist_md(waitlist_url):
    return f"""{MARKER_START}
## 🌀 Release fractal

Release fractals are currently **free for public repositories**. Private-repo support is
on the way — **[join the waitlist →]({waitlist_url})**.
{MARKER_END}
"""


def set_output(**kw):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for k, v in kw.items():
            f.write(f"{k}={v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--head")
    ap.add_argument("--since"); ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--image-base-url", default=".")
    ap.add_argument("--private", default="false")
    ap.add_argument("--waitlist-url", default="https://github.com/razvandimescu/fractal-of-the-day")
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--out", default="release_fractal_out")
    ap.add_argument("--model", default="", help="Ollama model for tier-1 classification; off if empty")
    ap.add_argument("--cache", default="", help="path to the classification cache JSON")
    ap.add_argument("--no-changelog", action="store_true",
                    help="keep mode: image + fingerprint only, so the host's own notes stand")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    comment_path = out / "comment.md"

    if a.private.lower() == "true":
        comment_path.write_text(waitlist_md(a.waitlist_url))
        set_output(skipped="true", comment_path=str(comment_path))
        print("private repo -> waitlist no-op (demand signal)")
        return

    sig = gather(a.base, a.head, a.since, a.max)
    if not sig["commits"]:
        comment_path.write_text(f"{MARKER_START}\n_No commits to summarize._\n{MARKER_END}\n")
        set_output(skipped="true", comment_path=str(comment_path))
        print("no commits in range")
        return

    p = derive_params(sig)
    image_name = f"fractal_{p['seed']}.webp"
    image_url = f"{a.image_base_url.rstrip('/')}/{image_name}"
    print(f"range={sig['label']} commits={sig['n_commits']} authors={sig['n_authors']} "
          f"churn=+{sig['insertions']}/-{sig['deletions']}")
    print(f"seed={p['seed']} focal={p['focal']} use_ref={p['use_ref']} decorated={p['decorated']}")
    print("rendering…")

    img, _ = daily.render_combo(p["seed"], p["focal"], p["use_ref"],
                                n_candidates=a.candidates, scorer=daily.Scorer("heuristic"),
                                decorated=p["decorated"])
    image_path = out / image_name
    img.resize((RENDER_PX, RENDER_PX), Image.LANCZOS).save(
        image_path, "WEBP", quality=WEBP_QUALITY, method=6)
    log = None
    if not a.no_changelog:
        classifier = changelog.Classifier(model=a.model or None, cache_path=a.cache or None)
        log = changelog.changelog_md(sig["commits"], classifier)
    comment_path.write_text(comment_md(sig, image_url, log))

    set_output(skipped="false", seed=str(p["seed"]),
               image_path=str(image_path), image_name=image_name,
               comment_path=str(comment_path))
    print(f"wrote {image_path} + {comment_path}")


if __name__ == "__main__":
    main()
