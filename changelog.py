"""Tiered, cached commit classification for the release changelog.

  tier 0  conventional-commit prefix   free, deterministic, always on
  tier 1  local LLM (Ollama)           classifies the NON-conforming remainder
  cache   sha256(subject) -> category  subjects are immutable, so never reclassify

Mirrors daily.Scorer: a capable tier that degrades gracefully to the cheap one when its
backend is absent. The LLM only *picks a bucket* from a fixed enum — it never rewrites a
commit subject, so the changelog can never fabricate a change that didn't happen.
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

CATEGORIES = ["feat", "fix", "perf", "refactor", "docs", "test", "build", "ci", "chore", "other"]
SECTIONS = [
    ("feat", "✨ Features"), ("fix", "🐛 Fixes"), ("perf", "⚡ Performance"),
    ("refactor", "🧹 Refactoring"), ("docs", "📝 Docs"), ("test", "✅ Tests"),
    ("build", "📦 Build"), ("ci", "🤖 CI"), ("chore", "🔧 Chores"), ("other", "📋 Other"),
]
TYPE_RE = re.compile(r"^(\w+)(\([^)]*\))?(!)?:\s*(.*)$")


def conventional_type(subject):
    m = TYPE_RE.match(subject)
    return m.group(1) if m and m.group(1) in CATEGORIES else None


# Leading-verb -> category. Free, deterministic floor for imperative commits and the
# graceful-degradation target when the LLM is unavailable.
VERB_TYPE = {
    "add": "feat", "implement": "feat", "introduce": "feat", "create": "feat",
    "support": "feat", "enable": "feat",
    "fix": "fix", "resolve": "fix", "correct": "fix", "patch": "fix", "handle": "fix",
    "guard": "fix", "prevent": "fix", "tolerate": "fix", "repair": "fix",
    "optimize": "perf", "optimise": "perf", "speed": "perf",
    "refactor": "refactor", "rename": "refactor", "move": "refactor", "simplify": "refactor",
    "extract": "refactor", "restructure": "refactor", "reorganize": "refactor",
    "document": "docs", "doc": "docs", "docs": "docs",
    "test": "test", "cover": "test",
    "bump": "chore", "upgrade": "chore", "pin": "chore", "remove": "chore",
    "delete": "chore", "drop": "chore",
}


def heuristic_type(subject):
    m = re.match(r"[A-Za-z]+", subject)
    return VERB_TYPE.get(m.group(0).lower(), "other") if m else "other"


def _key(subject):
    return hashlib.sha256(subject.encode()).hexdigest()[:16]


class Classifier:
    """`classify(subjects) -> [category]`. Conventional commits resolve for free; the rest
    hit the cache, then the LLM (if configured). `backend` is an injectable
    `callable(list[str]) -> list[str]` used for tests in place of Ollama."""

    def __init__(self, model=None, cache_path=None, backend=None,
                 ollama_url="http://localhost:11434"):
        self.model = model
        self.backend = backend
        self.ollama_url = ollama_url
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text() or "{}")
            except json.JSONDecodeError:
                pass                                     # empty/corrupt cache -> start fresh

    def classify(self, subjects):
        out = [None] * len(subjects)
        misses = []
        for i, s in enumerate(subjects):
            t = conventional_type(s)
            if t:
                out[i] = t
            elif _key(s) in self.cache:
                out[i] = self.cache[_key(s)]
            else:
                misses.append((i, s))

        if misses:
            labels = None
            if self.model or self.backend:
                try:
                    labels = self._run_llm([s for _, s in misses])
                except Exception as e:                       # unreachable model -> heuristic
                    print(f"  [classifier] LLM unavailable ({e}); falling back to heuristic")
            if labels:
                for (i, s), lab in zip(misses, labels):
                    out[i] = lab if lab in CATEGORIES else heuristic_type(s)
                    self.cache[_key(s)] = out[i]             # cache only real LLM results
                self._save()
            else:
                for i, s in misses:                          # deterministic floor, not cached
                    out[i] = heuristic_type(s)
        return out

    def _run_llm(self, subjects):
        labels = (self.backend or self._ollama)(subjects)
        if len(labels) != len(subjects):
            raise ValueError(f"expected {len(subjects)} labels, got {len(labels)}")
        return labels

    def _ollama(self, subjects):
        numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(subjects))
        schema = {"type": "object", "required": ["labels"], "properties": {
            "labels": {"type": "array", "items": {"type": "string", "enum": CATEGORIES}}}}
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": schema,                            # grammar-constrained to the enum
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content":
                 "You label git commit messages by type. Reply ONLY with JSON "
                 '{"labels": [...]} — one label per numbered line, in order. Each label is '
                 f"exactly one of: {', '.join(CATEGORIES)}. Examples: 'Add X'->feat, "
                 "'Bump deps'->chore, 'Speed up Y'->perf, 'Fix crash'->fix."},
                {"role": "user", "content": numbered},
            ],
        }).encode()
        req = urllib.request.Request(f"{self.ollama_url}/api/chat", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            content = json.loads(r.read())["message"]["content"]
        return json.loads(content)["labels"]

    def _save(self):
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache, indent=0, sort_keys=True))


def changelog_md(commits, classifier):
    cats = classifier.classify([c["subject"] for c in commits])
    buckets = {key: [] for key, _ in SECTIONS}
    for c, cat in zip(commits, cats):
        m = TYPE_RE.match(c["subject"])
        text = m.group(4) if (m and m.group(1) in CATEGORIES) else c["subject"]
        buckets[cat if cat in buckets else "other"].append(f"- {text} (`{c['short']}`)")
    out = [f"### {title}\n" + "\n".join(buckets[key])
           for key, title in SECTIONS if buckets[key]]
    return "\n\n".join(out) or "_No commits in range._"
