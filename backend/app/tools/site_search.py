"""
BM25 search over the portfolio site index (TARS site questions).

Pure, in-process ranking over the chunks of ``site-index.json`` (loaded by
``app.tools.site_index``). Each chunk is one section of one page; its title,
heading and text are indexed together with equal weight. Scores are BM25
normalised to ``[0, 1)`` per query, so a single threshold (``min_score``) means
the same thing for a two-word and a ten-word question.

Usage::

    from app.tools.site_search import SiteSearch, search_site

    search = SiteSearch(chunks)
    hit = search_site(search, "how does TARS score leads", min_score=0.3)
    if hit:
        print(hit.url, hit.score, [c.heading for c in hit.chunks])

Design notes:
- Own implementation (stdlib only): the BM25 libraries need numpy, which the
  image doesn't have, and 138 chunks build in milliseconds.
- IDF is the non-negative Lucene form. A query term the index has never seen
  gets the maximum IDF in the normalising bound and adds nothing to the score,
  so off-topic questions fall under the threshold.
- Every hit comes from a single page, so the answer's sources and its one link
  always agree.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteChunk:
    """One section of one site page (a ``chunks[]`` entry of site-index.json v1)."""

    id: str
    url: str
    title: str
    heading: str
    text: str


@dataclass(frozen=True)
class SiteHit:
    """
    The best-matching page for a question.

    ``url`` / ``title`` come from the best chunk; ``chunks`` are up to *k* of
    that page's chunks, best first; ``score`` is the best chunk's normalised score.
    """

    url: str
    title: str
    score: float
    chunks: tuple[SiteChunk, ...]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z0-9]+")

# English function words, question words, and words that frame a question about
# the site or about Mark's work ("what did Mark write in that article", "has Mark
# worked with X", "what's his experience with Y"): they carry no topic.
_STOPWORD_TEXT = """
    a about above after again against all also am an and any are as at be because
    been before being below between both but by can could did do does doing done
    down during each else etc even ever every few for from further get gets got had
    has have having he her here hers him his how however if in into is it its itself
    just like made make makes many may me might more most much must my no nor not now
    of off on once one only or other our ours out over own per please quite rather
    really same she should so some such than that the their theirs them then there
    these they this those through to too under until up upon us use used uses using
    very via was we were what when where which while who whom whose why will with
    within without would yes yet you your yours
    tell explain show know want wondering curious anything something thing things
    article articles post posts blog wrote write writes written writeup writeups
    read page pages site website mark
    worked working learn learned learnt experience experiences
    explained explaining showed shown showing told telling reading writing posted
    posting learning knew known knowing wanted wanting wondered
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

# All-caps "IT" (information technology) is a topic, not the pronoun "it": three
# of the site's pages are about IT work. Only the exact spelling "IT" is kept.
_KEEP_ACRONYMS = frozenset({"IT"})


def tokenize(text: str) -> list[str]:
    """
    Lower-case content tokens for indexing and querying.

    NFKC, ``[A-Za-z0-9]+`` runs, lower-cased; drop 1-character tokens and
    stopwords (except the acronym ``IT``), then a minimal plural fold
    (``agents`` -> ``agent``, ``apis`` -> ``api``; ``-ss``/``-us``/``-sis`` endings
    are kept: ``process``, ``status``, ``analysis``). Stopwords are checked before
    the fold (so ``this`` doesn't become ``thi``) and after it (so ``explains`` and
    ``websites`` drop like ``explain`` and ``website``). The same function runs on
    documents and queries, so the fold only has to be consistent, not
    linguistically right.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(unicodedata.normalize("NFKC", text)):
        tok = raw.lower()
        if raw in _KEEP_ACRONYMS:
            tokens.append(tok)
            continue
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "sis")):
            tok = tok[:-1]
            if tok in STOPWORDS:
                continue
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class SiteSearch:
    """BM25 (k1=1.2, b=0.75) over ``title + heading + text`` of each chunk."""

    K1 = 1.2
    B = 0.75

    def __init__(self, chunks: Sequence[SiteChunk]) -> None:
        self.chunks: tuple[SiteChunk, ...] = tuple(chunks)
        docs = [tokenize(f"{c.title} {c.heading} {c.text}") for c in self.chunks]
        self._tf = [Counter(doc) for doc in docs]
        self._len = [len(doc) for doc in docs]
        n = len(docs)
        self._avg_len = sum(self._len) / n if n else 0.0
        df = Counter(term for tf in self._tf for term in tf)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._unseen_idf = math.log(1 + (n + 0.5) / 0.5)

    def rank(self, query: str) -> list[tuple[int, float]]:
        """
        ``(chunk index, normalised score)`` for every chunk that matches at least
        one query term, best first (ties by index). ``[]`` when the query has no
        content terms.

        normalised = bm25(q, d) / sum over unique query terms of idf(t) * (k1 + 1),
        which is in ``[0, 1)`` because each term's BM25 weight is below idf * (k1 + 1).
        """
        terms = set(tokenize(query))
        if not terms:
            return []
        bound = sum(self._idf.get(t, self._unseen_idf) for t in terms) * (self.K1 + 1)

        scored: list[tuple[int, float]] = []
        for i, (tf, length) in enumerate(zip(self._tf, self._len, strict=True)):
            raw = 0.0
            for term in terms:
                freq = tf.get(term)
                if freq:
                    norm = self.K1 * (1 - self.B + self.B * length / self._avg_len)
                    raw += self._idf[term] * freq * (self.K1 + 1) / (freq + norm)
            if raw > 0:
                scored.append((i, raw / bound))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


def _page(url: str) -> str:
    """The page a chunk belongs to: its url without a ``#fragment``."""
    return url.split("#", 1)[0]


def search_site(
    search: SiteSearch, query: str, *, min_score: float, k: int = 3
) -> SiteHit | None:
    """
    Return the best page for *query*, or ``None`` when nothing scores at least
    *min_score*.

    The best chunk picks the page; the hit carries up to *k* of that page's
    chunks in rank order, so every source of the answer is on the linked page.
    """
    ranked = search.rank(query)
    if not ranked or ranked[0][1] < min_score:
        return None

    best_index, best_score = ranked[0]
    best = search.chunks[best_index]
    page = _page(best.url)
    same_page = [search.chunks[i] for i, _ in ranked if _page(search.chunks[i].url) == page]
    return SiteHit(url=best.url, title=best.title, score=best_score, chunks=tuple(same_page[:k]))
