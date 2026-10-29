"""
Tests for the site-index BM25 search (app.tools.site_search).

Pure logic over a tiny inline corpus; no I/O. The calibrated default threshold
is pinned against the committed fixture in test_site_index.py.
"""

from __future__ import annotations

import pytest

from app.tools.site_search import SiteChunk, SiteSearch, search_site, tokenize

BASE = "https://www.markandrewmarquez.com/portfolio"

CORPUS = [
    SiteChunk("bot:0", f"{BASE}/bot/", "Sales Lead Bot", "Overview",
              "A LangGraph chatbot that qualifies leads and writes them to Salesforce."),
    SiteChunk("bot:1", f"{BASE}/bot/", "Sales Lead Bot", "Scoring",
              "Lead scoring is deterministic: six dimensions add up to 100 points."),
    SiteChunk("bot:2", f"{BASE}/bot/", "Sales Lead Bot", "Evals",
              "Recorded cassettes replay the router and the extraction prompts in CI."),
    SiteChunk("bot:3", f"{BASE}/bot/", "Sales Lead Bot", "Deployment",
              "The backend runs on Azure Container Apps and scales to zero."),
    SiteChunk("monitor:0", f"{BASE}/monitor/", "Infrastructure Monitor", "Overview",
              "CloudWatch Synthetics canaries probe endpoints and Terraform builds the stack."),
    SiteChunk("monitor:1", f"{BASE}/monitor/", "Infrastructure Monitor", "Status page",
              "A static status page shows uptime for every endpoint."),
    SiteChunk("ad:0", f"{BASE}/ad/", "Two-Site Active Directory", "Overview",
              "Ansible builds two domain controllers on KVM and tests disaster recovery."),
]


@pytest.fixture
def search() -> SiteSearch:
    return SiteSearch(CORPUS)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_casefolds_and_splits_on_punctuation(self):
        assert tokenize("LangGraph, Salesforce; Azure!") == ["langgraph", "salesforce", "azure"]

    def test_drops_stopwords_and_site_framing_words(self):
        assert tokenize("What did Mark write in the article about canaries?") == ["canarie"]

    def test_drops_framing_words_about_marks_work(self):
        assert tokenize("Has Mark worked with LangGraph? What's his experience?") == ["langgraph"]

    def test_inflected_framing_words_drop_after_the_plural_fold(self):
        assert tokenize("Which article explains it") == []
        assert tokenize("sites websites marks reads shows tells knows wants explains learns") == []

    def test_stopwords_that_end_in_s_are_dropped_before_the_fold(self):
        # Without the first check, "this" and "does" would fold to "thi" and "doe".
        assert tokenize("this does canaries") == ["canarie"]

    def test_keeps_the_acronym_IT_but_not_the_pronoun(self):
        assert tokenize("It was IT support work") == ["it", "support", "work"]
        assert tokenize("it support") == ["support"]

    def test_drops_single_characters(self):
        assert tokenize("a b c x9 v2") == ["x9", "v2"]

    def test_plural_fold_is_minimal(self):
        assert tokenize("agents apis process status analysis") == [
            "agent", "api", "process", "status", "analysis",
        ]

    def test_nfkc_normalises_compatibility_characters(self):
        # Full-width letters fold to ASCII before matching.
        assert tokenize("ＡＷＳ Ｌａｍｂｄａ") == ["aws", "lambda"]

    def test_empty_and_stopword_only_text(self):
        assert tokenize("") == []
        assert tokenize("what is it about?") == []


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestRank:
    def test_best_chunk_matches_the_question(self, search):
        ranked = search.rank("how is lead scoring calculated")
        assert CORPUS[ranked[0][0]].id == "bot:1"

    def test_title_and_heading_terms_count(self, search):
        # "Deployment" only appears in that chunk's heading.
        assert CORPUS[search.rank("deployment")[0][0]].id == "bot:3"
        # "Active Directory" only appears in that chunk's title.
        assert CORPUS[search.rank("active directory")[0][0]].id == "ad:0"

    def test_scores_are_normalised_below_one(self, search):
        ranked = search.rank("terraform canaries cloudwatch synthetics endpoints")
        assert ranked
        assert all(0.0 < score < 1.0 for _, score in ranked)

    def test_scores_are_sorted_best_first(self, search):
        scores = [score for _, score in search.rank("lead scoring salesforce")]
        assert scores == sorted(scores, reverse=True)

    def test_unseen_term_lowers_the_score(self, search):
        on_topic = search.rank("terraform canaries")[0][1]
        diluted = search.rank("terraform canaries kubernetes helm")[0][1]
        assert diluted < on_topic

    @pytest.mark.parametrize(
        ("inflected", "base"),
        [
            ("Which article explains how lead scoring works?", "lead scoring works"),
            ("Which of the websites covers terraform canaries?", "covers terraform canaries"),
        ],
    )
    def test_inflected_framing_words_do_not_lower_the_score(self, search, inflected, base):
        assert search.rank(inflected) == search.rank(base)

    def test_query_with_no_content_terms_returns_nothing(self, search):
        assert search.rank("what did you write about?") == []
        assert search.rank("") == []

    def test_no_match_returns_nothing(self, search):
        assert search.rank("kubernetes helm") == []

    def test_repeated_query_words_count_once(self, search):
        assert search.rank("terraform") == search.rank("terraform terraform terraform")

    def test_empty_corpus(self):
        assert SiteSearch([]).rank("terraform") == []


# ---------------------------------------------------------------------------
# search_site (threshold + one page per hit)
# ---------------------------------------------------------------------------

class TestSearchSite:
    def test_hit_carries_up_to_k_chunks_from_the_best_page_only(self, search):
        hit = search_site(search, "lead bot salesforce scoring azure", min_score=0.0)
        assert hit is not None
        assert hit.url == f"{BASE}/bot/"
        assert hit.title == "Sales Lead Bot"
        assert len(hit.chunks) == 3
        assert {c.url for c in hit.chunks} == {hit.url}

    def test_k_limits_the_chunks(self, search):
        hit = search_site(search, "lead bot salesforce scoring azure", min_score=0.0, k=1)
        assert hit is not None
        assert len(hit.chunks) == 1

    def test_hit_score_is_the_best_chunk_score(self, search):
        hit = search_site(search, "terraform canaries", min_score=0.0)
        assert hit is not None
        assert hit.score == search.rank("terraform canaries")[0][1]
        assert hit.chunks[0].id == "monitor:0"

    def test_below_min_score_is_a_miss(self, search):
        score = search.rank("terraform canaries")[0][1]
        assert search_site(search, "terraform canaries", min_score=score) is not None
        assert search_site(search, "terraform canaries", min_score=score + 1e-9) is None

    def test_no_match_is_a_miss(self, search):
        assert search_site(search, "weather in paris", min_score=0.0) is None

    def test_fragment_urls_group_by_page(self):
        chunks = [
            SiteChunk("p:0", f"{BASE}/p/#intro", "Page", "Intro", "terraform canaries"),
            SiteChunk("p:1", f"{BASE}/p/#more", "Page", "More", "terraform modules"),
            SiteChunk("q:0", f"{BASE}/q/", "Other", "Intro", "terraform state"),
        ]
        hit = search_site(SiteSearch(chunks), "terraform canaries", min_score=0.0)
        assert hit is not None
        assert hit.url == f"{BASE}/p/#intro"
        assert [c.id for c in hit.chunks] == ["p:0", "p:1"]
