"""Abstention gating and its calibration.

The scores below are real measurements from bge-reranker-v2-m3 against ФЗ-14 (see
docs/SPEC.md §4). They are pinned here so a future threshold change cannot silently
reintroduce the failure they were written for: a question that retrieved exactly the right
article being refused because the threshold was picked from Russian examples alone.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.pipeline import Fragment, Pipeline

# The first threshold guessed for this system, before anything was measured. Kept as a
# constant because two tests below exist to show what it would have done.
UNCALIBRATED_GUESS = 0.30

# (language, best cross-encoder score, answerable?) — measured, not invented.
MEASURED = [
    ("ru", 0.999, True),   # «минимальный размер уставного капитала» -> Статья 14
    ("ru", 0.999, True),   # «как участник может выйти из общества»  -> Статья 26
    ("ru", 1.000, True),   # «компетенция общего собрания»           -> Статья 33
    ("ru", 0.000, False),  # VAT question, not covered by ФЗ-14
    ("ru", 0.007, False),  # minimum-wage question, not covered by ФЗ-14
    ("zh", 0.119, True),   # 最低注册资本      -> Статья 14
    ("zh", 0.251, True),   # 参与者如何退出公司 -> Статья 26
    ("zh", 0.129, True),   # 股东大会的职权     -> Статья 33
    ("zh", 0.467, True),   # 重大交易          -> Статья 46
    ("zh", 0.000, False),  # VAT question in Chinese
    ("zh", 0.000, False),  # minimum-wage question in Chinese
    ("en", 0.371, True),   # minimum charter capital -> Статья 14
    ("en", 0.827, True),   # participant withdrawal  -> Статья 26
    ("en", 0.000, False),  # VAT question in English
]

ANSWERABLE = [(lang, score) for lang, score, ok in MEASURED if ok]
OFF_TOPIC = [(lang, score) for lang, score, ok in MEASURED if not ok]


@pytest.fixture
def pipeline():
    get_settings.cache_clear()
    yield Pipeline()
    get_settings.cache_clear()


def _fragments(score: float) -> list[Fragment]:
    payload = {"act_id": "fz-14-ooo", "article_number": "14"}
    return [Fragment(payload=payload, score=score, retrieval_score=score)]


@pytest.mark.parametrize(("language", "score", "answerable"), MEASURED)
def test_every_measured_query_is_gated_correctly(pipeline, language, score, answerable):
    assert pipeline._should_abstain(_fragments(score), language) is not answerable


def test_the_two_score_bands_do_not_overlap():
    """The threshold is only meaningful because answerable and off-topic separate cleanly."""
    assert min(score for _, score in ANSWERABLE) > max(score for _, score in OFF_TOPIC)


def test_threshold_sits_between_the_bands(pipeline):
    threshold = pipeline.abstain_threshold_for("ru")
    assert max(score for _, score in OFF_TOPIC) < threshold
    assert threshold < min(score for _, score in ANSWERABLE)


def test_a_russian_calibrated_threshold_would_refuse_correct_answers():
    """Why the threshold is low: the cross-encoder's scale depends on the language pair.

    Guards the regression, not just the fix — if someone raises the threshold back toward
    a value that "looks safe" for Russian, this states the cost in other languages.
    """
    refused = [
        (lang, score) for lang, score in ANSWERABLE if score < UNCALIBRATED_GUESS
    ]
    # Three of the four Chinese questions, every one of which retrieved the right article.
    assert {lang for lang, _ in refused} == {"zh"}
    assert len(refused) == 3
    # And English clears it by only 24%, so it was one probe away from the same fate.
    english = [score for lang, score in ANSWERABLE if lang == "en"]
    assert min(english) < UNCALIBRATED_GUESS * 1.3


def test_chinese_scores_far_below_russian_despite_correct_retrieval():
    russian = [score for lang, score in ANSWERABLE if lang == "ru"]
    chinese = [score for lang, score in ANSWERABLE if lang == "zh"]
    assert max(chinese) < min(russian)


def test_per_language_override_is_applied(pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline._settings, "abstain_threshold_by_language", {"zh": 0.9}
    )
    assert pipeline.abstain_threshold_for("zh") == 0.9
    assert pipeline._should_abstain(_fragments(0.467), "zh") is True
    # Other languages keep the default.
    assert pipeline._should_abstain(_fragments(0.467), "ru") is False


def test_unknown_language_falls_back_to_the_default(pipeline):
    assert pipeline.abstain_threshold_for("de") == get_settings().abstain_threshold


def test_no_hits_always_abstains(pipeline):
    assert pipeline._should_abstain([], "ru") is True
    assert pipeline._should_abstain([], "zh") is True


def test_without_the_reranker_the_threshold_does_not_apply(pipeline, monkeypatch):
    """RRF scores are not on the cross-encoder's scale, so the threshold is meaningless."""
    monkeypatch.setattr(pipeline._settings, "rerank_enabled", False)
    assert pipeline._should_abstain(_fragments(0.001), "ru") is False
    assert pipeline._should_abstain([], "ru") is True
