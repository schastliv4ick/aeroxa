"""Context diversification.

Observed on real data: an English query against ФЗ-14 returned three parts of Статья 26
and nothing else. Parts of one long article score alike on a query about that article, so
without a cap a single Статья fills the whole prompt and the model never sees the related
norms it needs to answer well.
"""

from __future__ import annotations

import pytest

from app.rag.pipeline import Fragment, _diversify


def _fragment(article: str, score: float, part: int = 0, act: str = "fz-14-ooo") -> Fragment:
    return Fragment(
        payload={"act_id": act, "article_number": article, "part_index": part},
        score=score,
        retrieval_score=score,
    )


def _articles(fragments: list[Fragment]) -> list[str]:
    return [fragment.payload["article_number"] for fragment in fragments]


def test_one_article_cannot_fill_the_whole_context():
    candidates = [_fragment("26", 0.9 - i / 100, part=i) for i in range(5)]
    candidates += [_fragment("8", 0.5), _fragment("33", 0.4)]
    assert _articles(_diversify(candidates, limit=4, max_per_article=2)) == ["26", "26", "8", "33"]


def test_the_best_fragment_always_survives():
    candidates = [_fragment("26", 0.99), _fragment("26", 0.98), _fragment("8", 0.10)]
    assert _diversify(candidates, limit=3, max_per_article=1)[0].score == 0.99


def test_context_is_not_left_short_by_the_cap():
    """If diversity cannot fill the quota, capped fragments come back rather than a gap."""
    candidates = [_fragment("26", 0.9 - i / 100, part=i) for i in range(5)]
    assert len(_diversify(candidates, limit=4, max_per_article=2)) == 4


def test_same_article_number_in_different_acts_is_not_conflated():
    candidates = [
        _fragment("14", 0.9, act="fz-14-ooo"),
        _fragment("14", 0.8, act="gk-rf-1"),
        _fragment("14", 0.7, act="tk-rf"),
    ]
    assert len(_diversify(candidates, limit=3, max_per_article=1)) == 3


def test_limit_is_respected():
    candidates = [_fragment(str(i), 0.9) for i in range(10)]
    assert len(_diversify(candidates, limit=6, max_per_article=2)) == 6


@pytest.mark.parametrize("cap", [0, -1])
def test_cap_can_be_disabled(cap):
    candidates = [_fragment("26", 0.9 - i / 100, part=i) for i in range(5)]
    assert _articles(_diversify(candidates, limit=3, max_per_article=cap)) == ["26", "26", "26"]
