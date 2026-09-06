"""Startup disk-space check.

Written after a real failure: the reranker download ran out of space on the cache drive
and surfaced as `OSError: Can't load the model for 'BAAI/bge-reranker-v2-m3'` from inside
an /ask request, minutes in, with a traceback that never mentioned disk space.
"""

from __future__ import annotations

import logging

import pytest

from app import preflight

REPOS = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "hf_cache_dir", lambda: tmp_path)
    return tmp_path


def _materialise(cache, repo_id: str, *, weights: bool = True) -> None:
    """Create a cache entry. With ``weights=False`` this reproduces an interrupted
    download: config and tokenizer present, model file still missing."""
    snapshot = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"w" * 1024)


def test_warns_when_disk_is_too_small_and_models_are_missing(cache, monkeypatch, caplog):
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 1.2)
    with caplog.at_level(logging.WARNING):
        preflight.check_model_cache(REPOS)
    warning = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "1.2 GB free" in warning
    assert "HF_HOME" in warning  # the message must name the fix, not just the problem


def test_silent_when_models_are_already_cached(cache, monkeypatch, caplog):
    """A small disk is irrelevant once the weights are on it — do not cry wolf."""
    for repo in REPOS:
        _materialise(cache, repo)
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 0.5)
    with caplog.at_level(logging.WARNING):
        preflight.check_model_cache(REPOS)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_partially_cached_still_warns(cache, monkeypatch, caplog):
    """Exactly the reported failure: BGE-M3 present, reranker not, disk full."""
    _materialise(cache, "BAAI/bge-m3")
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 1.2)
    with caplog.at_level(logging.WARNING):
        preflight.check_model_cache(REPOS)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_no_warning_when_disk_has_room(cache, monkeypatch, caplog):
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 250.0)
    with caplog.at_level(logging.WARNING):
        preflight.check_model_cache(REPOS)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_cache_path_is_always_logged(cache, monkeypatch, caplog):
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 250.0)
    with caplog.at_level(logging.INFO):
        preflight.check_model_cache(REPOS)
    assert str(cache) in "\n".join(r.getMessage() for r in caplog.records)


def test_unknown_free_space_does_not_raise(cache, monkeypatch, caplog):
    monkeypatch.setattr(preflight, "_free_gb", lambda _: None)
    with caplog.at_level(logging.INFO):
        preflight.check_model_cache(REPOS)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_check_never_raises_even_if_the_cache_path_is_broken(monkeypatch, caplog):
    """Startup must not die because a diagnostic failed."""
    monkeypatch.setattr(preflight, "hf_cache_dir", lambda: preflight.Path("Z:/does/not/exist"))
    preflight.check_model_cache(REPOS)


def test_hf_cache_dir_follows_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    import importlib

    from huggingface_hub import constants

    importlib.reload(constants)
    try:
        assert str(tmp_path) in str(preflight.hf_cache_dir())
    finally:
        monkeypatch.undo()
        importlib.reload(constants)


def test_interrupted_download_is_not_mistaken_for_a_cached_model(cache, monkeypatch, caplog):
    """The exact observed failure: a full disk leaves config.json and the tokenizer behind
    but no model.safetensors, and the service must still warn."""
    _materialise(cache, "BAAI/bge-m3")
    _materialise(cache, "BAAI/bge-reranker-v2-m3", weights=False)
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 1.2)
    with caplog.at_level(logging.WARNING):
        preflight.check_model_cache(REPOS)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_zero_byte_weight_file_does_not_count(cache, monkeypatch, caplog):
    for repo in REPOS:
        _materialise(cache, repo)
    (cache / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "abc123" / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(preflight, "_free_gb", lambda _: 1.2)
    with caplog.at_level(logging.WARNING):
        preflight.check_model_cache(REPOS)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]
