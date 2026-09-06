"""Startup checks for conditions that otherwise fail confusingly and late.

Model weights are downloaded lazily, on the first request that needs them. When the cache
disk is full, the failure therefore surfaces as
``OSError: Can't load the model for 'BAAI/bge-reranker-v2-m3'`` raised from inside an
``/ask`` call — several minutes in, with a traceback that points at transformers and says
nothing about disk space. Reporting it at startup instead costs one ``statvfs``.

Nothing here raises: a warning that lets the service start is more useful than a refusal,
since the weights may already be cached elsewhere or the disk may be freed while running.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# BGE-M3 (~6.4 GB) plus bge-reranker-v2-m3 (~2.1 GB) as Hugging Face actually stores them:
# every published weight format, not just the one FlagEmbedding loads.
REQUIRED_CACHE_GB = 9.0

_HF_ENV_HINT = (
    'set HF_HOME to a disk with room before starting — PowerShell: $env:HF_HOME = "D:\\hf-cache", '
    "bash: export HF_HOME=/path/to/cache. It must be a real environment variable; .env is read "
    "by this application, not by the Hugging Face libraries."
)


def hf_cache_dir() -> Path:
    """Where Hugging Face will actually put weights, honouring HF_HOME/HF_HUB_CACHE."""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - never let a diagnostic break startup
        return Path.home() / ".cache" / "huggingface" / "hub"


def _free_gb(path: Path) -> float | None:
    # Walk up to the nearest existing ancestor: the cache dir may not exist yet.
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1024**3
    except OSError:
        return None


# A snapshot directory is not evidence of a usable model. An interrupted download leaves
# the small files behind -- config.json, the tokenizer -- with the weights still a
# `.incomplete` blob, which is exactly the state a full disk produces.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def _models_present(cache: Path, repo_ids: list[str]) -> bool:
    """True when every repo has a snapshot containing actual weights."""
    for repo_id in repo_ids:
        folder = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots"
        if not folder.is_dir():
            return False
        weights = (
            path
            for path in folder.rglob("*")
            if path.suffix in _WEIGHT_SUFFIXES and path.is_file() and path.stat().st_size > 0
        )
        if not next(weights, None):
            return False
    return True


def check_model_cache(repo_ids: list[str]) -> None:
    """Log where weights will be cached, and warn if that disk cannot hold them."""
    cache = hf_cache_dir()
    free = _free_gb(cache)
    if free is None:
        log.info("model cache: %s (free space unknown)", cache)
        return

    log.info("model cache: %s (%.1f GB free)", cache, free)

    if _models_present(cache, repo_ids):
        return  # Already downloaded; free space no longer matters for startup.

    if free < REQUIRED_CACHE_GB:
        log.warning(
            "model cache disk has %.1f GB free but the models need about %.0f GB, and they are "
            "not fully cached at %s. The download will fail partway through the first request "
            "with a misleading 'Can't load the model' error. To fix: %s",
            free,
            REQUIRED_CACHE_GB,
            cache,
            _HF_ENV_HINT,
        )
