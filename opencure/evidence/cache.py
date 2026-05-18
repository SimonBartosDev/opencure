"""
Persistent on-disk cache for external evidence API calls.

Any function decorated with @disk_cached reads/writes
data/evidence_cache/<namespace>/<sha256>.json based on its arguments.
Cache TTL is respected via the file's mtime; default 30 days.

Typical use in an evidence module:

    from opencure.evidence.cache import disk_cached

    @disk_cached(namespace='pubmed', ttl_days=30)
    def search_pubmed(query, max_results=10):
        ... network call ...
        return result_list

Cache key is (namespace, sha256 of repr(args) + repr(sorted(kwargs.items()))),
so identical (drug, disease) lookups hit the cache regardless of the
Python session.

Why a disk cache and not lru_cache:
  - lru_cache is per-process, so every new screen-run re-hits PubMed from 0
  - PubMed / CT.gov / FAERS are rate-limited: cache hits = staying friendly
  - Evidence data changes slowly; 30-day TTL is medically reasonable
  - A 2nd re-screen using hot cache runs ~5-10× faster than cold

Estimated impact on the current full-screen pipeline:
  Cold run:  ~5-6 hours  (currently)
  Warm run:  ~45-90 minutes  (after this cache lands, on 2nd run)
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Callable


CACHE_DIR = Path("data/evidence_cache")
DEFAULT_TTL_DAYS = 30


def _args_key(args: tuple, kwargs: dict) -> str:
    """Stable hash of call arguments."""
    payload = {
        "args": [repr(a) for a in args],
        "kwargs": sorted(kwargs.items()),
    }
    s = json.dumps(payload, default=repr, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def _cache_path(namespace: str, key: str) -> Path:
    return CACHE_DIR / namespace / f"{key}.json"


def _is_fresh(path: Path, ttl_days: float) -> bool:
    if not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days <= ttl_days


def disk_cached(namespace: str, ttl_days: float = DEFAULT_TTL_DAYS) -> Callable:
    """Decorator: persist function results to data/evidence_cache/<namespace>/."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _args_key(args, kwargs)
            path = _cache_path(namespace, key)
            if _is_fresh(path, ttl_days):
                try:
                    with path.open() as f:
                        return json.load(f)
                except Exception:
                    pass  # corrupt entry; recompute
            result = fn(*args, **kwargs)
            # Only cache JSON-serializable results
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    json.dump(result, f)
            except (TypeError, ValueError):
                pass  # unserializable — return directly but don't cache
            return result
        return wrapper
    return decorator


def clear_cache(namespace: str | None = None) -> int:
    """Purge all cached entries, or just one namespace. Returns count deleted."""
    root = CACHE_DIR / namespace if namespace else CACHE_DIR
    if not root.exists():
        return 0
    count = 0
    for p in root.rglob("*.json"):
        try:
            p.unlink()
            count += 1
        except Exception:
            pass
    return count


def cache_stats() -> dict:
    """Return per-namespace entry count + total size on disk."""
    stats: dict[str, dict] = {}
    if not CACHE_DIR.exists():
        return stats
    for ns_dir in CACHE_DIR.iterdir():
        if not ns_dir.is_dir():
            continue
        files = list(ns_dir.glob("*.json"))
        stats[ns_dir.name] = {
            "entries": len(files),
            "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
        }
    return stats


if __name__ == "__main__":
    print("Current cache stats:")
    for ns, s in cache_stats().items():
        print(f"  {ns:<20s}  {s['entries']:>6} entries  {s['size_mb']:>6} MB")
