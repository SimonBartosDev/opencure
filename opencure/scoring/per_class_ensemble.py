"""Per-class ensemble routing for v7.

The shared ``ensemble_v5.pkl`` averages across 93 diseases of widely
different therapeutic mechanism — anti-helminthics for Schisto, kinase
inhibitors for cancers, enzyme replacement for Gaucher, antipsychotics
for schizophrenia. v7 adds a small layer of *per-class* logistic heads
trained on the same 6-feature representation, but specialised to the
dominant mechanism of each class.

Architecture:
- 6 classes (parasitic, viral, bacterial, oncology, rare_metabolic,
  chronic_systemic) defined in ``opencure/eval/disease_classes.yaml``.
- Per-class head at ``data/models/ensemble_v7_<class>.pkl`` (same
  pickle shape as ``ensemble_v5.pkl``: ``{"model": clf, "feature_keys": (...)}``).
- ``route_disease(name) -> class`` resolves a disease name to its class.
- ``load_class_head(class_name)`` returns ``(model, feature_keys)`` or
  ``None`` if the head hasn't been trained yet — caller falls back to
  the shared head.
- ``score_with_routing(disease, p_features)`` is the public entry: takes
  the disease name + feature dict, routes to the right head, scores.

Fail-open contract: routing degrades to the shared head whenever the
class can't be resolved or the per-class artifact is missing. The
platform never breaks because of a missing per-class file.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

DISEASE_CLASSES_PATH = Path("opencure/eval/disease_classes.yaml")
PER_CLASS_MODEL_DIR = Path("data/models")


def _per_class_path(class_name: str) -> Path:
    return PER_CLASS_MODEL_DIR / f"ensemble_v7_{class_name}.pkl"


# ---- Disease → class ---------------------------------------------------

_CLASS_CACHE: dict[str, str] | None = None


def _normalize(name: str) -> str:
    """Match disease names tolerantly — strip case, punctuation drift."""
    return name.lower().strip().replace("'", "").replace("’", "")


def load_disease_class_map(
    path: Path = DISEASE_CLASSES_PATH,
) -> dict[str, str]:
    """Return ``{normalized_disease_name: class_name}``.

    Memoized; reload by clearing the module-level cache.
    """
    global _CLASS_CACHE
    if _CLASS_CACHE is not None:
        return _CLASS_CACHE
    if yaml is None:
        raise RuntimeError("PyYAML is required to load disease_classes.yaml")
    if not path.exists():
        _CLASS_CACHE = {}
        return _CLASS_CACHE

    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, str] = {}
    for class_name, diseases in raw.items():
        for d in diseases or []:
            out[_normalize(d)] = class_name
    _CLASS_CACHE = out
    return out


def route_disease(disease_name: str) -> Optional[str]:
    """Return the class name for ``disease_name`` or ``None`` if unmapped.

    ``None`` means "use the shared ensemble head" — not an error.
    """
    if not disease_name:
        return None
    classes = load_disease_class_map()
    return classes.get(_normalize(disease_name))


# ---- Class-head loader -------------------------------------------------

_HEAD_CACHE: dict[str, tuple[Any, tuple[str, ...]] | None] = {}


def load_class_head(
    class_name: str,
) -> Optional[tuple[Any, tuple[str, ...]]]:
    """Return ``(sklearn classifier, feature_keys)`` or ``None`` if absent.

    Memoized — first call hits disk, subsequent calls are O(1).
    """
    if class_name in _HEAD_CACHE:
        return _HEAD_CACHE[class_name]

    path = _per_class_path(class_name)
    if not path.exists():
        _HEAD_CACHE[class_name] = None
        return None
    try:
        with path.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception:
        _HEAD_CACHE[class_name] = None
        return None

    if isinstance(bundle, dict) and "model" in bundle:
        result = bundle["model"], tuple(bundle.get("feature_keys", ()))
    else:
        result = bundle, ()  # bare model — caller supplies feature keys
    _HEAD_CACHE[class_name] = result
    return result


def score_with_routing(
    disease_name: str,
    feats: dict[str, float],
    *,
    shared_model,
    shared_feature_keys: tuple[str, ...],
) -> tuple[float, str]:
    """Score a candidate via the per-class head when available.

    Returns ``(prob, head_used)`` where ``head_used`` is one of the
    class names or ``"shared"``. The tag flows into result JSONs so
    downstream consumers (and the methods paper) can audit which head
    produced each prediction.
    """
    class_name = route_disease(disease_name)
    if class_name is not None:
        head = load_class_head(class_name)
        if head is not None and head[1]:
            model, feature_keys = head
            X = np.asarray([[feats[k] for k in feature_keys]], dtype=float)
            return float(model.predict_proba(X)[0, 1]), class_name

    # Fallback: shared head.
    X = np.asarray([[feats[k] for k in shared_feature_keys]], dtype=float)
    return float(shared_model.predict_proba(X)[0, 1]), "shared"


def reset_caches() -> None:
    """Clear memoized state — useful for tests that swap the YAML."""
    global _CLASS_CACHE
    _CLASS_CACHE = None
    _HEAD_CACHE.clear()
