"""
Binding novelty re-ranking.

The scoring pipeline ranks candidates by ``combined_score``, which is blind to
whether a drug is already an established treatment for the disease. That lets a
standard-of-care drug occupy the surfaced top-K — uninteresting for repurposing
and a credibility problem in outreach material (e.g. hydromorphone, an opioid
already standard-of-care, ranking #1 for sickle cell disease).

``apply_novelty_ranking()`` re-orders a disease's candidate list so genuine
repurposing leads are surfaced first and already-known treatments are demoted
to the tail — kept in the output and flagged, for transparency. It rewrites
``rank`` (1-indexed) and sets ``is_repurposing_candidate`` on every candidate.
"""

from __future__ import annotations

# novelty_level values that mean "already known for this disease"
_KNOWN_LEVELS = {"KNOWN", "ESTABLISHED"}


def is_repurposing_candidate(cand: dict) -> bool:
    """True when the candidate is a genuine repurposing lead — i.e. not an
    already-established treatment for this disease.

    A candidate is demoted when either signal says it is already known:
      - ``is_known_treatment`` is True (authoritative DRKG treats-edge lookup)
      - ``novelty_level`` is KNOWN or ESTABLISHED (literature/trial heuristic)

    Candidates with no novelty signal at all are treated as repurposing leads
    (we only demote on positive evidence of being known).
    """
    if cand.get("is_known_treatment") is True:
        return False
    if str(cand.get("novelty_level", "")).upper() in _KNOWN_LEVELS:
        return False
    return True


def apply_novelty_ranking(candidates: list[dict]) -> list[dict]:
    """Re-order ``candidates`` so repurposing leads precede known treatments.

    Within each partition the candidates stay ordered by ``combined_score``
    descending. Rewrites ``rank`` (1-indexed) and sets
    ``is_repurposing_candidate`` on every candidate. Returns the same list
    objects, re-ordered (also mutates the input list's contents in place).
    """
    for c in candidates:
        c["is_repurposing_candidate"] = is_repurposing_candidate(c)

    def sort_key(c: dict) -> tuple[int, float]:
        try:
            score = float(c.get("combined_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        # repurposing leads first (0 < 1); then highest combined_score first
        return (0 if c["is_repurposing_candidate"] else 1, -score)

    ordered = sorted(candidates, key=sort_key)
    for i, c in enumerate(ordered, 1):
        c["rank"] = i
    return ordered
