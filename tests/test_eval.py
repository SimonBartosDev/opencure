"""Tests for held-out evaluation harness + time-sliced benchmark."""

import pytest
from pathlib import Path


class TestGroundTruth:

    @pytest.mark.integration
    def test_drkg_treats_edges_loadable(self):
        from opencure.eval.ground_truth import load_drkg_treats
        pairs = load_drkg_treats()
        assert len(pairs) > 1000, f"expected >1k DRKG treats pairs, got {len(pairs)}"
        # Spot-check format
        for d, s in pairs[:5]:
            assert d.startswith("Compound::")
            assert s.startswith("Disease::")

    def test_split_deterministic(self):
        from opencure.eval.ground_truth import split_holdout
        pairs = [(f"A{i}", f"B{i}") for i in range(100)]
        t1, h1 = split_holdout(pairs, frac=0.2, seed=42)
        t2, h2 = split_holdout(pairs, frac=0.2, seed=42)
        assert t1 == t2 and h1 == h2, "split with same seed must be deterministic"
        assert len(h1) == 20


class TestHoldoutFiles:

    def test_holdout_files_exist(self):
        assert Path("data/eval/holdout_test.jsonl").exists(), \
            "run `python3 -m opencure.eval.ground_truth` to create"


class TestTimeSliced:

    def test_time_sliced_benchmark_exists_or_reproducible(self):
        p = Path("data/eval/time_sliced_test.jsonl")
        # Either it's already built, or the builder is runnable
        if not p.exists():
            pytest.skip("time-sliced benchmark not built yet")
        import json
        lines = p.read_text().splitlines()
        assert len(lines) > 100, "time-sliced benchmark should have >100 pairs"
        # Check format
        for line in lines[:3]:
            d = json.loads(line)
            assert "compound" in d and "disease" in d
            assert d.get("first_approval_year", 0) >= 2020
