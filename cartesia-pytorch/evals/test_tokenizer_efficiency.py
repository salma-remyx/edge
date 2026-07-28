# Copyright (c) 2024, Cartesia.

"""Tests for the multilingual tokenizer-fragmentation benchmark.

The fragmentation math is exercised with a fake tokenizer so the core
capability is verified without a GPU, model weights, or HuggingFace downloads.
A guarded test additionally confirms the benchmark wires into the existing
``evals.generation`` (``choose_model`` / ``time_bench``) contract.
"""

import pytest
from evals.tokenizer_efficiency import (
    SAMPLE_CORPUS,
    count_tokens,
    estimate_decode_disparity,
    format_report,
    fragmentation_stats,
    multilingual_report,
)


class _FakeTokenizer:
    """Deterministic stand-in: one token per ASCII letter, three per non-ASCII.

    Simulates how a Latin-centric tokenizer fragments non-Latin scripts, so the
    cross-language disparity asserted below is meaningful.
    """

    def encode(self, text, add_special_tokens=False):
        ids = []
        for char in text:
            if char.isascii() and not char.isspace():
                ids.append(ord(char))
            elif not char.isascii():
                ids.extend([1000 + ord(char), 2000 + ord(char), 3000 + ord(char)])
        return ids


def test_count_tokens_uses_encode_without_special_tokens():
    """Whitespace contributes no id and ASCII letters map one-to-one."""
    tok = _FakeTokenizer()
    assert count_tokens(tok, "abc") == 3
    assert count_tokens(tok, "a b") == 2


def test_fragmentation_stats_metrics():
    """Core density metrics are computed from token, char, and byte counts."""
    stats = fragmentation_stats(_FakeTokenizer(), "ab")  # two ASCII letters
    assert stats.tokens == 2
    assert stats.chars == 2
    assert stats.n_bytes == 2
    assert stats.words == 1
    assert stats.tokens_per_char == pytest.approx(1.0)
    assert stats.bytes_per_token == pytest.approx(1.0)


def test_multilingual_report_disparity_favors_baseline():
    """Non-Latin text fragments 3x and sorts ahead of the baseline language."""
    tok = _FakeTokenizer()
    # English: three ASCII letters -> three tokens over three chars.
    # Hindi: each Devanagari char -> three tokens = nine tokens over three chars.
    corpus = {"English": "abc", "Hindi": "कखग"}
    reports = multilingual_report(tok, corpus, baseline="English")
    by_lang = {report.language: report for report in reports}
    assert by_lang["English"].tokens_per_char_ratio == pytest.approx(1.0)
    # Hindi fragments 3x: nine tokens / three chars vs three tokens / three chars.
    assert by_lang["Hindi"].tokens_per_char_ratio == pytest.approx(3.0)
    # Sorted by descending fragmentation -> Hindi first.
    assert reports[0].language == "Hindi"


def test_multilingual_report_rejects_missing_baseline():
    """An unknown baseline language raises KeyError before any scoring."""
    with pytest.raises(KeyError):
        multilingual_report(_FakeTokenizer(), {"English": "abc"}, baseline="Klingon")


def test_estimate_decode_disparity_combines_per_token_cost():
    """Fragmentation ratio multiplied by per-token cost yields ms/char."""
    tok = _FakeTokenizer()
    reports = multilingual_report(tok, {"English": "abc", "Hindi": "कखग"}, baseline="English")
    disparity = estimate_decode_disparity(reports, baseline="English", ms_per_token=2.0)
    # English: 1 tok/char * 2 ms = 2 ms/char; ratio 1.0.
    assert disparity["English"]["ms_per_char"] == pytest.approx(2.0)
    assert disparity["English"]["per_char_cost_ratio"] == pytest.approx(1.0)
    # Hindi: 3 tok/char * 2 ms = 6 ms/char; ratio 3.0.
    assert disparity["Hindi"]["ms_per_char"] == pytest.approx(6.0)
    assert disparity["Hindi"]["per_char_cost_ratio"] == pytest.approx(3.0)


def test_estimate_decode_disparity_without_per_token_cost():
    """Without per-token cost only the fragmentation ratio is reported."""
    tok = _FakeTokenizer()
    reports = multilingual_report(tok, {"English": "abc", "Hindi": "कखग"}, baseline="English")
    disparity = estimate_decode_disparity(reports, baseline="English")
    assert disparity["Hindi"] == {"tokens_per_char_ratio": pytest.approx(3.0)}
    assert "ms_per_char" not in disparity["Hindi"]


def test_format_report_renders_every_language():
    """The text table lists every language in the corpus."""
    reports = multilingual_report(_FakeTokenizer(), SAMPLE_CORPUS, baseline="English")
    table = format_report(reports, baseline="English")
    for language in SAMPLE_CORPUS:
        assert language in table


def test_contract_alignment_with_generation():
    """The benchmark's contract target evals.generation exposes choose_model/time_bench."""
    generation = pytest.importorskip("evals.generation")
    assert callable(getattr(generation, "choose_model", None))
    assert callable(getattr(generation, "time_bench", None))
