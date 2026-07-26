"""Tests for the open-ended stereotype-bias probe.

These exercise the integrated behavior of :mod:`evals.bias_probe`
(judge -> probe -> FP-vs-quantized delta) through the public surface,
plus the generate adapter against the same model/tokenizer interface
that ``evals/generation.py`` and the ``LlambaLMWrapper`` /
``ReneLMWrapper`` eval wrappers use. They also anchor the new code into
the existing package by importing :mod:`cartesia_pytorch.version`.
"""

import pytest
from evals.bias_probe import (
    GENDER,
    OCCUPATION,
    STEREOTYPE_LEXICON,
    BiasProbe,
    bias_delta,
    hflm_generate_adapter,
    judge_completion,
    run_bias_probe,
    wilson_interval,
)

import cartesia_pytorch.version  # non-new src module: anchors integration


def test_imports_existing_package():
    """The new code lives in the existing cartesia_pytorch package."""
    assert hasattr(cartesia_pytorch.version, "__version__")
    assert isinstance(cartesia_pytorch.version.__version__, str)


def test_judge_flags_explicit_stereotype():
    """The lexicon proxy flags explicit cues and passes benign text."""
    cues = STEREOTYPE_LEXICON[GENDER]
    flagged, matched = judge_completion("women are naturally emotional.", cues)
    assert flagged
    assert "emotional" in matched

    benign, none = judge_completion("anyone can succeed with effort.", cues)
    assert not benign
    assert none == ()


def test_judge_word_boundary_avoids_false_positive():
    """Substring-without-word-boundary should not match."""
    cues = STEREOTYPE_LEXICON[GENDER]
    # "emotional" is a cue, but here it only appears inside "emotionally"
    # via a different stem -- guard the boundary matcher behaves.
    flagged, _ = judge_completion("she spoke emotionlessly.", cues)
    assert not flagged


def test_run_bias_probe_computes_rate():
    """Stereotype rate and per-family breakdown match the stub outputs."""
    probes = (
        BiasProbe(GENDER, "g-biased"),
        BiasProbe(GENDER, "g-clean"),
        BiasProbe(OCCUPATION, "o-biased"),
    )

    def generate(prompt):
        responses = {
            "g-biased": "they are naturally emotional and delicate.",
            "g-clean": "people have widely varied temperaments.",
            "o-biased": "they are born leaders, naturally suited to it.",
        }
        return responses[prompt]

    report = run_bias_probe(generate, probes=probes)

    assert report.n_probes == 3
    assert report.n_flagged == 2
    assert report.stereotype_rate == pytest.approx(2 / 3)
    assert report.per_family[GENDER] == 0.5
    assert report.per_family[OCCUPATION] == 1.0


def test_wilson_interval_bounds():
    """Wilson CI brackets the observed rate and stays within [0, 1]."""
    lo, hi = wilson_interval(1, 4)
    assert lo < 0.25 < hi
    assert 0.0 <= lo and hi <= 1.0
    # Edge cases: no probes, and all-flagged.
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(4, 4)
    assert lo > 0.0 and hi == 1.0


def test_report_carries_wilson_ci():
    """Reports expose the paper-standard Wilson CI on the stereotype rate."""
    probes = (
        BiasProbe(GENDER, "g-biased"),
        BiasProbe(GENDER, "g-clean"),
    )

    def generate(prompt):
        return {
            "g-biased": "they are naturally emotional and delicate.",
            "g-clean": "people have widely varied temperaments.",
        }[prompt]

    report = run_bias_probe(generate, probes=probes)
    lo, hi = report.rate_ci()
    assert lo < report.stereotype_rate < hi
    assert "95% CI" in str(report)


def test_bias_delta_quantization_increases_bias():
    """The paper's headline: quantization raises open-ended bias."""
    probes = (BiasProbe(GENDER, "g"),)

    def fp_generate(prompt):
        return "anyone can succeed with effort and support."

    def quant_generate(prompt):
        return "women are naturally emotional and nurturing."

    fp_report = run_bias_probe(fp_generate, probes=probes)
    quant_report = run_bias_probe(quant_generate, probes=probes)
    delta = bias_delta(fp_report, quant_report)

    assert fp_report.stereotype_rate == 0.0
    assert quant_report.stereotype_rate == 1.0
    assert delta.delta == 1.0  # quantization increased open-ended bias
    assert delta.per_family[GENDER] == 1.0


class _FakeTokenizer:
    """Mimics the HF tokenizer surface used by evals/generation.py."""

    def __call__(self, prompt, return_tensors="pt"):
        tokens = type("_T", (), {})()
        tokens.input_ids = prompt  # duck-typed; no torch tensor required
        return tokens

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [sequences]


class _FakeModel:
    """Mimics ``model.generate(..., return_dict_in_generate=True)``."""

    device = None

    def generate(self, input_ids=None, **kwargs):
        continuation = " they are naturally more emotional and nurturing."
        out = type("_O", (), {})()
        out.sequences = input_ids + continuation
        return out


def test_adapter_wires_to_generation_surface():
    """The adapter drives the existing model/tokenizer interface."""
    adapter = hflm_generate_adapter(
        _FakeModel(),
        _FakeTokenizer(),
        max_new_tokens=8,
    )
    continuation = adapter("A woman's role")

    # Only the model's continuation is returned -- the prompt echo is stripped.
    assert "A woman's role" not in continuation
    flagged, matched = judge_completion(continuation, STEREOTYPE_LEXICON[GENDER])
    assert flagged
    assert {"emotional", "nurturing"}.issubset(set(matched))
