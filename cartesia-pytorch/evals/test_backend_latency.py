# Copyright (c) 2024, Aviv Bick, Kevin Li.

"""Tests for the cross-backend latency benchmark.

The protocol/report tests below are dependency-free and run anywhere. The
final test is guarded on the full Edge model stack (torch + transformers +
mamba_ssm + flash_attn) and exercises the lazy import of the existing
``cartesia_pytorch`` model modules.
"""

import time
import types

import backend_latency as bl
import pytest


class _FakeGenerateModel:
    """Minimal stand-in for an Edge LMHeadModel.

    Records the kwargs passed to ``generate`` so the wiring test can assert
    that the runner drives the repo's real ``.generate`` contract.
    """

    def __init__(self, prompt_len):
        """Store the prompt length and prepare a call log."""
        self.prompt_len = prompt_len
        self.calls = []

    def generate(self, **kwargs):
        """Record kwargs and return an object with a ``sequences`` attribute."""
        self.calls.append(kwargs)
        return types.SimpleNamespace(sequences=[[0] * (kwargs["max_length"])])


def test_compare_backends_picks_fastest_runner():
    """The report's fastest() must identify the lowest-latency backend."""
    fast = bl.CallableRunner("pytorch-eager", lambda: None, units=4)
    slow = bl.CallableRunner("pytorch-compiled", lambda: time.sleep(0.003), units=4)
    report = bl.compare_backends([fast, slow], warmup=1, repeats=3)

    table = report.format_table()
    assert "pytorch-eager" in table
    assert "pytorch-compiled" in table
    assert report.fastest().name == "pytorch-eager"


def test_bench_callable_records_repeated_samples():
    """bench_callable must run warmup + repeats and summarize the samples."""
    stats = bl.bench_callable("x", lambda: None, warmup=2, repeats=5)
    assert stats.repeats == 5
    assert stats.warmup == 2
    assert len(stats.samples_ms) == 5
    assert stats.min_ms == min(stats.samples_ms)
    assert stats.min_ms <= stats.mean_ms <= max(stats.samples_ms)


def test_make_generate_runner_drives_generate_contract():
    """The runner must call model.generate with generation.py's kwargs.

    This is the integration assertion: the benchmark drives the repo's
    ``model.generate(input_ids=..., max_length=..., cg=...,
    return_dict_in_generate=...)`` interface, timed by compare_backends.
    """
    prompt_len, genlen = 7, 4
    input_ids = types.SimpleNamespace(shape=(1, prompt_len))
    model = _FakeGenerateModel(prompt_len)
    runner = bl.make_generate_runner(
        model,
        input_ids,
        genlen,
        prompt_len=prompt_len,
        name="pytorch-eager",
    )

    report = bl.compare_backends([runner], warmup=1, repeats=2)
    assert report.fastest().name == "pytorch-eager"

    assert len(model.calls) == 3  # 1 warmup + 2 timed
    call = model.calls[-1]
    assert call["input_ids"] is input_ids
    assert call["max_length"] == prompt_len + genlen
    assert call["cg"] is True
    assert call["return_dict_in_generate"] is True
    assert call["output_scores"] is False
    assert call["enable_timing"] is False
    # units default to genlen, so ms/unit is mean latency per token.
    assert report.stats[0].units == genlen


def test_make_generate_runner_routes_through_generate_fn():
    """Passing generate_fn routes calls through the alternate strategy."""
    prompt_len, genlen = 3, 2
    model = _FakeGenerateModel(prompt_len)
    expected = model.generate
    compiled = []

    def fake_compiled(**kwargs):
        """Wrap the real generate so we can observe the routing."""
        compiled.append(True)
        return expected(**kwargs)

    runner = bl.make_generate_runner(
        model,
        types.SimpleNamespace(shape=(1, prompt_len)),
        genlen,
        prompt_len=prompt_len,
        name="pytorch-compiled",
        generate_fn=fake_compiled,
    )
    bl.compare_backends([runner], warmup=0, repeats=1)
    assert compiled  # the alternate strategy actually drove generation


def test_resolve_model_class_imports_existing_rene_module():
    """resolve_model_class must resolve to the real cartesia_pytorch class.

    Guarded on the full Edge stack; skips where the model deps are absent.
    """
    for dep in ("torch", "transformers", "mamba_ssm", "flash_attn"):
        pytest.importorskip(dep)
    # Make the ``cartesia_pytorch`` namespace package importable regardless of
    # the pytest CWD (its parent, ``cartesia-pytorch/``, hosts the package).
    import os
    import sys

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
    from cartesia_pytorch.Rene.rene import ReneLMHeadModel

    assert bl.resolve_model_class("Rene") is ReneLMHeadModel
    assert bl.resolve_model_class("Llamba-1B") is LlambaLMHeadModel
