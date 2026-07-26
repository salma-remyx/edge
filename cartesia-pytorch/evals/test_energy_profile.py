"""Tests for the phase-split energy profiling wired into ``evals.generation``.

``time_bench`` now reports prefill (prompt processing) and decode (generation)
separately via ``energy_profile.report_phases``, instead of collapsing them
into a single number. The report-logic tests below run anywhere; the wiring
test imports the call-site module ``evals.generation`` and is skipped when its
CUDA-only dependencies (torch, mamba_ssm, flash_attn) are unavailable.
"""

import os
import sys

# Make ``evals`` importable regardless of the directory pytest is invoked from.
_CARTESIA_PYTORCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CARTESIA_PYTORCH not in sys.path:
    sys.path.insert(0, _CARTESIA_PYTORCH)

from evals.energy_profile import report_phases  # noqa: E402


def test_decode_token_dominates_energy():
    """Each decode token costs more than each prefill token, so decode dominates energy."""
    report = report_phases(
        total_ms=505.0,
        prefill_ms=5.0,
        prompt_len=100,
        gen_len=100,
        power_w=15.0,
    )
    # Decode time is total minus prefill.
    assert report.decode_ms == 500.0
    # Per-token decode cost is ~100x the per-token prefill cost.
    assert report.decode_prefill_ratio == 100.0
    # Constant-power model: energy proportional to time -> decode ~99% of energy.
    assert report.prefill_energy_mj == 15.0 * 5.0
    assert report.decode_energy_mj == 15.0 * 500.0
    assert report.decode_energy_share > 0.98
    formatted = report.format()
    assert "decode/prefill" in formatted
    assert "decode energy share" in formatted


def test_phase_split_reported_without_power():
    """The prefill/decode split is reported even when no power estimate is supplied."""
    report = report_phases(total_ms=210.0, prefill_ms=10.0, prompt_len=50, gen_len=200)
    assert report.power_w is None
    assert report.prefill_energy_mj is None
    assert report.decode_ms == 200.0
    assert report.decode_us_per_tok == 1000.0  # 200 ms / 200 tokens
    assert report.prefill_us_per_tok == 200.0  # 10 ms / 50 tokens


def test_total_only_when_prefill_unmeasured():
    """Without a prefill measurement only the combined total is reported."""
    report = report_phases(total_ms=300.0, prompt_len=100, gen_len=100, power_w=10.0)
    assert report.prefill_ms is None
    assert report.decode_ms is None
    assert report.decode_prefill_ratio is None
    assert report.decode_energy_share is None


def test_time_bench_wired_to_report_phases():
    """The call-site module wires ``time_bench`` to the energy profiler."""
    import inspect

    try:
        from evals import generation
    except Exception as exc:  # CUDA-only deps unavailable in this sandbox.
        import pytest

        pytest.skip(f"evals.generation requires CUDA deps to import: {exc}")

    source = inspect.getsource(generation.time_bench)
    assert "from .energy_profile import report_phases" in source
    assert "report_phases(" in source
    signature = inspect.signature(generation.time_bench)
    assert "model" in signature.parameters
    assert "power_w" in signature.parameters
