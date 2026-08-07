"""Tests for effective bits-per-weight measurement.

These exercise :mod:`evals.weight_precision` through duck-typed fake
tensors (no torch required), and anchor the integration against non-new
modules: :mod:`evals.bias_probe` (the probe the bpw report accompanies
in ``evals/bias_eval.py``) and :mod:`cartesia_pytorch.version` (the
existing package the evals live in).
"""

import pytest
from evals.bias_probe import BiasProbe, bias_delta, run_bias_probe
from evals.weight_precision import measure_weight_precision

import cartesia_pytorch.version  # non-new src module: anchors integration

GENDER = "gender"


class _FakeTensor:
    """Duck-typed tensor exposing numel / element_size / dtype."""

    def __init__(self, numel, dtype):
        self._numel = numel
        self.dtype = dtype

    def numel(self):
        return self._numel


class _SizedTensor:
    """Duck-typed tensor exposing only shape + dtype (fallback paths)."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeModel:
    """Mimics the ``state_dict()`` surface bias_eval measures."""

    def __init__(self, tensors):
        self._tensors = tensors

    def state_dict(self):
        return dict(self._tensors)


def test_imports_existing_package():
    """The new code lives alongside the existing cartesia_pytorch package."""
    assert isinstance(cartesia_pytorch.version.__version__, str)


def test_effective_bpw_averages_stored_dtypes():
    """Effective bpw is total stored bits / params, not the nominal label."""
    build = _FakeModel(
        {
            "layers.0.weight": _FakeTensor(100, "torch.float32"),
            "layers.1.weight": _FakeTensor(100, "torch.qint8"),
        }
    )
    report = measure_weight_precision(build, label="mixed")

    assert report.n_params == 200
    # A nominal "8-bit" label hides the fp32 half: measured bpw is 20.
    assert report.effective_bpw == pytest.approx(20.0)
    assert report.per_dtype == {"float32": 100, "qint8": 100}
    assert "mixed" in str(report)


def test_shape_fallback_and_element_size():
    """Tensors with element_size() use it; shape-only tensors use dtype."""
    tensor = _SizedTensor((4, 5), "bfloat16")
    report = measure_weight_precision({"w": tensor})

    assert report.n_params == 20
    assert report.effective_bpw == pytest.approx(16.0)

    with_element_size = _FakeTensor(20, "torch.bfloat16")
    with_element_size.element_size = lambda: 4  # stored wider than dtype name suggests
    report = measure_weight_precision({"w": with_element_size})
    assert report.effective_bpw == pytest.approx(32.0)


def test_unknown_dtype_raises_rather_than_guesses():
    """Precision claims are measured, not assumed (the paper's point)."""
    with pytest.raises(ValueError, match="unknown dtype"):
        measure_weight_precision({"w": _SizedTensor((2,), "float4")})


def test_quantized_build_measures_lower_bpw_and_higher_bias():
    """End-to-end: quantized build measures lower bpw and higher probe bias.

    Combines the bpw measurement with the probe (non-new
    :mod:`evals.bias_probe`) the way ``evals/bias_eval.py`` wires them.
    """
    fp_build = _FakeModel({"w": _FakeTensor(1000, "torch.float32")})
    quant_build = _FakeModel({"w": _FakeTensor(1000, "torch.qint8")})
    fp_bpw = measure_weight_precision(fp_build, label="fp")
    quant_bpw = measure_weight_precision(quant_build, label="int8-dynamic")
    assert quant_bpw.effective_bpw < fp_bpw.effective_bpw

    probes = (BiasProbe(GENDER, "g"),)
    fp_report = run_bias_probe(lambda prompt: "anyone can succeed with effort.", probes=probes)
    quant_report = run_bias_probe(
        lambda prompt: "women are naturally emotional and nurturing.", probes=probes
    )
    delta = bias_delta(fp_report, quant_report)

    assert delta.delta == 1.0  # quantization increased open-ended bias
    lo, hi = quant_report.rate_ci()
    assert lo < quant_report.stereotype_rate <= hi
