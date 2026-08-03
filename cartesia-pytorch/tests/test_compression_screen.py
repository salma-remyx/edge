"""Tests for the data-free compression-fidelity screen.

Two things are covered:

1. The screen module's core operator-specificity dissociation -- at a *matched*
   error rate, coherent SVD truncation is flagged while incoherent magnitude
   pruning is not, because the governing axis is the *coherence* of the
   compression error, not the magnitude of the damage (arXiv:2607.28196).
2. Integration of the screen into the existing generation benchmark CLI via
   ``evals.generation.run_compression_screen`` (imports a NON-NEW module and
   exercises the wiring edit on real ``nn.Module`` instances).
"""

import pytest
import torch
import torch.nn as nn
from evals.generation import run_compression_screen

from cartesia_pytorch.utils.compression_screen import (
    DEFAULT_SCREEN_THRESHOLD,
    ScreenReport,
    format_screen_report,
    screen_state_dicts,
    screen_tensor,
)

WEIGHT_NAME = "backbones.linear.weight"


def _decaying_spectrum_weight(n, decay=4.0, seed=1):
    """A realistic dense weight matrix with a decaying singular spectrum.

    Real trained weights are dense but low effective rank, so truncating the
    small spectral tail is a *gentle* (low error-rate) yet coherent (low-rank)
    compression -- exactly the regime the paper warns about.

    Args:
        n: Matrix side length.
        decay: Spectral decay steepness.
        seed: RNG seed for reproducibility.

    Returns:
        An ``(n, n)`` float tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    unit_u, _ = torch.linalg.qr(torch.randn(n, n, generator=generator))
    unit_v, _ = torch.linalg.qr(torch.randn(n, n, generator=generator))
    spectrum = torch.exp(-torch.linspace(0.0, decay, n))
    return (unit_u * spectrum) @ unit_v.T


def _svd_truncate(weight, drop):
    """Coherent low-rank compression: drop the smallest ``drop`` singular values."""
    keep = min(weight.shape) - drop
    unit_u, sing, unit_vh = torch.linalg.svd(weight, full_matrices=False)
    return unit_u[:, :keep] @ torch.diag(sing[:keep]) @ unit_vh[:keep, :]


def _magnitude_prune_matched(weight, target_error):
    """Incoherent compression: zero smallest-magnitude entries to match a norm."""
    pruned = weight.clone()
    flat = pruned.flatten()
    order = torch.argsort(weight.abs().flatten())
    target_sq = target_error**2
    cumulative = 0.0
    index = 0
    while cumulative < target_sq and index < order.numel():
        position = order[index]
        cumulative += flat[position].item() ** 2
        flat[position] = 0.0
        index += 1
    return pruned


def test_coherent_svd_flagged_incoherent_pruning_not_matched_error_rate():
    """Core dissociation: matched damage, coherence is the governing axis."""
    weight = _decaying_spectrum_weight(256)
    weight_svd = _svd_truncate(weight, drop=4)
    weight_prune = _magnitude_prune_matched(weight, (weight - weight_svd).norm().item())

    svd_report = screen_state_dicts({WEIGHT_NAME: weight}, {WEIGHT_NAME: weight_svd})
    prune_report = screen_state_dicts({WEIGHT_NAME: weight}, {WEIGHT_NAME: weight_prune})

    # Same damage magnitude: the magnitude of the error does not predict the
    # outcome (error-rates matched to within 10%).
    relative_diff = abs(svd_report.error_rate - prune_report.error_rate) / svd_report.error_rate
    assert relative_diff < 0.1
    # Coherence separates them: SVD error is low-rank, pruning error is spread.
    assert svd_report.coherent_fraction > 5 * prune_report.coherent_fraction
    # The screen flags the coherent low-rank build and passes the pruning build
    # at the default fixed threshold.
    assert svd_report.flagged is True
    assert prune_report.flagged is False


def test_unchanged_weights_score_zero_and_pass():
    """An identical checkpoint is not flagged."""
    weight = _decaying_spectrum_weight(128)
    report = screen_state_dicts({WEIGHT_NAME: weight}, {WEIGHT_NAME: weight.clone()})
    assert report.score == 0.0
    assert report.coherent_fraction == 0.0
    assert report.flagged is False


def test_screen_tensor_shape_mismatch_raises():
    """A shape mismatch between original and compressed is a hard error."""
    with pytest.raises(ValueError):
        screen_tensor(torch.randn(8, 8), torch.randn(8, 7), name="bad")


def test_state_dicts_skips_bias_and_unmatched_keys():
    """1D (bias-like) tensors and unmatched keys are skipped, not screened."""
    weight = torch.randn(16, 16)
    original = {"w": weight, "bias": torch.randn(16), "extra": torch.randn(16, 16)}
    compressed = {"w": _svd_truncate(weight, 2), "bias": torch.randn(16)}
    report = screen_state_dicts(original, compressed)
    assert report.n_screened == 1


def test_threshold_is_configurable():
    """A higher threshold lets a coherent build pass; lower flags more."""
    weight = _decaying_spectrum_weight(256)
    weight_svd = _svd_truncate(weight, drop=4)
    loose = screen_state_dicts({WEIGHT_NAME: weight}, {WEIGHT_NAME: weight_svd}, threshold=1.0)
    strict = screen_state_dicts({WEIGHT_NAME: weight}, {WEIGHT_NAME: weight_svd}, threshold=0.0)
    assert loose.flagged is False
    assert strict.flagged is True
    assert DEFAULT_SCREEN_THRESHOLD > 0.0


def test_format_screen_report_contains_axes_and_verdict():
    """The rendered report surfaces both axes, the score, and the verdict."""
    weight = _decaying_spectrum_weight(64)
    weight_svd = _svd_truncate(weight, drop=4)
    report = screen_state_dicts({WEIGHT_NAME: weight}, {WEIGHT_NAME: weight_svd})
    text = format_screen_report(report)
    assert "error-rate" in text
    assert "coherent-fraction" in text
    assert "verdict" in text


class _TinyModel(nn.Module):
    """A minimal ``nn.Module`` exposing a weight via ``state_dict``."""

    def __init__(self, weight):
        super().__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)


def test_run_compression_screen_flags_coherent_compression():
    """Integration: the generation CLI hook flags a compressed module."""
    weight = _decaying_spectrum_weight(64)
    baseline = _TinyModel(weight)
    compressed = _TinyModel(_svd_truncate(weight, drop=4))

    report = run_compression_screen(compressed, baseline)

    assert isinstance(report, ScreenReport)
    assert report.n_screened >= 1
    assert report.flagged is True


def test_run_compression_screen_passes_unchanged_model():
    """Integration: an unchanged model is not flagged through the hook."""
    weight = _decaying_spectrum_weight(64)
    baseline = _TinyModel(weight)
    identical = _TinyModel(weight)

    report = run_compression_screen(identical, baseline)

    assert isinstance(report, ScreenReport)
    assert report.score == 0.0
    assert report.flagged is False
