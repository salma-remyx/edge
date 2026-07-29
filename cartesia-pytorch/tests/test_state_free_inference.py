"""Tests for state-free (transfer-function) SSM inference and its wiring.

Exercises the RTF state-free path both directly and through the hook added to
the existing ``DiscreteMamba2`` mixer, and checks the paper's central claim that
state-free inference cost is independent of the state size while the native scan
scales with it.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

# The Llamba package __init__ eagerly imports the full model stack
# (transformers, mamba_ssm), which is unavailable in the unit-test environment.
# Register lightweight namespace packages so importing the mixer submodules does
# not trigger it. This loads the real discrete_mamba2 / state_free source -- only
# the package boilerplate is bypassed.
_MIXERS = Path(__file__).resolve().parents[1] / "cartesia_pytorch" / "Llamba" / "mixers"
_LLAMBA = _MIXERS.parent
for _name, _path in [
    ("cartesia_pytorch", _LLAMBA.parent),
    ("cartesia_pytorch.Llamba", _LLAMBA),
    ("cartesia_pytorch.Llamba.mixers", _MIXERS),
]:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = [str(_path)]
        _mod.__package__ = _name
        sys.modules[_name] = _mod

from cartesia_pytorch.Llamba.mixers import state_free  # noqa: E402
from cartesia_pytorch.Llamba.mixers.discrete_mamba2 import DiscreteMamba2  # noqa: E402


def _system(channels, n, seed=0):
    """Sample a stable system and its RTF rational coefficients (float64)."""
    g = torch.Generator().manual_seed(seed)
    poles, residues, h_0 = state_free.stable_rtf_system(channels, n, generator=g)
    poles, residues, h_0 = poles.double(), residues.double(), h_0.double()
    a, b_tilde = state_free.rational_coefficients(poles, residues)
    return poles, residues, h_0, a, b_tilde


def _rel_err(a_tensor, b_tensor):
    return (a_tensor - b_tensor).abs().max().item() / (b_tensor.abs().max().item() + 1e-30)


@pytest.mark.parametrize("n", [1, 8, 32])
def test_state_free_matches_native_scan(n):
    """The state-free path reproduces the exact diagonal-scan output."""
    batch, channels, seq_len = 2, 3, 256
    poles, residues, h_0, a, b_tilde = _system(channels, n)
    u = torch.randn(
        batch, channels, seq_len, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
    )
    y_free = state_free.state_free_forward(u, a, b_tilde, h_0)
    y_scan = state_free.native_scan_forward(u, poles, residues, h_0)
    assert _rel_err(y_free, y_scan) < 1e-9


def test_wiring_discrete_mamba2_state_free_forward():
    """The hook on the existing DiscreteMamba2 mixer delegates to the RTF path."""
    batch, channels, n, seq_len = 2, 4, 16, 128
    poles, residues, h_0, a, b_tilde = _system(channels, n)
    u = torch.randn(
        batch, channels, seq_len, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
    )
    y_wired = DiscreteMamba2.state_free_forward(u, a, b_tilde, h_0)
    y_scan = state_free.native_scan_forward(u, poles, residues, h_0)
    assert _rel_err(y_wired, y_scan) < 1e-9


def test_scaling_state_free_independent_of_state_size():
    """Native cost grows with n; state-free cost and state are flat in n."""
    results = state_free.measure_state_size_scaling(
        seq_len=1024, state_sizes=(16, 64, 256), num_channels=16, batch_size=1, repeats=1
    )
    assert [r["native_state_elements"] for r in results] == [16, 64, 256]
    assert all(r["state_free_state_elements"] == 0 for r in results)
    # Native compute grows (linearly) with n; state-free compute is identical
    # across n, i.e. independent of the state size.
    native_flops = [r["native_flops"] for r in results]
    assert native_flops == sorted(native_flops) and native_flops[-1] > native_flops[0]
    assert len({r["state_free_flops"] for r in results}) == 1


def test_mixer_shape_and_finite():
    """The state-free mixer maps (B, L, d_model) -> (B, L, d_model), finite."""
    mixer = state_free.StateFreeMixer(d_model=12, state_size=32).double()
    out = mixer(torch.randn(3, 100, 12, dtype=torch.float64))
    assert out.shape == (3, 100, 12)
    assert torch.isfinite(out).all()
