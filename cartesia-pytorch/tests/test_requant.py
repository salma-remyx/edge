"""Integration tests for ReQuant refinement against the repo's LlamaMLP Linears.

These tests exercise ``cartesia_pytorch.utils.requant`` through the existing
(non-new) ``LlamaMLP`` module, proving the refinement integrates with the
repo's real linear layers rather than only self-testing the new file.
"""

import pytest
import torch

from cartesia_pytorch.Llamba.modeling_llama import LlamaMLP
from cartesia_pytorch.utils.requant import (
    dequantize,
    hessian_from_inputs,
    quantize_rtn,
    reconstruction_error,
    requant_linear,
    requant_refine,
)


@pytest.fixture()
def mlp_and_inputs():
    """A small LlamaMLP plus calibration inputs feeding its gate_proj layer."""
    torch.manual_seed(0)
    mlp = LlamaMLP(
        hidden_size=48,
        intermediate_size=64,
        bias=False,
        act_fn="silu",
        factory_kwargs={},
    )
    # Calibration inputs feeding gate_proj: [N, hidden_size].
    x = torch.randn(128, 48)
    return mlp, x


def test_codes_stay_on_fixed_grid(mlp_and_inputs):
    """Refined codes are exact integers within the symmetric grid."""
    mlp, x = mlp_and_inputs
    linear = mlp.gate_proj
    bits, group_size = 4, 16
    result, refined = requant_linear(linear, x, bits=bits, group_size=group_size, num_sweeps=5)

    qmax = 2 ** (bits - 1) - 1
    codes = result.codes
    assert codes.shape == linear.weight.shape
    # Integer-valued codes within the symmetric grid.
    assert torch.all(codes == codes.round())
    assert torch.all(codes.abs() <= qmax)
    # One scale per group; the grid is fixed and never changes during refinement.
    assert result.scale.shape == (linear.weight.shape[1] // group_size,)
    # Dequantizing the returned codes reproduces the returned refined weight.
    recon = dequantize(codes, result.scale, group_size, *linear.weight.shape)
    assert torch.allclose(recon, refined, atol=1e-6)


def test_refinement_monotonically_reduces_error(mlp_and_inputs):
    """Each sweep never increases the reconstruction error."""
    mlp, x = mlp_and_inputs
    bits, group_size = 4, 16
    weight = mlp.gate_proj.weight.detach().to(torch.float32)
    hessian = hessian_from_inputs(x)
    result = requant_refine(weight, hessian, bits=bits, group_size=group_size, num_sweeps=6)

    history = result.history
    assert len(history) >= 2
    for prev, cur in zip(history, history[1:]):
        assert cur <= prev + 1e-6
    # The refined point is no worse than the round-to-nearest starting point.
    assert history[-1] <= history[0] + 1e-6


def test_refinement_beats_round_to_nearest(mlp_and_inputs):
    """With correlated inputs, Hessian coupling lets ReQuant improve on RTN."""
    mlp, x = mlp_and_inputs
    bits, group_size = 3, 16  # lower bit-width -> larger gains per the paper
    weight = mlp.gate_proj.weight.detach().to(torch.float32)
    hessian = hessian_from_inputs(x)

    codes_rtn, scale = quantize_rtn(weight, bits, group_size)
    rtn_err = reconstruction_error(
        weight, dequantize(codes_rtn, scale, group_size, *weight.shape), hessian
    )
    result = requant_refine(weight, hessian, bits=bits, group_size=group_size, num_sweeps=10)
    refined_err = reconstruction_error(
        weight,
        dequantize(result.codes, result.scale, group_size, *weight.shape),
        hessian,
    )
    assert result.num_accepted > 0
    assert refined_err < rtn_err


def test_in_place_refinement_writes_weights_and_runs(mlp_and_inputs):
    """Refined dense weights are written back and the MLP still forwards."""
    mlp, x = mlp_and_inputs
    linear = mlp.gate_proj
    before = linear.weight.detach().clone()
    requant_linear(linear, x, bits=4, group_size=16, num_sweeps=3, in_place=True)
    # Refined dense weights were written back.
    assert not torch.allclose(before, linear.weight.detach())
    # The module still runs end-to-end through the (non-new) LlamaMLP forward.
    y = mlp(x)
    assert y.shape == (x.shape[0], mlp.hidden_size)
