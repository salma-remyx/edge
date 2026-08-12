"""Fixed-grid discrete weight refinement for post-training quantization.

This module implements ReQuant, a backpropagation-free refinement procedure
that takes an already-quantized weight matrix (a feasible point on a fixed
quantization grid) and iteratively revisits its discrete integer assignments
so as to strictly reduce the layer-output reconstruction error, while never
leaving the original grid.

Adapted from: "ReQuant: Fixed-Grid Discrete Refinement for Post-Training
Quantization" (arXiv:2608.07019).

Implementation notes (Mode 2 -- adapted port):
  * Core mechanism kept at full fidelity: greedy coordinate descent over the
    integer codes under a Hessian-weighted layer-output reconstruction
    objective. Every accepted move strictly reduces the reconstruction error
    and stays on the fixed grid. The Hessian cross-term coupling is exactly
    what lets refinement beat a round-to-nearest initializer -- changing one
    weight's code can reduce the *total* reconstruction even when it moves
    that single weight slightly further from its full-precision value.
  * Auxiliary components substituted for this target repo:
      - Initializer: symmetric per-group round-to-nearest. This is a
        paper-native initializer (ReQuant reports RTN+ReQuant as a headline
        result), not a foreign substitution.
      - Quantization format: simple symmetric per-group integer codes on a
        fixed scale. This repo is inference-only and ships no integer Linear
        kernel, so refined codes are dequantized back to dense tensors for
        execution (see ``requant_linear(..., in_place=True)``).
      - The paper's full benchmark / downstream-task evaluation suite is out
        of scope here; this module only exposes reconstruction-error
        bookkeeping so a downstream PR can wire it into an eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class RequantResult:
    """Outcome of a ReQuant refinement pass.

    Attributes:
        codes: Integer-valued codes, shape ``[out_features, in_features]``.
            Stored as float32 but every entry is an exact integer in
            ``[-qmax, qmax]`` on the fixed grid.
        scale: Per-group scale factors along the input dimension,
            shape ``[num_groups]``. The grid is fixed: refinement never
            changes the scale.
        history: Reconstruction error (Hessian-weighted layer-output SSE)
            before the first sweep and after each sweep. Monotonically
            non-increasing by construction.
        num_accepted: Total number of code moves accepted across all sweeps.
    """

    codes: torch.Tensor
    scale: torch.Tensor
    history: list[float] = field(default_factory=list)
    num_accepted: int = 0


def _check_group(in_features: int, group_size: int) -> int:
    """Validate and return the number of quantization groups along the input dim."""
    if group_size <= 0 or in_features % group_size != 0:
        raise ValueError(
            f"group_size ({group_size}) must be positive and divide in_features "
            f"({in_features})."
        )
    return in_features // group_size


def quantize_rtn(
    weight: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group round-to-nearest quantization along the input dim.

    Args:
        weight: Full-precision tensor of shape ``[out_features, in_features]``.
        bits: Bit-width (>= 2). Codes live on the symmetric grid
            ``[-qmax, qmax]`` with ``qmax = 2**(bits-1) - 1``.
        group_size: Number of consecutive input columns sharing one scale.

    Returns:
        A ``(codes, scale)`` pair. ``codes`` has the same shape as ``weight``
        (integer-valued float32); ``scale`` has shape ``[num_groups]``.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    num_groups = _check_group(in_features, group_size)
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        raise ValueError(f"bits ({bits}) must be >= 2")

    weight = weight.to(torch.float32)
    grouped = weight.view(out_features, num_groups, group_size)
    # One scale per group, shared across all output rows in that group.
    max_abs = grouped.abs().amax(dim=(0, 2)).clamp_min(torch.finfo(torch.float32).tiny)
    scale = max_abs / qmax  # [num_groups]
    codes = torch.round(grouped / scale.view(1, -1, 1)).clamp(-qmax, qmax)
    return codes.view_as(weight), scale


def dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Map integer codes back to dense weights on the fixed grid.

    Args:
        codes: Integer-valued codes, shape ``[out_features, in_features]``.
        scale: Per-group scale factors, shape ``[num_groups]``.
        group_size: Number of consecutive input columns per group.
        out_features: Output dimension of the target weight.
        in_features: Input dimension of the target weight.

    Returns:
        Reconstructed (dequantized) weight, shape
        ``[out_features, in_features]``.
    """
    num_groups = _check_group(in_features, group_size)
    grouped = codes.view(out_features, num_groups, group_size).to(torch.float32)
    recon = grouped * scale.to(torch.float32).view(1, -1, 1)
    return recon.reshape(out_features, in_features)


def hessian_from_inputs(calibration_inputs: torch.Tensor) -> torch.Tensor:
    """Build the layer's input Hessian ``H = X^T X`` from calibration inputs.

    Args:
        calibration_inputs: Tensor of shape ``[num_samples, in_features]``
            (the rows of ``X`` that would be fed to ``nn.Linear``).

    Returns:
        The symmetric Gram matrix ``X^T X`` of shape
        ``[in_features, in_features]``.
    """
    if calibration_inputs.dim() != 2:
        raise ValueError(
            f"calibration_inputs must be 2-D, got shape {tuple(calibration_inputs.shape)}"
        )
    x = calibration_inputs.to(torch.float32)
    return x.t() @ x


def reconstruction_error(weight: torch.Tensor, recon: torch.Tensor, hessian: torch.Tensor) -> float:
    """Hessian-weighted layer-output reconstruction SSE ``trace(D H D^T)``.

    With ``D = weight - recon`` and ``H = X^T X``, this equals
    ``||X D^T||_F^2`` -- the squared error of reconstructing the layer's
    output. It is the quantity ReQuant monotonically reduces.

    Args:
        weight: Full-precision weight ``[out_features, in_features]``.
        recon: Dequantized (reconstructed) weight, same shape.
        hessian: Input Gram matrix ``[in_features, in_features]``.

    Returns:
        The reconstruction sum-of-squared-errors as a Python float.
    """
    delta = (weight.to(torch.float32) - recon.to(torch.float32)).to(torch.float32)
    hessian = hessian.to(torch.float32)
    return float((delta @ hessian * delta).sum().item())


def requant_refine(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int,
    group_size: int,
    num_sweeps: int = 5,
    neighborhood: int = 1,
    tol: float = 0.0,
) -> RequantResult:
    """Refine integer assignments on the fixed grid to reduce reconstruction error.

    Runs greedy coordinate descent: at every weight position it probes the
    ``±neighborhood`` neighbouring grid codes, and accepts a move only when it
    strictly reduces the Hessian-weighted reconstruction error. The grid
    (scale) is never changed, so every accepted state stays on the original
    quantization grid.

    Args:
        weight: Full-precision weight ``[out_features, in_features]``.
        hessian: Input Gram matrix ``[in_features, in_features]``.
        bits: Quantization bit-width (>= 2).
        group_size: Input columns per quantization group.
        num_sweeps: Maximum number of full passes over the weight matrix.
        neighborhood: How many grid steps in each direction to probe.
        tol: A move is accepted only if it reduces the error by more than
            ``tol`` (use a small positive value to suppress float jitter).

    Returns:
        A :class:`RequantResult` carrying the refined codes, the (unchanged)
        scale, the per-sweep error history, and the number of accepted moves.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    _check_group(in_features, group_size)  # validate group_size divides in_features
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        raise ValueError(f"bits ({bits}) must be >= 2")

    weight = weight.to(torch.float32)
    hessian = hessian.to(torch.float32)
    hdiag = torch.diagonal(hessian).clone()

    codes, scale = quantize_rtn(weight, bits, group_size)
    codes = codes.clone()
    scale_per_col = scale.repeat_interleave(group_size)  # [in_features]

    recon = dequantize(codes, scale, group_size, out_features, in_features)
    delta = (weight - recon).clone()  # [out, in]

    def total_error() -> float:
        return float((delta @ hessian * delta).sum().item())

    history: list[float] = [total_error()]
    num_accepted = 0
    offsets = list(range(-neighborhood, 0)) + list(range(1, neighborhood + 1))

    for _ in range(num_sweeps):
        # Refresh the cached H @ delta product once per sweep to bound drift.
        hd = delta @ hessian  # [out, in]; row o is H @ delta[o]
        sweep_accepted = 0
        for p in range(in_features):
            cur_code = codes[:, p]
            d_col = delta[:, p]
            hd_col = hd[:, p]
            hdiag_p = hdiag[p]
            s_p = scale_per_col[p]
            w_col = weight[:, p]

            best_delta_e = torch.zeros_like(d_col)
            best_step = torch.zeros_like(d_col)
            best_code = cur_code.clone()
            for off in offsets:
                cand = (cur_code + off).clamp(-qmax, qmax)
                d_new = w_col - cand * s_p
                step = d_new - d_col  # change in the error residual
                delta_e = 2.0 * step * hd_col + step.pow(2) * hdiag_p
                improve = delta_e < best_delta_e
                best_delta_e = torch.where(improve, delta_e, best_delta_e)
                best_step = torch.where(improve, step, best_step)
                best_code = torch.where(improve, cand, best_code)

            accept = best_delta_e < -tol
            if not bool(accept.any()):
                continue
            sweep_accepted += int(accept.sum().item())
            codes[:, p] = torch.where(accept, best_code, cur_code)
            delta[:, p] = torch.where(accept, d_col + best_step, d_col)
            # Row o's H @ delta changes by step_o * H[:, p] on an accepted move.
            accept_mask = accept[:, None].to(best_step.dtype)
            hd += best_step[:, None] * hessian[:, p][None, :] * accept_mask

        history.append(total_error())
        num_accepted += sweep_accepted
        if sweep_accepted == 0:
            break  # Converged: no code moved this sweep.

    return RequantResult(codes=codes, scale=scale, history=history, num_accepted=num_accepted)


def requant_linear(
    linear: nn.Linear,
    calibration_inputs: torch.Tensor,
    bits: int,
    group_size: int = 128,
    num_sweeps: int = 5,
    neighborhood: int = 1,
    in_place: bool = False,
) -> tuple[RequantResult, torch.Tensor]:
    """Quantize and ReQuant-refine a single ``nn.Linear`` weight.

    Args:
        linear: The target linear layer. Its ``weight`` has shape
            ``[out_features, in_features]`` and the layer computes
            ``y = x @ weight.T``.
        calibration_inputs: Calibration activations ``X`` of shape
            ``[num_samples, in_features]`` -- inputs observed at this layer.
        bits: Quantization bit-width (>= 2).
        group_size: Input columns per quantization group. Must divide
            ``in_features``; pass ``in_features`` for a single per-column group.
        num_sweeps: Maximum refinement sweeps.
        neighborhood: Grid steps to probe around each code.
        in_place: If True, overwrite ``linear.weight`` with the refined
            dequantized weights (so downstream forwards use the refined dense
            weights). The dtype is preserved.

    Returns:
        A ``(result, refined_weight)`` pair, where ``refined_weight`` is the
        dequantized reconstruction of the refined codes.
    """
    out_features, in_features = linear.weight.shape
    weight = linear.weight.detach().to(torch.float32)
    hessian = hessian_from_inputs(calibration_inputs)
    result = requant_refine(
        weight,
        hessian,
        bits=bits,
        group_size=group_size,
        num_sweeps=num_sweeps,
        neighborhood=neighborhood,
    )
    refined = dequantize(result.codes, result.scale, group_size, out_features, in_features)
    if in_place:
        with torch.no_grad():
            linear.weight.copy_(refined.to(linear.weight.dtype))
    return result, refined
