"""Data-free screen for the compression-fidelity blind spot.

Mode-2 adapted port of "Fidelity Is Not Safety: Gently-Compressed LLMs Pass
Every Data-Free Quality Guard Yet Invent Procedure Steps in Agentic Execution"
(arXiv:2607.28196).

Gently low-rank-compressed LLMs (e.g. SVD weight truncation) clear the standard
quality stack (perplexity, MMLU, random-probe fidelity) yet invent procedure
steps as agents, while magnitude pruning matched to the same perplexity does
not. The blind spot is structural: a fidelity probe is a fidelity oracle, so it
cannot see the governing axis -- the *coherence* of the compression error times
its *rate*.

The paper's data-free screen is a two-axis statistic of the weight error
``E = W - W'``:

* ``error_rate`` -- ``||E||_F / ||W||_F``, the relative damage.
* ``coherent_fraction`` -- ``sigma_max(E)^2 / ||E||_F^2``, the error energy in
  the dominant singular direction. Low-rank (SVD truncation) error concentrates
  here; spread (magnitude-pruning / rounding) error does not.

A build is flagged when ``coherent_fraction * error_rate`` exceeds a fixed
threshold, reproducing the operator-specific dissociation: at a matched error
rate, SVD-truncated weights are flagged and magnitude-pruned weights are not.

Kept at full fidelity: the two-axis statistic and the coherence-times-rate
mechanism. Intentionally out of scope (downstream PR): the agentic SOP
"invented-step" canary and its paired-confidence-interval harness. This is the
data-free gate that decides whether to run that canary before agentic deploy,
using only the original and compressed weight tensors: no data, no inference.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch

# Fixed screen threshold on ``coherent_fraction * error_rate``. The paper
# reports fixed thresholds that generalize across architectures; this default
# is conservative (flags gentle low-rank builds, lets pruning/quantization
# pass) and is configurable from the CLI / function argument.
DEFAULT_SCREEN_THRESHOLD: float = 1e-3
# Tensors smaller than this along any screened axis are skipped: tiny norms
# and biases contribute numerical noise rather than signal.
DEFAULT_MIN_DIM: int = 2


@dataclass
class TensorScreen:
    """Two-axis compression-error statistic for a single weight tensor.

    Attributes:
        name: Parameter name (or label).
        weight_norm: ``||W||_F`` of the original tensor.
        error_norm: ``||E||_F = ||W - W'||_F`` of the compression error.
        coherent_fraction: ``sigma_max(E)^2 / ||E||_F^2`` in ``(0, 1]``. High
            for low-rank (coherent) error, ~``1/rank`` for spread error.
    """

    name: str
    weight_norm: float
    error_norm: float
    coherent_fraction: float

    @property
    def error_rate(self) -> float:
        """Relative Frobenius damage ``||E||_F / ||W||_F``."""
        return self.error_norm / self.weight_norm if self.weight_norm else 0.0

    @property
    def score(self) -> float:
        """Coherence-times-rate screen value for this tensor."""
        return self.coherent_fraction * self.error_rate


@dataclass
class ScreenReport:
    """Aggregated screen verdict for a compressed model.

    Attributes:
        error_rate: Global relative Frobenius error pooled over screened
            tensors, ``sqrt(sum ||E_i||^2) / sqrt(sum ||W_i||^2)``.
        coherent_fraction: Error-energy-weighted mean of per-tensor coherent
            fractions -- "how coherent is the compression error overall".
        threshold: Fixed flagging threshold.
        n_screened: Number of weight tensors screened.
        per_tensor: Per-tensor breakdown (largest scores first).
    """

    error_rate: float
    coherent_fraction: float
    threshold: float
    n_screened: int
    per_tensor: Sequence[TensorScreen] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Coherence-times-rate screen value for the model."""
        return self.coherent_fraction * self.error_rate

    @property
    def flagged(self) -> bool:
        """Whether the build is flagged by the data-free screen."""
        return self.score > self.threshold


def _matricize(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape an n-d tensor to 2D for spectral analysis.

    The first axis (output features) is preserved; all remaining axes are
    flattened, which is the standard view for spectral concentration of
    embedding / conv / linear weights.

    Args:
        tensor: A >=2D tensor.

    Returns:
        A 2D view of ``tensor``.
    """
    if tensor.ndim <= 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def _coherent_fraction(error: torch.Tensor, top_k: int = 1) -> float:
    """Fraction of error energy in the dominant singular direction(s).

    Args:
        error: 2D compression-error matrix.
        top_k: Number of dominant singular directions counted as "coherent".
            ``1`` (the single dominant direction) is the most discriminative
            choice for the gentle-compression regime.

    Returns:
        ``sum_{i<=top_k} sigma_i(E)^2 / ||E||_F^2`` in ``[0, 1]``.
    """
    total = torch.linalg.vector_norm(error).item()
    if total == 0.0:
        return 0.0
    if top_k <= 1:
        # sigma_max(E) -- cheaper than a full SVD and all we need for k=1.
        spectral = torch.linalg.matrix_norm(error, ord=2).item()
        return (spectral * spectral) / (total * total)
    k = min(top_k, min(error.shape))
    singular_values = torch.linalg.svdvals(error)
    top_energy = (singular_values[:k] ** 2).sum().item()
    return top_energy / (total * total)


def screen_tensor(
    weight: torch.Tensor,
    compressed_weight: torch.Tensor,
    name: str = "",
    top_k: int = 1,
) -> TensorScreen:
    """Compute the two-axis compression-error statistic for one tensor.

    Args:
        weight: Original weight tensor (>=2D).
        compressed_weight: Compressed weight tensor, same shape as ``weight``.
        name: Optional label for the tensor.
        top_k: Coherent-energy top-k (see :func:`_coherent_fraction`).

    Returns:
        The per-tensor :class:`TensorScreen`.

    Raises:
        ValueError: If the shapes differ or the original weight is all zeros.
    """
    # Cast to CPU float32 so the screen is robust to dtype (bf16/fp16/quantized
    # packing) and device (baseline vs. loaded model) differences, and never
    # peaks GPU memory on large checkpoints.
    w = weight.detach().to(dtype=torch.float32, device="cpu")
    wc = compressed_weight.detach().to(dtype=torch.float32, device="cpu")
    if w.shape != wc.shape:
        raise ValueError(f"Shape mismatch for {name!r}: {tuple(w.shape)} vs {tuple(wc.shape)}")
    weight_norm = torch.linalg.vector_norm(w).item()
    if weight_norm == 0.0:
        raise ValueError(f"Original weight {name!r} has zero norm; cannot screen.")
    error = w - wc
    error_norm = torch.linalg.vector_norm(error).item()
    coherent = _coherent_fraction(_matricize(error), top_k=top_k) if error_norm else 0.0
    return TensorScreen(
        name=name,
        weight_norm=weight_norm,
        error_norm=error_norm,
        coherent_fraction=coherent,
    )


def screen_state_dicts(
    original: Mapping[str, torch.Tensor],
    compressed: Mapping[str, torch.Tensor],
    threshold: float = DEFAULT_SCREEN_THRESHOLD,
    min_dim: int = DEFAULT_MIN_DIM,
    top_k: int = 1,
    name_globs: Optional[Sequence[str]] = None,
) -> ScreenReport:
    """Screen a compressed model against its original weights.

    Only floating-point tensors with >=2D and every axis >= ``min_dim`` are
    screened (norms and biases are skipped to avoid noise). The model-level
    ``error_rate`` pools Frobenius energy across tensors; the model-level
    ``coherent_fraction`` is the error-energy-weighted mean of per-tensor
    coherent fractions, so untouched tensors (zero error) do not dilute it.

    Args:
        original: Original model state dict.
        compressed: Compressed model state dict (same keys / shapes).
        threshold: Fixed flagging threshold on ``coherent_fraction * error_rate``.
        min_dim: Minimum size along each screened axis.
        top_k: Coherent-energy top-k passed to :func:`screen_tensor`.
        name_globs: Optional ``fnmatch`` patterns; if given, only matching
            parameter names are screened.

    Returns:
        The aggregated :class:`ScreenReport`.
    """
    per_tensor: list[TensorScreen] = []
    total_err_sq = 0.0
    total_weight_sq = 0.0
    weighted_coherent = 0.0
    for name, weight in original.items():
        if name not in compressed:
            continue
        if not torch.is_floating_point(weight) or weight.ndim < 2:
            continue
        if any(dim < min_dim for dim in weight.shape):
            continue
        if name_globs is not None and not any(
            fnmatch.fnmatch(name, pattern) for pattern in name_globs
        ):
            continue
        result = screen_tensor(weight, compressed[name], name=name, top_k=top_k)
        per_tensor.append(result)
        total_err_sq += result.error_norm**2
        total_weight_sq += result.weight_norm**2
        weighted_coherent += result.coherent_fraction * result.error_norm**2

    per_tensor.sort(key=lambda item: item.score, reverse=True)
    error_rate = (total_err_sq / total_weight_sq) ** 0.5 if total_weight_sq else 0.0
    coherent_fraction = weighted_coherent / total_err_sq if total_err_sq else 0.0
    return ScreenReport(
        error_rate=error_rate,
        coherent_fraction=coherent_fraction,
        threshold=threshold,
        n_screened=len(per_tensor),
        per_tensor=per_tensor,
    )


def format_screen_report(report: ScreenReport, per_tensor_limit: int = 10) -> str:
    """Render a :class:`ScreenReport` as a human-readable string.

    Args:
        report: The screen report to render.
        per_tensor_limit: Maximum number of per-tensor rows to include.

    Returns:
        A multi-line string summarizing the two axes, the score, and the
        verdict, followed by the worst-offending tensors.
    """
    verdict = (
        "FLAG (coherent low-rank error -- screen before agentic deploy)"
        if report.flagged
        else "PASS"
    )
    lines = [
        "Compression-fidelity blind-spot screen (data-free)",
        "===================================================",
        f"  tensors screened : {report.n_screened}",
        f"  error-rate       : {report.error_rate:.6g}",
        f"  coherent-fraction: {report.coherent_fraction:.6g}",
        f"  score (cf x er)  : {report.score:.6g}",
        f"  threshold        : {report.threshold:.6g}",
        f"  verdict          : {verdict}",
    ]
    if report.per_tensor:
        lines.append("  top offending tensors (by cf x er):")
        for item in report.per_tensor[:per_tensor_limit]:
            lines.append(
                f"    {item.score:.6g}  cf={item.coherent_fraction:.4g} "
                f"er={item.error_rate:.4g}  {item.name}"
            )
    return "\n".join(lines)
