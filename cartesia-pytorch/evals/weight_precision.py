"""Effective bits-per-weight measurement for model builds.

One of the headline contributions of *QuantiBias: Benchmarking
Quantization-Induced Bias in LLMs* (arXiv:2607.21063, Section 3) is that
nominal quantization labels ("4-bit", "8-bit") systematically overstate
the actual compression: scales, zero-points, metadata, and layers left
in full precision all add stored bits. The paper therefore measures
*effective* bits-per-weight (bpw) directly from each stored weight
tensor rather than trusting the label, and finds labels misstate
quantization by 11-40%.

This module ports that measurement to the repo's PyTorch builds. The
paper's reference implementation inspects GGUF files for llama.cpp; here
we substitute the target-native equivalent (adapted port, Mode 2): the
stored tensors of an in-memory HF-style build, via its ``state_dict()``
(which includes quantized packed weights and scales), its
``named_parameters()``, or any iterable of ``(name, tensor)`` pairs.

The module is torch-free: tensors are duck-typed (``numel`` /
``element_size`` / ``dtype`` / ``shape``), so it stays importable and
testable in lightweight environments without torch installed.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

#: Stored bits per element for the dtypes this repo's builds use. Names
#: are matched case-insensitively with any ``torch.`` / framework prefix
#: stripped, so ``torch.bfloat16`` and ``bfloat16`` both resolve.
_DTYPE_BITS = {
    "float64": 64,
    "double": 64,
    "float32": 32,
    "float": 32,
    "bfloat16": 16,
    "float16": 16,
    "half": 16,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "int64": 64,
    "long": 64,
    "int32": 32,
    "qint32": 32,
    "int16": 16,
    "short": 16,
    "int8": 8,
    "qint8": 8,
    "uint8": 8,
    "quint8": 8,
    "byte": 8,
    "char": 8,
    "bool": 8,
}


@dataclass
class WeightPrecisionReport:
    """Measured weight precision of one model build.

    ``effective_bpw`` is ``total_bits / n_params`` over every stored
    tensor -- the paper's replacement for the nominal quantization
    label. ``per_dtype`` maps each stored dtype name to its parameter
    count so callers can see which layers stayed in full precision.
    """

    label: str
    n_params: int
    total_bits: int
    effective_bpw: float
    per_dtype: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        """Render a short human-readable summary of the measurement."""
        name = self.label or "build"
        lines = [
            f"{name}: {self.n_params} params, effective {self.effective_bpw:.2f} bits/weight",
            "per-dtype param counts:",
        ]
        for dtype, count in sorted(self.per_dtype.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {dtype}: {count} ({count / self.n_params:.1%})")
        return "\n".join(lines)


def _dtype_name(tensor) -> str:
    """Return the normalized dtype name for a duck-typed tensor."""
    raw = str(getattr(tensor, "dtype", "float32")).lower()
    return raw.rsplit(".", 1)[-1]


def _numel(tensor) -> int:
    """Return the element count of a duck-typed tensor."""
    for attr in ("numel", "nelement"):
        method = getattr(tensor, attr, None)
        if callable(method):
            return int(method())
    n = 1
    for dim in getattr(tensor, "shape", ()):
        n *= int(dim)
    return n


def _bits_per_element(tensor, dtype_name: str) -> int:
    """Return the stored bits per element of a duck-typed tensor.

    Prefers the tensor's own ``element_size()`` (authoritative for
    torch tensors, including quantized packed weights); falls back to
    the :data:`_DTYPE_BITS` table. Raises ``ValueError`` for unknown
    dtypes rather than guessing -- the paper's point is that precision
    claims must be measured, not assumed.
    """
    element_size = getattr(tensor, "element_size", None)
    if callable(element_size):
        return int(element_size()) * 8
    try:
        return _DTYPE_BITS[dtype_name]
    except KeyError:
        raise ValueError(
            f"unknown dtype {dtype_name!r}: cannot measure stored bits. "
            "Add it to _DTYPE_BITS or use tensors exposing element_size()."
        ) from None


def _iter_named_tensors(build) -> Iterable[Tuple[str, object]]:
    """Yield ``(name, tensor)`` pairs from a model or tensor mapping."""
    state_dict = getattr(build, "state_dict", None)
    if callable(state_dict):
        return state_dict().items()
    named_parameters = getattr(build, "named_parameters", None)
    if callable(named_parameters):
        return named_parameters()
    items = getattr(build, "items", None)
    if callable(items):
        return items()
    return iter(build)


def measure_weight_precision(
    build,
    label: str = "",
    skip: Optional[tuple[str, ...]] = None,
) -> WeightPrecisionReport:
    """Measure the effective bits-per-weight of a model build.

    Args:
        build: A model exposing ``state_dict()`` (preferred -- includes
            quantized packed weights and scales) or ``named_parameters()``,
            or any mapping / iterable of ``(name, tensor)`` pairs.
        label: Human-readable build name for the report (e.g. ``"fp"``,
            ``"int8-dynamic"``).
        skip: Optional tensor-name substrings to exclude (e.g. norms or
            embeddings the caller does not count as weights).

    Returns:
        A :class:`WeightPrecisionReport` with the effective bpw -- the
        measured replacement for the nominal quantization label.
    """
    n_params = 0
    total_bits = 0
    per_dtype: dict[str, int] = {}
    for name, tensor in _iter_named_tensors(build):
        if skip and any(part in name for part in skip):
            continue
        dtype_name = _dtype_name(tensor)
        count = _numel(tensor)
        n_params += count
        total_bits += count * _bits_per_element(tensor, dtype_name)
        per_dtype[dtype_name] = per_dtype.get(dtype_name, 0) + count
    effective_bpw = total_bits / n_params if n_params else 0.0
    return WeightPrecisionReport(
        label=label,
        n_params=n_params,
        total_bits=total_bits,
        effective_bpw=effective_bpw,
        per_dtype=per_dtype,
    )
