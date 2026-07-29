"""State-free SSM inference via the transfer-function (RTF) approach.

Implements the core inference mechanism of "State-Free Inference of
State-Space Models: The Transfer Function Approach" (Bick et al., 2024,
https://arxiv.org/abs/2405.06147). The paper parametrizes a state-space model
by its *rational transfer function* in the frequency domain and shows that the
convolution kernel's spectrum can then be produced with a single pair of FFTs,
giving a sequence-parallel, **state-free** inference path whose memory and
compute are ``O(L)`` and ``O(L log L)`` and do **not** grow with the state
dimension ``n`` -- whereas a recurrent/scan path carries an ``n``-dimensional
state at every step (``O(L * n)`` compute, ``O(n)`` state memory).

Two inference paths are provided for the same underlying system:

* :func:`native_scan_forward` -- the diagonal-SSM recurrence (the scan baseline
  the paper contrasts against; one stable first-order mode per state dimension).
* :func:`state_free_forward` -- the RTF transfer-function path: build the
  impulse response with one forward-FFT pair + element-wise division + one
  inverse FFT (Algorithm 1), then convolve in the frequency domain. No state is
  materialized, so the cost is independent of ``n``.

This is an adapted port (Mode 2): the paper's transfer-function *inference*
machinery is kept at full fidelity, while the full data-dependent
parametrization / input projections are substituted with direct per-channel
rational-transfer-function parameters, and the paper's separate benchmark suite
is replaced by :func:`measure_state_size_scaling`.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn


def _next_fast_len(n: int) -> int:
    """Return the smallest power of two ``>= n`` for FFT-friendly lengths."""
    return 1 << (n - 1).bit_length()


def stable_rtf_system(
    num_channels: int,
    state_size: int,
    max_radius: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a random *stable* SSM in diagonal (pole/residue) form.

    Poles are drawn uniformly in ``[-max_radius, max_radius]`` (strictly inside
    the unit disk), so every first-order mode is stable and decays
    geometrically. Real poles keep the coefficients real.

    Args:
        num_channels: Number of independent SISO channels ``C``.
        state_size: State dimension ``n`` (number of modes / TF order).
        max_radius: Maximum pole magnitude, ``< 1`` for stability.
        generator: Optional torch generator for reproducibility.

    Returns:
        Tuple ``(poles, residues, h_0)`` of shapes ``(C, n)``, ``(C, n)``,
        ``(C,)``. ``poles`` are the per-mode decay rates, ``residues`` the
        per-mode readout weights, and ``h_0`` the feedthrough term.
    """
    poles = (torch.rand(num_channels, state_size, generator=generator) * 2 - 1) * max_radius
    residues = torch.randn(num_channels, state_size, generator=generator)
    h_0 = torch.randn(num_channels, generator=generator)
    return poles, residues, h_0


def rational_coefficients(
    poles: torch.Tensor,
    residues: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a diagonal (pole/residue) SSM to RTF rational coefficients.

    Builds the transfer function

        H(z) = h_0 + (b_1 z^-1 + ... + b_n z^-n) / (1 + a_1 z^-1 + ... + a_n z^-n)

    from the diagonal form. The denominator is ``a(w) = prod_i (1 - lambda_i w)``
    and the numerator is ``b(w) = w * sum_i residue_i * a(w) / (1 - lambda_i w)``,
    which has no z^0 term by construction (matching the paper's Eq. 4).

    Args:
        poles: Per-mode decay rates ``(C, n)``.
        residues: Per-mode readout weights ``(C, n)``.

    Returns:
        Tuple ``(a, b_tilde)`` of denominator and numerator coefficients, each
        of shape ``(C, n)`` (leading ``1`` / ``0`` terms are implicit).
    """
    channels, n = poles.shape
    # Denominator a(w) = prod_i (1 - lambda_i w) via repeated poly mult.
    a_full = poles.new_ones((channels, 1))
    for i in range(n):
        grown = a_full.new_zeros((channels, a_full.shape[1] + 1))
        grown[:, :-1] += a_full
        grown[:, 1:] += a_full * (-poles[:, i : i + 1])
        a_full = grown  # (C, n+1), a_full[:, 0] == 1
    a = a_full[:, 1:].contiguous()

    # Numerator b(w) = w * sum_i residue_i * q^(i)(w), where
    # q^(i)(w) = a(w) / (1 - lambda_i w) with q_0 = 1, q_k = a_k + lambda_i q_{k-1}.
    b_tilde = poles.new_zeros((channels, n))
    for i in range(n):
        lam = poles[:, i : i + 1]  # (C, 1)
        q = poles.new_ones((channels, 1))  # q_0 = 1
        for k in range(1, n):
            q_k = a_full[:, k : k + 1] + lam * q[:, -1:]
            q = torch.cat([q, q_k], dim=-1)  # (C, k+1)
        # b_tilde[:, j] += residue_i * q_j  (the leading w shifts indices by one).
        b_tilde = b_tilde + residues[:, i : i + 1] * q
    return a, b_tilde.contiguous()


def rtf_impulse_response(
    a: torch.Tensor,
    b_tilde: torch.Tensor,
    h_0: torch.Tensor,
    kernel_len: int,
) -> torch.Tensor:
    """Build the SSM impulse response from its rational transfer function.

    Implements Algorithm 1 of the paper: the transfer function is evaluated on
    ``kernel_len`` frequencies by zero-padding the numerator and denominator
    coefficient vectors to length ``kernel_len`` and taking one forward FFT of
    each, an element-wise division, and one inverse FFT. Because every operation
    acts on length-``kernel_len`` vectors regardless of how many of the leading
    ``n`` coefficients are non-zero, the cost is
    ``O(kernel_len log kernel_len)`` and is **independent of the state size n**.

    Args:
        a: Denominator coefficients ``(C, n)`` (the leading ``1`` is implicit).
        b_tilde: Numerator coefficients ``(C, n)`` (the leading ``0`` is
            implicit).
        h_0: Feedthrough term ``(C,)``.
        kernel_len: Number of impulse-response samples ``L`` to produce.

    Returns:
        Real impulse response of shape ``(C, max(kernel_len, n + 1))``.
    """
    channels = a.shape[0]
    n = a.shape[1]
    # An order-n rational function needs at least n + 1 impulse samples; grow the
    # FFT length if a short kernel was requested for a high-order system.
    fft_len = max(kernel_len, n + 1)
    a_bar = a.new_zeros((channels, fft_len))
    b_bar = b_tilde.new_zeros((channels, fft_len))
    a_bar[:, 0] = 1.0
    a_bar[:, 1 : n + 1] = a
    b_bar[:, 1 : n + 1] = b_tilde

    a_spec = torch.fft.fft(a_bar, dim=-1)
    b_spec = torch.fft.fft(b_bar, dim=-1)
    h_spec = b_spec / a_spec + h_0.unsqueeze(-1)
    return torch.fft.ifft(h_spec, dim=-1).real


def state_free_forward(
    u: torch.Tensor,
    a: torch.Tensor,
    b_tilde: torch.Tensor,
    h_0: torch.Tensor,
    kernel_len: Optional[int] = None,
) -> torch.Tensor:
    """State-free SSM inference via frequency-domain (FFT) convolution.

    The output is computed as a linear convolution of the input with the
    transfer-function impulse response, evaluated entirely in the frequency
    domain. No ``n``-dimensional state is ever materialized, so peak activation
    memory and compute are ``O(L log L)`` and independent of the state size.

    Args:
        u: Input sequence ``(B, C, L)``.
        a: Denominator coefficients ``(C, n)``.
        b_tilde: Numerator coefficients ``(C, n)``.
        h_0: Feedthrough term ``(C,)``.
        kernel_len: Impulse-response length to truncate to. Defaults to ``L``.

    Returns:
        Output sequence ``(B, C, L)``.
    """
    _, _, seq_len = u.shape
    n = a.shape[1]
    if kernel_len is None:
        # Need at least n + 1 samples to represent an order-n transfer function.
        kernel_len = max(seq_len, n + 1)
    kernel = rtf_impulse_response(a, b_tilde, h_0, kernel_len)

    fft_len = _next_fast_len(seq_len + kernel_len - 1)
    u_spec = torch.fft.fft(u, n=fft_len, dim=-1)
    k_spec = torch.fft.fft(kernel, n=fft_len, dim=-1)
    y_spec = u_spec * k_spec.unsqueeze(0)
    return torch.fft.ifft(y_spec, dim=-1).real[..., :seq_len]


def native_scan_forward(
    u: torch.Tensor,
    poles: torch.Tensor,
    residues: torch.Tensor,
    h_0: torch.Tensor,
) -> torch.Tensor:
    """Recurrent (stateful) diagonal-SSM inference -- the scan baseline.

    Runs the diagonal recurrence ``x_i,t = lambda_i x_i,t-1 + u_t`` for each of
    the ``n`` modes and reads out ``y_t = h_0 u_t + sum_i residue_i x_i,t-1``. An
    ``n``-dimensional state is carried and updated at every one of the ``L``
    steps, so this path is the paper's ``O(L * n)``-compute / ``O(n)``-state
    contrast to the state-free path. Each mode is a stable first-order
    recursion, so the baseline is numerically stable.

    Args:
        u: Input sequence ``(B, C, L)``.
        poles: Per-mode decay rates ``(C, n)``.
        residues: Per-mode readout weights ``(C, n)``.
        h_0: Feedthrough term ``(C,)``.

    Returns:
        Output sequence ``(B, C, L)``.
    """
    batch, channels, seq_len = u.shape
    n = poles.shape[1]
    # state[..., i] holds x_i,t-1; initialized to zero (x_i,-1 = 0).
    state = u.new_zeros((batch, channels, n))
    outputs = u.new_zeros((batch, channels, seq_len))
    for t in range(seq_len):
        u_t = u[:, :, t]
        outputs[:, :, t] = h_0 * u_t + torch.einsum("cn,bcn->bc", residues, state)
        state = poles * state + u_t.unsqueeze(-1)
    return outputs


class StateFreeMixer(nn.Module):
    """A transfer-function (state-free) SSM sequence mixer.

    Presents the ``(B, L, d_model)`` contract used elsewhere in the package (see
    :class:`discrete_mamba2.DiscreteMamba2`), treating each of the ``d_model``
    dimensions as an independent SISO system parametrized by its rational
    transfer function and inferred state-free via FFT convolution. The
    parameters are the transfer-function coefficients directly (the paper's
    parametrization), so inference cost is ``O(L log L)`` regardless of the
    state size -- the state dimension governs expressivity, not inference cost.

    Parameters are initialized from stable poles so the mixer is stable by
    default. Data-dependent input/output projections and cross-channel mixing
    are intentionally omitted -- the deliverable is the *inference kernel*, not a
    trained model.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int = 64,
        max_radius: float = 0.5,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the mixer with stable per-channel transfer functions.

        Args:
            d_model: Input/output dimension (number of SISO channels).
            state_size: Order ``n`` of each channel's rational transfer function.
            max_radius: Maximum pole magnitude for stable initialization.
            device: Device for parameters.
            dtype: Dtype for parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        poles, residues, h_0 = stable_rtf_system(d_model, state_size, max_radius=max_radius)
        a, b_tilde = rational_coefficients(poles, residues)
        self.a = nn.Parameter(a.to(device=device, dtype=dtype))
        self.b_tilde = nn.Parameter(b_tilde.to(device=device, dtype=dtype))
        self.h_0 = nn.Parameter(h_0.to(device=device, dtype=dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Run state-free inference over the sequence.

        Args:
            u: Input ``(B, L, d_model)``.

        Returns:
            Output ``(B, L, d_model)``.
        """
        x = u.transpose(1, 2)
        y = state_free_forward(x, self.a, self.b_tilde, self.h_0)
        return y.transpose(1, 2)


def _time(fn, repeats: int) -> float:
    """Return the median wall-clock time (ms) of ``repeats`` runs of ``fn``."""
    fn()  # warm up so lazy initialization does not skew the measurement
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _well_conditioned_rational(
    num_channels: int,
    state_size: int,
    scale: float = 0.01,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample well-conditioned transfer-function coefficients directly.

    RTF parametrizes the rational coefficients ``a, b_tilde`` directly (they are
    the trained parameters), rather than deriving them from poles. Small-magnitude
    coefficients keep ``1 + a_1 z^-1 + ...`` bounded away from zero on the unit
    circle, so the FFT division in :func:`rtf_impulse_response` stays
    well-conditioned at any state size. Used by the benchmark, which measures
    *cost* (correctness is covered separately by the diagonal baseline).

    Args:
        num_channels: Number of SISO channels ``C``.
        state_size: State dimension ``n`` (TF order).
        scale: Standard deviation of the random coefficients.
        generator: Optional torch generator for reproducibility.

    Returns:
        Tuple ``(a, b_tilde, h_0)`` of shapes ``(C, n)``, ``(C, n)``, ``(C,)``.
    """
    a = torch.randn(num_channels, state_size, generator=generator) * scale
    b_tilde = torch.randn(num_channels, state_size, generator=generator) * scale
    h_0 = torch.randn(num_channels, generator=generator) * scale
    return a, b_tilde, h_0


def measure_state_size_scaling(
    seq_len: int = 4096,
    state_sizes=(16, 64, 256, 1024),
    num_channels: int = 64,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float64,
    repeats: int = 3,
) -> list[dict]:
    """Benchmark state-free vs. native-scan inference across state sizes.

    For each state size ``n`` the two paths are timed on an ``n``-dimensional
    system. The benchmark measures *cost*, not correctness (the two paths use
    independently drawn systems of the same dimension): the native scan uses a
    stable diagonal system, the state-free path uses directly-parametrized RTF
    coefficients. The result makes the paper's central claim concrete -- the
    state-free time is roughly flat in ``n`` (cost lives in the ``L``-length
    FFTs), while the native scan grows with ``n`` (it carries and updates an
    ``n``-dim state at every step). Deterministic FLOP estimates are included so
    the scaling is checkable independent of wall-clock noise.

    Args:
        seq_len: Sequence length ``L``.
        state_sizes: State dimensions ``n`` to sweep.
        num_channels: Number of SISO channels ``C``.
        batch_size: Batch size ``B``.
        dtype: Compute dtype (float64 keeps the FFT division well-conditioned).
        repeats: Timed repetitions per measurement (median is reported).

    Returns:
        List of per-``n`` result dicts: ``state_size``, ``native_ms``,
        ``state_free_ms``, ``native_state_elements`` (``= n``),
        ``state_free_state_elements`` (``= 0``), ``native_flops``
        (``= 2 * L * B * C * n``), and ``state_free_flops`` (independent of ``n``
        while ``n <= L``).
    """
    results = []
    for n in state_sizes:
        poles, residues, native_h0 = stable_rtf_system(num_channels, n)
        poles = poles.to(dtype=dtype)
        residues = residues.to(dtype=dtype)
        native_h0 = native_h0.to(dtype=dtype)
        a, b_tilde, free_h0 = _well_conditioned_rational(num_channels, n)
        a = a.to(dtype=dtype)
        b_tilde = b_tilde.to(dtype=dtype)
        free_h0 = free_h0.to(dtype=dtype)
        u = torch.randn(batch_size, num_channels, seq_len, dtype=dtype)
        native_ms = _time(lambda: native_scan_forward(u, poles, residues, native_h0), repeats)
        state_free_ms = _time(lambda: state_free_forward(u, a, b_tilde, free_h0), repeats)
        # Native: two C*n MAC einsums per step over L steps. State-free: a small
        # constant number of length-(~2L) FFTs over the (B, C) grid.
        native_flops = 2 * seq_len * batch_size * num_channels * n
        fft_len = _next_fast_len(2 * seq_len)
        state_free_flops = (
            batch_size * num_channels * 6 * fft_len * max(1, fft_len.bit_length() - 1)
        )
        results.append(
            {
                "state_size": n,
                "native_ms": native_ms,
                "state_free_ms": state_free_ms,
                "native_state_elements": n,
                "state_free_state_elements": 0,
                "native_flops": native_flops,
                "state_free_flops": state_free_flops,
            }
        )
    return results
