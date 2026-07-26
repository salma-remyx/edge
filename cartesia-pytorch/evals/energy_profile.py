"""Phase-split energy profiling for small-language-model inference.

The dominant on-device inference cost is *decoding* (output tokens), not
*prefill* (input tokens): average inference power is approximately a
model-intrinsic constant, so energy scales linearly with wall-clock time, and
each decode token costs an order of magnitude more time than each prefill
token. ``time_bench`` historically collapsed prompt processing and decoding
into a single number; this module reports the two phases separately and
estimates per-phase energy under a constant-power model (``energy = power *
time``).

Adapted from "Seeing is Free, Speaking is Not: Uncovering the True Energy
Bottleneck in Edge VLM Inference" (arXiv:2607.09520). The paper's hardware
power meter is substituted with a configurable average-power estimate -- a
faithful proxy because the paper's own first finding is that power is a
model-intrinsic constant (under 5% variation across inputs). The paper's
multi-model / multi-resolution benchmark sweep is out of scope here and left
to downstream evaluation.
"""

from dataclasses import dataclass

# Energy in millijoules under a constant-power model: 1 W * 1 ms = 1 mJ.
_WATT_MS_TO_MILLIJOULE = 1.0
_MICROSECONDS_PER_MILLISECOND = 1000.0


@dataclass(frozen=True)
class PhaseEnergyReport:
    """Latency and energy breakdown of one generation, split by phase.

    Attributes:
        prompt_len: Number of input (prompt) tokens.
        gen_len: Number of generated (output) tokens.
        total_ms: Total generation wall-clock (prefill + decode), in milliseconds.
        prefill_ms: Prompt-processing time in ms, or None if unmeasured.
        decode_ms: Decode time in ms, or None if prefill was not measured.
        power_w: Average inference power in watts used for the energy estimate.
        prefill_us_per_tok: Amortized prefill cost per prompt token, in microseconds.
        decode_us_per_tok: Per-token decode cost, in microseconds.
        decode_prefill_ratio: Per-token decode cost divided by per-token prefill cost.
        prefill_energy_mj: Estimated prefill energy in millijoules.
        decode_energy_mj: Estimated decode energy in millijoules.
        decode_energy_share: Fraction of total energy attributable to decode.
    """

    prompt_len: int
    gen_len: int
    total_ms: float
    prefill_ms: float | None
    decode_ms: float | None
    power_w: float | None
    prefill_us_per_tok: float | None
    decode_us_per_tok: float | None
    decode_prefill_ratio: float | None
    prefill_energy_mj: float | None
    decode_energy_mj: float | None
    decode_energy_share: float | None

    def format(self):
        """Render the report as a human-readable multi-line string.

        Returns:
            The formatted report. Only phase and energy lines that could be
            computed from the supplied measurements are included.
        """
        lines = [f"total time: {self.total_ms:.0f}ms"]
        if self.prefill_ms is not None and self.decode_ms is not None:
            lines.append(
                f"prefill (prompt processing): {self.prefill_ms:.1f}ms "
                f"[{self.prefill_us_per_tok:.1f} us/tok over {self.prompt_len} toks]"
            )
            lines.append(
                f"decode (generation):         {self.decode_ms:.1f}ms "
                f"[{self.decode_us_per_tok:.1f} us/tok over {self.gen_len} toks]"
            )
            if self.decode_prefill_ratio is not None:
                lines.append(f"per-token decode/prefill cost: {self.decode_prefill_ratio:.1f}x")
        if self.prefill_energy_mj is not None and self.decode_energy_mj is not None:
            lines.append(
                f"estimated energy @ {self.power_w:g} W -> "
                f"prefill {self.prefill_energy_mj:.0f} mJ, "
                f"decode {self.decode_energy_mj:.0f} mJ"
            )
            if self.decode_energy_share is not None:
                lines.append(f"decode energy share: {self.decode_energy_share * 100:.0f}%")
        return "\n".join(lines)


def report_phases(
    *,
    total_ms,
    prompt_len,
    gen_len,
    prefill_ms=None,
    decode_ms=None,
    power_w=None,
):
    """Build a ``PhaseEnergyReport`` from measured phase timings.

    Args:
        total_ms: Total generation wall-clock (prefill + decode), in milliseconds.
        prompt_len: Number of input (prompt) tokens.
        gen_len: Number of generated (output) tokens.
        prefill_ms: Prompt-processing time in ms. When None, phases cannot be
            split and only the total is reported.
        decode_ms: Decode time in ms. Derived as ``total_ms - prefill_ms`` when
            omitted. Clamped at zero so a noisy measurement cannot go negative.
        power_w: Average inference power in watts. When None, the constant-power
            energy estimate is skipped (latency phases are still reported).

    Returns:
        A ``PhaseEnergyReport`` describing the per-phase latency and (when
        ``power_w`` is given) per-phase energy.
    """
    if prefill_ms is not None:
        if decode_ms is None:
            decode_ms = max(total_ms - prefill_ms, 0.0)
        prefill_us_per_tok = prefill_ms * _MICROSECONDS_PER_MILLISECOND / max(prompt_len, 1)
        decode_us_per_tok = decode_ms * _MICROSECONDS_PER_MILLISECOND / max(gen_len, 1)
        decode_prefill_ratio = (
            decode_us_per_tok / prefill_us_per_tok if prefill_us_per_tok > 0 else None
        )
    else:
        decode_ms = None
        prefill_us_per_tok = None
        decode_us_per_tok = None
        decode_prefill_ratio = None

    if power_w is not None and prefill_ms is not None and decode_ms is not None:
        prefill_energy_mj = power_w * prefill_ms * _WATT_MS_TO_MILLIJOULE
        decode_energy_mj = power_w * decode_ms * _WATT_MS_TO_MILLIJOULE
        total_energy_mj = prefill_energy_mj + decode_energy_mj
        decode_energy_share = decode_energy_mj / total_energy_mj if total_energy_mj > 0 else None
    else:
        prefill_energy_mj = None
        decode_energy_mj = None
        decode_energy_share = None

    return PhaseEnergyReport(
        prompt_len=prompt_len,
        gen_len=gen_len,
        total_ms=total_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        power_w=power_w,
        prefill_us_per_tok=prefill_us_per_tok,
        decode_us_per_tok=decode_us_per_tok,
        decode_prefill_ratio=decode_prefill_ratio,
        prefill_energy_mj=prefill_energy_mj,
        decode_energy_mj=decode_energy_mj,
        decode_energy_share=decode_energy_share,
    )
