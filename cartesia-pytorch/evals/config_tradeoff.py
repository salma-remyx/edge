"""Configuration trade-off analysis for SSM inference.

Adapted from "Attention to Detail: Evaluating Energy, Performance, and
Accuracy Trade-offs Across vLLM Configurations" (arXiv:2607.09172). That study
sweeps an inference engine's configuration axes and measures energy x latency x
accuracy across them to find Pareto-optimal settings, reporting that the effects
are model- and workload-dependent and that no configuration is universally
optimal.

This module keeps that trade-off-analysis *methodology* and substitutes the
axes this repo actually exposes for its Mamba/SSM models -- CUDA-graph on/off,
dtype, prompt length, generation length, and model choice -- for vLLM's
attention-kernel / prefix-cache / chunked-prefil axes. Model loading reuses
``evals.generation.choose_model``. Energy is ``power_w * latency`` when a device
power figure is supplied, mirroring the repo's ``--power_w`` convention;
otherwise the frontier is computed over the available dimensions. Accuracy is a
pluggable dimension (supply an ``accuracy_provider``); the default live measure
returns ``None`` for it, since wiring the full lm-evaluation-harness task suite
(see ``evals.cartesia_lm_eval``) is a separate, larger effort.

Run from the ``cartesia-pytorch`` directory::

    python -m evals.config_tradeoff --models Llamba-1B,Llamba-8B \
        --cg on,off --dtype bfloat16,float16 --power_w 350

``--dry_grid`` enumerates the sweep without loading any model.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace

# Heavy dependencies (torch, cartesia_pytorch, transformers) are imported lazily
# inside ``live_measure_fn`` so the analysis functions stay importable and
# testable without a CUDA toolchain.

#: Inference knobs swept by this analysis.
AXES = ("model", "cg", "dtype", "promptlen", "genlen")
#: Default model when a grid omits the ``model`` axis.
DEFAULT_MODEL = "Llamba-1B"
#: Metric -> optimization direction ("min" = smaller is better).
OBJECTIVE_DIRECTION = {"latency_ms": "min", "energy_j": "min", "accuracy": "max"}


@dataclass(frozen=True)
class Config:
    """A single inference configuration point in the sweep grid."""

    model: str
    cg: bool
    dtype: str
    promptlen: int
    genlen: int

    def axis_values(self) -> dict[str, object]:
        """Return the per-axis values keyed by axis name."""
        return {
            "model": self.model,
            "cg": self.cg,
            "dtype": self.dtype,
            "promptlen": self.promptlen,
            "genlen": self.genlen,
        }


@dataclass(frozen=True)
class Record:
    """A measured (config, metrics) row produced by the sweep."""

    config: Config
    latency_ms: float
    energy_j: float | None = None
    accuracy: float | None = None


@dataclass
class SweepResult:
    """Outcome of a configuration sweep: records, objectives, frontier."""

    records: list[Record]
    objectives: list[str]
    frontier: list[Record] = field(default_factory=list)


def enumerate_configs(grid: dict[str, list]) -> list[Config]:
    """Cartesian-product a ``{axis: [levels]}`` grid into ``Config`` points."""
    models = grid.get("model", [DEFAULT_MODEL])
    cgs = grid.get("cg", [True])
    dtypes = grid.get("dtype", ["bfloat16"])
    promptlens = grid.get("promptlen", [100])
    genlens = grid.get("genlen", [100])
    points = []
    for model, cg, dtype, promptlen, genlen in product(models, cgs, dtypes, promptlens, genlens):
        points.append(
            Config(
                model=model,
                cg=bool(cg),
                dtype=dtype,
                promptlen=int(promptlen),
                genlen=int(genlen),
            )
        )
    return points


def _sign(metric: str) -> int:
    """Return +1 for minimize objectives, -1 for maximize objectives."""
    return -1 if OBJECTIVE_DIRECTION.get(metric, "min") == "max" else 1


def _dominates(a: Record, b: Record, objectives: list[str]) -> bool:
    """Return True if record ``a`` Pareto-dominates record ``b``."""
    better_everywhere = all(
        _sign(m) * getattr(a, m) <= _sign(m) * getattr(b, m) for m in objectives
    )
    better_somewhere = any(_sign(m) * getattr(a, m) < _sign(m) * getattr(b, m) for m in objectives)
    return better_everywhere and better_somewhere


def pareto_frontier(records: list[Record], objectives: list[str]) -> list[Record]:
    """Return the non-dominated records over ``objectives``.

    Records missing any selected objective cannot be compared on the full
    objective set and are excluded from the frontier.
    """
    active = [r for r in records if all(getattr(r, m) is not None for m in objectives)]
    frontier = []
    for record in active:
        if not any(
            other is not record and _dominates(other, record, objectives) for other in active
        ):
            frontier.append(record)
    return frontier


def main_effects(records: list[Record], axis: str, metric: str) -> dict[object, float]:
    """Mean ``metric`` grouped by the levels of ``axis``."""
    buckets: dict[object, list[float]] = {}
    for record in records:
        value = getattr(record, metric)
        if value is None:
            continue
        buckets.setdefault(record.config.axis_values()[axis], []).append(float(value))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items()}


def axis_effect_spread(records: list[Record], axis: str, metric: str) -> float:
    """Range (max - min) of per-level mean ``metric`` across one ``axis``."""
    means = list(main_effects(records, axis, metric).values())
    if len(means) < 2:
        return 0.0
    return max(means) - min(means)


def rank_axes(
    records: list[Record], metric: str, axes: tuple[str, ...] = AXES
) -> list[tuple[str, float]]:
    """Rank ``axes`` by how much they move ``metric`` (largest spread first)."""
    scored = [(axis, axis_effect_spread(records, axis, metric)) for axis in axes]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def interaction_effects(
    records: list[Record], axis_a: str, axis_b: str, metric: str
) -> dict[tuple[object, object], float]:
    """Mean ``metric`` for each ``(axis_a, axis_b)`` level pair."""
    buckets: dict[tuple[object, object], list[float]] = {}
    for record in records:
        value = getattr(record, metric)
        if value is None:
            continue
        values = record.config.axis_values()
        key = (values[axis_a], values[axis_b])
        buckets.setdefault(key, []).append(float(value))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items()}


def run_sweep(
    grid: dict[str, list],
    measure_fn: Callable[..., Record],
    power_w: float | None = None,
    accuracy_provider: Callable[[Config], float | None] | None = None,
) -> SweepResult:
    """Enumerate configs, measure each, and compute the Pareto frontier.

    ``measure_fn`` has the signature ``measure_fn(config, power_w=...,
    accuracy_provider=...) -> Record``. Objectives are latency plus energy when a
    device power figure is supplied, plus accuracy when any record carries one.
    """
    records = [
        measure_fn(config, power_w=power_w, accuracy_provider=accuracy_provider)
        for config in enumerate_configs(grid)
    ]
    objectives = ["latency_ms"]
    if power_w:
        objectives.append("energy_j")
    if any(record.accuracy is not None for record in records):
        objectives.append("accuracy")
    return SweepResult(
        records=records, objectives=objectives, frontier=pareto_frontier(records, objectives)
    )


def _best(records: list[Record], metric: str) -> Record | None:
    """Return the record with the best ``metric`` value, or ``None``."""
    scored = [record for record in records if getattr(record, metric) is not None]
    if not scored:
        return None
    return min(scored, key=lambda r: _sign(metric) * getattr(r, metric))


def _format_record(record: Record) -> str:
    """Render a record as a compact one-line summary."""
    cfg = record.config
    parts = [
        f"model={cfg.model}",
        f"cg={cfg.cg}",
        f"dtype={cfg.dtype}",
        f"latency={record.latency_ms:.3g}ms",
    ]
    if record.energy_j is not None:
        parts.append(f"energy={record.energy_j:.3g}J")
    if record.accuracy is not None:
        parts.append(f"acc={record.accuracy:.3g}")
    return "  ".join(parts)


def build_report(result: SweepResult, axes: tuple[str, ...] = AXES) -> str:
    """Render a human-readable energy/performance/accuracy trade-off report."""
    lines = [f"# Config trade-off sweep ({len(result.records)} configs measured)"]
    lines.append(f"objectives: {', '.join(result.objectives)}")
    for metric in result.objectives:
        ranked = rank_axes(result.records, metric, axes)
        if ranked and ranked[0][1] > 0:
            top_axis, spread = ranked[0]
            lines.append(f"- {metric} mainly driven by '{top_axis}' (spread {spread:.3g})")
    lines.append("")
    lines.append(f"Pareto frontier ({len(result.frontier)} configs):")
    for record in result.frontier:
        lines.append("  " + _format_record(record))
    lines.append("")
    for metric in result.objectives:
        best = _best(result.records, metric)
        if best is not None:
            lines.append(f"best {metric}: {_format_record(best)}")
    return "\n".join(lines)


def live_measure_fn(
    config: Config,
    power_w: float | None = None,
    accuracy_provider: Callable[[Config], float | None] | None = None,
) -> Record:
    """Measure a config on a real model (requires torch + cartesia_pytorch).

    Loads the model via ``evals.generation.choose_model`` and times one
    generation pass at the config's CUDA-graph / dtype settings. Energy is
    ``power_w * latency`` when a power figure is supplied.
    """
    import torch  # noqa: PLC0415  -- lazy so the module imports without CUDA
    from evals.generation import choose_model  # noqa: PLC0415

    args = SimpleNamespace(
        model=config.model,
        promptlen=config.promptlen,
        genlen=config.genlen,
        repeats=1,
        seed=123,
        dtype=config.dtype,
    )
    model, tokenizer = choose_model(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    model.to(dtype=getattr(torch, config.dtype))
    model.eval()
    input_ids = torch.randint(1, 1000, (1, config.promptlen), dtype=torch.long, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(
        input_ids=input_ids,
        max_length=input_ids.shape[1] + config.genlen,
        cg=config.cg,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    energy_j = power_w * (latency_ms / 1000.0) if power_w else None
    accuracy = accuracy_provider(config) if accuracy_provider else None
    return Record(config=config, latency_ms=latency_ms, energy_j=energy_j, accuracy=accuracy)


def _bool_levels(raw: str) -> list[bool]:
    """Parse a comma list of on/off-ish tokens into booleans."""
    truthy = {"on", "true", "1", "yes", "y"}
    return [token.strip().lower() in truthy for token in raw.split(",") if token.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the config trade-off sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", type=str, default=DEFAULT_MODEL, help="Comma-separated model names."
    )
    parser.add_argument("--cg", type=str, default="on,off", help="Comma-separated on/off levels.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Comma-separated dtypes.")
    parser.add_argument(
        "--promptlen", type=str, default="100", help="Comma-separated prompt lengths."
    )
    parser.add_argument(
        "--genlen", type=str, default="100", help="Comma-separated generation lengths."
    )
    parser.add_argument(
        "--power_w", type=float, default=None, help="Device power in watts for the energy axis."
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--dry_grid",
        action="store_true",
        help="Print the enumerated configs without loading any model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build the grid, sweep, and print the trade-off report."""
    args = parse_args(argv)
    grid = {
        "model": [m.strip() for m in args.models.split(",") if m.strip()],
        "cg": _bool_levels(args.cg),
        "dtype": [d.strip() for d in args.dtype.split(",") if d.strip()],
        "promptlen": [int(x) for x in args.promptlen.split(",") if x.strip()],
        "genlen": [int(x) for x in args.genlen.split(",") if x.strip()],
    }
    if args.dry_grid:
        for config in enumerate_configs(grid):
            print(config)
        return
    result = run_sweep(grid, live_measure_fn, power_w=args.power_w)
    print(build_report(result))


if __name__ == "__main__":
    main()
