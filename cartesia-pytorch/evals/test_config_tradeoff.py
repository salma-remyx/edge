"""Tests for ``evals.config_tradeoff``.

The analysis tests (sweep enumeration, Pareto frontier, main/interaction
effects, end-to-end sweep with an injected measure function, report) are pure
and run anywhere. The integration test asserts that ``live_measure_fn`` delegates
model loading to the existing ``evals.generation`` module; it requires the CUDA
toolchain and is skipped when ``torch`` is unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from evals.config_tradeoff import (
    Config,
    Record,
    SweepResult,
    build_report,
    enumerate_configs,
    interaction_effects,
    live_measure_fn,
    main,
    main_effects,
    pareto_frontier,
    rank_axes,
    run_sweep,
)


def _rec(model, cg, dtype, latency, energy=None, accuracy=None):
    """Build a Record with a fixed prompt/gen length for brevity."""
    return Record(
        config=Config(model=model, cg=cg, dtype=dtype, promptlen=10, genlen=10),
        latency_ms=latency,
        energy_j=energy,
        accuracy=accuracy,
    )


def test_enumerate_configs_is_cartesian_product():
    """The grid expands into the full cross product of axis levels."""
    grid = {"model": ["A", "B"], "cg": [True, False], "dtype": ["bfloat16"]}
    configs = enumerate_configs(grid)
    assert len(configs) == 4
    keys = {(c.model, c.cg, c.dtype) for c in configs}
    assert keys == {
        ("A", True, "bfloat16"),
        ("A", False, "bfloat16"),
        ("B", True, "bfloat16"),
        ("B", False, "bfloat16"),
    }


def test_pareto_frontier_drops_dominated_on_latency_and_energy():
    """A record worse on both objectives is excluded from the frontier."""
    records = [
        _rec("A", True, "bfloat16", latency=10, energy=5),
        _rec("A", False, "bfloat16", latency=20, energy=8),
        _rec("B", True, "float16", latency=12, energy=4),
    ]
    frontier = pareto_frontier(records, ["latency_ms", "energy_j"])
    assert {(r.config.model, r.config.cg) for r in frontier} == {("A", True), ("B", True)}


def test_pareto_frontier_handles_accuracy_dimension():
    """The 3-D frontier trades accuracy against latency and energy."""
    records = [
        _rec("A", True, "bfloat16", latency=10, energy=5, accuracy=0.90),
        _rec("B", True, "float16", latency=12, energy=4, accuracy=0.95),
        _rec("C", True, "float16", latency=20, energy=8, accuracy=0.80),
    ]
    frontier = pareto_frontier(records, ["latency_ms", "energy_j", "accuracy"])
    assert {r.config.model for r in frontier} == {"A", "B"}


def test_records_missing_an_objective_are_excluded_from_frontier():
    """A record without accuracy cannot compete on the 3-D objective set."""
    records = [
        _rec("A", True, "bfloat16", latency=10, energy=5, accuracy=0.90),
        _rec("B", True, "float16", latency=12, energy=4, accuracy=None),
    ]
    frontier = pareto_frontier(records, ["latency_ms", "energy_j", "accuracy"])
    assert {r.config.model for r in frontier} == {"A"}


def test_main_effects_group_by_axis_level():
    """main_effects averages the metric within each level of an axis."""
    records = [
        _rec("A", True, "bfloat16", latency=10),
        _rec("A", False, "bfloat16", latency=20),
        _rec("B", True, "bfloat16", latency=40),
    ]
    assert main_effects(records, "model", "latency_ms") == {"A": 15.0, "B": 40.0}


def test_rank_axes_puts_largest_spread_first():
    """The axis whose levels move the metric most ranks first."""
    records = [
        _rec("A", True, "bfloat16", latency=10),
        _rec("B", True, "bfloat16", latency=100),
        _rec("A", True, "float16", latency=11),
        _rec("B", True, "float16", latency=101),
    ]
    ranked = rank_axes(records, "latency_ms")
    assert ranked[0][0] == "model"
    assert ranked[0][1] > ranked[1][1]


def test_interaction_effects_pairs_two_axes():
    """interaction_effects reports the mean per (axis_a, axis_b) pair."""
    records = [
        _rec("A", True, "bfloat16", latency=10),
        _rec("A", False, "bfloat16", latency=20),
        _rec("B", True, "bfloat16", latency=40),
        _rec("B", False, "bfloat16", latency=60),
    ]
    inter = interaction_effects(records, "model", "cg", "latency_ms")
    assert inter[("A", True)] == 10.0
    assert inter[("B", False)] == 60.0


def test_run_sweep_uses_injected_measure_fn_and_builds_frontier():
    """run_sweep drives an injected measure function and derives objectives."""

    def fake_measure(config, power_w=None, accuracy_provider=None):
        base = {"A": 10.0, "B": 50.0}[config.model]
        latency = base + (5.0 if not config.cg else 0.0)
        energy = power_w * latency / 1000.0 if power_w else None
        accuracy = accuracy_provider(config) if accuracy_provider else None
        return Record(config=config, latency_ms=latency, energy_j=energy, accuracy=accuracy)

    grid = {"model": ["A", "B"], "cg": [True, False]}
    result = run_sweep(grid, fake_measure, power_w=200.0)
    assert len(result.records) == 4
    assert "latency_ms" in result.objectives
    assert "energy_j" in result.objectives
    # model A with cg=True is fastest and lowest-energy -> on the frontier.
    assert ("A", True) in {(r.config.model, r.config.cg) for r in result.frontier}
    report = build_report(result)
    assert "Pareto frontier" in report
    assert "energy_j" in report
    assert "model" in report  # top driver axis is named


def test_build_report_with_accuracy_objective_names_driver():
    """When accuracy is present it becomes an objective in the report."""
    records = [
        _rec("A", True, "bfloat16", latency=10, energy=5, accuracy=0.9),
        _rec("B", True, "bfloat16", latency=100, energy=50, accuracy=0.5),
    ]
    result = SweepResult(
        records=records,
        objectives=["latency_ms", "energy_j", "accuracy"],
        frontier=pareto_frontier(records, ["latency_ms", "energy_j", "accuracy"]),
    )
    report = build_report(result)
    assert "accuracy" in report


def test_main_dry_grid_prints_enumerated_configs(capsys):
    """--dry_grid enumerates configs without touching any model loader."""
    main(["--models", "A,B", "--cg", "on,off", "--dtype", "bfloat16", "--dry_grid"])
    out = capsys.readouterr().out
    assert "model='A'" in out and "model='B'" in out
    assert "cg=True" in out and "cg=False" in out


def test_live_measure_fn_delegates_to_evals_generation(monkeypatch):
    """live_measure_fn must load the model via evals.generation.choose_model.

    Requires the CUDA toolchain (torch + cartesia_pytorch); skipped otherwise.
    """
    pytest.importorskip("torch")
    try:
        import evals.generation as generation
    except Exception as exc:  # pragma: no cover - depends on the full CUDA toolchain
        pytest.skip(f"evals.generation unavailable: {exc}")

    calls: dict[str, object] = {}

    class FakeModel:
        """Minimal stand-in for a loaded LM head model."""

        def to(self, *args, **kwargs):
            """No-op placement/dtype cast."""
            return self

        def eval(self):
            """No-op eval mode."""
            return self

        def generate(self, **kwargs):
            """Record the cg setting and return a sequences-bearing object."""
            calls["generate_kwargs"] = kwargs
            return SimpleNamespace(sequences=None)

    def fake_choose_model(args):
        """Pretend to load, recording which model was requested."""
        calls["model_arg"] = args.model
        return FakeModel(), SimpleNamespace(eos_token_id=0)

    monkeypatch.setattr(generation, "choose_model", fake_choose_model)
    config = Config(model="Llamba-1B", cg=False, dtype="bfloat16", promptlen=8, genlen=4)
    record = live_measure_fn(config, power_w=300.0)
    assert calls["model_arg"] == "Llamba-1B"
    assert calls["generate_kwargs"]["cg"] is False
    assert isinstance(record, Record)
    assert record.latency_ms >= 0.0
    assert record.energy_j is not None  # power_w supplied -> energy measured
