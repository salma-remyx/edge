# Copyright (c) 2024, Aviv Bick, Kevin Li.

"""Cross-backend inference-latency benchmark for Edge SSM models.

Adapted from "Benchmarking Edge Inference Strategies for Deep Learning
Models in Industrial Machine Vision" (arxiv:2607.11356v1), which compares
plain PyTorch, ONNX Runtime, OpenVINO and TensorRT on inference time across
CPU/GPU hardware and reports the fastest strategy per device. This module
ports that methodology -- warmup plus repeated timings and a
backend-versus-backend comparison table that names the fastest strategy --
to this repo's own multi-backend identity (PyTorch/MLX/Metal).

The timing and reporting core is backend-agnostic and dependency-free, so it
runs anywhere. The PyTorch binding drives the same ``model.generate(...)``
interface already used by ``evals/generation.py``; the CUDA-graph (``cg``)
flag, ``return_dict_in_generate`` and sampling kwargs are forwarded
identically, so the timed work matches the existing generation benchmark.
Other backends (an MLX or Metal runner) plug in by supplying a ``step()``
callable -- they are not stubbed here because this environment cannot run
them, and the protocol is the deliverable rather than any one backend.
"""

import argparse
import statistics
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Protocol


@dataclass
class LatencyStats:
    """Latency measurements for a single backend runner.

    Attributes:
        name: Backend or strategy name, e.g. ``"pytorch-eager"``.
        repeats: Number of timed iterations recorded.
        warmup: Number of untimed warmup iterations.
        samples_ms: Per-iteration wall-clock latencies in milliseconds.
        units: Work units produced per iteration (e.g. generated tokens).
    """

    name: str
    repeats: int
    warmup: int
    samples_ms: list = field(default_factory=list)
    units: int = 1

    @property
    def mean_ms(self):
        """Mean latency per iteration in milliseconds."""
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self):
        """Median latency per iteration in milliseconds."""
        return statistics.median(self.samples_ms)

    @property
    def min_ms(self):
        """Minimum observed latency per iteration in milliseconds."""
        return min(self.samples_ms) if self.samples_ms else 0.0

    @property
    def std_ms(self):
        """Population standard deviation of the per-iteration latencies."""
        return statistics.pstdev(self.samples_ms) if len(self.samples_ms) > 1 else 0.0

    @property
    def ms_per_unit(self):
        """Mean latency per work unit (e.g. milliseconds per token)."""
        return self.mean_ms / self.units if self.units else self.mean_ms


class BackendRunner(Protocol):
    """A single inference backend or strategy to be benchmarked."""

    name: str

    def prepare(self) -> None:
        """Set up the backend before timing (eval mode, compile, device move)."""

    def step(self) -> object:
        """Run one inference unit and return its output."""


@dataclass
class CallableRunner:
    """Wrap any callables as a backend runner.

    Attributes:
        name: Backend or strategy name.
        step_fn: Zero-argument callable performing one inference unit.
        prepare_fn: Optional setup callable run once before timing.
        sync_fn: Optional callable to flush async work after each step.
        units: Work units produced per step, for ms/token-style metrics.
    """

    name: str
    step_fn: Callable[[], object]
    prepare_fn: Optional[Callable[[], None]] = None
    sync_fn: Optional[Callable[[], None]] = None
    units: int = 1

    def prepare(self) -> None:
        """Run the optional setup callable."""
        if self.prepare_fn is not None:
            self.prepare_fn()

    def step(self) -> object:
        """Run one inference unit via ``step_fn``."""
        return self.step_fn()


def bench_callable(
    name,
    step_fn,
    *,
    warmup=3,
    repeats=10,
    sync_fn=None,
    units=1,
):
    """Time ``step_fn`` with warmup plus repeats and return latency stats.

    Args:
        name: Backend or strategy name for the resulting stats.
        step_fn: Zero-argument callable performing one inference unit.
        warmup: Untimed iterations used to prime caches and kernels.
        repeats: Timed iterations whose latencies are recorded.
        sync_fn: Optional callable to flush async work before the timed
            region and after every step (e.g. ``torch.cuda.synchronize``).
        units: Work units produced per step, for ms/token-style metrics.

    Returns:
        LatencyStats summarizing the timed iterations.
    """
    for _ in range(warmup):
        step_fn()
        if sync_fn is not None:
            sync_fn()
    if sync_fn is not None:
        sync_fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        step_fn()
        if sync_fn is not None:
            sync_fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return LatencyStats(
        name=name,
        repeats=repeats,
        warmup=warmup,
        samples_ms=samples,
        units=units,
    )


class BackendLatencyReport:
    """Collect latency stats across backends and compare them."""

    def __init__(self):
        """Initialize an empty report."""
        self._stats = []

    def add(self, stats):
        """Append a backend's latency stats.

        Args:
            stats: LatencyStats for one backend.
        """
        self._stats.append(stats)

    @property
    def stats(self):
        """The collected per-backend latency stats."""
        return list(self._stats)

    def fastest(self):
        """Return the stats of the backend with the lowest mean latency.

        This is the per-device "fastest inference strategy" conclusion that
        the benchmarking paper reports.
        """
        if not self._stats:
            raise ValueError("no backend stats recorded")
        return min(self._stats, key=lambda s: s.mean_ms)

    def format_table(self):
        """Render a backend-vs-backend latency comparison table.

        Returns:
            One row per backend with mean / median / min latency (and latency
            per unit), followed by a footer naming the fastest backend.
        """
        header = (
            f"{'backend':<20} {'mean(ms)':>10} {'median(ms)':>11} "
            f"{'min(ms)':>9} {'std(ms)':>9} {'ms/unit':>9}"
        )
        lines = [header, "-" * len(header)]
        for s in self._stats:
            lines.append(
                f"{s.name:<20} {s.mean_ms:>10.3f} {s.median_ms:>11.3f} "
                f"{s.min_ms:>9.3f} {s.std_ms:>9.3f} {s.ms_per_unit:>9.3f}"
            )
        if self._stats:
            best = self.fastest()
            lines.append(f"fastest: {best.name} ({best.mean_ms:.3f} ms/iter)")
        return "\n".join(lines)


def compare_backends(runners, *, warmup=3, repeats=10):
    """Benchmark each runner and return a latency comparison report.

    Args:
        runners: Sequence of BackendRunner instances to benchmark.
        warmup: Untimed warmup iterations per runner.
        repeats: Timed iterations per runner.

    Returns:
        BackendLatencyReport with one entry per runner.
    """
    report = BackendLatencyReport()
    for runner in runners:
        runner.prepare()
        stats = bench_callable(
            runner.name,
            runner.step,
            warmup=warmup,
            repeats=repeats,
            sync_fn=getattr(runner, "sync_fn", None),
            units=getattr(runner, "units", 1),
        )
        report.add(stats)
    return report


def make_generate_runner(
    model,
    input_ids,
    genlen,
    *,
    prompt_len,
    name,
    cg=True,
    eos_token_id=None,
    units=None,
    generate_fn=None,
):
    """Build a runner that times ``model.generate(...)`` like generation.py.

    The runner forwards the same ``cg``, ``return_dict_in_generate``,
    ``output_scores`` and ``enable_timing`` flags that ``evals/generation.py``
    passes, so the timed work matches the existing generation benchmark
    (sampling uses the model's greedy defaults, since latency is the measured
    quantity). It is dependency-free: ``torch`` is only required when the
    model is later prepared for timing.

    Args:
        model: Object exposing ``generate(input_ids=..., max_length=...)``.
        input_ids: Prompt token ids (a tensor once torch is available).
        genlen: Tokens to generate per step.
        prompt_len: Prompt length used to compute ``max_length``.
        name: Backend or strategy name for the runner.
        cg: Whether to enable CUDA-graph generation (Rene/Llamba support it).
        eos_token_id: Optional end-of-sequence id forwarded to ``generate``.
        units: Work units per step; defaults to ``genlen`` (ms/token).
        generate_fn: Optional callable overriding ``model.generate`` (e.g. a
            ``torch.compile``-wrapped generate), so the same kwargs flow to a
            different inference strategy.

    Returns:
        A CallableRunner whose ``step`` performs one full generation.
    """
    generate = partial(
        generate_fn if generate_fn is not None else model.generate,
        cg=cg,
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        eos_token_id=eos_token_id,
    )
    max_length = prompt_len + genlen

    def _step():
        return generate(input_ids=input_ids, max_length=max_length)

    return CallableRunner(
        name=name,
        step_fn=_step,
        units=genlen if units is None else units,
    )


def resolve_model_class(name):
    """Map a model name to its Edge model class (lazy import).

    Args:
        name: One of ``"Rene"`` or a ``"Llamba-*"`` variant.

    Returns:
        The model's ``LMHeadModel`` class from ``cartesia_pytorch``.
    """
    if name == "Rene":
        from cartesia_pytorch.Rene.rene import ReneLMHeadModel

        return ReneLMHeadModel
    if name.startswith("Llamba"):
        from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel

        return LlambaLMHeadModel
    raise ValueError(f"unknown model: {name}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Edge backend inference latency (paper 2607.11356v1)."
    )
    parser.add_argument("--prompt", type=str, default="Rene Descartes was")
    parser.add_argument("--promptlen", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="Llamba-1B",
        choices=["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
    )
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["eager"],
        choices=["eager", "compiled"],
        help="PyTorch strategies to compare; add MLX/Metal via CallableRunner.",
    )
    return parser.parse_args()


@dataclass
class _LoadedModel:
    """A loaded model and tokenizer plus the inputs to benchmark.

    Attributes:
        model: The Edge LMHeadModel, ready for ``.generate``.
        tokenizer: The matching tokenizer (None for raw-token timing).
        input_ids: Prompt token ids on the target device.
        prompt_len: Prompt length, for ``max_length`` bookkeeping.
        eos_token_id: End-of-sequence id forwarded to ``generate``.
        cg: Whether CUDA-graph generation is supported for this model.
    """

    model: object
    tokenizer: object
    input_ids: object
    prompt_len: int
    eos_token_id: object
    cg: bool


def load_model(args, device):
    """Load the Edge model and prepare benchmark inputs.

    Mirrors ``evals/generation.py``'s ``choose_model`` plus prompt
    tokenization, but returns the pieces a latency runner needs.

    Args:
        args: Parsed CLI arguments.
        device: Target torch device string.

    Returns:
        _LoadedModel ready to be turned into backend runners.
    """
    import torch
    from transformers import AutoTokenizer

    model_cls = resolve_model_class(args.model)
    if args.model == "Rene":
        model = model_cls.from_pretrained("cartesia-ai/Rene-v0.1-1.3b-pytorch")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    else:
        hf_id = {"Llamba-1B": "cartesia-ai/Llamba-1B", "Llamba-3B": "cartesia-ai/Llamba-3B"}
        tok_id = {
            "Llamba-1B": "meta-llama/Llama-3.2-1B",
            "Llamba-3B": "meta-llama/Llama-3.2-3B",
            "Llamba-8B": "meta-llama/Llama-3.1-8B",
        }
        model = model_cls.from_pretrained(hf_id.get(args.model, "cartesia-ai/" + args.model))
        tokenizer = AutoTokenizer.from_pretrained(tok_id[args.model])

    torch.manual_seed(args.seed)
    model.to(device=device)
    model.to(dtype=getattr(torch, args.dtype))
    model.eval()

    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    return _LoadedModel(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        prompt_len=int(input_ids.shape[1]),
        eos_token_id=tokenizer.eos_token_id,
        cg=True,
    )


def build_runners(loaded, args, device):
    """Build the requested backend runners from a loaded model.

    Args:
        loaded: _LoadedModel produced by ``load_model``.
        args: Parsed CLI arguments (``--backends`` selects strategies).
        device: Target torch device string.

    Returns:
        List of CallableRunner instances, one per requested strategy.
    """
    import torch

    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    runners = []
    for backend in args.backends:
        generate_fn = torch.compile(loaded.model.generate) if backend == "compiled" else None
        runner = make_generate_runner(
            loaded.model,
            loaded.input_ids,
            args.genlen,
            prompt_len=loaded.prompt_len,
            name=f"pytorch-{backend}",
            cg=loaded.cg,
            eos_token_id=loaded.eos_token_id,
            generate_fn=generate_fn,
        )
        runner.sync_fn = sync_fn
        runners.append(runner)
    return runners


def main():
    """Run the cross-backend latency benchmark and print the comparison."""
    import torch

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_model(args, device)
    runners = build_runners(loaded, args, device)
    print(f"\nLatency benchmark for {args.model} on {device} ({args.dtype})")
    print(f"Number of parameters: {sum(p.numel() for p in loaded.model.parameters())}")
    report = compare_backends(runners, warmup=args.warmup, repeats=args.repeats)
    print()
    print(report.format_table())


if __name__ == "__main__":
    main()
