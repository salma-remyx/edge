# Copyright (c) 2024, Aviv Bick, Kevin Li.

r"""Benchmark state-free (transfer-function) SSM inference vs. the native scan.

Sweeps the state size ``n`` and reports, for each inference path, the wall-clock
time, the number of carried state elements, and a FLOP estimate. This makes
concrete the central claim of "State-Free Inference of State-Space Models: The
Transfer Function Approach" (Bick et al., 2024,
https://arxiv.org/abs/2405.06147): state-free inference cost is ``O(L log L)``
and independent of the state size, while the native recurrent scan carries and
updates an ``n``-dimensional state at every step and so scales with ``n``.

Run from the ``cartesia-pytorch`` directory (so ``cartesia_pytorch`` is
importable), the same way ``evals/generation.py`` is run:

    python evals/state_free_inference.py
    python -m evals.state_free_inference --seq-len 8192 --state-sizes 16 64 256 1024
"""

import argparse

from cartesia_pytorch.Llamba.mixers.state_free import measure_state_size_scaling


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length L.")
    parser.add_argument(
        "--state-sizes",
        type=int,
        nargs="+",
        default=[16, 64, 256, 1024],
        help="State dimensions n to sweep.",
    )
    parser.add_argument("--num-channels", type=int, default=64, help="Number of SISO channels C.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size B.")
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats per measurement (median)."
    )
    return parser.parse_args()


def format_results(results):
    """Format a scaling sweep as a human-readable table string.

    Args:
        results: List of per-``n`` result dicts from
            :func:`measure_state_size_scaling`.

    Returns:
        A formatted multi-line table.
    """
    header = (
        f"{'state n':>8} | {'native (ms)':>12} | {'state-free (ms)':>15} | "
        f"{'speedup':>8} | {'native state':>12} | {'free state':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        speedup = r["native_ms"] / r["state_free_ms"] if r["state_free_ms"] > 0 else float("inf")
        lines.append(
            f"{r['state_size']:>8} | {r['native_ms']:>12.2f} | {r['state_free_ms']:>15.2f} | "
            f"{speedup:>7.1f}x | {r['native_state_elements']:>12} | "
            f"{r['state_free_state_elements']:>10}"
        )
    return "\n".join(lines)


def main():
    """Run the state-size scaling benchmark and print the results."""
    args = parse_args()
    print(
        f"State-free (transfer-function) vs native-scan inference "
        f"(L={args.seq_len}, C={args.num_channels}, B={args.batch_size})\n"
    )
    results = measure_state_size_scaling(
        seq_len=args.seq_len,
        state_sizes=tuple(args.state_sizes),
        num_channels=args.num_channels,
        batch_size=args.batch_size,
        repeats=args.repeats,
    )
    print(format_results(results))
    print(
        "\nNative-scan time/state grow with n; state-free time/state are flat "
        "(cost lives in the L-length FFTs, not an n-dim recurrence)."
    )


if __name__ == "__main__":
    main()
