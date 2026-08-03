"""Data-free compression-fidelity screen CLI.

Screen a compressed Llamba / Rene checkpoint against its uncompressed baseline
with the two-axis weight-error statistic (coherent-fraction x error-rate) from
"Fidelity Is Not Safety..." (arXiv:2607.28196). Standard quality guards
(perplexity, MMLU, fidelity probes) pass on gently low-rank-compressed models
that nonetheless invent procedure steps as agents; this screen flags those
builds before agentic deployment. It needs only the two weight sets: no data,
no inference.

Example::

    python -m evals.compression_screen \
        --original cartesia-ai/Llamba-8B \
        --compressed ./my-svd-truncated-llamba-8b \
        --model-type Llamba
"""

import argparse

from cartesia_pytorch.utils.compression_screen import (
    DEFAULT_SCREEN_THRESHOLD,
    format_screen_report,
    screen_state_dicts,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=str, required=True, help="Uncompressed baseline.")
    parser.add_argument(
        "--compressed", type=str, required=True, help="Compressed model under test."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["Llamba", "Rene"],
        help="Which model class to load both checkpoints as.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCREEN_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--name-glob",
        type=str,
        action="append",
        default=None,
        help="fnmatch pattern(s) restricting which params are screened; repeatable.",
    )
    return parser.parse_args()


def load_state_dict(path, model_type, dtype):
    """Load a checkpoint's state dict as the given model type.

    The model backends are imported lazily so this CLI stays importable without
    the accelerator-only dependencies.

    Args:
        path: HuggingFace repo id or local path.
        model_type: ``"Llamba"`` or ``"Rene"``.
        dtype: Floating point dtype name to load weights as.

    Returns:
        The model's ``state_dict()``.

    Raises:
        ValueError: If ``model_type`` is unknown.
    """
    import torch

    if model_type == "Llamba":
        from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel as Model
    elif model_type == "Rene":
        from cartesia_pytorch.Rene.rene import ReneLMHeadModel as Model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model = Model.from_pretrained(path)
    model.to(dtype=getattr(torch, dtype))
    return model.state_dict()


def main():
    """Run the data-free compression screen and print the report."""
    args = parse_args()
    original = load_state_dict(args.original, args.model_type, args.dtype)
    compressed = load_state_dict(args.compressed, args.model_type, args.dtype)
    report = screen_state_dicts(
        original,
        compressed,
        threshold=args.threshold,
        top_k=args.top_k,
        name_globs=args.name_glob,
    )
    print(format_screen_report(report))


if __name__ == "__main__":
    main()
