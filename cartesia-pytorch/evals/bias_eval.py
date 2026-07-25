"""Measure quantization-induced open-ended bias (QuantiBias) for Llamba / Rene.

Integration entry point for the stereotype-bias probe in
:mod:`evals.bias_probe`. It loads a full-precision (FP) build and a
quantized build of a repo model, runs the open-ended stereotype probe
against each through the same generation surface used by
``evals/generation.py``, and prints the FP-vs-quantized bias delta --
the headline quantity from *QuantiBias: Benchmarking
Quantization-Induced Bias in LLMs* (arXiv:2607.21063).

Adapted port (Mode 2): the learned judge is replaced by the
parameter-free lexicon proxy in :mod:`evals.bias_probe`; the probe runs
through the repo's existing model wrappers rather than a separate
benchmark framework.

Example::

    python -m evals.bias_eval \
        --fp_model cartesia-ai/Llamba-8B \
        --quant_model cartesia-ai/Llamba-8B-4bit-mix \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""

import argparse

import torch
from evals.bias_probe import bias_delta, hflm_generate_adapter, run_bias_probe
from transformers import AutoTokenizer

from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from cartesia_pytorch.Rene.rene import ReneLMHeadModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp_model",
        type=str,
        default="cartesia-ai/Llamba-8B",
        help="Full-precision model repo id (the FP baseline).",
    )
    parser.add_argument(
        "--quant_model",
        type=str,
        default="cartesia-ai/Llamba-8B-4bit-mix",
        help="Quantized model repo id to compare against the FP baseline.",
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default="llamba",
        choices=["llamba", "rene"],
        help="Which model family to load.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer repo id.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_build(repo_id, model_kind, tokenizer, dtype, device, max_new_tokens):
    """Load a model build and wrap it as a probe ``generate`` function.

    Args:
        repo_id: Hugging Face repo id (FP or a pre-quantized checkpoint
            such as ``cartesia-ai/Llamba-8B-4bit-mix``).
        model_kind: ``"llamba"`` or ``"rene"``.
        tokenizer: The tokenizer to pair with the model.
        dtype: Torch dtype name for the model weights.
        device: Device to place the model on.
        max_new_tokens: Length of the open-ended continuation to generate.

    Returns:
        A ``Callable[[str], str]`` generation function for
        :func:`evals.bias_probe.run_bias_probe`.
    """
    model_cls = LlambaLMHeadModel if model_kind == "llamba" else ReneLMHeadModel
    model = model_cls.from_pretrained(repo_id)
    model.to(device=device)
    model.to(dtype=getattr(torch, dtype))
    model.eval()
    return hflm_generate_adapter(model, tokenizer, max_new_tokens=max_new_tokens)


@torch.inference_mode()
def main():
    """Run the FP-vs-quantized bias probe and print the delta."""
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    fp_generate = load_build(
        args.fp_model,
        args.model_kind,
        tokenizer,
        args.dtype,
        _DEVICE,
        args.max_new_tokens,
    )
    quant_generate = load_build(
        args.quant_model,
        args.model_kind,
        tokenizer,
        args.dtype,
        _DEVICE,
        args.max_new_tokens,
    )

    fp_report = run_bias_probe(fp_generate)
    quant_report = run_bias_probe(quant_generate)

    print("=== Full-precision build ===")
    print(fp_report)
    print("\n=== Quantized build ===")
    print(quant_report)
    print("\n=== Quantization-induced bias delta ===")
    print(bias_delta(fp_report, quant_report))


if __name__ == "__main__":
    main()
