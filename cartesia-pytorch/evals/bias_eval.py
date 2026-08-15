"""Measure quantization-induced open-ended bias (QuantiBias) for Llamba / Rene.

Integration entry point for the stereotype-bias probe in
:mod:`evals.bias_probe`. It loads a full-precision (FP) build of a repo
model, derives a quantized build from it (or loads a quantized
checkpoint explicitly), runs the open-ended stereotype probe against
each through the same generation surface used by ``evals/generation.py``,
and prints the FP-vs-quantized bias delta -- the headline quantity from
*QuantiBias: Benchmarking Quantization-Induced Bias in LLMs*
(arXiv:2607.21063). It also reports each build's *effective*
bits-per-weight (:mod:`evals.weight_precision`), the paper's measured
replacement for nominal quantization labels.

Adapted port (Mode 2): the learned judge is replaced by the
parameter-free lexicon proxy in :mod:`evals.bias_probe`; the probe runs
through the repo's existing model wrappers rather than a separate
benchmark framework; and since the repo ships no quantized PyTorch
checkpoints (quantized Llamba builds are MLX-only), the quantized build
is derived on the fly via int8 dynamic quantization of linear layers
unless ``--quant_model`` names an explicit checkpoint.

Example::

    python -m evals.bias_eval \
        --fp_model cartesia-ai/Llamba-8B \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""

import argparse

import torch
from evals.bias_probe import bias_delta, hflm_generate_adapter, run_bias_probe
from evals.weight_precision import measure_weight_precision
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
        default="",
        help="Quantized model repo id to compare against the FP baseline. "
        "If empty (default), the quantized build is derived on the fly from "
        "--fp_model via int8 dynamic quantization of linear layers.",
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


def load_model(repo_id, model_kind, dtype, device):
    """Load a repo model checkpoint (FP or a quantized build).

    Args:
        repo_id: Hugging Face repo id.
        model_kind: ``"llamba"`` or ``"rene"``.
        dtype: Torch dtype name for the model weights.
        device: Device to place the model on.

    Returns:
        The loaded model in eval mode.
    """
    model_cls = LlambaLMHeadModel if model_kind == "llamba" else ReneLMHeadModel
    model = model_cls.from_pretrained(repo_id)
    model.to(device=device)
    model.to(dtype=getattr(torch, dtype))
    model.eval()
    return model


def quantize_linear_in_place(model):
    """Quantize ``nn.Linear`` weights to int8 via dynamic quantization.

    On-the-fly substitute for a pre-quantized checkpoint: the repo ships
    no quantized PyTorch builds (quantized Llamba checkpoints are
    MLX-only), so the quantized build is derived from the FP checkpoint
    with torch's built-in dynamic quantization. Only ``nn.Linear``
    weights are quantized; SSM / normalization layers stay in the load
    dtype, which the effective-bpw report reflects honestly.

    Args:
        model: The loaded FP model; modified in place.
    """
    quantize_dynamic = getattr(torch.ao.quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:  # older torch layouts
        quantize_dynamic = torch.quantization.quantize_dynamic
    quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


@torch.inference_mode()
def main():
    """Run the FP-vs-quantized bias probe and print the delta."""
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = load_model(args.fp_model, args.model_kind, args.dtype, _DEVICE)
    fp_precision = measure_weight_precision(model.state_dict(), label="fp")
    fp_generate = hflm_generate_adapter(model, tokenizer, max_new_tokens=args.max_new_tokens)
    fp_report = run_bias_probe(fp_generate)

    if args.quant_model:
        model = load_model(args.quant_model, args.model_kind, args.dtype, _DEVICE)
        quant_label = args.quant_model
    else:
        quantize_linear_in_place(model)
        quant_label = "int8-dynamic"
    quant_precision = measure_weight_precision(model.state_dict(), label=quant_label)
    quant_generate = hflm_generate_adapter(model, tokenizer, max_new_tokens=args.max_new_tokens)
    quant_report = run_bias_probe(quant_generate)

    print("=== Full-precision build ===")
    print(fp_precision)
    print(fp_report)
    print("\n=== Quantized build ===")
    print(quant_precision)
    print(quant_report)
    print("\n=== Quantization-induced bias delta ===")
    print(bias_delta(fp_report, quant_report))


if __name__ == "__main__":
    main()
