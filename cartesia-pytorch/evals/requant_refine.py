# Copyright (c) 2024, Aviv Bick, Kevin Li.

"""ReQuant refinement benchmark for Llamba / Rene PyTorch models.

Loads a model, captures calibration activations at every ``nn.Linear`` via
forward hooks, applies the ReQuant fixed-grid discrete refinement to each
linear weight, and reports the reduction in Hessian-weighted reconstruction
error along with the change in perplexity on a short prompt.

This is the evaluation entry point for the ``utils/requant`` capability. It is
intended to be run on a CUDA device with downloaded weights, mirroring
``evals/generation.py``::

    python -m evals.requant_refine --model Llamba-1B --bits 4 --group-size 128 \
        --num-sweeps 5 --num-calib-tokens 2048

Adapted from "ReQuant: Fixed-Grid Discrete Refinement for Post-Training
Quantization" (arXiv:2608.07019). See ``cartesia_pytorch/utils/requant.py`` for
the method and the scope of the adaptation.
"""

import argparse

import torch
from torch import nn
from transformers import AutoTokenizer

from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from cartesia_pytorch.Rene.rene import ReneLMHeadModel
from cartesia_pytorch.utils.requant import (
    dequantize,
    hessian_from_inputs,
    quantize_rtn,
    reconstruction_error,
    requant_refine,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Llamba-1B",
        choices=["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-sweeps", type=int, default=5)
    parser.add_argument("--neighborhood", type=int, default=1)
    parser.add_argument("--calib-prompt", type=str, default="The quick brown fox jumps over")
    parser.add_argument("--num-calib-tokens", type=int, default=2048)
    parser.add_argument("--eval-prompt", type=str, default="Once upon a time")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    return parser.parse_args()


def choose_model(args):
    """Load the model and tokenizer based on the model name."""
    name = args.model
    if name == "Llamba-1B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-1B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    elif name == "Llamba-3B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-3B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    elif name == "Llamba-8B":
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    elif name == "Rene":
        model = ReneLMHeadModel.from_pretrained("cartesia-ai/Rene-v0.1-1.3b-pytorch")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    else:
        raise NotImplementedError
    return model, tokenizer


def calibration_ids(tokenizer, prompt, num_tokens):
    """Build a calibration token sequence of the requested length."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < num_tokens:
        # Pad by repeating the prompt so the Hessian sees enough rows.
        repeats = (num_tokens + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, repeats)[:, :num_tokens]
    return ids


def collect_linear_inputs(model, input_ids):
    """Capture the input activations of every ``nn.Linear`` in one forward pass.

    Args:
        model: The language model.
        input_ids: Token ids of shape ``[1, seq_len]``.

    Returns:
        A dict mapping ``id(linear)`` to a ``[num_rows, in_features]`` tensor
        of captured input activations (float32, detached).
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(module):
        def hook(_module, args):
            x = args[0]
            captured[id(_module)] = x.detach().to(torch.float32).reshape(-1, x.shape[-1])

        return hook

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(module)))

    try:
        with torch.inference_mode():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def compute_perplexity(model, input_ids):
    """Compute causal-language-model perplexity on ``input_ids``."""
    with torch.inference_mode():
        logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    return float(torch.exp(loss).item())


def _valid_group_size(in_features, preferred):
    """Return a group size that divides ``in_features``, falling back gracefully."""
    if in_features % preferred == 0:
        return preferred
    for candidate in (64, 32, 16, 8):
        if in_features % candidate == 0:
            return candidate
    return in_features  # whole row as a single group


@torch.inference_mode()
def run_requant(model, linear_inputs, args):
    """Refine every ``nn.Linear`` in place; return reconstruction-error totals."""
    total_before = 0.0
    total_after = 0.0
    n_layers = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        inputs = linear_inputs.get(id(module))
        if inputs is None or inputs.shape[0] == 0:
            continue
        out_features, in_features = module.weight.shape
        group_size = _valid_group_size(in_features, args.group_size)
        weight = module.weight.detach().to(torch.float32)
        hessian = hessian_from_inputs(inputs)

        codes, scale = quantize_rtn(weight, args.bits, group_size)
        before = reconstruction_error(
            weight, dequantize(codes, scale, group_size, out_features, in_features), hessian
        )
        result = requant_refine(
            weight,
            hessian,
            bits=args.bits,
            group_size=group_size,
            num_sweeps=args.num_sweeps,
            neighborhood=args.neighborhood,
        )
        refined = dequantize(result.codes, result.scale, group_size, out_features, in_features)
        after = reconstruction_error(weight, refined, hessian)

        module.weight.copy_(refined.to(module.weight.dtype))
        total_before += before
        total_after += after
        n_layers += 1
    return total_before, total_after, n_layers


@torch.inference_mode()
def main():
    """Main entry point for the ReQuant refinement benchmark."""
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = choose_model(args)
    model.to(device=device)
    model.to(dtype=getattr(torch, args.dtype))
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    calib_ids = calibration_ids(tokenizer, args.calib_prompt, args.num_calib_tokens)
    eval_ids = tokenizer(args.eval_prompt, return_tensors="pt").input_ids.to(device)
    if eval_ids.shape[1] < 2:
        raise ValueError("eval_prompt must tokenize to at least 2 tokens")

    baseline_ppl = compute_perplexity(model, eval_ids)
    print(f"Baseline perplexity: {baseline_ppl:.4f}")

    linear_inputs = collect_linear_inputs(model, calib_ids)
    total_before, total_after, n_layers = run_requant(model, linear_inputs, args)

    refined_ppl = compute_perplexity(model, eval_ids)

    reduction = 1.0 - (total_after / total_before) if total_before > 0 else 0.0
    print(f"\nReQuant refinement over {n_layers} nn.Linear layers:")
    print(
        f"  reconstruction error: {total_before:.4f} -> {total_after:.4f} "
        f"({100.0 * reduction:.2f}% reduction)"
    )
    print(
        f"  perplexity:           {baseline_ppl:.4f} -> {refined_ppl:.4f} "
        f"(delta {refined_ppl - baseline_ppl:+.4f})"
    )


if __name__ == "__main__":
    main()
