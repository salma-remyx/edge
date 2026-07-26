# Copyright (c) 2024, Aviv Bick, Kevin Li.

import argparse
import time
from functools import partial

import torch
from transformers import AutoTokenizer

from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from cartesia_pytorch.Rene.rene import ReneLMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Rene Descartes was")
    parser.add_argument("--promptlen", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="Llamba-1B",
        choices=["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
    )
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    # Sampling arguments
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--power_w",
        type=float,
        default=None,
        help=(
            "Average inference power in watts. When set, time_bench reports a "
            "constant-power per-phase energy estimate (energy = power * time)."
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def time_bench(args, input_ids, generate_fn, model=None, power_w=None):
    """Benchmark generation time, split into prefill and decode phases.

    Reports prompt processing (prefill) and decoding (decode) separately and, when
    ``power_w`` is given, estimates per-phase energy under a constant-power model
    (energy = power * time). Each decode token costs far more than each prefill
    token, so output length -- not input length -- dominates both latency and
    energy. Adapted from "Seeing is Free, Speaking is Not" (arXiv:2607.09520).
    """
    from .energy_profile import report_phases

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.repeats):
        out = generate_fn(input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen)
    torch.cuda.synchronize()
    total_ms = (time.time() - start) / args.repeats * 1000

    # Prefill (prompt processing) measured in isolation: a single forward pass
    # over the prompt. Best-effort: if the raw forward is unsupported, time_bench
    # falls back to reporting only the combined total.
    prefill_ms = None
    if model is not None:
        try:
            torch.cuda.synchronize()
            prefill_start = time.time()
            for _ in range(args.repeats):
                model(input_ids)
            torch.cuda.synchronize()
            prefill_ms = (time.time() - prefill_start) / args.repeats * 1000
        except Exception:
            prefill_ms = None

    prompt_len = len(input_ids[0])
    gen_len = len(out.sequences[0]) - prompt_len

    # Print stats
    print(f"\nTiming results for {args.model} model:")
    print(f"Prompt length: {prompt_len}, generation length: {gen_len}")
    print(f"prompt processing + decoding time: {total_ms:.0f}ms")

    report = report_phases(
        total_ms=total_ms,
        prefill_ms=prefill_ms,
        prompt_len=prompt_len,
        gen_len=gen_len,
        power_w=power_w,
    )
    if prefill_ms is not None or power_w is not None:
        print(report.format())
    return report


def choose_model(args):
    """Load the model and tokenizer based on the model name."""
    name = args.model
    # Load model
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


@torch.inference_mode()
def main():
    """Main function for generation benchmarking."""
    # Parse arguments
    args = parse_args()
    torch.manual_seed(args.seed)

    # Load model
    model, tokenizer = choose_model(args)

    # Prepare model
    model.to(device=device)
    model.to(dtype=getattr(torch, args.dtype))
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Tokenize prompt
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (1, args.promptlen), dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)

    # Prepare generation function
    generate_fn = partial(
        model.generate,
        cg=args.model in ["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Generate
    out = generate_fn(input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen)

    if args.prompt is not None:
        print(
            "Generated text:\n",
            tokenizer.batch_decode(sequences=out.sequences.tolist(), skip_special_tokens=True)[0],
        )

    time_bench(args, input_ids, generate_fn, model=model, power_w=args.power_w)


if __name__ == "__main__":
    main()
