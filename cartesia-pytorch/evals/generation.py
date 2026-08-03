# Copyright (c) 2024, Aviv Bick, Kevin Li.

import argparse
import time
from functools import partial

import torch

from cartesia_pytorch.utils.compression_screen import format_screen_report, screen_state_dicts

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
    # Data-free compression-fidelity screen (arXiv:2607.28196): flags coherent
    # low-rank compression of the loaded model vs this baseline before deploy.
    parser.add_argument(
        "--compression-baseline",
        type=str,
        default=None,
        help="Baseline model name to screen the loaded model against.",
    )
    parser.add_argument(
        "--compression-threshold",
        type=float,
        default=None,
        help="Flag threshold on coherent-fraction x error-rate (uses screen default).",
    )
    parser.add_argument("--compression-top-k", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def time_bench(args, input_ids, generate_fn):
    """Benchmark the generation time."""
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.repeats):
        out = generate_fn(input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen)
    torch.cuda.synchronize()

    # Print stats
    print(f"\nTiming results for {args.model} model:")
    print(
        f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}"
    )
    print(f"prompt processing + decoding time: {(time.time() - start) / args.repeats * 1000:.0f}ms")


def choose_model(args):
    """Load the model and tokenizer based on the model name."""
    # Imported lazily so the benchmark CLI stays importable without the
    # accelerator-only model backends (e.g. on CPU / in tests).
    from transformers import AutoTokenizer

    from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
    from cartesia_pytorch.Rene.rene import ReneLMHeadModel

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


def run_compression_screen(model, baseline_model, threshold=None, top_k=1):
    """Screen ``model`` against ``baseline_model`` with the data-free weight screen.

    Flags coherent low-rank compression (e.g. SVD truncation) that is too gentle
    to move the generation benchmark. Needs only the two weight sets.
    """
    kwargs = {} if threshold is None else {"threshold": threshold}
    return screen_state_dicts(
        original=baseline_model.state_dict(), compressed=model.state_dict(), top_k=top_k, **kwargs
    )


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

    # Optional data-free screen of the loaded model vs an uncompressed baseline
    # (arXiv:2607.28196): catches coherent low-rank compression the benchmark misses.
    if args.compression_baseline is not None:
        saved_model = args.model
        args.model = args.compression_baseline
        baseline, _ = choose_model(args)
        args.model = saved_model
        report = run_compression_screen(
            model,
            baseline,
            threshold=args.compression_threshold,
            top_k=args.compression_top_k,
        )
        print(format_screen_report(report))

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

    time_bench(args, input_ids, generate_fn)


if __name__ == "__main__":
    main()
