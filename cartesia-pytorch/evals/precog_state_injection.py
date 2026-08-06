# Copyright (c) 2024, Cartesia, Inc.
r"""Compare RAG prefill latency: in-context retrieval vs. PRECOG O(1) injection.

Adapted from PRECOG (Pre-Computed Context Injection, arXiv:2608.02560): the corpus
is encoded once into the SSM recurrent state; each query then injects that state
instead of re-prefilling the corpus, collapsing prefill from ``O(L_corpus)`` to
``O(1)`` per query.

Run from the ``cartesia-pytorch`` directory::

    python -m evals.precog_state_injection \\
        --model Llamba-1B --corpus-len 8192 --query-len 32 --repeats 5
"""

import argparse
import statistics
import time

import torch

from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from cartesia_pytorch.Llamba.mixers.corpus_state_cache import (
    build_inference_params,
    encode_document,
    inject_states,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

_LLAMBA_REPOS = {
    "Llamba-1B": "cartesia-ai/Llamba-1B",
    "Llamba-3B": "cartesia-ai/Llamba-3B",
    "Llamba-8B": "cartesia-ai/Llamba-8B",
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Llamba-1B", choices=list(_LLAMBA_REPOS))
    parser.add_argument("--corpus-len", type=int, default=8192)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"]
    )
    return parser.parse_args()


def load_model(name):
    """Load a pretrained Llamba model by short name."""
    return LlambaLMHeadModel.from_pretrained(_LLAMBA_REPOS[name])


def _maybe_sync():
    """Synchronize CUDA timing, if a GPU is present."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def time_prefill(model, input_ids, repeats):
    """Return the median wall time (ms) of ``repeats`` fresh prefill passes."""
    samples = []
    for _ in range(repeats):
        params = build_inference_params(model, 1, input_ids.shape[1])
        _maybe_sync()
        start = time.perf_counter()
        model(input_ids, inference_params=params)
        _maybe_sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


@torch.inference_mode()
def main():
    """Encode a corpus once and compare per-query prefill cost both ways."""
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_model(args.model)
    model.to(device=device)
    model.to(dtype=getattr(torch, args.dtype))
    model.eval()

    corpus = torch.randint(1, 1000, (1, args.corpus_len), device=device)
    query = torch.randint(1, 1000, (1, args.query_len), device=device)

    # Baseline: in-context RAG re-prefills the corpus with every query.
    baseline_ms = time_prefill(model, torch.cat([corpus, query], dim=1), args.repeats)

    # PRECOG: encode the corpus once into the recurrent state.
    encode_start = time.perf_counter()
    corpus_params = build_inference_params(model, 1, args.corpus_len)
    corpus_state = encode_document(model, corpus_params, corpus)
    encode_ms = (time.perf_counter() - encode_start) * 1e3

    # Each query injects the corpus state (O(1)) and prefills only the query.
    precog_samples = []
    for _ in range(args.repeats):
        params = build_inference_params(model, 1, args.query_len)
        inject_states(model.backbone.layers, params, corpus_state)
        _maybe_sync()
        start = time.perf_counter()
        model(query, inference_params=params)
        _maybe_sync()
        precog_samples.append((time.perf_counter() - start) * 1e3)
    precog_ms = statistics.median(precog_samples)

    print(
        f"\nPRECOG state injection on {args.model} (corpus={args.corpus_len}, "
        f"query={args.query_len}, repeats={args.repeats}):\n"
    )
    print(f"  in-context RAG prefill (corpus+query): {baseline_ms:10.1f} ms/query")
    print(f"  PRECOG corpus encode (one-time):       {encode_ms:10.1f} ms")
    print(f"  PRECOG inject + query prefill:         {precog_ms:10.3f} ms/query")
    print(f"  per-query speedup:                     {baseline_ms / precog_ms:10.0f}x")


if __name__ == "__main__":
    main()
