# Copyright (c) 2024, Cartesia.

"""Multilingual tokenizer-fragmentation benchmark.

Quantifies how a fixed pre-training tokenizer fragments text across languages
and the resulting per-character decode-cost disparity -- the efficiency axis
motivated by "In-Place Tokenizer Expansion for Pre-trained LLMs"
(arXiv:2607.15232). The paper's headline result is that an under-served
language can require several times more tokens per character than the
tokenizer's primary language, which scales per-character decode latency,
compute, and energy by the same factor on a compact model.

This is a Mode-2 (adapted) port: the paper's core *measurement* -- per-language
fragmentation and its decode-cost impact -- is implemented at full fidelity,
while the paper's *expansion recipe* (BPE-merge continuation, embedding-row
initialization, two-stage adaptation) is intentionally out of scope because it
requires continued-pre-training infrastructure the evals harness does not host.

The module imports cleanly without a model or GPU: the fragmentation analysis
is pure Python over any tokenizer exposing ``encode`` (e.g. HuggingFace
``AutoTokenizer``). Heavy imports (``torch``, ``transformers``, the existing
``evals.generation`` ``choose_model`` / ``time_bench`` contract) are deferred to
the optional ``--decode-latency`` CLI path.
"""

import argparse
import json
import time
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

# Tokenizer HuggingFace repos, mirroring ``evals.generation.choose_model`` so
# the tokenizer-only path (default) loads the same tokenizer the model uses
# without paying for the full model download.
TOKENIZER_REPOS = {
    "Rene": "allenai/OLMo-1B-hf",
    "Llamba-1B": "meta-llama/Llama-3.2-1B",
    "Llamba-3B": "meta-llama/Llama-3.2-3B",
    "Llamba-8B": "meta-llama/Llama-3.1-8B",
}

MODEL_CHOICES = list(TOKENIZER_REPOS)

# One parallel sentence per language (a translation of a fixed English prompt).
# Cross-script languages the paper highlights (Hindi, Vietnamese, Thai) are
# included alongside the Latin-script baseline. Replace with FLORES or a corpus
# file (``--corpus``) for real benchmarking.
SAMPLE_CORPUS = {
    "English": "The quick brown fox jumps over the lazy dog near the river.",
    "Spanish": "El rapido zorro marron salta sobre el perro perezoso cerca del rio.",
    "Vietnamese": "Con cao nau nhanh nhen nhay qua con cho luoi bieng gan dong song.",
    "Hindi": "तेज़ भूरी लोमड़ी नदी के पास आलसी कुत्ते के ऊपर कूद जाती है।",
    "Thai": "สุนัขจิ้งจอกสีน้ำตาลว่องไวกระโดดข้ามสุนัขขี้เกียจใกล้แม่น้ำ",
    "Chinese": "敏捷的棕色狐狸跳过河边的懒狗。",
    "Arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من النهر.",
    "Russian": "Быстрая бурая лиса прыгает через ленивую собаку у реки.",
}


@dataclass
class FragmentationStats:
    """Tokenizer fragmentation metrics for a single text sample.

    Attributes:
        tokens: Number of tokenizer output ids.
        chars: Number of Unicode code points in the text.
        n_bytes: UTF-8 byte length of the text.
        words: Whitespace-delimited word count (best-effort; unreliable for
            unsegmented scripts such as Thai and Chinese).
    """

    tokens: int
    chars: int
    n_bytes: int
    words: int

    @property
    def tokens_per_char(self) -> float:
        """Tokens emitted per Unicode character (script-agnostic density)."""
        return self.tokens / self.chars if self.chars else float("nan")

    @property
    def bytes_per_token(self) -> float:
        """UTF-8 bytes carried per emitted token."""
        return self.n_bytes / self.tokens if self.tokens else float("nan")

    @property
    def tokens_per_word(self) -> float:
        """Tokens per whitespace word (caveat: unsegmented scripts)."""
        return self.tokens / self.words if self.words else float("nan")


@dataclass
class LanguageReport:
    """Per-language fragmentation plus disparity versus a baseline language.

    Attributes:
        language: Language name.
        stats: Raw fragmentation metrics.
        tokens_per_char_ratio: ``tokens_per_char`` relative to the baseline
            language; the paper's "N-times more tokens" axis (baseline == 1.0).
    """

    language: str
    stats: FragmentationStats
    tokens_per_char_ratio: float


def count_tokens(tokenizer, text: str) -> int:
    """Return the number of ids ``tokenizer`` produces for ``text``.

    Works with any tokenizer exposing ``encode(text, add_special_tokens=...)``
    (e.g. HuggingFace tokenizers) or a callable returning ``input_ids``.
    """
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
    encoded = tokenizer(text)
    ids = getattr(encoded, "input_ids", encoded)
    return len(ids)


def fragmentation_stats(tokenizer, text: str) -> FragmentationStats:
    """Compute fragmentation metrics for ``text`` under ``tokenizer``."""
    return FragmentationStats(
        tokens=count_tokens(tokenizer, text),
        chars=len(text),
        n_bytes=len(text.encode("utf-8")),
        words=len(text.split()),
    )


def multilingual_report(
    tokenizer,
    corpus: Mapping[str, str],
    baseline: str = "English",
) -> Sequence[LanguageReport]:
    """Score every language in ``corpus`` and its disparity versus ``baseline``.

    Args:
        tokenizer: Tokenizer exposing ``encode`` or ``__call__``.
        corpus: Mapping ``{language name: sample text}``.
        baseline: Language whose ``tokens_per_char`` defines ratio 1.0.

    Returns:
        Reports sorted by descending fragmentation (most fragmented first).

    Raises:
        KeyError: If ``baseline`` is not a key in ``corpus``.
        ValueError: If the baseline text is empty.
    """
    if baseline not in corpus:
        raise KeyError(f"baseline language {baseline!r} not in corpus")
    base = fragmentation_stats(tokenizer, corpus[baseline])
    if not base.chars:
        raise ValueError("baseline text is empty")
    base_density = base.tokens_per_char
    reports = []
    for language, text in corpus.items():
        stats = fragmentation_stats(tokenizer, text)
        reports.append(
            LanguageReport(
                language=language,
                stats=stats,
                tokens_per_char_ratio=stats.tokens_per_char / base_density,
            )
        )
    reports.sort(key=lambda report: report.tokens_per_char_ratio, reverse=True)
    return reports


def estimate_decode_disparity(
    reports: Sequence[LanguageReport],
    baseline: str = "English",
    ms_per_token: float | None = None,
) -> dict[str, dict[str, float]]:
    """Combine fragmentation with per-token decode cost into per-character cost.

    Mirrors the paper's final step: per-character decode cost is
    ``tokens_per_char * ms_per_token``. Because per-token decode cost is
    language-independent for a single tokenizer, the cost ratio equals the
    fragmentation ratio; supplying ``ms_per_token`` additionally yields the
    absolute per-character latency, which is what the paper reports as a
    per-character decode speedup across devices.

    Args:
        reports: Output of ``multilingual_report``.
        baseline: Language whose cost defines the ratio 1.0.
        ms_per_token: Measured per-token decode latency in ms. If ``None``,
            only the fragmentation-based ratio is returned.

    Returns:
        Per-language ``{metric: value}`` where metrics include
        ``tokens_per_char_ratio`` and, when ``ms_per_token`` is given,
        ``ms_per_char`` and ``per_char_cost_ratio``.

    Raises:
        KeyError: If ``baseline`` is not present in ``reports``.
    """
    base_report = next((report for report in reports if report.language == baseline), None)
    if base_report is None:
        raise KeyError(f"baseline language {baseline!r} not in reports")
    base_density = base_report.stats.tokens_per_char
    out: dict[str, dict[str, float]] = {}
    for report in reports:
        entry: dict[str, float] = {"tokens_per_char_ratio": report.tokens_per_char_ratio}
        if ms_per_token is not None:
            ms_per_char = report.stats.tokens_per_char * ms_per_token
            entry["ms_per_char"] = ms_per_char
            base_cost = base_density * ms_per_token
            entry["per_char_cost_ratio"] = ms_per_char / base_cost if base_cost else float("nan")
        out[report.language] = entry
    return out


def format_report(reports: Sequence[LanguageReport], baseline: str = "English") -> str:
    """Render ``reports`` as a text table (baseline language under ``baseline``)."""
    lines = [f"Tokenizer fragmentation vs baseline='{baseline}'", ""]
    header = f"{'language':<12}{'tokens':>8}{'tok/char':>11}{'bytes/tok':>11}{'ratio':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for report in reports:
        row = (
            f"{report.language:<12}{report.stats.tokens:>8}"
            f"{report.stats.tokens_per_char:>11.3f}{report.stats.bytes_per_token:>11.2f}"
            f"{report.tokens_per_char_ratio:>9.2f}"
        )
        lines.append(row)
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark multilingual tokenizer fragmentation and decode-cost disparity.",
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="Llamba-3B")
    parser.add_argument(
        "--baseline",
        default="English",
        help="corpus language whose tokenization density defines ratio 1.0",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="path to a JSON {language: text} corpus; defaults to a built-in sample",
    )
    parser.add_argument(
        "--decode-latency",
        action="store_true",
        help="also load the model and measure per-token decode latency via choose_model",
    )
    parser.add_argument("--genlen", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args(argv)


def _load_corpus(path: str) -> Mapping[str, str]:
    """Load a ``{language: text}`` corpus from a JSON file."""
    with open(path, encoding="utf-8") as handle:
        corpus = json.load(handle)
    if not isinstance(corpus, dict) or not all(isinstance(value, str) for value in corpus.values()):
        raise ValueError("corpus file must be a JSON object mapping language to text")
    return corpus


def _load_tokenizer(model_name: str):
    """Lazily load the tokenizer that ``model_name`` ships with."""
    from transformers import AutoTokenizer  # local import: keeps module import-free

    return AutoTokenizer.from_pretrained(TOKENIZER_REPOS[model_name])


def _measure_ms_per_token(model_name: str, genlen: int, dtype: str) -> float:
    """Load the model via the existing ``choose_model`` contract and time decode.

    Returns the measured per-token decode latency in milliseconds. The model is
    loaded through ``evals.generation.choose_model`` (the same loader the
    generation benchmark uses) and timed with a one-shot generation pass.
    """
    import torch  # local import: only needed for the decode-latency path
    from evals.generation import choose_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args_ns = argparse.Namespace(model=model_name)
    model, tokenizer = choose_model(args_ns)
    model.to(device=device)
    model.to(dtype=getattr(torch, dtype))
    model.eval()
    prompt_ids = tokenizer("a", return_tensors="pt").input_ids.to(device=device)
    generate_fn = partial(
        model.generate,
        cg=True,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    out = generate_fn(input_ids=prompt_ids, max_length=prompt_ids.shape[1] + genlen)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000.0
    n_new_tokens = out.sequences.shape[1] - prompt_ids.shape[1]
    return elapsed_ms / max(n_new_tokens, 1)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the multilingual tokenizer-fragmentation benchmark."""
    args = parse_args(argv)
    corpus = _load_corpus(args.corpus) if args.corpus else dict(SAMPLE_CORPUS)
    tokenizer = _load_tokenizer(args.model)
    reports = multilingual_report(tokenizer, corpus, baseline=args.baseline)
    print(format_report(reports, baseline=args.baseline))
    if args.decode_latency:
        ms_per_token = _measure_ms_per_token(args.model, args.genlen, args.dtype)
        disparity = estimate_decode_disparity(
            reports, baseline=args.baseline, ms_per_token=ms_per_token
        )
        print(f"\nMeasured per-token decode latency: {ms_per_token:.3f} ms/tok")
        print(f"{'language':<12}{'ms/char':>10}{'x baseline':>12}")
        for language, entry in disparity.items():
            print(
                f"{language:<12}{entry['ms_per_char']:>10.3f}"
                f"{entry['per_char_cost_ratio']:>12.2f}"
            )


if __name__ == "__main__":
    main()
