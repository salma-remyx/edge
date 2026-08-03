"""Compare self-refinement vs. multi-agent reasoning for local SSMs.

Adapted from "Two Calls Beat Five Agents: Evaluating Multi-Agent Pipelines
Against Self-Refinement for Local Language Models" (arXiv:2607.26922): for
local ~7B models a cheap two-call self-refinement loop matches or beats
structured multi-agent pipelines (which suffer error accumulation) at a
fraction of the token cost, and the inter-agent communication format
(plaintext vs. JSON) matters more than architectural complexity.

Mode-2 adapted port. The strategy-comparison mechanism is kept at full
fidelity; two auxiliaries are substituted with target-native equivalents:

  * crewAI (the paper's framework, which expects an OpenAI-compatible server
    that the repo does not expose) is replaced by a role-prompt loop over the
    repo's existing ``choose_model() -> model.generate()`` contract.
  * The paper's 500/164-item GSM8K and HumanEval suites are replaced by a
    small built-in problem set; full-suite accuracy belongs in a follow-up.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# A ``generate`` callable maps a prompt string to the model's completion text.
GenerateFn = Callable[[str], str]
# A token counter returns an approximate token count for a string.
TokenCounter = Callable[[str], int]


@dataclass
class Problem:
    """A single grade-school math word problem with a numeric answer."""

    question: str
    answer: str


@dataclass
class SolveResult:
    """The outcome of one strategy on one problem."""

    answer: str | None
    calls: int
    tokens: int


# A tiny built-in problem set standing in for GSM8K. Answers follow the
# standard "#### <number>" convention so answer extraction is uniform.
BUILTIN_PROBLEMS: list[Problem] = [
    Problem(question="A baker has 3 trays of 6 muffins each. How many muffins total?", answer="18"),
    Problem(question="Maya read 12 pages Monday and twice as many Tuesday. How many Tuesday?", answer="24"),
    Problem(question="A parking lot has 4 rows of 15 cars. How many cars total?", answer="60"),
    Problem(question="Tom had 50 dollars, spent 18 then 7. How many dollars left?", answer="25"),
    Problem(question="A school orders 9 boxes of 12 pencils. How many pencils total?", answer="108"),
    Problem(question="Lena runs 5 km a day for 8 days. How many km total?", answer="40"),
    Problem(question="A basket has 4 apples and 3 times as many oranges. How many oranges?", answer="12"),
    Problem(question="A theater has 120 seats and 84 are taken. How many empty?", answer="36"),
]

_DIRECT_PROMPT = (
    "Solve the math problem. End your answer on a new line as "
    "'#### <number>'.\n\nProblem: {question}"
)

_REFINE_PROMPT = (
    "You are checking a math solution. Find any mistake and give the corrected "
    "solution. End your answer on a new line as '#### <number>'.\n\n"
    "Problem: {question}\n\nProposed solution:\n{solution}"
)

_SOLVER_PROMPT = (
    "You are the Solver agent. Solve the math problem step by step. "
    "End your answer on a new line as '#### <number>'.\n\nProblem: {question}"
)

_CRITIC_PROMPT = (
    "You are the Critic agent. Review the proposed solution for errors and "
    "state whether it is correct. If it is wrong, give the correct final "
    "answer as '#### <number>'.\n\n{communication}"
)

_AGGREGATE_PROMPT = (
    "Combine the Solver's solution and the Critic's review into a single "
    "final answer. End with '#### <number>'.\n\n"
    "Problem: {question}\n\nSolver:\n{solver}\n\nCritic:\n{critic}"
)


def extract_answer(text: str) -> str | None:
    """Return the model's final numeric answer, or ``None`` if not found.

    Looks first for the GSM8K-style ``#### <number>`` marker, then falls back
    to the last integer anywhere in the text.
    """
    if not text:
        return None
    marker = "####"
    for line in reversed(text.splitlines()):
        if marker in line:
            tail = line.split(marker, 1)[1].strip()
            num = _first_number(tail)
            if num is not None:
                return num
    return _first_number(text)


def _first_number(text: str) -> str | None:
    """Return the first integer/decimal found in *text*, else ``None``."""
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0).lstrip("0") or "0" if match else None


def grade(predicted: str | None, gold: str) -> bool:
    """Return ``True`` when *predicted* numerically equals *gold*."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return predicted.strip() == gold.strip()


def default_token_counter(text: str) -> int:
    """Return a tokenizer-free token estimate (whitespace split)."""
    return len(text.split())


def make_token_counter(tokenizer) -> TokenCounter:
    """Build a token counter from a tokenizer, falling back to whitespace."""
    if tokenizer is None:
        return default_token_counter
    encode = getattr(tokenizer, "encode", None)

    def _count(text: str) -> int:
        if encode is None:
            return default_token_counter(text)
        try:
            return len(encode(text))
        except Exception:  # noqa: BLE001 - fall back if encoding fails
            return default_token_counter(text)

    return _count


def direct_solve(
    generate: GenerateFn, problem: Problem, token_counter: TokenCounter
) -> SolveResult:
    """Solve in a single generate call (the paper's 'direct' baseline)."""
    prompt = _DIRECT_PROMPT.format(question=problem.question)
    completion = generate(prompt)
    return SolveResult(
        answer=extract_answer(completion),
        calls=1,
        tokens=token_counter(prompt) + token_counter(completion),
    )


def self_refine_v1(
    generate: GenerateFn, problem: Problem, token_counter: TokenCounter
) -> SolveResult:
    """Solve, then reflect and revise in a second call (the paper's V1)."""
    solve_prompt = _SOLVER_PROMPT.format(question=problem.question)
    solution = generate(solve_prompt)
    refine_prompt = _REFINE_PROMPT.format(question=problem.question, solution=solution)
    revised = generate(refine_prompt)
    answer = extract_answer(revised) or extract_answer(solution)
    tokens = (
        token_counter(solve_prompt)
        + token_counter(solution)
        + token_counter(refine_prompt)
        + token_counter(revised)
    )
    return SolveResult(answer=answer, calls=2, tokens=tokens)


def multi_agent(
    generate: GenerateFn,
    problem: Problem,
    token_counter: TokenCounter,
    *,
    fmt: str = "plaintext",
) -> SolveResult:
    """Run a 3-role solver/critic/aggregator loop (the paper's multi-agent arm).

    *fmt* selects the inter-agent communication format. ``"json"`` wraps the
    solver's output in a JSON envelope, which the paper shows increases error
    accumulation for local models relative to plain text.
    """
    solver_prompt = _SOLVER_PROMPT.format(question=problem.question)
    solution = generate(solver_prompt)

    if fmt == "json":
        communication = json.dumps(
            {"question": problem.question, "proposed_solution": solution}
        )
    else:
        communication = f"Problem: {problem.question}\n\nProposed solution:\n{solution}"
    critic_prompt = _CRITIC_PROMPT.format(communication=communication)
    critique = generate(critic_prompt)

    aggregate_prompt = _AGGREGATE_PROMPT.format(
        question=problem.question, solver=solution, critic=critique
    )
    final = generate(aggregate_prompt)

    answer = extract_answer(final) or extract_answer(solution)
    tokens = sum(
        token_counter(p)
        for p in (solver_prompt, solution, critic_prompt, critique, aggregate_prompt, final)
    )
    return SolveResult(answer=answer, calls=3, tokens=tokens)


STRATEGIES = {
    "direct": direct_solve,
    "self_refine": self_refine_v1,
    "multi_agent": multi_agent,
}


def run_benchmark(
    generate: GenerateFn,
    problems: Sequence[Problem],
    strategies: Sequence[str],
    token_counter: TokenCounter,
    *,
    agent_format: str = "plaintext",
) -> dict:
    """Run each strategy over each problem and report accuracy and token use.

    Returns a dict keyed by strategy name with ``accuracy`` (0..1),
    ``avg_tokens``, ``total_calls`` and ``num_problems``.
    """
    results: dict[str, dict] = {}
    for name in strategies:
        fn = STRATEGIES[name]
        correct = 0
        total_tokens = 0
        total_calls = 0
        for problem in problems:
            if name == "multi_agent":
                outcome = fn(generate, problem, token_counter, fmt=agent_format)
            else:
                outcome = fn(generate, problem, token_counter)
            correct += int(grade(outcome.answer, problem.answer))
            total_tokens += outcome.tokens
            total_calls += outcome.calls
        n = len(problems)
        results[name] = {
            "accuracy": correct / n if n else 0.0,
            "avg_tokens": total_tokens / n if n else 0.0,
            "total_calls": total_calls,
            "num_problems": n,
        }
    return results


def decode_completion(out, tokenizer, prompt_len: int) -> str:
    """Decode only the newly generated tokens from a ``model.generate`` result.

    *out* follows the repo's ``return_dict_in_generate=True`` contract: it
    exposes ``.sequences`` (a tensor or list of token-id rows). The first
    ``prompt_len`` tokens of the first row are the prompt and are skipped.
    """
    sequences = out.sequences
    rows = sequences.tolist() if hasattr(sequences, "tolist") else sequences
    first_row = rows[0]
    new_ids = first_row[prompt_len:]
    decoded = tokenizer.batch_decode([new_ids], skip_special_tokens=True)
    return decoded[0] if isinstance(decoded, list) else decoded


def make_generate_fn(model, tokenizer, **gen_kwargs) -> GenerateFn:
    """Adapt the repo's ``model.generate`` contract into ``generate(prompt)``.

    Heavy imports (``torch``) are deferred so the rest of the module stays
    importable without a CUDA stack. Mirrors the call shape used in
    ``evals/generation.py``.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = gen_kwargs.pop("max_new_tokens", 256)

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        out = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            **gen_kwargs,
        )
        return decode_completion(out, tokenizer, input_ids.shape[1])

    return generate


def _strategies_from_args(args) -> list[str]:
    raw = getattr(args, "strategies", "all")
    if raw in (None, "all", ""):
        return list(STRATEGIES)
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _problems_from_args(args) -> list[Problem]:
    n = getattr(args, "bench_problems", None) or len(BUILTIN_PROBLEMS)
    return list(BUILTIN_PROBLEMS[:n])


def run_strategy_bench(
    model, tokenizer, args, generate_fn: GenerateFn | None = None
) -> dict:
    """Entry point dispatched from ``evals/generation.py``.

    When *generate_fn* is ``None`` it is built from *model* and *tokenizer*
    via :func:`make_generate_fn`. Injecting *generate_fn* keeps this function
    unit-testable without a CUDA stack.
    """
    if generate_fn is None:
        generate_fn = make_generate_fn(
            model, tokenizer, **_gen_kwargs_from_args(args)
        )
    token_counter = make_token_counter(tokenizer)
    results = run_benchmark(
        generate_fn,
        _problems_from_args(args),
        _strategies_from_args(args),
        token_counter,
        agent_format=getattr(args, "agent_format", "plaintext"),
    )
    print_report(results, agent_format=getattr(args, "agent_format", "plaintext"))
    return results


def _gen_kwargs_from_args(args) -> dict:
    """Pull sampling kwargs from *args*, mirroring ``evals/generation.py``."""
    return {
        "temperature": getattr(args, "temperature", 1.0),
        "top_k": getattr(args, "top_k", 1),
        "top_p": getattr(args, "top_p", 1.0),
        "min_p": getattr(args, "min_p", 0.0),
        "repetition_penalty": getattr(args, "repetition_penalty", 1.0),
    }


def print_report(results: dict, *, agent_format: str) -> None:
    """Print a human-readable comparison of the strategies."""
    print("\nSelf-refinement vs. multi-agent strategy comparison")
    print(f"multi-agent communication format: {agent_format}")
    header = f"{'strategy':<14}{'accuracy':>10}{'avg tokens':>12}{'calls':>8}"
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        print(
            f"{name:<14}{stats['accuracy'] * 100:>9.1f}%"
            f"{stats['avg_tokens']:>12.0f}{stats['total_calls']:>8}"
        )
    print(
        "\nPaper insight: for local models, communication format and "
        "implementation details can dominate architectural complexity; "
        "watch the token cost of the multi-agent arm."
    )


def parse_args(argv: Sequence[str] | None = None):
    """Parse arguments for the standalone ``python -m evals.refine_vs_agents`` CLI."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="Rene",
        choices=["Rene", "Llamba-1B", "Llamba-3B", "Llamba-8B"],
    )
    parser.add_argument("--bench-problems", type=int, default=len(BUILTIN_PROBLEMS))
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated subset of: direct,self_refine,multi_agent (or 'all').",
    )
    parser.add_argument(
        "--agent-format",
        type=str,
        default="plaintext",
        choices=["plaintext", "json"],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    return parser.parse_args(argv)


def main() -> None:
    """Load a model via ``choose_model`` and run the strategy benchmark."""
    from evals.generation import choose_model

    args = parse_args()
    model, tokenizer = choose_model(args)
    run_strategy_bench(model, tokenizer, args)


if __name__ == "__main__":
    main()
