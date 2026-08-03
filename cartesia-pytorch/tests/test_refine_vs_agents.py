"""Tests for the self-refinement vs. multi-agent strategy benchmark.

These run without a CUDA stack: the strategy logic is exercised through an
injected ``generate`` callable, and the repo's ``model.generate`` contract is
exercised through :func:`decode_completion` fakes.
"""

from types import SimpleNamespace

from cartesia_pytorch.version import __version__ as CARTESIA_VERSION
from evals.refine_vs_agents import (
    BUILTIN_PROBLEMS,
    decode_completion,
    direct_solve,
    extract_answer,
    grade,
    make_token_counter,
    multi_agent,
    run_benchmark,
    run_strategy_bench,
    self_refine_v1,
)


class FakeGenerate:
    """Records prompts and returns scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        """Record the prompt and return the next scripted response."""
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "#### 0"


class FakeSequences:
    """Mimics a tensor's ``.tolist()`` for the generate contract."""

    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        """Return the stored rows, mimicking a tensor."""
        return self._rows


class FakeOutput:
    """Mimics ``return_dict_in_generate=True`` output (``.sequences``)."""

    def __init__(self, rows):
        self.sequences = FakeSequences(rows)


class FakeTokenizer:
    """Decodes token-id rows to a deterministic string."""

    def batch_decode(self, sequences, skip_special_tokens=True):
        """Decode the first token-id row to a space-joined string."""
        return [" ".join(str(token) for token in sequences[0])]


def test_answer_extraction_handles_marker_and_fallback():
    """``#### N`` wins; otherwise the last integer is used."""
    assert extract_answer("work...\n#### 42") == "42"
    assert extract_answer("the answer is 7") == "7"
    assert extract_answer("no number here") is None


def test_grade_is_numeric():
    """Numeric equality is tolerant of formatting but not of wrong values."""
    assert grade("18", "18") is True
    assert grade("018", "18") is True
    assert grade("19", "18") is False
    assert grade(None, "18") is False


def test_direct_solve_uses_one_call():
    """The direct baseline consumes a single generate call."""
    problem = BUILTIN_PROBLEMS[0]
    gen = FakeGenerate(["#### " + problem.answer])
    result = direct_solve(gen, problem, make_token_counter(None))
    assert result.calls == 1
    assert grade(result.answer, problem.answer)


def test_self_refine_uses_two_calls():
    """Self-refinement issues exactly two generate calls (solve + revise)."""
    problem = BUILTIN_PROBLEMS[0]
    gen = FakeGenerate(["#### 999", "#### " + problem.answer])
    result = self_refine_v1(gen, problem, make_token_counter(None))
    assert result.calls == 2
    assert len(gen.prompts) == 2
    assert grade(result.answer, problem.answer)


def test_multi_agent_call_count_and_format():
    """The multi-agent arm issues three calls and respects the format flag."""
    problem = BUILTIN_PROBLEMS[0]
    responses = ["#### " + problem.answer, "correct", "#### " + problem.answer]

    plain = FakeGenerate(responses)
    multi_agent(plain, problem, make_token_counter(None), fmt="plaintext")
    assert len(plain.prompts) == 3
    assert "Proposed solution:" in plain.prompts[1]
    assert '"proposed_solution"' not in plain.prompts[1]

    json_mode = FakeGenerate(list(responses))
    multi_agent(json_mode, problem, make_token_counter(None), fmt="json")
    assert len(json_mode.prompts) == 3
    assert '"proposed_solution"' in json_mode.prompts[1]


def test_run_benchmark_accuracy_and_token_accounting():
    """A perfect fake generate yields full accuracy and per-strategy token shape."""
    problem = BUILTIN_PROBLEMS[0]

    def always_correct(prompt):
        return "#### " + problem.answer

    results = run_benchmark(
        always_correct,
        [problem],
        ["direct", "self_refine", "multi_agent"],
        make_token_counter(None),
    )
    assert results["direct"]["accuracy"] == 1.0
    assert results["self_refine"]["accuracy"] == 1.0
    assert results["multi_agent"]["accuracy"] == 1.0
    # More calls cost more tokens: multi_agent (3) > self_refine (2) > direct (1).
    assert results["multi_agent"]["total_calls"] == 3
    assert results["self_refine"]["total_calls"] == 2
    assert results["direct"]["total_calls"] == 1
    assert results["multi_agent"]["avg_tokens"] > results["direct"]["avg_tokens"]


def test_decode_completion_skips_prompt_tokens():
    """Only newly generated tokens are decoded from the generate contract."""
    out = FakeOutput([[10, 11, 12, 13, 14]])
    text = decode_completion(out, FakeTokenizer(), prompt_len=2)
    assert text == "12 13 14"


def test_decode_completion_handles_plain_rows():
    """A plain list of rows (no ``.tolist``) is also accepted."""
    out = SimpleNamespace(sequences=[[1, 2, 3]])
    text = decode_completion(out, FakeTokenizer(), prompt_len=1)
    assert text == "2 3"


def test_run_strategy_bench_wiring_with_injected_generate():
    """The generation.py dispatch target runs end-to-end without a model.

    Exercises the exact entry point that ``evals/generation.py`` calls when
    ``--strategy-bench`` is set, with ``generate_fn`` injected so no CUDA stack
    is required. Confirms the benchmark integrates with the existing CLI args
    shape and reports the package version under test.
    """
    assert CARTESIA_VERSION, "expected the real cartesia_pytorch package on the path"
    problem = BUILTIN_PROBLEMS[0]
    args = SimpleNamespace(
        strategy_bench=True,
        bench_problems=1,
        strategies="direct,multi_agent",
        agent_format="plaintext",
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
    )

    def always_correct(prompt):
        return "#### " + problem.answer

    results = run_strategy_bench(
        model=None, tokenizer=None, args=args, generate_fn=always_correct
    )
    assert results["direct"]["accuracy"] == 1.0
    assert results["multi_agent"]["total_calls"] == 3
    assert results["direct"]["num_problems"] == 1
