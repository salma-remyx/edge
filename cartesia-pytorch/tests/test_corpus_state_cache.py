"""Tests for ``corpus_state_cache`` (PRECOG O(1) SSM state injection).

The modules under ``cartesia_pytorch.Llamba`` cannot be imported via the package
path in environments lacking the CUDA-only ``mamba_ssm`` dependency (their
``__init__`` pulls it in), so the capability module and the existing
``LlambaConfig`` are loaded directly by file path. The full real-mixer path is
covered by a test that is skipped when ``mamba_ssm`` is unavailable.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_LLAMBA_DIR = _HERE.parent / "cartesia_pytorch" / "Llamba"
_MIXERS_DIR = _LLAMBA_DIR / "mixers"


def _load_module(path: Path, name: str):
    """Load a Python module directly from a file path, bypassing package init."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csc = _load_module(_MIXERS_DIR / "corpus_state_cache.py", "corpus_state_cache")
LlambaConfig = _load_module(
    _LLAMBA_DIR / "configuration_llamba.py", "configuration_llamba"
).LlambaConfig


def _make_state(seed: int, layers=2):
    """Build a deterministic CorpusState with random per-layer conv/ssm tensors."""
    gen = torch.Generator().manual_seed(seed)
    conv = {i: torch.randn(1, 4, 8, generator=gen) for i in range(layers)}
    ssm = {i: torch.randn(1, 4, 8, 4, generator=gen) for i in range(layers)}
    return csc.CorpusState(conv=conv, ssm=ssm)


class _FakeMixer:
    """Mimics the ``DiscreteMamba2`` attributes the injection touches."""

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.inject_initial_states = False


class _FakeLayer:
    """Wraps a mixer like ``Block`` does (``layer.mixer``)."""

    def __init__(self, layer_idx):
        self.mixer = _FakeMixer(layer_idx)


class _FakeInferenceParams:
    """Mimics the ``key_value_memory_dict`` surface of ``InferenceParams``."""

    def __init__(self, cache):
        self.key_value_memory_dict = cache


def test_fingerprint_is_a_unit_vector():
    """The retrieval fingerprint is L2-normalized over the flattened ssm state."""
    state = csc.CorpusState(conv={0: torch.zeros(1, 4, 8)}, ssm={0: torch.ones(1, 4, 8, 4)})
    fp = csc.CorpusStateStore.fingerprint(state)
    assert fp.shape[0] == 1 * 4 * 8 * 4
    assert torch.allclose(fp.norm(), torch.tensor(1.0), atol=1e-6)


def test_retrieve_returns_closest_corpus_state():
    """Cosine retrieval surfaces the corpus document matching the query state."""
    store = csc.CorpusStateStore()
    store.add("alpha", _make_state(seed=0))
    store.add("beta", _make_state(seed=1))
    query = _make_state(seed=0)  # identical fingerprints -> cosine 1.0
    (key, _), *_ = store.retrieve(query, top_k=1)
    assert key == "alpha"


def test_inject_states_seeds_cache_and_flips_flag():
    """Injection overwrites the cache in place and enables the mixer flag."""
    layers = [_FakeLayer(0), _FakeLayer(1)]
    ip = _FakeInferenceParams(
        {
            0: {"conv": torch.zeros(1, 4, 8), "ssm": torch.zeros(1, 4, 8, 4)},
            1: {"conv": torch.zeros(1, 4, 8), "ssm": torch.zeros(1, 4, 8, 4)},
        }
    )
    state = csc.CorpusState(
        conv={0: torch.full((1, 4, 8), 3.0), 1: torch.full((1, 4, 8), 5.0)},
        ssm={0: torch.full((1, 4, 8, 4), 7.0), 1: torch.full((1, 4, 8, 4), 9.0)},
    )
    csc.inject_states(layers, ip, state)
    assert torch.all(ip.key_value_memory_dict[0]["ssm"] == 7.0)
    assert torch.all(ip.key_value_memory_dict[1]["conv"] == 5.0)
    assert all(layer.mixer.inject_initial_states for layer in layers)


def test_snapshot_then_inject_restores_state():
    """A snapshot round-trips through inject onto a fresh cache unchanged."""
    layers = [_FakeLayer(0)]
    src = _FakeInferenceParams({0: {"conv": torch.randn(1, 4, 8), "ssm": torch.randn(1, 4, 8, 4)}})
    state = csc.snapshot_states(src)
    dst = _FakeInferenceParams({0: {"conv": torch.zeros(1, 4, 8), "ssm": torch.zeros(1, 4, 8, 4)}})
    csc.inject_states(layers, dst, state)
    assert torch.allclose(dst.key_value_memory_dict[0]["ssm"], src.key_value_memory_dict[0]["ssm"])
    assert torch.allclose(
        dst.key_value_memory_dict[0]["conv"], src.key_value_memory_dict[0]["conv"]
    )


def test_ssm_state_shape_matches_config():
    """The derived ssm shape matches the DiscreteMamba2 contract for a LlambaConfig."""
    config = LlambaConfig(
        vocab_size=32,
        d_model=16,
        n_layer=2,
        ssm_cfg={
            "d_state": 64,
            "n_v_heads": 8,
            "n_qk_heads": 8,
            "expand": 2,
            "chunk_size": 16,
            "activation": "identity",
            "bias": False,
        },
    )
    # headdim = expand * d_model // n_v_heads = 2 * 16 // 8 = 4
    assert csc.ssm_state_shape(config, batch_size=1) == torch.Size([1, 8, 4, 64])


def test_end_to_end_with_real_mixer():
    """The real DiscreteMamba2 exposes the wiring flag and accepts injection.

    Skipped when the CUDA-only ``mamba_ssm``/``causal_conv1d`` deps are absent.
    """
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("causal_conv1d")
    from cartesia_pytorch.Llamba.mixers.discrete_mamba2 import DiscreteMamba2

    mixer = DiscreteMamba2(
        d_model=16,
        layer_idx=0,
        d_state=16,
        n_v_heads=8,
        n_qk_heads=8,
        d_conv=4,
        expand=2,
        chunk_size=16,
    ).eval()
    assert mixer.inject_initial_states is False  # default; wiring edit is present
    cache = mixer.allocate_inference_cache(batch_size=1, max_seqlen=8)
    layer = _FakeLayer(0)
    layer.mixer = mixer
    state = csc.CorpusState(
        conv={0: torch.full_like(cache["conv"], 1.5)},
        ssm={0: torch.full_like(cache["ssm"], 2.5)},
    )
    csc.inject_states([layer], _FakeInferenceParams({0: cache}), state)
    assert mixer.inject_initial_states is True
    assert torch.all(cache["ssm"] == 2.5)
