"""O(1) corpus-state injection for Llamba SSM mixers.

Implements the core mechanism of PRECOG (Pre-Computed Context Injection,
arXiv:2608.02560): pre-encode a document corpus offline into the fixed-size,
position-agnostic recurrent hidden state of a State-Space Model and inject the
best-matching state directly at query time, collapsing retrieval-augmented
prefill from ``O(L_context)`` to ``O(1)`` per query.

The mechanism rides on the ``DiscreteMamba2`` inference-cache contract::

    state = {"conv": (B, d_conv, conv_dim), "ssm": (B, n_v_heads, headdim, d_state)}

The recurrent ``ssm`` state is "a complete summary of everything the model has
read" -- exactly the property PRECOG exploits. Copying a pre-encoded per-layer
``ssm`` state into a fresh cache is the ``O(1)`` injection, and flipping the
mixer's ``inject_initial_states`` flag makes the following prefill start from it.

This is a Mode-2 (adapted) port. Substitutions vs. the paper:

* Retrieval of the best-matching corpus state uses a parameter-free cosine
  fingerprint of the per-layer ``ssm`` state instead of the paper's learned
  matcher (see :meth:`CorpusStateStore.retrieve`).
* SMC hierarchical / cognitive-domain memory consolidation is out of scope;
  only flat corpus-state retrieval and single-state injection are provided.
* No benchmark harness lives here; ``evals.precog_state_injection`` measures the
  prefill-cost collapse.
"""

from typing import Dict, List, Tuple

import torch

StateDict = Dict[str, torch.Tensor]


class CorpusState:
    """Snapshot of a mixer stack's recurrent state after reading a corpus.

    Attributes:
        conv: per-layer conv states, keyed by layer index.
        ssm: per-layer SSM recurrent states, keyed by layer index.
    """

    def __init__(self, conv: Dict[int, torch.Tensor], ssm: Dict[int, torch.Tensor]):
        """Store per-layer conv/ssm tensors (cloned defensively on read)."""
        self.conv = dict(conv)
        self.ssm = dict(ssm)
        if set(self.conv) != set(self.ssm):
            raise ValueError("conv and ssm must be keyed by the same layer indices.")

    def __len__(self) -> int:
        """Return the number of layers captured in this snapshot."""
        return len(self.ssm)

    def to(self, dtype=None, device=None) -> "CorpusState":
        """Return a new ``CorpusState`` with every tensor cast to dtype/device."""
        return CorpusState(
            conv={i: t.to(dtype=dtype, device=device) for i, t in self.conv.items()},
            ssm={i: t.to(dtype=dtype, device=device) for i, t in self.ssm.items()},
        )


def snapshot_states(inference_params) -> CorpusState:
    """Read the per-layer ``{conv, ssm}`` cache out of an ``InferenceParams``.

    Works with any object exposing ``key_value_memory_dict`` whose values are the
    ``{"conv", "ssm"}`` dicts produced by
    :meth:`DiscreteMamba2.allocate_inference_cache`.

    Args:
        inference_params: the cache to read from (``InferenceParams``-like).

    Returns:
        A :class:`CorpusState` holding a cloned copy of every layer's state.
    """
    conv: Dict[int, torch.Tensor] = {}
    ssm: Dict[int, torch.Tensor] = {}
    for layer_idx, layer_state in inference_params.key_value_memory_dict.items():
        conv[layer_idx] = layer_state["conv"].clone()
        ssm[layer_idx] = layer_state["ssm"].clone()
    return CorpusState(conv=conv, ssm=ssm)


def ssm_state_shape(config, batch_size: int = 1) -> torch.Size:
    """The recurrent ``ssm`` state shape a ``DiscreteMamba2`` stack allocates.

    Mirrors :meth:`DiscreteMamba2.allocate_inference_cache`:
    ``(batch, n_v_heads, headdim, d_state)`` where
    ``headdim = expand * d_model // n_v_heads``. Reading dims from a
    ``LlambaConfig`` keeps this tied to the model's own contract.

    Args:
        config: a ``LlambaConfig`` (or compatible) exposing ``d_model`` and
            ``ssm_cfg``.
        batch_size: leading batch dimension.

    Returns:
        The ``ssm`` cache tensor shape for one layer.
    """
    ssm_cfg = config.ssm_cfg
    expand = ssm_cfg.get("expand", 1)
    n_v_heads = ssm_cfg["n_v_heads"]
    d_state = ssm_cfg["d_state"]
    headdim = (expand * config.d_model) // n_v_heads
    return torch.Size([batch_size, n_v_heads, headdim, d_state])


class CorpusStateStore:
    """Flat store of pre-encoded corpus states with parameter-free retrieval.

    Each document's per-layer ``ssm`` state is flattened and L2-normalized into a
    single fingerprint; retrieval is max-cosine-similarity (a dot product over the
    normalized fingerprints). This stands in for PRECOG's learned matcher.
    """

    def __init__(self):
        """Initialize an empty store."""
        self._keys: List = []
        self._states: List[CorpusState] = []
        self._fingerprints: List[torch.Tensor] = []

    def __len__(self) -> int:
        """Return the number of encoded corpus documents."""
        return len(self._keys)

    @staticmethod
    def fingerprint(state: CorpusState) -> torch.Tensor:
        """Flatten the per-layer ``ssm`` states into one L2-normalized vector."""
        vec = torch.cat([state.ssm[i].reshape(-1) for i in sorted(state.ssm)])
        return vec / vec.norm().clamp_min(1e-8)

    def add(self, key, state: CorpusState) -> None:
        """Register a pre-encoded corpus ``state`` under ``key``."""
        self._keys.append(key)
        self._states.append(state)
        self._fingerprints.append(self.fingerprint(state))

    def retrieve(self, query: CorpusState, top_k: int = 1) -> List[Tuple]:
        """Return the ``(key, state)`` pairs whose fingerprint best matches ``query``.

        Args:
            query: a corpus state to match against (e.g. encoded from the query).
            top_k: number of best matches to return.

        Returns:
            Up to ``top_k`` ``(key, CorpusState)`` tuples, best match first.
        """
        if not self._keys:
            return []
        query_fp = self.fingerprint(query)
        sims = torch.stack(self._fingerprints) @ query_fp
        k = min(top_k, len(self._keys))
        best = torch.argsort(sims, descending=True)[:k].tolist()
        return [(self._keys[i], self._states[i]) for i in best]


@torch.inference_mode()
def encode_document(model, inference_params, input_ids) -> CorpusState:
    """Run ``input_ids`` through ``model`` once and snapshot the resulting state.

    The forward pass fills the per-layer recurrent cache; the snapshot then holds
    the model's summary of the document. This is the offline corpus pre-encoding.

    Args:
        model: a ``LlambaLMHeadModel`` (or any module whose ``forward`` accepts
            ``inference_params`` and routes it to the mixer stack).
        inference_params: a freshly allocated cache (see
            :func:`build_inference_params`).
        input_ids: ``(batch, seqlen)`` token ids for one corpus document.

    Returns:
        The encoded :class:`CorpusState`.
    """
    model(input_ids, inference_params=inference_params)
    return snapshot_states(inference_params)


def inject_states(layers, inference_params, state: CorpusState) -> None:
    """Seed a fresh cache with a pre-encoded corpus state (the ``O(1)`` injection).

    Overwrites each layer's ``conv``/``ssm`` cache in place and flips the mixer's
    ``inject_initial_states`` flag so the next prefill starts from the seeded SSM
    state instead of zeros.

    Args:
        layers: the mixer stack, e.g. ``model.backbone.layers``; each layer must
            expose ``.mixer.layer_idx`` and ``.mixer.inject_initial_states``.
        inference_params: the cache to seed (``InferenceParams``-like).
        state: a previously captured :class:`CorpusState`.
    """
    cache = inference_params.key_value_memory_dict
    for layer in layers:
        layer_idx = layer.mixer.layer_idx
        if layer_idx not in state.ssm:
            raise KeyError(f"corpus state is missing layer {layer_idx}.")
        layer_state = cache[layer_idx]
        layer_state["conv"].copy_(state.conv[layer_idx].to(layer_state["conv"].dtype))
        layer_state["ssm"].copy_(state.ssm[layer_idx].to(layer_state["ssm"].dtype))
        # Activate the `initial_states` path in DiscreteMamba2.forward.
        layer.mixer.inject_initial_states = True


def build_inference_params(model, batch_size: int, max_seqlen: int, dtype=None):
    """Allocate an ``InferenceParams`` and wire the model's cache into it.

    Lazily imports ``InferenceParams`` from ``mamba_ssm`` so this module stays
    importable without the CUDA-only dependency.

    Args:
        model: a model exposing ``allocate_inference_cache``.
        batch_size: batch dimension for the cache tensors.
        max_seqlen: maximum sequence length the cache must hold.
        dtype: optional cache dtype; defaults to the mixer's parameter dtype.

    Returns:
        A populated ``InferenceParams`` ready to pass to ``model.forward``.

    Raises:
        ImportError: if ``mamba_ssm`` is not installed.
    """
    try:
        from mamba_ssm.models.mixer_seq_simple import InferenceParams
    except ImportError as exc:  # pragma: no cover - exercised only with mamba_ssm
        raise ImportError(
            "build_inference_params requires mamba_ssm; install it to allocate a cache."
        ) from exc
    params = InferenceParams(
        max_seqlen=max_seqlen,
        max_batch_size=batch_size,
        seqlen_offset=0,
        key_value_memory_dict={},
    )
    params.key_value_memory_dict = model.allocate_inference_cache(
        batch_size, max_seqlen, dtype=dtype
    )
    return params
