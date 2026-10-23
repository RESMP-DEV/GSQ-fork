"""Model-family dispatch and the Qwen4Exp topology contract.

This module is intentionally free of the heavy GSQ runtime imports (torch,
transformers, accelerate): the dispatch decision and the Qwen4Exp layer
topology are pure data so they can be verified on a host without a GPU stack.
The wrapper that consumes the contract lives in ``src.models.qwen4_exp``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# ── Model families ────────────────────────────────────────────────────────

FAMILY_LLAMA = "llama"
FAMILY_QWEN35 = "qwen35"
FAMILY_QWEN3 = "qwen3"
FAMILY_QWEN4_EXP = "qwen4_exp"
FAMILY_KIMI_K25 = "kimi_k25"
FAMILY_KIMI = "kimi"
FAMILY_GEMMA4 = "gemma4"

# ``model_type`` strings that name the Qwen4Exp architecture. The list is a
# convenience for readers; ``is_qwen4_exp_model_type`` also accepts related
# spellings (``qwen4exp``, ``qwen4_exp_moe``, ``qwen4_moe_exp``).
QWEN4_EXP_MODEL_TYPES = frozenset(
    {
        "qwen4_exp",
        "qwen4exp",
        "qwen4_exp_moe",
        "qwen4_moe_exp",
    }
)


# ``model_type`` values that identify a family when the checkpoint path says
# nothing. The mapping is explicit: an unknown ``model_type`` raises instead
# of being routed by a substring match.
CONFIG_MODEL_TYPE_FAMILIES = {
    "qwen3": FAMILY_QWEN3,
    "qwen3_moe": FAMILY_QWEN3,
    "qwen3_next": FAMILY_QWEN3,
    "qwen35": FAMILY_QWEN35,
    "qwen35_moe": FAMILY_QWEN35,
    "qwen36": FAMILY_QWEN35,
    "qwen36_moe": FAMILY_QWEN35,
    "llama": FAMILY_LLAMA,
    "llama2": FAMILY_LLAMA,
    "llama3": FAMILY_LLAMA,
    "kimi_k2": FAMILY_KIMI,
    "kimi_k25": FAMILY_KIMI_K25,
    "kimi_k2_5": FAMILY_KIMI_K25,
    "gemma4": FAMILY_GEMMA4,
}


def normalize_model_type(model_type: Any) -> str:
    """Normalize a config ``model_type`` string for comparison."""
    if not isinstance(model_type, str):
        return ""
    return model_type.strip().lower().replace("-", "_").replace(".", "_")


def is_qwen4_exp_model_type(model_type: Any) -> bool:
    """Return True when *model_type* names the Qwen4Exp architecture."""
    normalized = normalize_model_type(model_type)
    if not normalized:
        return False
    if normalized in QWEN4_EXP_MODEL_TYPES:
        return True
    return normalized.startswith("qwen4") and ("exp" in normalized or "expert" in normalized)


def is_qwen4_model_type(model_type: Any) -> bool:
    """Return True for any Qwen4-family model type (Exp or not)."""
    return normalize_model_type(model_type).startswith("qwen4")


def config_model_types(config: Any) -> tuple[str, ...]:
    """Return the ``model_type`` strings of *config* and its nested text config.

    Qwen4Exp checkpoints are multimodal-style: the decoder stack is described
    by ``config.text_config`` while ``config.model_type`` names the outer
    architecture. Both are inspected so a nested Qwen4Exp decoder is still
    detected when the outer type is generic.
    """
    if config is None:
        return ()
    collected: list[str] = []
    for candidate in (config, getattr(config, "text_config", None)):
        model_type = getattr(candidate, "model_type", None)
        if isinstance(model_type, str) and model_type and model_type not in collected:
            collected.append(model_type)
    return tuple(collected)


def classify_model(model_name: str, config: Any = None) -> str:
    """Classify *model_name* into one of the FAMILY_* families.

    The checkpoint's own ``model_type`` is consulted *before* any name-based
    routing, so a Qwen4Exp checkpoint is never silently served by the generic
    Qwen3 wrappers. Unknown Qwen4 variants and Qwen4-shaped names without a
    confirming config raise instead of falling through to a generic family.
    """
    name_lower = (model_name or "").lower()
    model_types = config_model_types(config)

    # 1) The Qwen4 family is decided by config first: fail closed on anything
    #    that is not the wired-up Qwen4Exp topology.
    for model_type in model_types:
        if is_qwen4_exp_model_type(model_type):
            return FAMILY_QWEN4_EXP
    for model_type in model_types:
        if is_qwen4_model_type(model_type):
            raise ValueError(
                f"Unsupported Qwen4 checkpoint: model_type '{model_type}' "
                f"({model_name!r}). GSQ wires up the Qwen4Exp topology only; "
                "refusing to dispatch this checkpoint to a generic wrapper."
            )

    # 2) A Qwen4-shaped name without a confirming model_type is ambiguous and
    #    must not fall through to the generic Qwen3 wrappers.
    if "qwen4" in name_lower:
        raise ValueError(
            f"Checkpoint name {model_name!r} looks like the Qwen4 family but the "
            "config does not report a Qwen4Exp model_type. Refusing to dispatch "
            "it to the generic Qwen3 wrappers."
        )

    # 3) Legacy name-based routing (order preserved from main.py).
    if "llama" in name_lower:
        return FAMILY_LLAMA
    if "qwen3.5" in name_lower or "qwen3.6" in name_lower:
        return FAMILY_QWEN35
    if "qwen3" in name_lower:
        return FAMILY_QWEN3
    if "k2.5" in name_lower or "k2_5" in name_lower:
        return FAMILY_KIMI_K25
    if "kimi" in name_lower:
        return FAMILY_KIMI
    if "gemma-4-31b" in name_lower:
        return FAMILY_GEMMA4

    # 4) Config-driven fallback for checkpoints whose path says nothing. The
    #    table is explicit on purpose: a substring match could route an
    #    architecture variant into a wrapper that does not implement it.
    for model_type in model_types:
        family = CONFIG_MODEL_TYPE_FAMILIES.get(normalize_model_type(model_type))
        if family is not None:
            return family

    raise ValueError(f"Unsupported model family: {model_name}")


# ── Qwen4Exp topology contract ────────────────────────────────────────────

# Checkpoint-side prefixes that may hold the Qwen4Exp decoder stack. A
# multimodal Qwen4Exp export nests it under ``model.language_model``; a
# text-only export keeps it at ``model``.
QWEN4EXP_LAYER_PREFIX_CANDIDATES = (
    "model.language_model.layers",
    "model.layers",
)

# Attention sub-modules that may carry a layer's attention weights.
QWEN4EXP_ATTENTION_MODULES = ("self_attn", "linear_attn")

# Router, shared expert (PLE) and fused routed experts of a MoE layer.
QWEN4EXP_ROUTER_MODULE = "mlp.gate"
QWEN4EXP_SHARED_EXPERT_MODULE = "mlp.shared_expert"
QWEN4EXP_ROUTED_EXPERT_MODULE = "mlp.experts"

# Fused 3D parameters that hold the routed experts of a routed layer.
QWEN4EXP_FUSED_EXPERT_PARAMS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)

# Trailing components that identify a parameter *file* rather than a module.
_PARAMETER_LEAF_NAMES = frozenset(
    {
        "weight",
        "bias",
        "scale",
        "inv_freq",
        "e_score_correction_bias",
        "A_log",
        "dt_bias",
    }
)

_GROUP_NAMES = (
    "attention",
    "norms",
    "router",
    "shared_expert",
    "routed_experts",
    "extra",
)


def _classify_layer_name(relative_name: str) -> str:
    """Classify a layer-relative checkpoint name into a placement group."""
    if relative_name == QWEN4EXP_ROUTER_MODULE or relative_name.startswith(
        QWEN4EXP_ROUTER_MODULE + "."
    ):
        return "router"
    if relative_name.startswith(QWEN4EXP_SHARED_EXPERT_MODULE):
        # Covers ``shared_expert``, ``shared_experts`` and the
        # ``shared_expert_gate`` of a PLE layer: they move together.
        return "shared_expert"
    if relative_name.startswith(QWEN4EXP_ROUTED_EXPERT_MODULE):
        return "routed_experts"
    head = relative_name.split(".", 1)[0]
    if head in QWEN4EXP_ATTENTION_MODULES:
        return "attention"
    if head == "norm" or head.endswith("norm"):
        return "norms"
    return "extra"


def group_layer_names(layer_prefix: str, names: Iterable[str]) -> dict[int, list[str]]:
    """Group checkpoint parameter names by decoder layer index."""
    marker = f"{layer_prefix}."
    grouped: dict[int, list[str]] = {}
    for name in names:
        if not name.startswith(marker):
            continue
        head = name[len(marker) :].split(".", 1)[0]
        if head.isdigit():
            grouped.setdefault(int(head), []).append(name)
    return grouped


def module_path_of(name: str) -> str:
    """Return the module path owning the parameter *name*."""
    parent, _, leaf = name.rpartition(".")
    if parent and leaf in _PARAMETER_LEAF_NAMES:
        return parent
    return name


@dataclass(frozen=True)
class Qwen4ExpLayerTopology:
    """Placement groups of a single Qwen4Exp decoder layer.

    Every tuple holds exact checkpoint-side parameter names so placement and
    export can be resolved without guessing whether a name is a module or a
    parameter file. ``extra`` collects the hyper-connection and otherwise
    unclassified parameters of the layer.
    """

    layer_idx: int
    prefix: str
    param_names: tuple[str, ...] = ()
    attention: tuple[str, ...] = ()
    norms: tuple[str, ...] = ()
    router: tuple[str, ...] = ()
    shared_expert: tuple[str, ...] = ()
    routed_experts: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    fused_expert_params: tuple[str, ...] = ()

    @property
    def has_routed_experts(self) -> bool:
        return bool(self.routed_experts)

    @property
    def has_shared_expert(self) -> bool:
        return bool(self.shared_expert)

    @property
    def has_extra(self) -> bool:
        """True when the layer carries parameters outside the known groups.

        Such parameters are the hyper-connection vectors of the Qwen4Exp
        residual structure: they move with the layer, but their forward
        semantics are not modeled yet.
        """
        return bool(self.extra)

    @property
    def is_empty(self) -> bool:
        """True when no checkpoint parameter could be attributed to the layer."""
        return not self.param_names

    @property
    def routed_expert_param_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.param_names
            if name.startswith(f"{self.prefix}.{QWEN4EXP_ROUTED_EXPERT_MODULE}")
        )

    @property
    def non_expert_param_names(self) -> tuple[str, ...]:
        routed = set(self.routed_expert_param_names)
        return tuple(name for name in self.param_names if name not in routed)


def build_qwen4exp_layer_topology(
    layer_idx: int, layer_prefix: str, names: Iterable[str]
) -> Qwen4ExpLayerTopology:
    """Build the placement topology of one Qwen4Exp decoder layer.

    *names* is any iterable of checkpoint-side parameter names; only the ones
    that belong to this layer are used.
    """
    prefix = f"{layer_prefix}.{layer_idx}"
    marker = prefix + "."
    layer_names = [name for name in names if name.startswith(marker)]

    buckets: dict[str, list[str]] = {group: [] for group in _GROUP_NAMES}
    for name in layer_names:
        relative = name[len(marker) :]
        buckets[_classify_layer_name(relative)].append(name)

    routed = buckets["routed_experts"]
    fused = tuple(
        name for name in (f"{prefix}.{param}" for param in QWEN4EXP_FUSED_EXPERT_PARAMS) if routed
    )

    return Qwen4ExpLayerTopology(
        layer_idx=layer_idx,
        prefix=prefix,
        param_names=tuple(sorted(layer_names)),
        attention=tuple(buckets["attention"]),
        norms=tuple(buckets["norms"]),
        router=tuple(buckets["router"]),
        shared_expert=tuple(buckets["shared_expert"]),
        routed_experts=tuple(routed),
        extra=tuple(buckets["extra"]),
        fused_expert_params=fused,
    )


def build_qwen4exp_topology(
    layer_prefix: str, names: Iterable[str], num_layers: int
) -> dict[int, Qwen4ExpLayerTopology]:
    """Build the placement topology of every Qwen4Exp decoder layer."""
    names = list(names)
    topology: dict[int, Qwen4ExpLayerTopology] = {}
    for layer_idx in range(num_layers):
        topology[layer_idx] = build_qwen4exp_layer_topology(layer_idx, layer_prefix, names)
    return topology


def resolve_layer_prefix(
    checkpoint_names: Iterable[str] = (),
    model_names: Iterable[str] = (),
) -> str:
    """Resolve the checkpoint-side decoder-layer prefix of a Qwen4Exp model.

    The checkpoint namespace is preferred; the instantiated model namespace is
    the fallback for dummy (shape-only) runs where no checkpoint index exists.
    """
    checkpoint = list(checkpoint_names)
    for candidate in QWEN4EXP_LAYER_PREFIX_CANDIDATES:
        marker = candidate + "."
        if any(name.startswith(marker) for name in checkpoint):
            return candidate
    model = list(model_names)
    for candidate in QWEN4EXP_LAYER_PREFIX_CANDIDATES:
        model_side = candidate.replace(".language_model", "")
        marker = model_side + "."
        if any(name.startswith(marker) for name in model):
            # Fall back to the namespace the model itself uses: dummy
            # (shape-only) runs have no checkpoint index to match against.
            return model_side
    return QWEN4EXP_LAYER_PREFIX_CANDIDATES[0]


# ── Fail-closed helpers ───────────────────────────────────────────────────


def qwen4exp_not_implemented(what: str) -> NotImplementedError:
    """Return the error used for every unimplemented Qwen4Exp boundary.

    GSQ support for the Qwen4Exp architecture covers checkpoint dispatch and
    weight placement only. Anything that would quantize or export the fused
    per-expert weights must fail loudly instead of writing a silently wrong
    checkpoint.
    """
    return NotImplementedError(
        f"Qwen4Exp: {what} is not implemented. GSQ support for the Qwen4Exp "
        "architecture covers checkpoint dispatch and weight placement only; no "
        "quantized Qwen4Exp checkpoint is produced."
    )


__all__ = [
    "CONFIG_MODEL_TYPE_FAMILIES",
    "FAMILY_GEMMA4",
    "FAMILY_KIMI",
    "FAMILY_KIMI_K25",
    "FAMILY_LLAMA",
    "FAMILY_QWEN3",
    "FAMILY_QWEN35",
    "FAMILY_QWEN4_EXP",
    "QWEN4_EXP_MODEL_TYPES",
    "QWEN4EXP_ATTENTION_MODULES",
    "QWEN4EXP_FUSED_EXPERT_PARAMS",
    "QWEN4EXP_LAYER_PREFIX_CANDIDATES",
    "QWEN4EXP_ROUTED_EXPERT_MODULE",
    "QWEN4EXP_ROUTER_MODULE",
    "QWEN4EXP_SHARED_EXPERT_MODULE",
    "Qwen4ExpLayerTopology",
    "build_qwen4exp_layer_topology",
    "build_qwen4exp_topology",
    "classify_model",
    "config_model_types",
    "group_layer_names",
    "is_qwen4_exp_model_type",
    "is_qwen4_model_type",
    "module_path_of",
    "normalize_model_type",
    "qwen4exp_not_implemented",
    "resolve_layer_prefix",
]
