"""Host-safe checks for the GSQ Qwen4Exp dispatch and topology contract.

Two layers are covered:

* the pure contract in ``src/models/__init__.py`` (family classification, the
  Qwen4Exp layer topology, the fail-closed helpers). That module deliberately
  avoids the heavy GSQ runtime imports, so these checks run anywhere.
* the wrapper in ``src/models/qwen4_exp.py`` needs torch plus the GSQ runtime
  dependencies (accelerate, compressed-tensors, datasets). Those checks are
  skipped when the runtime is not installed.

Run:
    python tests/test_qwen4_exp_wrapper.py
    # or
    pytest tests/test_qwen4_exp_wrapper.py
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_contract():
    """Import the dispatch/topology contract without the GSQ runtime stack.

    ``src/__init__.py`` imports the data pipeline, which pulls in ``datasets``.
    The contract does not need it, so fall back to loading ``src/models`` from
    its file when the package import is unavailable.
    """
    try:
        import src.models as contract  # noqa: PLC0415

        return contract
    except Exception:  # pragma: no cover - depends on the host environment
        spec = importlib.util.spec_from_file_location(
            "gsq_qwen4exp_models_contract",
            REPO_ROOT / "src" / "models" / "__init__.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve __module__ here
        spec.loader.exec_module(module)
        return module


CONTRACT = _load_contract()

_RUNTIME_MODULES = (
    "torch",
    "accelerate",
    "compressed_tensors",
    "datasets",
    "safetensors",
)

try:  # pytest is optional: the file also runs as a plain script.
    import pytest
except ImportError:  # pragma: no cover - bare host
    pytest = None


def _runtime_guard():
    """Skip a runtime-gated check, with or without pytest."""
    modules = ", ".join(_RUNTIME_MODULES)
    message = f"GSQ runtime dependencies ({modules}) are not installed"
    if pytest is None:
        print(f"skip: {message}")
        return False
    pytest.skip(message)


# ── helpers ───────────────────────────────────────────────────────────────


def _assert_raises(exc_type, func, *args, **kwargs):
    """Assert that *func* raises *exc_type*; works without pytest."""
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        return exc
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


def _config(model_type, text_model_type=None):
    config = SimpleNamespace(model_type=model_type)
    if text_model_type is not None:
        config.text_config = SimpleNamespace(model_type=text_model_type)
    return config


LAYER_PREFIX = "model.language_model.layers"


def _layer_names(layer_idx=0, prefix=LAYER_PREFIX):
    """Checkpoint parameter names of a Qwen4Exp decoder layer."""
    base = f"{prefix}.{layer_idx}"
    return [
        f"{base}.input_layernorm.weight",
        f"{base}.post_attention_layernorm.weight",
        f"{base}.self_attn.q_proj.weight",
        f"{base}.self_attn.k_proj.weight",
        f"{base}.self_attn.v_proj.weight",
        f"{base}.self_attn.o_proj.weight",
        f"{base}.self_attn.q_norm.weight",
        f"{base}.self_attn.k_norm.weight",
        f"{base}.mlp.gate.weight",
        f"{base}.mlp.gate.e_score_correction_bias",
        f"{base}.mlp.shared_expert.gate_proj.weight",
        f"{base}.mlp.shared_expert.up_proj.weight",
        f"{base}.mlp.shared_expert.down_proj.weight",
        f"{base}.mlp.shared_expert_gate.weight",
        f"{base}.mlp.experts.gate_up_proj",
        f"{base}.mlp.experts.down_proj",
        f"{base}.hyper_connection.attn_scale",
        f"{base}.hyper_connection.mlp_scale",
    ]


# ── dispatch ──────────────────────────────────────────────────────────────


def test_qwen4_exp_model_type_dispatches_to_the_qwen4_exp_family():
    family = CONTRACT.classify_model("Qwen/Qwen4-Exp-30B-A3B", _config("qwen4_exp"))
    assert family == CONTRACT.FAMILY_QWEN4_EXP, family


def test_generic_qwen3_dispatch_cannot_silently_select_qwen4_exp():
    """The config decides the family before the generic Qwen3 branch."""
    for name in (
        "qwen3-mirror/qwen4_exp",
        "Qwen/Qwen3.5-Exp",
        "/models/qwen3-qwen4exp",
    ):
        family = CONTRACT.classify_model(name, _config("qwen4_exp"))
        assert family == CONTRACT.FAMILY_QWEN4_EXP, (name, family)
        assert family != CONTRACT.FAMILY_QWEN3, name


def test_nested_text_config_model_type_is_inspected():
    family = CONTRACT.classify_model("Qwen/Qwen4-Exp-Omni", _config("qwen4_exp_omni", "qwen4_exp"))
    assert family == CONTRACT.FAMILY_QWEN4_EXP, family


def test_unknown_qwen4_variant_fails_closed():
    _assert_raises(
        ValueError,
        CONTRACT.classify_model,
        "Qwen/Qwen4-Next",
        _config("qwen4_next"),
    )


def test_qwen4_name_without_a_confirming_config_fails_closed():
    _assert_raises(ValueError, CONTRACT.classify_model, "/models/qwen4_exp", None)


def test_qwen4_name_conflicting_with_the_config_fails_closed():
    _assert_raises(
        ValueError,
        CONTRACT.classify_model,
        "/models/qwen4_exp",
        _config("qwen3_moe"),
    )


def test_qwen3_family_still_dispatches_to_the_generic_wrappers():
    assert (
        CONTRACT.classify_model("Qwen/Qwen3-30B-A3B", _config("qwen3_moe")) == CONTRACT.FAMILY_QWEN3
    )
    assert CONTRACT.classify_model("Qwen/Qwen3-8B", _config("qwen3")) == CONTRACT.FAMILY_QWEN3


def test_legacy_name_dispatch_is_preserved():
    cases = {
        "meta-llama/Llama-3.1-8B": CONTRACT.FAMILY_LLAMA,
        "Qwen/Qwen3.5-35B-A3B": CONTRACT.FAMILY_QWEN35,
        "Qwen/Qwen3.6-35B": CONTRACT.FAMILY_QWEN35,
        "moonshotai/Kimi-K2-Instruct": CONTRACT.FAMILY_KIMI,
        "moonshotai/Kimi-K2.5-Instruct": CONTRACT.FAMILY_KIMI_K25,
        "moonshotai/Kimi-K2_5-Instruct": CONTRACT.FAMILY_KIMI_K25,
        "google/gemma-4-31b-it": CONTRACT.FAMILY_GEMMA4,
    }
    for name, expected in cases.items():
        family = CONTRACT.classify_model(name, None)
        assert family == expected, (name, family, expected)


def test_config_drives_a_checkpoint_whose_name_says_nothing():
    family = CONTRACT.classify_model("/models/exp-30b", _config("qwen4_exp"))
    assert family == CONTRACT.FAMILY_QWEN4_EXP, family


def test_unsupported_family_still_raises():
    _assert_raises(ValueError, CONTRACT.classify_model, "/models/mystery", None)


def test_unknown_model_type_is_not_routed_by_a_substring_match():
    """An unrecognized model_type raises instead of matching a family loosely."""
    _assert_raises(ValueError, CONTRACT.classify_model, "/models/mystery", _config("llama4"))
    _assert_raises(ValueError, CONTRACT.classify_model, "/models/mystery", _config("qwen3_next_v2"))


def test_known_model_types_drive_a_checkpoint_without_a_name_hint():
    assert CONTRACT.classify_model("/models/weights", _config("qwen3_moe")) == CONTRACT.FAMILY_QWEN3
    assert (
        CONTRACT.classify_model("/models/weights", _config("qwen35_moe")) == CONTRACT.FAMILY_QWEN35
    )


def test_main_routes_qwen4_exp_before_the_generic_qwen3_branch():
    """The wiring in main.py must reach the Qwen4Exp wrapper first."""
    tree = ast.parse((REPO_ROOT / "main.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model_wrapper"
    )
    branches = []
    node = next((n for n in function.body if isinstance(n, ast.If)), None)
    while node is not None:
        branches.append((ast.unparse(node.test), ast.unparse(node)))
        orelse = node.orelse
        node = orelse[0] if orelse and isinstance(orelse[0], ast.If) else None
    qwen4_index = next(
        (i for i, (test, _) in enumerate(branches) if "FAMILY_QWEN4_EXP" in test),
        None,
    )
    qwen3_index = next(
        (i for i, (test, _) in enumerate(branches) if "FAMILY_QWEN3" in test),
        None,
    )
    assert qwen4_index is not None, "no Qwen4Exp branch in get_model_wrapper"
    assert qwen3_index is not None, "no Qwen3 branch in get_model_wrapper"
    assert qwen4_index < qwen3_index, branches
    assert "Qwen4ExpWrapper" in branches[qwen4_index][1], branches[qwen4_index]


# ── topology contract ─────────────────────────────────────────────────────


def test_topology_groups_attention_norms_router_and_experts():
    names = _layer_names()
    topology = CONTRACT.build_qwen4exp_layer_topology(0, LAYER_PREFIX, names)
    base = f"{LAYER_PREFIX}.0"

    assert set(topology.attention) == {
        f"{base}.self_attn.q_proj.weight",
        f"{base}.self_attn.k_proj.weight",
        f"{base}.self_attn.v_proj.weight",
        f"{base}.self_attn.o_proj.weight",
        f"{base}.self_attn.q_norm.weight",
        f"{base}.self_attn.k_norm.weight",
    }
    assert set(topology.norms) == {
        f"{base}.input_layernorm.weight",
        f"{base}.post_attention_layernorm.weight",
    }
    assert set(topology.router) == {
        f"{base}.mlp.gate.weight",
        f"{base}.mlp.gate.e_score_correction_bias",
    }
    assert set(topology.shared_expert) == {
        f"{base}.mlp.shared_expert.gate_proj.weight",
        f"{base}.mlp.shared_expert.up_proj.weight",
        f"{base}.mlp.shared_expert.down_proj.weight",
        f"{base}.mlp.shared_expert_gate.weight",
    }
    assert set(topology.routed_experts) == {
        f"{base}.mlp.experts.gate_up_proj",
        f"{base}.mlp.experts.down_proj",
    }
    assert set(topology.extra) == {
        f"{base}.hyper_connection.attn_scale",
        f"{base}.hyper_connection.mlp_scale",
    }
    assert topology.has_routed_experts
    assert topology.has_shared_expert
    assert topology.has_extra
    assert not topology.is_empty


def test_topology_names_the_fused_3d_expert_parameters():
    topology = CONTRACT.build_qwen4exp_layer_topology(0, LAYER_PREFIX, _layer_names())
    base = f"{LAYER_PREFIX}.0"
    assert topology.fused_expert_params == (
        f"{base}.mlp.experts.gate_up_proj",
        f"{base}.mlp.experts.down_proj",
    )


def test_topology_separates_the_routed_expert_parameters():
    topology = CONTRACT.build_qwen4exp_layer_topology(0, LAYER_PREFIX, _layer_names())
    base = f"{LAYER_PREFIX}.0"
    assert set(topology.routed_expert_param_names) == {
        f"{base}.mlp.experts.gate_up_proj",
        f"{base}.mlp.experts.down_proj",
    }
    assert f"{base}.mlp.experts.gate_up_proj" not in topology.non_expert_param_names
    assert f"{base}.mlp.shared_expert.gate_proj.weight" in (topology.non_expert_param_names)


def test_topology_supports_a_per_expert_checkpoint_layout():
    """A checkpoint that stores per-expert 2D weights still names the 3D tensors."""
    base = f"{LAYER_PREFIX}.3"
    names = [
        f"{base}.mlp.experts.0.gate_proj.weight",
        f"{base}.mlp.experts.0.up_proj.weight",
        f"{base}.mlp.experts.0.down_proj.weight",
        f"{base}.mlp.experts.1.gate_proj.weight",
        f"{base}.mlp.experts.1.up_proj.weight",
        f"{base}.mlp.experts.1.down_proj.weight",
    ]
    topology = CONTRACT.build_qwen4exp_layer_topology(3, LAYER_PREFIX, names)
    assert len(topology.routed_experts) == 6
    assert topology.fused_expert_params == (
        f"{base}.mlp.experts.gate_up_proj",
        f"{base}.mlp.experts.down_proj",
    )


def test_topology_marks_a_ple_layer_without_routed_experts():
    """A PLE layer that only carries a shared expert has no fused experts."""
    base = f"{LAYER_PREFIX}.1"
    names = [
        f"{base}.mlp.shared_expert.gate_proj.weight",
        f"{base}.mlp.shared_expert.up_proj.weight",
        f"{base}.mlp.shared_expert.down_proj.weight",
    ]
    topology = CONTRACT.build_qwen4exp_layer_topology(1, LAYER_PREFIX, names)
    assert topology.has_shared_expert
    assert not topology.has_routed_experts
    assert topology.fused_expert_params == ()


def test_topology_covers_every_layer():
    names = _layer_names(0) + _layer_names(1)
    topology = CONTRACT.build_qwen4exp_topology(LAYER_PREFIX, names, 2)
    assert sorted(topology) == [0, 1]
    assert not topology[0].is_empty
    assert not topology[1].is_empty


def test_layer_prefix_resolution_prefers_the_checkpoint_namespace():
    checkpoint_names = _layer_names(0)
    assert CONTRACT.resolve_layer_prefix(checkpoint_names, []) == LAYER_PREFIX
    model_names = ["model.layers.0.self_attn.q_proj.weight"]
    assert CONTRACT.resolve_layer_prefix([], model_names) == "model.layers"
    assert CONTRACT.resolve_layer_prefix([], []) == CONTRACT.QWEN4EXP_LAYER_PREFIX_CANDIDATES[0]


# ── dispatch through main.py ──────────────────────────────────────────────


def test_main_dispatch_sends_a_qwen4_exp_config_to_the_qwen4_exp_family():
    main_module = _import_main()
    calls = []

    def fake_loader(model_name):
        calls.append(model_name)
        return _config("qwen4_exp")

    original = main_module.load_dispatch_config
    main_module.load_dispatch_config = fake_loader
    try:
        family, _ = main_module.resolve_model_family("Qwen/Qwen3-Exp-mirror")
        assert family == CONTRACT.FAMILY_QWEN4_EXP, family
        assert calls == ["Qwen/Qwen3-Exp-mirror"], calls
    finally:
        main_module.load_dispatch_config = original


def test_main_dispatch_keeps_the_legacy_name_routing():
    main_module = _import_main()

    def boom(model_name):
        raise AssertionError(f"the config must not be loaded for {model_name!r}")

    original = main_module.load_dispatch_config
    main_module.load_dispatch_config = boom
    try:
        family, config = main_module.resolve_model_family("meta-llama/Llama-3.1-8B")
        assert family == CONTRACT.FAMILY_LLAMA, family
        assert config is None
    finally:
        main_module.load_dispatch_config = original


def test_main_dispatch_reports_an_unsupported_family():
    main_module = _import_main()

    original = main_module.load_dispatch_config
    main_module.load_dispatch_config = lambda model_name: None
    try:
        _assert_raises(ValueError, main_module.resolve_model_family, "/models/mystery")
    finally:
        main_module.load_dispatch_config = original


def test_dispatch_chains_a_config_load_failure():
    main_module = _import_main()

    def boom(model_name):
        raise ConnectionError("checkpoint config is unavailable")

    original = main_module.load_dispatch_config
    main_module.load_dispatch_config = boom
    try:
        error = _assert_raises(ValueError, main_module.resolve_model_family, "/models/mystery")
    finally:
        main_module.load_dispatch_config = original
    assert "config load failed" in str(error), error
    assert isinstance(error.__cause__, ConnectionError), error.__cause__


# ── fail-closed boundaries ────────────────────────────────────────────────


def test_fail_closed_helper_returns_a_not_implemented_error():
    error = CONTRACT.qwen4exp_not_implemented("a thing")
    assert isinstance(error, NotImplementedError), type(error)
    assert "Qwen4Exp" in str(error)
    assert "a thing" in str(error)


# ── wrapper level ──────────────────────────────────────────────────────────


def _module_missing(name):
    """True when *name* is neither installed nor stubbed yet."""
    if name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return True


def _install_runtime_stubs():
    """Install minimal stand-ins for the absent GSQ runtime dependencies.

    The stand-ins are faithful only for the placement code paths under test
    (module/parameter device moves). They are not a substitute for
    accelerate, compressed-tensors, datasets or lion-pytorch anywhere else,
    and they are never installed when the real package is available.
    """
    import contextlib
    import importlib.machinery
    import types

    import torch

    def _new_module(name):
        # A real __spec__ keeps importlib.util.find_spec() working for the
        # stubbed packages (transformers probes accelerate that way).
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return module

    if _module_missing("accelerate"):
        accelerate = _new_module("accelerate")
        accelerate_utils = _new_module("accelerate.utils")

        def init_empty_weights(*args, **kwargs):
            return contextlib.nullcontext()

        def set_module_tensor_to_device(model, param_name, device, value=None, dtype=None):
            parts = param_name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            leaf = parts[-1]
            old = getattr(module, leaf)
            if value is None:
                value = torch.empty(old.shape, dtype=dtype or old.dtype, device=device)
            else:
                value = value.to(device=device, dtype=dtype or value.dtype)
            setattr(
                module,
                leaf,
                torch.nn.Parameter(value, requires_grad=old.requires_grad),
            )

        accelerate.init_empty_weights = init_empty_weights
        accelerate.utils = accelerate_utils
        accelerate_utils.set_module_tensor_to_device = set_module_tensor_to_device
        sys.modules["accelerate"] = accelerate
        sys.modules["accelerate.utils"] = accelerate_utils

    if _module_missing("compressed_tensors"):
        compressed = _new_module("compressed_tensors")

        class PackedQuantizationCompressor:
            def compress(self, *args, **kwargs):
                raise NotImplementedError("compressed-tensors stub")

            def decompress(self, *args, **kwargs):
                raise NotImplementedError("compressed-tensors stub")

        compressed.PackedQuantizationCompressor = PackedQuantizationCompressor
        sys.modules["compressed_tensors"] = compressed

    if _module_missing("datasets"):
        datasets = _new_module("datasets")

        def load_dataset(*args, **kwargs):
            raise RuntimeError("datasets stub")

        datasets.load_dataset = load_dataset
        sys.modules["datasets"] = datasets

    if _module_missing("wandb"):
        wandb = _new_module("wandb")

        def _unavailable(*args, **kwargs):
            raise RuntimeError("wandb stub")

        wandb.init = _unavailable
        wandb.log = _unavailable
        wandb.finish = _unavailable
        wandb.run = None
        sys.modules["wandb"] = wandb

    if _module_missing("lion_pytorch"):
        lion = _new_module("lion_pytorch")

        class Lion:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("lion-pytorch stub")

        lion.Lion = Lion
        sys.modules["lion_pytorch"] = lion


def _wrapper_class():
    """Return the Qwen4Exp wrapper class.

    Missing runtime dependencies are replaced by stand-ins so the placement
    paths can be exercised on a host without the full GSQ stack. An import
    error that is not caused by a missing dependency is raised, not hidden.
    """
    if _module_missing("torch"):
        return None
    missing = [name for name in _RUNTIME_MODULES if _module_missing(name)]
    if missing:
        _install_runtime_stubs()
    from src.models.qwen4_exp import Qwen4ExpWrapper

    return Qwen4ExpWrapper


def _import_main():
    """Import main.py, installing stand-ins for absent runtime dependencies."""
    for name in ("wandb", "datasets", "lion_pytorch", "accelerate", "compressed_tensors"):
        if _module_missing(name):
            _install_runtime_stubs()
            break
    import main as main_module

    return main_module


def _tiny_wrapper():
    """Build a wrapper instance around tiny stand-in modules."""
    import torch
    from torch import nn

    class TinyExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 4, 3))
            self.down_proj = nn.Parameter(torch.zeros(2, 3, 4))

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(3)
            self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(3, 3)})
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(3, 2)
            self.mlp.experts = TinyExperts()
            self.hyper_scale = nn.Parameter(torch.zeros(3))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyLayer()])

    wrapper_cls = _wrapper_class()
    if wrapper_cls is None:
        return None
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.model = TinyModel()
    wrapper.model.to("meta")
    wrapper.dummy = True
    wrapper.device = "cpu"
    wrapper.dtype = torch.float32
    wrapper.meta_init_std = 0.02
    wrapper.layer_prefix = "model.layers"
    wrapper._model_layer_prefix = "model.layers"
    wrapper.topology = CONTRACT.build_qwen4exp_topology(
        "model.layers",
        [name for name, _ in wrapper.model.named_parameters()],
        1,
    )
    return wrapper


def test_wrapper_constructor_resolves_the_multimodal_topology():
    """The constructor resolves model.language_model.layers and the topology."""
    import torch
    from torch import nn

    _import_main()  # installs the stand-ins for absent runtime dependencies
    from src.models import base as base_module
    from src.models.qwen4_exp import Qwen4ExpWrapper

    class TinyExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 3))
            self.down_proj = nn.Parameter(torch.zeros(4, 3, 8))

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(3, 3)})
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(3, 4)
            self.mlp.experts = TinyExperts()

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

    model = TinyModel()
    checkpoint_names = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "shard.safetensors",
        "model.language_model.layers.0.mlp.experts.down_proj": "shard.safetensors",
        "model.language_model.layers.1.mlp.experts.gate_up_proj": "shard.safetensors",
        "model.language_model.layers.1.mlp.experts.down_proj": "shard.safetensors",
    }

    original_init = base_module.BaseModelWrapper.__init__

    def fake_init(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        self.model = model
        self.model.config = SimpleNamespace(num_experts=4, moe_intermediate_size=8)
        self.dtype = dtype
        self.device = device
        self.dummy = False
        self._name_to_shard = checkpoint_names
        self._single_shard = None

    base_module.BaseModelWrapper.__init__ = fake_init
    try:
        wrapper = Qwen4ExpWrapper("dummy", None, 1, 128, "cpu", torch.float32)
    finally:
        base_module.BaseModelWrapper.__init__ = original_init

    assert wrapper.layer_prefix == "model.language_model.layers"
    assert wrapper._model_layer_prefix == "model.layers"
    assert wrapper.num_layers == 2
    assert wrapper.num_experts == 4
    assert wrapper.is_moe
    assert wrapper.fused_experts
    assert wrapper.fused_expert_intermediate_size == 8
    assert wrapper.get_layer_module(0) is model.model.layers[0]
    assert wrapper.topology[0].fused_expert_params == (
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    )


def test_wrapper_constructor_refuses_a_non_dummy_checkpoint_without_decoder_names():
    import torch

    _import_main()
    from src.models import base as base_module
    from src.models.qwen4_exp import Qwen4ExpWrapper

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])

    original_init = base_module.BaseModelWrapper.__init__

    def fake_init(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        self.model = TinyModel()
        self.model.config = SimpleNamespace(num_experts=4, moe_intermediate_size=8)
        self.dtype = dtype
        self.device = device
        self.dummy = False
        self._name_to_shard = {}
        self._single_shard = None

    base_module.BaseModelWrapper.__init__ = fake_init
    try:
        error = _assert_raises(
            RuntimeError,
            Qwen4ExpWrapper,
            "dummy",
            None,
            1,
            128,
            "cpu",
            torch.float32,
        )
    finally:
        base_module.BaseModelWrapper.__init__ = original_init
    assert "Refusing to guess the topology" in str(error), error


def test_wrapper_layer_prefixes_cover_every_placement_group():
    wrapper = _tiny_wrapper()
    if wrapper is None:
        _runtime_guard()
        return
    topology = wrapper.topology[0]
    prefixes = wrapper._layer_prefixes("model.layers.0")
    assert set(prefixes["non_mlp"]) == set(
        topology.attention + topology.norms + topology.router + topology.extra
    )
    assert set(prefixes["mlp"]) == set(topology.shared_expert + topology.routed_experts)
    assert prefixes["mlp_offload_params"] == list(topology.fused_expert_params)


def test_wrapper_moves_a_layer_between_the_device_and_meta():
    """Fused experts, the shared expert and hyper-connection weights travel."""
    wrapper = _tiny_wrapper()
    if wrapper is None:
        _runtime_guard()
        return
    layer = wrapper.model.model.layers[0]

    assert layer.self_attn["q_proj"].weight.device.type == "meta"
    assert layer.mlp.experts.gate_up_proj.device.type == "meta"
    assert layer.hyper_scale.device.type == "meta"

    wrapper.move_layer_to_gpu("model.layers.0")
    assert layer.self_attn["q_proj"].weight.device.type == "cpu"
    assert layer.mlp.experts.gate_up_proj.device.type == "cpu"
    assert layer.mlp.experts.gate_up_proj.shape == (2, 4, 3)
    assert layer.hyper_scale.device.type == "cpu"

    wrapper.offload_to_meta("model.layers.0")
    assert layer.self_attn["q_proj"].weight.device.type == "meta"
    assert layer.mlp.experts.gate_up_proj.device.type == "meta"
    assert layer.hyper_scale.device.type == "meta"


def test_wrapper_export_accepts_exact_parameter_names():
    wrapper = _tiny_wrapper()
    if wrapper is None:
        _runtime_guard()
        return
    import tempfile

    wrapper.save_dir = tempfile.mkdtemp(prefix="gsq_qwen4exp_export_")
    topology = wrapper.topology[0]
    # Exact parameter names must not raise the way get_submodule would.
    wrapper.save_prefixes_to_disc(list(topology.extra))
    assert os.listdir(wrapper.save_dir) == []


def test_wrapper_unimplemented_boundaries_raise():
    wrapper = _tiny_wrapper()
    if wrapper is None:
        _runtime_guard()
        return
    _assert_raises(
        NotImplementedError,
        wrapper.get_layer_initialization,
        None,
        None,
        None,
        None,
    )
    _assert_raises(NotImplementedError, wrapper.get_mlp_input, None)
    _assert_raises(NotImplementedError, wrapper.get_mlp_output, None)
    _assert_raises(NotImplementedError, wrapper.save_moe_experts_to_disc)
    _assert_raises(NotImplementedError, wrapper.ppl_evaluation)


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except BaseException as exc:  # noqa: BLE001 - report and continue
            if type(exc).__name__ == "Skipped":
                print(f"skip {test.__name__}: {exc}")
                continue
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
