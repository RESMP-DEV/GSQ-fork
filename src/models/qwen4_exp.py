"""GSQ wrapper for the Qwen4Exp architecture.

Scope, deliberately narrow:

* the checkpoint reaches this wrapper through ``model_type`` dispatch
  (:func:`src.models.classify_model`), never through the generic Qwen3
  wrappers;
* the decoder stack lives under ``model.language_model.layers`` and its
  parameters are grouped by the Qwen4Exp topology contract
  (:class:`src.models.Qwen4ExpLayerTopology`), which covers the fused routed
  experts, the shared expert of a PLE layer, and the hyper-connection
  parameters of the residual structure;
* weights move between the accelerator and ``meta``;
* every boundary that still needs per-expert 3D quantization, per-expert
  output export, or the Qwen4Exp forward split raises
  :class:`NotImplementedError` instead of writing a silently wrong checkpoint.
"""

from __future__ import annotations

import os

from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file

from . import (
    QWEN4EXP_LAYER_PREFIX_CANDIDATES,
    Qwen4ExpLayerTopology,
    build_qwen4exp_topology,
    qwen4exp_not_implemented,
    resolve_layer_prefix,
)
from .base import BaseModelWrapper


class Qwen4ExpWrapper(BaseModelWrapper):
    """Placement-only GSQ wrapper for Qwen4Exp checkpoints."""

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)

        checkpoint_names = self._checkpoint_names()
        model_names = [name for name, _ in self.model.named_parameters()]
        self.layer_prefix = resolve_layer_prefix(checkpoint_names, model_names)
        self._model_layer_prefix = self.layer_prefix.replace(".language_model", "")

        try:
            layers = self.model.get_submodule(self._model_layer_prefix)
        except AttributeError as exc:  # pragma: no cover - config/model mismatch
            raise RuntimeError(
                f"Qwen4Exp: decoder stack '{self._model_layer_prefix}' was not "
                "found on the instantiated model. Refusing to guess the topology."
            ) from exc
        self.num_layers = len(layers)
        if self.num_layers == 0:
            raise RuntimeError(
                f"Qwen4Exp: decoder stack '{self._model_layer_prefix}' is empty; "
                "refusing to build a topology without layers."
            )

        config = self.model.config
        self.num_experts = (
            getattr(config, "num_experts", None)
            or getattr(config, "n_routed_experts", None)
            or getattr(config, "num_local_experts", None)
            or 0
        )
        self.is_moe = True
        self.fused_experts = True
        self.fused_expert_intermediate_size = getattr(
            config, "moe_intermediate_size", None
        ) or getattr(config, "intermediate_size", None)

        source_names = checkpoint_names or model_names
        self.topology = build_qwen4exp_topology(self.layer_prefix, source_names, self.num_layers)

    # ── topology resolution ───────────────────────────────────────────────

    def _checkpoint_names(self):
        """Return the checkpoint-side parameter names under the decoder stack.

        The names come from the safetensors index (or the shard header), not
        from a tensor load, so resolving the topology stays cheap.
        """
        if self.dummy:
            return []
        if self._name_to_shard is not None:
            all_names = list(self._name_to_shard.keys())
        else:
            with safe_open(self._single_shard, framework="pt", device="cpu") as f:
                all_names = list(f.keys())
        best: list[str] = []
        for candidate in QWEN4EXP_LAYER_PREFIX_CANDIDATES:
            marker = candidate + "."
            names = [name for name in all_names if name.startswith(marker)]
            if len(names) > len(best):
                best = names
        return best

    def _topology_for(self, layer_name) -> Qwen4ExpLayerTopology:
        layer_idx = int(str(layer_name).split(".")[-1])
        topology = self.topology.get(layer_idx)
        if topology is None:
            raise KeyError(f"Qwen4Exp: no topology entry for layer {layer_idx} ({layer_name!r}).")
        if topology.is_empty:
            raise RuntimeError(
                f"Qwen4Exp: no checkpoint parameters resolved for layer "
                f"{layer_idx} under '{topology.prefix}'. Refusing to move an "
                "unknown layer."
            )
        return topology

    # ── BaseModelWrapper contract ─────────────────────────────────────────

    def _layer_prefixes(self, layer_name):
        topology = self._topology_for(layer_name)
        prefixes = {
            "non_mlp": list(topology.attention)
            + list(topology.norms)
            + list(topology.router)
            + list(topology.extra),
            "mlp": list(topology.shared_expert) + list(topology.routed_experts),
        }
        if topology.fused_expert_params:
            prefixes["mlp_offload_params"] = list(topology.fused_expert_params)
        return prefixes

    def get_layer_module(self, idx):
        return self.model.get_submodule(f"{self._model_layer_prefix}.{idx}")

    def move_embed_to(self, device):
        parent = self._stack_parent()
        names = self._names_from_ckpt([f"{parent}.embed_tokens", "model.embed_tokens"])
        if device == "cuda":
            self._set_tensors(names)
            rotary = getattr(getattr(self.model, "model", None), "rotary_emb", None)
            if rotary is not None:
                self.model.model.rotary_emb = rotary.to("cuda")
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        parent = self._stack_parent()
        names = self._names_from_ckpt([f"{parent}.norm", "model.norm", "lm_head"])
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    # ── movement ──────────────────────────────────────────────────────────

    def move_layer_to_gpu(self, layer_name):
        """Materialize one decoder layer on the accelerator.

        Fused routed experts are handled by the base fused-expert machinery:
        a checkpoint that stores the 3D ``gate_up_proj``/``down_proj`` tensors
        is copied as-is, while a checkpoint that stores per-expert 2D weights
        is sliced into the fused tensors.
        """
        topology = self._topology_for(layer_name)
        pairs = self._names_from_ckpt(topology.param_names)
        self._set_tensors(pairs)

    def offload_to_meta(self, layer_name):
        """Offload one decoder layer to ``meta``.

        Shared-expert and hyper-connection parameters are offloaded by name;
        the fused routed experts are offloaded through their 3D tensors
        because a per-expert checkpoint layout has no matching model
        parameter to target.
        """
        topology = self._topology_for(layer_name)
        self._offload_names_to_meta(self._names_from_ckpt(topology.non_expert_param_names))
        for param_name in topology.fused_expert_params:
            set_module_tensor_to_device(
                self.model,
                param_name.replace(".language_model", ""),
                "meta",
            )

    def _stack_parent(self):
        """Return the checkpoint-side parent of the decoder stack."""
        parent = self.layer_prefix.rsplit(".layers", 1)[0]
        return parent or self.layer_prefix

    # ── export ────────────────────────────────────────────────────────────

    def save_prefixes_to_disc(self, prefixes, exclude=None):
        """Save raw (non-quantized) layer parameters to disk.

        The Qwen4Exp topology names parameters exactly, so prefixes are
        resolved against the model parameters instead of ``get_submodule``;
        this keeps hyper-connection parameters exportable even when their
        parent is a bare ``nn.Parameter`` on the decoder layer.
        """
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return
        excluded = list(exclude or [])
        os.makedirs(self.save_dir, exist_ok=True)
        params = dict(self.model.named_parameters())
        for prefix in prefixes:
            if any(token in prefix for token in excluded):
                continue
            model_side = prefix.replace(".language_model", "")
            to_save = {}
            if model_side in params:
                tensor = params[model_side]
                if tensor.is_cuda:
                    to_save[prefix] = tensor.detach().cpu()
            else:
                for name, tensor in params.items():
                    if name.startswith(model_side + ".") and tensor.is_cuda:
                        local = name[len(model_side) + 1 :]
                        to_save[f"{prefix}.{local}"] = tensor.detach().cpu()
            if not to_save:
                continue
            safe = prefix.replace(".", "_")
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)

    def save_moe_experts_to_disc(self):
        raise qwen4exp_not_implemented("export of the per-expert quantized 3D fused expert weights")

    # ── fail-closed boundaries ────────────────────────────────────────────

    def get_layer_initialization(self, trainer, gpt_all, config, logging, is_attn=False):
        raise qwen4exp_not_implemented(
            "per-expert 3D quantization (GPTQ/GSQ) of the fused routed experts"
        )

    def get_mlp_input(self, layer_input):
        raise qwen4exp_not_implemented(
            "the Qwen4Exp layer forward split: the residual/hyper-connection "
            "semantics of the attention block are not modeled yet"
        )

    def get_mlp_output(self, mlp_input_batch):
        raise qwen4exp_not_implemented(
            "the Qwen4Exp MoE forward: shared-expert gating, PLE routing and "
            "hyper-connection combination are not modeled yet"
        )

    def ppl_evaluation(self, read_from_disk=-1):
        raise qwen4exp_not_implemented(
            "perplexity evaluation (it depends on the unimplemented Qwen4Exp forward split)"
        )
