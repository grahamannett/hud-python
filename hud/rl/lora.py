import math
import re
from typing import Dict, List

import torch
import torch.nn as nn

from hud.rl.logger import console


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer.

    ΔW = B @ A, with A ∈ R^(r x in_features), B ∈ R^(out_features x r)
    Forward: base(x) + (x @ A.T) @ B.T * (alpha / r)
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), "LoRALinear expects nn.Linear as base_layer"
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 0.0

        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.empty(base_layer.out_features, self.rank))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base = self.base_layer(x)
        if self.rank == 0 or self.scaling == 0.0:
            return base
        x_d = self.dropout(x)
        delta = (x_d @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return base + delta * self.scaling


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    mod: nn.Module = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def _set_module_by_name(model: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _find_target_linear_modules(model: nn.Module, patterns: List[str]) -> List[str]:
    targets: List[str] = []
    regexes = [re.compile(p) for p in patterns]
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(r.search(n) for r in regexes):
            targets.append(n)
    return targets


def _should_keep_trainable(param_name: str, modules_to_save: List[str]) -> bool:
    if not modules_to_save:
        return False
    regexes = [re.compile(p) for p in modules_to_save]
    if any(r.search(param_name) for r in regexes):
        return True
    # Also test parent module name
    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
    return any(r.search(module_name) for r in regexes)


def freeze_all_except_lora_and_specified(model: nn.Module, modules_to_save: List[str]) -> None:
    for name, p in model.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            p.requires_grad_(True)
        elif _should_keep_trainable(name, modules_to_save):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def apply_lora_to_model(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: List[str],
    modules_to_save: List[str] | None = None,
) -> None:
    """Insert LoRALinear wrappers for matching nn.Linear modules and freeze others.

    Should be called before FSDP wrapping.
    """
    if rank <= 0:
        console.warning_log("LoRA rank <= 0; skipping LoRA application")
        return

    targets = _find_target_linear_modules(model, target_modules)
    if not targets:
        console.warning_log("LoRA target modules not found; check patterns")
        return

    for name in targets:
        base = _get_module_by_name(model, name)
        if not isinstance(base, nn.Linear):
            continue
        lora = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        _set_module_by_name(model, name, lora)

    freeze_all_except_lora_and_specified(model, modules_to_save or [])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_adapter_params = 0
    lora_adapted_params = 0
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_adapter_params += module.lora_A.numel() + module.lora_B.numel()
            lora_adapted_params += module.base_layer.weight.numel()
    fully_trainable = trainable_params - lora_adapter_params
    adapted_or_trainable = lora_adapted_params + fully_trainable
    console.info_log(
        f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params"
    )
    console.info_log(
        f"LoRA: {fully_trainable:,} fully trainable; {adapted_or_trainable:,}/{total_params:,} adapted or trainable"
    )


def has_lora_layers(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def merge_lora_weights_inplace(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Add LoRA deltas into base_layer in-place and zero LoRA weights. Returns original LoRA params."""
    original: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            original[name] = {"lora_A": module.lora_A.data.clone(), "lora_B": module.lora_B.data.clone()}
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.base_layer.weight.data.add_(delta)
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
    return original


def restore_lora_weights_inplace(model: nn.Module, original: Dict[str, Dict[str, torch.Tensor]]) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in original:
            module.lora_A.data.copy_(original[name]["lora_A"])
            module.lora_B.data.copy_(original[name]["lora_B"])
            # Remove the added delta to keep training consistent
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.base_layer.weight.data.sub_(delta)


def clean_lora_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove LoRA-specific tensors and replace `.base_layer.` with `.` for HF/vLLM compat."""
    clean: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if "lora_A" in k or "lora_B" in k:
            continue
        new_k = k.replace(".base_layer.", ".") if ".base_layer." in k else k
        clean[new_k] = v
    return clean

