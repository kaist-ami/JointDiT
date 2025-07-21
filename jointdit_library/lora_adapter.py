import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional

class LoRALinear(nn.Module):
    """
    A Low-Rank Adaptation (LoRA) wrapper for an existing nn.Linear layer.
    Applies the base layer to half the batch and a LoRA-adjusted path to the other half,
    then concatenates the results.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer

        # LoRA down- and up-projection layers
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        # Optional dropout on the LoRA branch
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Scaling factor
        self.alpha = alpha
        self.scaling = alpha / r

        # Initialize LoRA weights
        init.zeros_(self.lora_B.weight)
        init.xavier_uniform_(self.lora_A.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        if batch % 2 != 0:
            raise ValueError("Batch size must be even when using LoRALinear.")

        # Split batch in half
        x_base, x_lora = x[:batch//2], x[batch//2:]

        # Base path
        out_base = self.base_layer(x_base)
        # LoRA-enriched path
        out_lora = (
            self.base_layer(x_lora)
            + self.lora_B(self.lora_A(self.lora_dropout(x_lora))) * self.scaling
        )

        # Concatenate along the batch dimension
        return torch.cat([out_base, out_lora], dim=0)


def add_lora_to_attention(
    model: nn.Module,
    module_name: str,
    r: int = 4,
    alpha: int = 1,
    dropout: float = 0.0
):
    """
    Replace every nn.Linear submodule whose name ends with `module_name`
    with a LoRALinear wrapper preserving the original layer.
    """
    for name, module in model.named_modules():
        if name.endswith(module_name) and isinstance(module, nn.Linear):
            lora_wrapper = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, lora_wrapper)


def add_linear_to_double_blocks(model: nn.Module, additional_param: bool = False):
    """
    In each double-stream block, insert a new nn.Linear called 'joint1'.
    Input/output dims are twice the block's qkv dimension, plus optional extra param.
    """
    for block in model.double_blocks:
        dim = block.img_attn.qkv.in_features * 2
        in_dim = dim + 512 if additional_param else dim
        out_dim = dim
        joint = nn.Linear(in_dim, out_dim, bias=True)
        init.zeros_(joint.weight)
        init.zeros_(joint.bias)
        block.add_module("joint1", joint)


def add_linear_to_single_blocks(model: nn.Module, additional_param: bool = False):
    """
    In each single-stream block, insert a new nn.Linear called 'joint2'.
    Input/output dims are twice the block's linear2 output dim, plus optional extra param.
    """
    for block in model.single_blocks:
        dim = block.linear2.out_features * 2
        in_dim = dim + 512 if additional_param else dim
        out_dim = dim
        joint = nn.Linear(in_dim, out_dim, bias=True)
        init.zeros_(joint.weight)
        init.zeros_(joint.bias)
        block.add_module("joint2", joint)
