from dataclasses import dataclass
import math
from typing import Iterable, List, Optional

import torch
from torch import nn


@dataclass
class ELoRARouting:
    alpha: torch.Tensor   # [B, K]   skill mixture
    gate: torch.Tensor    # [B, N]   joint gates over the union of robot graphs (0 off-robot)


class ELoRAContext:
    """Mutable holder shared by all wrapped layers of one policy."""

    def __init__(self):
        self.routing: Optional[ELoRARouting] = None


class ELoRALinear(nn.Module):
    """y = W_base x + (scale / r) * B_eff (A_eff x), A_eff = sum_k alpha_k A_k, B_eff = sum_i g_i B_i."""

    def __init__(self, base: nn.Linear, ctx: ELoRAContext, num_skills: int, num_nodes: int,
                 rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.ctx = ctx
        self.rank = rank
        self.scaling = scale / rank
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(num_skills, rank, d_in))
        self.B = nn.Parameter(torch.zeros(num_nodes, d_out, rank))
        for k in range(num_skills):
            nn.init.kaiming_uniform_(self.A[k], a=math.sqrt(5))
        # save_base=False drops the wrapped frozen weights from state_dict() (used when the
        # base model is a pretrained backbone that is reloaded from its own source).
        self.save_base = True
        hook = getattr(self, "register_state_dict_post_hook", None) or self._register_state_dict_hook
        hook(self._drop_base_hook)

    @staticmethod
    def _drop_base_hook(module, state_dict, prefix, local_metadata):
        if not module.save_base:
            for k in [k for k in state_dict if k.startswith(prefix + "base.")]:
                state_dict.pop(k)

    @property
    def weight(self):  
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        r = self.ctx.routing
        if r is None:
            return y
        dt = self.A.dtype
        A_eff = torch.einsum("bk,krd->brd", r.alpha.to(dt), self.A)         # [B, r, d_in]
        B_eff = torch.einsum("bn,nor->bor", r.gate.to(dt), self.B)          # [B, d_out, r]
        squeeze = x.dim() == 2
        xb = x.unsqueeze(1) if squeeze else x                                # [B, T, d_in]
        if xb.shape[0] != A_eff.shape[0]:
            raise ValueError(f"E-LoRA routing batch {A_eff.shape[0]} != input batch {xb.shape[0]}")
        delta = torch.einsum("btr,bor->bto", torch.einsum("btd,brd->btr", xb.to(dt), A_eff), B_eff)
        delta = delta.squeeze(1) if squeeze else delta
        return y + (self.scaling * delta).to(y.dtype)

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, skills={self.A.shape[0]}, nodes={self.B.shape[0]}, "
                f"in={self.base.in_features}, out={self.base.out_features}")


def inject_elora(root: nn.Module, targets: Iterable[str], ctx: ELoRAContext, num_skills: int,
                 num_nodes: int, rank: int = 4, scale: float = 1.0) -> List[str]:
    targets = tuple(targets)
    replaced = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, ELoRALinear):
            continue
        if not name.endswith(targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        if isinstance(parent, nn.MultiheadAttention):
            raise ValueError(f"{name}: nn.MultiheadAttention bypasses out_proj.forward; "
                             "E-LoRA cannot be applied there")
        setattr(parent, attr, ELoRALinear(module, ctx, num_skills, num_nodes, rank, scale))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"no nn.Linear matched E-LoRA targets {targets}")
    return replaced


def elora_parameters(root: nn.Module):
    for m in root.modules():
        if isinstance(m, ELoRALinear):
            yield m.A
            yield m.B
