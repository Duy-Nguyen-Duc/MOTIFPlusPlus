from typing import Dict, List, Optional

import torch

PRIMITIVES: List[str] = ["REACH", "PRESHAPE", "GRASP", "CARRY", "REORIENT", "PUSH", "RELEASE"]
SLOT_NAMES: List[str] = ["translation", "orientation", "gripper"]
DEFAULT_SLOT_MASK: List[List[int]] = [
    [1, 1, 0],  # REACH
    [0, 0, 1],  # PRESHAPE
    [0, 0, 1],  # GRASP
    [1, 0, 0],  # CARRY
    [0, 1, 0],  # REORIENT
    [1, 0, 0],  # PUSH
    [0, 0, 1],  # RELEASE
]


def frames_to_tokens(x: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """
    Per-frame labels [B, H] -> per-token labels [B, n_tokens].
    """
    if x.dim() != 2:
        raise ValueError(f"expected [B, H], got {tuple(x.shape)}")
    H = x.shape[1]
    centre = ((torch.arange(n_tokens, device=x.device, dtype=torch.float32) + 0.5)
              * H / n_tokens).long().clamp_(0, H - 1)
    return x[:, centre]


def boundary_counts(gt_change: torch.Tensor, pred_change: torch.Tensor, tol: int = 1):
    """
    Match boundaries with +-tol token tolerance. Both [B, M-1] bool.
    """
    def dilate(m):
        k = 2 * tol + 1
        return torch.nn.functional.max_pool1d(m.float().unsqueeze(1), k, stride=1, padding=tol
                                              ).squeeze(1) > 0.5
    pred_d, gt_d = dilate(pred_change), dilate(gt_change)
    tp = (gt_change & pred_d).sum().item()
    fn = (gt_change & ~pred_d).sum().item()
    fp = (pred_change & ~gt_d).sum().item()
    return tp, fp, fn


class PrimitiveTyingMeter:
    """
    Accumulates how well free code choices follow the slot mask.
    update() takes the nearest-neighbour code per slot token BEFORE label
    gating, and idle codes are the reserved codebook rows [0, num_slots).
    """

    def __init__(self, num_primitives: int, num_slots: int, slot_mask: Optional[torch.Tensor] = None,
                 tol: int = 1):
        self.P, self.S, self.tol = num_primitives, num_slots, tol
        self.slot_mask = slot_mask.detach().cpu().float() if slot_mask is not None else None
        self.reset()

    def reset(self):
        self.active = torch.zeros(self.P, self.S, dtype=torch.float64)
        self.total = torch.zeros(self.P, self.S, dtype=torch.float64)
        self.tp = self.fp = self.fn = 0
        self.windows = 0

    @torch.no_grad()
    def update(self, idx_per_slot: torch.Tensor, prim_tok: torch.Tensor):
        """idx_per_slot [B, S, M] long (free choice); prim_tok [B, M] long."""
        B, S, M = idx_per_slot.shape
        is_idle = idx_per_slot < S                                   # reserved rows
        active = (~is_idle).permute(0, 2, 1).reshape(-1, S).double().cpu()   # [B*M, S]
        p = prim_tok.reshape(-1).cpu().clamp(0, self.P - 1)
        self.active.index_add_(0, p, active)
        self.total.index_add_(0, p, torch.ones_like(active))
        if M > 1:
            gt = prim_tok[:, 1:] != prim_tok[:, :-1]
            pred = (is_idle[:, :, 1:] != is_idle[:, :, :-1]).any(1)
            tp, fp, fn = boundary_counts(gt, pred, self.tol)
            self.tp += tp; self.fp += fp; self.fn += fn
        self.windows += B

    def summary(self) -> Dict:
        rate = (self.active / self.total.clamp_min(1)).numpy()
        out = {
            "windows": self.windows,
            "activation_rate": {PRIMITIVES[p] if p < len(PRIMITIVES) else str(p):
                                [round(float(rate[p, s]), 4) for s in range(self.S)]
                                for p in range(self.P)},
            "tokens_per_primitive": [int(self.total[p, 0]) for p in range(self.P)],
            "boundary_precision": self.tp / max(self.tp + self.fp, 1),
            "boundary_recall": self.tp / max(self.tp + self.fn, 1),
        }
        pr, rc = out["boundary_precision"], out["boundary_recall"]
        out["boundary_f1"] = 2 * pr * rc / max(pr + rc, 1e-9)
        if self.slot_mask is not None:
            mask = self.slot_mask.numpy()
            seen = (self.total > 0).numpy()
            err = abs(rate - mask)[seen]
            out["mask_agreement"] = float(1.0 - err.mean()) if err.size else float("nan")
            # the two halves of the claim: active slots stay on, inactive slots go idle
            on = rate[(mask > 0.5) & seen]
            off = rate[(mask < 0.5) & seen]
            out["active_slot_on_rate"] = float(on.mean()) if on.size else float("nan")
            out["inactive_slot_idle_rate"] = float(1.0 - off.mean()) if off.size else float("nan")
        return out
