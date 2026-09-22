"""Optional slot-wise cross-embodiment alignment; legacy InfoNCE stays unchanged."""
import torch
import torch.nn.functional as F


def compute_slot_cross_robot_infonce(
    z_e, languages, progress, embodiment_ids, num_slots=3,
    tau=0.1, sigma=0.02, max_progress_delta=0.04,
):
    """Align corresponding slots across robots within a supported task/phase.

    Pool time inside each slot, never across slots. Positives require matching
    language, different embodiment and progress distance <= max_progress_delta.
    Gaussian weights retain the legacy progress preference inside this window.
    All non-self samples are denominator candidates. Unsupported anchors are
    excluded; an unsupported batch returns a differentiable zero.
    """
    b, tokens, dim = z_e.shape
    if tokens % num_slots or tau <= 0 or sigma <= 0 or max_progress_delta <= 0:
        raise ValueError("Invalid slot shape or contrastive scale")
    if len(languages) != b or embodiment_ids.numel() != b:
        raise ValueError("Each window needs one task and embodiment")
    emb = F.normalize(z_e.float().reshape(b, num_slots, -1, dim).mean(2), dim=-1)
    sim = torch.einsum("isd,jsd->sij", emb, emb) / tau
    eye = torch.eye(b, device=z_e.device, dtype=torch.bool)
    sim = sim.masked_fill(eye[None], -torch.inf)
    p0 = progress.reshape(b, -1)[:, 0].float()
    dp = (p0[:, None] - p0[None]).abs()
    same_task = torch.tensor([[x == y for y in languages] for x in languages], device=z_e.device)
    ids = embodiment_ids.reshape(b)
    positive = same_task & (ids[:, None] != ids[None]) & (dp <= max_progress_delta) & ~eye
    valid = positive.any(1)
    stats = {
        "slot_valid_anchor_frac": valid.float().mean().detach(),
        "slot_positive_pairs": positive.sum().detach(),
        "slot_pos_per_anchor": positive.sum(1).float().mean().detach(),
    }
    if not valid.any():
        loss = z_e.sum() * 0.0
        per_slot = loss.expand(num_slots)
    else:
        # Select supported rows before logsumexp to avoid all--inf gradients.
        logits = sim[:, valid, :]
        log_weights = (-(dp / sigma).square()).masked_fill(~positive, -torch.inf)
        per_slot = (torch.logsumexp(logits, -1) -
                    torch.logsumexp(logits + log_weights[valid][None], -1)).mean(1)
        loss = per_slot.mean()
    stats["slot_infonce_loss"] = loss.detach()
    for slot in range(num_slots):
        stats[f"slot_infonce_{slot}"] = per_slot[slot].detach()
    return loss, stats
