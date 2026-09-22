"""Versioned gripper polarity for the factorized motif teacher only.

Inputs remain the existing, min/max-normalized episode-relative state. The
reflection is self-inverse and does not change robot-native state or actions.
"""
import torch

from data.embodiment_tags import EMBODIMENT_TAG_MAPPING


LEGACY = "legacy"
OPEN_POSITIVE = "maniskill_open_positive_v1"
MARKER = "_motif_gripper_convention_version"


def convention_version(convention):
    if convention not in (LEGACY, OPEN_POSITIVE):
        raise ValueError(f"Unknown motif gripper convention: {convention}")
    return int(convention == OPEN_POSITIVE)


def reflect_relative_gripper(state, embodiment_id):
    """Reflect xArm's last channel, preserving all other values and gradients."""
    if state.ndim != 3 or state.shape[-1] != 8:
        raise ValueError("Gripper polarity v1 expects [batch, time, 8] rel_state")
    if embodiment_id is None:
        raise ValueError("Gripper polarity v1 requires embodiment_id")
    if embodiment_id.dtype not in (torch.int32, torch.int64):
        raise ValueError("embodiment_id must be an integer tensor")
    ids = embodiment_id.reshape(-1).to(device=state.device)
    if ids.numel() != state.shape[0]:
        raise ValueError("Expected one embodiment_id per trajectory")
    known = [EMBODIMENT_TAG_MAPPING[tag] for tag in
             ("panda_rel", "xarm6_rel", "widowxai_rel")]
    if not bool(torch.stack([ids == value for value in known]).any(0).all()):
        raise ValueError("Gripper polarity v1 has no calibration for this embodiment")
    sign = torch.where(ids == EMBODIMENT_TAG_MAPPING["xarm6_rel"], -1, 1)
    result = state.clone()
    result[..., 7] = state[..., 7] * sign[:, None].to(state.dtype)
    return result


def check_checkpoint_convention(state_dict, prefix, expected, error_msgs):
    """Reject semantic mismatches even under load_state_dict(strict=False).

    Stage-2 checkpoints intentionally omit the entire frozen teacher. An absent
    nested module is allowed; the predictor retains its own convention marker.
    """
    if prefix and not any(key.startswith(prefix) for key in state_dict):
        return
    marker = state_dict.get(prefix + MARKER)
    actual = 0 if marker is None else (
        int(marker.item()) if torch.is_tensor(marker) and marker.numel() == 1 else -1
    )
    if actual != expected:
        error_msgs.append(
            f"{prefix}motif gripper convention mismatch: checkpoint={actual}, "
            f"configured={expected}; rebuild from the matching teacher/config"
        )
