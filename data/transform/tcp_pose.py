from typing import Any

import numpy as np
import torch

from data.transform.base import ModalityTransform


class TcpPosePassthrough(ModalityTransform):
    """Raw state.position + state.rotation (before normalisation) -> data["tcp_pose"] [T, 7] for the geometry tokens."""
    out_key: str = "tcp_pose"

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        parts = [torch.as_tensor(np.asarray(data[k], dtype=np.float32)) for k in self.apply_to]
        data[self.out_key] = torch.cat(parts, dim=-1)
        return data
