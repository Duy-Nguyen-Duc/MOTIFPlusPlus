"""LeRobotSingleDataset variant that reads frames from a pre-decoded memmap
(built by data/process/prextract_frames.py) instead of seeking mp4s per sample.

Only `get_video` is overridden. The converter writes exactly one frame per
parquet row at constant fps, so the parent's timestamp->frame mapping is the
identity and the memmap can be indexed with step indices directly.
"""

import json
from pathlib import Path

import numpy as np

from .dataset import LeRobotSingleDataset


class FrameCachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, *args, frame_cache_dir: str | Path, **kwargs):
        self._frame_cache_dir = Path(frame_cache_dir)
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._offsets = {}
        for key in self.modality_keys.get("video", []):
            alias = key.replace("video.", "")
            meta_path = self._frame_cache_dir / f"frames_{alias}.json"
            bin_path = self._frame_cache_dir / f"frames_{alias}.u8"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"no frame cache for {alias} at {meta_path}; run "
                    f"data/process/prextract_frames.py first"
                )
            meta = json.load(open(meta_path))
            # mmap_mode='r' -- workers share one page-cache copy instead of each
            # materialising the array.
            self._cache[alias] = np.load(bin_path.as_posix(), mmap_mode="r")
            self._offsets[alias] = {int(k): tuple(v) for k, v in meta["offsets"].items()}

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Same clamping the parent applies before looking frames up.
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        alias = key.replace("video.", "")
        start, length = self._offsets[alias][int(trajectory_id)]
        idx = start + np.minimum(step_indices, length - 1)
        return np.asarray(self._cache[alias][idx])
