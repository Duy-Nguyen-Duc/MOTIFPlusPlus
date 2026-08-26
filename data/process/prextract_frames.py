"""Pre-decode every episode's video frames into a uint8 memmap.

Why: LeRobotSingleDataset.get_video opens a fresh decord VideoReader per sample and
seeks, so one batch of N samples costs 2N open+seek+decode cycles. Stages 2 and 3
run 30 and 60 epochs over the same ~900 videos, re-decoding all of them every epoch.
Measured cost of that path: ~0.5 samples/s.

What this stores: frames at their NATIVE resolution, decoded but otherwise untouched.
VideoCrop / VideoResize / VideoColorJitter still run per sample in the transform
stack, so augmentation stays stochastic and training behaviour is unchanged. Only
the decode is hoisted out.

Layout, per robot and per video key:
    <root>/<robot>/frames_<key>.u8      raw uint8, shape (total_frames, H, W, 3)
    <root>/<robot>/frames_<key>.json    {"shape": [...], "offsets": {ep_index: [start, len]}}

A memmap rather than RAM so dataloader workers share one copy through the page
cache instead of each holding its own, and so the cache survives between stages.

Usage:
    python data/process/prextract_frames.py \
        --dataset_root ./demo_data_processed --out_root ./demo_data/frame_cache
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.video import get_all_frames

VIDEO_KEYS = ["observation.images.image", "observation.images.wrist_image"]
KEY_ALIAS = {"observation.images.image": "image",
             "observation.images.wrist_image": "wrist_image"}


def episode_lengths(dataset_dir: Path) -> dict[int, int]:
    out = {}
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            out[int(e["episode_index"])] = int(e["length"])
    return out


def extract_robot(dataset_dir: Path, out_dir: Path, backend: str, overwrite: bool,
                  only_episodes=None):
    """Cache one robot's frames.

    `only_episodes` restricts the cache to the episodes a given K-shot mask actually
    selects. The full 896-episode cache is 50 GB and does not fit the disk quota here;
    the K=5 mask needs 627 episodes / 33.8 GB. Offsets are keyed by episode_index, so a
    subset cache is looked up exactly like a full one -- it simply raises KeyError if
    training ever asks for an episode outside the mask, rather than silently
    substituting the wrong frames. Rebuild when K changes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = episode_lengths(dataset_dir)
    eps = sorted(lengths if only_episodes is None else
                 [e for e in lengths if e in set(only_episodes)])
    total = sum(lengths[e] for e in eps)

    for full_key in VIDEO_KEYS:
        alias = KEY_ALIAS[full_key]
        bin_path = out_dir / f"frames_{alias}.u8"
        meta_path = out_dir / f"frames_{alias}.json"
        if meta_path.exists() and not overwrite:
            print(f"  skip {alias} (exists)")
            continue

        # Probe the first episode for the frame shape.
        probe = get_all_frames(
            (dataset_dir / "videos" / "chunk-000" / full_key /
             f"episode_{eps[0]:06d}.mp4").as_posix(),
            video_backend=backend, video_backend_kwargs={},
        )
        h, w, c = probe.shape[1:]
        shape = (total, h, w, c)
        print(f"  {alias}: {total} frames -> {np.prod(shape)/1e9:.1f} GB  {shape}")

        mm = np.lib.format.open_memmap(
            bin_path.as_posix(), mode="w+", dtype=np.uint8, shape=shape
        )
        offsets, cursor = {}, 0
        for ep in tqdm(eps, desc=f"    {alias}", leave=False):
            n = lengths[ep]
            frames = get_all_frames(
                (dataset_dir / "videos" / "chunk-000" / full_key /
                 f"episode_{ep:06d}.mp4").as_posix(),
                video_backend=backend, video_backend_kwargs={},
            )
            # A container can hold a frame more or fewer than the parquet rows;
            # clamp so index == step for every step the dataset will ask for.
            if frames.shape[0] < n:
                pad = np.repeat(frames[-1:], n - frames.shape[0], axis=0)
                frames = np.concatenate([frames, pad], axis=0)
            mm[cursor:cursor + n] = frames[:n]
            offsets[str(ep)] = [cursor, n]
            cursor += n
        mm.flush()
        del mm
        with open(meta_path, "w") as f:
            json.dump({"shape": list(shape), "offsets": offsets,
                       "video_key": full_key}, f)
        print(f"    wrote {bin_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True,
                   help="parent of the per-robot canonicalized datasets")
    p.add_argument("--out_root", required=True)
    p.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    p.add_argument("--video_backend", default="decord")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--episodes_json", default=None,
                   help="JSON {robot: [episode_index, ...]} limiting the cache to a "
                        "K-shot mask's episodes (see masked_episodes_K5.json)")
    args = p.parse_args()

    subsets = json.load(open(args.episodes_json)) if args.episodes_json else {}
    for robot in args.robots:
        only = subsets.get(robot)
        print(f"[{robot}]" + (f"  ({len(only)} masked episodes)" if only else "  (all episodes)"))
        extract_robot(Path(args.dataset_root) / robot,
                      Path(args.out_root) / robot,
                      args.video_backend, args.overwrite, only_episodes=only)
    print("PREXTRACT_DONE")


if __name__ == "__main__":
    main()
