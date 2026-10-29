"""Pre-decode every episode's video frames into a frame cache.

Why: LeRobotSingleDataset.get_video opens a fresh decord VideoReader per sample and
seeks, so one batch of N samples costs 2N open+seek+decode cycles. Stages 2 and 3
run 30 and 60 epochs over the same ~900 videos, re-decoding all of them every epoch.
Measured cost of that path: ~0.5 samples/s.

What this stores: frames at their NATIVE resolution, decoded but otherwise untouched.
VideoCrop / VideoResize / VideoColorJitter still run per sample in the transform
stack, so augmentation stays stochastic and training behaviour is unchanged. Only
the decode is hoisted out.

Two on-disk formats, both read transparently by data/frame_cache_dataset.py:

  raw  (default)   <root>/<robot>/frames_<key>.u8      uint8 memmap (total_frames, H, W, 3)
                   <root>/<robot>/frames_<key>.json    {"shape", "offsets": {ep: [start, len]}, "video_key"}
  webp             <root>/<robot>/frames_<key>.webp.bin  concatenated lossless-WebP frames
                   <root>/<robot>/frames_<key>.idx.npy   int64 (total_frames + 1,) byte offsets into .bin
                   <root>/<robot>/frames_<key>.json      as above plus "format": "webp"
                   Lossless (bit-identical to raw after decode), 3.9x smaller on these renders
                   (50 KB vs 192 KB per 256x256 frame), decode 0.9 ms/frame with cv2. The full
                   896-episode cache is ~12 GB instead of ~47 GB. Encoding is slow (~70 ms/frame),
                   hence --workers.

A memmap rather than RAM so dataloader workers share one copy through the page
cache instead of each holding its own, and so the cache survives between stages.

Usage:
    python data/process/prextract_frames.py \
        --dataset_root ./demo_data_processed --out_root ./demo_data/frame_cache_full --format webp --workers 8
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.video import get_all_frames

VIDEO_KEYS = ["observation.images.image", "observation.images.wrist_image"]
KEY_ALIAS = {"observation.images.image": "image",
             "observation.images.wrist_image": "wrist_image"}
WEBP_LOSSLESS = 101   # cv2: IMWRITE_WEBP_QUALITY > 100 selects lossless encoding


def episode_lengths(dataset_dir: Path) -> dict[int, int]:
    out = {}
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            out[int(e["episode_index"])] = int(e["length"])
    return out


def _encode_webp(frame: np.ndarray) -> bytes:
    import cv2
    cv2.setNumThreads(1)
    ok, buf = cv2.imencode(".webp", np.ascontiguousarray(frame), [cv2.IMWRITE_WEBP_QUALITY, WEBP_LOSSLESS])
    assert ok, "webp encode failed"
    return buf.tobytes()


def _episode_frames(dataset_dir: Path, full_key: str, ep: int, n: int, backend: str) -> np.ndarray:
    frames = get_all_frames(
        (dataset_dir / "videos" / "chunk-000" / full_key / f"episode_{ep:06d}.mp4").as_posix(),
        video_backend=backend, video_backend_kwargs={},
    )
    # A container can hold a frame more or fewer than the parquet rows;
    # clamp so index == step for every step the dataset will ask for.
    if frames.shape[0] < n:
        pad = np.repeat(frames[-1:], n - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames[:n]


def extract_robot(dataset_dir: Path, out_dir: Path, backend: str, overwrite: bool,
                  only_episodes=None, fmt: str = "raw", workers: int = 8):
    """Cache one robot's frames.

    `only_episodes` restricts the cache to the episodes a given K-shot mask actually
    selects. Offsets are keyed by episode_index, so a subset cache is looked up exactly
    like a full one -- it simply raises KeyError if training ever asks for an episode
    outside the mask, rather than silently substituting the wrong frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = episode_lengths(dataset_dir)
    eps = sorted(lengths if only_episodes is None else
                 [e for e in lengths if e in set(only_episodes)])
    total = sum(lengths[e] for e in eps)

    for full_key in VIDEO_KEYS:
        alias = KEY_ALIAS[full_key]
        meta_path = out_dir / f"frames_{alias}.json"
        if meta_path.exists() and not overwrite:
            print(f"  skip {alias} (exists)")
            continue

        # Probe the first episode for the frame shape.
        probe = _episode_frames(dataset_dir, full_key, eps[0], lengths[eps[0]], backend)
        h, w, c = probe.shape[1:]
        shape = (total, h, w, c)
        offsets, cursor = {}, 0

        if fmt == "raw":
            bin_path = out_dir / f"frames_{alias}.u8"
            print(f"  {alias}: {total} frames -> {np.prod(shape)/1e9:.1f} GB raw {shape}")
            mm = np.lib.format.open_memmap(bin_path.as_posix(), mode="w+", dtype=np.uint8, shape=shape)
            for ep in tqdm(eps, desc=f"    {alias}", leave=False):
                n = lengths[ep]
                mm[cursor:cursor + n] = _episode_frames(dataset_dir, full_key, ep, n, backend)
                offsets[str(ep)] = [cursor, n]
                cursor += n
            mm.flush()
            del mm
            meta = {"shape": list(shape), "offsets": offsets, "video_key": full_key}
        elif fmt == "webp":
            bin_path = out_dir / f"frames_{alias}.webp.bin"
            idx_path = out_dir / f"frames_{alias}.idx.npy"
            print(f"  {alias}: {total} frames -> lossless webp {shape}, {workers} encoder workers")
            byte_idx = [0]
            with open(bin_path, "wb") as fh, Pool(workers) as pool:
                for ep in tqdm(eps, desc=f"    {alias}", leave=False):
                    n = lengths[ep]
                    frames = _episode_frames(dataset_dir, full_key, ep, n, backend)
                    for blob in pool.imap(_encode_webp, list(frames), chunksize=4):
                        fh.write(blob)
                        byte_idx.append(byte_idx[-1] + len(blob))
                    offsets[str(ep)] = [cursor, n]
                    cursor += n
            np.save(idx_path.as_posix(), np.asarray(byte_idx, dtype=np.int64))
            meta = {"format": "webp", "shape": list(shape), "offsets": offsets, "video_key": full_key,
                    "bytes": byte_idx[-1]}
            print(f"    {alias}: {byte_idx[-1]/1e9:.2f} GB on disk ({byte_idx[-1]/max(total,1)/1024:.1f} KB/frame)")
        else:
            raise ValueError(fmt)

        assert cursor == total
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"    wrote {meta_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True,
                   help="parent of the per-robot canonicalized datasets")
    p.add_argument("--out_root", required=True)
    p.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    p.add_argument("--video_backend", default="decord")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--format", choices=["raw", "webp"], default="raw")
    p.add_argument("--workers", type=int, default=8, help="webp encoder processes")
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
                      args.video_backend, args.overwrite, only_episodes=only,
                      fmt=args.format, workers=args.workers)
    print("PREXTRACT_DONE")


if __name__ == "__main__":
    main()
