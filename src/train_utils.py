import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# /dev/shm is 256 MB here and every in-flight dataloader batch lives in it
# (~1.2 MB/sample), so build_loader bounds num_workers * prefetch_factor *
# batch_size explicitly. The default "file_descriptor" sharing strategy is kept:
# "file_system" does not move tensors out of shm and can race the consumer.

from data.dataset import LeRobotSingleDataset
from data.data_config import DATA_CONFIG_MAP
from data.schema import EmbodimentTag
from data.multi_lerobot_wrapper import MultiDatasetWrapper

from data.collator import MOTIFCollator

log = logging.getLogger(__name__)

def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def rng_state():
    """Capture every RNG a resume needs to continue the same stream.

    Must survive torch.load(weights_only=True): numpy's state holds an ndarray
    whose reconstructor is rejected there, so it is stored as a plain list.
    """
    import random, numpy as np, torch
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], [int(x) for x in np_state[1]],
                  int(np_state[2]), int(np_state[3]), float(np_state[4])),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(st):
    """Restore what rng_state() captured. Returns False if the checkpoint has none."""
    import random, numpy as np, torch
    if not st:
        return False
    random.setstate(st["python"])
    ns = st["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if st.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in st["torch_cuda"]])
    return True

def load_yaml(rel_path: str):
    from hydra.utils import to_absolute_path
    from omegaconf import OmegaConf
    path = to_absolute_path(os.path.join("config", rel_path)) if not os.path.isabs(rel_path) else rel_path
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

def build_dataset(dc):
    paths, embs, cfgs, backs = dc["dataset_path"], dc["embodiment_tag"], dc["data_config"], dc["video_backend"]
    if 'modifiy_tasks_num' not in dc:
        dc["modifiy_tasks_num"] = [None] * len(paths)
    modift_tasks_num = dc["modifiy_tasks_num"]
    assert len(paths) == len(embs) == len(cfgs) == len(backs) == len(modift_tasks_num)
    datasets = []
    # drop_video: stage 1 never reads batch["images"], so skip video decoding and
    # its transforms entirely. Stages 2 and 3 consume the images; leave this off.
    drop_video = dc.get("drop_video", False)

    for p, e, c, v, m in zip(paths, embs, cfgs, backs, modift_tasks_num):
        modiality_configs = DATA_CONFIG_MAP[c].modality_config()
        transforms = DATA_CONFIG_MAP[c].transform()
        if drop_video:
            video_keys = set(modiality_configs.pop("video").modality_keys)
            kept = []
            for t in transforms.transforms:
                # Drop transforms whose only targets are video keys.
                targets = set(getattr(t, "apply_to", []) or [])
                if targets and targets <= video_keys:
                    continue
                if hasattr(t, "video_concat_order"):
                    t.video_concat_order = []
                kept.append(t)
            transforms.transforms = kept
        # Adjust transform based on window size
        if "window_size" in dc:
            modiality_configs['action'].delta_indices = list(range(dc["window_size"]))
            transforms.transforms[-1].action_horizon = dc["window_size"]
        common = dict(
            dataset_path=p,
            modality_configs=modiality_configs,
            transforms=transforms,
            embodiment_tag=EmbodimentTag(e),
            video_backend=v,
            modifiy_tasks_num=m,
            balance_tasks=dc.get("balance_tasks", False),
        )
        # Read pixels from the pre-decoded memmap cache (prextract_frames.py)
        # instead of seeking mp4s per sample.
        cache_root = dc.get("frame_cache_root")
        if cache_root and not drop_video:
            from data.frame_cache_dataset import FrameCachedLeRobotSingleDataset
            datasets.append(FrameCachedLeRobotSingleDataset(
                frame_cache_dir=os.path.join(cache_root, os.path.basename(str(p).rstrip("/"))),
                **common,
            ))
        else:
            datasets.append(LeRobotSingleDataset(**common))
    if dc.get("sampling_weights") is None:
        sw = [1.0 / len(datasets)] * len(datasets)
    else:
        s = torch.tensor(dc["sampling_weights"], dtype=torch.float32)
        sw = (s / s.sum()).tolist()
    return MultiDatasetWrapper(datasets=datasets, sampling_weights=sw)

def build_loader(dataset, dc, device: str):
    collator = MOTIFCollator(
        normalize_images=dc["collator"]["normalize_images"],
        device="cpu" if device.startswith("cuda") else device
    )
    # Bound the /dev/shm queue depth explicitly (see module header):
    # num_workers * prefetch_factor * batch_size * 1.2 MB must stay under 256 MB.
    # Defaults to torch's prefetch_factor of 2 when the key is absent.
    loader_kwargs = {}
    if dc["num_workers"] > 0 and dc.get("prefetch_factor") is not None:
        loader_kwargs["prefetch_factor"] = dc["prefetch_factor"]
    data_loader = DataLoader(
        dataset,
        batch_size=dc["batch_size"],
        shuffle=dc["shuffle"],
        num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"],
        drop_last=dc["drop_last"],
        persistent_workers=dc["persistent_workers"] and dc["num_workers"] > 0,
        collate_fn=collator,
        **loader_kwargs,
    )
    log.info(f"DataLoader: {len(data_loader)} batches, {len(data_loader.dataset)} samples per batch")
    return data_loader

class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom: Linearly increase from absolute warmup_lr to base_lr
    """
    def __init__(self, optimizer, warmup_lr, base_lr, warmup_iters, last_epoch=-1):
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_iters:
            # After warmup ends, LR is no longer controlled by this scheduler
            return [self.base_lr for _ in self.optimizer.param_groups]
        # Current warmup ratio
        alpha = self.last_epoch / max(1, self.warmup_iters)
        lr = self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
        return [lr for _ in self.optimizer.param_groups]

def sched_steps_per_epoch(cfg_train, loader) -> int:
    """Optimizer steps performed per epoch -- the unit the LR schedule is sized in.

    ``cfg.train.iters_per_epoch`` is already an optimizer-step budget (the training
    loops break on ``iters >= it``), so it passes through unchanged. Otherwise every
    batch in the loader is consumed and the trailing partial accumulation window is
    flushed, hence the ceiling division.
    """
    override = cfg_train.get("iters_per_epoch", 0)
    if override:
        return max(1, int(override))
    ga = max(1, cfg_train.get("grad_accum_steps", 1))
    return max(1, -(-len(loader) // ga))


def build_optim_sched(model, oc, epochs: int, steps_per_epoch: int):
    """Build the AdamW optimizer and its warmup+cosine schedule.

    ``steps_per_epoch`` is counted in optimizer steps (``sched.step()`` runs once
    per accumulation window, not per batch); derive it with
    ``sched_steps_per_epoch()``.
    """
    decay, no_decay, codebook_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vq.codebook" in n:
            codebook_params.append(p)
            continue
        if any(k in n for k in ["bias", "LayerNorm", "layernorm", "ln", "norm"]):
            no_decay.append(p)
        else:
            decay.append(p)
    base_lr = oc["lr"]
    warmup_lr = oc.get("warmup_lr", 1e-9)
    warmup_ratio = oc.get("warmup_ratio", 0.05)
    T_max_factor = oc.get("T_max_factor", 1.0)

    codebook_lr = oc.get("codebook_lr", base_lr * 3.0)

    param_groups = [
        {"params": decay, "weight_decay": oc["weight_decay"], "lr": base_lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
    ]

    if len(codebook_params) > 0:
        param_groups.append(
            {
                "params": codebook_params,
                "weight_decay": 0.0,   
                "lr": codebook_lr,     
            }
        )
    opt = torch.optim.AdamW(
        param_groups,
        lr=base_lr,  
        betas=(oc["beta1"], oc["beta2"]),
        eps=oc["eps"],
    )
    total_iters = epochs * max(1, steps_per_epoch)
    T_total = int(total_iters * T_max_factor)
    warmup_iters = int(T_total * warmup_ratio)
    warmup_iters = max(1, warmup_iters)
    warmup_sched = LinearWarmupLR(
        optimizer=opt,
        warmup_lr=warmup_lr,
        base_lr=base_lr,
        warmup_iters=warmup_iters,
    )

    cosine_iters = max(1, T_total - warmup_iters)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cosine_iters,
        eta_min=oc["min_lr"],
    )

    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters],
    )

    return opt, sched


def count_params_B(model, print_frozen=False):
    total = 0
    trainable = 0
    
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            log.info(f"[Trainable] {name}: {tuple(p.shape)}")
        else:
            if print_frozen:
                log.info(f"[Frozen]     {name}: {tuple(p.shape)}")
    
    log.info(f"Trainable/Total params: {trainable} / {total}")
    log.info(f"Trainable/Total params: {trainable/1e9:.4f} B / {total/1e9:.4f} B")
    return trainable, total

def get_grad_norm(model: torch.nn.Module) -> float:
    """calculates the norm of the gradients of the model parameters"""
    params = [p for p in model.parameters() if p.grad is not None]
    if len(params) == 0:
        return 0.0
    device = params[0].grad.device
    total = torch.zeros((), device=device)
    for p in params:
        total += p.grad.detach().pow(2).sum()
    return total.sqrt().item()