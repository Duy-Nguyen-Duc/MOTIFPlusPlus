import os
import errno
import glob
import time
import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import wandb

from model.stage2_3 import build_model

from train_utils import (
    set_seed,
    build_dataset,
    build_loader,
    build_optim_sched,
    sched_steps_per_epoch,
    rng_state,
    load_rng_state,
    count_params_B,
    get_grad_norm
)

log = logging.getLogger(__name__)


def _is_fs_error(e):
    """Transient filesystem failure (quota / no space / IO) rather than a bug or CUDA error."""
    if isinstance(e, OSError):
        return e.errno in (errno.EDQUOT, errno.ENOSPC, errno.EIO)
    s = str(e)
    return "PytorchStreamWriter" in s or "file write failed" in s or "unexpected pos" in s


def save_checkpoint(payload, path, tries=40, wait=90.0):
    """torch.save through a temp file + rename, retried on transient filesystem errors.

    The workspace fileset shares a GPFS quota with other projects; when it is full, torch.save
    raises RuntimeError (PytorchStreamWriter ... file write failed) or OSError (EDQUOT/ENOSPC).
    Crashing there cost three resumes on 2026-10-21, so wait and retry instead. Returns True
    on success. tries=None retries until the write succeeds (used for the final epoch, whose
    file the chain scripts wait for).
    """
    tmp = path + ".tmp"
    n = 0
    while True:
        n += 1
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return True
        except (RuntimeError, OSError) as e:
            if not _is_fs_error(e):
                raise
            try:
                os.remove(tmp)
            except OSError:
                pass
            if tries is not None and n >= tries:
                log.error(f"=> giving up on {path} after {n} attempts: {e}")
                return False
            log.warning(f"=> save of {path} failed (attempt {n}: {e}); retrying in {wait:.0f}s")
            time.sleep(wait)


@hydra.main(version_base=None, config_path="../config", config_name="train_stage3")
def main(cfg: DictConfig):
    # Print/log the current complete configuration
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.seed)
    device = cfg.device

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    # Wandb init
    wandb_run = None
    if cfg.get("wandb", {}).get("enable", False):
        wandb_mode = "offline" if cfg.wandb.get("offline", False) else "online"
        wandb_run = wandb.init(
            project=cfg.wandb.get("project", "stage3-train"),
            name=cfg.wandb.get("name", os.path.basename(out_dir)),
            tags=cfg.wandb.get("tags", None),
            notes=cfg.wandb.get("notes", None),
            dir=out_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
            resume="allow",
        )
        if cfg.wandb.get("log_code", True):
            wandb.run.log_code(".")

    # Build dataset & loader
    dataset = build_dataset(cfg.dataset)
    dataset.save_metadata(out_dir)
    loader = build_loader(dataset, cfg.dataset, device)

    # Build model/optimizer/scheduler/loss
    # ---- mixed precision -------------------------------------------------
    _amp_mode = str(cfg.train.get("amp", "off")).lower()
    if _amp_mode in ("bf16", "bfloat16"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("train.amp=bf16 but this GPU does not support bfloat16")
        log.info("mixed precision: bf16 autocast ENABLED")
        def _amp_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # TF32 for the fp32 ops that remain (optimizer, LayerNorm reductions).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif _amp_mode in ("off", "none", "fp32"):
        from contextlib import nullcontext
        def _amp_ctx():
            return nullcontext()
    else:
        raise ValueError(f"train.amp must be bf16 or off, got {_amp_mode!r}")

    model = build_model(cfg.models.stage3).to(device)
    iters_per_epoch = (
        len(loader) if cfg.train.iters_per_epoch == 0 else cfg.train.iters_per_epoch
    )
    cfg.optim = cfg.optim.stage3
    # Sized in OPTIMIZER steps, not batches -- see sched_steps_per_epoch.
    optim, sched = build_optim_sched(
        model, cfg.optim, cfg.train.epochs, sched_steps_per_epoch(cfg.train, loader)
    )

    # Count model parameters
    count_params_B(model)

    if wandb_run is not None and cfg.wandb.get("watch", False):
        wandb.watch(
            model,
            log=cfg.wandb.get("watch_log", "gradients"),
            log_freq=cfg.wandb.get("watch_freq", 100),
        )

    global_iters = 0
    start_epoch = 1

    # Resume support: checkpoints carry optimizer/scheduler/RNG so a resumed run
    # continues the same schedule and data stream.
    resume_from = cfg.train.get("resume_from", None)
    if resume_from == "auto":
        # last.pth is written every epoch and is always the newest state; the
        # numbered checkpoints only start at skip_epochs.
        last = os.path.join(out_dir, "last.pth")
        if os.path.exists(last):
            resume_from = last
        else:
            existing = sorted(glob.glob(os.path.join(out_dir, "epoch_*.pth")))
            resume_from = existing[-1] if existing else None
    if resume_from:
        log.info(f"=> resuming from {resume_from}")
        ck = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        if "optimizer" not in ck:
            # Refuse to "resume" without optimizer state: that would restart the
            # LR schedule at epoch 1 with warm weights.
            raise RuntimeError(
                f"{resume_from} has no optimizer state. Resuming from it would "
                "restart the LR schedule at epoch 1 with pre-trained weights, "
                "which is not a continuation. Delete it or train from scratch "
                "into a fresh output_dir."
            )
        optim.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        global_iters = ck.get("global_iters", 0)
        start_epoch = ck["epoch"] + 1
        restored = load_rng_state(ck.get("rng"))
        log.info(f"=> resumed at epoch {start_epoch}, rng_restored={restored}")
        if not restored:
            log.warning("=> no RNG state in checkpoint: the shuffle and augmentation "
                        "stream restarts, so this run is NOT equivalent to an "
                        "uninterrupted one")

    # Train loop
    # Fine-tuning: weights only (strict=False, so new heads start fresh), fresh
    # optimizer and schedule. Ignored once a resume checkpoint exists in out_dir,
    # so resume_from=auto keeps working across relaunches.
    init_from = cfg.train.get("init_from", None)
    if init_from and not resume_from:
        ck = torch.load(init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        frozen = ("mkl.dino_encoder.", "mkl.text_encoder.")
        log.info(f"=> initialised weights from {init_from} (epoch {ck.get('epoch')}), fresh optimizer/schedule")
        log.info(f"   missing (new) keys: {[k for k in missing if not k.startswith(frozen)]}; unexpected: {list(unexpected)}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        # Kept on the GPU: a per-step .item() would sync the host every iteration.
        total_loss = torch.zeros((), device=device)
        n_seen, iters = 0, 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()

        grad_norm = 0.0

        for step, batch in enumerate(loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # Mixed precision, OPT-IN via train.amp (default "off" so every arm
            # trained before 2026-09-05 reproduces bit-for-bit).
            #
            # Why: measured on the perslot stage-2 run, the A100 sat at ~55%
            # "utilisation" with 2165 MiB of 81920 used (2.6%) and CPU load 8/32
            # -- nothing was saturated. Everything ran fp32, so the tensor cores
            # were idle; DINOv2 ViT-B is exactly the dense-matmul workload they
            # exist for. bf16 needs no GradScaler (unlike fp16): its exponent
            # range matches fp32, so there is nothing to underflow.
            with _amp_ctx():
                raw_loss = model(batch, causal_attn=False)
            # Backward OUTSIDE autocast: gradients accumulate in fp32 master
            # weights, which is what keeps bf16 training numerically equivalent.
            loss = raw_loss / max(1, cfg.train.grad_accum_steps)
            loss.backward()

            if (step + 1) % cfg.train.grad_accum_steps == 0:
                if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train.grad_clip_norm
                    ).item()
                else:
                    grad_norm = get_grad_norm(model)

                optim.step()
                optim.zero_grad(set_to_none=True)

                if cfg.optim["T_max_factor"]:
                    sched.step()
                iters += 1
                global_iters += 1

            bs = batch["state"].shape[0]
            total_loss += raw_loss.detach() * bs
            n_seen += bs

            if (
                cfg.train.log_interval > 0
                and iters % max(1, cfg.train.log_interval) == 0
            ):
                log_payload = {
                    "epoch": epoch,
                    "iter": iters,
                    "global_iter": global_iters,
                    "lr": optim.param_groups[0]["lr"],
                    "loss": raw_loss.item(),
                    "grad_norm": grad_norm,
                }
                for k, v in (getattr(model, "last_aux", None) or {}).items():
                    log_payload[k] = float(v)
                log.info(json.dumps(log_payload, ensure_ascii=False))

                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss_step": raw_loss.item(),
                            "train/lr": optim.param_groups[0]["lr"],
                            "train/grad_norm": grad_norm,
                            "train/epoch": epoch,
                        },
                        step=global_iters,
                    )

            if cfg.train.iters_per_epoch > 0 and iters >= cfg.train.iters_per_epoch:
                break

        # Flush the trailing partial accumulation window; otherwise its gradients
        # are discarded by the zero_grad at the top of the next epoch.
        if cfg.train.grad_accum_steps > 1 and (step + 1) % cfg.train.grad_accum_steps != 0:
            if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip_norm
                ).item()
            else:
                grad_norm = get_grad_norm(model)
            optim.step()
            optim.zero_grad(set_to_none=True)
            if cfg.optim["T_max_factor"]:
                sched.step()

        if not cfg.optim["T_max_factor"]:
            sched.step()

        avg_loss = total_loss.item() / max(1, n_seen)
        epoch_time = time.time() - t0

        epoch_payload = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time": epoch_time,
            "lr": optim.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }
        log.info(json.dumps(epoch_payload, ensure_ascii=False))

        if wandb_run is not None:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/epoch_time": epoch_time,
                    "train/lr_epoch": optim.param_groups[0]["lr"],
                    "train/grad_norm_epoch": grad_norm,
                },
                step=global_iters,
            )

        # Save weights/checkpoints
        is_after_skip = epoch >= cfg.train.skip_epochs
        is_checkpoint_interval = epoch % cfg.train.ckpt_interval == 0
        is_last_epoch = epoch == cfg.train.epochs

        full_state = model.state_dict()
        exclude_prefixes = [
            "mkl.dino_encoder.",
            "mkl.text_encoder."
        ]
        filtered_state = {
            k: v for k, v in full_state.items()
            if not any(k.startswith(prefix) for prefix in exclude_prefixes)
        }
        payload = {
            "epoch": epoch,
            "model": filtered_state,
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "global_iters": global_iters,
            "rng": rng_state(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }

        # Rolling per-epoch checkpoint (numbered ones only start at skip_epochs).
        # Temp-file + rename so a crash mid-write cannot leave a truncated file; writes are
        # retried while the shared quota is exhausted (see save_checkpoint).
        last_path = os.path.join(out_dir, "last.pth")
        save_checkpoint(payload, last_path)

        if is_after_skip and (is_checkpoint_interval or is_last_epoch):
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch:04d}.pth")
            saved = save_checkpoint(payload, ckpt_path, tries=None if is_last_epoch else 40)
            if saved:
                log.info(f"=> saved {ckpt_path}")

            if saved and wandb_run is not None and cfg.wandb.get("log_ckpt", False):
                artifact = wandb.Artifact(
                    name=f"ckpt-epoch-{epoch:04d}",
                    type="model",
                    metadata={"epoch": epoch},
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()