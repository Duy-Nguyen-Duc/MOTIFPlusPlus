import os
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

@hydra.main(version_base=None, config_path="../config", config_name="train_stage2")
def main(cfg: DictConfig):
    # Print/Log the full current configuration
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.seed)
    device = cfg.device

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    # >>> wandb init
    wandb_run = None
    if cfg.get("wandb", {}).get("enable", False):
        wandb_mode = "offline" if cfg.wandb.get("offline", False) else "online"
        wandb_run = wandb.init(
            project=cfg.wandb.get("project", "stage2-train"),
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
    loader = build_loader(dataset, cfg.dataset, device)

    # Build model/optimizer/scheduler/loss
    # ---- mixed precision (opt-in, default off) ---------------------------
    # Measured 2026-09-05 on the perslot stage-2 run: A100 at ~55% utilisation,
    # 2165 MiB of 81920 used (2.6%), CPU load 8/32 -- nothing saturated, because
    # everything ran fp32 (tensor cores idle) with batch 8 accumulated 16x.
    # bf16 needs no GradScaler: its exponent range matches fp32.
    _amp_mode = str(cfg.train.get("amp", "off")).lower()
    if _amp_mode in ("bf16", "bfloat16"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("train.amp=bf16 but this GPU does not support bfloat16")
        log.info("mixed precision: bf16 autocast ENABLED")
        def _amp_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif _amp_mode in ("off", "none", "fp32"):
        from contextlib import nullcontext
        def _amp_ctx():
            return nullcontext()
    else:
        raise ValueError(f"train.amp must be bf16 or off, got {_amp_mode!r}")

    model = build_model(cfg.models.stage2).to(device)
    iters_per_epoch = (
        len(loader) if cfg.train.iters_per_epoch == 0 else cfg.train.iters_per_epoch
    )
    cfg.optim = cfg.optim.stage2
    # Sized in OPTIMIZER steps, not batches -- see sched_steps_per_epoch.
    optim, sched = build_optim_sched(
        model, cfg.optim, cfg.train.epochs, sched_steps_per_epoch(cfg.train, loader)
    )

    # Count model parameters
    count_params_B(model)

    # >>> wandb watch (optional)
    if wandb_run is not None and cfg.wandb.get("watch", False):
        wandb.watch(
            model,
            log=cfg.wandb.get("watch_log", "gradients"),
            log_freq=cfg.wandb.get("watch_freq", 100),
        )

    global_iters = 0
    start_epoch = 1

    # Resume support: checkpoints carry optimizer/scheduler/RNG so a resumed run
    # continues the same schedule and data stream. Stage 2 has no last.pth --
    # only the interval checkpoints exist, so "auto" picks the newest of those.
    resume_from = cfg.train.get("resume_from", None)
    if resume_from == "auto":
        existing = sorted(glob.glob(os.path.join(out_dir, "epoch_*.pth")))
        resume_from = existing[-1] if existing else None
    if resume_from:
        log.info(f"=> resuming from {resume_from}")
        ck = torch.load(resume_from, map_location="cpu")
        # strict=False: checkpoints hold only stage-2 weights; the frozen
        # vq_vae/dino/text encoders were rebuilt from their own ckpts above.
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

    # train loop
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        total_loss, n_seen, iters = 0.0, 0, 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()

        grad_norm = 0.0 

        for step, batch in enumerate(loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            with _amp_ctx():
                raw_loss = model(batch, causal_attn=False)

            # Gradient accumulation
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
            total_loss += raw_loss.item() * bs
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

        avg_loss = total_loss / max(1, n_seen)
        epoch_time = time.time() - t0

        epoch_payload = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time": epoch_time,
            "lr": optim.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }
        log.info(json.dumps(epoch_payload, ensure_ascii=False))

        # >>> wandb epoch log
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

        # Save weights
        if epoch % cfg.train.ckpt_interval == 0 or (epoch == cfg.train.epochs and epoch % cfg.train.ckpt_interval != 0):
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch:04d}.pth")

            # Only save stage2 weights
            full_state = model.state_dict()
            exclude_prefixes = [
                "vq_vae.",
                "dino_encoder.",
                "text_encoder."
            ]
            filtered_state = {
                k: v for k, v in full_state.items()
                if not any(k.startswith(prefix) for prefix in exclude_prefixes)
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model": filtered_state,
                    "optimizer": optim.state_dict(),
                    "scheduler": sched.state_dict(),
                    "global_iters": global_iters,
                    "rng": rng_state(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            log.info(f"=> saved {ckpt_path}")

            # >>> wandb optional ckpt upload
            if wandb_run is not None and cfg.wandb.get("log_ckpt", False):
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