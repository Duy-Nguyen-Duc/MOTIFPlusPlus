import os
import glob
import time
import json
import logging
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn.functional as F
import wandb

from model.stage1 import compute_cross_robot_infonce, LabelRemapper
from model.stage1_graph import effect_loss
from model.stage1_factorized import ConditionalEmbodimentDiscriminator
from model.stage1_graph import build_model, TASK_IDS
from model.primitive_tying import PRIMITIVES, frames_to_tokens
from data.embodiment_tags import EMBODIMENT_TAG_MAPPING

from train_utils import (
    set_seed, build_dataset, build_loader, build_optim_sched, sched_steps_per_epoch,
    rng_state, load_rng_state, count_params_B, get_grad_norm,
)

log = logging.getLogger(__name__)


class RoutingReport:
    """Per-epoch: code usage, alpha sharpness, soft/hard recon gap, code purity."""

    def __init__(self, codebook_size: int, num_primitives: int, device):
        self.K, self.P = codebook_size, num_primitives
        self.code_prim = torch.zeros(codebook_size, num_primitives, device=device)
        self.reset()

    def reset(self):
        self.code_prim.zero_()
        self.sums = {"entropy": 0.0, "alpha_max": 0.0, "balance": 0.0, "recon": 0.0, "recon_hard": 0.0,
                     "effect": 0.0, "effect_r2": 0.0}
        self.n = 0

    @torch.no_grad()
    def update(self, out, target, prim_tok=None):
        self.n += 1
        self.sums["entropy"] += float(out["routing_entropy"])
        self.sums["alpha_max"] += float(out["alpha_max"])
        self.sums["balance"] += float(out["load_balance"])
        self.sums["recon"] += float(((out["recon"] - target) ** 2).mean())
        if "recon_hard" in out:
            self.sums["recon_hard"] += float(((out["recon_hard"] - target) ** 2).mean())
        if prim_tok is not None:
            flat = out["idx"].reshape(-1) * self.P + prim_tok.reshape(-1)
            self.code_prim += torch.bincount(flat, minlength=self.K * self.P).view(self.K, self.P).float()

    def summary(self):
        s = {k: v / max(self.n, 1) for k, v in self.sums.items()}
        total = self.code_prim.sum()
        if total > 0:
            usage = self.code_prim.sum(1)
            p = usage / total
            s["codes_used"] = int((usage > 0).sum())
            s["perplexity"] = float((-(p * (p + 1e-10).log()).sum()).exp())
            s["code_purity"] = float(self.code_prim.max(1).values.sum() / total)
            per_prim = {}
            for j, name in enumerate(PRIMITIVES):
                col = self.code_prim[:, j]
                if col.sum() == 0:
                    continue
                srt = col.sort(descending=True).values.cumsum(0) / col.sum()
                per_prim[name] = {"tokens": int(col.sum()), "codes_90pct": int((srt < 0.9).sum() + 1),
                                  "top_code": int(col.argmax())}
            s["per_primitive"] = per_prim
            top = self.code_prim.argmax(1)
            s["code_majority_primitive"] = {int(k): PRIMITIVES[int(top[k])] for k in range(self.K) if usage[k] > 0}
        return s


@hydra.main(version_base=None, config_path="../config", config_name="train_stage1_graph")
def main(cfg: DictConfig):
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    set_seed(cfg.seed)
    device = cfg.device
    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)
    code_usage_path = os.path.join(out_dir, "code_usage")
    os.makedirs(code_usage_path, exist_ok=True)
    report_dir = os.path.join(out_dir, "routing")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    wandb_run = None
    if cfg.get("wandb", {}).get("enable", False):
        wandb_run = wandb.init(project=cfg.wandb.get("project", "stage1-train"),
                               name=cfg.wandb.get("name", os.path.basename(out_dir)),
                               tags=cfg.wandb.get("tags", None), notes=cfg.wandb.get("notes", None),
                               dir=out_dir, config=OmegaConf.to_container(cfg, resolve=True),
                               mode="offline" if cfg.wandb.get("offline", False) else "online",
                               resume="allow")

    dataset = build_dataset(cfg.dataset)
    loader = build_loader(dataset, cfg.dataset, device)
    model = build_model(cfg.models.stage1).to(device)

    prim_cond = bool(cfg.get("prim_conditioning", False))
    infonce_cond = cfg.get("infonce_cond", "primitive" if prim_cond else "task")
    if infonce_cond not in ("task", "primitive"):
        raise ValueError(f"infonce_cond must be task|primitive, got {infonce_cond}")
    infonce_sigma = float(cfg.get("infonce_sigma", 0.1 if infonce_cond == "primitive" else 0.02))
    infonce_min_weight = float(cfg.get("infonce_min_weight", 1e-3 if infonce_cond == "primitive" else 0.0))

    if cfg.get("use_adversarial", False):
        model.discriminator = ConditionalEmbodimentDiscriminator(
            input_dim=cfg.models.stage1.latent_dim, num_robots=cfg.get("num_robots", 2),
            num_tasks=len(PRIMITIVES) if prim_cond else len(TASK_IDS),
            num_progress_bins=cfg.get("num_progress_bins", 4)).to(device)
        valid_ids = [EMBODIMENT_TAG_MAPPING[tag] for tag in cfg.dataset["embodiment_tag"]]
        remapper = LabelRemapper(valid_ids, device=device)

    cfg.optim = cfg.optim.stage1
    optim, sched = build_optim_sched(model, cfg.optim, cfg.train.get("scheduler_epochs", cfg.train.epochs),
                                     sched_steps_per_epoch(cfg.train, loader))
    count_params_B(model)
    report = RoutingReport(model.codebook_size, model.num_primitives, device)

    global_iters, start_epoch = 0, 1
    resume_from = cfg.train.get("resume_from")
    if resume_from == "auto":                  # last.pth, else the newest interval checkpoint
        last = os.path.join(out_dir, "last.pth")
        existing = sorted(glob.glob(os.path.join(out_dir, "epoch_*.pth")))
        resume_from = last if os.path.exists(last) else (existing[-1] if existing else None)
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optim.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        global_iters = checkpoint["global_iters"]
        start_epoch = checkpoint["epoch"] + 1
        load_rng_state(checkpoint["rng"])
        log.info("Resuming epoch %s, optimizer step %s", start_epoch, global_iters)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        total_loss, n_seen, iters = 0.0, 0, 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()
        grad_norm = 0.0
        epoch_counts = torch.zeros(model.codebook_size, device=device)
        infonce_sum, infonce_n = 0.0, 0
        report.reset()

        for step, batch in enumerate(loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            out = model(batch, causal_attn=False)
            target = batch["rel_state"]
            outputs = model.compute_loss(out["recon"], target)
            h, indices = out["h"], out["idx"]

            # routing regularisers
            frac = epoch - 1 + (step + 1) / len(loader)
            ent_ramp = min(1.0, frac / max(float(cfg.get("entropy_ramp_epochs", 3)), 1e-8))
            outputs["loss"] = (outputs["loss"]
                               + float(cfg.get("lambda_entropy", 0.0)) * ent_ramp * out["routing_entropy"]
                               + float(cfg.get("lambda_balance", 0.0)) * out["load_balance"])

            # causal effect loss (off in this experiment)
            lambda_effect = float(cfg.get("lambda_effect", 0.0))
            if lambda_effect > 0:
                if "obj_rel" not in batch:
                    raise ValueError("lambda_effect > 0 needs batch['obj_rel'] (dataset=dataset_maniskill_prim_obj)")
                eff, eff_r2 = effect_loss(out["effect_pred"], batch["obj_rel"], batch.get("obj_valid"),
                                          scale=model.effect_scale, clip=float(cfg.get("effect_clip", 1.0)))
                outputs["loss"] += lambda_effect * eff
                outputs["effect_loss"], outputs["effect_r2"] = eff.detach(), eff_r2
                report.sums["effect"] += float(eff); report.sums["effect_r2"] += float(eff_r2)

            # primitive labels per token
            prim_tok = out.get("prim_tok")
            if prim_cond and prim_tok is None:
                raise ValueError("prim_conditioning=true needs batch['prim_id'] (dataset_maniskill_prim)")
            prim_prog_tok = prim_center = prim_prog_center = None
            if prim_tok is not None:
                B_, M_ = prim_tok.shape
                prim_prog_tok = frames_to_tokens(batch["prim_prog"].view(B_, -1), M_)
                prim_center = prim_tok[:, M_ // 2].tolist()
                prim_prog_center = prim_prog_tok[:, M_ // 2: M_ // 2 + 1]
            report.update(out, target, prim_tok)

            # cross-robot InfoNCE on the morphology-free tokens
            infonce_loss = None
            if cfg.add_info_NCE:
                if infonce_cond == "primitive":
                    nce_labels, nce_progress = prim_center, prim_prog_center
                else:
                    nce_labels, nce_progress = batch["language"], batch["progress"]
                infonce_loss, _ = compute_cross_robot_infonce(
                    z_e=h, languages=nce_labels, progress=nce_progress, tau=0.1,
                    sigma=infonce_sigma, hard_delta=cfg.get("hard_delta", None),
                    min_weight=infonce_min_weight)
                outputs["loss"] += cfg.lambda_infonce * infonce_loss
                outputs["infonce_loss"] = infonce_loss

            # embodiment adversary on the morphology-free tokens
            if cfg.get("use_adversarial", False):
                p = float(global_iters) / (cfg.train.get("scheduler_epochs", cfg.train.epochs) * len(loader))
                alpha_grl = 2.0 / (1.0 + math.exp(-10 * p)) - 1
                disc_input = h.reshape(-1, h.size(-1))
                if prim_cond:
                    task_ids, prog_tok = prim_tok.reshape(-1), prim_prog_tok.reshape(-1)
                else:
                    task_ids = torch.tensor([TASK_IDS.get(l, len(TASK_IDS)) for l in batch["language"]],
                                            device=device).unsqueeze(1).repeat(1, h.size(1)).view(-1)
                    prog_tok = batch["progress"].mean(1).unsqueeze(1).repeat(1, h.size(1)).view(-1)
                disc_logits = model.discriminator(disc_input, task_ids, prog_tok, alpha=alpha_grl)
                target_ids = remapper(batch["embodiment_id"].long()).unsqueeze(1).repeat(1, h.size(1)).view(-1)
                adv_loss = F.cross_entropy(disc_logits, target_ids)
                outputs["loss"] += cfg.get("lambda_adv", 0.05) * adv_loss
                outputs["adv_loss"] = adv_loss.detach()
                outputs["adv_acc"] = (disc_logits.argmax(1) == target_ids).float().mean().detach()

            code_usage = model.compute_code_usage(indices)
            perplexity_step = model.compute_perplexity(indices)
            topk_step = model.compute_topk_usage(indices, ks=(10, 20))
            with torch.no_grad():
                epoch_counts += model._counts_from_indices(indices)
            raw_loss = outputs["loss"]
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}")
            (raw_loss / max(1, cfg.train.grad_accum_steps)).backward()

            if (step + 1) % cfg.train.grad_accum_steps == 0:
                if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm).item()
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
            if infonce_loss is not None:
                infonce_sum += float(infonce_loss) * bs
                infonce_n += bs

            if cfg.train.log_interval > 0 and iters % max(1, cfg.train.log_interval) == 0:
                payload = {
                    "epoch": epoch, "iter": iters, "global_iter": global_iters,
                    "lr": optim.param_groups[0]["lr"], "loss": raw_loss.item(),
                    "recon_loss": outputs["recon_loss"].item(),
                    "routing_entropy": float(out["routing_entropy"]),
                    "alpha_max": float(out["alpha_max"]),
                    "load_balance": float(out["load_balance"]),
                    "code_usage": code_usage.item(), "perplexity": perplexity_step.item(),
                    "top10_frac": topk_step["top10_frac"].item(), "top20_frac": topk_step["top20_frac"].item(),
                    "grad_norm": grad_norm,
                }
                if "recon_hard" in out:
                    payload["recon_hard"] = float(((out["recon_hard"] - target) ** 2).mean())
                if infonce_loss is not None:
                    payload["infonce_loss"] = float(infonce_loss)
                if "effect_loss" in outputs:
                    payload["effect_loss"] = float(outputs["effect_loss"])
                    payload["effect_r2"] = float(outputs["effect_r2"])
                if "adv_loss" in outputs:
                    payload["adv_loss"] = float(outputs["adv_loss"])
                    payload["adv_acc"] = float(outputs["adv_acc"])
                log.info(json.dumps(payload, ensure_ascii=False))
                if wandb_run is not None:
                    wandb.log({f"train/{k}_step": v for k, v in payload.items()}, step=global_iters)

            if cfg.train.iters_per_epoch > 0 and iters >= cfg.train.iters_per_epoch:
                break

        if cfg.train.grad_accum_steps > 1 and (step + 1) % cfg.train.grad_accum_steps != 0:
            if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm).item()
            optim.step()
            global_iters += 1
            optim.zero_grad(set_to_none=True)
            if cfg.optim["T_max_factor"]:
                sched.step()
        if not cfg.optim["T_max_factor"]:
            sched.step()

        with torch.no_grad():
            p_epoch = epoch_counts / epoch_counts.sum().clamp_min(1.0)
            perplexity_epoch = (-(p_epoch * (p_epoch + 1e-10).log()).sum()).exp()
        summary = report.summary()
        with open(os.path.join(report_dir, f"epoch_{epoch:04d}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        epoch_payload = {
            "epoch": epoch, "avg_loss": total_loss / max(1, n_seen), "time": time.time() - t0,
            "lr": optim.param_groups[0]["lr"], "grad_norm": grad_norm,
            "perplexity_epoch": perplexity_epoch.item(),
            "codes_used": summary.get("codes_used"),
            "routing_entropy_epoch": summary["entropy"], "alpha_max_epoch": summary["alpha_max"],
            "recon_epoch": summary["recon"], "recon_hard_epoch": summary["recon_hard"],
            "code_purity": summary.get("code_purity"),
        }
        if infonce_n:
            epoch_payload["infonce_loss_epoch"] = infonce_sum / infonce_n
        log.info(json.dumps(epoch_payload, ensure_ascii=False))
        if wandb_run is not None:
            wandb.log({f"train/{k}_epoch": v for k, v in epoch_payload.items() if v is not None}, step=global_iters)

        model.plot_and_reset_code_usage(epoch, code_usage_path)

        payload = {"epoch": epoch, "model": model.state_dict(), "optimizer": optim.state_dict(),
                   "scheduler": sched.state_dict(), "global_iters": global_iters, "rng": rng_state(),
                   "cfg": OmegaConf.to_container(cfg, resolve=True)}
        # rolling checkpoint for resume_from=auto (write + rename)
        last_path = os.path.join(out_dir, "last.pth")
        torch.save(payload, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        if epoch % cfg.train.ckpt_interval == 0 or epoch == cfg.train.epochs:
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch:04d}.pth")
            torch.save(payload, ckpt_path)
            log.info(f"=> saved {ckpt_path}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
