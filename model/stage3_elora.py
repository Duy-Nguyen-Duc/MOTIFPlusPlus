import copy
import logging
from typing import Dict, List, Sequence

import contextlib
import warnings

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from model.elora import ELoRAContext, ELoRARouting, inject_elora, elora_parameters
from model.stage1_graph import GraphRoutedMotifLearning
from model.stage2_3 import MotifRoboticPolicy

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _deterministic_kernels():
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


class GraphELoRARouter(nn.Module):
    """g_r, node embeddings and joint gates from a trainable copy of the stage-1 GNN."""

    def __init__(self, teacher: GraphRoutedMotifLearning, gate_hidden: int = 64):
        super().__init__()
        self.gnn = copy.deepcopy(teacher.gnn)
        self.task_embed = copy.deepcopy(teacher.task_embed)
        if getattr(self.task_embed, "proj", None) is not None:
            # Frozen T5 features are identical for both: reuse the teacher's encoder/cache.
            self.task_embed._encode = teacher.task_embed._encode
        for p in self.parameters():
            p.requires_grad_(True)
        self.register_buffer("emb2graph", teacher.emb2graph.clone(), persistent=False)
        hidden = self.gnn.in_proj.out_features
        self.gate = nn.Sequential(nn.Linear(hidden, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.num_nodes = int(self.gnn.gid.numel())

    def forward(self, embodiment_id: torch.Tensor, languages: List[str]):
        device = embodiment_id.device
        uniq = list(dict.fromkeys(languages))
        e_task = self.task_embed(uniq, device)
        with _deterministic_kernels():
            outs = [self.gnn(e_task[u]) for u in range(len(uniq))]
        g_table = torch.stack([o[0] for o in outs])                 # [U, R, Dg]
        n_table = torch.stack([o[1] for o in outs])                 # [U, N, Dg]
        u_idx = torch.tensor([uniq.index(l) for l in languages], device=device)
        r_idx = self.emb2graph[embodiment_id.long().view(-1)]
        if (r_idx < 0).any():
            raise ValueError(f"no kinematic graph for embodiment ids "
                             f"{embodiment_id.view(-1)[r_idx < 0].unique().tolist()}")
        g = g_table[u_idx, r_idx]                                   # [B, Dg]
        n = n_table[u_idx]                                          # [B, N, Dg]
        on_robot = (self.gnn.gid.unsqueeze(0) == r_idx.unsqueeze(1)).to(n.dtype)   # [B, N]
        gates = torch.sigmoid(self.gate(n).squeeze(-1)) * on_robot
        return g, gates, n


class MotifRoboticPolicyELoRA(MotifRoboticPolicy):
    def __init__(self, elora_rank: int = 4, elora_targets: Sequence[str] = ("ff.net.0.proj", "ff.net.2"),
                 elora_scale: float = 1.0, elora_freeze_base: bool = False, elora_base_ckpt: str = None,
                 elora_gate_hidden: int = 64, **kwargs):
        super().__init__(**kwargs)
        teacher = self.mkl.vq_vae
        if not isinstance(teacher, GraphRoutedMotifLearning):
            raise TypeError("MOTIFStage3ELoRA needs a graph-routed stage-1 teacher (MOTIFStage1Graph)")
        self.router = GraphELoRARouter(teacher, gate_hidden=elora_gate_hidden)
        self.elora_ctx = ELoRAContext()
        self.elora_targets = list(elora_targets)
        self.elora_layers = inject_elora(self.flow_matching_head.model, self.elora_targets, self.elora_ctx,
                                         num_skills=teacher.codebook_size, num_nodes=self.router.num_nodes,
                                         rank=elora_rank, scale=elora_scale)
        n_elora = sum(p.numel() for p in elora_parameters(self))
        n_router = sum(p.numel() for p in self.router.parameters())
        log.info("E-LoRA: %d layers, rank %d, %.2fM adapter params, %.2fM router params",
                 len(self.elora_layers), elora_rank, n_elora / 1e6, n_router / 1e6)

        self.elora_freeze_base = elora_freeze_base
        if elora_base_ckpt:
            ck = torch.load(elora_base_ckpt, map_location="cpu")
            # The base checkpoint's DiT linears are now wrapped: remap "<name>.weight" ->
            # "<name>.base.weight" for every E-LoRA site.
            state = {}
            for k, v in ck["model"].items():
                for site in self.elora_layers:
                    pre = f"flow_matching_head.model.{site}."
                    if k.startswith(pre):
                        k = pre + "base." + k[len(pre):]
                        break
                state[k] = v
            missing, unexpected = self.load_state_dict(state, strict=False)
            bad = [k for k in missing if not (k.startswith(("router.", "mkl.dino_encoder.", "mkl.text_encoder."))
                                              or ".A" in k or ".B" in k)]
            if bad or unexpected:
                raise ValueError(f"base ckpt mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")
            log.info("loaded frozen base policy from %s", elora_base_ckpt)
        if elora_freeze_base:
            if not elora_base_ckpt:
                raise ValueError("elora_freeze_base=true needs elora_base_ckpt")
            for p in self.parameters():
                p.requires_grad_(False)
            for p in elora_parameters(self):
                p.requires_grad_(True)
            for p in self.router.parameters():
                p.requires_grad_(True)

    # -- routing ------------------------------------------------------------------
    def _set_routing(self, key: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            alpha = self.mkl.vq_vae.vq.logits(key).softmax(-1).mean(1)        # [B, K]
        g, gates, _ = self.router(batch["embodiment_id"], batch["language"])
        self.elora_ctx.routing = ELoRARouting(alpha=alpha, gate=gates)
        return g

    def _clear_routing(self):
        self.elora_ctx.routing = None

    def _inputs(self, batch, train: bool):
        encoded = self.mkl.encode_raw(batch)
        obs_emb, masks = self.get_observation_embedding(batch, encoded=encoded)
        key, _, _ = self.mkl.get_key_from_raw(*encoded, state=batch.get("state"),
                                              state_mask=batch.get("state_mask"),
                                              embodiment_id=batch.get("embodiment_id"))
        if train and self.training and self.key_noise_std > 0:
            key = key + self.key_noise_std * key.std() * torch.randn_like(key)
        self._set_routing(key, batch)
        latent_action = self._latent_from_key(key)
        state = batch["state"]
        if train and self.training and self.state_noise_std > 0:
            noise = self.state_noise_std * torch.randn_like(state)
            if "state_mask" in batch:
                noise = noise * batch["state_mask"]
            state = state + noise
        inp1 = BatchFeature(data={"backbone_features": obs_emb, "backbone_attention_mask": ~masks.bool()})
        data = {"embodiment_id": batch["embodiment_id"], "state": state, "latent_action": latent_action}
        if train:
            data.update(action=batch["action"], action_mask=batch["action_mask"])
        return inp1, BatchFeature(data=data)

    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        if not self.add_latent_action:
            raise ValueError("MOTIFStage3ELoRA needs add_latent_action=true (routing comes from the key)")
        inp1, inp2 = self._inputs(batch, train=True)
        try:
            return self.flow_matching_head(inp1, inp2).loss
        finally:
            self._clear_routing()

    @torch.no_grad()
    def get_action(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        inp1, inp2 = self._inputs(batch, train=False)
        try:
            return self.flow_matching_head.get_action(inp1, inp2)
        finally:
            self._clear_routing()
