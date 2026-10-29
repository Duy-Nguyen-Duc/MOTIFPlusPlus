from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from model.kinematic_graph import KinematicGraph, load_graphs
from model.modules.blocks import VectorQuantizer
from model.primitive_tying import frames_to_tokens
from model.stage1 import ActionMotifLearning

# Instruction -> id for the table text encoder (same order as the trainers).
TASK_IDS = {
    "Push the cube across the surface to the target location.": 0,
    "Pick up the sphere and place it precisely into the target.": 1,
    "Pull the cube toward the target marker.": 2,
    "Lift the peg and stand it upright.": 3,
    "Pick up the cube.": 4,
    "Stack the cube on top of the other cube.": 5,
}
DEFAULT_ROBOTS = {"panda": "panda_rel", "xarm6": "xarm6_rel", "widowxai": "widowxai_rel"}


# --------------------------------------------------------------------------- text
class TaskTextEmbedder(nn.Module):
    """
    e_task = L(instruction) from a frozen encoder, then a trainable projection.
    """

    def __init__(self, out_dim: int, text_encoder: str = "t5-base"):
        super().__init__()
        self.text_encoder = text_encoder
        if text_encoder == "table":
            self.table = nn.Embedding(len(TASK_IDS) + 1, out_dim)
            self.proj = None
        else:
            self.proj = nn.Linear(768, out_dim)
            self._lm = None                       # [model, tokenizer], lazily loaded
            self._cache: Dict[str, torch.Tensor] = {}

    def _encode(self, sentences: List[str], device) -> torch.Tensor:
        missing = [s for s in dict.fromkeys(sentences) if s not in self._cache]
        if missing:
            if self._lm is None:
                from transformers import T5EncoderModel, T5Tokenizer
                lm = T5EncoderModel.from_pretrained(self.text_encoder).eval()
                for p in lm.parameters():
                    p.requires_grad = False
                self._lm = [lm, T5Tokenizer.from_pretrained(self.text_encoder)]
            lm, tok = self._lm
            lm.to(device)
            with torch.no_grad():
                enc = tok(missing, return_tensors="pt", padding=True).to(device)
                hid = lm(**enc).last_hidden_state                       # [n, T, 768]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hid * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            for s, v in zip(missing, pooled):
                self._cache[s] = v.detach()
        return torch.stack([self._cache[s].to(device) for s in sentences])

    def forward(self, sentences: List[str], device) -> torch.Tensor:
        if self.text_encoder == "table":
            ids = torch.tensor([TASK_IDS.get(s, len(TASK_IDS)) for s in sentences], device=device)
            return self.table(ids)
        return self.proj(self._encode(sentences, device))


# --------------------------------------------------------------------------- GNN
class MPNNLayer(nn.Module):
    """
    Message Passing Neural Network Layer.
    """

    def __init__(self, hidden: int, d_edge: int, d_task: int, layer_norm: bool = True):
        super().__init__()
        self.w_self = nn.Linear(hidden, hidden)
        self.msg = nn.Sequential(nn.Linear(2 * hidden + d_edge + d_task, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden) if layer_norm else nn.Identity()

    def forward(self, n, src, dst, e, e_task_nodes):
        # message j -> i along edge (src=j, dst=i); N(i) = all j with an edge into i
        m = self.msg(torch.cat([n[dst], n[src], e, e_task_nodes[dst]], dim=-1))
        agg = torch.zeros(n.shape, dtype=m.dtype, device=n.device).index_add_(0, dst, m)   # m may be bf16 under autocast
        return self.norm(F.relu(self.w_self(n) + agg))


class TaskConditionedMPNN(nn.Module):
    """L message-passing layers over the disjoint union of the robot graphs, then
    mean-pool per graph -> g_r [R, hidden] for one task embedding."""

    def __init__(self, graphs: List[KinematicGraph], d_task: int, hidden: int = 128,
                 layers: int = 3, layer_norm: bool = True):
        super().__init__()
        x, src, dst, e, gid = [], [], [], [], []
        offset = 0
        for g_i, g in enumerate(graphs):
            x.append(torch.tensor(g.node_feats, dtype=torch.float32))
            src.append(torch.tensor(g.edge_src) + offset)
            dst.append(torch.tensor(g.edge_dst) + offset)
            e.append(torch.tensor(g.edge_feats, dtype=torch.float32))
            gid.append(torch.full((g.num_nodes,), g_i, dtype=torch.long))
            offset += g.num_nodes
        self.register_buffer("x", torch.cat(x), persistent=False)
        self.register_buffer("src", torch.cat(src), persistent=False)
        self.register_buffer("dst", torch.cat(dst), persistent=False)
        self.register_buffer("e", torch.cat(e), persistent=False)
        self.register_buffer("gid", torch.cat(gid), persistent=False)
        self.num_graphs = len(graphs)
        self.in_proj = nn.Linear(self.x.shape[1], hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden, self.e.shape[1], d_task, layer_norm)
                                     for _ in range(layers)])
        counts = torch.bincount(self.gid, minlength=self.num_graphs).float()
        self.register_buffer("inv_counts", 1.0 / counts.clamp_min(1.0), persistent=False)

    def forward(self, e_task: torch.Tensor) -> torch.Tensor:
        """e_task: [d_task] (one task) -> g: [R, hidden], node embeddings [N, hidden]."""
        n = self.in_proj(self.x)
        e_task_nodes = e_task.unsqueeze(0).expand(n.shape[0], -1)
        for layer in self.layers:
            n = layer(n, self.src, self.dst, self.e, e_task_nodes)
        g = torch.zeros(self.num_graphs, n.shape[1], device=n.device, dtype=n.dtype)
        g = g.index_add(0, self.gid, n) * self.inv_counts.unsqueeze(1)
        return g, n


# --------------------------------------------------------------------------- routing
class RoutingCodebook(VectorQuantizer):
    """Shared codebook addressed by attention instead of nearest neighbour.

    forward(q):  alpha = softmax(q . W_K c / sqrt(d) / tau), z_soft = alpha @ C,
                 idx = argmax alpha, z_hard = C[idx]   (usage counted on idx)
    inference_forward(q): the argmax route, returned as (z_hard, idx, -logits) so the
    stage-2 / stage-3 callers see the VectorQuantizer contract."""

    def __init__(self, num_latents: int, latent_dim: int, temperature: float = 1.0,
                 code_restart: bool = False, logit_scale_init: float = 10.0):
        super().__init__(num_latents, latent_dim, code_restart=code_restart)
        nn.init.normal_(self.codebook.weight, std=0.1)
        self.key_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.temperature = float(temperature)
        self.cosine = True
        self.log_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.scale = 1.0 / math.sqrt(latent_dim)

    def logits(self, q: torch.Tensor) -> torch.Tensor:
        keys = self.key_proj(self.codebook.weight)                     # [K, d]
        if self.cosine:
            sim = torch.einsum("...d,kd->...k", F.normalize(q, dim=-1), F.normalize(keys, dim=-1))
            return sim * self.log_scale.exp() / self.temperature
        return torch.einsum("...d,kd->...k", q, keys) * self.scale / self.temperature

    def forward(self, q: torch.Tensor):
        logits = self.logits(q)
        alpha = logits.softmax(-1)                                     # [..., K]
        z_soft = alpha @ self.codebook.weight
        idx = logits.argmax(-1)
        z_hard = self.codebook(idx)
        if not self.training or self.code_restart:
            self.update_usage(idx)
        return z_soft, z_hard, alpha, idx, logits

    @torch.no_grad()
    def inference_forward(self, q: torch.Tensor):
        logits = self.logits(q)
        idx = logits.argmax(-1)
        return self.codebook(idx), idx, -logits


# --------------------------------------------------------------------------- model
class GraphRoutedMotifLearning(ActionMotifLearning):
    def __init__(
        self,
        state_dim: int = 8,
        model_dim: int = 256,
        latent_dim: int = 256,
        codebook_size: int = 128,
        window_size: int = 32,
        depth_enc: int = 4,
        depth_dec: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_w: int = 16,
        use_sinusoidal_pe: bool = False,
        max_len: int = 2048,
        vq_beta: float = 0.25,
        downsample_factor: int = 2,
        depth_enc_low: int = 0,
        depth_dec_low: int = 0,
        add_embodiment_token: bool = False,
        # graph / routing
        graph_dir: str = "./assets/kinematic_graphs",
        robots: Dict[str, str] = None,          # graph name -> embodiment tag
        graph_hidden: int = 128,
        graph_layers: int = 3,
        graph_layer_norm: bool = True,
        task_dim: int = 64,
        text_encoder: str = "t5-base",
        route_temperature: float = 1.0,
        hard_training: bool = False,            # True: straight-through argmax code to the decoder
        decoder_cond: str = "graph",            # graph | none
        code_restart: bool = False,
        num_primitives: int = 7,
        # causal effect head: Delta o over each token's span from z^skill only
        effect_scale: float = 10.0,             # metres -> decimetres before the MSE
        effect_hidden: int = 256,
    ):
        super().__init__(
            state_dim=state_dim, model_dim=model_dim, latent_dim=latent_dim,
            codebook_size=codebook_size, window_size=window_size, depth_enc=depth_enc,
            depth_dec=depth_dec, num_heads=num_heads, dropout=dropout, local_w=local_w,
            use_sinusoidal_pe=use_sinusoidal_pe, max_len=max_len, vq_beta=vq_beta,
            downsample_factor=downsample_factor, depth_enc_low=depth_enc_low,
            depth_dec_low=depth_dec_low, add_embodiment_token=add_embodiment_token,
        )

        self.forward_for_keys = True
        self.vq = RoutingCodebook(codebook_size, latent_dim, temperature=route_temperature,
                                  code_restart=code_restart)
        robots = dict(robots or DEFAULT_ROBOTS)
        self.robot_names = list(robots.keys())
        graphs = load_graphs(graph_dir, self.robot_names)
        self.gnn = TaskConditionedMPNN([graphs[r] for r in self.robot_names], task_dim,
                                       hidden=graph_hidden, layers=graph_layers,
                                       layer_norm=graph_layer_norm)
        emb2graph = torch.full((max(EMBODIMENT_TAG_MAPPING.values()) + 1,), -1, dtype=torch.long)
        for g_i, (name, tag) in enumerate(robots.items()):
            emb2graph[EMBODIMENT_TAG_MAPPING[tag]] = g_i
        self.register_buffer("emb2graph", emb2graph, persistent=False)
        self.task_embed = TaskTextEmbedder(task_dim, text_encoder=text_encoder)
        self.route_query = nn.Linear(graph_hidden + latent_dim, latent_dim)
        self.decoder_cond = decoder_cond
        if decoder_cond == "graph":
            self.dec_cond = nn.Linear(graph_hidden, latent_dim)
        elif decoder_cond != "none":
            raise ValueError(f"decoder_cond must be graph|none, got {decoder_cond}")
        self.hard_training = hard_training
        self.num_primitives = num_primitives
        self.effect_scale = float(effect_scale)
        self.effect_head = nn.Sequential(nn.Linear(latent_dim, effect_hidden), nn.ReLU(),
                                         nn.Linear(effect_hidden, 3))

    # -- morphology -----------------------------------------------------------
    def morphology(self, embodiment_id: torch.Tensor, languages: List[str]) -> torch.Tensor:
        """g_r per sample [B, graph_hidden]: one GNN pass per distinct instruction in
        the batch over all robot graphs, then gather (robot, task) per sample."""
        device = embodiment_id.device
        uniq = list(dict.fromkeys(languages))
        e_task = self.task_embed(uniq, device)                          # [U, d_task]
        g_table = torch.stack([self.gnn(e_task[u])[0] for u in range(len(uniq))])  # [U, R, Dg]
        u_idx = torch.tensor([uniq.index(l) for l in languages], device=device)
        r_idx = self.emb2graph[embodiment_id.long().view(-1)]
        if (r_idx < 0).any():
            bad = embodiment_id.view(-1)[r_idx < 0].unique().tolist()
            raise ValueError(f"no kinematic graph for embodiment ids {bad}")
        return g_table[u_idx, r_idx]

    # -- forward --------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        progress, x = batch["progress"], batch["rel_state"]
        h = self.encoder(x, progress, causal=causal_attn)               # H_m  [B, M, D]
        g = self.morphology(batch["embodiment_id"], batch["language"])  # g_r  [B, Dg]
        g_tok = g.unsqueeze(1).expand(-1, h.shape[1], -1)
        q = self.route_query(torch.cat([g_tok, h], dim=-1))             # q_m  [B, M, D]
        z_soft, z_hard, alpha, idx, logits = self.vq(q)

        if self.hard_training:
            z_dec = z_soft + (z_hard - z_soft).detach()                 # straight-through argmax
        elif self.training:
            z_dec = z_soft
        else:
            z_dec = z_hard                                              # inference: discrete code
        dec_in = z_dec
        if self.decoder_cond == "graph":
            dec_in = dec_in + self.dec_cond(g).unsqueeze(1)
        if self.add_embodiment_token:
            dec_in = dec_in + self.embodiment_token(batch["embodiment_id"].long().view(-1)).unsqueeze(1)
        recon = self.decoder(dec_in, progress, causal=causal_attn)

        ent = -(alpha * (alpha + 1e-10).log()).sum(-1).mean()           # mean token entropy
        mean_alpha = alpha.reshape(-1, alpha.shape[-1]).mean(0)
        balance = (mean_alpha * (mean_alpha * alpha.shape[-1] + 1e-10).log()).sum()  # KL(mean || uniform)
        out = dict(recon=recon, z_e=q, h=h, z_q=z_dec, z=z_hard, idx=idx, alpha=alpha, g=g,
                   routing_entropy=ent, load_balance=balance,
                   alpha_max=alpha.max(-1).values.mean().detach(),
                   effect_pred=self.effect_head(z_dec))                 # [B, M, 3]
        if self.training and not self.hard_training:
            # soft/hard gap: what the decoder loses when stage 3 feeds it the argmax code
            with torch.no_grad():
                hard_in = z_hard + (dec_in - z_dec)
                out["recon_hard"] = self.decoder(hard_in, progress, causal=causal_attn)
        prim = batch.get("prim_id")
        if prim is not None:
            out["prim_tok"] = frames_to_tokens(prim.long().view(prim.shape[0], -1),
                                               h.shape[1]).clamp(0, self.num_primitives - 1)
        return out

    def compute_loss(self, x_hat, x, z_e=None, z=None) -> Dict[str, torch.Tensor]:
        """Reconstruction only; the routing regularisers are weighted in the trainer
        (lambda_entropy sharpens alpha, lambda_balance spreads codebook usage)."""
        rec = ((x_hat - x) ** 2).mean()
        return {"loss": rec, "recon_loss": rec, "vq_loss": torch.zeros_like(rec)}


# def object_displacement_targets(obj_rel: torch.Tensor, n_tokens: int, scale: float,
#                                 clip: float) -> torch.Tensor:
#     """obj_rel [B, H, 3] (metres, relative to the episode start) -> Delta o per token
#     [B, M, 3]: object position at the end of token m's span minus at its start, scaled
#     and clipped (one widowxai episode drops the object off the table)."""
#     B, H, _ = obj_rel.shape
#     ds = max(H // n_tokens, 1)
#     start_idx = torch.arange(n_tokens, device=obj_rel.device) * ds
#     end_idx = (start_idx + ds).clamp(max=H - 1)
#     delta = obj_rel[:, end_idx] - obj_rel[:, start_idx]
#     return (delta * scale).clamp(-clip, clip)


# def effect_loss(effect_pred: torch.Tensor, obj_rel: torch.Tensor, obj_valid=None,
#                 scale: float = 10.0, clip: float = 1.0):
#     """L_effect = MSE(Delta o, f_effect(z^skill)) over valid samples; also returns the
#     R^2 against the batch mean (0 = predicting the mean, 1 = perfect)."""
#     target = object_displacement_targets(obj_rel, effect_pred.shape[1], scale, clip)
#     if obj_valid is not None:
#         w = obj_valid.reshape(obj_valid.shape[0], -1)[:, 0].float().view(-1, 1, 1)
#     else:
#         w = torch.ones(target.shape[0], 1, 1, device=target.device)
#     se = ((effect_pred - target) ** 2) * w
#     denom = (w.sum() * target.shape[1] * target.shape[2]).clamp_min(1.0)
#     mse = se.sum() / denom
#     with torch.no_grad():
#         mean = (target * w).sum((0, 1), keepdim=True) / (w.sum() * target.shape[1]).clamp_min(1.0)
#         var = (((target - mean) ** 2) * w).sum() / denom
#         r2 = 1.0 - mse.detach() / var.clamp_min(1e-8)
#     return mse, r2


MODEL_REGISTRY = {"MOTIFStage1Graph": GraphRoutedMotifLearning}


def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    return MODEL_REGISTRY[name](**kwargs)
