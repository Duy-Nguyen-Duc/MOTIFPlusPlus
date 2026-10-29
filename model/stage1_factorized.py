"""Factorized action-motif learning (Stage 1 variant).

Motivation, measured on the reproduced K=5 baseline: the monolithic codebook is
already fully shared across embodiments (128/128 codes appear on all three robots,
and 100% of a few-shot pair's codes exist in source data), but the *sequences* are
not -- 66% of bigrams and 81% of trigrams occur on only one robot, and a target
pair sees only ~30% of its bigrams in its 5 demonstrations. MOTIF learns a shared
alphabet with a private grammar.

Hypothesis: a monolithic code binds translation, orientation and gripper into one
symbol, so a transition is a *joint* transition and the space of joint transitions
is combinatorially large. Per-slot codes make transitions per-slot, so e.g. a
gripper-close transition should be shared across robots even when the arm
transition is not.

Design decisions that differ from the original proposal, and why:

* **Cross-slot attention lives in the decoder, not the encoder.** If slots attend
  to each other before quantization, each code already carries information about
  the other slots, the factorization is nominal, and the routing losses fight the
  encoder. Encoding is therefore per-slot and independent; coordination is modelled
  downstream. This is what "non-duplicated motif content with coordinated decoding"
  actually implies.
* **No `z_coord` channel.** With three tightly coupled slots on a single
  manipulator it has nothing to represent that decoder cross-slot mixing does not
  already carry. It returns with a bimanual embodiment.
* **Shared codebook across slots and embodiments**, so "the same normalized motion
  pattern reuses the same code" is testable across both axes.

The embodiment discriminator is *conditional*. Under the paper's interleaved mask
each robot has a different task mix (four source tasks at ~48 demos, two targets at
K), so the marginal motif distribution differs across embodiments because the task
mix differs, not because the motion style differs. A marginal discriminator can
exploit that, and the gradient reversal then strips task information out of the
motifs. Conditioning on (task, progress bin) removes the confound.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gripper_conventions import (
    LEGACY, MARKER, check_checkpoint_convention, convention_version,
    reflect_relative_gripper,
)
from model.modules.blocks import (PerSlotQuantizer, ResidualVectorQuantizer,
                                  VectorQuantizer)
from model.stage1 import (
    ActionMotifLearning,
    LocalTransformerEncoder,
    LocalTransformerDecoder,
    GradientReversal,
)
from model.primitive_tying import DEFAULT_SLOT_MASK, frames_to_tokens

# (name, start, end) into the 8-D canonicalized rel_state:
# xyz | wxyz quaternion | gripper
DEFAULT_SLOTS: List[Tuple[str, int, int]] = [
    ("translation", 0, 3),
    ("orientation", 3, 7),
    ("gripper", 7, 8),
]


class FactorizedActionMotifLearning(ActionMotifLearning):
    """Per-slot encoders + shared codebook + cross-slot decoding.

    Interface matches ActionMotifLearning so the Stage-1 training loop is unchanged:
    ``z_e`` / ``z_q`` / ``idx`` are returned with the slot axis folded into the token
    axis ([B, S*H_ds, ...]), which leaves InfoNCE (mean-pools over tokens), the
    adversarial head (flattens to [N, D]) and the codebook statistics all valid.
    """

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
        slots: List[Tuple[str, int, int]] = None,
        cross_slot: bool = True,
        add_embodiment_token: bool = False,
        latent_dynamics: bool = False,
        gripper_adapter: bool = False,
        rvq_levels: int = 1,
        per_slot_codebook: bool = False,
        gripper_convention: str = LEGACY,
        slot_gating: bool = False,
        slot_mask: List[List[int]] = None,
        num_primitives: int = 7,
        idle_margin: float = 0.0,
    ):
        # Build the parent with the same hyperparameters, then discard its single
        # encoder/decoder: we reuse its VQ, loss and codebook-statistics helpers.
        super().__init__(
            state_dim=state_dim, model_dim=model_dim, latent_dim=latent_dim,
            codebook_size=codebook_size, window_size=window_size,
            depth_enc=depth_enc, depth_dec=depth_dec, num_heads=num_heads,
            dropout=dropout, local_w=local_w, use_sinusoidal_pe=use_sinusoidal_pe,
            max_len=max_len, vq_beta=vq_beta, downsample_factor=downsample_factor,
            depth_enc_low=depth_enc_low, depth_dec_low=depth_dec_low,
            add_embodiment_token=add_embodiment_token,
        )

        self.gripper_convention = gripper_convention
        version = convention_version(gripper_convention)
        if version:
            if state_dim != 8 or [tuple(s) for s in (slots or DEFAULT_SLOTS)] != DEFAULT_SLOTS:
                raise ValueError("Gripper polarity v1 requires the default eight-channel slots")
            # No random initialization; legacy parameter/RNG streams stay intact.
            self.register_buffer(MARKER, torch.tensor(version, dtype=torch.int64))

        # Opt-in residual quantization. rvq_levels=1 leaves `self.vq` exactly as
        # the parent built it, so every existing arm is bit-identical.
        #
        # Measured motivation (EXPERIMENTS.md, "The quantizer is the
        # constraint"): one 128-code level simultaneously provides transfer
        # (embodiment per token 0.654 -> 0.261) and destroys scene precision
        # (gripper->goal 2.51 -> 3.85 cm). Level 1 keeps the first job; the
        # deeper levels carry what it rounds away.
        # Independent codebook per slot instead of one shared across slots.
        # See PerSlotQuantizer for the measured caveat: on grounded's final
        # checkpoint 87% of used codes are already slot-exclusive, so at the same
        # codebook_size this is mainly a ~2.5x per-slot capacity increase.
        # NOTE: sharing across EMBODIMENTS is untouched -- that is the transfer
        # claim, and it is orthogonal to sharing across slots.
        self.per_slot_codebook = per_slot_codebook
        assert not (per_slot_codebook and rvq_levels > 1), \
            "per_slot_codebook and rvq_levels>1 not combined -- one change at a time"
        if per_slot_codebook:
            self.vq = PerSlotQuantizer(
                num_latents=codebook_size, latent_dim=latent_dim,
                num_slots=len(slots or DEFAULT_SLOTS), code_restart=True,
            )

        self.rvq_levels = rvq_levels
        if rvq_levels > 1:
            self.vq = ResidualVectorQuantizer(
                num_latents=codebook_size,
                latent_dim=latent_dim,
                num_levels=rvq_levels,
                code_restart=True,
            )
        del self.encoder
        del self.decoder

        self.slots = [tuple(s) for s in (slots or DEFAULT_SLOTS)]
        self.cross_slot = cross_slot
        S = len(self.slots)
        self.num_slots = S

        common = dict(
            model_dim=model_dim, latent_dim=latent_dim, num_heads=num_heads,
            dropout=dropout, local_w=local_w, use_sinusoidal_pe=use_sinusoidal_pe,
            max_len=max_len, downsample_factor=self.downsample_factor,
        )
        self.encoders = nn.ModuleList([
            LocalTransformerEncoder(state_dim=(e - s), depth_high=depth_enc,
                                    depth_low=depth_enc_low, **common)
            for _, s, e in self.slots
        ])
        self.decoders = nn.ModuleList([
            LocalTransformerDecoder(state_dim=(e - s), depth_high=depth_dec,
                                    depth_low=depth_dec_low, **common)
            for _, s, e in self.slots
        ])
        # Cross-slot mixing: each slot decoder sees every slot's code, projected
        # back to latent_dim. Coordination without entangling the codes themselves.
        self.mix = nn.ModuleList([nn.Linear(S * latent_dim, latent_dim) for _ in self.slots])

        # Latent-dynamics / transition model (mechanism after Wang et al.
        # arXiv:2406.01968): a small shared MLP predicts the next-step encoder
        # latent from the current quantized code (+ a slot embedding, since the
        # three slots have different dynamics). Trained with MSE to the detached
        # next latent; gradients flow into the encoders through the
        # straight-through z_q, pressuring transitions -- the measured transfer
        # bottleneck ("shared alphabet, private grammar") -- to be predictable
        # by one embodiment-invariant model. The case-study failure mode is
        # literally a missing p(next motif | motif).
        self.latent_dynamics = latent_dynamics
        if latent_dynamics:
            self.dyn_slot_emb = nn.Embedding(S, latent_dim)
            self.dyn_mlp = nn.Sequential(
                nn.Linear(latent_dim, 2 * latent_dim), nn.GELU(),
                nn.Linear(2 * latent_dim, latent_dim),
            )

        # Per-embodiment gripper adapter (SPACE arXiv:2606.24049 submodule):
        # zero-init FiLM on the gripper slot's decoder input, indexed by the
        # global embodiment id. Lets the shared gripper codes stay universal
        # while the decoder grounds them per robot -- the gripper slot is the
        # only one whose cross-embodiment support regressed (Robotiq vs
        # parallel-finger kinematics). Decoder-side only; encoders and codes
        # (all that stages 2/3 consume) are untouched at inference.
        self.gripper_adapter = gripper_adapter
        if gripper_adapter:
            self.gripper_slot_idx = next(
                i for i, (n, _, _) in enumerate(self.slots) if n == "gripper")
            self.grip_film = nn.Embedding(16, 2 * latent_dim)
            nn.init.zeros_(self.grip_film.weight)  # identity at start

        # Primitive -> slot tying (model/primitive_tying.py). With per-frame
        # primitive labels in the batch (batch["prim_id"], *_rel_prim data
        # config), every slot the current primitive does not use is forced onto
        # that slot's reserved idle code: codebook row s for slot s. The usual
        # codebook/commitment terms then pull both the idle vector and the
        # encoder output together, so at inference -- no labels, plain
        # nearest-neighbour -- an inactive slot lands on idle by itself. Slot
        # tokens the primitive does use quantize freely over the whole codebook
        # (idle included, so a genuinely still slot may still say so). Small
        # residual motion in an idle slot is explained by the cross-slot decoder
        # mix from the active slots' codes, not by the idle slot's own code.
        # Without labels (stage 2/3 teacher calls) the forward is unchanged.
        self.slot_gating = slot_gating
        self.gate_with_labels = True
        self.num_primitives = num_primitives
        # > 0: hinge on inactive tokens, d(z_e, idle) + margin <= d(z_e, nearest other code)
        self.idle_margin = float(idle_margin)
        if slot_gating:
            if per_slot_codebook or rvq_levels > 1:
                raise ValueError("slot_gating needs the plain shared VectorQuantizer")
            table = torch.tensor(slot_mask if slot_mask is not None else DEFAULT_SLOT_MASK,
                                 dtype=torch.float32)
            if tuple(table.shape) != (num_primitives, S):
                raise ValueError(f"slot_mask must be [{num_primitives}, {S}], got {tuple(table.shape)}")
            # Not persisted: the config carries it, and old checkpoints stay loadable.
            self.register_buffer("slot_mask_table", table, persistent=False)
            self.vq.reserved = S

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        check_checkpoint_convention(
            state_dict, prefix, convention_version(self.gripper_convention), error_msgs
        )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _slot_gate(self, batch: Dict[str, torch.Tensor], n_tokens: int):
        """Per-token (primitive id, slot gate) from per-frame labels, or (None, None)."""
        if not self.slot_gating or batch.get("prim_id") is None:
            return None, None
        prim = batch["prim_id"]
        prim_tok = frames_to_tokens(prim.long().view(prim.shape[0], -1), n_tokens)   # [B, M]
        prim_tok = prim_tok.clamp(0, self.num_primitives - 1)
        gate = self.slot_mask_table[prim_tok] if self.gate_with_labels else None      # [B, M, S]
        return gate, prim_tok

    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        progress, state_seq = batch["progress"], batch["rel_state"]
        if self.gripper_convention != LEGACY:
            if batch.get("rel_state_gripper_convention", LEGACY) != LEGACY:
                raise ValueError("Teacher expects legacy rel_state; do not canonicalize twice")
            state_seq = reflect_relative_gripper(state_seq, batch.get("embodiment_id"))

        z_e_list = [self.encoders[i](state_seq[..., s:e], progress, causal=causal_attn)
                    for i, (_, s, e) in enumerate(self.slots)]
        gate, prim_tok = self._slot_gate(batch, z_e_list[0].shape[1])

        z_q_list, z_list, idx_list, idx_nn_list, margin_terms = [], [], [], [], []
        for i in range(self.num_slots):
            z_e_i = z_e_list[i]
            z_q_i, z_i, _, idx_i, _ = (self.vq(z_e_i, slot=i) if self.per_slot_codebook
                                       else self.vq(z_e_i))
            idx_nn_list.append(idx_i)                      # free nearest-neighbour choice
            if gate is not None:
                active = gate[:, :, i] > 0.5                  # [B, M]
                idx_i = torch.where(active, idx_i, torch.full_like(idx_i, i))
                z_i = self.vq.codebook(idx_i)                 # re-lookup keeps codebook grads
                # Inactive tokens feed the idle row itself to the decoder (gradient
                # into the row), with no straight-through path to the encoder: with
                # ST the recon gradient kept pushing z_e while the decoder input never
                # moved, so inactive z_e drifted away from idle (vq_loss rose and the
                # free nearest neighbour picked other codes). Only the commitment term
                # in compute_loss now ties inactive z_e to the idle row.
                z_q_i = torch.where(active.unsqueeze(-1), z_q_i, z_i)
                if self.idle_margin > 0 and self.training and (~active).any():
                    # Free nearest neighbour must land on idle for inactive tokens:
                    # the commitment term alone leaves them within ~1 unit of idle
                    # while active codes for tiny motions sit just as close.
                    # Codebook detached: with codebook gradients the hinge scattered
                    # the active codes (vq loss x10, recon x2); only the encoder moves.
                    d = torch.cdist(z_e_i[~active], self.vq.codebook.weight.detach())  # [n, K]
                    d_other = d[:, self.num_slots:].min(1).values
                    margin_terms.append(F.relu(d[:, i] - d_other + self.idle_margin))
                with torch.no_grad():                         # usage credit for forced idle
                    self.vq.usage[i] += (~active).sum()
            z_q_list.append(z_q_i); z_list.append(z_i); idx_list.append(idx_i)

        if self.cross_slot:
            ctx = torch.cat(z_q_list, dim=-1)          # [B, H_ds, S*latent]
            dec_in = [self.mix[i](ctx) for i in range(self.num_slots)]
        else:
            dec_in = z_q_list

        if self.add_embodiment_token:
            tok = self.embodiment_token(batch["embodiment_id"].long().view(-1)).unsqueeze(1)
            dec_in = [d + tok for d in dec_in]

        if self.gripper_adapter:
            gi = self.gripper_slot_idx
            film = self.grip_film(batch["embodiment_id"].long().view(-1))  # [B, 2D]
            scale, shift = film.chunk(2, dim=-1)
            dec_in = list(dec_in)
            dec_in[gi] = dec_in[gi] * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        recon = torch.zeros_like(state_seq)
        for i, (_, s, e) in enumerate(self.slots):
            recon[..., s:e] = self.decoders[i](dec_in[i], progress, causal=causal_attn)

        if self.gripper_convention != LEGACY:
            # Keep the public reconstruction API in the input convention. This
            # reflection makes its MSE exactly the canonical-space MSE, while
            # the existing training loop and reconstruction probes stay valid.
            recon = reflect_relative_gripper(recon, batch["embodiment_id"])

        out = dict(
            recon=recon,
            z_e=torch.cat(z_e_list, dim=1),            # [B, S*H_ds, latent]
            z_q=torch.cat(z_q_list, dim=1),
            z=torch.cat(z_list, dim=1),
            idx=torch.cat(idx_list, dim=1),            # [B, S*H_ds]
            idx_per_slot=torch.stack(idx_list, dim=1), # [B, S, H_ds] for analysis
        )
        if self.slot_gating:
            out["idx_nn_per_slot"] = torch.stack(idx_nn_list, dim=1)   # [B, S, H_ds] before gating
        if prim_tok is not None:
            out["prim_tok"] = prim_tok                                 # [B, H_ds]
            out["slot_gate"] = self.slot_mask_table[prim_tok]           # [B, H_ds, S]
        if margin_terms:
            out["idle_margin_loss"] = torch.cat(margin_terms).mean()

        if self.latent_dynamics:
            dyn = 0.0
            for i in range(self.num_slots):
                sid = self.dyn_slot_emb.weight[i]
                pred = self.dyn_mlp(z_q_list[i][:, :-1] + sid)     # [B, H_ds-1, D]
                dyn = dyn + F.mse_loss(pred, z_e_list[i][:, 1:].detach())
            out["dyn_loss"] = dyn / self.num_slots
        return out


class ConditionalEmbodimentDiscriminator(nn.Module):
    """Embodiment discriminator conditioned on (task, progress bin).

    Predicts the embodiment from a motif token *given* what task is being performed
    and how far through it the window sits, so gradient reversal removes only the
    embodiment-specific component rather than the task signal that correlates with
    embodiment through the interleaved data allocation.
    """

    def __init__(self, input_dim: int, num_robots: int, num_tasks: int = 6,
                 num_progress_bins: int = 4, cond_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.num_progress_bins = num_progress_bins
        self.task_emb = nn.Embedding(num_tasks + 1, cond_dim)   # +1 = unknown task
        self.prog_emb = nn.Embedding(num_progress_bins, cond_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2 * cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_robots),
        )

    def forward(self, z, task_ids, progress, alpha: float = 1.0):
        """z: [N, D]; task_ids: [N] long; progress: [N] float in [0, 1]."""
        z = GradientReversal.apply(z, alpha)
        bins = torch.clamp((progress * self.num_progress_bins).long(),
                           0, self.num_progress_bins - 1)
        h = torch.cat([z, self.task_emb(task_ids), self.prog_emb(bins)], dim=-1)
        return self.net(h)


MODEL_REGISTRY = {"MOTIFStage1Factorized": FactorizedActionMotifLearning}


def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    return MODEL_REGISTRY[name](**kwargs)
