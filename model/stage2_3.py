import numpy as np
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer
from transformers.feature_extraction_utils import BatchFeature
from einops import rearrange, repeat
from typing import Dict, List

from model.stage1 import ActionMotifLearning
from model.gripper_conventions import (
    LEGACY, MARKER, check_checkpoint_convention, convention_version,
)
from model.modules.perceiver_resampler_with_mask import PerceiverResampler
from model.modules.flow_matching_action_head import FlowmatchingActionHead

log = logging.getLogger(__name__)

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class MultimodalMotifPredictor(nn.Module):
    """Multimodal key learning model using text, vision and Perceiver fusion."""

    def __init__(
        self,
        vq_vae_ckpt: str = None,
        model_dim: int = 512,
        perceiver_depth: int = 6,
        perceiver_heads: int = 8,
        perceiver_dim_head: int = 64,
        perceiver_max_tokens: int = 1000,
        aux_trace_weight: float = 0.0,
        slot_id_embed: bool = False,
        trace_target: str = "self",
        condition_on_state: bool = False,
        state_dim: int = 10,
        max_embodiments: int = 10,
        teacher_eval: bool = False,
        state_conditioning_mode: str = "state_id",
    ) -> None:
        super().__init__()
        self.aux_trace_weight = aux_trace_weight
        self.slot_id_embed = slot_id_embed
        self.condition_on_state = condition_on_state
        self.teacher_eval = teacher_eval
        if state_conditioning_mode not in ("state_id", "state", "none"):
            raise ValueError(f'Unknown state_conditioning_mode: {state_conditioning_mode}')
        self.state_conditioning_mode = state_conditioning_mode
        # "self"   -- regress rel_state, the EE pose relative to the episode's
        #             first frame (proprioceptive; what the dyntrace arm used).
        # "object" -- regress the gripper->object offset, the geometry the motif
        #             pipeline never carried. Needs the object-augmented dataset.
        assert trace_target in ("self", "object"), trace_target
        self.trace_target = trace_target

        if vq_vae_ckpt is not None:
            vq_vae_cfg_path = Path(vq_vae_ckpt).parent / "config.json"
            with open(vq_vae_cfg_path, "r") as f:
                cfg = json.load(f)
            stage1_cls_name = cfg['models']['stage1'].get("class_name", "MOTIFStage1")
            cfg = {k: v for k, v in cfg['models']['stage1'].items() if k != "class_name"}
            # Dispatch on the class the stage-1 checkpoint was trained with: the
            # factorized variant emits one token stream per kinematic slot.
            if stage1_cls_name == "MOTIFStage1Dino":
                # Latent actions defined by observation change: N tokens for one
                # transition, not window_size/downsample per slot. The frozen
                # DINO encoder inside it is loaded but never trained here.
                from model.stage1_dino import DinoLatentActionLearning
                self.vq_vae = DinoLatentActionLearning(**cfg)
                num_slots = 1
            elif stage1_cls_name == "MOTIFStage1Graph":
                # Graph-guided routing: one token stream; keys are the routing
                # queries q_m from forward(batch) (forward_for_keys), and the
                # codebook is addressed by attention (RoutingCodebook).
                from model.stage1_graph import GraphRoutedMotifLearning
                self.vq_vae = GraphRoutedMotifLearning(**cfg)
                num_slots = 1
            elif stage1_cls_name in ("MOTIFStage1Factorized", "MOTIFStage1Grounded"):
                from model.stage1_factorized import FactorizedActionMotifLearning
                if stage1_cls_name == "MOTIFStage1Grounded":
                    # The grounded arm's visual FDM is a TRAINING-TIME constraint
                    # on the codes. Stage 2 needs only the per-slot encoders and
                    # the codebook, so it builds the plain factorized model and
                    # lets the unused visual weights fall out of the strict=False
                    # load -- no second DINO, and no frame pair required here.
                    cfg = {k: v for k, v in cfg.items() if k not in (
                        "vis_model_dim", "vis_depth", "vis_heads", "dino_pool",
                        "view_index", "dino_name", "dino_dim", "max_embodiments")}
                self.vq_vae = FactorizedActionMotifLearning(**cfg)
                num_slots = len(cfg.get("slots") or [1, 2, 3])
            else:
                cfg = {k: v for k, v in cfg.items() if k not in ("slots", "cross_slot")}
                self.vq_vae = ActionMotifLearning(**cfg)
                num_slots = 1
            checkpoint = torch.load(vq_vae_ckpt, map_location="cpu")
            self.vq_vae.load_state_dict(checkpoint["model"], strict=False)  # TODO: discriminator
            for param in self.vq_vae.parameters():
                param.requires_grad = False
            self.vq_vae.vq.code_restart = False
            log.info("Loaded VQVAE from %s", vq_vae_ckpt)

            teacher_convention = getattr(self.vq_vae, "gripper_convention", LEGACY)
            self.motif_gripper_version = convention_version(teacher_convention)
            if self.motif_gripper_version:
                # Stage 2 omits vq_vae weights when saving. Keep its semantic
                # marker here so Stage 3 cannot load a predictor for another
                # convention into this teacher, even with strict=False.
                self.register_buffer(MARKER, torch.tensor(self.motif_gripper_version,
                                                         dtype=torch.int64))

            # Perceiver
            code_dim = cfg["latent_dim"]
            if stage1_cls_name == "MOTIFStage1Dino":
                num_latents = int(cfg["num_latents"])
            else:
                ratio = cfg["window_size"] / cfg["downsample_factor"]

                if ratio <= 0 or not float(ratio).is_integer():
                    raise ValueError("window_size / downsample_factor must be a positive integer.")

                num_latents = int(ratio) * num_slots

            # The DINO stage 1 needs a frame PAIR, so its dataset yields
            # images [B, v, t=2, ...]. Stage 2 and stage 3 must see only the
            # CURRENT frame -- feeding the perceiver or the flow head the
            # +1 s frame would leak the future they are meant to predict from.
            self.obs_frames = 1 if stage1_cls_name == "MOTIFStage1Dino" else None
            self.perceiver = PerceiverResampler(
                dim=model_dim,
                depth=perceiver_depth,
                dim_head=perceiver_dim_head,
                heads=perceiver_heads,
                max_tokens=perceiver_max_tokens,
                num_latents=num_latents,
            )

            self.to_code_dim = nn.Linear(model_dim, code_dim)

            # Slot-identity embedding: the perceiver's key tokens map to slots
            # purely by position ([slot0 tokens | slot1 | slot2]); this adds an
            # explicit learned group embedding so which-slot-am-I-predicting is
            # architectural rather than something 48 individual query latents
            # must each rediscover. Zero-init: exact no-op at start.
            self.num_slots = num_slots
            if self.slot_id_embed:
                self.slot_emb = nn.Embedding(num_slots, model_dim)
                nn.init.zeros_(self.slot_emb.weight)

            # Auxiliary end-effector-trace prediction (behavior-aligned
            # representation, after arXiv:2607.27549): each predicted key token
            # must also regress the canonicalized rel_state channels of its slot,
            # window-averaged over that token's downsample window. Grounds the
            # vision+language prediction in explicit embodiment-invariant
            # geometry instead of only the frozen encoder's latents.
            if self.aux_trace_weight > 0:
                if stage1_cls_name == "MOTIFStage1Factorized":
                    from model.stage1_factorized import DEFAULT_SLOTS
                    self.trace_slots = [tuple(s) for s in
                                        (cfg.get("slots") or DEFAULT_SLOTS)]
                else:
                    self.trace_slots = [("all", 0, cfg["state_dim"])]
                out_dims = ([3] * len(self.trace_slots) if trace_target == "object"
                            else [e - s for _, s, e in self.trace_slots])
                self.trace_heads = nn.ModuleList([
                    nn.Linear(code_dim, o) for o in out_dims
                ])

        # Vision encoder (DINO)
        dino_dim = 768
        self.dino_transform = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_encoder.requires_grad_(False)
        self.vision_proj = nn.Linear(dino_dim, model_dim)

        # Text encoder (T5)
        t5_dim = 768
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(t5_dim, model_dim)

        if condition_on_state:
            # Only current rollout-available proprioception enters the predictor.
            self.state_proj = nn.Sequential(nn.Linear(state_dim, model_dim), nn.GELU(),
                                            nn.Linear(model_dim, model_dim))
            self.state_embodiment = nn.Embedding(max_embodiments, model_dim)
            # Allocate identical modules/RNG draws across ablation arms, but
            # freeze and bypass unused branches. Common initial weights match F2.
            if state_conditioning_mode != 'state_id':
                self.state_embodiment.requires_grad_(False)
            if state_conditioning_mode == 'none':
                self.state_proj.requires_grad_(False)

        # sentence -> unpadded T5 last_hidden_state, keyed by (device, sentence).
        # Plain dict so it stays out of state_dict().
        self._text_cache = {}

        # Frozen encoders stay in eval mode; see train().
        self.dino_encoder.eval()
        self.text_encoder.eval()
        if self.teacher_eval:
            self.vq_vae.eval()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        check_checkpoint_convention(
            state_dict, prefix, getattr(self, "motif_gripper_version", 0), error_msgs
        )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def train(self, mode: bool = True):
        """Keep the frozen encoders in eval mode: requires_grad_(False) stops
        gradients but not T5's dropout, which would make the text features
        stochastic and uncacheable."""
        super().train(mode)
        self.dino_encoder.eval()
        self.text_encoder.eval()
        if self.teacher_eval:
            self.vq_vae.eval()
        return self

    def _condition_observation(self, obs_seq, masks, state=None, state_mask=None, embodiment_id=None):
        mode = getattr(self, 'state_conditioning_mode', 'state_id')
        if not self.condition_on_state or mode == 'none':
            return obs_seq, masks
        if state is None or (mode == 'state_id' and embodiment_id is None):
            raise ValueError('State conditioning requires current state; state_id also requires embodiment_id')
        if state.ndim != 3 or state.shape[1] != 1:
            raise ValueError(f'Expected current state [B,1,D], got {tuple(state.shape)}')
        if state_mask is not None:
            state = state.masked_fill(~state_mask.bool(), 0)
        token = self.state_proj(state)
        if mode == 'state_id':
            token = token + self.state_embodiment(embodiment_id)[:, None, :]
        return torch.cat([obs_seq, token], dim=1), torch.cat([masks, masks.new_ones((len(masks), 1))], dim=1)

    # Encode text
    def _encode_text(self, sentences: List[str], device = 'cuda'):
        if self.text_encoder.training:
            # Dropout live -> output is stochastic; do not cache a noise draw.
            return self._encode_text_uncached(sentences, device)

        cache = self._text_cache
        dev_key = str(device)
        missing = [s for s in dict.fromkeys(sentences) if (dev_key, s) not in cache]
        if missing:
            encoding = self.tokenizer(missing, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                hidden = self.text_encoder(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                ).last_hidden_state
            lengths = encoding["attention_mask"].sum(dim=1).tolist()
            for sentence, row, length in zip(missing, hidden, lengths):
                # Store unpadded: a sentence's rows do not depend on the padding
                # batch it was encoded with.
                cache[(dev_key, sentence)] = row[:length].clone()

        rows = [cache[(dev_key, s)] for s in sentences]
        max_len = max(row.shape[0] for row in rows)
        lang_emb = rows[0].new_zeros(len(rows), max_len, rows[0].shape[-1])
        attention_mask = torch.zeros(
            len(rows), max_len, dtype=torch.long, device=lang_emb.device
        )
        for i, row in enumerate(rows):
            lang_emb[i, : row.shape[0]] = row
            attention_mask[i, : row.shape[0]] = 1

        # Padded rows stay zero; all consumers mask them out downstream.
        return lang_emb, attention_mask

    def _encode_text_uncached(self, sentences: List[str], device = 'cuda'):
        encoding = self.tokenizer(sentences, return_tensors="pt", padding=True).to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state, attention_mask

    # Encode images
    def _encode_image(self, images: torch.Tensor):
        # Single choke point for the observation, so the frame-pair dataset's
        # future frame cannot reach stage 2's perceiver or stage 3's flow head
        # through any path. Inert unless the stage-1 class is the DINO one.
        if getattr(self, "obs_frames", None):
            images = images[:, :, :self.obs_frames]
        b, v, t, c, h, w = images.shape
        images = rearrange(images, "b v t c h w -> (b t v) c h w")
        images = self.dino_transform(images)

        with torch.no_grad():
            feats = self.dino_encoder.forward_features(images)
            emb = feats["x_norm_patchtokens"]  # [(B*T*V), N, dino_dim]

        emb = rearrange(emb, "(b t v) n d -> b (t v n) d", b=b, t=t, v=v)
        mask = torch.ones(b, emb.shape[1], dtype=torch.int64, device=emb.device)
        return emb, mask

    # Frozen-encoder pass before any projection, split out so stage 3 can run
    # the encoders once and share the result (see MotifRoboticPolicy.forward).
    def encode_raw(self, batch: Dict[str, torch.Tensor]):
        lang_emb, lang_mask = self._encode_text(batch["language"], device=batch["images"].device)
        img_emb, img_mask = self._encode_image(batch["images"])
        return lang_emb, lang_mask, img_emb, img_mask

    def _encode_observation_from_raw(self, lang_emb, lang_mask, img_emb, img_mask):
        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)
        return obs_seq, masks

    # Encode observation (language + image)
    def _encode_observation(self, batch: Dict[str, torch.Tensor]):
        lang_emb, lang_mask = self._encode_text(batch["language"], device=batch["images"].device)
        img_emb, img_mask = self._encode_image(batch["images"])

        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)
        return self._condition_observation(obs_seq, masks, batch.get('state'),
                                           batch.get('state_mask'), batch.get('embodiment_id'))

    # Forward pass
    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_seq, masks = self._encode_observation(batch)
        key_token = self.perceiver(obs_seq, masks)

        # The factorized stage 1 has per-slot encoders (no single .encoder); its
        # forward concatenates slot streams along the token axis.
        if hasattr(self.vq_vae, "encoder") and not getattr(self.vq_vae, "forward_for_keys", False):
            z_e = self.vq_vae.encoder(batch["rel_state"], batch["progress"], causal=causal_attn)
        else:
            z_e = self.vq_vae(batch, causal_attn=causal_attn)["z_e"]
        key = self._project_key(key_token)

        z, idx_e, dis_e = self.vq_vae.vq.inference_forward(z_e)
        z_k, idx_k, dis_k = self.vq_vae.vq.inference_forward(key)

        loss = F.mse_loss(z_e, key)
        if self.aux_trace_weight > 0:
            tgt = (batch["ee_to_obj"] if self.trace_target == "object"
                   else batch["rel_state"])
            loss = loss + self.aux_trace_weight * self._trace_loss(key, tgt)
        return loss

    def _trace_loss(self, key: torch.Tensor, rel_state: torch.Tensor):
        """key: [B, S*H_ds, code_dim]; rel_state: [B, H, C], H = H_ds * ds."""
        B, N, D = key.shape
        S = len(self.trace_slots)
        k = key.view(B, S, N // S, D)
        H_ds = N // S
        ds = rel_state.shape[1] // H_ds
        target = rel_state.view(B, H_ds, ds, rel_state.shape[-1]).mean(dim=2)
        if self.trace_target == "object":
            # Every slot must carry where the object is relative to the gripper.
            losses = [F.mse_loss(self.trace_heads[i](k[:, i]), target)
                      for i in range(len(self.trace_slots))]
        else:
            losses = [
                F.mse_loss(self.trace_heads[i](k[:, i]), target[..., s:e])
                for i, (_, s, e) in enumerate(self.trace_slots)
            ]
        return sum(losses) / len(losses)

    def _project_key(self, key_token: torch.Tensor) -> torch.Tensor:
        """Optionally add the slot-identity embedding, then project to code dim.
        key_token: [B, S*H_ds, model_dim], slots concatenated along tokens."""
        if getattr(self, "slot_id_embed", False):
            B, N, D = key_token.shape
            H = N // self.num_slots
            slot_ids = torch.arange(self.num_slots, device=key_token.device)
            key_token = key_token + self.slot_emb(slot_ids).repeat_interleave(H, dim=0)
        return self.to_code_dim(key_token)

    # Inference: obtain key tokens
    @torch.no_grad()
    def get_key(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_seq, masks = self._encode_observation(batch)
        key_token = self.perceiver(obs_seq, masks)
        return self._project_key(key_token), obs_seq, masks

    @torch.no_grad()
    def get_key_from_raw(self, lang_emb, lang_mask, img_emb, img_mask,
                         state=None, state_mask=None, embodiment_id=None):
        obs_seq, masks = self._encode_observation_from_raw(
            lang_emb, lang_mask, img_emb, img_mask
        )
        obs_seq, masks = self._condition_observation(obs_seq, masks, state, state_mask, embodiment_id)
        key_token = self.perceiver(obs_seq, masks)
        return self._project_key(key_token), obs_seq, masks


class MotifRoboticPolicy(nn.Module):
    """Multimodal key learning model using text, vision and Perceiver fusion."""

    def __init__(
        self,
        mkl_cfg: dict = None,
        mkl_ckpt: str = None,
        add_latent_action: bool = False,
        flow_matching_cfg: dict = None,
        state_noise_std: float = 0.0,
        key_noise_std: float = 0.0,
        latent_mode: str = "code",
        geometry_cfg: dict = None,
    ) -> None:
        super().__init__()

        # What actually reaches the flow head as `latent_action`.
        #   code          -- z_q, the quantized motif. DEFAULT, every prior arm.
        #   key           -- the continuous key, no quantization.
        #   residual      -- key - z_q, only what quantization discards.
        #   code_residual -- concat([z_q, key - z_q]); the code plus what the
        #                    codebook threw away. Needs latent_action_dim = 2x.
        # Motivation (2026-09-02): probing the pipeline showed the ee-obj
        # auxiliary drove the gripper->object offset into the CONTINUOUS key to
        # 1.03 cm, but after vq.inference_forward the same probe reads 9.25 cm --
        # the stage-1 codebook has no codes that distinguish object positions, so
        # nearest-neighbour assignment discards ~90% of it. stage 3 only ever saw
        # the code, which is why Transfer stayed at 51.00 vs dyn's 53.00.
        #   code_trace    -- concat([z_q, W(trace)]), where trace is the FROZEN
        #                    stage-2 trace head's own output on the same key:
        #                    the gripper->object offset it was trained to
        #                    regress (ee-obj: 1.12 cm). Route 3 of the 09-02
        #                    list, and the only one of the three still standing.
        #                    Needs latent_action_dim = 2x, like code_residual.
        #   code_zerores  -- concat([z_q, 0]); EVAL-ONLY ablation of a
        #                    code_residual checkpoint. Same weights, same width,
        #                    residual half held at zero, so it answers whether
        #                    the residual channel is load-bearing at rollout
        #                    without retraining anything.
        assert latent_mode in ("code", "key", "residual", "code_residual",
                               "code_zerores", "code_trace"), latent_mode
        self.latent_mode = latent_mode

        # Training-time robustness noise (both inert at 0.0, and never applied
        # outside self.training, so get_action / eval are unchanged):
        #   state_noise_std -- gaussian noise on the flow head's state input, in
        #     normalized state units. The observed eval failure mode is a stall
        #     once the rollout drifts off the K=5 demo manifold; conditioning on
        #     perturbed states during training is the cheap DART-style
        #     counterpart of collecting corrective data.
        #   key_noise_std -- gaussian noise on the predicted key BEFORE vector
        #     quantization, scaled by the key tensor's own std. Simulates
        #     stage-2 key misprediction on off-manifold observations, so the
        #     action head does not overfit to exactly-right motif conditioning.
        self.state_noise_std = state_noise_std
        self.key_noise_std = key_noise_std

        if mkl_cfg is None:
            mkl_cfg = {}

        self.add_latent_action = add_latent_action
        if self.add_latent_action and mkl_ckpt is None:
            raise ValueError("Must provide mkl_ckpt if add_latent_action is True.")

        # MultimodalMotifPredictor (frozen)
        mkl_cfg = {k: v for k, v in mkl_cfg.items() if k != "class_name"}
        self.mkl = MultimodalMotifPredictor(**mkl_cfg)

        if mkl_ckpt is not None:
            checkpoint = torch.load(mkl_ckpt, map_location="cpu")
            missing, unexpected = self.mkl.load_state_dict(checkpoint["model"], strict=False)
            for k in missing:
                if not k.startswith(tuple(["vq_vae", "dino_encoder", "text_encoder"])):
                    raise ValueError(f"Missing key: {k}")
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
            log.info(f"Loaded mkl_ckpt from {mkl_ckpt}")

        # Freeze stage2 modules
        for param in self.mkl.parameters():  
            param.requires_grad = False

        # code_trace: a trainable projection of the frozen trace head's 3-dim
        # output up to the code width, so the second half of the latent is an
        # explicit gripper->object channel rather than the 256-dim grab-bag the
        # keyres residual turned out to be. The heads themselves stay frozen
        # inside self.mkl; only this projection learns.
        if self.latent_mode == "code_trace":
            if not hasattr(self.mkl, "trace_heads"):
                raise ValueError("latent_mode=code_trace needs a stage-2 ckpt "
                                 "trained with aux_trace_weight > 0")
            if self.mkl.trace_target != "object":
                raise ValueError("latent_mode=code_trace expects "
                                 "trace_target=object (ee-obj's stage 2)")
            code_dim = self.mkl.to_code_dim.out_features
            self.trace_proj = nn.Linear(3, code_dim)

        if flow_matching_cfg.get("motif_position_mode", "none") == "slot_time":
            flow_matching_cfg = dict(flow_matching_cfg)
            slots = self.mkl.num_slots
            tokens = self.mkl.perceiver.num_queries
            if tokens % slots:
                raise ValueError(f"Cannot divide {tokens} motif tokens into {slots} slots")
            for name, actual in (("motif_num_slots", slots),
                                 ("motif_tokens_per_slot", tokens // slots)):
                configured = flow_matching_cfg.get(name)
                if configured is not None and configured != actual:
                    raise ValueError(f"{name}={configured} disagrees with loaded teacher: {actual}")
                flow_matching_cfg[name] = actual
        self.flow_matching_head = FlowmatchingActionHead(flow_matching_cfg)
        self.flow_matching_head.set_trainable_parameters(
            tune_projector=True,
            tune_diffusion_model=True,
        )

        dino_dim = 768
        self.vision_proj = nn.Linear(dino_dim, flow_matching_cfg['hidden_size'])
        t5_dim = 768
        self.lang_proj = nn.Linear(t5_dim, flow_matching_cfg['hidden_size'])

        # Metric 3D positional encodings on the wrist-camera tokens (model/geometry_tokens.py);
        # needs batch["tcp_pose"] from a *_geo data config. Zero-initialised: inert at start.
        self.geometry = None
        if geometry_cfg and geometry_cfg.get("enabled", True):
            from model.geometry_tokens import GeometryTokens
            gcfg = {k: v for k, v in dict(geometry_cfg).items() if k != "enabled"}
            if "robots" in gcfg:
                gcfg["robots"] = dict(gcfg["robots"])
            self.geometry = GeometryTokens(hidden_size=flow_matching_cfg['hidden_size'], **gcfg)

    def get_observation_embedding(self, batch: Dict[str, torch.Tensor], encoded=None):
        if encoded is None:
            encoded = self.mkl.encode_raw(batch)
        lang_emb, lang_mask, img_emb, img_mask = encoded

        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)
        if self.geometry is not None:
            img_emb = self.geometry(img_emb, batch)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)

        return obs_seq, masks

    def _latent_from_key(self, key: torch.Tensor) -> torch.Tensor:
        """Build the flow head's latent_action from the predicted key."""
        z_q, _, _ = self.mkl.vq_vae.vq.inference_forward(key)
        if self.latent_mode == "code":
            return z_q
        if self.latent_mode == "key":
            return key
        if self.latent_mode == "residual":
            return key - z_q
        if self.latent_mode == "code_zerores":
            return torch.cat([z_q, torch.zeros_like(z_q)], dim=-1)
        if self.latent_mode == "code_trace":
            # Each slot's own head, applied to that slot's key tokens. In object
            # mode every slot regresses the same shared offset, so the three
            # predictions are three views of one quantity.
            B, N, D = key.shape
            S = len(self.mkl.trace_slots)
            k = key.view(B, S, N // S, D)
            tr = torch.cat([self.mkl.trace_heads[i](k[:, i]) for i in range(S)],
                           dim=1)                                   # [B, N, 3]
            return torch.cat([z_q, self.trace_proj(tr)], dim=-1)
        return torch.cat([z_q, key - z_q], dim=-1)   # code_residual

    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        # One shared frozen-encoder pass for the policy branch and mkl's key branch.
        encoded = self.mkl.encode_raw(batch)
        obs_emb, masks = self.get_observation_embedding(batch, encoded=encoded)
        if self.add_latent_action:
            key, _, _ = self.mkl.get_key_from_raw(*encoded, state=batch.get('state'),
                                                state_mask=batch.get('state_mask'),
                                                embodiment_id=batch.get('embodiment_id'))
            if self.training and self.key_noise_std > 0:
                key = key + self.key_noise_std * key.std() * torch.randn_like(key)
            latent_action = self._latent_from_key(key)
        else:
            latent_action = None
        state = batch['state']
        if self.training and self.state_noise_std > 0:
            noise = self.state_noise_std * torch.randn_like(state)
            if 'state_mask' in batch:
                noise = noise * batch['state_mask']
            state = state + noise
        action_head_input1 = BatchFeature(
            data={
                "backbone_features": obs_emb,
                "backbone_attention_mask": ~masks.bool()    # mask is True for padding tokens
                }
        )
        head_inputs = {
            "embodiment_id": batch['embodiment_id'],
            "state": state,
            "action": batch['action'],
            "action_mask": batch['action_mask'],
            "latent_action": latent_action,
        }
        if "ee_to_obj" in batch:
            # Gripper->object offset [B, T, 3] (object-augmented datasets only);
            # consumed by the head's auxiliary loss when aux_obj_weight > 0.
            head_inputs["ee_to_obj"] = batch["ee_to_obj"]
        action_head_input2 = BatchFeature(data=head_inputs)
        action_head_outputs = self.flow_matching_head(action_head_input1, action_head_input2)
        # Auxiliary terms for the trainer's log line (empty unless the head emits them).
        self.last_aux = {k: v for k, v in action_head_outputs.items() if k.startswith("aux_")}
        return action_head_outputs.loss

    @torch.no_grad()
    def get_action(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        encoded = self.mkl.encode_raw(batch)
        obs_emb, masks = self.get_observation_embedding(batch, encoded=encoded)
        if self.add_latent_action:
            key, _, _ = self.mkl.get_key_from_raw(*encoded, state=batch.get('state'),
                                                state_mask=batch.get('state_mask'),
                                                embodiment_id=batch.get('embodiment_id'))
            latent_action = self._latent_from_key(key)
        else:
            latent_action = None
        action_head_input1 = BatchFeature(
            data={
                "backbone_features": obs_emb, 
                "backbone_attention_mask": ~masks.bool()    # mask is True for padding tokens
                }
        )
        action_head_input2 = BatchFeature(
            data={
                "embodiment_id": batch['embodiment_id'], 
                "state": batch['state'],
                "latent_action": latent_action,
                }
        ) 
        action_head_outputs = self.flow_matching_head.get_action(action_head_input1, action_head_input2)
        return action_head_outputs


MODEL_REGISTRY = {
    "MOTIFStage2": MultimodalMotifPredictor,
    "MOTIFStage3": MotifRoboticPolicy,
}

def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name == "MOTIFStage3ELoRA" and name not in MODEL_REGISTRY:
        from model.stage3_elora import MotifRoboticPolicyELoRA   # lazy: it imports this module
        MODEL_REGISTRY[name] = MotifRoboticPolicyELoRA
    if name == "MOTIFStage3VLA" and name not in MODEL_REGISTRY:
        from model.vla_elora import VLAELoRAPolicy
        MODEL_REGISTRY[name] = VLAELoRAPolicy
    if name == "MOTIFStage3PI05" and name not in MODEL_REGISTRY:
        from model.pi05_elora import PI05ELoRAPolicy   # frozen pi0.5 + E-LoRA + prompts
        MODEL_REGISTRY[name] = PI05ELoRAPolicy
    if name == "MOTIFStage3GR00T" and name not in MODEL_REGISTRY:
        from model.gr00t_elora import GR00TELoRAPolicy   # frozen GR00T N1.5 + E-LoRA + prompts
        MODEL_REGISTRY[name] = GR00TELoRAPolicy
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    return MODEL_REGISTRY[name](**kwargs)
