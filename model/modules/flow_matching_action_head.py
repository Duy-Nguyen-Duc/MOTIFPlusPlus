# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from .action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer
from model.modules.blocks import CategorySpecificLinear, CategorySpecificMLP


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)
        
    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig:
    """General-purpose config class for FlowmatchingActionHead."""

    add_pos_embed: bool = True
    diffusion_model_cfg: Dict[str, Any] = field(
        default_factory=dict,  
        metadata={"help": "Diffusion model configuration."},
    )
    input_embedding_dim: int = 1024
    backbone_embedding_dim: int = 1024

    add_latent_action: bool = False
    latent_action_dim: int = 256
    motif_position_mode: str = "none"
    motif_num_slots: Optional[int] = None
    motif_tokens_per_slot: Optional[int] = None

    hidden_size: int = 1024
    max_seq_len: int = 1024
    max_action_dim: Optional[int] = None
    max_state_dim: Optional[int] = None
    action_horizon: Optional[int] = None
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: Optional[int] = None
    max_num_embodiments: int = 10
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    load_pretrained_det_decode_layer_path: Optional[str] = None
    detection_coeff: float = 1.0

    freeze_decode_layer: bool = False
    expand_batch: Optional[int] = None
    use_vlln: bool = True

    # Auxiliary gripper->object regression from the DiT's own hidden states
    # (fix 2 of eval_results/analysis/graph_rollout_diagnosis.md: the policy
    # mislocalises the object and stops short of it). When aux_obj_weight > 0 a
    # small head on the mean DiT output token regresses the current-frame
    # offset action_input.ee_to_obj[:, 0] (metres, scaled by aux_obj_scale so
    # 1 cm = 0.1 at the default 10). Weight 0 (default) allocates nothing, so
    # every earlier checkpoint loads and trains unchanged. get_action ignores it.
    aux_obj_weight: float = 0.0
    aux_obj_scale: float = 10.0
    aux_obj_clip: float = 0.5   # metres; the widowxai side-car has a 1.5 m glitch episode

    vl_self_attention_cfg: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Config for VL self-attention transformer."},
    )

    @classmethod
    def from_dict(
        cls,
        cfg: Mapping[str, Any],
    ) -> "FlowmatchingActionHeadConfig":
        """Create a config instance from a dictionary.Useful when loading JSON/YAML configs."""
        return cls(**cfg)

    @classmethod
    def from_any(
        cls,
        cfg: Union["FlowmatchingActionHeadConfig", Mapping[str, Any], None],
    ) -> "FlowmatchingActionHeadConfig":
        """
        Flexible constructor:
        - If given a config instance, return it directly.
        - If given a dict, convert it to a config.
        - If None, return the default config.
        """
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, Mapping):
            return cls.from_dict(cfg)
        raise TypeError(f"Unsupported config type: {type(cfg)}")


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Union[FlowmatchingActionHeadConfig, Mapping[str, Any], None] = None,
    ):
        super().__init__()
        config = FlowmatchingActionHeadConfig.from_any(config)

        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.max_action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=config.max_action_dim,
        )

        if config.add_latent_action:
            self.latent_action_encoder = nn.Sequential(
                nn.Linear(config.latent_action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, self.hidden_size),
            )
        self.add_latent_action = config.add_latent_action

        self.aux_obj_weight = config.aux_obj_weight
        self.aux_obj_scale = config.aux_obj_scale
        if self.aux_obj_weight > 0:
            self.obj_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 3),
            )

        self.use_vlln = config.use_vlln
        if self.use_vlln:
            self.vl_self_attention = SelfAttentionTransformer(**config.vl_self_attention_cfg)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.apply(self._init_weights)

        # Opt-in identity after projection, where quantization cannot discard it.
        # Allocate zero parameters after existing initialization: old weights and
        # the RNG stream match the control exactly at initialization.
        if config.motif_position_mode not in ("none", "slot_time"):
            raise ValueError(f"Unknown motif_position_mode: {config.motif_position_mode}")
        if config.motif_position_mode == "slot_time":
            if not config.add_latent_action:
                raise ValueError("slot_time requires add_latent_action=True")
            if not config.motif_num_slots or not config.motif_tokens_per_slot:
                raise ValueError("slot_time requires positive motif slot/time counts")
            if config.motif_num_slots < 1 or config.motif_tokens_per_slot < 1:
                raise ValueError("motif slot/time counts must be positive")
            self.motif_slot_embedding = nn.Parameter(
                torch.zeros(config.motif_num_slots, self.hidden_size))
            self.motif_time_embedding = nn.Parameter(
                torch.zeros(config.motif_tokens_per_slot, self.hidden_size))

        print(
            "Total number of Action Head parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, CategorySpecificLinear):
            nn.init.normal_(module.W, mean=0.0, std=0.02)
            nn.init.zeros_(module.b)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()        

    def encode_latent_action(self, latent_action: torch.Tensor) -> torch.Tensor:
        latent = self.latent_action_encoder(latent_action)
        if self.config.motif_position_mode == "slot_time":
            slots = self.config.motif_num_slots
            times = self.config.motif_tokens_per_slot
            if latent.ndim != 3 or latent.shape[1] != slots * times:
                raise ValueError(f"Expected {slots} x {times} slot-major motif tokens; "
                                 f"got {tuple(latent.shape)}")
            identity = (self.motif_slot_embedding[:, None, :]
                        + self.motif_time_embedding[None, :, :]).reshape(slots * times, -1)
            latent = latent + identity.to(dtype=latent.dtype).unsqueeze(0)
        return latent

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.config.motif_position_mode == "slot_time":
            for name in ("motif_slot_embedding", "motif_time_embedding"):
                if prefix + name not in state_dict:
                    error_msgs.append(f"Required motif identity weight missing: {prefix}{name}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
                
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.add_latent_action:
                self.latent_action_encoder.requires_grad_(False)
                if self.config.motif_position_mode == "slot_time":
                    self.motif_slot_embedding.requires_grad_(False)
                    self.motif_time_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        # if self.freeze_decode_layer:
        #     self.decode_layer.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_mask = backbone_output["backbone_attention_mask"]
        backbone_features = self.vl_self_attention(backbone_features, backbone_mask)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, 
                backbone_output: BatchFeature, 
                action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        if self.use_vlln:
            backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        if self.add_latent_action:
            latent_action_emb = self.encode_latent_action(action_input.latent_action)
            sa_embs = torch.cat((state_features, latent_action_emb, action_features), dim=1)
        else:
            sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_embs = vl_embeds

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=None if self.use_vlln else backbone_output.backbone_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        ee_to_obj = action_input.get("ee_to_obj", None) if self.aux_obj_weight > 0 else None
        if ee_to_obj is not None:
            # Current-frame offset: the window starts at the observation frame.
            target = ee_to_obj[:, 0] if ee_to_obj.dim() == 3 else ee_to_obj
            target = target.clamp(-self.config.aux_obj_clip, self.config.aux_obj_clip)
            pred_obj = self.obj_head(model_output.mean(dim=1))
            aux = F.mse_loss(pred_obj, (target * self.aux_obj_scale).to(pred_obj.dtype))
            loss = loss + self.aux_obj_weight * aux
            output_dict["loss"] = loss
            output_dict["aux_obj_loss"] = aux.detach()
            output_dict["aux_obj_rmse_cm"] = (aux.detach().sqrt() / self.aux_obj_scale * 100.0)
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        if self.use_vlln:
            backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.max_action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # These conditioning features do not depend on the denoising timestep.
        latent_action_emb = (
            self.encode_latent_action(action_input.latent_action)
            if self.add_latent_action else None
        )
        if self.config.add_pos_embed:
            pos_ids = torch.arange(self.action_horizon, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)

        obj_steps = []
        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # Join vision, language, state and action embedding along sequence dimension.
            if self.add_latent_action:
                sa_embs = torch.cat((state_features, latent_action_emb, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=None if self.use_vlln else backbone_output.backbone_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            if getattr(self, "obj_head", None) is not None:
                # gripper->object offset read off the DiT tokens at this denoising step (metres)
                obj_steps.append(self.obj_head(model_output.mean(dim=1)) / self.aux_obj_scale)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        if obj_steps:
            # exposed for test-time steering (eval/rollout_video.py --steer): [B, num_steps, 3] and its mean
            self.last_obj_pred_steps = torch.stack(obj_steps, dim=1).detach()
            self.last_obj_pred = self.last_obj_pred_steps.mean(dim=1)
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
