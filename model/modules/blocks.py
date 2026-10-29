import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)

class ProgressPosEnc(nn.Module):
    """
    p: [B, T] in [0,1]  ->  PE: [B, T, d_model]
    """
    def __init__(self, d_model=256, num_freqs=16):
        super().__init__()
        # log-spaced frequencies
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), num_freqs))  # [F]
        self.register_buffer("freqs", 2*math.pi*freqs, persistent=False)
        self.proj = nn.Linear(2*num_freqs, d_model)

    def forward(self, p):  # p: [B,T]
        ang = p.unsqueeze(-1) * self.freqs          # [B,T,F]
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,T,2F]
        return self.proj(feat)                      # [B,T,d_model]
    

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=4, causal=True, no_pad=False):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride, no_pad=no_pad)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class CausalDeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalDeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        last_n = self.kernel_size-self.stride
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x

class DeConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalDeConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        else:
            conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, output_padding=stride-1)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=[5,3], stride=[2,2], n_groups=8, causal=True, residual=False, pooling_layers=[]):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size[i], 
                stride[i], 
                n_groups=n_groups, 
                causal=causal
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=sum(stride))
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, 'pooling'):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)


class TimestepEncoder(nn.Module):
    """Encodes integer diffusion timesteps into continuous embeddings.

    This layer mirrors the common "time embedding" pathway used in diffusion models.
    It first maps raw timestep scalars to a higher-dimensional sinusoidal space
    (via :class:`diffusers.models.embeddings.Timesteps`), then passes that through
    a small MLP (:class:`TimestepEmbedding`).

    Args:
        embedding_dim (int): Size of the final time embedding.
        compute_dtype (torch.dtype, optional): Dtype used during computation for
            the projection step. Defaults to ``torch.float32``.

    Shape:
        - Input: ``timesteps`` of shape ``(N,)`` or ``(N, 1)`` with integer-like values.
        - Output: tensor of shape ``(N, embedding_dim)``.

    Example:
        >>> enc = TimestepEncoder(embedding_dim=512)
        >>> t = torch.randint(0, 1000, (8,))
        >>> temb = enc(t)  # (8, 512)
    """

    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        # Timesteps: turns scalar timestep into a sinusoidal embedding of size 256.
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        # TimestepEmbedding: projects 256-dim sinusoidal vector to embedding_dim.
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        """Compute the time embedding.

        Args:
            timesteps (Tensor): Integer-like timesteps, shape ``(N,)`` or ``(N, 1)``.

        Returns:
            Tensor: Time embeddings of shape ``(N, embedding_dim)``.
        """
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned by an external embedding.

    The input is normalized and then scaled and shifted with parameters produced
    from a conditioning vector (e.g., a time embedding). This is a common pattern
    in conditional generative models.

    Args:
        embedding_dim (int): Dimension of the conditioning embedding.
        norm_elementwise_affine (bool, optional): If ``True``, apply learnable affine
            parameters inside the base LayerNorm. Defaults to ``False``.
        norm_eps (float, optional): Epsilon for numerical stability in LayerNorm.
            Defaults to ``1e-5``.
        chunk_dim (int, optional): Unused here but kept for API parity. Defaults to 0.

    Shape:
        - Input: ``x`` of shape ``(N, L, D)`` or ``(N, D)``; ``temb`` of shape ``(N, embedding_dim)``.
        - Output: same shape as ``x``.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        # Base LayerNorm applied to x before adaptive scale/shift.
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply adaptive LayerNorm.

        Args:
            x (Tensor): Input features of shape ``(N, L, D)`` or ``(N, D)``.
            temb (Tensor, optional): Conditioning embedding of shape ``(N, embedding_dim)``.

        Returns:
            Tensor: Normalized and adaptively scaled/shifted features with same shape as ``x``.
        """
        temb = self.linear(self.silu(temb))
        # Split into scale and shift of shape (N, D)
        scale, shift = temb.chunk(2, dim=1)
        # Broadcast scale/shift over sequence/time dimension if present
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class CategorySpecificLinear(nn.Module):
    """Linear layer with per-category parameters.

    Instead of a single weight matrix, this layer holds ``num_categories`` different
    sets of weights/biases and selects one set per item using ``cat_ids``.

    Args:
        num_categories (int): Number of distinct categories.
        input_dim (int): Input feature dimension.
        hidden_dim (int): Output feature dimension.

    Shape:
        - Input: ``x`` of shape ``(N, L, input_dim)``; ``cat_ids`` of shape ``(N,)`` (ints in ``[0, num_categories)``).
        - Output: tensor of shape ``(N, L, hidden_dim)``.
    """

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # Per-category weights/biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        """Apply the category-specific affine transform.

        Args:
            x (Tensor): Input of shape ``(N, L, input_dim)``.
            cat_ids (LongTensor): Category ids of shape ``(N,)``.

        Returns:
            Tensor: Output of shape ``(N, L, hidden_dim)``.
        """
        selected_W = self.W[cat_ids]            # (N, input_dim, hidden_dim)
        selected_b = self.b[cat_ids]            # (N, hidden_dim)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with per-category parameters.

    This composes two :class:`CategorySpecificLinear` layers with a ReLU activation
    in between, allowing different categories to learn different projections.

    Args:
        num_categories (int): Number of categories.
        input_dim (int): Input feature size for the first layer.
        hidden_dim (int): Hidden width.
        output_dim (int): Output feature size.

    Shape:
        - Input: ``x`` of shape ``(N, L, input_dim)``; ``cat_ids`` of shape ``(N,)``.
        - Output: tensor of shape ``(N, L, output_dim)``.
    """

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        """Compute forward pass through the category-specific MLP.

        Args:
            x (Tensor): Input of shape ``(N, L, input_dim)``.
            cat_ids (LongTensor): Category ids of shape ``(N,)``.

        Returns:
            Tensor: Output of shape ``(N, L, output_dim)``.
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class BasicTransformerBlock(nn.Module):
    """A minimal Transformer block with optional cross-attention.

    The block provides:
    1) (Optional) sinusoidal positional embeddings.
    2) Self-attention **or** encoder cross-attention using :class:`nn.MultiheadAttention`
       (with ``batch_first=True`` to accept ``(N, L, D)`` tensors).
    3) A feed-forward network from :mod:`diffusers.models.attention`.

    Notes:
        - If ``encoder_hidden_states`` is provided at forward time, a **cross-attention**
          operation is performed (query = normalized hidden states, key/value = encoder states).
          Otherwise, standard **self-attention** is used.
        - ``encoder_attention_mask`` is passed as ``key_padding_mask`` to PyTorch MHA, where
          ``True`` entries indicate positions to **ignore**.

    Args:
        dim (int): Embedding dimension.
        num_attention_heads (int): Number of attention heads.
        attention_head_dim (int): *Unused here*; kept for API parity.
        dropout (float, optional): Dropout probability inside attention/FFN. Defaults to 0.0.
        cross_attention_dim (int, optional): *Unused here*; kept for API parity.
        activation_fn (str, optional): Activation in the FFN (e.g., ``"geglu"``). Defaults to ``"geglu"``.
        attention_bias (bool, optional): If ``True``, enables bias in attention projections.
        norm_elementwise_affine (bool, optional): If ``True``, LayerNorms have affine params.
        norm_type (str, optional): Normalization type. ``"ada_norm"`` uses :class:`AdaLayerNorm`.
        norm_eps (float, optional): Epsilon for LayerNorm.
        final_dropout (bool, optional): If ``True``, apply dropout to attention output.
        positional_embeddings (str, optional): If ``"sinusoidal"``, add sinusoidal position encoding.
        num_positional_embeddings (int, optional): Sequence length for position encoding.
        ff_inner_dim (int, optional): Hidden width of the FFN.
        ff_bias (bool, optional): If ``True``, enable linear biases in FFN.

    Shape:
        - Input: ``hidden_states`` of shape ``(N, L, D)``.
        - Output: same shape as input.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        # Optional positional embedding
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # 1) Normalization before attention
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        # 2) Self- or Cross-Attention (PyTorch MHA supports key_padding_mask)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            bias=attention_bias,
            batch_first=True
        )

        # 3) Feed-forward network and post-attention norm
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Run attention + feed-forward with residual connections.

        Args:
            hidden_states (Tensor): Input of shape ``(N, L, D)``.
            attention_mask (Tensor, optional): *Unused here.* Consider pre-masking inputs
                if required.
            encoder_hidden_states (Tensor, optional): If given, perform cross-attention
                over these encoder states of shape ``(N, S, D)``.
            encoder_attention_mask (BoolTensor, optional): Key padding mask for attention
                (``True`` positions are ignored). Shape ``(N, S)`` for cross-attention
                or ``(N, L)`` for self-attention.
            temb (Tensor, optional): Time/context embedding if ``norm_type == 'ada_norm'``.

        Returns:
            Tensor: Output tensor of shape ``(N, L, D)``.
        """
        # Pre-norm (optionally adaptive)
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # Optional positional embeddings
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # Cross-attention if encoder states are provided; otherwise self-attention.
        if encoder_hidden_states is not None:
            attn_output, _ = self.attn1(
                query=norm_hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                key_padding_mask=encoder_attention_mask,
                attn_mask=attention_mask,
            )
        else:
            attn_output, _ = self.attn1(
                query=norm_hidden_states,
                key=norm_hidden_states,
                value=norm_hidden_states,
                key_padding_mask=encoder_attention_mask,
                attn_mask=attention_mask,
            )

        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        # Residual connection
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward + residual
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

class VectorQuantizer(nn.Module):
    """Vector-quantization layer with a learnable codebook and optional code restart.
    from https://github.com/OpenDriveLab/UniVLA/blob/main/latent_action_model/genie/modules/blocks.py#L362

    Given continuous inputs, this layer picks the nearest codebook vector for each
    position, updates (softly) usage statistics, and returns a straight-through
    estimator for backpropagation.

    Args:
        num_latents (int): Size of the codebook (number of codes).
        latent_dim (int): Dimensionality of each code vector.
        code_restart (bool, optional): If ``True``, periodically re-initialize
            "dead" codes based on usage statistics. Defaults to ``True``.

    Returns (from ``forward``):
        Tuple[Tensor, Tensor, Tensor, Tensor]: ``(z_q, z, x, indices)`` where
        - ``z_q``: straight-through quantized output (same shape as ``x``),
        - ``z``: nearest codebook embeddings,
        - ``x``: original inputs (for loss terms outside),
        - ``indices``: indices of selected codes.

    Notes:
        - Usage is tracked per index. A code is considered "dead" if it hasn't
          been used within an epoch; such codes can be randomly restarted.
    """

    def __init__(self, num_latents: int, latent_dim: int, code_restart: bool = True) -> None:
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        # Small symmetric init improves early training stability.
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)

        # Initialize a usage buffer (not persisted to checkpoints by default).
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)
        self.num_latents = num_latents

        self.code_restart = code_restart
        # Rows [0, reserved) are idle codes pinned by a slot-gated model
        # (model/stage1_factorized.py, slot_gating): never counted dead and never
        # a restart source, so they are shaped only by the codebook/commitment
        # terms of the tokens gated onto them.
        self.reserved = 0

    def update_usage(self, min_enc) -> None:
        """Increment usage counters for the selected code indices.

        Args:
            min_enc (LongTensor): Code indices of shape ``(*)``.
        """
        for idx in min_enc:
            self.usage[idx] = self.usage[idx] + 1  # Add used code

    def random_restart(self) -> None:
        """Randomly restart all codes that were not used in the last epoch.

        This helps avoid permanently "dead" codes by re-initializing them with
        values sampled from other (active) codes.
        """
        if self.code_restart:
            # Codes with usage < 1 are considered dead.
            reserved = getattr(self, "reserved", 0)
            dead_codes = torch.nonzero(self.usage < 1).squeeze(1)
            dead_codes = dead_codes[dead_codes >= reserved]
            candidates = torch.arange(reserved, self.num_latents)
            rand_codes = candidates[torch.randperm(len(candidates))[0:len(dead_codes)]]
            print(f"Restarting {len(dead_codes)} codes")
            with torch.no_grad():
                self.codebook.weight[dead_codes] = self.codebook.weight[rand_codes]

            # Propagate to inner VQ if present (hierarchical VQ setups).
            if hasattr(self, "inner_vq"):
                self.inner_vq.random_restart()

    def reset_usage(self) -> None:
        """Reset per-code usage counters (e.g., between epochs)."""
        if self.code_restart:
            # Reset usage between epochs
            self.usage.zero_()

            if hasattr(self, "inner_vq"):
                self.inner_vq.reset_usage()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Quantize inputs against the codebook using nearest neighbor lookups.

        Args:
            x (Tensor): Input features; last dimension must equal ``latent_dim``.
                Shape can be ``(..., latent_dim)``.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - ``z_q`` (Tensor): Straight-through quantized outputs (same shape as ``x``).
                - ``z`` (Tensor): Codebook embeddings corresponding to nearest codes.
                - ``x`` (Tensor): The original inputs (passed through for convenience).
                - ``indices`` (LongTensor): Indices of the selected codes.
        """
        # Compute pairwise distances between x and all codebook vectors.
        distance = torch.cdist(x, self.codebook.weight)

        # Select nearest code index and retrieve corresponding embeddings.
        indices = torch.argmin(distance, dim=-1)
        # indices = torch.randint(0, 31, (8,4)).to('cuda')
        z = self.codebook(indices)

        # Update per-code usage statistics (during training or when restarts are on).
        if not self.training or self.code_restart:
            self.update_usage(indices)

        # Straight-through estimator: forward uses z, backward uses identity.
        z_q = x + (z - x).detach()
        return z_q, z, x, indices, distance
    
    @torch.no_grad()
    def inference_forward(self, x: Tensor):
        """
        A lightweight inference version of forward():
        - same nearest-neighbor lookup as forward
        - no autograd / no backprop
        - no usage update / no straight-through trick needed
        """
        # Compute distances to all codebook vectors
        distance = torch.cdist(x, self.codebook.weight)

        # Pick nearest code indices
        indices = torch.argmin(distance, dim=-1)

        # Get nearest embeddings from codebook
        z = self.codebook(indices)

        return z, indices, distance


class ResidualVectorQuantizer(nn.Module):
    """Multi-level VQ: level 1 quantizes x, each later level quantizes the residual.

    Why (measured 2026-09-03, EXPERIMENTS.md "The quantizer is the constraint").
    A single 128-code level has to do two incompatible jobs at once, and the
    ledger has the numbers for both:

      * it MAKES TRANSFER WORK -- embodiment identity per token drops 0.654 in
        the continuous key to 0.261 after quantization (the `grounded` gate),
        0.937 -> 0.299 on ee-obj. Rounding to a shared vocabulary is what strips
        robot identity.
      * it DESTROYS SCENE PRECISION -- `grounded`'s visual FDM drove the
        gripper->goal read to 2.51 cm in the key, the DINO ceiling, and
        quantization returned it to 3.85 cm. ee-obj: 1.03 -> 9.25 cm.

    Residual quantization separates the two. Level 1 stays the coarse shared
    alphabet, and that is where the embodiment-blindness claim lives; the deeper
    levels encode what level 1 rounds away, so the SUM can carry precision the
    alphabet does not have to.

    Standard construction (SoundStream / EnCodec). NOT "more codes": dyn already
    took one level 128 -> 192 and moved nothing, villa-X uses 32 codes and UniVLA
    16. More LEVELS, not a bigger level.

    Interface is deliberately identical to VectorQuantizer so nothing downstream
    changes: `forward` returns the same 5-tuple, `inference_forward` the same
    3-tuple (both with z = the SUM over levels, and the LEVEL-1 indices and
    distances, which is what the perplexity/usage logging and the code-restart
    machinery consume). `cumulative_levels` is the extra hook the gate needs to
    probe level 1 alone against all levels.

    The commitment/codebook loss is computed outside, by
    MOTIFStage1.compute_loss, from (z_e, z). With z = sum of levels that is the
    correct joint objective: d/dz_l ||sum_l z_l - z_e||^2 is the same for every
    level, so all codebooks are fitted to explain the input jointly while the
    nearest-neighbour assignment stays greedy on residuals.
    """

    def __init__(self, num_latents: int, latent_dim: int, num_levels: int = 3,
                 code_restart: bool = True) -> None:
        super().__init__()
        assert num_levels >= 1, num_levels
        self.levels = nn.ModuleList([
            VectorQuantizer(num_latents, latent_dim, code_restart=code_restart)
            for _ in range(num_levels)
        ])
        self.num_latents = num_latents
        self.num_levels = num_levels
        self.code_restart = code_restart

    @property
    def codebook(self) -> nn.Embedding:
        """Level 1's codebook -- the shared alphabet, and what callers asking for
        `vq.codebook` (stage 2/3 reading the code width) mean."""
        return self.levels[0].codebook

    @property
    def usage(self) -> Tensor:
        """Level 1's usage counters.

        MOTIFStage1.plot_and_reset_code_usage reads `vq.usage` directly at every
        epoch boundary, so this is not optional -- omitting it crashed the first
        rvq launch at the end of epoch 1 (2026-09-03 16:20). Level 1 is the right
        answer for the same reason `codebook` returns level 1's: the usage plot
        is about the shared alphabet. Per-level counters remain on
        `self.levels[i].usage` for anyone who wants the deeper ones.
        """
        return self.levels[0].usage

    def _sync_restart(self) -> None:
        for lv in self.levels:
            lv.code_restart = self.code_restart

    def random_restart(self) -> None:
        self._sync_restart()
        if self.code_restart:
            for lv in self.levels:
                lv.random_restart()

    def reset_usage(self) -> None:
        self._sync_restart()
        if self.code_restart:
            for lv in self.levels:
                lv.reset_usage()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        residual = x
        z_total, idx0, dist0 = 0.0, None, None
        for i, lv in enumerate(self.levels):
            distance = torch.cdist(residual, lv.codebook.weight)
            indices = torch.argmin(distance, dim=-1)
            z_l = lv.codebook(indices)
            if not self.training or lv.code_restart:
                lv.update_usage(indices)
            z_total = z_total + z_l
            # Detached so the greedy assignment of level i+1 does not backprop
            # through level i's codebook; the gradient path is z_total.
            residual = residual - z_l.detach()
            if i == 0:
                idx0, dist0 = indices, distance
        z_q = x + (z_total - x).detach()
        return z_q, z_total, x, idx0, dist0

    @torch.no_grad()
    def inference_forward(self, x: Tensor):
        residual = x
        z_total, idx0, dist0 = 0.0, None, None
        for i, lv in enumerate(self.levels):
            distance = torch.cdist(residual, lv.codebook.weight)
            indices = torch.argmin(distance, dim=-1)
            z_l = lv.codebook(indices)
            z_total = z_total + z_l
            residual = residual - z_l
            if i == 0:
                idx0, dist0 = indices, distance
        return z_total, idx0, dist0

    @torch.no_grad()
    def cumulative_levels(self, x: Tensor):
        """[z_1, z_1+z_2, ...] -- for the gate, which must report level 1 alone
        (the alphabet: embodiment must stay near chance) separately from the full
        sum (the geometry: ee->goal must move toward the DINO ceiling)."""
        residual, running, out = x, 0.0, []
        for lv in self.levels:
            indices = torch.argmin(torch.cdist(residual, lv.codebook.weight), dim=-1)
            z_l = lv.codebook(indices)
            running = running + z_l
            residual = residual - z_l
            out.append(running.clone())
        return out


class PerSlotQuantizer(nn.Module):
    """One independent codebook per kinematic slot, instead of one shared over all.

    Motivation (user, 2026-09-04): translation (R^3, metres), orientation (SO(3)
    via quaternions) and the gripper (1-D) are topologically distinct, and a
    single shared vocabulary makes them compete for the same vectors and learn
    entangled geometric compromises.

    Measured caveat, on grounded's epoch_0020 over 30 720 tokens: the shared
    codebook has already largely self-partitioned. 110 of 127 used codes (87%)
    are touched by exactly ONE slot; translation/orientation/gripper use 52/45/51
    codes each; only ~20% of each slot's token mass lands on a code another slot
    also uses. So the competition this removes is mostly already absent, and the
    dominant effect of per-slot codebooks at the same `codebook_size` is a ~2.5x
    capacity increase per slot (128 each instead of ~50 of a shared 128), not a
    topological fix. Run the capacity-matched variant (codebook_size ~= 128/3)
    alongside to separate the two.

    Interface matches VectorQuantizer so nothing downstream changes, with one
    addition: `forward(x, slot=i)` selects a codebook explicitly (the stage-1
    path already loops over slots), while `inference_forward(x)` on a full
    [B, S*H, D] key dispatches by token position, which is what stage 2/3 need.
    All slots share one `codebook_size`; indices are LOCAL to a slot, so the
    usage/perplexity statistics aggregate across slots as before.
    """

    def __init__(self, num_latents: int, latent_dim: int, num_slots: int = 3,
                 code_restart: bool = True) -> None:
        super().__init__()
        self.slots = nn.ModuleList([
            VectorQuantizer(num_latents, latent_dim, code_restart=code_restart)
            for _ in range(num_slots)
        ])
        self.num_latents = num_latents
        self.num_slots = num_slots
        self.code_restart = code_restart

    @property
    def codebook(self) -> nn.Embedding:
        return self.slots[0].codebook

    @property
    def usage(self) -> Tensor:
        """Summed over slots -- indices are slot-local, so this is the aggregate
        histogram the epoch-boundary plot has always shown."""
        return torch.stack([s.usage for s in self.slots], dim=0).sum(0)

    def _sync(self) -> None:
        for s in self.slots:
            s.code_restart = self.code_restart

    def random_restart(self) -> None:
        self._sync()
        if self.code_restart:
            for s in self.slots:
                s.random_restart()

    def reset_usage(self) -> None:
        self._sync()
        if self.code_restart:
            for s in self.slots:
                s.reset_usage()

    def forward(self, x: Tensor, slot: int = 0):
        return self.slots[slot](x)

    @torch.no_grad()
    def inference_forward(self, x: Tensor):
        """x: [B, S*H, D] -- split by token position, one codebook per slot."""
        B, N, D = x.shape
        assert N % self.num_slots == 0, (N, self.num_slots)
        H = N // self.num_slots
        zs, idxs, ds = [], [], []
        for i, s in enumerate(self.slots):
            z, idx, dist = s.inference_forward(x[:, i * H:(i + 1) * H])
            zs.append(z); idxs.append(idx); ds.append(dist)
        return torch.cat(zs, 1), torch.cat(idxs, 1), torch.cat(ds, 1)
