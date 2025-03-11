import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Any, Dict
from torch.utils.checkpoint import checkpoint  # for gradient checkpointing

###############################################################################
# Utility: Sin Embedding, Norm, LayerScale
###############################################################################

def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    positions = positions.to(dtype)
    freq_seq = torch.arange(half, device=positions.device, dtype=dtype).view(1, 1, -1)
    phase = positions / (max_period ** (freq_seq / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    raise ValueError(f"Unknown norm type: {norm_type}")

class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4, channel_last: bool = True):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x if self.channel_last else self.scale[:, None] * x

###############################################################################
# Basic Multihead Attention
###############################################################################

class BasicMultiheadAttention(nn.Module):
    """
    Multi-head self/cross attention using torch’s built-in scaled_dot_product_attention
    for memory efficiency, plus optional generator yields.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        cross_attention: bool = False,
        use_generator: Optional[bool] = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.cross_attention = cross_attention
        self.use_generator = use_generator

        # Single in-proj layer
        in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias

        # Out projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def set_use_generator(self, flag : bool):
        self.use_generator = flag
        
    def init_qkv(self):
        """
        Splits in_proj_weight,bias into q_proj, k_proj, v_proj
        """
        d = self.embed_dim
        W = self.in_proj_weight  # [3*d, d]
        B = self.in_proj_bias    # [3*d] or None

        self.q_proj = nn.Linear(d, d, bias=(B is not None))
        self.k_proj = nn.Linear(d, d, bias=(B is not None))
        self.v_proj = nn.Linear(d, d, bias=(B is not None))

        self.q_proj.weight.data = W[:d, :]
        self.k_proj.weight.data = W[d:2*d, :]
        self.v_proj.weight.data = W[2*d:, :]
        if B is not None:
            self.q_proj.bias.data = B[:d]
            self.k_proj.bias.data = B[d:2*d]
            self.v_proj.bias.data = B[2*d:]

        # Remove original in_proj references
        del self.in_proj_weight
        del self.in_proj_bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, T, C]
            attn_mask (optional): shape compatible with scaled_dot_product_attention
        Returns:
            x: attention output [B, T, C]
        """
        B, Tq, C = query.shape
        Tk = key.shape[1]

        # If using causal attention and Tq == Tk, set is_causal
        is_causal = self.causal and (Tq == Tk)

        # Compute Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head [B, num_heads, T, head_dim]
        head_dim = C // self.num_heads
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        # Use PyTorch 2.0+ scaled_dot_product_attention
        # This automatically handles q scaling by sqrt(dim), so do NOT scale q yourself
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,           # shape [B, num_heads, Tq, Tk] or broadcastable
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        # shape of attn_output: [B, num_heads, Tq, head_dim]

        # Merge heads
        x = rearrange(attn_output, "b h t d -> b t (h d)")

        # If generator logic is toggled on, yield partial states
        if self.use_generator:
            wrap_attn_output = [{
                "attn_output": x,
                "query": query,
                "q": q
            }]
            yield wrap_attn_output
            x = wrap_attn_output[0]["attn_output"]

        # Out projection
        x = self.out_proj(x)
        return x

###############################################################################
# TransformerEncoderLayer with optional cross-attn
###############################################################################

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Single transformer layer with BasicMultiheadAttention,
    optional cross-attention, and optional layer-scale.
    No streaming logic or caching.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        bias_ff: bool = True,
        bias_attn: bool = True,
        causal: bool = False,
        cross_attention: bool = False,
        layer_scale: Optional[float] = None,
        norm: str = "layer_norm",
        use_generator: Optional[bool] = False,
        **kwargs
    ):
        super().__init__(
            d_model,
            num_heads,
            dim_feedforward,
            dropout,
            batch_first=True,
            **kwargs
        )
        # Overwrite self_attn
        self.self_attn = BasicMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias_attn,
            causal=causal,
            cross_attention=False,
            use_generator=use_generator
        )

        # Redefine feed-forward with optional bias
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias_ff)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias_ff)

        self.cross_attention = None
        self.use_generator = use_generator
        if cross_attention:
            self.cross_attention = BasicMultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias_attn,
                causal=False,
                cross_attention=True
            )
            self.dropout_cross = nn.Dropout(dropout)
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5)
            self.layer_scale_cross = LayerScale(d_model, layer_scale) if layer_scale else nn.Identity()

        # Optional layer-scale
        self.layer_scale_1 = LayerScale(d_model, layer_scale) if layer_scale else nn.Identity()
        self.layer_scale_2 = LayerScale(d_model, layer_scale) if layer_scale else nn.Identity()

        # Regenerate norms if needed
        self.norm1 = create_norm_fn(norm, d_model)
        self.norm2 = create_norm_fn(norm, d_model)

    def init_qkv(self):
        self.self_attn.init_qkv()
        if self.cross_attention:
            self.cross_attention.init_qkv()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        cross_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = src
        # Pre-norm or post-norm logic from nn.TransformerEncoderLayer
        if self.norm_first:
            # 1) Self-attn
            norm_x = self.norm1(x)
            sa_out = yield from self.self_attn(norm_x, norm_x, norm_x, attn_mask=src_mask)
            x = x + self.layer_scale_1(self.dropout1(sa_out))

            # 2) Cross-attn
            if cross_src is not None and self.cross_attention:
                ca_out = self.cross_attention(self.norm_cross(x), cross_src, cross_src)
                x = x + self.layer_scale_cross(self.dropout_cross(ca_out))

            # 3) Feed-forward
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
            return x + self.layer_scale_2(self.dropout2(ff_out))
        else:
            # 1) Self-attn
            sa_out = yield from self.self_attn(x, x, x, attn_mask=src_mask)
            x = self.norm1(x + self.layer_scale_1(self.dropout1(sa_out)))

            # 2) Cross-attn
            if cross_src is not None and self.cross_attention:
                ca_out = self.cross_attention(x, cross_src, cross_src)
                x = self.norm_cross(x + self.layer_scale_cross(self.dropout_cross(ca_out)))

            # 3) Feed-forward
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.norm2(x + self.layer_scale_2(self.dropout2(ff_out)))

###############################################################################
# Memory-Efficient Transformer: wraps layer calls in torch.checkpoint
###############################################################################

class Transformer(nn.Module):
    """
    Stacked TransformerEncoderLayer with optional sinusoidal embeddings.
    Supports gradient checkpointing for memory efficiency,
    without altering layer forward logic.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
        cross_attention: bool = False,
        layer_scale: Optional[float] = None,
        positional_embedding: str = "sin",
        max_period: float = 10000.0,
        positional_scale: float = 1.0,
        use_generator: bool = False,
        memory_efficient: bool = False,
        **kwargs
    ):
        super().__init__()
        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.use_generator = use_generator
        self.memory_efficient = memory_efficient

        # Build layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                causal=causal,
                cross_attention=cross_attention,
                layer_scale=layer_scale,
                use_generator=use_generator,
                **kwargs
            )
            for _ in range(num_layers)
        ])

    def init_qkv(self):
        for layer in self.layers:
            layer.init_qkv()

    def set_use_generator(self, flag : bool):
        self.use_generator = flag
        for layer in self.layers:
            self.layer.set_use_generator(flag)

    def _apply_layer(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        cross_src: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Helper that either calls layer(x, src_mask, cross_src)
        directly or uses torch.checkpoint for memory efficiency.
        """
        def forward_fn(*inputs):
            return layer(*inputs)

        if not self.memory_efficient:
            return layer(x, src_mask=src_mask, cross_src=cross_src)
        else:
            # checkpoint requires all inputs to be tensors or
            # have them in a tuple.
            # We pass them as a single tuple for convenience.
            return checkpoint(forward_fn, x, src_mask, cross_src)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        cross_src: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Optionally add sinusoidal embeddings
        if "sin" in self.positional_embedding:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        # Pass through each layer, potentially with gradient checkpointing
        for layer in self.layers:
            out = yield from self._apply_layer(layer, x, src_mask, cross_src)
            x = out

        return x

    def make_optim_group(
        self,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None
    ):
        group = {"params": list(self.parameters())}
        if lr is not None:
            group["lr"] = lr
        if weight_decay is not None:
            group["weight_decay"] = weight_decay
        return group
