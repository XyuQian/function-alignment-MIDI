import torch
from torch import Tensor
from torch.nn import Module, Linear, Dropout
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from .mha_func import multi_head_attention_forward
from typing import Optional, Tuple
from shoelace.utils.network_utils import generator_switch

class MultiheadAttention(Module):
    """
    Implements Multihead Attention as described in 'Attention Is All You Need'.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device=None,
        dtype=None,
        use_generator: bool = False,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_generator = use_generator
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
            constant_(self.k_proj.bias, 0.0)
            constant_(self.v_proj.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Computes multi-head attention using scaled dot-product attention.
        """

        attn_output = generator_switch(multi_head_attention_forward(
            query,
            query,
            query,
            self.embed_dim,
            self.num_heads,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.dropout,
            self.out_proj,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=is_causal,
            training=self.training,
            use_generator=self.use_generator
        ), use_generator=self.use_generator)

        return attn_output
