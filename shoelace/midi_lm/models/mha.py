import torch
from torch import Tensor
from torch.nn import Module, Linear, Dropout
from torch.nn.init import constant_, xavier_uniform_
from .mha_func import multi_head_attention_forward
from typing import Optional, Tuple, Dict


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
        self.kv_cache = None
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


    def reset_cache(self):
        self.kv_cache = None

    def forward(
            self,
            query: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Computes multi-head attention using scaled dot-product attention.
        
        Args:
            query: Input tensor of shape (batch_size, tgt_len, embed_dim).
            key_padding_mask: Optional mask for keys.
            need_weights: Whether to return attention weights (unused here).
            attn_mask: Optional attention mask.
            is_causal: Whether to apply a causal mask.
            kv_cache: Optional dictionary containing key/value cache for inference.
            
        Returns:
            A tuple (attn_output, kv_cache) where attn_output is the attention output,
            and kv_cache is the updated key/value cache.
        """
        
        attn_output, kv_cache = yield from multi_head_attention_forward(
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
            kv_cache=self.kv_cache,
            use_generator=self.use_generator
        )
        self.kv_cache = kv_cache
        return attn_output
