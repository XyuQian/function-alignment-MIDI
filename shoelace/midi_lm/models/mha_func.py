import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from shoelace.utils.network_utils import make_yield


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim: int,
        num_heads: int,
        q_proj: Callable[[Tensor], Tensor],
        k_proj: Callable[[Tensor], Tensor],
        v_proj: Callable[[Tensor], Tensor],
        dropout_p: float,

        out_proj: Callable[[Tensor], Tensor],
        key_padding_mask: Optional[Tensor] = None,
        training: bool = True,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        use_generator: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute multi-head attention forward pass using scaled dot-product attention.
    """
    batch_size, tgt_len, embed_dim = query.shape
    assert embed_dim == embed_dim, "Embedding dimensions must match."
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    if not training:
        dropout_p = 0.0
    q = q_proj(query)
    k = k_proj(key)
    v = v_proj(value)

    q = q.contiguous().view(batch_size, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.contiguous().view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    v = v.contiguous().view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch_size, key.shape[1]), \
            "key_padding_mask shape must match (batch_size, key_length)"

        # Expand to (batch_size, 1, 1, key_len) and broadcast over heads & query positions
        key_padding_mask = key_padding_mask[:, None, None, :]  # Shape: (batch_size, 1, 1, key_len)
        key_padding_mask = key_padding_mask.masked_fill(key_padding_mask, float('-inf'))  # Set padding to -inf
        # print(attn_mask.shape, key_padding_mask.shape)
        # If attn_mask is also provided, combine them
        is_causal = False
        if attn_mask is not None:
            attn_mask = attn_mask + key_padding_mask
        else:
            attn_mask = key_padding_mask

    attn_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, is_causal)
    attn_output = (
        attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size * tgt_len, embed_dim)
    )

    if use_generator:
        yield_output = {
            "attn_output": attn_output,
            "query": query,
            "q": q
        }
        wrap_attn_output = [yield_output]
        make_yield(wrap_attn_output)
        attn_output = wrap_attn_output[0]["attn_output"]

    attn_output = out_proj(attn_output)
    attn_output = attn_output.view(batch_size, tgt_len, attn_output.size(1))
    return attn_output
