from torch import Tensor
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict

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
        kv_cache: Optional[Dict[str, Tensor]] = None,
        use_generator: bool = False,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    """
    Compute multi-head attention forward pass using scaled dot-product attention.
    Supports key/value caching for faster inference.
    
    Args:
        query: Input tensor of shape (batch_size, tgt_len, embed_dim).
        key: Key tensor.
        value: Value tensor.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        q_proj, k_proj, v_proj: Projection functions for query, key, and value.
        dropout_p: Dropout probability.
        out_proj: Final projection function.
        key_padding_mask: Optional mask for keys.
        training: If False, inference mode is assumed.
        attn_mask: Optional attention mask.
        is_causal: Whether to apply a causal mask.
        kv_cache: Optional dictionary containing cached tensors for inference.
                  Expected keys: "past_k", "past_v", "past_q", "past_query".
        use_generator: If True, yields a dictionary with intermediate outputs.
        
    Returns:
        A tuple (attn_output, kv_cache). In training mode, kv_cache remains unchanged.
    """
    batch_size, tgt_len, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    # Disable dropout during inference
    if not training:
        dropout_p = 0.0

    q = q_proj(query)
    k = k_proj(key)
    v = v_proj(value)

    # Reshape projections to (batch_size, num_heads, seq_len, head_dim)
    q = q.contiguous().view(batch_size, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.contiguous().view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    v = v.contiguous().view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    # If a key/value cache is provided (for inference), append the new keys/values and queries to the cached ones.
    if not training and kv_cache is not None:
        past_k = kv_cache.get("past_k")
        past_v = kv_cache.get("past_v")
        past_q = kv_cache.get("past_q")
        past_query = kv_cache.get("past_query")
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        past_q = torch.cat([past_q, q], dim=2)
        past_query = torch.cat([past_query, query], dim=1)
        

        kv_cache["past_k"] = k
        kv_cache["past_v"] = v
        kv_cache["past_q"] = past_q
        kv_cache["past_query"] = past_query


    if is_causal and attn_mask is None:
        tgt_len, src_len = q.shape[2], k.shape[2]
        causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=q.device), diagonal=1).bool()
        causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        is_causal = False
        
    

    if key_padding_mask is not None:
        # Ensure key_padding_mask shape matches key length
        assert key_padding_mask.shape == (batch_size, k.shape[2]), \
            "key_padding_mask shape must match (batch_size, key_length)"
        # Expand mask to (batch_size, 1, 1, key_length)
        key_padding_mask = key_padding_mask[:, None, None, :]
        key_padding_mask = key_padding_mask.masked_fill(key_padding_mask, float('-inf'))
        if attn_mask is not None:
            attn_mask = attn_mask + key_padding_mask
        else:
            attn_mask = key_padding_mask
    
    print(q.shape, k.shape, v.shape, attn_mask.shape)
    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size * tgt_len, embed_dim)

    if use_generator:
        if not training and kv_cache is not None:
            wrap_attn_output = [{
                "attn_output": attn_output,
                "query": past_query,
                "q": past_q
            }]
        else:
            wrap_attn_output = [{
                "attn_output": attn_output,
                "query": query,
                "q": q
            }]
        yield wrap_attn_output
        attn_output = wrap_attn_output[0]["attn_output"]

    attn_output = out_proj(attn_output)
    attn_output = attn_output.view(batch_size, tgt_len, attn_output.size(1))
    if kv_cache is None:
        kv_cache = {
            "past_q": q,
            "past_k": k,
            "past_v": v,
            "past_query": query
        }
    return attn_output, kv_cache
