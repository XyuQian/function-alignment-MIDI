import torch
import copy
from torch.nn import ModuleList
from torch import Tensor
from typing import Any, Callable, Optional, Union
import torch.nn.functional as F


import torch
import torch.nn.functional as F

def sample(logits, top_k_val=20, temperature=1.0):
    """
    Samples the next token from logits using top-k sampling.
    
    This function supports 2D logits (batch, vocab_size) and 3D logits 
    (batch, seq_len, vocab_size).
    
    Args:
        logits: Tensor of shape (batch, vocab_size) or (batch, seq_len, vocab_size)
        top_k_val: The number of top tokens to consider.
        temperature: Temperature factor for scaling logits.
        
    Returns:
        A tensor containing the sampled token indices with shape:
          - (batch, 1) for 2D logits, or
          - (batch, seq_len, 1) for 3D logits.
    """
    logits = logits / temperature

    if logits.dim() == 2:
        # For logits of shape (batch, vocab_size)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k_val, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices.gather(-1, sampled_indices)
        return next_token  # Shape: (batch, 1)
    
    elif logits.dim() == 3:
        # For logits of shape (batch, seq_len, vocab_size)
        batch, seq_len, vocab_size = logits.shape
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k_val, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        # Flatten the batch and sequence dimensions for sampling.
        flat_probs = top_k_probs.reshape(-1, top_k_val)
        sampled_indices = torch.multinomial(flat_probs, num_samples=1)
        flat_topk_indices = top_k_indices.reshape(-1, top_k_val)
        next_token_flat = flat_topk_indices.gather(-1, sampled_indices)
        # Reshape back to (batch, seq_len, 1)
        next_token = next_token_flat.view(batch, seq_len, 1)
        return next_token
    else:
        raise ValueError("logits must be either a 2D or 3D tensor")


def generator_switch(x, use_generator, use_from=True):
    if not use_generator:
        return x
    if use_from:
        make_yield_from(x)
    else:
        make_yield(x)


def make_yield_from(x):
    res = yield from x
    return res

def make_yield(x):
    yield x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def freeze(model, is_freeze=True):
    for n, p in model.named_parameters():
        p.requires_grad = not is_freeze


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def print_params(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)


def init_weights(model):
    for n, p in model.named_parameters():
        torch.nn.init.xavier_uniform_(p.weight)


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]
