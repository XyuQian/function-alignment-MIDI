import torch
from torch import Tensor
from typing import Callable, Optional, Union
from torch.nn import Module, Linear, Dropout, LayerNorm
import torch.nn.functional as F
from shoelace.utils.network_utils import _get_clones, _get_activation_fn
from .mha import MultiheadAttention


class TransformerEncoderLayer(Module):
    """
    Transformer Encoder Layer with support for LoRA and generator-based forward passes.
    """
    __constants__ = ["norm_first"]

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            device=None,
            dtype=None,
            use_generator: bool = False,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first,
            use_generator=use_generator, **factory_kwargs
        )

        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation
        

        self.use_generator = use_generator

    def reset_cache(self):
        self.self_attn.reset_cache()

    def forward(
            self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False
    ):
        """
        Forward pass with optional generator-based execution.
        """
        

        x = src
        # print(x[0, 0, 100:300])
        # # print("-----------------------------------")
        
        if self.norm_first:
            x = x + (yield from self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            ))
            x = x + self._ff_block(self.norm2(x))
        else:
            
            x = self.norm1(
                x
                + (yield from self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal
                                                  )))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  is_causal: bool = False):
        """
        Self-attention block with optional generator-based support.
        """
        x = (yield from self.self_attn(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal=is_causal
        ))
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Feed-forward block with activation and dropout.
        """
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class TransformerEncoder(Module):
    """
    Transformer Encoder supporting generator-based execution and LoRA modifications.
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers: int, norm: Optional[Module] = None, use_generator: bool = False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.use_generator = use_generator
        


    def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None):
        """
        Forward pass with optional generator-based execution.
        """
        output = src

        for layer in self.layers:
            output = yield from layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    def load_from_torch_model(self, state_dict):
    
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            # Extract parameters for the current layer
            layer_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items() if k.startswith(prefix)
            }
            # Process multi-head attention weights specially
            if "self_attn.in_proj_weight" in layer_state:
                in_proj_weight = layer_state.pop("self_attn.in_proj_weight")
                d_model = layer.self_attn.embed_dim
                # Split the fused weight into q, k, and v weights.
                layer.self_attn.q_proj.weight.data.copy_(in_proj_weight[:d_model, :])
                layer.self_attn.k_proj.weight.data.copy_(in_proj_weight[d_model:2 * d_model, :])
                layer.self_attn.v_proj.weight.data.copy_(in_proj_weight[2 * d_model:3 * d_model, :])
            if "self_attn.in_proj_bias" in layer_state:
                in_proj_bias = layer_state.pop("self_attn.in_proj_bias")
                d_model = layer.self_attn.embed_dim
                layer.self_attn.q_proj.bias.data.copy_(in_proj_bias[:d_model])
                layer.self_attn.k_proj.bias.data.copy_(in_proj_bias[d_model:2 * d_model])
                layer.self_attn.v_proj.bias.data.copy_(in_proj_bias[2 * d_model:3 * d_model])
            # Load out projection weights directly.
            if "self_attn.out_proj.weight" in layer_state:
                layer.self_attn.out_proj.weight.data.copy_(layer_state.pop("self_attn.out_proj.weight"))
            if "self_attn.out_proj.bias" in layer_state:
                layer.self_attn.out_proj.bias.data.copy_(layer_state.pop("self_attn.out_proj.bias"))
            # Load remaining parameters (e.g. for linear layers and layer norms)
            layer.load_state_dict(layer_state, strict=False)
    
        # Load the global normalization weights if present.
        if self.norm is not None:
            norm_state = {k[len("norm."):]: v for k, v in state_dict.items() if k.startswith("norm.")}
            self.norm.load_state_dict(norm_state)

