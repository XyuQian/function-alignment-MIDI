import logging
import math
import typing as tp
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from ..modules.transformer import Transformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn
from ..utils import utils

from shoelace.utils.network_utils import generator_switch, print_params

logger = logging.getLogger(__name__)

ConditionTensors = tp.Dict[str, ConditionType]
# Single-pass CFG: we either have unconditional + conditional together or no conditions
CFGConditions = ConditionTensors


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """Returns partial initializer based on method and dimension."""
    std = 1 / math.sqrt(input_dim)
    if init_depth is not None:
        std /= math.sqrt(2 * init_depth)
    if method == 'gaussian':
        return partial(torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3*std, b=3*std)
    elif method == 'uniform':
        bound = math.sqrt(3) * std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    raise ValueError(f"Unsupported init method: {method}")


def init_layer(
    module: nn.Module,
    method: str,
    init_depth: tp.Optional[int] = None,
    zero_bias_init: bool = False
):
    """Apply an initialization scheme to a linear or embedding layer."""
    if isinstance(module, nn.Linear):
        init_fn = get_init_fn(method, module.in_features, init_depth)
        if module.weight.device.type == 'cpu' and module.weight.dtype == torch.float16:
            w = module.weight.float()
            init_fn(w)
            module.weight.data.copy_(w.half())
        else:
            init_fn(module.weight)
        if zero_bias_init and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        init_fn = get_init_fn(method, module.embedding_dim)
        if module.weight.device.type == 'cpu' and module.weight.dtype == torch.float16:
            w = module.weight.float()
            init_fn(w)
            module.weight.data.copy_(w.half())
        else:
            init_fn(module.weight)


class ScaledEmbedding(nn.Embedding):
    """Embedding layer with optional scaled learning rate."""
    def __init__(self, num_embeddings, embedding_dim, lr=None):
        super().__init__(num_embeddings, embedding_dim)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    logits: torch.Tensor
    mask: torch.Tensor


def filter_kwargs(kwargs: dict) -> dict:
    exclude_keys = {
        "past_context", "custom", "kv_repeat", "memory_efficient",
        "attention_as_float32", "xpos", "checkpointing",
        "qk_layer_norm", "qk_layer_norm_cross", "two_step_cfg",
        "attention_dropout"
    }
    return {k: v for k, v in kwargs.items() if k not in exclude_keys}

class LMModel(nn.Module):
    """
    Transformer-based language model that interleaves multiple streams of codes
    and handles single-step classifier-free guidance (CFG), without streaming or two-step CFG.
    """
    def __init__(
        self,
        pattern_provider: CodebooksPatternProvider,
        condition_provider: ConditioningProvider,
        fuser: ConditionFuser,
        n_q: int = 8,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = 'layer_norm',
        norm_first: bool = False,
        emb_lr: tp.Optional[float] = None,
        bias_proj: bool = True,
        weight_init: tp.Optional[str] = None,
        depthwise_init: tp.Optional[str] = None,
        zero_bias_init: bool = False,
        cfg_dropout: float = 0.0,
        cfg_coef: float = 1.0,
        attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {},
        use_generator: tp.Optional[bool] = False,
        **kwargs
    ):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)

        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        self.n_q = n_q
        self.dim = dim
        embed_dim = self.card + 1  # Extra for special token
        self.use_generator = use_generator

        self.pattern_provider = pattern_provider

        # Embeddings
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])

        # Transformer
        self.transformer = Transformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            norm_first=norm_first,
            use_generator=use_generator,
            **filter_kwargs(kwargs)
        )
        self.out_norm = create_norm_fn(norm, dim) if norm_first else None

        # Output heads
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])

        # Init
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module] = None  # optional usage

        # print_params(self)

    def init_qkv(self):
        self.transformer.init_qkv()

    def _init_weights(self, weight_init, depthwise_init, zero_bias_init):
        if weight_init is None:
            return
        for emb_layer in self.emb:
            init_layer(emb_layer, weight_init, None, zero_bias_init)
        for idx, t_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            t_layer.apply(
                partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            )
        for linear in self.linears:
            init_layer(linear, weight_init, None, zero_bias_init)

    @property
    def special_token_id(self):
        return self.card

    @property
    def num_codebooks(self):
        return self.n_q

    def forward(
        self,
        sequence: torch.Tensor,
        conditions: tp.List[ConditioningAttributes] = None,
        condition_tensors: tp.Optional[ConditionTensors] = None
    ) -> torch.Tensor:
        # sequence [B, K, S]
        B, K, S = sequence.shape
        x = sum(self.emb[i](sequence[:, i]) for i in range(K))

        # If no precomputed condition_tensors, encode them
        if condition_tensors is None and conditions is not None:
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            condition_tensors = self.condition_provider(tokenized)

        if condition_tensors is not None:
            x, x_cross = self.fuser(x, condition_tensors)
        else:
            x_cross = None
        out = generator_switch(self.transformer(x, cross_src=x_cross),
                               use_generator=self.use_generator)
        if self.out_norm:
            out = self.out_norm(out)

        logits = torch.stack([self.linears[i](out) for i in range(K)], dim=-2)
        if self.fuser.fuse2cond['prepend']:
            logits = logits[:, :, -S:]
        return logits

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        cfg_conditions: CFGConditions,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None
    ) -> torch.Tensor:
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp

        # Single-pass CFG: unconditional + conditional in same forward
        if cfg_conditions:
            sequence = torch.cat([sequence, sequence], dim=0)  # double batch
        all_logits = model(sequence, [], cfg_conditions)
        if cfg_conditions:
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
        else:
            logits = all_logits

        # logits -> [B, K, card, T], focus on last time step
        logits = logits.permute(0, 1, 3, 2)[..., -1]
        if use_sampling and temp > 0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_tok = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_tok = utils.sample_top_k(probs, k=top_k)
            else:
                next_tok = utils.multinomial(probs, num_samples=1)
        else:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tok

    @torch.no_grad()
    def generate(
        self,
        prompt: tp.Optional[torch.Tensor] = None,
        conditions: tp.List[ConditioningAttributes] = [],
        num_samples: tp.Optional[int] = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        remove_prompts: bool = False,
        check: bool = False,
        callback: tp.Optional[tp.Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        # Infer batch size
        if num_samples is None:
            if prompt is not None:
                num_samples = prompt.shape[0]
            elif conditions:
                num_samples = len(conditions)
            else:
                num_samples = 1

        first_param = next(iter(self.parameters()))
        device = first_param.device
        if prompt is None:
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1

        gen_codes = torch.full((B, K, max_gen_len), unknown_token, device=device, dtype=torch.long)
        gen_codes[..., :T] = prompt

        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        start_seq_idx = pattern.get_first_step_with_timesteps(T)
        if start_seq_idx is None:
            return gen_codes[..., :T]

        # Single-step CFG: merge conditions + null_conditions
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            merged_conds = conditions + null_conditions
            tokenized = self.condition_provider.tokenize(merged_conds)
            cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        prev = 0
        for step in range(start_seq_idx, gen_sequence.shape[-1]):
            curr_seq = gen_sequence[..., prev:step]
            next_tok = self._sample_next_token(
                curr_seq,
                cfg_conditions,
                use_sampling,
                temp,
                top_k,
                top_p,
                cfg_coef=cfg_coef
            )
            valid_mask = mask[..., step:step+1].expand(B, -1)
            next_tok[~valid_mask] = self.special_token_id
            unknown_slice = (gen_sequence[..., step:step+1] == unknown_token)
            gen_sequence[..., step:step+1] = torch.where(
                unknown_slice, next_tok, gen_sequence[..., step:step+1]
            )
            prev = step
            if callback:
                callback(step - start_seq_idx + 1, gen_sequence.shape[-1] - start_seq_idx)

        out_codes, idxs, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
        out_codes = out_codes[..., (T if remove_prompts else 0):max_gen_len]
        return out_codes
