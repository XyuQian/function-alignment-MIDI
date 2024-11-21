import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from shoelace.pianorollLM.pianoroll_lm_with_baby import PositionalEncoding
from shoelace.utils.network_utils import freeze


def init_weights_A(layer):
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))


def init_weights_B(layer):
    nn.init.constant_(layer.weight, 0)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 1)


class LoRALayer(nn.Module):
    def __init__(self, in_dim, embed_dim, frozen_weight, alpha, low_rank_dim=16, frozen_bias=None):
        super(LoRALayer, self).__init__()
        self.a = nn.Linear(in_dim, low_rank_dim, bias=False)
        self.b = nn.Linear(low_rank_dim, embed_dim, bias=False)
        init_weights_A(self.a)
        init_weights_B(self.b)
        self.frozen_weights = {
            "w": frozen_weight,
            "b": frozen_bias}

        self.alpha = alpha

    def set_config(self, device):
        frozen_weights = self.frozen_weights
        self.frozen_weights = {
            "w": frozen_weights["w"].to(device),
            "b": frozen_weights["b"].to(device) if frozen_weights["b"] is not None else None}

    def forward(self, x):
        weight = self.frozen_weights["w"] + (self.b.weight @ self.a.weight) * self.alpha
        return F.linear(x, weight, bias=self.frozen_weights["b"])


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, in_dim, embed_dim,
                 num_heads, dropout=0.0):
        super(LowRankMultiheadAttention, self).__init__()

        self.dropout = dropout

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.k_linear = nn.Linear(in_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(in_dim, embed_dim, bias=False)
        self.prompt = nn.Parameter(torch.randn(1, 1, in_dim), requires_grad=True)

        self.pos_linear = nn.Linear(in_dim, embed_dim, bias=False)

        self.pos = PositionalEncoding(d_model=in_dim)

        self.gates = nn.Parameter(torch.zeros([1]), requires_grad=True)
        self.attn_dropout = nn.Dropout(dropout)

    def set_config(self, device):
        self.pos.set_config(device)
        pass

    def compute_attention(self, query, key, value, attn_mask):

        num_heads = self.num_heads
        head_dim = self.head_dim

        batch_size = len(key)
        t_len = key.shape[1]
        key = key.view(-1, t_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(-1, t_len, num_heads, head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # If shape is (t_q, t_k), broadcast to (batch_size, num_heads, t_q, t_k)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, t_q, t_k)
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return attn_output

    def forward(self, x_a, x_b, debug, mode="a2b", cur_step=None, full_b_step=50, full_a_step=801, attn_mask=None):

        pos_short = self.pos.r_pos["relative"][:, :full_b_step]
        pos_long = F.interpolate(pos_short.transpose(1, 2), size=(full_a_step,)).transpose(1, 2)


        if mode == "a2b":
            pos = self.pos_linear(pos_long)
            pos = pos.view(1, full_a_step, self.num_heads, self.head_dim).transpose(1, 2)
            debug = debug + pos[:, :, : debug.shape[-2]]
            kv = self.pos(x_b)
        else:
            pos = self.pos_linear(pos_short)
            pos = pos.view(1, full_b_step, self.num_heads, self.head_dim).transpose(1, 2)
            debug = debug + pos[:, :, : debug.shape[-2]]
            kv = x_b + pos_long[:, :x_b.shape[1]]


        kv = torch.concat([self.prompt.repeat(len(kv), 1, 1), kv], 1)
        condition_output = self.compute_attention(query=debug,
                                                  key=self.k_linear(kv),
                                                  value=self.v_linear(kv),
                                                  attn_mask=attn_mask)
        return condition_output * self.gates
