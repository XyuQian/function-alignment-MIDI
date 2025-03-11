import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def init_weights_A(layer: nn.Module):
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))


def init_weights_B(layer: nn.Module):
    nn.init.constant_(layer.weight, 0)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4097):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("r_pos", pe.unsqueeze(0))

    def forward(self, x_len: int, index=None):
        if index is None:
            return self.r_pos[:, :x_len, :]
        return self.r_pos.squeeze(0)[index[..., 0]]


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, low_rank_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.embed_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self.k_linear = self._create_low_rank_mlp(in_dim, low_rank_dim, out_dim)
        self.v_linear = self._create_low_rank_mlp(in_dim, low_rank_dim, out_dim)
        self.q_pos_linear = self._create_low_rank_mlp(out_dim, low_rank_dim, out_dim)
        self.k_pos_linear = self._create_low_rank_mlp(out_dim, low_rank_dim, out_dim)

        self.prompt = nn.Parameter(torch.randn(1, 1, in_dim), requires_grad=True)
        self.gate = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.pos_encoding = PositionalEncoding(d_model=out_dim)

    @staticmethod
    def _create_low_rank_mlp(in_dim: int, low_rank_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, out_dim, bias=False)
        )

    def forward(self, hidden_a, hidden_b, attn_mask):
        vanilla_attn_output = hidden_a["attn_output"]

        q = hidden_a["q"]

        kv_x = hidden_b["query"]


        

        q_len, kv_len = q.shape[2], kv_x.shape[1]

        prompt = torch.repeat(self.prompt, [len(kv_x), 1, 1])
        kv_x = torch.concat([self.prompt, kv_x], 1)

        print(q.shape, kv_x.shape, attn_mask.shape)
        key = self.k_linear(kv_x)
        value = self.v_linear(kv_x)

        attn_output = self.compute_attention(q, key, value, attn_mask)


        # q = q.transpose(1, 2).reshape(q.shape[0], q_len, -1) + self.q_pos_linear(self.pos_encoding(q_len, pos_a))
        # kv_x = torch.cat([self.prompt.repeat(len(kv_x), 1, 1), kv_x], dim=1)
        # kv_pos = F.pad(self.pos_encoding(kv_len, pos_b), (0, 0, 1, 0))

        # key = self.k_linear(kv_x) + self.k_pos_linear(kv_pos)
        # value = self.v_linear(kv_x)
        return attn_output * self.gate + vanilla_attn_output

    def compute_attention(self, q, key, value, attn_mask):
        batch_size, kv_len = key.shape[0], key.shape[1]
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask.dim() == 2 else attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return attn_output


class SholaceParam(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, num_heads,
                 low_rank_dim):
        super().__init__()
        self.cross_attn = nn.ModuleList([
            LowRankMultiheadAttention(in_dim, out_dim, num_heads, low_rank_dim)
            for _ in range(n_layers)])

    def forward(self, hidden_a, hidden_b, layer_idx, attn_mask=None):
        return self.cross_attn[layer_idx](hidden_a, hidden_b, attn_mask)
