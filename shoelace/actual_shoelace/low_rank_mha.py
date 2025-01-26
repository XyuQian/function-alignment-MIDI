import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def init_weights_A(layer):
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))


def init_weights_B(layer):
    nn.init.constant_(layer.weight, 0)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096 + 1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.r_pos = {"relative": pe}

    def set_config(self, device):
        self.r_pos["relative"] = self.r_pos["relative"].to(device)

    def forward(self, x_len, index=None):
        pe = self.r_pos["relative"]
        if index is None:
            return pe[:, :x_len, :]
        else:
            pe = pe.squeeze(0)
            return pe[index[..., 0]]
            # pos = []

            # n_pos = index.shape[-1]
            # for k in range(n_pos):
            #     pos.append(pe[..., k::n_pos][index[..., k]])
            #
            # return x + torch.stack(pos, -1).flatten(2, 3)


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, in_dim, embed_dim,
                 num_heads, multi_factor,
                 long_first, low_rank_dim=64, dropout=0.0):
        super(LowRankMultiheadAttention, self).__init__()

        self.dropout = dropout

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.k_linear = nn.Sequential(
            nn.Linear(in_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, embed_dim, bias=False))
        self.v_linear = nn.Sequential(
            nn.Linear(in_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, embed_dim, bias=False))

        self.prompt = nn.Parameter(torch.randn(1, 1, in_dim), requires_grad=True)

        self.q_pos_linear = nn.Sequential(
            nn.Linear(embed_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, embed_dim, bias=False))

        self.k_pos_linear = nn.Sequential(
            nn.Linear(embed_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, embed_dim, bias=False))

        self.pos_encoding = PositionalEncoding(d_model=embed_dim)

        self.gates = nn.Parameter(torch.zeros([1]), requires_grad=True)
        self.attn_dropout = nn.Dropout(dropout)

    def set_config(self, device):
        self.pos_encoding.set_config(device)
        pass

    def compute_attention(self, q, key, value, attn_mask):

        num_heads = self.num_heads
        head_dim = self.head_dim

        batch_size = len(key)
        kv_len = key.shape[1]
        b_sz = len(key)

        key = key.view(b_sz, kv_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(b_sz, kv_len, num_heads, head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, key.transpose(-2, -1)) / (head_dim ** 0.5)
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

    def forward(self, q, kv_x, pos_a=None, pos_b=None, mask=None):
        q_len = q.shape[2]
        kv_len = kv_x.shape[1]
        nq_heads = q.shape[1]
        q = q.transpose(1, 2).view(len(q), q_len, -1)
        query_pos = self.pos_encoding(x_len=q_len,
                                      index=pos_a)
        q = q + self.q_pos_linear(query_pos)
        q = q.view(len(q), q_len, nq_heads, -1).transpose(1, 2)

        key_pos = self.pos_encoding(x_len=kv_len, index=pos_b)

        key_pos = F.pad(key_pos, (0, 0, 1, 0), "constant", 0)

        kv = torch.concat([self.prompt.repeat(len(kv_x), 1, 1), kv_x], 1)

        condition_output = self.compute_attention(q=q,
                                                  key=self.k_linear(kv) + self.k_pos_linear(key_pos),
                                                  # key=self.k_linear(kv),
                                                  value=self.v_linear(kv),
                                                  attn_mask=mask)
        return condition_output * self.gates
