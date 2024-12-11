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
    def __init__(self, d_model, max_len=1000, multi_factor=16, long_first=True):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        long_pe = F.interpolate(pe.transpose(1, 2), size=(int(max_len * multi_factor),)).transpose(1, 2)


        self.pos = {"a_pe": long_pe if long_first else pe,
                    "b_pe": pe if long_first else long_pe}

    def set_config(self, device):
        for key in self.pos:
            self.pos[key] = self.pos[key].to(device)

    def forward(self, x_a, x_a_idx, x_b, x_b_idx):
        return x_a + self.pos["a_pe"][:, x_a_idx:x_a_idx + x_a.shape[1], :], \
               x_b + self.pos["b_pe"][:, x_b_idx:x_b_idx + x_b.shape[1], :]


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
        # self.k_linear = nn.Linear(in_dim, embed_dim, bias=False)
        # self.v_linear = nn.Linear(in_dim, embed_dim, bias=False)
        # self.pos_linear = nn.Linear(in_dim, embed_dim, bias=False)
        self.prompt = nn.Parameter(torch.randn(1, 1, in_dim), requires_grad=True)

        self.pos_linear = nn.Sequential(
            nn.Linear(in_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, embed_dim, bias=False))

        self.pos_encoding = PositionalEncoding(multi_factor=multi_factor,
                                               long_first=long_first,
                                               d_model=in_dim)

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

    def forward(self, q, kv_x, q_start_idx=0, kv_start_idx=0, attn_mask=None):
        q_len = q.shape[2]
        nq_heads = q.shape[1]
        q = q.transpose(1, 2).view(len(q), q_len, -1)
        zero_query = torch.zeros([len(q), q_len, kv_x.shape[-1]]).to(q.device)
        query_pos, kv_x = self.pos_encoding(x_a=zero_query,
                                            x_a_idx=q_start_idx,
                                            x_b=kv_x,
                                            x_b_idx=kv_start_idx)
        q = q + self.pos_linear(query_pos)
        q = q.view(len(q), q_len, nq_heads, -1).transpose(1, 2)
        kv_x = torch.concat([self.prompt.repeat(len(kv_x), 1, 1), kv_x], 1)
        condition_output = self.compute_attention(q=q,
                                                  key=self.k_linear(kv_x),
                                                  value=self.v_linear(kv_x),
                                                  attn_mask=attn_mask)
        return condition_output * self.gates
