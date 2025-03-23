import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange


def get_pe(d_model, max_len=10000):
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def init_weights_A(layer: nn.Module):
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))


def init_weights_B(layer: nn.Module):
    nn.init.constant_(layer.weight, 0)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 1)


def create_low_rank_mlp(in_dim: int, low_rank_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(in_dim, low_rank_dim, bias=False),
        nn.Linear(low_rank_dim, out_dim, bias=False)
    )

class PositionalEncoding(nn.Module):
    def __init__(self, low_rank_dim: int, d_model: int, n_indices: int, max_len: int = 15000):
        super().__init__()
        
        self.n_indices = n_indices

        self.pos_linear = nn.ModuleList([create_low_rank_mlp(d_model, low_rank_dim, d_model) for _ in range(n_indices)])
        

    def forward(self, pe, index):
        inputs = [pe[index[..., i]] for i in range(self.n_indices)]
        return sum([self.pos_linear[i](x) for i, x in enumerate(inputs)])


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, 
            num_heads: int, 
            tasks: list,
            low_rank_dim: int = 64, 
            n_in_indices: int = 1,
            n_out_indices: int = 1,
            n_prompts: int = 5,
            dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.embed_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self.k_linear = create_low_rank_mlp(in_dim, low_rank_dim, out_dim)
        self.v_linear = create_low_rank_mlp(in_dim, low_rank_dim, out_dim)
        # self.q_pos_linear = PositionalEncoding(low_rank_dim=low_rank_dim,
        #                                         d_model=out_dim,
        #                                         n_indices=n_out_indices)
        # self.k_pos_linear = PositionalEncoding(low_rank_dim=low_rank_dim,
        #                                         d_model=out_dim,
        #                                         n_indices=n_in_indices)

        self.k_pos_linear =  create_low_rank_mlp(out_dim, low_rank_dim, out_dim)
        self.q_pos_linear =  create_low_rank_mlp(out_dim, low_rank_dim, out_dim)



        self.prompt = nn.Parameter(torch.randn(len(tasks), n_prompts, in_dim), requires_grad=True)
        self.gates = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.tasks = {
            task : i for i, task in enumerate(tasks)
        }

    def forward(self, pe, hidden_a, hidden_b, indices_a, indices_b, tasks, attn_mask):
        vanilla_attn_output = hidden_a["attn_output"]
        
        prompt = torch.stack([self.prompt[self.tasks[t]] for t in tasks], 0)

        q = hidden_a["q"]
        
        if hidden_b is None:
            
            kv_x = prompt
            key_pos = torch.zeros([len(kv_x), kv_x.shape[1], self.embed_dim]).to(kv_x.device)
            print("none", key_pos.shape)
            
        else:
            
            kv_x = hidden_b["query"]
            key_pos = F.pad(pe[indices_b], (0, 0, prompt.shape[1], 0), "constant", 0)
            kv_x = torch.concat([prompt, kv_x], 1)

        
        key = self.k_linear(kv_x)  + self.k_pos_linear(key_pos)
        value = self.v_linear(kv_x)
        
        q_pos = self.q_pos_linear(pe[indices_a])
        q = q + rearrange(q_pos, "b t (h d) -> b h t d", h=self.num_heads)
        attn_output = self.compute_attention(q, key, value, attn_mask)

        

        return attn_output * self.gates + vanilla_attn_output

    def compute_attention(self, q, key, value, attn_mask):
        batch_size, kv_len = key.shape[0], key.shape[1]
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask.dim() == 2 else attn_mask.unsqueeze(1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return attn_output


class SholaceParam(nn.Module):
    def __init__(self, n_layers, out_dim, **kwargs):
        super().__init__()
        self.register_buffer("pe", get_pe(out_dim))
        self.cross_attn = nn.ModuleList([
            LowRankMultiheadAttention(out_dim=out_dim, **kwargs)
            for _ in range(n_layers)])
        

    def forward(self, layer_idx, **kwargs):
        return self.cross_attn[layer_idx](pe=self.pe, **kwargs)
