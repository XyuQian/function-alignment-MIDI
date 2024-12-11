import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from shoelace.utils.network_utils import print_params

STRIDE = 32
SEQ_LEN = 24


def sample(logits, top_k_val=20, temperature=1.):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k_val, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    top_k_indices = top_k_indices.flatten(0, 1)
    top_k_probs = top_k_probs.flatten(0, 1)
    next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, num_samples=1))
    next_token = next_token.view(-1, 1)
    return next_token


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400 + 1):
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

    def forward(self, x):
        pe = self.r_pos["relative"]
        x_len = x.shape[1]
        return x + pe[:, :x_len, :]


class BabyLLM(nn.Module):
    def __init__(self, embedding_dim=1024, n_layers=4):
        super(BabyLLM, self).__init__()
        self.in_layer = nn.Embedding(384 + 2 + STRIDE + 1, embedding_dim)

        self.pos_encoding = nn.Parameter(torch.randn(1, SEQ_LEN*2 - 1, 1), requires_grad=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=n_layers)
        self.out_layer = nn.Linear(embedding_dim, 385 + STRIDE, bias=False)
        self.sos = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

    def forward(self, tgt, memory):
        tgt = self.in_layer(tgt) + self.pos_encoding
        sos = self.sos.repeat(len(tgt), 1, 1)
        tgt = torch.concat([sos, tgt], 1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=attn_mask,
                                      tgt_is_causal=True)
        return self.out_layer(decoder_output)

    def inference(self, memory, acc_mask=False, top_k=10, temperature=1.):
        sos = self.sos.repeat(len(memory), 1, 1)
        tgt = sos
        decoded_sequence = None

        for i in range(self.n_codebooks):
            if acc_mask and i // 4 % 2 == 1:
                next_token = torch.zeros_like(decoded_sequence[:, :1]) + 512
            else:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt[0])).to(tgt.device)
                decoder_output = self.decoder(tgt, memory,
                                              tgt_mask=attn_mask,
                                              tgt_is_causal=True)
                logits = self.output_layer(decoder_output[:, -1:])
                next_token = sample(logits, top_k_val=top_k, temperature=temperature)

            decoded_sequence = torch.concat([decoded_sequence, next_token],
                                            -1) if decoded_sequence is not None else next_token
            if i < self.n_codebooks - 1:
                next_embedding = self.in_layer(next_token) + self.pos_encoding[:, i]
                tgt = torch.concat([tgt, next_embedding], 1)
        return decoded_sequence


class InputEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(InputEmbedding, self).__init__()
        self.sos = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.pitch_embedding = nn.Embedding(384 + 2, embedding_dim // STRIDE)

    def forward(self, x, padding=True, n=None):
        if padding and x is None:
            return self.sos.repeat(n, 1, 1)
        out = self.pitch_embedding(x).flatten(2, 3)
        if padding:
            sos = self.sos.repeat(len(out), 1, 1)
            out = torch.concat([sos, out], 1)
        return out


class MelodyLM(nn.Module):
    def __init__(self, embedding_dim=1024, num_heads=8, num_layers=8):
        super(MelodyLM, self).__init__()
        self.input_embed = InputEmbedding(embedding_dim=1024)
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim)

        decoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                batch_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.baby_llm = BabyLLM(embedding_dim=1024,
                                n_layers=4)
        self.n_pr_words = 385
        self.embedding_dim = 1024

    def set_config(self, device):
        self.pos_encoding.set_config(device)

    def set_training(self, device):
        pass

    def prepare_for_lora(self, mode):
        self.transformer_decoder.prepare_for_lora(mode)

    def forward(self, seq, melody):
        x = seq[:, :-1]
        src_padding_mask = torch.where(seq[:, :, 0] == self.n_pr_words, float('-inf'), 0)
        embed_x = self.input_embed(x.long())
        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x.shape[1]).to(embed_x.device)
        decoder_output = self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)
        memory = decoder_output.flatten(0, 1)[:, None]

        baby_target = melody.flatten(0, 1)
        outputs = self.baby_llm(tgt=baby_target.flatten(1, 2)[:, :-1],
                                memory=memory)
        target = torch.stack([baby_target[..., 0] + STRIDE, baby_target[..., 1]], -1)
        target[..., 1][baby_target[..., 1] > STRIDE] = -1
        target[..., 0][baby_target[..., 0] > 384] = -1
        target = target.flatten(1, 2)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        outputs = outputs.flatten(0, 1)
        target = target.flatten()
        r = torch.rand_like(target.float()) < .5
        melody_loss = loss_fn(outputs[r], target[r])
        return melody_loss

    def encode2roll(self, x, melody_mask=None, with_padding=False, n=None):
        target = x
        if x is None:
            embed_x = self.input_embed(None, n=n)
        else:
            input_x = x[:, :-1] if self.training else x
            embed_x = self.input_embed(input_x)

        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x.shape[1]).to(embed_x.device)
        src_padding_mask = torch.where(target[:, :, 0] == self.n_pr_words, float('-inf'), 0) if with_padding else None
        if melody_mask is not None:
            # non_sil = (melody_mask > 0).flatten()
            diff = torch.abs(melody_mask[:, 1:] - melody_mask[:, :-1]) > 0
            diff = torch.concat([torch.ones_like(diff[:, :1]), diff], 1)
            diff = diff.flatten()
            melody_mask = (torch.rand(diff.shape) > .5).long()
            melody_mask[diff > 0] = 1

        return self.transformer_decoder.encode2roll(
            melody_mask=melody_mask,
            target=target,
            src=embed_x_with_pos,
            mask=attn_mask,
            src_key_padding_mask=src_padding_mask,
            is_causal=True
        )

    def roll(self, param_dict, layer_idx, fn):

        return self.transformer_decoder.roll(param_dict=param_dict, layer_idx=layer_idx, fn=fn)

    def roll2end(self, param_dict, with_acc_loss=True):
        decoder_output = self.transformer_decoder.roll2end(param_dict)
        memory = decoder_output.flatten(0, 1)[:, None]

        target = param_dict["target"].flatten(0, 1)

        outputs = self.baby_llm(tgt=target[:, :-1],
                                memory=memory)
        melody_mask = param_dict["melody_mask"]

        outputs = outputs.view(-1, 3, 2, 4, 512)
        melody_outputs = outputs[:, :, 0].flatten(1, 2)
        acc_outputs = outputs[:, :, 1].flatten(1, 2)
        target = target.view(-1, 3, 2, 4)
        melody_target = target[:, :, 0].flatten(1, 2)
        acc_target = target[:, :, 1].flatten(1, 2)

        loss_fn = nn.CrossEntropyLoss(ignore_index=512)
        if melody_mask is None:
            melody_loss = loss_fn(melody_outputs.flatten(0, 1),
                                  melody_target.flatten())
        else:
            melody_loss = loss_fn(melody_outputs[melody_mask > 0].flatten(0, 1),
                                  melody_target[melody_mask > 0].flatten())
        acc_loss = loss_fn(acc_outputs.flatten(0, 1), acc_target.flatten()) if with_acc_loss else 0
        return melody_loss + acc_loss

    def sample_next_tokens(self, param_dict, acc_mask, top_k=32, temperature=1.):
        decoder_output = self.transformer_decoder.roll2end(param_dict)
        memory = decoder_output[:, -1:]
        next_token = self.baby_llm.inference(memory=memory,
                                             acc_mask=acc_mask,
                                             temperature=temperature,
                                             top_k=top_k)

        return next_token[:, None]

    def inference(self, x, refine_fn, activation, max_len=100, top_k=10, temperature=1.):
        embed_x = self.input_embed(x)
        decoded_sequence = []
        print(x.shape, max_len)
        assert x.shape[1] < max_len
        # print(x.shape)
        # melody_sequence = []
        for i, _ in tqdm(enumerate(range(max_len - x.shape[1])), total=max_len - x.shape[1],
                         desc=f"inference"):
            embedding = self.pos_encoding(embed_x)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(embedding.shape[1]).to(embedding.device)
            decoder_output = self.transformer_decoder(embedding,
                                                      is_causal=True,
                                                      mask=attn_mask)
            decoder_output = decoder_output[:, -1:]
            # melody_prob = self.melody_out(decoder_output)
            # melody_token = sample(melody_prob, top_k_val=top_k, temperature=temperature)
            # melody_sequence.append(melody_token[: None])

            next_token = self.baby_llm.inference(memory=decoder_output,
                                                 # melody=melody_token,
                                                 refine_fn=refine_fn,
                                                 activation=activation,
                                                 temperature=temperature,
                                                 top_k=top_k)

            next_token, activation = refine_fn(next_token[:, None], activation=activation)
            decoded_sequence.append(next_token)
            embed_x = torch.concat([embed_x,
                                    self.input_embed(next_token, padding=False)], 1)

        return torch.concat(decoded_sequence, 1)

    def save_weights(self, model_path):
        torch.save(self.state_dict(), model_path)
