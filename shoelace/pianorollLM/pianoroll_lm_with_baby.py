import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import numpy as np
from shoelace.pianoroll_vq.base_vq import LinearBlock


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
    def __init__(self, d_model, max_len=1000 + 1):
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


class ContextPosEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000 + 1):
        super(ContextPosEncoding, self).__init__()
        self.pos_encoding = nn.Sequential(
            LinearBlock(1, 256),
            LinearBlock(256, 2048),
            LinearBlock(2048, d_model)
        )
        self.re_pos = PositionalEncoding(d_model=d_model, max_len=max_len)

    def set_config(self, device):
        self.re_pos.set_config(device)

    def forward(self, x, ctx_pos):

        return self.re_pos(x) + self.pos_encoding(ctx_pos.unsqueeze(-1))


class BabyLLM(nn.Module):
    def __init__(self, n_words=512, memory_dim=512, embedding_dim=512, n_codebooks=10, n_layers=3):
        super(BabyLLM, self).__init__()
        self.in_layer = nn.Embedding(n_words + 1, embedding_dim)

        self.pos_emb = nn.Parameter(torch.randn(1, n_codebooks - 1, 1), requires_grad=True)
        # self.mem_linear = nn.Linear(memory_dim, embedding_dim, bias=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=n_layers)
        self.output_layer = nn.Linear(embedding_dim, n_words, bias=False)
        self.sos = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.mask_token = n_words
        self.n_codebooks = n_codebooks

    def forward(self, tgt, memory, melody=None):
        tgt = self.in_layer(tgt) + self.pos_emb
        sos = self.sos.repeat(len(tgt), 1, 1)
        tgt = torch.concat([sos, tgt], 1)
        # memory = self.mem_linear(memory)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(self.n_codebooks).to(tgt.device)
        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=attn_mask,
                                      tgt_is_causal=True)
        out = self.output_layer(decoder_output)
        return out

    def inference(self, memory, top_k=10, temperature=1.):
        sos = self.sos.repeat(len(memory), 1, 1)
        tgt = sos
        decoded_sequence = None
        for i in range(self.n_codebooks):
            attn_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt[0])).to(tgt.device)
            decoder_output = self.decoder(tgt, memory,
                                          tgt_mask=attn_mask,
                                          tgt_is_causal=True)
            logits = self.output_layer(decoder_output[:, -1:])
            next_token = sample(logits, top_k_val=top_k, temperature=temperature)

            decoded_sequence = torch.concat([decoded_sequence, next_token],
                                            -1) if decoded_sequence is not None else next_token
            if i < self.n_codebooks - 1:
                next_embedding = self.in_layer(next_token) + self.pos_emb[:, i]
                tgt = torch.concat([tgt, next_embedding], 1)
        return decoded_sequence


class InputEmbedding(nn.Module):
    def __init__(self, n_words=512,
                 n_codebooks=None, embedding_dim=None):
        super(InputEmbedding, self).__init__()
        if n_codebooks is None:
            n_codebooks = [4, 4, 4]
            embedding_dim = [512, 256, 256]

        self.layers = nn.ModuleList(
            nn.ModuleList(
                nn.Embedding(n_words + 1, embedding_dim[i]) for _ in range(n_codebooks[i])
            ) for i in range(len(n_codebooks))
        )
        self.sos = nn.Parameter(torch.randn(1, 1, sum(embedding_dim)), requires_grad=True)
        self.mask_token = n_words
        self.n_codebooks = n_codebooks
        self.n_words = n_words

    def forward(self, x, padding=True, n=None):
        if padding and x is None:
            return self.sos.repeat(n, 1, 1)

        n_codebooks = self.n_codebooks
        n = 0
        outs = []
        for i in range(len(n_codebooks)):
            ind = x[:, :, n:n + n_codebooks[i]]
            n += n_codebooks[i]
            emb = 0.
            for j in range(n_codebooks[i]):
                emb += self.layers[i][j](ind[:, :, j])
            outs.append(emb)
        out = torch.concat(outs, -1)
        if padding:
            sos = self.sos.repeat(len(x), 1, 1)
            out = torch.concat([sos, out], 1)
        return out


class PianoRollLM(nn.Module):
    def __init__(self, embedding_dim=1024, num_heads=8, num_layers=8):
        super(PianoRollLM, self).__init__()
        self.input_embedding = InputEmbedding(n_words=512,
                                              n_codebooks=[4, 4, 4],
                                              embedding_dim=[512, 256, 256])
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, max_len=1000)

        decoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                batch_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.baby_llm = BabyLLM(n_words=512,
                                memory_dim=1024,
                                embedding_dim=1024,
                                n_codebooks=12,
                                n_layers=4)
        self.n_pr_words = 512
        self.embedding_dim = 1024
        self.n_first_layers = 2
        self.hn_layers = (num_layers - self.n_first_layers * 2) // 2
        self.n_decoder_layers = num_layers

    def set_config(self, device):
        self.pos_encoding.set_config(device)

    def set_training(self, device):
        pass

    def prepare_for_lora(self, mode):
        self.transformer_decoder.prepare_for_lora(mode)

    def res_decode(self, embed_x_with_pos, x, ctx_pos):
        mask_schedule = [.6, .8, .9, .95]

        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x_with_pos.shape[1]).to(
            embed_x_with_pos.device)
        src_padding_mask = torch.where(x[:, :, 0] == 512, float('-inf'), 0) if x is not None else None
        transformer_decoder = self.transformer_decoder
        param_dict = transformer_decoder.encode2roll(
            src=embed_x_with_pos,
            mask=attn_mask,
            src_key_padding_mask=src_padding_mask,
            is_causal=True,
            target=None,
        )
        n_layers = len(transformer_decoder.layers)
        x_res = x
        if x is None:
            x_res = torch.zeros_like(embed_x_with_pos[:, :, :12]).long()
        mask_indices = []

        pos_decay = torch.arange(embed_x_with_pos.shape[1]).to(embed_x_with_pos.device)
        pos_decay = 2. / (torch.exp(-1 * pos_decay / 20.) + 1) - 1  # 2/(e^(-x/100) + 1)  - 1
        for i in range(n_layers // 2):
            output = param_dict["output"]
            mask_r = torch.rand_like(x_res[0, :, 0].float())
            mask_idx = mask_r * pos_decay < mask_schedule[i]
            output = output[:, mask_idx]
            pos_decay = pos_decay[mask_idx]
            mask_indices = [mask_idx] + mask_indices
            x_res = x_res[:, mask_idx]
            src_padding_mask = torch.where(x_res[:, :, 0] == 512, float('-inf'), 0) if x is not None else None
            attn_mask = nn.Transformer.generate_square_subsequent_mask(x_res.shape[1]).to(
                x_res.device)

            param_dict["src_key_padding_mask_for_layers"] = src_padding_mask
            param_dict["mask"] = attn_mask
            param_dict["output"] = output
            param_dict = transformer_decoder.roll(param_dict, i)

        mask = None
        for i in range(1, len(mask_indices)):
            mask = mask_indices[i].clone()
            mask[mask_indices[i]] = mask_indices[i - 1]
            mask_indices[i] = mask
        output = self.pos_encoding(torch.zeros_like(embed_x_with_pos), ctx_pos)
        src_padding_mask = torch.where(x[:, :, 0] == 512, float('-inf'), 0) if x is not None else None
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x_with_pos.shape[1]).to(
            embed_x_with_pos.device)
        output[:, mask] = param_dict["output"]
        param_dict["src_key_padding_mask_for_layers"] = src_padding_mask
        param_dict["mask"] = attn_mask
        param_dict["output"] = output

        for i in range(n_layers // 2, n_layers):
            param_dict = transformer_decoder.roll(param_dict, i)

        decoder_output = transformer_decoder.roll2end(param_dict)
        return decoder_output


    def yield_forward(self, midi_seq):
        input_x = midi_seq[:, :-1]

        baby_target = midi_seq

        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x.shape[1]).to(embed_x.device)
        src_padding_mask = torch.where(midi_seq[:, :, 0] == 512, float('-inf'), 0)

        decoder_output = yield from self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)
        memory = decoder_output.flatten(0, 1)[:, None]
        baby_target = baby_target.flatten(0, 1)

        outputs = self.baby_llm(tgt=baby_target[:, :-1],
                                memory=memory)
        return outputs

    def forward(self, x):
        input_x = x[:, :-1]

        baby_target = target = x
        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        src_padding_mask = torch.where(x[:, :, 0] == 512, float('-inf'), 0) if x is not None else None

        decoder_output = self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)

        # decoder_output = self.res_decode(embed_x_with_pos, x=x, ctx_pos=ctx_pos)
        memory = decoder_output.flatten(0, 1)[:, None]

        baby_target = baby_target.flatten(0, 1)
        outputs = self.baby_llm(tgt=baby_target[:, :-1],
                                memory=memory)
        loss_fn = nn.CrossEntropyLoss(ignore_index=512)
        acc_loss = loss_fn(outputs.flatten(0, 1), target.flatten())
        return acc_loss


    def sample_next_tokens(self, param_dict, top_k=32, temperature=1.):
        decoder_output = self.transformer_decoder.roll2end(param_dict)
        memory = decoder_output[:, -1:]
        next_token = self.baby_llm.inference(memory=memory,

                                             temperature=temperature,
                                             top_k=top_k)

        return next_token[:, None]

    def inference(self, x, ctx_pos, max_len=100, top_k=10, temperature=1., mask_ratio=0.):
        embed_x = self.input_embedding(x)
        decoded_sequence = []
        print(x.shape, max_len)
        assert x.shape[1] < max_len
        prompt_len = x.shape[1]

        for i, _ in tqdm(enumerate(range(max_len - prompt_len)), total=max_len - prompt_len,
                         desc=f"inference"):
            # print(embed_x.shape, ctx_pos.shape)
            embedding = self.pos_encoding(embed_x, ctx_pos[:, :embed_x.shape[1]])
            attn_mask = nn.Transformer.generate_square_subsequent_mask(embedding.shape[1]).to(embedding.device)
            decoder_output = self.transformer_decoder(embedding,
                                                      is_causal=True,
                                                      mask=attn_mask)
            # decoder_output = self.res_decode(embed_x_with_pos=embed_x,
            #                                  x=None,
            #                                  ctx_pos=ctx_pos[:, :embed_x.shape[1]])

            decoder_output = decoder_output[:, -1:]
            next_token = self.baby_llm.inference(memory=decoder_output,
                                                 temperature=temperature,
                                                 top_k=top_k)

            next_token = next_token[:, None]
            decoded_sequence.append(next_token)
            embed_x = torch.concat([embed_x,
                                    self.input_embedding(next_token, padding=False)], 1)

        return torch.concat(decoded_sequence, 1)

    def save_weights(self, model_path):
        torch.save(self.state_dict(), model_path)
