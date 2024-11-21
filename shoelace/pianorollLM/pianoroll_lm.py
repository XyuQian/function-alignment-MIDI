import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200 + 1):
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
    def __init__(self, n_words=512, embedding_dim=1024, n_codebooks=4, n_layers=4):
        super(BabyLLM, self).__init__()
        self.in_layer = nn.Embedding(n_words + 1, embedding_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, n_codebooks - 1, 1), requires_grad=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=n_layers)
        self.output_layer = nn.Linear(embedding_dim, n_words, bias=False)
        self.sos = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.mask_token = n_words
        self.n_codebooks = n_codebooks

    def forward(self, tgt, memory):
        tgt = self.in_layer(tgt) + self.pos_emb
        sos = self.sos.repeat(len(tgt), 1, 1)
        tgt = torch.concat([sos, tgt], 1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(self.n_codebooks).to(tgt.device)
        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=attn_mask,
                                      tgt_is_causal=True)
        out = self.output_layer(decoder_output)
        return out

    def inference(self, memory, refine_fn, activation, top_k=10, temperature=1.):
        sos = self.sos.repeat(len(memory), 1, 1)
        tgt = sos
        decoded_sequence = None
        sample_id = 3
        for i in range(self.n_codebooks):
            attn_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt[0])).to(tgt.device)
            decoder_output = self.decoder(tgt, memory,
                                          tgt_mask=attn_mask,
                                          tgt_is_causal=True)
            logits = self.output_layer(decoder_output[:, -1:])

            temp_val = temperature
            top_k_val = top_k
            logits = logits / temp_val

            top_k_logits, top_k_indices = torch.topk(logits, k=top_k_val, dim=-1)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            # print(i, top_k_probs.shape, top_k_indices[0])
            #print(i, top_k_probs[0])
            idx = torch.arange(top_k).to(top_k_probs.device) + 1
            print(i, "is in", idx[top_k_indices[sample_id, 0] == activation[sample_id, i]], top_k_probs[sample_id, 0][top_k_indices[sample_id, 0] == activation[sample_id, i]])
            print(i, "top1", top_k_probs[sample_id, 0, 0].item(), top_k_indices[sample_id, 0, 0].item())
            top_k_indices = top_k_indices.flatten(0, 1)
            top_k_probs = top_k_probs.flatten(0, 1)
            next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, num_samples=1))
            next_token = next_token.view(-1, 1)
            # next_token = top_k_indices[:, :, 0]

            # print("select", next_token[0], activation[0, i])
            # if i < 12:
            #     next_token = activation[:, i: i + 1]

            decoded_sequence = torch.concat([decoded_sequence, next_token],
                                            -1) if decoded_sequence is not None else next_token
            if i < self.n_codebooks - 1:
                next_embedding = self.in_layer(next_token) + self.pos_emb[:, i]
                tgt = torch.concat([tgt, next_embedding], 1)

            # if i == 7:
            #
            #     decoded_sequence = activation
            #
            #     if i < self.n_codebooks - 1:
            #         tgt = self.in_layer(decoded_sequence)
            #         tgt = tgt + self.pos_emb[:, :i + 1]
            #         tgt = torch.concat([sos, tgt], 1)

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

    def forward(self, x, padding=True):
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
            sos = self.sos.repeat(len(out), 1, 1)
            out = torch.concat([sos, out], 1)
        return out


class PianoRollLM(nn.Module):
    def __init__(self, embedding_dim=1024, num_heads=8, num_layers=8):
        super(PianoRollLM, self).__init__()
        self.input_embedding = InputEmbedding(n_words=512,
                                              n_codebooks=[4, 4, 4],
                                              embedding_dim=[512, 256, 256])
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer,
                                                         num_layers=num_layers)
        self.baby_llm = BabyLLM(n_words=512,
                                embedding_dim=1024,
                                n_codebooks=12,
                                n_layers=4)
        self.n_pr_words = 512

    def set_config(self, device):
        self.positional_encoding.set_config(device)

    def forward(self, x):
        input_x = x[:, :-1]
        target = x
        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.positional_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x.shape[1]).to(embed_x.device)
        src_padding_mask = torch.where(target[:, :, 0] == self.n_pr_words, float('-inf'), 0)
        decoder_output = self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)
        memory = decoder_output.flatten(0, 1)[:, None]

        target = target.flatten(0, 1)
        outputs = self.baby_llm(tgt=target[:, :-1],
                                memory=memory)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.n_pr_words)
        tf_loss = loss_fn(outputs.flatten(0, 1), target.long().flatten())
        return tf_loss

    def inference(self, x, refine_fn, activation, max_len=100, top_k=10, temperature=1.):
        embed_x = self.input_embedding(x)
        decoded_sequence = []
        assert x.shape[1] < max_len
        print(x.shape)

        for i, _ in tqdm(enumerate(range(max_len - x.shape[1])), total=max_len - x.shape[1],
                         desc=f"inference"):
            embedding = self.positional_encoding(embed_x)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(embedding.shape[1]).to(embedding.device)
            decoder_output = self.transformer_decoder(embedding,
                                                      is_causal=True,
                                                      mask=attn_mask)
            decoder_output = decoder_output[:, -1:]
            next_token = self.baby_llm.inference(memory=decoder_output,
                                                 refine_fn=refine_fn,
                                                 activation=activation[:, i],
                                                 temperature=temperature,
                                                 top_k=top_k)
            # next_token = next_token[:, None]
            next_token, activation = refine_fn(next_token[:, None], activation=activation)
            decoded_sequence.append(next_token)
            embed_x = torch.concat([embed_x,
                                    self.input_embedding(next_token, padding=False)], 1)
            # if i == 5:
            #     break
        return torch.concat(decoded_sequence, 1)

    def save_weights(self, model_path):
        torch.save(self.state_dict(), model_path)
