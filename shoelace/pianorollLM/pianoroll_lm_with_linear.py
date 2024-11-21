import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from shoelace.pianoroll_vq.base_vq import LinearBlock
from tqdm import tqdm
from shoelace.utils.network_utils import print_params


def sample(logits, top_k_val=20, temperature=1.):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k_val, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    # top_k_indices = top_k_indices.flatten(0, 1)
    # top_k_probs = top_k_probs.flatten(0, 1)
    next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, num_samples=1))
    next_token = next_token.flatten()
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
    def __init__(self, n_words=512, embedding_dim=1024, n_codebooks=10, n_layers=4):
        super(BabyLLM, self).__init__()
        self.in_layer = nn.Embedding(n_words + 1, embedding_dim)

        self.pos_emb = nn.Parameter(torch.randn(1, n_codebooks - 1, 1), requires_grad=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8,
                                                   batch_first=True)
        self.melody_mem = nn.Embedding(128 + 1, embedding_dim)
        self.mem_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=n_layers)
        self.output_layer = nn.Linear(embedding_dim, n_words, bias=False)
        self.sos = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.mask_token = n_words
        self.n_codebooks = n_codebooks

    def forward(self, tgt, memory, melody):
        tgt = self.in_layer(tgt) + self.pos_emb
        sos = self.sos.repeat(len(tgt), 1, 1)
        tgt = torch.concat([sos, tgt], 1)
        memory = self.mem_linear(memory) + self.melody_mem(melody)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(self.n_codebooks).to(tgt.device)
        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=attn_mask,
                                      tgt_is_causal=True)
        out = self.output_layer(decoder_output)
        return out

    def inference(self, memory, melody, refine_fn, activation, top_k=10, temperature=1.):
        sos = self.sos.repeat(len(memory), 1, 1)
        tgt = sos
        decoded_sequence = None
        memory = self.mem_linear(memory) + self.melody_mem(melody)
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
    def __init__(self, n_words=None, embedding_dim=None, has_sos=True):
        super(InputEmbedding, self).__init__()
        if n_words is None:
            n_words = [
                [128, 320],
                [512, 512, 512, 512],
                [512, 512, 512, 512],
                [512, 512, 512, 512]
            ]
            embedding_dim = [
                256, 256, 256, 256
            ]

        self.layers = nn.ModuleList(
            nn.ModuleList(
                nn.Embedding(n_words[i][j] + 1, embedding_dim[i]) for j in range(len(n_words[i]))
            ) for i in range(len(n_words))
        )
        if has_sos:
            self.sos = nn.Parameter(torch.randn(1, 1, sum(embedding_dim)), requires_grad=True)
        self.n_words = n_words
        self.embedding_dim = embedding_dim

    def prepare(self, x, melody, chords):
        outs = [[x[:, :, i * 4 + j] for j in range(4)] for i in range(3)]
        return [[melody, chords]] + outs

    def encode(self, x, padding=True):

        n_words = self.n_words
        # print("--------------------", n_words, len(x), len(x[0]))
        outs = []
        for i in range(len(n_words)):
            emb = 0.
            for j in range(len(n_words[i])):
                emb += self.layers[i][j](x[i][j])
            outs.append(emb)
        out = torch.concat(outs, -1)
        if padding:
            sos = self.sos.repeat(len(out), 1, 1)
            out = torch.concat([sos, out], 1)
        return out


class CondSampler(nn.Module):
    def __init__(self, context_dim, cond_size, out_dim, embedding_dim):
        super(CondSampler, self).__init__()
        self.context_linear = nn.Linear(context_dim, embedding_dim[0], bias=False)
        if cond_size is not None:
            self.cond_linear = InputEmbedding(n_words=cond_size, embedding_dim=embedding_dim[1:], has_sos=False)
        else:
            self.cond_linear = None
        # self.norm = nn.LayerNorm(sum(embedding_dim), eps=1e-5, bias=False)
        self.out_linear = nn.Sequential(
            LinearBlock(sum(embedding_dim), 2048),
            nn.Linear(2048, out_dim))
        self.embedding_dim = embedding_dim

    def forward(self, cond, context):
        context = self.context_linear(context)
        if self.cond_linear is not None:
            cond = self.cond_linear.encode(cond, padding=False)
            out = torch.concat([context, cond], -1)
        else:
            out = context
        return self.out_linear(out)


class OutputEmbedding(nn.Module):
    def __init__(self, n_words, embedding_dim, context_dim):
        super(OutputEmbedding, self).__init__()
        self.melody_sampler = nn.Sequential(
            LinearBlock(context_dim, 2048),
            nn.Linear(2048, n_words[0][0], bias=False))
        self.chords_sampler = CondSampler(context_dim=context_dim, cond_size=[n_words[0][:1]],
                                          out_dim=n_words[0][1], embedding_dim=[context_dim] + embedding_dim[:1])
        self.rvq_sampler = nn.ModuleList(
            [nn.ModuleList(
                [CondSampler(context_dim=context_dim,
                             cond_size=n_words[:i + 1] if j == 0 else n_words[:i + 1] + [n_words[i + 1][:j]],
                             out_dim=n_words[i + 1][j],
                             embedding_dim=[context_dim] + embedding_dim[:i + 1 + int(j > 0)])
                 for j in range(len(n_words[i + 1]))]) for i in range(len(n_words) - 1)]
        )

    def forward(self, x, context):
        melody_pred = self.melody_sampler(context)
        chords_pred = self.chords_sampler(cond=[x[0][:1]], context=context)
        rvq_preds = []
        for i in range(len(x) - 1):
            for j in range(len(x[i + 1])):
                rvq_preds.append(self.rvq_sampler[i][j](
                    cond=x[:i + 1] if j == 0 else x[:i + 1] + [x[i + 1][:j]],
                    context=context
                ))
        rvq_pred = torch.stack(rvq_preds, 2)

        return melody_pred, chords_pred, rvq_pred

    def inference(self, context, refine_fn, activation, temperature, top_k):
        melody_pred = self.melody_sampler(context)
        # print("++++++++++++++++", melody_pred.shape, context.shape)
        melody_token = sample(melody_pred, top_k_val=top_k, temperature=temperature)
        conds = [[melody_token]]

        chords_pred = self.chords_sampler(cond=conds, context=context)
        chords_token = sample(chords_pred, top_k_val=top_k, temperature=temperature)
        conds = [[melody_token, chords_token]]

        for i in range(3):
            for j in range(4):
                rvq_pred = self.rvq_sampler[i][j](
                    cond=conds,
                    context=context
                )
                rvq_token = sample(rvq_pred, top_k_val=top_k, temperature=temperature)
                if j == 0:
                    conds.append([rvq_token])
                else:
                    conds[-1].append(rvq_token)
        rvq_tokens = [torch.stack(t, -1) for t in conds[1:]]
        rvq_tokens = torch.concat(rvq_tokens, -1)

        melody_tokens = conds[0][0]
        chord_tokens = conds[0][1]

        return rvq_tokens, melody_tokens, chord_tokens



class PianoRollLM(nn.Module):
    def __init__(self, embedding_dim=1024, num_heads=16, num_layers=12):
        super(PianoRollLM, self).__init__()
        self.input_embedding = InputEmbedding()
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim)

        decoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                batch_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = OutputEmbedding(context_dim=embedding_dim,
                                           n_words=self.input_embedding.n_words,
                                           embedding_dim=self.input_embedding.embedding_dim)
        # self.baby_llm = BabyLLM(n_words=512,
        #                         embedding_dim=1024,
        #                         n_codebooks=12,
        #                         n_layers=4)
        self.n_pr_words = 512
        self.embedding_dim = 1024

    def set_config(self, device):
        self.positional_encoding.set_config(device)

    def set_training(self, device):
        pass

    def prepare_for_lora(self, mode):
        self.transformer_decoder.prepare_for_lora(mode)

    def forward(self, x, melody, chords):
        input_tokens = self.input_embedding.prepare(x[:, :-1], melody[:, :-1], chords[:, :-1])
        embed_x = self.input_embedding.encode(input_tokens)

        embed_x_with_pos = self.positional_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x.shape[1]).to(embed_x.device)
        src_padding_mask = torch.where(x[:, :, 0] == self.n_pr_words, float('-inf'), 0)
        decoder_output = self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)

        target_tokens = self.input_embedding.prepare(x, melody, chords)
        melody_pred, chords_pred, rvq_pred = self.output_proj(target_tokens, context=decoder_output)
        # print(chords_pred.shape, chords.shape, rvq_pred.shape, x.shape)
        # loss_fn = nn.CrossEntropyLoss(ignore_index=self.n_pr_words)
        # tf_loss = loss_fn(outputs.flatten(0, 1), target.long().flatten())
        loss_fn = nn.CrossEntropyLoss(ignore_index=128)
        melody_loss = loss_fn(melody_pred.flatten(0, 1), melody.long().flatten())

        loss_fn = nn.CrossEntropyLoss(ignore_index=320)
        chords_loss = loss_fn(chords_pred.flatten(0, 1), chords.long().flatten())

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.n_pr_words)
        rvq_loss = loss_fn(rvq_pred.flatten(0, 2), x.flatten().long())

        return melody_loss, chords_loss, rvq_loss

    def encode2roll(self, x, melody):
        input_x = x[:, :-1]
        target = x
        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.positional_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embed_x.shape[1]).to(embed_x.device)
        src_padding_mask = torch.where(target[:, :, 0] == self.n_pr_words, float('-inf'), 0)
        return self.transformer_decoder.encode2roll(
            target=target,
            melody=melody,
            src=embed_x_with_pos,
            mask=attn_mask,
            src_key_padding_mask=src_padding_mask,
            is_causal=True
        )

    def roll(self, param_dict, layer_idx, fn):
        return self.transformer_decoder.roll(param_dict=param_dict, layer_idx=layer_idx, fn=fn)

    def roll2end(self, param_dict):
        decoder_output = self.transformer_decoder.roll2end(param_dict)
        memory = decoder_output.flatten(0, 1)[:, None]

        target = param_dict["target"].flatten(0, 1)
        melody = param_dict["melody"].flatten(0, 1)
        outputs = self.baby_llm(tgt=target[:, :-1],
                                memory=memory,
                                melody=melody.unsqueeze(1))
        melody_pred = self.melody_out(decoder_output)

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.n_pr_words)
        tf_loss = loss_fn(outputs.flatten(0, 1), target.long().flatten())
        loss_fn = nn.CrossEntropyLoss(ignore_index=128)
        melody_loss = loss_fn(melody_pred.flatten(0, 1), melody.long().flatten())
        return outputs, melody_pred, tf_loss, melody_loss

    def inference(self, x,
                  melody,
                  chords,
                  refine_fn, activation,
                  max_len=100, top_k=10, temperature=1.):
        decoded_sequence = []
        assert x.shape[1] < max_len
        print(x.shape, chords.shape, melody.shape)
        input_tokens = self.input_embedding.prepare(x, melody, chords)
        for i in range(len(input_tokens)):
            print("------------------------")
            for j in range(len(input_tokens[i])):
                print(input_tokens[i][j].shape)

        embed_x = self.input_embedding.encode(input_tokens)
        melody_sequence = []

        for i, _ in tqdm(enumerate(range(max_len - x.shape[1])), total=max_len - x.shape[1],
                         desc=f"inference"):
            embedding = self.positional_encoding(embed_x)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(embedding.shape[1]).to(embedding.device)
            decoder_output = self.transformer_decoder(embedding,
                                                      is_causal=True,
                                                      mask=attn_mask)
            next_token, next_melody, next_chords = self.output_proj.inference(context=decoder_output[:, -1],
                                                                              refine_fn=refine_fn,
                                                                              activation=activation,
                                                                              temperature=temperature,
                                                                              top_k=top_k)

            next_input_tokens = self.input_embedding.prepare(next_token[:, None],
                                                             next_melody[:, None],
                                                             next_chords[:, None])
            decoded_sequence.append(next_token)
            embed_x = torch.concat([embed_x,
                                    self.input_embedding.encode(next_input_tokens, padding=False)], 1)
            melody_sequence.append(next_melody)

        return torch.stack(decoded_sequence, 1), torch.stack(melody_sequence, 1)

    def save_weights(self, model_path):
        torch.save(self.state_dict(), model_path)
