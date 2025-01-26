import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .transformer_encoder_gen import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from .preprocess_MIDI import SEG_RES


def sample(logits, top_k_val=20, temperature=1.):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k_val, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    top_k_indices = top_k_indices.flatten(0, 1)
    top_k_probs = top_k_probs.flatten(0, 1)
    next_token = top_k_indices.gather(-1, torch.multinomial(top_k_probs, num_samples=1))
    next_token = next_token.view(-1, 1)
    return next_token


SOS = 130
PAD = 131
N_ONSET = 132
N_INSTRUMENT = 132
N_PITCH = 132
N_DUR_X = 132
N_DUR_Y = 132
N_VELOCITY = 132
MAX_SEQ_LEN = 1500


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048 + 1):
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
    def __init__(self, n_words, memory_dim, n_steps, embedding_dim, n_layers, n_heads):
        super(BabyLLM, self).__init__()
        self.in_layer = nn.Embedding(n_words, embedding_dim)
        self.mem_linear = nn.Linear(memory_dim, embedding_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, n_steps, 1), requires_grad=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_heads,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=n_layers)
        self.output_layer = nn.Linear(embedding_dim, n_words, bias=False)
        self.n_steps = n_steps

    def forward(self, tgt, memory):
        tgt = self.in_layer(tgt) + self.pos_emb
        memory = self.mem_linear(memory)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(self.n_steps).to(tgt.device)
        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=attn_mask,
                                      tgt_is_causal=True)
        out = self.output_layer(decoder_output)
        return out

    @torch.no_grad()
    def inference(self, memory, top_k=10, temperature=1.):
        memory = self.mem_linear(memory)
        tgt = torch.zeros([len(memory), 1]).to(memory.device).long() + SOS
        for i in range(self.n_steps):
            attn_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt[0])).to(tgt.device)
            input_x = self.in_layer(tgt) + self.pos_emb[:, :i + 1]
            decoder_output = self.decoder(input_x, memory,
                                          tgt_mask=attn_mask,
                                          tgt_is_causal=True)
            logits = self.output_layer(decoder_output[:, -1:])
            next_token = sample(logits, top_k_val=top_k, temperature=temperature)
            tgt = torch.concat([tgt, next_token], -1)
        tgt = tgt[:, 1:]
        tgt[tgt[:, 0] == SEG_RES, 1:] = PAD
        return tgt


class InputEmbedding(nn.Module):
    def __init__(self, n_words, embedding_dim):
        super(InputEmbedding, self).__init__()
        self.layers = nn.ModuleList(
            nn.Embedding(n, embedding_dim) for n in n_words
        )
        self.n_words = n_words

    def forward(self, x):
        return sum([self.layers[i](x[..., i]) for i in range(len(self.n_words))])


class MIDILM(nn.Module):
    def __init__(self, param, baby_param):
        super(MIDILM, self).__init__()
        n_words = [N_ONSET, N_INSTRUMENT,
                   N_PITCH, N_DUR_X, N_DUR_Y,
                   N_VELOCITY]
        embedding_dim = param["embedding_dim"]
        num_heads = param["num_heads"]
        num_layers = param["num_layers"]
        self.input_embedding = InputEmbedding(n_words=n_words,
                                              embedding_dim=embedding_dim)
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim)

        decoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                batch_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.baby_llm = BabyLLM(**baby_param)

    def set_config(self, device):
        self.pos_encoding.set_config(device)

    def set_training(self, device):
        pass

    def prepare_for_lora(self, mode):
        self.transformer_decoder.prepare_for_lora(mode)

    def yield_forward(self, midi_seq, return_memory=False):
        x = midi_seq
        input_x = F.pad(x[:, :-1], (0, 0, 1, 0), "constant", SOS)
        baby_input_x = F.pad(x[:, :, :-1], (1, 0), "constant", SOS)

        # print(input_x.shape, baby_input_x.shape)
        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        src_padding_mask = torch.where(x[:, :, 0] == PAD, float('-inf'), 0) if x is not None else None
        decoder_output = yield from self.transformer_decoder(embed_x_with_pos,
                                                             src_key_padding_mask=src_padding_mask,
                                                             is_causal=True,
                                                             mask=attn_mask)
        if return_memory:
            yield decoder_output
        else:
            memory = decoder_output.flatten(0, 1)[:, None]

            baby_input_x = baby_input_x.flatten(0, 1)
            outputs = self.baby_llm(tgt=baby_input_x,
                                    memory=memory)

            yield outputs

    def lora_forward(self, x):

        input_x = F.pad(x[:, :-1], (0, 0, 1, 0), "constant", SOS)
        baby_input_x = F.pad(x[:, :, :-1], (1, 0), "constant", SOS)
        target = x

        # print(input_x.shape, baby_input_x.shape)
        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        src_padding_mask = torch.where(x[:, :, 0] == PAD, float('-inf'), 0) if x is not None else None

        decoder_output = self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)

        memory = decoder_output.flatten(0, 1)[:, None]

        baby_input_x = baby_input_x.flatten(0, 1)
        outputs = self.baby_llm(tgt=baby_input_x,
                                memory=memory)

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        acc_loss = loss_fn(outputs.flatten(0, 1), target.flatten())
        return acc_loss


    def forward(self, x):
        input_x = F.pad(x[:, :-1], (0, 0, 1, 0), "constant", SOS)
        baby_input_x = F.pad(x[:, :, :-1], (1, 0), "constant", SOS)
        target = x

        # print(input_x.shape, baby_input_x.shape)
        embed_x = self.input_embedding(input_x)
        embed_x_with_pos = self.pos_encoding(embed_x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        src_padding_mask = torch.where(x[:, :, 0] == PAD, float('-inf'), 0) if x is not None else None

        decoder_output = self.transformer_decoder(embed_x_with_pos,
                                                  src_key_padding_mask=src_padding_mask,
                                                  is_causal=True,
                                                  mask=attn_mask)

        memory = decoder_output.flatten(0, 1)[:, None]

        baby_input_x = baby_input_x.flatten(0, 1)
        outputs = self.baby_llm(tgt=baby_input_x,
                                memory=memory)

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        acc_loss = loss_fn(outputs.flatten(0, 1), target.flatten())
        return acc_loss

    @torch.no_grad()
    def inference(self, x, max_len=512, top_k=32, temperature=1.):
        x = F.pad(x, (0, 0, 1, 0), "constant", SOS)
        embed_x = self.input_embedding(x)
        decoded_sequence = []
        assert x.shape[1] < max_len
        prompt_len = x.shape[1]

        for i, _ in tqdm(enumerate(range(max_len - prompt_len)), total=max_len - prompt_len,
                         desc=f"inference"):
            embedding = self.pos_encoding(embed_x)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(embedding.shape[1]).to(embedding.device)
            decoder_output = self.transformer_decoder(embedding,
                                                      is_causal=True,
                                                      mask=attn_mask)
            decoder_output = decoder_output[:, -1:]
            next_token = self.baby_llm.inference(memory=decoder_output,
                                                 temperature=temperature,
                                                 top_k=top_k)

            next_token = next_token[:, None]
            decoded_sequence.append(next_token)
            embed_x = torch.concat([embed_x, self.input_embedding(next_token)], 1)
        seq = torch.concat(decoded_sequence, 1)
        seq = torch.concat([x[:, 1:], seq], 1)
        return seq

    def save_weights(self, model_path):
        torch.save(self.state_dict(), model_path)
