from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from shoelace.utils.network_utils import sample
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from shoelace.midi_lm.models.config import PAD, SOS, N_ONSET, \
    N_INSTRUMENT, N_PITCH, N_DUR_X, N_DUR_Y, N_VELOCITY, SEG_RES

import torch


def generate_attention_mask(seq_len: int, attn_window: int, mask_prob: float, device: torch.device) -> torch.Tensor:
    """
    Generate a custom attention mask.

    Args:
        seq_len (int): Total sequence length.
        attn_window (int): Number of previous tokens each token can fully attend to.
        mask_prob (float): Probability of masking tokens outside the attention window.
        device (torch.device): The device to place tensors on.

    Returns:
        torch.Tensor: Attention mask where `-inf` means masked attention.
    """
    # Initialize attention mask with Transformer default
    attention_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    # Create index matrix
    # indices = torch.arange(seq_len, device=device).reshape(1, -1)
    # indices = indices.expand(seq_len, -1)  # More efficient than torch.tile
    #
    # # Apply attention window constraint
    # mask = indices < (indices.T - attn_window)
    #
    # # Apply random masking beyond the attention window
    # # random_mask = torch.rand(seq_len, seq_len, device=device) < mask_prob
    # attention_mask[mask] = float('-inf')

    return attention_mask


def check_fn(tgt: torch.Tensor, token: torch.Tensor, pre_token: torch.Tensor, i: int) -> bool:
    """
    Checks whether a newly sampled token is valid based on prior constraints.

    Args:
        tgt (Tensor): The current generated sequence (batch_size, seq_len).
        token (Tensor): The newly proposed token (batch_size, 1) or None.
        pre_token (Tensor): The previous token constraints (batch_size, seq_len).
        i (int): The current step in the generation process.

    Returns:
        bool: True if the token should be rejected and resampled, False otherwise.
    """
    # If no token is sampled yet, force resampling
    if token is None:
        return True

    # If there's no prior token constraint, accept the token
    if pre_token is None:
        return False

    # Ensure correct shapes
    tgt = torch.cat([tgt, token], dim=-1)  # Append token to target sequence
    pre_token = pre_token.squeeze(1)  # Remove singleton dimension if present
    token = token.squeeze(1)  # Ensure token has correct shape
    tgt = tgt.squeeze(1)  # Ensure tgt has correct shape

    # If `i > 1`, accept the token without checking
    if i > 1:
        return False

    # Special conditions for `i == 0`
    if i == 0:
        for j in range(len(tgt)):
            if SEG_RES > pre_token[j, 0] > token[j]:
                return True  # Resample if token violates segment boundary rules

    # Special conditions for `i == 1`
    else:
        for j in range(len(token)):
            if i == 1 and (tgt[j, i] == SEG_RES or pre_token[j, i] == SEG_RES):
                continue  # Skip checks if SEG_RES token is present

            if tgt[j, i] == pre_token[j, i - 1] and token[j] < pre_token[j, i]:
                print(token[j], pre_token[j, i], "heeeeeeee")  # Debug print
                return True  # Resample if constraints are violated

    return False  # Token is valid





class BabyLLM(nn.Module):
    def __init__(self, n_words, memory_dim, n_steps, embedding_dim, n_layers, n_heads):
        """
        A lightweight transformer-based model for sequence generation.
        """
        super().__init__()
        self.in_layer = nn.Embedding(n_words, embedding_dim)
        self.mem_linear = nn.Linear(memory_dim, embedding_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, n_steps, 1), requires_grad=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(embedding_dim, n_words, bias=False)
        self.n_steps = n_steps

    @torch.no_grad()
    def inference(self, memory, pre_token, top_k=32, temperature=1.0):
        """
        Performs inference step by step, generating new tokens from memory.
        """
        # print(memory.shape)
        memory = self.mem_linear(memory)
        tgt = torch.zeros([memory.shape[0], 1], dtype=torch.long, device=memory.device) + SOS
        
        for i in range(self.n_steps):
            attn_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
            input_x = self.in_layer(tgt) + self.pos_emb[:, :tgt.shape[1], :]
            decoder_output = self.decoder(input_x, memory, tgt_mask=attn_mask, tgt_is_causal=True)
            logits = self.output_layer(decoder_output[:, -1])  # Ensure correct shape
            next_token = None

            # while check_fn(tgt, next_token, pre_token, i):
            next_token = sample(logits, top_k_val=top_k, temperature=temperature)

            tgt = torch.cat([tgt, next_token[:, None]], dim=1)

        # Apply SEG_RES mask
        tgt = tgt[:, 1:]  # Remove initial SOS
        tgt[tgt[:, 0] == SEG_RES, 1:] = PAD  # Masking similar to first script
        return tgt

    def forward(self, tgt, memory):
        tgt = self.in_layer(tgt) + self.pos_emb
        memory = self.mem_linear(memory)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(self.n_steps).to(tgt.device)
        decoder_output = self.decoder(tgt, memory, tgt_mask=attn_mask, tgt_is_causal=True)
        return self.output_layer(decoder_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2049):
        """
        Implements positional encoding to provide position information to sequences.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("r_pos", pe.unsqueeze(0))
        self.cache = 0

    def reset_cache(self):
        self.cache = 0

    def forward(self, x):
        if self.training:
            x = x + self.r_pos[:, :x.shape[1], :]
            return x
        else:
            sid = self.cache
            eid = sid + x.shape[1]
            self.cache = eid

            return x + self.r_pos[:, sid :eid, :]


class InputEmbedding(nn.Module):
    def __init__(self, n_words, embedding_dim):
        """
        Embedding layer for multiple input features.
        """
        super().__init__()
        self.layers = nn.ModuleList(nn.Embedding(n, embedding_dim) for n in n_words)

    def forward(self, x):
        return sum(self.layers[i](x[..., i]) for i in range(len(self.layers)))


class MIDILM(nn.Module):
    def __init__(self, param, baby_param, use_generator=False):
        """
        Transformer-based model for MIDI sequence modeling.
        """
        super().__init__()
        self.use_generator = use_generator
        embedding_dim = param["embedding_dim"]
        self.input_embedding = InputEmbedding(n_words=[N_ONSET, N_INSTRUMENT, N_PITCH, N_DUR_X, N_DUR_Y, N_VELOCITY],
                                              embedding_dim=embedding_dim)
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim)
        decoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=param["num_heads"], batch_first=True, use_generator=use_generator)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers=param["num_layers"], use_generator=use_generator)
        self.baby_llm = BabyLLM(**baby_param)
        self.reset_cache()

    def set_use_generator(self, flag: bool):
        self.use_generator = flag
        self.transformer_decoder.set_use_generator(flag)

    def reset_cache(self, reset_sos=True):
        if reset_sos:
            self.cache = False
        self.pos_encoding.reset_cache()
        self.transformer_decoder.reset_cache()

    def yield_forward(self, x, return_loss=True, return_memory=False, with_sos=True, **kwargs):
        """
        Forward pass for MIDI language modeling.
        """
        input_x = F.pad(x, (0, 0, 1, 0), "constant", SOS) if with_sos else x
        
        if return_loss:
            input_x = input_x[:, :-1]

        embed_x = self.input_embedding(input_x)
        
        embed_x_with_pos = self.pos_encoding(embed_x)
        
        # attn_mask = generate_attention_mask(seq_len=embed_x.shape[1],
        #                                     attn_window=512,
        #                                     mask_prob=0.7, device=embed_x.device)

        src_padding_mask = x[:, :, 0] == PAD if return_loss else None

        decoder_output = yield from self.transformer_decoder(embed_x_with_pos,
                                                             src_key_padding_mask=src_padding_mask,
                                                             is_causal=True,
                                                             mask=None)
        
        if return_memory:
            yield decoder_output

        memory = decoder_output.flatten(0, 1)[:, None]
        baby_input_x = F.pad(x[:, :, :-1], (1, 0), "constant", SOS).flatten(0, 1)
        outputs = self.baby_llm(tgt=baby_input_x, memory=memory)
        if return_loss:
            yield nn.CrossEntropyLoss(ignore_index=PAD)(outputs.flatten(0, 1), x.flatten())
        yield outputs

    def load_from_torch_model(self, path: str):
        """
        Loads weights from a checkpoint into the MIDILM model.
        
        The checkpoint is expected to contain keys with prefixes:
        - "baby_llm." for baby_llm,
        - "transformer_decoder." for transformer_decoder,
        - "input_embedding." for input_embedding.
        """
        # Load the checkpoint on CPU.
        state_dict = torch.load(path, map_location="cpu")
        
        # Extract and load baby_llm weights.
        baby_llm_state = {k[len("baby_llm."):]: v for k, v in state_dict.items() if k.startswith("baby_llm.")}
        if not baby_llm_state:
            raise KeyError("Checkpoint does not contain 'baby_llm' weights.")
        self.baby_llm.load_state_dict(baby_llm_state)
        
        # Extract and load transformer_decoder weights.
        transformer_decoder_state = {k[len("transformer_decoder."):]: v for k, v in state_dict.items() if k.startswith("transformer_decoder.")}
        if not transformer_decoder_state:
            raise KeyError("Checkpoint does not contain 'transformer_decoder' weights.")
        self.transformer_decoder.load_from_torch_model(transformer_decoder_state)
        
        # Extract and load input_embedding weights.
        input_embedding_state = {k[len("input_embedding."):]: v for k, v in state_dict.items() if k.startswith("input_embedding.")}
        if not input_embedding_state:
            raise KeyError("Checkpoint does not contain 'input_embedding' weights.")
        self.input_embedding.load_state_dict(input_embedding_state)
        
        print(f"Successfully loaded weights from {path}")


    @torch.no_grad()
    def yield_inference(self, x, max_len, last_chunk=True, top_k=32, temperature=1.0):
        """
        Performs inference by generating a sequence step-by-step.
        """

        decoded_sequence = [None]
        prompt = x
        cur_timing = ((x[0, :, 0] == SEG_RES).sum() -1) * SEG_RES
        assert cur_timing >= max_len
        next_token = None
        for _ in tqdm(range(max_len - cur_timing), initial=cur_timing, desc="MidiLM Inference", total_len=max_len):
            while not next_token or not next_token[0, 0] == SEG_RES:
                if self.use_generator:
                    decoder_output = yield from self(prompt, return_memory=True, 
                        return_loss=False, with_sos=not self.cache, return_val=False)
                else:
                    decoder_output = self(prompt, return_memory=True, 
                        return_loss=False, with_sos=not self.cache, return_val=True)
                self.cache = True
            
                decoder_output = decoder_output[:, -1:]
                # print(decoder_output[0, 0, 100:300], "here")
                next_token = self.baby_llm.inference(memory=decoder_output,
                                                pre_token=decoded_sequence[-1],
                                                temperature=temperature, top_k=top_k)
                decoded_sequence.append(next_token[:, None])
                prompt = next_token[:, None]

        yield {
            "output" : torch.concat([x] + decoded_sequence[1:], 1)
        }
        

    def forward(self, input_ids, return_val=True, **kwargs):
        generator = self.yield_forward(input_ids, **kwargs)
        if self.use_generator:
            return generator
        elif return_val:
            return next(generator)
        return generator

    @torch.no_grad()
    def inference(self, input_ids, **kwargs):
        generator = self.yield_inference(input_ids, **kwargs)
        if self.use_generator:
            return generator
        else:
            return next(generator)["output"]

    def get_cache(self):
        return self.transformer_decoder.get_cache()


    def save_weights(self, model_path):
        """
        Saves the model's state dictionary.
        """
        torch.save(self.state_dict(), model_path)


if __name__ == "__main__":
    import torch

    from shoelace.midi_lm.models.config import baby_param, midi_lm_param

    # Initialize models
    midi_lm = MIDILM(midi_lm_param, baby_param)

    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    vocab_size = 128

    tgt = torch.randint(0, vocab_size, (batch_size, seq_len, 6))
    # Forward pass for MIDILM
    tgt[0, -4:] = PAD
    loss = midi_lm(tgt)

    print("MIDILM Loss:", loss.item())  # Should return a loss value
