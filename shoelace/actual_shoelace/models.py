import torch
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_mha import LowRankMultiheadAttention

from shoelace.utils.network_utils import freeze, print_params
from shoelace.pfMIDILM.MIDILM import PAD, SEG_RES
from shoelace.actual_shoelace.utils import make_index
from tqdm import tqdm

from peft import LoraModel, LoraConfig, get_peft_model
from shoelace.audiocraft.utils import utils

STOP_ITER = "stop_iter"


class EmptyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def save_lora_weights(network, path):
    state = network.state_dict()
    for name in list(state.keys()):
        if "lora_A" not in name.split(".") and "lora_B" not in name.split("."):
            state.pop(name)

    torch.save(state, path + ".lora.pth")


def get_musicgen(sec, device, is_gen, is_tuned):
    if is_gen:
        from shoelace.adapt_musicgen_gen.models.musicgen import MusicGen
    else:
        from shoelace.adapt_musicgen.models.musicgen import MusicGen

    mg = MusicGen.get_pretrained(name='large', device=device)
    mg.set_generation_params(duration=sec, extend_stride=16, top_k=250)
    mg.lm.prepare_inputs_for_generation = None
    mg.lm.generation_config = None
    mg.lm.init_qkv()
    target_modules = ["self_attn.q_proj",
                      "self_attn.k_proj",
                      "self_attn.v_proj",
                      "self_attn.out_proj"]
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.01,
    )
    mg.lm = get_peft_model(mg.lm, config)
    if not is_tuned:
        freeze(mg.lm)
    return mg


def get_midi_lm(device, is_tuned, is_gen):
    from shoelace.pfMIDILM.config_1024_8_12_512_8_3 import midi_lm_param, baby_param
    if is_gen:
        from shoelace.pfMIDILM.MIDILM_gen import MIDILM
    else:
        from shoelace.pfMIDILM.MIDILM import MIDILM

    n_layers = midi_lm_param["num_layers"]

    midi_lm = MIDILM(param=midi_lm_param,
                     baby_param=baby_param)
    midi_lm.load_state_dict(torch.load("exp/midi_lm/latest_1_9000.pth", map_location="cpu"))

    midi_lm.set_config(device)
    midi_lm.prepare_for_lora(mode="config")
    midi_lm.prepare_inputs_for_generation = None
    midi_lm.generation_config = None
    target_modules = ["self_attn.q_proj",
                      "self_attn.k_proj",
                      "self_attn.v_proj"]
    for i in range(n_layers):
        target_modules.append(f"transformer_decoder.layers.{i}.self_attn.out_proj")

    # print("---------------------")
    # print_params(midi_lm)
    # assert False
    freeze(midi_lm)

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.01,
    )
    peft_midi_lm = get_peft_model(midi_lm, config)
    if is_gen or is_tuned:
        freeze(midi_lm.baby_llm, is_freeze=False)
    elif not is_tuned:
        freeze(peft_midi_lm)
    return peft_midi_lm, n_layers


class MIDILMGEN(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        lm, _ = get_midi_lm(device, is_gen=True, is_tuned=True)
        self.lm = lm
        print_params(self)

    def forward(self, midi_seq):
        loss = self.lm.lora_forward(midi_seq)
        return {
            "loss": loss
        }

    def inference(self, x, max_len=512, top_k=32, temperature=1.):
        return self.lm.inference(x, max_len=max_len, top_k=top_k, temperature=temperature)

    def save_weights(self, path):
        save_lora_weights(self, path)
        torch.save(self.lm.baby_llm.state_dict(), path + ".baby.pth")

    def load_weights(self, path):
        self.lm.baby_llm.load_state_dict(torch.load(path + ".baby.pth", map_location="cpu"))
        self.load_state_dict(torch.load(path + ".lora.pth", map_location="cpu"),
                             strict=False)


class AudioLMGEN(nn.Module):
    def __init__(self, sec, device="cuda", frame_rate=50):
        super().__init__()
        mg = get_musicgen(sec, device, is_gen=True, is_tuned=True)
        self.musicgen = mg
        self.lm = mg.lm
        self.max_duration = sec
        self.frame_rate = frame_rate
        self.autocast = mg.autocast

    def set_training(self):
        self.lm.train()

    def forward(self, audio_seq):

        lm = self.lm
        with self.autocast:
            pred = lm.compute_predictions(codes=audio_seq.transpose(1, 2),
                                          conditions=[None] * len(audio_seq))
            loss = self.loss_func(audio_seq=audio_seq, pred=pred)
        return {
            "loss": loss
        }

    def prepare_for_infer(self, prompt, max_gen_len, num_samples, device):
        lm = self.lm
        pattern = lm.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, lm.num_codebooks, 0), dtype=torch.long, device=device)
        else:
            prompt = prompt.transpose(1, 2)

        B, K, T = prompt.shape
        start_offset = T
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        gen_codes[..., :start_offset] = prompt
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, lm.special_token_id)
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None
        return gen_sequence, mask, start_offset_sequence

    def sample_next_tokens(self, logits, mask, offset, gen_sequence,
                           use_sampling: bool = True,
                           temp: float = 1.0,
                           top_k: int = 250,
                           top_p: float = 0.0):

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        unknown_token = -1
        valid_mask = mask[..., offset:offset + 1].expand(len(logits), -1, -1)
        next_token[~valid_mask] = self.lm.special_token_id
        gen_sequence[..., offset:offset + 1] = torch.where(
            gen_sequence[..., offset:offset + 1] == unknown_token,
            next_token, gen_sequence[..., offset:offset + 1]
        )
        return gen_sequence

    @torch.no_grad()
    def inference(self, audio_prompt,
                  audio_max_gen_len,
                  num_samples,
                  audio_top_k,
                  audio_walk_steps,
                  device):
        max_gen_len = audio_max_gen_len - 3
        gen_sequence, mask, start_offset_sequence = self.prepare_for_infer(audio_prompt,
                                                                           max_gen_len,
                                                                           num_samples,
                                                                           device)
        offset = start_offset_sequence
        for i, step_unit in enumerate(audio_walk_steps):
            curr_sequence = gen_sequence[..., :offset]
            for j in range(step_unit):
                with self.musicgen.autocast:
                    gen = self.lm.yield_inference(
                        curr_sequence)
                    for k in range(self.n_layers):
                        yield next(gen)
                    logits = next(gen)

            gen_sequence = self.sample_next_tokens(logits, mask, offset,
                                                   gen_sequence,
                                                   top_k=audio_top_k)
            offset += step_unit

        out_codes = self.decode(gen_sequence,
                                max_gen_len=audio_max_gen_len)
        out_codes = out_codes[..., :max_gen_len]
        yield out_codes

    def decode(self, gen_sequence, max_gen_len):
        unknown_token = -1
        pattern = self.lm.pattern_provider.get_pattern(max_gen_len)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
        return out_codes

    def get_input_embeddings(self):
        return self.lm.emb

    def loss_func(self, audio_seq, pred):
        audio_seq = audio_seq.transpose(1, 2)
        pred = pred.logits

        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        audio_pred = pred[:, :, :-3]
        audio_target = audio_seq[:, :, :-3]
        return loss_fn(audio_pred.flatten(0, 2), audio_target.long().flatten())

    def load_weights(self, path):
        self.load_state_dict(torch.load(path + ".lora.pth", map_location="cpu"),
                             strict=False)

    def save_weights(self, path):
        save_lora_weights(self, path)


class MIDILM(nn.Module):
    def __init__(self, sec, is_tuned, n_layers=None, device="cuda", frame_rate=50, is_gen=False):
        super().__init__()
        lm, n_layers = get_midi_lm(device, is_gen=is_gen, is_tuned=is_tuned)
        self.lm = lm
        self.max_duration = sec
        self.frame_rate = frame_rate
        self.autocast = EmptyContext()
        self.n_layers = n_layers
        self.is_gen = is_gen
        print_params(self)

    def forward(self, midi_seq):
        out = yield from self.lm.yield_forward(midi_seq)
        yield out

    def inference_midi(self, x, max_len=512, top_k=32, temperature=1.):
        return self.lm.inference(x, max_len=max_len, top_k=top_k, temperature=temperature)

    def sample_next_tokens(self, logits,
                           gen_sequence,
                           midi_prompt,
                           top_k,
                           temperature=1.):
        next_token = self.lm.baby_llm.inference(memory=logits[:, -1:],
                                                temperature=temperature,
                                                top_k=top_k)
        if gen_sequence.shape[1] < midi_prompt.shape[1]:
            offset = gen_sequence.shape[1]
            next_token = torch.where(midi_prompt[:, offset] < 0, next_token, midi_prompt[:, offset])
        return torch.concat([gen_sequence, next_token[:, None]], 1)



    @torch.no_grad()
    def inference(self, midi_prompt,
                  midi_max_gen_len,
                  num_samples,
                  midi_top_k,
                  midi_walk_steps,
                  device):
        seq_mask = (midi_prompt[:, :, 0] < 0).sum(0)
        start_offset = -1

        for i in range(len(seq_mask)):
            if seq_mask[i] > 0:
                start_offset = i
                break

        if start_offset == -1:
            gen_sequence = midi_prompt
        else:
            gen_sequence = midi_prompt[:, :start_offset]

        index, stop_sig = make_index(gen_sequence, chunk_len=midi_max_gen_len)

        for i, step_unit in enumerate(midi_walk_steps):
            gen = self.lm.yield_forward(gen_sequence.long(), cut_seq=False, return_memory=True)
            for k in range(self.n_layers):
                hidden_states = next(gen)
                if step_unit > 0:
                    yield hidden_states[0], hidden_states[1], hidden_states[2], index
                else:
                    yield hidden_states
            logits = next(gen)
            if step_unit > 0:
                gen_sequence = self.sample_next_tokens(logits,
                                                       gen_sequence,
                                                       midi_prompt,
                                                       top_k=midi_top_k)

                index, stop_sig = make_index(gen_sequence, chunk_len=midi_max_gen_len)

            if stop_sig:
                yield None, None, None, STOP_ITER
                break

            # offset += step_unit
        # gen_sequence = gen_sequence[:, :midi_max_gen_len]
        yield gen_sequence

    def loss_func(self, midi_seq, pred):
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        return loss_fn(pred.flatten(0, 1), midi_seq.long().flatten())

    def save_weights(self, path):
        save_lora_weights(self, path)
        torch.save(self.lm.baby_llm.state_dict(), path + ".baby.pth")

    def load_weights(self, path):
        self.lm.baby_llm.load_state_dict(torch.load(path + ".baby.pth", map_location="cpu"))
        self.load_state_dict(torch.load(path + ".lora.pth", map_location="cpu"),
                             strict=False)


class AudioLM(nn.Module):
    def __init__(self, sec, n_layers, is_tuned, device="cuda", frame_rate=50):
        super().__init__()
        mg = get_musicgen(sec, device, is_gen=False, is_tuned=is_tuned)
        self.musicgen = mg
        self.lm = mg.lm
        self.max_duration = sec
        self.frame_rate = frame_rate
        self.autocast = mg.autocast
        self.n_layers = n_layers

    def set_training(self):
        self.lm.train()

    def forward(self, audio_seq):
        with self.autocast:
            out = yield from self.lm.compute_predictions(codes=audio_seq.transpose(1, 2),
                                                         conditions=[None] * len(audio_seq))
        yield out

    def prepare_for_infer(self, prompt, max_gen_len, num_samples, device):
        lm = self.lm
        pattern = lm.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, lm.num_codebooks, 0), dtype=torch.long, device=device)
        else:
            prompt = prompt.transpose(1, 2)

        B, K, T = prompt.shape
        start_offset = T
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        gen_codes[..., :start_offset] = prompt
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, lm.special_token_id)
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None
        return gen_sequence, mask, start_offset_sequence

    def sample_next_tokens(self, logits, mask, offset, gen_sequence,
                           use_sampling: bool = True,
                           temp: float = 1.0,
                           top_k: int = 250,
                           top_p: float = 0.0):

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        unknown_token = -1
        valid_mask = mask[..., offset:offset + 1].expand(len(logits), -1, -1)
        next_token[~valid_mask] = self.lm.special_token_id
        gen_sequence[..., offset:offset + 1] = torch.where(
            gen_sequence[..., offset:offset + 1] == unknown_token,
            next_token, gen_sequence[..., offset:offset + 1]
        )
        return gen_sequence

    @torch.no_grad()
    def inference(self, audio_prompt,
                  audio_max_gen_len,
                  num_samples,
                  audio_top_k,
                  audio_walk_steps,
                  device):
        max_gen_len = audio_max_gen_len - 3
        gen_sequence, mask, start_offset_sequence = self.prepare_for_infer(audio_prompt,
                                                                           max_gen_len,
                                                                           num_samples,
                                                                           device)
        offset = start_offset_sequence
        for i, step_unit in enumerate(audio_walk_steps):
            curr_sequence = gen_sequence[..., :offset]
            with self.musicgen.autocast:
                gen = self.lm.yield_inference(
                    curr_sequence)
                for k in range(self.n_layers):
                    yield next(gen)
                logits = next(gen)
            if step_unit > 0:
                gen_sequence = self.sample_next_tokens(logits, mask, offset,
                                                       gen_sequence,
                                                       top_k=audio_top_k)
            offset += step_unit

        out_codes = self.decode(gen_sequence,
                                max_gen_len=audio_max_gen_len)
        out_codes = out_codes[..., :max_gen_len]
        yield out_codes

    def decode(self, gen_sequence, max_gen_len):
        unknown_token = -1
        pattern = self.lm.pattern_provider.get_pattern(max_gen_len)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
        return out_codes

    def get_input_embeddings(self):
        return self.lm.emb

    def loss_func(self, audio_seq, pred):
        audio_seq = audio_seq.transpose(1, 2)
        pred = pred.logits

        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        audio_pred = pred[:, :, :-3]
        audio_target = audio_seq[:, :, :-3]
        return loss_fn(audio_pred.flatten(0, 2), audio_target.long().flatten())

    def load_weights(self, path):
        self.load_state_dict(torch.load(path + ".lora.pth", map_location="cpu"),
                             strict=False)

    def save_weights(self, path):
        save_lora_weights(self, path)


class SholaceParam(nn.Module):
    def __init__(self,
                 n_layers=48,
                 a_embed_dim=2048,
                 b_embed_dim=1024,
                 num_heads=32,
                 low_rank_dim=64,
                 long_first=True,
                 multi_factor=16):
        super(SholaceParam, self).__init__()
        self.cross_attn = nn.ModuleList([
            LowRankMultiheadAttention(in_dim=b_embed_dim,
                                      embed_dim=a_embed_dim,
                                      num_heads=num_heads,
                                      low_rank_dim=low_rank_dim,
                                      long_first=long_first,
                                      multi_factor=multi_factor)
            for _ in range(n_layers)])

    def set_config(self, device):
        for layer in self.cross_attn:
            layer.set_config(device)

    def forward(self):
        for i in range(len(self.cross_attn)):
            yield self.cross_attn[i]
