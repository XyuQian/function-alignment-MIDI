import torch
import torch.nn as nn
from .low_rank_mha import LowRankMultiheadAttention
from shoelace.adapt_musicgen.musicgen_air import MusicGen
import numpy as np
from shoelace.utils.network_utils import freeze, print_params
from shoelace.pianorollLM.pianoroll_lm_with_baby import PianoRollLM, PositionalEncoding

from peft import LoraModel, LoraConfig, get_peft_model
from tqdm import tqdm
from shoelace.audiocraft.utils import utils


def save_lora_weights(network, path):
    state = network.state_dict()
    for name in list(state.keys()):
        if "lora_A" not in name.split(".") and "lora_B" not in name.split("."):
            state.pop(name)

    torch.save(state, path + ".lora.pth")


def get_musicgen(sec, device):
    mg = MusicGen.get_pretrained(name='large', device=device)
    mg.set_generation_params(duration=sec, extend_stride=16, top_k=200)
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
    return mg


def get_midi_lm(device="cuda"):
    midi_lm = PianoRollLM()
    midi_lm.load_state_dict(torch.load("save_models/llm_pop_vocals_3.pth", map_location="cpu"))
    midi_lm.prepare_for_lora(mode="config")
    midi_lm.prepare_inputs_for_generation = None
    target_modules = ["self_attn.q_proj",
                      "self_attn.k_proj",
                      "self_attn.v_proj",
                      "self_attn.o_proj"]
    freeze(midi_lm)

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.01,
    )
    midi_lm_peft = get_peft_model(midi_lm, config)

    return midi_lm_peft


def create_mask(a_len, b_len, device):
    dt = np.random.rand()
    mask_seq = torch.rand(a_len) * b_len * (1 + dt)
    mask_seq, _ = torch.sort(mask_seq)
    mask_seq_a = mask_seq[:, None].repeat(1, b_len)
    mask_seq_b = torch.arange(b_len)[None, :].repeat(a_len, 1)
    mask_a = torch.where(mask_seq_a > mask_seq_b, 0, float(-torch.inf))
    mask_b = torch.where(mask_seq_a > mask_seq_b, float(-torch.inf), 0).transpose(0, 1)

    mask = torch.rand_like(mask_a)
    mask_a[mask < dt/2] = float(-torch.inf)
    mask_a[:, 0] = 0
    mask = torch.rand_like(mask_b)
    mask_b[mask < dt/2] = float(-torch.inf)
    mask_b[:, 0] = 0
    return mask_a.to(device), mask_b.to(device)


class CondMusicgen(nn.Module):
    def __init__(self, sec, device="cuda", frame_rate=50):
        super().__init__()
        mg = get_musicgen(sec, device)
        self.musicgen = mg
        self.lm = mg.lm
        self.max_duration = sec
        self.frame_rate = frame_rate

    def set_training(self):
        self.lm.train()

    def forward(self, seq, desc, embed_fn, num_samples=1, mode="train",
                total_gen_len=None, prompt_tokens=None):
        mg = self.musicgen
        lm = self.lm
        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)
        with mg.autocast:
            out = lm.compute_predictions(codes=seq,
                                         embed_fn=embed_fn,
                                         conditions=attributes)
        return out

    def prepare_for_infer(self, desc, prompt, max_gen_len, num_samples, device):
        mg = self.musicgen
        lm = self.lm
        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)
        conditions = lm.cfg_dropout(attributes)
        conditions = lm.att_dropout(conditions)
        tokenized = lm.condition_provider.tokenize(conditions)
        condition_tensors = lm.condition_provider(tokenized)

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
        return condition_tensors, attributes, gen_sequence, mask, start_offset_sequence

    def test_generate(self, fn, desc, prompt, xlen):
        prompt = None
        mg = self.musicgen
        lm = self.lm
        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)

        all_tokens = [] if prompt is None else [prompt]
        stride_tokens = int(self.frame_rate * mg.extend_stride)
        current_gen_offset = 0
        prompt_length = prompt.shape[-1] if prompt is not None else 0
        prompt_tokens = prompt
        total_gen_len = xlen
        total_sec = total_gen_len / 50.
        print("generate", current_gen_offset, prompt_length, total_gen_len, len(desc))
        while current_gen_offset + prompt_length < total_gen_len:
            time_offset = current_gen_offset / self.frame_rate
            chunk_duration = min(total_sec - time_offset, self.max_duration)
            max_gen_len = int(chunk_duration * self.frame_rate)
            print(max_gen_len, prompt_length)
            if prompt_length >= max_gen_len:
                break
            # print("current_gen_offset / total ", current_gen_offset, "/", total_gen_len)
            with mg.autocast:
                gen_tokens = lm.yy_generate(num_samples=len(desc),
                                            emb_fn=fn,
                                            prompt=prompt_tokens,
                                            conditions=attributes,
                                            callback=None, max_gen_len=max_gen_len, **mg.generation_params)
            if prompt_tokens is None:
                all_tokens.append(gen_tokens)
            else:
                print("???????????????????????????????/")
                all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
            prompt_tokens = gen_tokens[:, :, stride_tokens:]
            prompt_length = prompt_tokens.shape[-1]
            current_gen_offset += stride_tokens
            if current_gen_offset > 50 * 80:
                break

        gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

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
        gen_sequence[..., offset:offset + 1] = next_token
        return gen_sequence

    def decode(self, gen_sequence, max_gen_len):
        unknown_token = -1
        pattern = self.lm.pattern_provider.get_pattern(max_gen_len)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
        return out_codes

    def get_input_embeddings(self):
        return self.lm.emb


class SholaceParam(nn.Module):
    def __init__(self,
                 n_a_layers=48,
                 n_b_layers=8,
                 a_embed_dim=2048, b_embed_dim=1024,
                 a_num_heads=32, b_num_heads=8,
                 dropout=0.1, ):
        super(SholaceParam, self).__init__()
        self.audio_self_attn = nn.ModuleList([
            LowRankMultiheadAttention(in_dim=b_embed_dim,
                                      embed_dim=a_embed_dim,
                                      num_heads=a_num_heads)
            for i in range(n_a_layers)])
        self.midi_self_attn = nn.ModuleList([
            LowRankMultiheadAttention(in_dim=a_embed_dim,
                                      embed_dim=b_embed_dim,
                                      num_heads=b_num_heads)
            for i in range(n_b_layers)])

    def set_config(self, device):
        for layer in self.audio_self_attn:
            layer.set_config(device)
        for layer in self.midi_self_attn:
            layer.set_config(device)

    def forward(self, x_a, x_b, layer_idx, mode, attn_mask, cur_step=None, debug=None):

        if mode == "a2b":
            return self.audio_self_attn[layer_idx](
                x_a, x_b, mode=mode, cur_step=cur_step,
                attn_mask=attn_mask,
                debug=debug,
            )
        else:
            return self.midi_self_attn[layer_idx](
                x_a, x_b, mode=mode, cur_step=cur_step,
                attn_mask=attn_mask,
                debug=debug
            )


class Sholace(nn.Module):
    def __init__(self, midi_lm_conf, audio_lm_conf, is_mono):
        super().__init__()
        self.cur_audio_step = None
        self.dropout_p = .1
        self.is_mono = is_mono

        n_midi_layers = midi_lm_conf["n_layers"]
        n_audio_layers = audio_lm_conf["n_layers"]

        self.midi_lm = midi_lm_conf["model"]
        self.audio_lm = audio_lm_conf["model"]

        self.adapter = SholaceParam(n_a_layers=n_audio_layers,
                                    n_b_layers=n_midi_layers)

        self.midi_state = None
        self.stride = 6

        self.cache = None
        self.hidden_states = {}
        self.cache_seq = {
            "midi": None,
            "audio": None
        }
        self.cur_audio_layer_idx = 0

    def save_weights(self, path):
        torch.save(self.adapter.state_dict(), path)
        save_lora_weights(self.audio_lm, path + ".audio")
        save_lora_weights(self.midi_lm, path + ".midi")

    def load_weights(self, path, device):
        self.adapter.load_state_dict(torch.load(path, map_location=device))
        self.audio_lm.load_state_dict(torch.load(path + ".audio.lora.pth", map_location=device), strict=False)
        self.midi_lm.load_state_dict(torch.load(path + ".midi.lora.pth", map_location=device), strict=False)

    def audio_emb_fn_unit(self, query, debug):
        audio_layer_idx = self.cur_audio_layer_idx
        self.hidden_states["audio"] = query

        if audio_layer_idx % self.stride == 0:
            midi_layer_idx = audio_layer_idx // self.stride
            self.cache = self.midi_lm.roll(param_dict=self.cache,
                                           layer_idx=midi_layer_idx,
                                           fn=self.midi_emb_fn)

        if self.cur_audio_step is None:
            cur_step = None
        else:
            cur_step = self.cur_audio_step[1]
        midi_state = self.hidden_states["hidden_state"]
        return self.adapter(x_a=query,
                            x_b=midi_state,
                            debug=debug,
                            cur_step=cur_step,
                            attn_mask=self.cache["audio_attn_mask"],
                            layer_idx=audio_layer_idx,
                            mode="a2b")

    def audio_emb_fn(self, idx, mode="set_layer_idx"):
        if mode == "set_layer_idx":
            self.cur_audio_layer_idx = idx
            return self.audio_emb_fn_unit
        elif mode == "update_interval":
            self.cur_audio_step = idx
            return self.audio_emb_fn

    def midi_emb_fn(self, hidden_state, q):
        self.hidden_states["hidden_state"] = hidden_state
        midi_layer_idx = self.cur_audio_layer_idx // self.stride
        audio_x = self.hidden_states["audio"]
        return self.adapter(x_a=hidden_state,
                            x_b=audio_x,
                            debug=q,
                            attn_mask=self.cache["midi_attn_mask"],
                            layer_idx=midi_layer_idx,
                            mode="b2a")

    def set_training(self, device):
        self.audio_lm.set_training()
        self.midi_lm.set_training(device)

    def set_config(self, device):
        self.adapter.set_config(device)
        self.midi_lm.set_config(device)

    def forward(self, audio_seq, midi_seq, melody_seq, desc):
        with torch.no_grad():
            self.cache = self.midi_lm.encode2roll(midi_seq, melody_mask=melody_seq)

        a_len, b_len = audio_seq.shape[-1], midi_seq.shape[1]
        a_len += 1

        if np.random.rand() > .5:
            audio_attn_mask, midi_attn_mask = create_mask(a_len, b_len, audio_seq.device)
        else:
            midi_attn_mask, audio_attn_mask = create_mask(b_len, a_len, audio_seq.device)

        self.cache["audio_attn_mask"] = audio_attn_mask
        self.cache["midi_attn_mask"] = midi_attn_mask

        audio_out = self.audio_lm(audio_seq, desc, embed_fn=self.audio_emb_fn)
        audio_out = audio_out.logits
        midi_loss = self.midi_lm.roll2end(self.cache, with_acc_loss=not self.is_mono)
        self.cache = None
        self.hidden_states = {}

        audio_pred = audio_out[:, :, :-3]
        audio_target = audio_seq[:, :, :-3]
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        audio_loss = loss_fn(audio_pred.flatten(0, 2), audio_target.long().flatten())
        return audio_loss, midi_loss

    @torch.no_grad
    def inference(self, midi_seq, audio_seq, desc, mode="a2b"):

        if mode == "a2b":
            max_gen_len = 50 * 16 - 2
            condition_tensors, attributes, gen_sequence, mask, start_offset_sequence \
                = self.audio_lm.prepare_for_infer(desc=desc,
                                                  prompt=None,
                                                  max_gen_len=max_gen_len,
                                                  num_samples=len(audio_seq),
                                                  device=audio_seq.device)
            gen_sequence_len = gen_sequence.shape[-1]
            self.cache = self.midi_lm.encode2roll(midi_seq)
            self.cache["audio_attn_mask"] = None
            self.cache["midi_attn_mask"] = None

            for offset in tqdm(range(start_offset_sequence, gen_sequence_len),
                               total=gen_sequence_len - start_offset_sequence,
                               desc=f"inference"):
                embed_fn = self.audio_emb_fn(idx=(0, offset), mode="update_interval")
                curr_sequence = gen_sequence[..., :offset]
                with self.audio_lm.musicgen.autocast:
                    logits = self.audio_lm.lm.yy_generate(
                        embed_fn, curr_sequence, conditions=[], condition_tensors=condition_tensors)
                gen_sequence = self.audio_lm.sample_next_tokens(logits, mask, offset, gen_sequence)
            predict_rvq = self.audio_lm.decode(gen_sequence, max_gen_len)
            return predict_rvq

        elif mode == "b2a":
            max_gen_len = audio_seq.shape[1]
            condition_tensors, attributes, gen_sequence, mask, start_offset_sequence \
                = self.audio_lm.prepare_for_infer(desc=desc,
                                                  prompt=audio_seq,
                                                  max_gen_len=max_gen_len,
                                                  num_samples=len(audio_seq),
                                                  device=audio_seq.device)
            gen_sequence_len = gen_sequence.shape[-1] // 16
            gen_sequence = gen_sequence[..., :max_gen_len]
            midi_pred = None

            for offset in tqdm(range(gen_sequence_len),
                               total=gen_sequence_len,
                               desc=f"inference"):
                self.cache = self.midi_lm.encode2roll(midi_pred, n=len(audio_seq))
                self.cache["audio_attn_mask"] = None
                self.cache["midi_attn_mask"] = None

                embed_fn = self.audio_emb_fn(idx=None, mode="update_interval")
                with self.audio_lm.musicgen.autocast:
                    _ = self.audio_lm.lm.yy_generate(
                        embed_fn, (gen_sequence + 0.).long(), conditions=[], condition_tensors=condition_tensors)
                next_midi_token = self.midi_lm.sample_next_tokens(self.cache,
                                                                  top_k=2,
                                                                  acc_mask=self.is_mono)
                midi_pred = next_midi_token if midi_pred is None else torch.cat([midi_pred, next_midi_token], 1)
            return midi_pred


def config_model(model, trainable_layers):
    freeze(model)
    for n, p in model.named_parameters():
        for layer in trainable_layers:
            if layer in n.split("."):
                p.requires_grad = True
    return model


class Yingyang(nn.Module):
    def __init__(self, is_mono, sec=15):
        super().__init__()
        audio_lm = CondMusicgen(sec)
        midi_lm = get_midi_lm()

        midi_lm_conf = {
            "model": midi_lm,
            "n_layers": 8
        }
        audio_lm_conf = {
            "model": audio_lm,
            "n_layers": 48
        }

        sholace = Sholace(midi_lm_conf=midi_lm_conf,
                          audio_lm_conf=audio_lm_conf,
                          is_mono=is_mono)
        # sholace = config_model(model=sholace,
        #                        trainable_layers=[
        #                            "adapter"
        #                        ])
        self.sholace = sholace
        print_params(self)

    def set_training(self, device):
        self.sholace.set_training(device)
        print_params(self)

    def set_config(self, device):
        self.sholace.set_config(device)

    def save_weights(self, path):
        self.sholace.save_weights(path)

    def load_weights(self, path, device="cpu"):
        self.sholace.load_weights(path, device)

    def forward(self, audio_seq, melody_seq, midi_seq, desc):
        return self.sholace(audio_seq=audio_seq,
                            melody_seq=melody_seq,
                            midi_seq=midi_seq,
                            desc=desc)

    def inference(self, midi_seq, audio_seq, desc, mode):
        with torch.no_grad():
            rvq_codes = self.sholace.inference(
                midi_seq=midi_seq,
                audio_seq=audio_seq,
                desc=desc, mode=mode)
        return rvq_codes
