import torch
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_mha import LowRankMultiheadAttention
from shoelace.adapt_musicgen.models.musicgen import MusicGen
import numpy as np
from shoelace.utils.network_utils import freeze, print_params
from shoelace.pianorollLM.pianoroll_lm_with_baby import PianoRollLM

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
    return mg


def get_midi_lm(device="cuda"):
    midi_lm = PianoRollLM()
    midi_lm.load_state_dict(torch.load("save_models/llm.pth", map_location="cpu"))
    midi_lm.set_config(device)
    midi_lm.prepare_for_lora(mode="config")
    midi_lm.prepare_inputs_for_generation = None
    midi_lm.generation_config = None
    target_modules = ["self_attn.q_proj",
                      "self_attn.k_proj",
                      "self_attn.v_proj",
                      "self_attn.o_proj"]
    freeze(midi_lm)

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.01,
    )
    midi_lm_peft = get_peft_model(midi_lm, config)

    return midi_lm_peft


def create_mask(a_len, b_len, device, mask_ratio=.7):
    mask_a = torch.zeros([a_len, b_len])
    mask = torch.rand_like(mask_a)
    mask_a[mask < mask_ratio] = float(-torch.inf)
    mask_b = torch.zeros_like(mask_a).transpose(0, 1) + float(-torch.inf)
    mask_a = F.pad(mask_a, (1, 0), "constant", 0)
    mask_b = F.pad(mask_b, (1, 0), "constant", 0)
    return mask_a.to(device), mask_b.to(device)


class AudioLM(nn.Module):
    def __init__(self, sec, device="cuda", frame_rate=50):
        super().__init__()
        mg = get_musicgen(sec, device)
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
            out = yield from lm.compute_predictions(codes=audio_seq,
                                                    conditions=[None] * len(audio_seq))
        yield out

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

    def loss_func(self, audio_seq, pred):
        pred = pred.logits
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        audio_pred = pred[:, :, :-3]
        audio_target = audio_seq[:, :, :-3]
        return loss_fn(audio_pred.flatten(0, 2), audio_target.long().flatten())


class MIDILM(nn.Module):
    def __init__(self, sec, device="cuda", frame_rate=50):
        super().__init__()
        lm = get_midi_lm(device)
        self.lm = lm
        self.max_duration = sec
        self.frame_rate = frame_rate
        self.autocast = None

    def loss_func(self, midi_seq, pred):
        loss_fn = nn.CrossEntropyLoss(ignore_index=512)
        return loss_fn(pred.flatten(0, 1), midi_seq.long().flatten())

    def forward(self, midi_seq):
        out = yield from self.lm.yield_forward(midi_seq)
        yield out


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


class Yinyang(nn.Module):
    def __init__(self, mode="vocals2mel", sec=15):
        super().__init__()
        model_factory = {
            "AudioLM": {
                "model": AudioLM,
                "n_layers": 48,
                "low_rank_dim": 64,
                "hidden_size": 2048,
                "n_heads": 32,
                "steps": 16,
                "seq_len": int(sec * 50 + 1),
                "param_list": ["audio_seq"],
            },
            "MIDILM": {
                "model": MIDILM,
                "low_rank_dim": 64,
                "n_layers": 8,
                "hidden_size": 1024,
                "n_heads": 8,
                "steps": 1,
                "seq_len": int(sec * 50 // 16),
                "param_list": ["midi_seq"]
            },
        }
        skip_layers = {
            "AudioLM-MIDILM": [6, 1],
            "MIDILM-AudioLM": [1, 6],
        }
        recipes = {
            "mel2vocals":
                {
                    "models": ["AudioLM", "MIDILM"],
                    "bi": False,
                },
            "vocals-mel":
                {
                    "models": ["MIDILM", "AudioLM"],
                    "bi": True,
                },
            "vocals2mel":
                {
                    "models": ["MIDILM", "AudioLM"],
                    "bi": False,
                },

        }

        target_recipe = recipes[mode]
        models = nn.ModuleList()
        adapters = nn.ModuleList()
        n_skip_layer_pairs = []
        names = []
        for i, m in enumerate(target_recipe["models"]):
            params = model_factory[m]
            models.append(params["model"](sec=sec))
            names.append(m)
            if i == len(target_recipe["models"]) - 1:
                n_skip_layer_pairs.append([-1, -1])
                break
            next_target = target_recipe["models"][i + 1]
            next_param = model_factory[next_target]
            n_skip_layer_pairs.append(skip_layers[m + "-" + next_target])
            shoelace = nn.ModuleList()
            multi_factor = params["steps"] / next_param["steps"] if params["steps"] >= next_param["steps"] else \
                next_param["steps"] / params["steps"]
            long_first = params["steps"] >= next_param["steps"]
            shoelace.append(
                SholaceParam(
                    n_layers=params["n_layers"],
                    a_embed_dim=params["hidden_size"],
                    b_embed_dim=next_param["hidden_size"],
                    low_rank_dim=params["low_rank_dim"],
                    num_heads=params["n_heads"],
                    multi_factor=multi_factor,
                    long_first=long_first,
                )
            )
            if target_recipe["bi"]:
                shoelace.append(
                    SholaceParam(
                        n_layers=next_param["n_layers"],
                        a_embed_dim=next_param["hidden_size"],
                        b_embed_dim=params["hidden_size"],
                        low_rank_dim=next_param["low_rank_dim"],
                        num_heads=next_param["n_heads"],
                        multi_factor=multi_factor,
                        long_first=not long_first,
                    )
                )
            adapters.append(shoelace)

        self.bi_di = target_recipe["bi"]
        self.models = models
        self.adapters = adapters
        self.names = names
        self.n_skip_layer_pairs = n_skip_layer_pairs
        self.n_layers = model_factory[self.names[0]]["n_layers"]
        self.seq_len = [model_factory[self.names[i]]["seq_len"] for i in range(len(self.models))]
        self.param_list = [model_factory[self.names[i]]["param_list"] for i in range(len(self.models))]
        print_params(self)

    def set_config(self, device):
        for adapter in self.adapters:
            for ad in adapter:
                ad.set_config(device)
        self.cur_device = device

    def save_weights(self, path):

        torch.save(self.adapters, path + ".adapters.pth")
        for i, model in enumerate(self.models):
            save_lora_weights(model, path + "." + self.names[i])

    def load_weights(self, path, device="cpu"):
        self.adapters.load_weights(path + ".adapters.pth", device)
        for model in self.models:
            model.load_weights(path, device)

    def stitch(self, model_gen, adapters, n_layers, n_skip_layer_pairs, masks, auto_cast, bi_di=True):
        a_skip_layers, b_skip_layers = n_skip_layer_pairs[0]
        out, query, q, kv_x = None, None, None, None
        if len(adapters) > 0:
            adapter = adapters[0]
            mask_a, mask_b = masks[0]
        else:
            adapter = None
            mask_a, mask_b = None, None

        for i in range(n_layers):
            out, query, q = next(model_gen[0])
            if a_skip_layers > 0 and i % a_skip_layers == 0:
                kv_out, kv_x, kv_q = self.stitch(model_gen[1:],
                                                 adapters[1:],
                                                 n_layers=b_skip_layers,
                                                 n_skip_layer_pairs=n_skip_layer_pairs[1:],
                                                 masks=masks[1:],
                                                 auto_cast=auto_cast[1:],
                                                 bi_di=bi_di)

                if bi_di:
                    if auto_cast[1] is not None:
                        with auto_cast[1]:
                            kv_out[0] = kv_out[0] + next(adapter[1])(q=kv_q,
                                                                     kv_x=query,
                                                                     mask=mask_b)
                    else:
                        kv_out[0] = kv_out[0] + next(adapter[1])(q=kv_q,
                                                                 kv_x=query,
                                                                 mask=mask_b)
                    kv_out[1] = i
            if adapter is not None:
                if auto_cast[0] is not None:
                    with auto_cast[0]:
                        out[0] = out[0] + next(adapter[0])(q=q,
                                                           kv_x=kv_x,
                                                           mask=mask_a)
                else:
                    out[0] = out[0] + next(adapter[0])(q=q,
                                                       kv_x=kv_x,
                                                       mask=mask_a)
                out[1] = i
        return out, query, q

    def forward(self, seqs):
        model_gen = []
        unpack_params = []
        for i, model in enumerate(self.models):
            param_names = self.param_list[i]
            params = {}
            for n in param_names:
                params[n] = seqs[n]
            model_gen.append(model(**params))
            unpack_params.append(params)

        adapters = []
        for i in range(len(self.adapters)):
            adapter_fn = []
            for ad in self.adapters[i]:
                adapter_fn.append(ad())
            adapters.append(adapter_fn)

        masks = []
        seq_len = self.seq_len
        for i in range(len(seqs) - 1):
            mask_a, mask_b = create_mask(seq_len[i],
                                         seq_len[i + 1],
                                         device=self.cur_device)
            masks.append([mask_a, mask_b])

        auto_cast = [self.models[i].autocast for i in range(len(self.models))]

        self.stitch(model_gen=model_gen,
                    adapters=adapters,
                    n_layers=self.n_layers,
                    n_skip_layer_pairs=self.n_skip_layer_pairs,
                    masks=masks,
                    auto_cast=auto_cast,
                    bi_di=self.bi_di)
        loss = {}
        for i, model in enumerate(model_gen):
            out = next(model)
            unpack_params[i]["pred"] = out
            loss[self.names[i]] = self.models[i].loss_func(**unpack_params[i])
        return loss
