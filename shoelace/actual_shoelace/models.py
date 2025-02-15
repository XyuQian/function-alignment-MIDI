import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .cross_attention import LowRankMultiheadAttention
from shoelace.utils.network_utils import freeze, print_params
from shoelace.utils.midi_config import PAD, SEG_RES
from shoelace.utils.midi_utils import make_index
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
    state = {k: v for k, v in network.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save(state, f"{path}.lora.pth")


def get_musicgen(sec, device, is_gen, is_tuned, r=32, lora_alpha=64):
    from shoelace.adapt_musicgen_gen.models.musicgen import MusicGen
    if is_gen else from shoelace.adapt_musicgen.models.musicgen import MusicGen
    mg = MusicGen.get_pretrained(name='large', device=device)
    mg.set_generation_params(duration=sec, extend_stride=16, top_k=250)
    mg.lm.init_qkv()
    config = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=lora_alpha,
                        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                        "self_attn.out_proj"], lora_dropout=0.01)
    mg.lm = get_peft_model(mg.lm, config)
    if not is_tuned:
        freeze(mg.lm)
    return mg


def get_midi_lm(device, is_tuned, is_gen, r=32, lora_alpha=64):
    from shoelace.pfMIDILM.config_1024_8_12_512_8_3 import midi_lm_param, baby_param
    from shoelace.pfMIDILM_gen.MIDILM import MIDILM
    if is_gen else from shoelace.pfMIDILM.MIDILM import MIDILM

    midi_lm = MIDILM(param=midi_lm_param, baby_param=baby_param)
    midi_lm.load_state_dict(torch.load("exp/midi_lm/latest_1_9000.pth", map_location="cpu"))
    midi_lm.set_config(device)
    midi_lm.prepare_for_lora(mode="config")
    target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] + [
        f"transformer_decoder.layers.{i}.self_attn.out_proj" for i in range(midi_lm_param["num_layers"])]
    freeze(midi_lm)

    config = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                        lora_dropout=0.01)
    peft_midi_lm = get_peft_model(midi_lm, config)
    if is_gen or is_tuned:
        freeze(midi_lm.baby_llm, is_freeze=False)
    elif not is_tuned:
        freeze(peft_midi_lm)
    return peft_midi_lm, midi_lm_param["num_layers"]


class MIDILMGEN(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.lm, _ = get_midi_lm(device, is_gen=True, is_tuned=True)
        print_params(self)

    def forward(self, midi_seq):
        return {"loss": self.lm.lora_forward(midi_seq)}

    def inference(self, x, max_len=512, top_k=32, temperature=1.0):
        return self.lm.inference(x, max_len=max_len, top_k=top_k, temperature=temperature)

    def save_weights(self, path):
        save_lora_weights(self, path)
        torch.save(self.lm.baby_llm.state_dict(), f"{path}.baby.pth")

    def load_weights(self, path):
        self.lm.baby_llm.load_state_dict(torch.load(f"{path}.baby.pth", map_location="cpu"))
        self.load_state_dict(torch.load(f"{path}.lora.pth", map_location="cpu"), strict=False)



