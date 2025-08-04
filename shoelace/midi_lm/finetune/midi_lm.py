import os
import torch
import torch.nn as nn
from types import SimpleNamespace
from peft import LoraConfig, get_peft_model

from ..models.midi_lm import MIDILM
from shoelace.utils.network_utils import print_params, freeze


def prepare_inputs_for_generation(input_ids, **kwargs):
    # Minimal example: return them as-is
    return {"input_ids": input_ids, **kwargs}


class FakeConfig:
    def __init__(self):
        self.model_type = None
        self.peft_type = "task_type"
        self.tie_word_embeddings = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class MIDILMLora(nn.Module):
    """
    A wrapper around MusicGen that applies LoRA to the transformer layers.
    """

    def __init__(self, model_path, r=32, lora_alpha=64, use_generator=False):
        super().__init__()
        # 1) Initialize base MusicGen model on the specified device
        from ..models.config import baby_param, midi_lm_param
        midi_lm = MIDILM(param=midi_lm_param, baby_param=baby_param, use_generator=use_generator)
        midi_lm.load_from_torch_model(model_path)
        # 2) Provide a dummy config so PEFT doesn't crash
        #    model_type can be anything recognized, e.g. "gpt2" or "mpt"
        midi_lm.config = FakeConfig()
        

        # 3) Insert a minimal 'prepare_inputs_for_generation' method
        midi_lm.prepare_inputs_for_generation = prepare_inputs_for_generation

        # 4) Configure LoRA
        target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ]
        for i in range(len(midi_lm.transformer_decoder.layers)):
            target_modules.append(f"transformer_decoder.layers.{i}.self_attn.out_proj")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.02,
        )
        self.lm = get_peft_model(midi_lm, lora_config)
        # freeze(self.lm, True)
        freeze(self.lm.baby_llm, False)
        # print_params(self)


    def set_use_generator(self, flag: bool):
        self.lm.set_use_generator(flag)

    def get_cache(self):
        return self.lm.get_cache()

    def reset_cache(self, reset_sos=True):
        self.lm.reset_cache(reset_sos)

    def decode(self, input_ids):
        return input_ids

    def forward(self, input_ids, **kwargs):
        """
        Forward pass for the LoRA-wrapped model.
        Adjust as needed for your LM signature.
        """
        
        return self.lm(input_ids, **kwargs)

    def inference(self, input_ids, **kwargs):
        return self.lm.inference(input_ids, **kwargs)

    def save_weights(self, path: str):
        """
        Saves only LoRA-related parameters.
        """
        state = self.state_dict()
        os.makedirs(path, exist_ok=True)
        for key in list(state.keys()):
            if "lora_A" not in key and "lora_B" not in key:
                state.pop(key)
        torch.save(state, os.path.join(path, "lora.pth"))
        torch.save(self.lm.baby_llm.state_dict(), os.path.join(path, "baby_llm.pth"))
        print(f"LoRA weights saved to: {path}.lora.pth")

    def load_weights(self, path: str):
        """
        Loads LoRA-only weights from saved .lora.pth file.
        """
        lora_path = os.path.join(path, "lora.pth")
        baby_path = os.path.join(path, "baby_llm.pth")
        self.load_state_dict(torch.load(lora_path, map_location="cpu"), strict=False)
        self.lm.baby_llm.load_state_dict(torch.load(baby_path, map_location="cpu"))
        print(f"LoRA weights loaded from: {path}")
