import torch
import torch.nn as nn
from types import SimpleNamespace
from peft import LoraConfig, get_peft_model

from ..models.musicgen import MusicGen
from shoelace.utils.network_utils import print_params



def reformat(state):
    res = {}
    for key in state:
        if str.startswith(key, "lm.base_model.model.transformer"):
            k = str.replace(key, "lm.base_model.model.transformer", "lm.base_model.model.lm.transformer")
            res[k] = state[key]
        else:
            res[key] = state[key]
    return res


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


class MusicGenLora(nn.Module):
    """
    A wrapper around MusicGen that applies LoRA to the transformer layers.
    """

    def __init__(self, name, device, r=32, lora_alpha=64, use_generator=False):
        super().__init__()
        # 1) Initialize base MusicGen model on the specified device
        musicgen = MusicGen(name=name, device=device, use_generator=use_generator)

        # 2) Provide a dummy config so PEFT doesn't crash
        #    model_type can be anything recognized, e.g. "gpt2" or "mpt"
        musicgen.config = FakeConfig()

        # 3) Insert a minimal 'prepare_inputs_for_generation' method
        musicgen.prepare_inputs_for_generation = prepare_inputs_for_generation

        # 4) Configure LoRA
        target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ]
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.02,
        )
        # 5) Convert musicgen to a LoRA-enabled model

        self.lm = get_peft_model(musicgen, lora_config)
        print_params(self)

    def set_use_generator(self, flag: bool):
        self.lm.set_use_generator(flag)

    def get_cache(self):
        return self.lm.get_cache()

    def reset_cache(self, reset_sos=True):
        self.lm.reset_cache(reset_sos)

    def decode(self, input_ids):
        return self.lm.decode(input_ids)

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
        for key in list(state.keys()):
            if "lora_A" not in key and "lora_B" not in key:
                state.pop(key)
        torch.save(state, path + ".lora.pth")
        print(f"LoRA weights saved to: {path}.lora.pth")

    def load_weights(self, path: str, strict: bool = False):
        """
        Loads LoRA-only weights from saved .lora.pth file.
        """
        
        lora_state = torch.load(path + ".lora.pth", map_location="cpu")
        lora_state = reformat(lora_state)
        self.load_state_dict(lora_state, strict=strict)
        print(f"LoRA weights loaded from: {path}.lora.pth")

    
