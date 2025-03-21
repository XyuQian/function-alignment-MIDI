
import torch
from shoelace.musicgen.finetune.musicgen import MusicGenLora
from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

device = "cuda" if torch.cuda.is_available() else "cpu"
IDX_PAD = 9999



MODEL_FACTORY = {
    "AudioLM": {
        "model": MusicGenLora,
        "kwargs": {
            "device": None,
            "name": "medium",
            "r": 32,
            "lora_alpha": 64,
        },
        
        "n_indices": 4,
        "layer_skip": 1,
        "n_layers": 48,
        "low_rank_dim": 64,
        "emb_dim": 1536,
        "num_heads": 24,
        "steps": 16,
        "checkpoint_path": None,
         "condition_model": "MIDILM"
        
    },
    "MIDILM": {
        "model": MIDILMLora,
        "kwargs": {
            "r": 32,
            "lora_alpha": 64,
            "model_path": "save_models/midi_lm_0309.pth"
        },
        
        "n_indices": 1,
        "layer_skip": 4,
        "n_layers": 12,
        "low_rank_dim": 64,
        "emb_dim": 1024,
        "num_heads": 8,
        "steps": 1,
        "seq_len": 0,
        "checkpoint_path": None,
        "condition_model_name": "AudioLM",
         
    },
}

MASK_TYPE = {
    "midi_2_audio":{
        "MIDILM": False,
        "AudioLM":True
    },
    "audio_2_midi":{
        "MIDILM": True,
        "AudioLM":False
    },
    "bi_di":{
        "MIDILM": True,
        "AudioLM":True
    }

}

