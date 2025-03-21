
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
        "tasks": ["vocals", "accompaniment", "beats", "chords", "full"]
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
        "tasks": ["melody", "accompaniment", "beats", "chords", "full"]
    },
}

MODEL_PAIRS = {
    "midi_2_audio":{
        "MIDILM":{
            "is_freeze": False,
            "condition_model": None
        },
        "AudioLM":{
            "is_freeze": False,
            "condition_model": "MIDILM"
        }
    },
    "audio_2_midi":{
        "MIDILM":{
            "is_freeze": False,
            "condition_model": "AudioLM"
        },
        "AudioLM":{
            "is_freeze": False,
            "condition_model": None
        }
    },
    "bi_di":{
        "MIDILM":{
            "is_freeze": False,
            "condition_model": "AudioLM"
        },
        "AudioLM":{
            "is_freeze": False,
            "condition_model": "MIDILM"
        }
    }

}

