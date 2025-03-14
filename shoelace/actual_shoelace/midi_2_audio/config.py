import torch
from shoelace.musicgen.finetune.musicgen import MusicGenLora
from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FACTORY = {
    "AudioLM": {
        "model": MusicGenLora,
        "kwargs": {
            "device": None,
            "name": "small",
            "r": 8,
            "lora_alpha": 16,
        },
        "layer_skip": 1,
        "n_layers": 24,
        "low_rank_dim": 64,
        "emb_dim": 1024,
        "num_heads": 16,
        "steps": 16,
        "checkpoint_path": None
    },
    "MIDILM": {
        "model": MIDILMLora,
        "kwargs": {
            "r": 8,
            "lora_alpha": 16,
            "model_path": "save_models/midi_lm_0309.pth"
        },
        "layer_skip": 2,
        "n_layers": 12,
        "low_rank_dim": 64,
        "emb_dim": 1024,
        "num_heads": 8,
        "steps": 1,
        "seq_len": 0,
        "checkpoint_path": "save_models/midi_lm_piano_cover"
    },
}