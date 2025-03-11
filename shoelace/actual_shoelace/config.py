
from shoelace.musicgen.finetune.musicgen import MusicGenLora
from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

MODEL_FACTORY = {
    "AudioLM": {
        "model": MusicGenLora,
        "kwargs": {
            "r": 8,
            "lora_alpha": 16,
        },
        "n_layers": 48,
        "low_rank_dim": 64,
        "hidden_size": 2048,
        "n_heads": 32,
        "steps": 16,
        "emb_dim": 2048,
        "checkpoint_path": None
    },
    "MIDILM": {
        "model": MIDILMLora,
        "n_layers": 48,
        "kwargs": {
            "r": 8,
            "lora_alpha": 16
        },
        "n_layers": 12,
        "low_rank_dim": 64,
        "hidden_size": 1024,
        "n_heads": 8,
        "steps": 1,
        "seq_len": 0,
        "checkpoint_path": None
    },
}