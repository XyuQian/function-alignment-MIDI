
from shoelace.musicgen.finetune.finetune.musicgen import MusicGenLora
from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

MODEL_FACTORY = {
    "AudioLM": {
        "model": MusicGenLora,
        "kwargs": {
            "n_layers": 48,
            "sec": 16,
            "r": 16,
            "lora_alpha": 8,
        },
        "low_rank_dim": 64,
        "hidden_size": 2048,
        "n_heads": 32,
        "steps": 16,
        "emb_dim": 2048,
        "checkpoint_path": None
    },
    "MIDILM": {
        "model": MIDILMLora,
        "kwargs": {
            "n_layers": 12,
            "sec": 16,
            "r": 16,
            "lora_alpha": 8
        },
        "low_rank_dim": 64,
        "hidden_size": 1024,
        "n_heads": 8,
        "steps": 1,
        "seq_len": 0,
        "checkpoint_path": None
    },
}