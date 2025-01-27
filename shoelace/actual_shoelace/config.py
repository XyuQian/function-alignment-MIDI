from .models import AudioLM, MIDILM

MODEL_FACTORY = {
    "AudioLM": {
        "model": AudioLM,
        "model_params": {
            "n_layers": 48,
            "sec": 16,
            "r": 32,
            "alpha_lora": 64,
            # "alpha_lora": 32
        },

        "low_rank_dim": 64,
        "hidden_size": 2048,
        "n_heads": 32,
        "steps": 16,
        "seq_len": 1,
        "param_list": ["audio_seq", "audio_index"],
        "inference_param_list": ["audio_index",
                                 "audio_prompt",
                                 "num_samples",
                                 "audio_top_k",
                                 "audio_max_gen_len",
                                 "audio_walk_steps",
                                 "device"],
        "out": "audio_pred",

    },

    "MIDILM": {
        "model": MIDILM,
        "model_params": {
            "n_layers": 12,
            "sec": 16,
            "r": 32,
            "alpha_lora": 64
        },
        "low_rank_dim": 64,
        "hidden_size": 1024,
        "n_heads": 8,
        "steps": 1,
        "seq_len": 0,
        "param_list": ["midi_seq", "midi_index"],
        "inference_param_list": ["midi_index",
                                 "midi_prompt",
                                 "num_samples",
                                 "midi_top_k",
                                 "midi_walk_steps",
                                 "midi_max_gen_len",
                                 "device"],
        "out": "midi_pred",
    },
}
SKIP_LAYERS = {
    "AudioLM-MIDILM": [4, 1],
    "MIDILM-AudioLM": [1, 4],
}
RECIPE = {
    "mel2vocals":
        {
            "models": ["AudioLM", "MIDILM"],
            "bi": False,
            "loss_weight": [1, 1],
            "model_weights_path": [None, "save_models/piano_lm/latest_39_end.pth"],
        },
    "vocals-mel":
        {
            "models": ["MIDILM", "AudioLM"],
            "bi": True,
            "loss_weight": [1, 1],

        },
    "vocals2mel":
        {
            "models": ["MIDILM", "AudioLM"],
            "bi": False,
            "loss_weight": [1, 1],
            "model_weights_path": [None, "exp/pop_song_musicgen/latest_19_end.pth"],
        },

}
