from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

MODEL_FACTORY = {
    "ScoreLM": {
        "model": MIDILMLora, # Your Score LM class
        "kwargs": {
            "r": 32,
            "lora_alpha": 64,
            "model_path": "exp/midi_lm_continue_phase_1/latest_1_9000.pth"
        },

        "emb_dim": 1024,
        "n_layers": 12,
        "layer_skip": 1, # Process every 1 layer

        "n_indices": 1, # Adapter hyperparameter           
        "low_rank_dim": 64, # Adapter hyperparameter
        "num_heads": 8, # Adapter hyperparameter
        "steps": 1,
        "seq_len": 0,
        "checkpoint_path": None,
        "cond_model_name": "PerformanceLM", # Model it's conditioned ON
    },
    
    "PerformanceLM": {
        "model": MIDILMLora, # Your Performance LM class
        "kwargs": {
            "r": 32,
            "lora_alpha": 64,
            "model_path": "exp/midi_lm_continue_phase_1/latest_1_9000.pth"
        },
        "emb_dim": 1024,
        "n_layers": 12,
        "layer_skip": 1, # Process every 1 layer

        "n_indices": 1, # Adapter hyperparameter           
        "low_rank_dim": 64, # Adapter hyperparameter
        "num_heads": 8, # Adapter hyperparameter
        "steps": 1,
        "seq_len": 0,
        "checkpoint_path": None,
        "cond_model_name": "ScoreLM", # Model it's conditioned ON
    }
}

TASKS = {
    "midi_conversion": {
        "perf_2_score": ["generate_score"], # Task ScoreLM performs when conditioned on PerformanceLM
        "score_2_perf": ["generate_performance"] # Task PerformanceLM performs when conditioned on ScoreLM
    },
}
MODEL_MAPPING = {
    "ScoreLM": "perf_2_score", # When ScoreLM is the main model, it performs the task of generating score
    "PerformanceLM": "score_2_perf" # When PerformanceLM is the main model, it performs the task of generating performance
}