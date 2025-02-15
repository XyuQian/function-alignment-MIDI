SEG_RES = 128
RES_EVENT = 129
SOS = 130
PAD = 131
N_ONSET = 132
N_INSTRUMENT = 132
N_PITCH = 132
N_DUR_X = 132
N_DUR_Y = 132
N_VELOCITY = 132
MAX_SEQ_LEN = 1500



midi_lm_param = {
    "embedding_dim": 1024,
    "num_heads": 8,
    "num_layers": 12,
}

baby_param = {
    "n_words": 132,
    "memory_dim": 1024,
    "n_steps": 6,
    "embedding_dim": 512,
    "n_layers": 3,
    "n_heads": 8,
}
