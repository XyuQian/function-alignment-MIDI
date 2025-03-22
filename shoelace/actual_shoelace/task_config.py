MODEL_MAPPING = {
    "AudioLM": "audio",
    "MIDILM": "midi"
}
TASKS = {
    "multi-tasks": {
        "midi": ["full", "melody", "accompaniment", "beats", "chords"],
        "audio": ["full", "vocals", "accompaniment", "beats", "chords"]
    },
    "complex": {
        "midi": ["full", "melody", "accompaniment", "chords"],
        "audio": ["full", "vocals", "accompaniment", "chords"]
    },
    "mono":{
        "audio": ["vocals"],
        "midi": ["melody"]
    },
    "full":{
        "audio": ["full"],
        "midi": ["full"]
    },
    "mel-acc":{
        "audio": ["full", "accompaniment", "vocals"],
        "midi": ["full", "accompaniment", "melody"]
    }
                
}
