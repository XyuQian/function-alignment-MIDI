MODEL_MAPPING = {
    "AudioLM": "audio",
    "MIDILM": "midi"
}
TASKS = {
    "multi-tasks": {
        "midi": ["full", "melody", "accompaniment", "beats", "chords"],
        "audio": ["full", "vocals", "accompaniment", "beats", "chords"]
    },
    "multi-track": {
        "midi": ["multi-track"],
        "audio": ["multi-track"]
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
