import os
import sys
import librosa
from shoelace.utils.sep_utils import separate
from shoelace.audiocraft.data.audio import audio_write
import torch
device = "cuda"


def process_data(file_lst_path):
    with open(file_lst_path, "r") as f:
        lines = f.readlines()
    lines = [line.rstrip().split("\t") for line in lines]
    for _, audio_path in lines:
        if os.path.exists(audio_path + ".acc.wav"):
            continue
        print(audio_path)
        wav, _ = librosa.load(audio_path, sr=32000)
        wav = torch.from_numpy(wav[None, None, ...])
        wavs = separate(wav, sample_rate=32000)
        for key in wavs:
            audio_write(audio_path + "." + key, wavs[key].squeeze(0).cpu(), sample_rate=32000, strategy="loudness", loudness_compressor=True)
        acc = sum([wavs[k] for k in wavs if not k == "vocals"])
        audio_write(audio_path + ".acc", acc.squeeze(0).cpu(), sample_rate=32000, strategy="loudness",
                    loudness_compressor=True)





if __name__ == "__main__":
    fid = sys.argv[1]
    tokens_folder = "data/formatted/groups/pop909_tokens"
    file_lst_path = f"data/formatted/groups/pop909_text/{fid}.lst"
    os.makedirs(
        tokens_folder, exist_ok=True
    )
    output_path = os.path.join(tokens_folder, f"{fid}.h5")
    process_data(file_lst_path)
