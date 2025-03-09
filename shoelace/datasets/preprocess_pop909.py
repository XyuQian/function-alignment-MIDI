import os
import sys
import numpy as np
import re
import torch
import json
from shoelace.datasets.preprocess_midi import load_midi
from shoelace.utils.encodec_utils import extract_rvq
import librosa
import h5py

device = "cuda"


def encode_unicode_string(normal_string):
    # Replace each character in the string with its #UXXXX representation
    encoded_string = ""
    for char in normal_string:
        if ord(char) > 127:  # Encode non-ASCII characters
            encoded_string += f"#U{ord(char):04X}"
        else:
            encoded_string += char  # Keep ASCII characters as they are
    return encoded_string


def decode_unicode_string(unicode_string):
    # Find all occurrences of #UXXXX and replace them with actual Unicode characters
    matches = re.findall(r'#U([0-9A-Fa-f]{4})', unicode_string)
    for match in matches:
        unicode_char = chr(int(match, 16))  # Convert the hexadecimal to an integer, then to a character
        unicode_string = unicode_string.replace(f'#U{match}', unicode_char)
    return unicode_string


def add_key(hf, dataset_name, data):
    if dataset_name in hf:
        del hf[dataset_name]  # Delete it if it exists
    hf[dataset_name] = data


def add_audio(path, st, hf, key):
    wav, sr = librosa.load(path, sr=32000)
    x = torch.from_numpy(wav[None, None, ...])
    rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
    rvq_codes = rvq_codes[st:]
    if key in hf:
        del hf[key]
    hf.create_dataset(key, data=rvq_codes.astype(np.int16))


def process_data(file_lst_path, output_path):
    with open(file_lst_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [line for line in lines]
    # with open("data/formatted/groups/pop909_chords_dict.json", 'r') as file:
    #     chords_dict = json.load(file)

    lines = [line.rstrip().split("\t") for line in lines]

    with h5py.File(output_path, "a") as hf:
        for midi_path, audio_path in lines:
            if midi_path in ["data/POP909/196/196.mid"]:
                continue
            # if midi_path + ".audio.mel" in hf:
            #     print("skip", midi_path)
            #     continue
            # else:
            #     print(midi_path)
            print(midi_path)
            results = load_midi(midi_path,
                                extract_melody=True,
                                return_onset=True,
                                remove_sil=False)

            if results is None:
                continue
            add_key(hf, midi_path + ".index", results["index"].astype(np.int32))
            add_key(hf, midi_path + ".events", results["events"].astype(np.int16))
            add_key(hf, midi_path + ".sos", results["sos"].astype(np.int32))
            add_key(hf, midi_path + ".res_events", results["res_events"].astype(np.int16))
            add_key(hf, midi_path + ".res_sos", results["res_sos"].astype(np.int32))
            add_key(hf, midi_path + ".valid_melody_seg", results["valid_melody_seg"].astype(np.bool))

            if not os.path.exists(audio_path):
                audio_path = encode_unicode_string(audio_path)
            wav, sr = librosa.load(audio_path)
            x = torch.from_numpy(wav[None, None, ...])
            rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
            st = results["onset"]
            rvq_codes = rvq_codes[st:]
            hf.create_dataset(midi_path + ".audio", data=rvq_codes.astype(np.int16))

            tmp_path = os.path.join(str.replace(audio_path, ".mp3", ""), "vocals.wav")
            add_audio(path=tmp_path, st=st, hf=hf, key=midi_path + ".audio.vocals")

            tmp_path = os.path.join(str.replace(audio_path, ".mp3", ""), "no_vocals.wav")
            add_audio(path=tmp_path, st=st, hf=hf, key=midi_path + ".audio.acc")


if __name__ == "__main__":
    fid = sys.argv[1]
    tokens_folder = "data/formatted/pop909/feature"
    file_lst_path = f"data/formatted/pop909/text/{fid}.lst"
    os.makedirs(
        tokens_folder, exist_ok=True
    )
    output_path = os.path.join(tokens_folder, f"{fid}.h5")

    process_data(file_lst_path, output_path)
