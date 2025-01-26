import os
import sys
import numpy as np
import re
import torch
import json
from .MIDI_preprocess import midi_to_matrix, load_chords
from .MIDI2tokens import remove_sil, midi_2_token, get_model
from shoelace.utils.encodec_utils import extract_rvq
import librosa
import h5py

device = "cuda"




def decode_unicode_string(unicode_string):
    # Find all occurrences of #UXXXX and replace them with actual Unicode characters
    matches = re.findall(r'#U([0-9A-Fa-f]{4})', unicode_string)
    for match in matches:
        unicode_char = chr(int(match, 16))  # Convert the hexadecimal to an integer, then to a character
        unicode_string = unicode_string.replace(f'#U{match}', unicode_char)
    return unicode_string



def process_data(file_lst_path, output_path):
    with open(file_lst_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [decode_unicode_string(line) for line in lines]
    with open("data/formatted/groups/pop909_chords_dict.json", 'r') as file:
        chords_dict = json.load(file)

    lines = [line.rstrip().split("\t") for line in lines]

    midi_rvq = get_model()
    scale = 16

    with h5py.File(output_path, "a") as hf:
        for midi_path, audio_path in lines:
            if midi_path in ["data/POP909/196/196.mid"]:
                continue
            if midi_path + ".audio.mel" in hf:
                print("skip", midi_path)
                continue
            else:
                print(midi_path)

            melody_data, _ = midi_to_matrix(path=midi_path, midi_data=None, mel_tag="MELODY", res=50, mode="melody")
            acc_data, melody = midi_to_matrix(path=midi_path, midi_data=None, mel_tag="MELODY",  res=50, mode="acc")
            chord_path = os.path.join("/".join(str.split(midi_path, "/")[:-1]), "chord_midi.txt")
            chords, chords_dict = load_chords(chord_path, chords_dict, res=50, revise_ch=False)
            chords = chords[:len(melody)]
            melody = np.stack([melody, chords], -1)

            _, st, ed = remove_sil(acc_data + melody_data)
            acc_midi_tokens = midi_2_token(midi_rvq, acc_data[st:ed])
            melody_midi_tokens = midi_2_token(midi_rvq, melody_data[st:ed])

            melody = melody[st:ed]
            melody = melody[:int(len(melody)) // scale * scale]
            melody = np.reshape(melody, [-1, 16, 2])
            ind = np.argmax(melody[:, :, 0], axis=1) + np.arange(len(melody)) * 16
            melody = melody.reshape(-1, 2)
            melody = melody[ind]

            hf.create_dataset(midi_path + ".acc", data=acc_midi_tokens.astype(np.int16))
            hf.create_dataset(midi_path + ".mel", data=melody_midi_tokens.astype(np.int16))
            hf.create_dataset(midi_path + ".melody", data=melody.astype(np.int16))

            wav, sr = librosa.load(audio_path)
            x = torch.from_numpy(wav[None, None, ...])
            rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
            rvq_codes = rvq_codes[st:ed]
            hf.create_dataset(midi_path + ".audio", data=rvq_codes.astype(np.int16))
            wav, sr = librosa.load(audio_path + ".acc.wav")
            x = torch.from_numpy(wav[None, None, ...])
            rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
            rvq_codes = rvq_codes[st:ed]
            hf.create_dataset(midi_path + ".audio.acc", data=rvq_codes.astype(np.int16))
            wav, sr = librosa.load(audio_path + ".vocals.wav")
            x = torch.from_numpy(wav[None, None, ...])
            rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
            rvq_codes = rvq_codes[st:ed]
            hf.create_dataset(midi_path + ".audio.mel", data=rvq_codes.astype(np.int16))



if __name__ == "__main__":
    fid = sys.argv[1]
    tokens_folder = "data/formatted/groups/pop909_tokens"
    file_lst_path = f"data/formatted/groups/pop909_text/{fid}.lst"
    os.makedirs(
        tokens_folder, exist_ok=True
    )
    output_path = os.path.join(tokens_folder, f"{fid}.h5")

    process_data(file_lst_path, output_path)
