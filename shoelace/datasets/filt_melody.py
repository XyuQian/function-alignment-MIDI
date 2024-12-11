import os
import sys
import numpy as np
import re
from .split_melody import extract_melody, CHRIS_MEL
from .MIDI_preprocess import midi_to_matrix
from .MIDI2tokens import remove_sil, midi_2_token, get_model

device = "cuda"
import h5py

MAX_FRAME = 10 * 60 * 50



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

    # # lines = [decode_unicode_string(line) for line in lines]
    # with open("data/formatted/groups/pop909_chords_dict.json", 'r') as file:
    #     chords_dict = json.load(file)

    lines = [line.rstrip() for line in lines]

    midi_rvq = get_model()
    scale = 16

    with h5py.File(output_path, "w") as hf:
        for midi_path in lines:
            print(midi_path)
            mel, midi_data = extract_melody(midi_path)
            melody_data, _ = midi_to_matrix(path=None, midi_data=midi_data,
                                            res=50, mode="melody", mel_tag=CHRIS_MEL)
            acc_data, melody = midi_to_matrix(path=None, midi_data=midi_data,
                                              res=50, mode="acc", mel_tag=CHRIS_MEL)
            _, st, ed = remove_sil(acc_data + melody_data)
            acc_midi_tokens = midi_2_token(midi_rvq, acc_data[st:ed])
            melody_midi_tokens = midi_2_token(midi_rvq, melody_data[st:ed])

            melody = melody[st:ed]
            melody = melody[:int(len(melody)) // scale * scale]
            melody = np.reshape(melody, [-1, 16])
            ind = np.argmax(melody, axis=1) + np.arange(len(melody)) * 16
            melody = melody.reshape(-1)
            melody = melody[ind]
            print(melody.shape)

            hf.create_dataset(midi_path + ".acc", data=acc_midi_tokens.astype(np.int16))
            hf.create_dataset(midi_path + ".mel", data=melody_midi_tokens.astype(np.int16))
            hf.create_dataset(midi_path + ".melody", data=melody.astype(np.int16))




if __name__ == "__main__":
    fid = sys.argv[1]
    tokens_folder = "data/formatted/las_melody/tokens"
    file_lst_path = f"data/formatted/las_melody/mel_acc_text/{fid}.lst"
    os.makedirs(
        tokens_folder, exist_ok=True
    )
    output_path = os.path.join(tokens_folder, f"{fid}.h5")

    process_data(file_lst_path, output_path)
