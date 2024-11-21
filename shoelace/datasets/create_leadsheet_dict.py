import os

import numpy as np

from .MIDI_preprocess import midi_to_matrix, extract_feature, load_chords
import json


def analysis_leedsheet(list_folder):
    chords_dict = {"N": 0}
    leedsheet_dict = {}

    leedsheet_mat = []
    for file_path in os.listdir(list_folder):
        path = os.path.join(list_folder, file_path)
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\t")[0]
                print(line)
                _, melody = midi_to_matrix(line, res=50)
                chord_path = os.path.join("/".join(str.split(line, "/")[:-1]), "chord_midi.txt")
                chords, chords_dict = load_chords(chord_path, chords_dict, res=50)
                chords = chords[:len(melody)]
                leedsheet_mat.append(np.unique(melody * 1000 + chords))
    leedsheet_mat = np.unique(np.concatenate(leedsheet_mat, 0))
    print(len(leedsheet_mat))
    for val in leedsheet_mat:
        leedsheet_dict[str(int(val))] = len(leedsheet_dict)
    print(len(chords_dict))
    with open("data/formatted/groups/chords_dict.json", "w") as outfile:
        json.dump(chords_dict, outfile, indent=4, sort_keys=False)


if __name__ == "__main__":
    file_list_folder = "data/formatted/groups/pop909_text"
    output_feature_folder = "data/formatted/groups/pop909_feature"
    analysis_leedsheet(file_list_folder)
