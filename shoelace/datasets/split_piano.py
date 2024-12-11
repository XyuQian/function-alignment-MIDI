import os
import sys

import pretty_midi
import numpy as np
from .split_melody import is_mono

device = "cuda"


def check_piano(path):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return False

    return len(midi_data.instruments) <= 5


def check_length(path):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return False

    return midi_data.get_end_time() > 2 * 60.





def check_has_mono_trk(path):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return False

    if len(midi_data.instruments) == 1:
        return False

    for instr in midi_data.instruments:
        if instr.is_drum:
            continue
        if "Melody" in instr.name or "melody" in instr.name or "MELODY" in instr.name or is_mono(instr):
            return True

    return False


def process_files(file_lst_path, output_path):
    with open(file_lst_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]

    outputs = []
    for i, midi_path in enumerate(lines):
        print(len(outputs), i, "/", len(lines), midi_path)
        if check_has_mono_trk(midi_path):
            outputs.append(midi_path)

    with open(output_path, "w") as f:
        f.writelines("\n".join(outputs))


if __name__ == "__main__":
    fid = sys.argv[1]
    target_folder = "data/formatted/las_melody/has_mono_trks"
    file_lst_path = f"data/formatted/las_melody/lt_5_trks/{fid}.lst"
    os.makedirs(
        target_folder, exist_ok=True
    )
    output_path = os.path.join(target_folder, f"{fid}.lst")

    process_files(file_lst_path, output_path)
