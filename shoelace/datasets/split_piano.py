import os
import sys

import h5py
import pretty_midi
import numpy as np
from .check_loop_gpu import compute_mean_similarity
from .split_melody import is_mono

device = "cuda"
RES = 50


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


def check(path):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return False, None
    # for instrument in midi_data.instruments:
    #     # instr_program = 128 if instrument.is_drum else instrument.program
    #     for note in instrument.notes:
    #         dur = note.end - note.start
    #         if dur >= 30.:
    #             return False, None

    if midi_data.get_end_time() > 60 * 20:
        print(midi_data.get_end_time())
        return False, None
    try:
        similarity = compute_mean_similarity(midi_data,
                                             window_size=RES * 5, similarity_threshold=0.35, use_cuda=True)
        return True, similarity
    except:
        return False, None


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
    data = []
    if os.path.exists(output_path + ".lst"):
        with open(output_path + ".lst", "r", encoding="utf-8") as f:
            data = f.readlines()
        data = [d.rstrip().split("\t\t") for d in data]
    data = {
        d[0]: d[1] for d in data
    }

    with open(output_path + ".lst", "a") as f:
        for i, midi_path in enumerate(lines):
            if midi_path in data:
                continue
            print(i, "/", len(lines), midi_path)
            try:
                is_valid, sim = check(midi_path)
                if is_valid:
                    f.writelines(f"{midi_path}\t\t{sim}\n")
                    f.flush()
            except:
                print("i am not happy")


if __name__ == "__main__":
    fid = sys.argv[1]
    target_folder = "data/formatted/las/dur_lt_30_with_sim_text"
    file_lst_path = f"data/formatted/las/dur_lt_30_text/{fid}.lst"
    os.makedirs(
        target_folder, exist_ok=True
    )
    output_path = os.path.join(target_folder, f"{fid}.lst")

    process_files(file_lst_path, output_path)
