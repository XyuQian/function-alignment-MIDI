import os
from multiprocessing import Process
import pretty_midi
import numpy as np
import h5py

def midi_to_matrix(path, midi_data, mel_tag, res=50, mode="melody"):
    if midi_data is None:
        try:
            midi_data = pretty_midi.PrettyMIDI(path)
        except:
            return None
    max_time = -1
    outs = []
    melody = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:

            if instrument.name == mel_tag:
                melody.append([note.pitch, note.start, note.end, note.velocity])
            elif mode == "acc":
                outs.append([note.pitch, note.start, note.end, note.velocity])
            if mode == "melody" and instrument.name == mel_tag:
                outs.append([note.pitch, note.start, note.end, note.velocity])
            if max_time < note.end:
                max_time = note.end

    # if max_time < 30:
    #     return None

    max_frame = round(max_time * res) + 2
    onsets = np.zeros([max_frame, 128])
    activations = np.zeros([max_frame, 128])
    melody_mat = np.zeros([max_frame])

    for line in outs:
        pitch, start, end, velocity = line
        start = int(start * res)
        end = int(end * res)
        if start >= end:
            continue
        activations[start:end, int(pitch)] = velocity
        onsets[start, int(pitch)] = 1000 + velocity

    for line in melody:
        pitch, start, end, velocity = line
        start = int(start * res)
        end = int(end * res)
        if start >= end:
            continue
        melody_mat[start:end] = int(pitch)

    activations[onsets > 0] = onsets[onsets > 0]
    return activations, melody_mat


def load_chords(path, chords_dict, res=50, revise_ch=True):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.split("\t") for line in lines]
    chords = np.zeros([int(float(lines[-1][1]) * 50) + 1])
    for line in lines:
        st = int(float(line[0]) * res)
        ed = int(float(line[1]) * res)
        ch = line[2].rstrip()
        if revise_ch and ch not in chords_dict:
            chords_dict[ch] = len(chords_dict)
        chords[st: ed] = chords_dict[ch]
    return chords, chords_dict


def extract_feature_unit(output_path, lines):
    with h5py.File(output_path, "w") as hf:
        for line in lines:
            print(line)
            data = midi_to_matrix(line, res=50)
            if data is None:
                print("error", line)
                continue
            assert True
            hf.create_dataset(line, data=data.astype(np.int16))


def extract_feature(file_list_folder, output_feature_folder, fn):
    ps = []
    for fname in os.listdir(file_list_folder):
        file_list_path = os.path.join(file_list_folder, fname)
        with open(file_list_path, "r") as f:
            lines = f.readlines()
        lines = [line.rstrip().split("\t")[0] for line in lines]
        output_path = os.path.join(output_feature_folder, str.replace(fname, ".lst", ".h5"))
        p = Process(target=fn, args=(output_path, lines))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()


if __name__ == "__main__":
    file_list_folder = "data/formatted/groups/pop909_text"
    output_feature_folder = "data/formatted/groups/pop909_feature"
    # file_list_folder = "data/formatted/groups/las_text"
    # output_feature_folder = "data/formatted/groups/las_feature"
    os.makedirs(file_list_folder, exist_ok=True)
    os.makedirs(output_feature_folder, exist_ok=True)
    extract_feature(file_list_folder, output_feature_folder, extract_feature_unit)
