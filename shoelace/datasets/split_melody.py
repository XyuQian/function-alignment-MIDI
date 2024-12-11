import os
import sys

import pretty_midi
import numpy as np

device = "cuda"
CHRIS_MEL = "CHRIS-MEL"


def midi2_array(instrument, res=50):
    n_max_frames = 10 * 60 * res
    notes = np.zeros([n_max_frames, 128])
    for note in instrument.notes:
        start = int((note.start - 0.05) * res)
        start = 0 if start < 0 else start
        end = int((note.end + 0.05) * res)
        pitch = int(note.pitch)
        notes[start:end, pitch] = 1
    return notes


def is_mono(instrument, res=50):
    n_max_frames = 10 * 60 * res
    notes = np.zeros([n_max_frames, 128])
    for note in instrument.notes:
        start = int(note.start * res)
        end = int(note.end * res)
        pitch = int(note.pitch)
        if start >= end:
            continue
        notes[start:end, pitch] = 1
    total_frames = (notes.sum(-1) > 0).sum()
    ol_frames = (notes.sum(-1) > 1).sum()
    print(instrument.name, ol_frames, total_frames / 10.)
    return ol_frames < total_frames / 10.


def get_active_frames(notes):
    dur = len(notes)
    st = int(.15 * dur)
    ed = int(.85 * dur)
    return (notes[st:ed].sum(-1) > 0).sum()


def extract_melody(path):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return None

    acc = []
    mel = {}

    for instr in midi_data.instruments:
        if instr.is_drum:
            continue
        notes = midi2_array(instr)
        is_mel = -1
        for key in ["Bass", "BASS", "bass"]:
            if key in instr.name:
                is_mel = 0
                break
        for key in ["Melody", "melody", "MELODY", "vocal", "Vocal", "VOCAL"]:
            if key in instr.name:
                is_mel = 1
                break
        if is_mel == 1 or (is_mel == -1 and is_mono(instr)):
            instr.name = instr.name + ".MEL"
            while instr.name in mel:
                instr.name = instr.name + ".1"
            mel[instr.name] = notes
        else:
            acc.append(notes)
    all_trks = acc + [mel[n] for n in mel]
    if len(all_trks) == 0:
        return None
    all_trk = sum(all_trks)
    total_frames = get_active_frames(all_trk)
    if total_frames == 0:
        return None

    print("mono:", [n for n in mel])
    active_trks = {}
    for n in mel:
        n_frames = get_active_frames(mel[n])
        nsil_r = n_frames * 1. / total_frames
        if nsil_r > 0.3:
            active_trks[n] = nsil_r

    if len(active_trks) == 0:
        return None

    mel = {n: mel[n] for n in active_trks}
    print("act:", [f"{n}##{active_trks[n]}" for n in active_trks])

    nonsil = all_trk.sum(-1) > 1
    all_curve = np.argmax(all_trk * (np.arange(128)[None, ...] + 1), -1)
    nonsil[all_curve == 0] = 0

    max_n_high = -1
    target = None
    for n in mel:
        mel[n] = np.argmax(mel[n] * (np.arange(128)[None, ...] + 1), -1)
        nonsil[mel[n] == 0] = 0
    nonsil = nonsil > 0
    all_curve = all_curve[nonsil]
    results = {}
    for n in mel:
        n_high = (mel[n][nonsil] >= all_curve).sum()
        r = n_high / nonsil.sum()
        if r >= .1:
            results[n] = r
            print("r: ", n, r, active_trks[n])
            if active_trks[n] > max_n_high:
                max_n_high = active_trks[n]
                target = n
    print("ori target", target, len(results))
    # if len(results) > 1 and target is not None:
    #     snd_results = {n: results[n] for n in results if not n == target}
    #     max_n_high = -1
    #     for n in snd_results:
    #         if snd_results[n] > max_n_high:
    #             max_n_high = snd_results[n]
    #             target = n
    print("target: ", target)
    print("----------------------------------------------------")
    if target is not None:
        for instr in midi_data.instruments:
            if instr.name == target:
                instr.name = CHRIS_MEL
                break
        return target, midi_data
    return None


def process_files(file_lst_path, output_path):
    with open(file_lst_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]

    outputs = []
    for i, midi_path in enumerate(lines):
        print(len(outputs), i, "/", len(lines), midi_path)
        melody_trk = extract_melody(midi_path)
        if melody_trk is not None:
            outputs.append(midi_path)
            print(melody_trk[0], midi_path)

    with open(output_path, "w") as f:
        f.writelines("\n".join(outputs))


if __name__ == "__main__":
    fid = sys.argv[1]
    target_folder = "data/formatted/las_melody/mel_acc_text_pre"
    file_lst_path = f"data/formatted/las_melody/has_mono_trks/{fid}.lst"
    os.makedirs(
        target_folder, exist_ok=True
    )
    output_path = os.path.join(target_folder, f"{fid}.lst")

    process_files(file_lst_path, output_path)
    # extract_melody("data/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs/5/56896d6f44804c0715f2efbe91dda307.mid")
    # extract_melody(sys.argv[1])
