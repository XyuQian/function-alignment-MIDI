import pretty_midi
import numpy as np
import os
import sys
import h5py

from shoelace.midi_lm.config import SEG_RES, RES_EVENT

saxophone_ranges = {
    "Soprano Sax": (60, 81),  # C4 to A5
    "Alto Sax": (50, 69),  # D3 to A4
    "Tenor Sax": (46, 67),  # A2 to G4
    "Baritone Sax": (36, 55),  # C2 to G3
}

saxophone_program = {
    "Soprano Sax": 64,
    "Alto Sax": 65,
    "Tenor Sax": 66,
    "Baritone Sax": 67
}


def analyze_melody(instrument: pretty_midi.Instrument) -> int:
    melody_notes = [note.pitch for note in instrument.notes]
    min_pitch, max_pitch = min(melody_notes), max(melody_notes)

    best_fit_sax, min_out_of_range = None, float("inf")
    for sax, (low, high) in saxophone_ranges.items():
        out_of_range_low = max(0, low - min_pitch)
        out_of_range_high = max(0, max_pitch - high)
        total_out_of_range = out_of_range_low + out_of_range_high

        if total_out_of_range < min_out_of_range:
            min_out_of_range = total_out_of_range
            best_fit_sax = sax

    return saxophone_program[best_fit_sax]


def load_midi(
        path: str,
        extract_melody: bool = False,
        res: int = 50,
        return_onset: bool = False,
        remove_sil: bool = False
):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return None

    notes = []
    for instrument in midi_data.instruments:
        instr_program = 128 if instrument.is_drum else instrument.program
        if extract_melody:
            instr_program = analyze_melody(instrument) if instrument.name in ["MELODY", "Voice"] else 0

        for note in instrument.notes:
            start, end = int(note.start * res), int(note.end * res)
            if start >= end:
                continue
            notes.append([start, instr_program, note.pitch, end, note.velocity, instrument.name in ["MELODY"]])

    if not notes:
        return None

    notes.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    onset, pre_seg, offset = (notes[0][0] if remove_sil else 0,) * 3

    events, valid_melody_seg, sos, res_events = [[SEG_RES, -1, -1, -1, -1, -1]], [], [0], []
    index, melody_flag = [[0, 0]], False

    for note in notes:
        while note[0] - pre_seg >= SEG_RES:
            pre_seg += SEG_RES
            sos.append(len(events))
            valid_melody_seg.append(melody_flag)
            events.append([SEG_RES, -1, -1, -1, -1, -1])
            index.append([pre_seg - offset, pre_seg - offset])
            melody_flag = False

        start, end, end_x, end_y = note[0] - pre_seg, note[3] - pre_seg, (note[3] - pre_seg) // SEG_RES, (
                    note[3] - pre_seg) % SEG_RES
        if note[5]:
            melody_flag = True

        if end_x > 0:
            for j in range(end_x):
                while len(res_events) <= len(sos) + j:
                    res_events.append([])
                res_events[len(sos) + j].append([RES_EVENT, note[1], note[2], end_x - j - 1, end_y, note[4]])

        index.append([note[0] - offset, note[3] - offset])
        events.append([start, note[1], note[2], end_x, end_y, note[4]])

    sos.append(len(events))
    valid_melody_seg.append(False)

    res_sos, cur_idx = [0], 0
    for res_e in res_events:
        cur_idx += len(res_e)
        res_sos.append(cur_idx)
    res_events = [v for e in res_events for v in e]

    if extract_melody:
        melody_window_len, non_sil_idx = 3, next((i for i, v in enumerate(valid_melody_seg) if v), 0)
        start_idx = -1
        for i in range(non_sil_idx, len(valid_melody_seg)):
            if valid_melody_seg[i]:
                if start_idx >= 0 and i - start_idx <= melody_window_len:
                    valid_melody_seg[start_idx: i] = True
                start_idx = -1
            else:
                if start_idx < 0:
                    start_idx = i

    results = {
        "events": np.asarray(events),
        "sos": np.asarray(sos),
        "res_events": np.asarray(res_events),
        "res_sos": np.asarray(res_sos),
        "index": np.asarray(index),
    }
    if return_onset:
        results["onset"] = onset
    if extract_melody:
        results["valid_melody_seg"] = valid_melody_seg

    return results


def extract_feature(file_path_lst: str, output_feature_path: str) -> None:
    with open(file_path_lst, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    with h5py.File(output_feature_path, "w") as hf:
        for line in lines:
            results = load_midi(line)
            if results is None:
                continue
            print(line)
            for key, value in results.items():
                hf.create_dataset(f"{line}.{key}", data=value.astype(np.int16 if "events" in key else np.int32))


if __name__ == "__main__":
    fid = sys.argv[1]
    file_path_lst = f"data/formatted/las/dur_lt_30_text/{fid}.lst"
    output_feature_folder = "data/formatted/las/midis"
    os.makedirs(output_feature_folder, exist_ok=True)
    output_feature_path = os.path.join(output_feature_folder, f"{fid}.h5")
    extract_feature(file_path_lst, output_feature_path)
