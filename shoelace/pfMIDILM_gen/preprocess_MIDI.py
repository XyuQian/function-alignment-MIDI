import pretty_midi
import numpy as np
import os
import sys
import h5py

SEG_RES = 128
RES_EVENT = 129

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


def analyze_melody(instrument):
    melody_notes = [note.pitch for note in instrument.notes]

    min_pitch = min(melody_notes)
    max_pitch = max(melody_notes)

    best_fit_sax = None
    min_out_of_range = float("inf")

    for sax, (low, high) in saxophone_ranges.items():
        out_of_range_low = max(0, low - min_pitch)  # Notes below the range
        out_of_range_high = max(0, max_pitch - high)  # Notes above the range
        total_out_of_range = out_of_range_low + out_of_range_high

        if total_out_of_range < min_out_of_range:
            min_out_of_range = total_out_of_range
            best_fit_sax = sax

    return saxophone_program[best_fit_sax]


def load_midi(path, extract_melody=False, res=50, return_onset=False, remove_sil=False):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return None
    notes = []
    for instrument in midi_data.instruments:
        instr_program = 128 if instrument.is_drum else instrument.program
        if extract_melody and instrument.name in ["MELODY"]:
            instr_program = analyze_melody(instrument)

        for note in instrument.notes:
            start = int(note.start * res)
            end = int(note.end * res)
            if start >= end:
                continue
            notes.append([start, instr_program, note.pitch, end, note.velocity, instrument.name in ["MELODY"]])

    if len(notes) == 0:
        return None
    notes.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    onset = notes[0][0] if remove_sil else 0
    pre_seg = offset = notes[0][0] if remove_sil else 0
    events = [[SEG_RES, -1, -1, -1, -1, -1]]
    valid_melody_seg = []
    index = [[0, 0]]
    sos = [0]
    res_events = []
    melody_flag = False
    for i, note in enumerate(notes):
        while note[0] - pre_seg >= SEG_RES:
            pre_seg += SEG_RES
            sos.append(len(events))
            valid_melody_seg.append(melody_flag)
            events.append([SEG_RES, -1, -1, -1, -1, -1])
            index.append([pre_seg - offset, pre_seg - offset])
            melody_flag = False

        start = note[0] - pre_seg
        end = note[3] - pre_seg
        end_x = end // SEG_RES
        end_y = end % SEG_RES
        if note[5]:
            melody_flag = True
        assert end_x < SEG_RES
        if end_x > 0:
            start_of_sos = len(sos)
            for j in range(end_x):
                while len(res_events) < start_of_sos + j + 1:
                    res_events.append([])
                res_events[start_of_sos + j].append([RES_EVENT,
                                                     note[1],
                                                     note[2],
                                                     end_x - j - 1,
                                                     end_y,
                                                     note[4]])
        index.append([note[0] - offset, note[3] - offset])
        events.append([start,
                       note[1],
                       note[2],
                       end_x,
                       end_y,
                       note[4]])
    sos.append(len(events))
    valid_melody_seg.append(False)
    res_sos = [0]
    cur_idx = 0
    for res_e in res_events:
        cur_idx += len(res_e)
        res_sos.append(cur_idx)
    res_events = [v for e in res_events for v in e]
    events = np.array(events)

    if extract_melody:
        melody_window_len = 3
        non_sil_idx = 0
        valid_melody_seg = np.asarray(valid_melody_seg)
        for i, has_melody in enumerate(valid_melody_seg):
            if has_melody:
                non_sil_idx = i
                break

        start_idx = -1

        for i in range(non_sil_idx, len(valid_melody_seg)):
            has_melody = valid_melody_seg[i]
            if has_melody:
                if start_idx >= 0:
                    if i - start_idx <= melody_window_len:
                        valid_melody_seg[start_idx: i] = True
                start_idx = -1
            else:
                if start_idx < 0:
                    start_idx = i


    results = {
        "events" : np.asarray(events),
        "sos": np.asarray(sos),
        "res_events": np.asarray(res_events),
        "res_sos": np.asarray(res_sos),
        "index": np.asarray(index),
    }

    if return_onset:
        results["onset"] = onset

    if extract_melody:
        results["valid_melody_seg"] = valid_melody_seg
    # print(valid_melody_seg)
    return results


def extract_feature(file_path_lst, output_feature_path):
    with open(file_path_lst, "r") as f:
        lines = f.readlines()

    for line in lines:
        events, sos, res_events, res_sos, index = load_midi(line)
    return
    lines = [line.rstrip() for line in lines]
    with h5py.File(output_feature_path, "w") as hf:
        for line in lines:
            events, sos, res_events, res_sos, index = load_midi(line)
            if events is not None:
                print(line)
                hf.create_dataset(line + ".events", data=events.astype(np.int16))
                hf.create_dataset(line + ".sos", data=sos.astype(np.int32))
                hf.create_dataset(line + ".res_events", data=res_events.astype(np.int16))
                hf.create_dataset(line + ".res_sos", data=res_sos.astype(np.int32))
                hf.create_dataset(line + ".index", data=index.astype(np.int32))
            # break


if __name__ == "__main__":
    fid = sys.argv[1]
    file_path_lst = f"data/formatted/las/dur_lt_30_text/{fid}.lst"
    output_feature_folder = "data/formatted/las/midis"
    os.makedirs(output_feature_folder, exist_ok=True)
    output_feature_path = os.path.join(output_feature_folder, fid + ".h5")
    extract_feature(file_path_lst, output_feature_path)
