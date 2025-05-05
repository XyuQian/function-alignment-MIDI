import pretty_midi
import numpy as np
import os
import sys
import glob
import h5py
from bisect import bisect_left, bisect_right

from shoelace.midi_lm.models.config import SEG_RES, RES_EVENT

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


def load_score(
        path: str,
        beat_txt: str = None,

        extract_melody: bool = False,
        res: int = 50,
        return_onset: bool = False,
        remove_sil: bool = False,
        melody_only: bool = False,
        acc_only: bool = False,

):
    # 1. Load beat info
    if beat_txt is None:
        return None
    
    beat_times, beat_idx, bar_idx = [], [], []
    with open(beat_txt, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines:
        t = float(line.split()[0])
        beat_times.append(t)

        # beat_idx is a loop of [1, 2, 3, 4]
        beat_idx.append(len(beat_idx) % 4 + 1)
        bar_idx.append(len(bar_idx) // 4 + 1)
    
    beat_duration = (beat_times[-1] - beat_times[0]) / len(beat_times)

    
    # 2. Load midi data and extract notes
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return None
    
    notes = []
    for instrument in midi_data.instruments:
        instr_program = 128 if instrument.is_drum else instrument.program
        if melody_only and not instrument.name == "MELODY":
            continue
        if acc_only and instrument.name == "MELODY":
            continue

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

    # 3. Build events
    for note in notes:
        while note[0] - pre_seg >= SEG_RES:
            pre_seg += SEG_RES
            sos.append(len(events))
            events.append([SEG_RES, -1, -1, -1, -1, -1])

        start, end, end_x, end_y = note[0] - pre_seg, note[3] - pre_seg, (note[3] - pre_seg) // SEG_RES, (
                    note[3] - pre_seg) % SEG_RES
        
        # rest_event: [RES_EVENT, instrument, pitch, , end_y, velocity]
        if end_x > 0:
            for j in range(end_x):
                while len(res_events) <= len(sos) + j:
                    res_events.append([])
                res_events[len(sos) + j].append([RES_EVENT, note[1], note[2], end_x - j - 1, end_y, note[4]])
        
        # event: [start, instrument, pitch, end_x, end_y, velocity]
        events.append([start, note[1], note[2], end_x, end_y, note[4]])
    
    sos.append(len(events))
    valid_melody_seg.append(False)
    
    res_sos, cur_idx = [0], 0
    for res_e in res_events:
        cur_idx += len(res_e)
        res_sos.append(cur_idx)
    res_events = [v for e in res_events for v in e] # flatten res_events

    # 4. Beat Quantization
    num_seg = len(sos) - 1
    seg_start = [i * SEG_RES for i in range(num_seg)]

    def quantize_to_beat(event, seg_idx):
        # event: [start, instrument, pitch, end_x, end_y, velocity]
        # seg_idx: index of the segment

        abs_start = event[0] + seg_start[seg_idx]

        # Find the closest beat time
        pos = bisect_right(beat_times, abs_start) - 1
        if pos < 0:
            beat_offset = 0
        beat_offset = int(abs_start - beat_times[pos])

        # Quantize the event to the closest beat
        total_duration = event[3] * SEG_RES + event[4]
        end_x = int(total_duration // beat_duration)
        end_y = (total_duration / beat_duration) - end_x

        return [beat_offset, event[1], event[2], end_x, round(end_y, 2)*100, event[5]]
    
    new_events = []
    for i in range(num_seg):
        seg_start_idx = sos[i]
        seg_end_idx = sos[i + 1] if i + 1 < num_seg else len(events)
        for j in range(seg_start_idx, seg_end_idx):
            new_events.append(quantize_to_beat(events[j], i))
    
    new_res_events = []
    for e in res_events:
        new_res_events.append(quantize_to_beat(e, 0))
    
    results = {
        "events": np.asarray(new_events),
        "sos": np.asarray(sos),
        "res_events": np.asarray(new_res_events),
        "res_sos": np.asarray(res_sos),
    }

    # results = {
    #     "events": np.asarray(events),
    #     "sos": np.asarray(sos),
    #     "res_events": np.asarray(res_events),
    #     "res_sos": np.asarray(res_sos),
    # }
    if return_onset:
        results["onset"] = onset

    return results


def load_midi(
        path: str,
        extract_melody: bool = False,
        res: int = 50,
        return_onset: bool = False,
        remove_sil: bool = False,
        melody_only: bool = False,
        acc_only: bool = False,

):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return None

    notes = []
    for instrument in midi_data.instruments:
        instr_program = 128 if instrument.is_drum else instrument.program
        if melody_only and not instrument.name == "MELODY":
            continue
        if acc_only and instrument.name == "MELODY":
            continue

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

    for note in notes:
        while note[0] - pre_seg >= SEG_RES:
            pre_seg += SEG_RES
            sos.append(len(events))
            events.append([SEG_RES, -1, -1, -1, -1, -1])

        start, end, end_x, end_y = note[0] - pre_seg, note[3] - pre_seg, (note[3] - pre_seg) // SEG_RES, (
                    note[3] - pre_seg) % SEG_RES

        if end_x > 0:
            for j in range(end_x):
                while len(res_events) <= len(sos) + j:
                    res_events.append([])
                res_events[len(sos) + j].append([RES_EVENT, note[1], note[2], end_x - j - 1, end_y, note[4]])

        events.append([start, note[1], note[2], end_x, end_y, note[4]])

    sos.append(len(events))
    valid_melody_seg.append(False)

    res_sos, cur_idx = [0], 0
    for res_e in res_events:
        cur_idx += len(res_e)
        res_sos.append(cur_idx)
    res_events = [v for e in res_events for v in e]

    results = {
        "events": np.asarray(events),
        "sos": np.asarray(sos),
        "res_events": np.asarray(res_events),
        "res_sos": np.asarray(res_sos),
    }
    if return_onset:
        results["onset"] = onset

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
    # fid = sys.argv[1]
    # file_path_lst = f"data/formatted/las/dur_lt_30_text/{fid}.lst"

    fid = "001"
    file_path_lst = f"data/formatted/rwc/midi_files_{fid}.lst"

    output_feature_folder = f"data/formatted/rwc/midis"
    os.makedirs(output_feature_folder, exist_ok=True)

    output_feature_path = os.path.join(output_feature_folder, f"{fid}.h5")
    # extract_feature(file_path_lst, output_feature_path)

    midi_path = f"data/rwc/AIST.RWC-MDB-P-2001.SMF_SYNC/RM-P{fid}.SMF_SYNC.MID"
    beat_path = f"data/rwc/AIST.RWC-MDB-P-2001.BEAT/RM-P{fid}.BEAT.TXT"
    
    results = load_score(midi_path, beat_path)

    for key, value in results.items():
        print(key, value.shape)
        print(value)

