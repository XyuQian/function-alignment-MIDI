import pretty_midi
import numpy as np
import os
import sys
import h5py

SEG_RES = 128
RES_EVENT = 129


def load_midi(path, res=50):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return None, None, None, None
    notes = []
    for instrument in midi_data.instruments:
        instr_program = 128 if instrument.is_drum else instrument.program
        for note in instrument.notes:
            start = int(note.start * res)
            end = int(note.end * res)
            if start >= end:
                continue
            notes.append([start, instr_program, note.pitch, end, note.velocity])
    if len(notes) == 0:
        return None, None, None, None
    notes.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    pre_seg = notes[0][0]
    events = [[SEG_RES, -1, -1, -1, -1, -1]]
    sos = []
    res_events = []
    flag = True
    for i, note in enumerate(notes):
        while note[0] - pre_seg >= SEG_RES:
            pre_seg += SEG_RES
            events.append([SEG_RES, -1, -1, -1, -1, -1])
            flag = True
        if flag:
            sos.append(len(events) - 1)
            flag = False
        start = note[0] - pre_seg
        end = note[3] - pre_seg
        end_x = end // SEG_RES
        end_y = end % SEG_RES
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
        events.append([start,
                       note[1],
                       note[2],
                       end_x,
                       end_y,
                       note[4]])
    sos.append(len(events))

    res_sos = [0]
    cur_idx = 0
    for res_e in res_events:
        cur_idx += len(res_e)
        res_sos.append(cur_idx)
    res_events = [v for e in res_events for v in e]

    return np.array(events), np.array(sos), np.array(res_events), np.array(res_sos)


def extract_feature(file_path_lst, output_feature_path):
    with open(file_path_lst, "r") as f:
        lines = f.readlines()

    lines = [line.rstrip() for line in lines]
    with h5py.File(output_feature_path, "w") as hf:
        for line in lines:
            events, sos, res_events, res_sos = load_midi(line)
            if events is not None:
                print(line)
                hf.create_dataset(line + ".events", data=events.astype(np.int16))
                hf.create_dataset(line + ".sos", data=sos.astype(np.int32))
                hf.create_dataset(line + ".res_events", data=res_events.astype(np.int16))
                hf.create_dataset(line + ".res_sos", data=res_sos.astype(np.int32))
            # break


if __name__ == "__main__":
    fid = sys.argv[1]
    file_path_lst = f"data/formatted/las/short_dur_text_beta/{fid}.lst"
    output_feature_folder = "data/formatted/las/midis"
    os.makedirs(output_feature_folder, exist_ok=True)
    output_feature_path = os.path.join(output_feature_folder, fid + ".h5")
    extract_feature(file_path_lst, output_feature_path)
