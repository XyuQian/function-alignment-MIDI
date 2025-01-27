import os
import torch
import numpy as np
import pretty_midi
from shoelace.pfMIDILM_gen.preprocess_MIDI import load_midi, SEG_RES, RES_EVENT
from shoelace.pfMIDILM_gen.MIDILM import PAD

device = "cuda"
SEQ_LEN = 512


def get_test_data(sec=256, res=50):
    names = [
        ["data/POP909/909/909.mid", 2],
        ["data/POP909/803/803.mid", 2],
        ["data/POP909/860/860.mid", 2],
        ["data/POP909/757/757.mid", 2],
        ["data/POP909/765/765.mid", 2],
        ["data/POP909/873/873.mid", 2],
        ["data/POP909/164/164.mid", 2],
        ["data/POP909/007/007.mid", 2],
        ["data/POP909/008/008.mid", 2],
        ["data/POP909/009/009.mid", 2],
        # "data/POP909/164/164.mid",
        # "data/POP909/007/007.mid",
        # "data/POP909/008/008.mid",
        # "data/POP909/009/009.mid",
        # ["data/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs/f/f370a190b7901932cae04037e29ef6cf.mid", 110],
        # ["data/rwc/RM-P001.SMF_SYNC.MID", 120]
    ]

    start_idx = 0
    seq = []
    prompt_len = []
    for i, (path, pl) in enumerate(names):
        prompt_len.append(pl)
        results = load_midi(path,
                            extract_melody=True,
                            return_onset=True)
        st, events, sos, res_events, res_sos, index, valid_melody_seg = results["onset"], \
                                                                        results["events"], \
                                                                        results["sos"], \
                                                                        results["res_events"], \
                                                                        results["res_sos"], \
                                                                        results["index"], \
                                                                        results["valid_melody_seg"]
        event_st = sos[start_idx]
        event_ed = event_st + SEQ_LEN
        if event_ed > len(events):
            event_ed = len(events)
        res_event_st = res_sos[start_idx]
        res_event_ed = res_sos[start_idx + 1]
        event = events[event_st: event_ed]
        # if res_event_ed > res_event_st:
        #     prefix = res_events[res_event_st: res_event_ed]
        #     event = np.concatenate([event[:1], prefix, event[1:]], 0)
        if len(event) > SEQ_LEN:
            event = event[:SEQ_LEN]
        event[event < 0] = PAD
        if len(event) < SEQ_LEN:
            event = np.pad(event, ((0, SEQ_LEN - len(event)), (0, 0)), "constant", constant_values=(PAD, PAD))
        seq.append(event)
    return torch.from_numpy(np.stack(seq, 0)), prompt_len


def add_notes(events, start_pos, instruments):
    start = events[0] + start_pos * SEG_RES
    instr = str(int(events[1]))
    pitch = events[2]
    end_x = events[3] + start_pos
    end_y = events[4]
    velocity = events[5]

    if instr not in instruments:
        instruments[instr] = []
    instruments[instr].append(
        [start, pitch, end_x * SEG_RES + end_y, velocity]
    )


def decode(path, events, res=50):
    assert events[0][0] == SEG_RES
    cur_idx = 1
    instruments = {}
    start_pos = 0
    while cur_idx < len(events) and events[cur_idx][0] == RES_EVENT:
        events[cur_idx][0] = 0
        add_notes(events[cur_idx], start_pos, instruments)
        cur_idx += 1
    while cur_idx < len(events):
        if events[cur_idx][0] in [PAD, RES_EVENT]:
            print(cur_idx, events[cur_idx], RES_EVENT, PAD)
            break
        if events[cur_idx][0] < SEG_RES:
            add_notes(events[cur_idx], start_pos, instruments)
        else:
            start_pos += 1
        cur_idx += 1

    midi = pretty_midi.PrettyMIDI()
    for instr in instruments:
        instr_id = int(instr)
        if instr_id == 128:
            program = pretty_midi.Instrument(program=0)
            program.is_drum = True
        else:
            program = pretty_midi.Instrument(program=instr_id)

        for event in instruments[instr]:
            st = event[0] * 1. / res
            ed = event[2] * 1. / res
            if ed - st < 0.001:
                continue
            note = pretty_midi.Note(velocity=event[3],
                                    pitch=event[1],
                                    start=st,
                                    end=ed)
            program.notes.append(note)

        midi.instruments.append(program)
    midi.write(path)


def store_midis(seq, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(len(seq)):
        path = os.path.join(output_folder, str(i) + ".mid")
        decode(path, seq[i].cpu().numpy())


def inference():
    # from shoelace.actual_shoelace.models import MIDILMGEN
    from shoelace.pfMIDILM.config_1024_8_12_512_8_3 import midi_lm_param, baby_param
    # model = MIDILMGEN(device=device)

    from shoelace.pfMIDILM_gen.MIDILM import MIDILM

    model = MIDILM(param=midi_lm_param,
                  baby_param=baby_param)
    model.load_state_dict(torch.load("exp/midi_lm/latest_0_24000.pth", map_location="cpu"), strict=False)
    # model.load_weights("save_models/piano_lm/latest_39_end.pth")
    model = model.to(device)
    model.set_config(device)
    model.eval()

    seq, prompt_len = get_test_data()
    seq = seq.to(device).long()

    prompt_len = SEQ_LEN // 10
    input_seq = seq[:, :prompt_len]
    res = model.inference(input_seq, max_len=SEQ_LEN, top_k=16, temperature=1.)
    for i in range(prompt_len * 2):
        print(i, prompt_len, "ref", seq[0, i])
        print(i, prompt_len, "ped", res[0, i])
        # print("------------------------------------")
    folder = os.path.join(output_folder, "pred")
    store_midis(res, folder)
    folder = os.path.join(output_folder, "groundtruth")
    store_midis(input_seq, folder)


def cut_ref(output_folder):
    seq, _ = get_test_data()
    seq = [
        seq[6, :60],
        seq[6, :70],
        seq[6, :80],
        seq[6, :90],
        seq[6, :100],
        seq[6, :110],
        seq[6, :120],
    ]
    store_midis(seq, output_folder)


if __name__ == "__main__":
    output_folder = "test_results"
    os.makedirs(output_folder, exist_ok=True)
    inference()
    # cut_ref(output_folder)
