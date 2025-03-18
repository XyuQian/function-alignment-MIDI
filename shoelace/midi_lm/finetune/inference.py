import os
import sys
import torch
import numpy as np
import pretty_midi
from shoelace.datasets.preprocess_midi import load_midi, SEG_RES, RES_EVENT
from shoelace.midi_lm.models.config import midi_lm_param, baby_param, PAD
from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

device = "cuda"
SEQ_LEN = 512


def get_test_data():
    """Loads test MIDI data for inference."""
    paths = [
        "data/POP909/909/909.mid",
        "data/POP909/803/803.mid",
        "data/POP909/860/860.mid",
        "data/POP909/757/757.mid"
    ]
    sequences = []

    for path in paths:
        results = load_midi(path, extract_melody=True, return_onset=True)
        if results is None:
            continue
        events = results["events"]
        sos = results["sos"]

        start_idx = 0
        event_start = sos[start_idx]
        event_end = min(event_start + SEQ_LEN, len(events))
        event = events[event_start:event_end]

        event[event < 0] = PAD
        if len(event) < SEQ_LEN:
            event = np.pad(event, ((0, SEQ_LEN - len(event)), (0, 0)), "constant", constant_values=(PAD, PAD))

        sequences.append(event)

    return torch.from_numpy(np.stack(sequences, axis=0)), len(sequences)


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


def save_midi_sequences(sequences, folder):
    """Saves generated sequences as MIDI files."""
    os.makedirs(folder, exist_ok=True)
    for i, seq in enumerate(sequences):
        decode(os.path.join(folder, f"{i}.mid"), seq.cpu().numpy())


def run_inference(model_path, output_folder):
    """Runs inference using a trained MIDI language model."""
    model = MIDILMLora(model_path="save_models/midi_lm_0309.pth")
    model.load_weights(model_path)

    model.to(device).eval()

    input_seq, num_samples = get_test_data()
    input_seq = input_seq.to(device).long()

    generated_seq = model.inference(input_seq[:, :128], max_len=15, top_k=16, temperature=1.0)

    # for i in range(20):
    #     print(generated_seq[0, i + 124])
    #     print(input_seq[0, i + 124])
    #     print("----------------------------")

    save_midi_sequences(generated_seq, os.path.join(output_folder, "generated"))
    save_midi_sequences(input_seq[:, :128], os.path.join(output_folder, "reference"))


if __name__ == "__main__":
    output_folder = "test_results"
    os.makedirs(output_folder, exist_ok=True)
    model_id = sys.argv[1]
    model_path = f"save_models/piano_lm_v1"
    run_inference(model_path, output_folder)
