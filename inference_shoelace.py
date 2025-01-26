import os
import sys

import librosa
import torch
import numpy as np
import torch.nn.functional as F
import pretty_midi
from shoelace.pfMIDILM.preprocess_MIDI import load_midi, SEG_RES, RES_EVENT
from shoelace.pfMIDILM.MIDILM import PAD

device = "cuda"
SEQ_LEN = 512

TOL_WIN = 5
MAX_DUR = int(15.36 * 50)
SEG_LEN = MAX_DUR // SEG_RES + TOL_WIN


def get_test_data(output_folder):
    from shoelace.utils.encodec_utils import audio_write
    from shoelace.utils.encodec_utils import extract_rvq
    names = [
        # ["data/POP909/909/909.mid",
        #  "data/pop909_audio/909-蓝精灵/original.mp3", 5],
        # ["data/POP909/803/803.mid",
        #  "data/pop909_audio/803-过火/original.mp3", 36],
        # ["data/POP909/860/860.mid",
        #  "data/pop909_audio/860-青城山下白素贞/original.mp3", 0],
        ["data/POP909/757/757.mid",
         "data/pop909_audio/757-菊花台/original.mp3", 37],
        # ["data/POP909/765/765.mid",
        #  "data/pop909_audio/765-虹之间/original.mp3", 56],
        # ["data/POP909/873/873.mid",
        #  "data/pop909_audio/873-飘雪/original.mp3", 1],
        # ["data/POP909/861/861.mid",
        #  "data/pop909_audio/861-青春修炼手册/original.mp3", 1],
        # ["data/POP909/663/663.mid",
        #  "data/pop909_audio/663-眉间雪/original.mp3", 1],
        # ["data/POP909/756/756.mid",
        #  "data/pop909_audio/756-莉莉安/original.mp3", 1],
        # ["data/POP909/164/164.mid",
        #  "data/pop909_audio/164-#U513f#U6b4c/original.mp3", 12]
        # "data/POP909/007/007.mid",
        # "data/POP909/008/008.mid",
        # "data/POP909/009/009.mid",
        # ["data/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs/f/f370a190b7901932cae04037e29ef6cf.mid", 110],
        # ["data/rwc/RM-P001.SMF_SYNC.MID", 120]
    ]

    seq = []
    prompt_len = []
    seq_index = []
    audio_seq = []
    for i, (path, audio_path, pl) in enumerate(names):

        results = load_midi(path,
                            extract_melody=True,
                            return_onset=True)
        # idx = events[:, 1] > 0
        # events[idx, 1] = 0
        # idx = res_events[:, 1] > 0
        # res_events[idx, 1] = 0
        st, events, sos, res_events, res_sos, index, valid_melody_seg = results["onset"], \
                                                                        results["events"], \
                                                                        results["sos"], \
                                                                        results["res_events"], \
                                                                        results["res_sos"], \
                                                                        results["index"], \
                                                                        results["valid_melody_seg"]
        start_idx = 0
        pl = start_idx
        prompt_len.append(pl)
        event_st = sos[start_idx]

        # event_ed = sos[start_idx + SEG_LEN]
        # if event_ed > len(events):
        #     event_ed = len(events)

        res_event_st = res_sos[start_idx]
        res_event_ed = res_sos[start_idx + 1]
        event = events[event_st:]
        # index = index[event_st:event_ed]
        index = index - index[0]

        if res_event_ed > res_event_st:
            prefix = res_events[res_event_st: res_event_ed]
            event = np.concatenate([event[:1], prefix, event[1:]], 0)
            prefix_pos = np.zeros([len(prefix), index.shape[-1]])
            prefix_pos[:, 1] = prefix[:, 3] * SEG_RES + prefix[:, 4]
            index = np.concatenate([index[:1], prefix_pos, index[1:]], 0)

        event[event < 0] = PAD
        seq.append(event)
        seq_index.append(index)
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        # audio_start = int(((pl * SEG_RES + st) / 50) * sr)
        # audio_start = int(128 / 50 * sr)
        audio_start = int(100 / 50 * sr)
        # audio_start = 0
        audio = torch.from_numpy(audio[audio_start:])
        x = audio[None, None, ...]
        rvq_codes = extract_rvq(x, sr).transpose(0, 1)
        audio_seq.append(rvq_codes)
        audio_write(os.path.join(output_folder, str(i)), audio, sr, strategy="loudness", loudness_compressor=True)

    min_len = min([len(s) for s in seq])
    seq = [s[:min_len] for s in seq]
    seq_index = [s[:min_len] for s in seq_index]

    min_len = min([len(s) for s in audio_seq])
    audio_seq = [s[:min_len] for s in audio_seq]
    audio_seq = torch.stack(audio_seq, 0)
    index = torch.from_numpy(np.stack(seq_index, 0))
    index = F.pad(index, (0, 0, 1, 0), "constant", 0)
    # print(index.shape)
    for s in seq:
        print(s.shape, np.max(s), np.min(s))
    return torch.from_numpy(np.stack(seq, 0)), \
           audio_seq, \
           index, \
           prompt_len


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
    # assert events[0][0] == SEG_RES
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
        decode(path, seq[i].cpu().numpy().astype(np.int64))


def inference(output_folder, model_path, mode):
    from shoelace.actual_shoelace.inference_helper import InferenceHelper
    from shoelace.utils.encodec_utils import save_rvq
    model = InferenceHelper(
        models=[
            [model_path, mode]
        ],
        device=device
    )
    midi_seq, audio_seq, midi_index, prompt_len = get_test_data(os.path.join(output_folder, "groundtruth"))
    midi_seq = midi_seq.to(device).long()
    audio_seq = audio_seq.to(device).long()
    midi_index = midi_index.to(device).long()

    res = model.inference(midi_seq=midi_seq,
                          audio_seq=audio_seq,
                          mode=mode)

    if "midi_pred" in res:
        midi_pred = res["midi_pred"]
        folder = os.path.join(output_folder, "pred")
        store_midis(midi_pred, folder)

    if "audio_pred" in res:
        audio_pred = res["audio_pred"]
        save_rvq(output_list=[os.path.join(output_folder, "pred", str(i)) for i in range(len(audio_pred))],
                 tokens=audio_pred.long())

    folder = os.path.join(output_folder, "groundtruth")
    store_midis(midi_seq, folder)


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
    for i in range(10):
        print(seq[0][i])
    store_midis(seq, output_folder)


def test_data_loader(output_folder):
    from shoelace.actual_shoelace.shoelace_dataset import ShoelaceDataset as Dataset
    from shoelace.utils.encodec_utils import save_rvq
    dataset = Dataset(path_folder="data/formatted/",
                      rid=0,
                      is_mono=False,
                      num_workers=0)
    dataset.reset_random_seed(0, 0)

    midi_pred = []
    for i in range(0, 10):
        midi_data, audio_data, midi_index = dataset.__getitem__(i)
        print(midi_index[..., 0])
        print(midi_index[..., 1])
        tokens = torch.from_numpy(audio_data[None, ...]).long().to(device)
        print(tokens.shape)
        tokens = tokens.transpose(1, 2)
        save_rvq(output_list=[os.path.join(output_folder, str(i))],
                 tokens=tokens)
        print(midi_data.shape)
        # for j in range(len(midi_data)):
        #     print(midi_data[j])
        midi = torch.from_numpy(midi_data)
        midi_pred.append(midi)
    store_midis(midi_pred, output_folder)


if __name__ == "__main__":
    mode = sys.argv[1]
    model_path = sys.argv[2]
    output_folder = os.path.join("test_results", mode)
    os.makedirs(output_folder, exist_ok=True)
    # test_data_loader(output_folder)
    inference(output_folder, model_path=model_path, mode=mode)
    # cut_ref(output_folder)
