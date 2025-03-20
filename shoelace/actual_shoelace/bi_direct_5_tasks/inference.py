import os
import sys
import glob
import torch.nn.functional as F
import numpy as np
import torch
import librosa
import argparse
from shoelace.utils.network_utils import transform_inputs
from shoelace.utils.encodec_utils import save_rvq, extract_rvq
from shoelace.midi_lm.models.config import SEG_RES, PAD
from shoelace.datasets.preprocess_midi import load_midi
from shoelace.musicgen.finetune.config import FRAME_RATE
from shoelace.datasets.utils import decode
from shoelace.actual_shoelace.bi_direct_5_tasks.inference_helper import InferenceHelper

device = "cuda"
SEQ_LEN = 512



def get_midi_data(path, chunk_frame, hop_frame, device, task):
    midi_configs = {
                "melody": {"melody_only": True, "acc_only": False, "extract_melody": True},
                "accompaniment": {"melody_only": False, "acc_only": True, "extract_melody": True},
                "full": {"melody_only": False, "acc_only": False, "extract_melody": True},
                "chords": {"melody_only": False, "acc_only": False, "extract_melody": False},
                "beats": {"melody_only": False, "acc_only": False, "extract_melody": False}
    }
    results = load_midi(path,  melody_only=midi_configs["melody_only"],
                    acc_only=midi_configs["acc_only"],
                    extract_melody=midi_configs["extract_melody"],
                    return_onset=True,
                    remove_sil=False)
                    
    assert results is not None
    events = results["events"]
    sos = results["sos"]
    res_events = results["res_events"]
    res_sos = results["res_sos"]

    chunk_len = int(chunk_frame//SEG_RES)
    hop_len = int(hop_frame//SEG_RES)
    for st_id in range(0, len(sos) - chunk_len, hop_len):
        
        event_st_id = sos[st_id]
        event_ed_id = sos[st_id + chunk_len]
        print(st_id, st_id + chunk_len, event_st_id, event_ed_id)
        if st_id + 1 < len(res_sos):
            res_st_id = res_sos[st_id]
            res_ed_id = res_sos[st_id + 1]
        else:
            res_st_id = res_ed_id = 0
        
        seq = events[event_st_id : event_ed_id]
        if res_st_id < res_ed_id:
            seq = np.concatenate([seq[:1], res_events[res_st_id : res_ed_id], seq[1:]], 0)
        input_ids = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        input_ids[input_ids < 0] = PAD
        midi_index = transform_inputs(input_ids[..., 0], SEG_RES).long().to(device)
        midi_index = F.pad(midi_index, (1, 0), "constant", 0)
        
        yield input_ids, midi_index

    seq = torch.from_numpy(events[:event_ed_id]).unsqueeze(0)
    seq[seq < 0] = PAD
    yield seq, None

def get_audio_data(path, chunk_frame, hop_frame, device, task):
    wav, sr = librosa.load(path, sr=32000)
    x = torch.from_numpy(wav[None, None, ...])
    rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
    for i in range(0, len(rvq_codes), hop_frame):
        ed = i + chunk_frame if i + chunk_frame < len(rvq_codes) else len(rvq_codes)
        audio_chunk = torch.from_numpy(rvq_codes[i : ed])
        index = torch.arange(len(audio_chunk))
        index = F.pad(index, (1, 0), "constant", 0)
        yield audio_chunk.unsqueeze(0).to(device), index.unsqueeze(0).to(device)
    yield torch.from_numpy(rvq_codes).unsqueeze(0).to(device), None



def run_inference_midi_2_audio(model_folder, output_folder, input_path, tasks):
    """Runs inference using a trained MIDI language model."""
    model_folder = os.path.join(model_folder, "midi_2_audio")
    os.makedirs(model_folder, exist_ok=True)
    chunk_frame = int(FRAME_RATE*15.36)
    hop_frame = int(FRAME_RATE*7.68)
    model = InferenceHelper(model_folder=model_folder, device=device)
    midi_data_generator = get_midi_data(input_path, task=tasks[0], 
                            chunk_frame=chunk_frame, hop_frame=hop_frame, device=device)

    generated_codes, input_ids = model.midi_2_audio(midi_data_generator, 
                chunk_frame=chunk_frame, hop_frame=hop_frame, top_k=150, tasks=tasks)

    fname = input_path.split("/")[-1].split(".mid")[0]
    decode(os.path.join(output_folder, f"{fname}.mid"), input_ids[0].cpu().numpy())
    save_rvq([os.path.join(output_folder, fname)], generated_codes)


def run_inference_audio_2_midi(model_folder, output_folder, input_path, tasks):
    """Runs inference using a trained MIDI language model."""
    model_folder = os.path.join(model_folder, "audio_2_midi")
    os.makedirs(model_folder, exist_ok=True)
    chunk_frame = int(FRAME_RATE*15.36)
    hop_frame = int(FRAME_RATE*5.12)
    model = InferenceHelper(model_folder=model_folder, device=device)
    
    audio_data_generator = get_audio_data(input_path, chunk_frame=chunk_frame, 
                                hop_frame=hop_frame, device=device, task=tasks[0])

    midi_codes, audio_codes = model.audio_2_midi(audio_data_generator, tasks=tasks,
                chunk_frame=chunk_frame, hop_frame=hop_frame, top_k=16)
    fname = input_path.split("/")[-1].split(".")[0]
    decode(os.path.join(output_folder, f"{fname}.mid"), midi_codes[0].cpu().numpy())
    save_rvq([os.path.join(output_folder, fname)], audio_codes)

if __name__ == "__main__":
    output_folder = "test_results"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--model_id', type=str, required=True)
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-m', '--midi_mode', type=str, required=True)
    parser.add_argument('-a', '--audio_mode', type=str, required=True)
    
    args = parser.parse_args()
    model_id = args.model_id
    input_path = args.input_path
    task = args.task
    midi_mode = args.midi_mode
    audio_mode = args.audio_mode

    model_folder = f"exp/bi_direct_medium_bi_mask_5_tasks/latest_{model_id}_end"
    if task == "a2m":
        tasks = [audio_mode, midi_mode]
        run_inference_audio_2_midi(model_folder, output_folder, input_path, tasks)
    elif task == "m2a":
        tasks = [midi_mode, audio_mode]
        run_inference_midi_2_audio(model_folder, output_folder, input_path, tasks)