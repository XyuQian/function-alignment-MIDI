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
from shoelace.pop909_training.inference_helper import InferenceHelper

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
    config = midi_configs[task]
    results = load_midi(path,  melody_only=config["melody_only"],
                    acc_only=config["acc_only"],
                    extract_melody=config["extract_melody"],
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
        
        if st_id + 1 < len(res_sos):
            res_st_id = res_sos[st_id]
            res_ed_id = res_sos[st_id + 1]
        else:
            res_st_id = res_ed_id = 0
        
        seq = events[event_st_id : event_ed_id + 1]
        if res_st_id < res_ed_id:
            seq = np.concatenate([seq[:1], res_events[res_st_id : res_ed_id], seq[1:]], 0)
        input_ids = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        input_ids[input_ids < 0] = PAD
        midi_index = transform_inputs(input_ids[..., 0], SEG_RES).long().to(device) + 1
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
        ed = i + chunk_frame 
        if ed  > len(rvq_codes):
            ed = len(rvq_codes) - 1
        audio_chunk = torch.from_numpy(rvq_codes[i : ed])
        index = torch.arange(i, ed) + 1
        index = F.pad(index, (1, 0), "constant", 0)
        yield audio_chunk.unsqueeze(0).to(device), index.unsqueeze(0).to(device)
    yield torch.from_numpy(rvq_codes[:ed]).unsqueeze(0).to(device), None



def run_inference_midi_2_audio(model, output_folder, input_path, fname, tasks):
    """Runs inference using a trained MIDI language model."""
    
    chunk_frame = int(FRAME_RATE*15.36)
    hop_frame = int(FRAME_RATE*7.68)
    fname = fname + "_" + "2".join(tasks)
    ref_path = os.path.join(output_folder, f"{fname}.mid")
    if os.path.exists(ref_path):
        print("skip", ref_path)
        return
    else:
        print("begin", ref_path)
        return
     
    midi_data_generator = get_midi_data(input_path, task=tasks[0], 
                            chunk_frame=chunk_frame, hop_frame=hop_frame, device=device)

    generated_codes, input_ids = model.midi_2_audio(midi_data_generator, 
                chunk_frame=chunk_frame, hop_frame=hop_frame, top_k=100, tasks=tasks)

    
    decode(os.path.join(output_folder, f"{fname}.mid"), input_ids[0].cpu().numpy())
    save_rvq([os.path.join(output_folder, fname)], generated_codes)


def run_inference_audio_2_midi(model, output_folder, input_path, fname, tasks):
    """Runs inference using a trained MIDI language model."""
    
    chunk_frame = int(FRAME_RATE*12.8)
    hop_frame = int(FRAME_RATE*5.12)
    fname = fname + "_" + "2".join(tasks)
    ref_path = os.path.join(output_folder, fname + ".wav")
    if os.path.exists(ref_path):
        print("skip", ref_path)
        return
    else:
        print("begin", ref_path)
        
    
    audio_data_generator = get_audio_data(input_path, chunk_frame=chunk_frame, 
                                hop_frame=hop_frame, device=device, task=tasks[0])

    midi_codes, audio_codes = model.audio_2_midi(audio_data_generator, tasks=tasks,
                chunk_frame=chunk_frame, hop_frame=hop_frame, top_k=16)
    
    
    decode(os.path.join(output_folder, f"{fname}.mid"), midi_codes[0].cpu().numpy())
    save_rvq([os.path.join(output_folder, fname)], audio_codes)

if __name__ == "__main__":
    output_folder = "test_results"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--model_id', type=str, required=True)
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-tt', '--task_type', type=str, required=True)
    parser.add_argument('-mt', '--model_type', type=str, required=True)
    parser.add_argument('-m', '--midi_mode', type=str, required=True)
    parser.add_argument('-a', '--audio_mode', type=str, required=True)
    parser.add_argument('-n', '--n_prompts', type=int, required=True)
    parser.add_argument('-f', '--filename', type=str, default="test")
    parser.add_argument('-mmt', '--model_task', type=str, required=True)
    
    args = parser.parse_args()
    model_id = args.model_id
    input_path = args.input_path
    task_type = args.task_type
    model_type = args.model_type
    midi_mode = args.midi_mode
    audio_mode = args.audio_mode
    model_task = args.model_task
    n_prompts = int(args.n_prompts)

    model_folder = f"exp/cma_{model_type}_{n_prompts}_{model_task}/latest_{model_id}_end"
    output_folder = os.path.join(output_folder, f"cma_{model_type}_{n_prompts}_{model_task}", 
        f"latest_{model_id}_end", task_type)
    os.makedirs(output_folder, exist_ok=True)

    model = InferenceHelper(model_folder=model_folder, n_prompts=n_prompts, 
                    model_type=model_type, device=device, task_type=model_task)

    
    if task_type == "audio_2_midi":
        tasks = [audio_mode, midi_mode]
        inference_fn = run_inference_audio_2_midi
       
    elif task_type == "midi_2_audio":
        tasks = [midi_mode, audio_mode]
        inference_fn = run_inference_midi_2_audio
    
    if str.endswith(input_path, ".lst"):
        with open(input_path, "r") as f:
            files = f.readlines()
        files = [f.rstrip() for f in files]
        for f in files:
            fname = "_".join(f.split("/"))
            inference_fn(model, output_folder, f, fname, tasks)
            break
            
    else:
        inference_fn(model, output_folder, input_path, args.filename, tasks)