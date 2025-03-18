import os
import sys
import torch.nn.functional as F
import numpy as np
import torch
from shoelace.utils.network_utils import transform_inputs

from shoelace.midi_lm.models.config import SEG_RES, PAD
from shoelace.datasets.preprocess_midi import load_midi
from shoelace.musicgen.finetune.config import FRAME_RATE
from shoelace.datasets.utils import decode
from shoelace.actual_shoelace.midi_2_audio.inference_helper import InferenceHelper
from shoelace.utils.encodec_utils import save_rvq
device = "cuda"
SEQ_LEN = 512



def get_midi_data(path, chunk_frame, hop_frame, device):
    results = load_midi(path, extract_melody=True, return_onset=True)
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



def run_inference(model_folder, output_folder, fid):
    """Runs inference using a trained MIDI language model."""
    chunk_frame = int(FRAME_RATE*15.36)
    hop_frame = int(FRAME_RATE*8)
    model = InferenceHelper(model_folder=model_folder, device=device)
    midi_data_generator = get_midi_data(f"data/POP909/{fid}/{fid}.mid", 
                            chunk_frame=chunk_frame, hop_frame=hop_frame, device=device)

    generated_codes, input_ids = model.inference(midi_data_generator, 
                chunk_frame=chunk_frame, hop_frame=hop_frame, top_k=150)

    decode(os.path.join(output_folder, f"{fid}.mid"), input_ids[0].cpu().numpy())
    save_rvq([os.path.join(output_folder, fid)], generated_codes)

if __name__ == "__main__":
    output_folder = "test_results/midi_2_audio"
    os.makedirs(output_folder, exist_ok=True)
    model_id = sys.argv[1]
    fid = sys.argv[2]
    model_folder = f"exp/midi_2_audio_medium/latest_{model_id}_end"
    run_inference(model_folder, output_folder, fid)
