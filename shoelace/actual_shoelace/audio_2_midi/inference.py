import os
import sys
import torch.nn.functional as F
import numpy as np
import torch
import librosa
from shoelace.musicgen.finetune.config import FRAME_RATE
from shoelace.datasets.utils import decode
from shoelace.actual_shoelace.audio_2_midi.inference_helper import InferenceHelper
from shoelace.utils.encodec_utils import save_rvq, extract_rvq
device = "cuda"
SEQ_LEN = 512



def get_audio_data(path, chunk_frame, hop_frame, device):
    wav, sr = librosa.load(path, sr=32000)
    x = torch.from_numpy(wav[None, None, ...])
    rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
    for i in range(0, len(rvq_codes), hop_frame):
        ed = i + chunk_frame if i + chunk_frame < len(rvq_codes) else len(rvq_codes)
        audio_chunk = rvq_codes[i : ed]
        index = torch.arange(len(audio_chunk) + 3)
        index = F.pad(index, (1, 0), "constant", 0)
        yield audio_chunk.unsqueeze(0).to(device), index.unsqueeze(0).to(device)
    yield rvq_codes.unsqueeze(0), None



def run_inference(model_folder, output_folder, fid):
    """Runs inference using a trained MIDI language model."""
    chunk_frame = int(FRAME_RATE*15.36)
    hop_frame = int(FRAME_RATE*8)
    model = InferenceHelper(model_folder=model_folder, device=device)
    audio_data_generator = get_audio_data(f"data/POP909/{fid}/{fid}.mid", 
                            chunk_frame=chunk_frame, hop_frame=hop_frame, device=device)

    midi_codes, audio_codes = model.inference(audio_data_generator, 
                chunk_frame=chunk_frame, hop_frame=hop_frame, top_k=150)

    decode(os.path.join(output_folder, f"{fid}.mid"), midi_codes[0].cpu().numpy())
    save_rvq([os.path.join(output_folder, fid)], audio_codes)

if __name__ == "__main__":
    output_folder = "test_results/audio_2_midi"
    os.makedirs(output_folder, exist_ok=True)
    model_id = sys.argv[1]
    fid = sys.argv[2]
    model_folder = f"exp/midi_2_audio_medium/latest_{model_id}_end"
    run_inference(model_folder, output_folder, fid)
