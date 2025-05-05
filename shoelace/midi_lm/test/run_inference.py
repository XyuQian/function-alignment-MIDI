import os
import sys
import torch
import numpy as np
import pretty_midi
from glob import glob
from tqdm import tqdm

from shoelace.datasets.preprocess_midi import load_midi, SEG_RES, RES_EVENT
from shoelace.midi_lm.models.config import midi_lm_param, baby_param, PAD
from shoelace.midi_lm.test.midi_lm import MIDILM
from shoelace.midi_lm.test.inference import add_notes, decode


device = "cuda"
SEQ_LEN = 1024


def get_test_data(input_folder):
    """Load test MIDI files and return tokenized prompt sequences."""
    
    sequences = []
    for file in os.listdir(input_folder):
        if not file.endswith('.mid'):
            continue
        path = os.path.join(input_folder, file)
        results = load_midi(path, extract_melody=False, return_onset=True)
        if results is None:
            raise ValueError(f"Invalid MIDI file: {path}")
        
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


def run_folder_inference(model_path, input_folder, output_folder):
    """Run inference on a single MIDI file and save generated output."""

    # Load model
    model = MIDILM(param=midi_lm_param, baby_param=baby_param)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.to(device).eval()

    # Load and tokenize the input MIDI
    input_seq, num_samples = get_test_data(input_folder)  # shape: [num_samples, seq_len, 6]
    input_seq = input_seq.to(device).long()

    # Run inference
    generated_seq = model.inference(
        input_seq[:, :128],  # 128 tokens (how many seconds?)
        max_len=SEQ_LEN,
        top_k=16,
        temperature=1.0
    )

    # print(f"Input sequence shape: {input_seq.shape}")
    # print(f"Generated sequence shape: {generated_seq.shape}")

    # Decode and save the output MIDI
    input_midis = sorted(glob(os.path.join(input_folder, '*.mid')))
    for i in range(num_samples):
        file_id = os.path.splitext(os.path.basename(input_midis[i]))[0]
        output_file = os.path.join(output_folder, f"{file_id}_Gen.mid")
        decode(output_file, generated_seq[i].cpu().numpy())



if __name__ == "__main__":

    model_id = 9000
    model_path = f"exp/midi_lm_continue_phase_1/latest_1_{model_id}.pth"

    for dataset in ['ASAP', 'MusicNet', 'RWC']:
        for mode in ['Performance', 'Score']:
            input_folder = f"data/samples/{dataset}/{mode}"
            output_folder = f"inference_results/{dataset}/{mode}/"
            os.makedirs(output_folder, exist_ok=True)

            print(f"Running inference on {input_folder}...")
            run_folder_inference(model_path, input_folder, output_folder)
            print(f"Results saved to {output_folder}")