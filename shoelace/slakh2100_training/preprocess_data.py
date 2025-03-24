#!/usr/bin/env python3
import os
import sys
import re
import numpy as np
import torch
import librosa
import h5py
from shoelace.datasets.preprocess_midi import load_midi
from shoelace.utils.encodec_utils import extract_rvq

device = "cuda"

#######################################
#           Unicode Utils             #
#######################################

def encode_unicode_string(normal_string):
    """Encode non-ASCII characters as #UXXXX."""
    encoded = ""
    for char in normal_string:
        if ord(char) > 127:
            encoded += f"#U{ord(char):04X}"
        else:
            encoded += char
    return encoded

def decode_unicode_string(unicode_string):
    """Decode #UXXXX sequences to Unicode characters."""
    matches = re.findall(r'#U([0-9A-Fa-f]{4})', unicode_string)
    for match in matches:
        unicode_string = unicode_string.replace(f'#U{match}', chr(int(match, 16)))
    return unicode_string

#######################################
#          HDF5 Helper Functions      #
#######################################

def add_key(hf, key, data):
    """Delete existing key and add new data."""
    if key in hf:
        del hf[key]
    hf[key] = data

def add_midi_data(fid, tag, hf, results):
    """Store MIDI arrays under keys formatted as <fid>.<tag>.<field>."""
    add_key(hf, f"{fid}.midi.{tag}.events", results["events"].astype(np.int16))
    add_key(hf, f"{fid}.midi.{tag}.sos", results["sos"].astype(np.int32))
    add_key(hf, f"{fid}.midi.{tag}.res_events", results["res_events"].astype(np.int16))
    add_key(hf, f"{fid}.midi.{tag}.res_sos", results["res_sos"].astype(np.int32))

#######################################
#         Audio Feature Utils         #
#######################################

def extract_audio_features(path, sr=32000):
    """Load audio and extract RVQ features."""
    wav, _ = librosa.load(path, sr=sr)
    x = torch.from_numpy(wav[None, None, ...])
    rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
    return rvq_codes

def add_audio_features(path, onset, hf, key, sr=32000):
    """Extract audio features starting at onset and store in HDF5."""
    features = extract_audio_features(path, sr)
    add_key(hf, key, features[onset:].astype(np.int16))

#######################################
#           Processing Logic          #
#######################################

def process_data(file_lst_path, data_folder, output_path):
    """
    For each song (line in the file list), process and store:
      - MIDI features for melody, accompaniment, full, chord, and beat.
      - Audio features for original, vocals, non-vocal (acc), chord, and beat.
    """
    with open(file_lst_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip().split("\t") for line in f.readlines()]
    
    with h5py.File(output_path, "a") as hf:
        for midi_path, audio_path in lines:
            print(midi_path, midi_path)

            midi_configs = [
                ("multi-track",   midi_path, {"melody_only": False, "acc_only": False, "extract_melody": False}),
                
            ]
            
            # Process each MIDI configuration.
            midi_results = {}
            for tag, path, params in midi_configs:
                results = load_midi(
                    path,
                    melody_only=params.get("melody_only", False),
                    acc_only=params.get("acc_only", False),
                    extract_melody=params.get("extract_melody", False),
                    return_onset=True,
                    remove_sil=False
                )
                if results is None:
                    print(f"Warning: load_midi failed for {midi_path} tag {tag}")
                    continue
                add_midi_data(midi_path, tag, hf, results)
                midi_results[tag] = results
            
            # Dictionary mapping audio keys to file paths.
            audio_mapping = {
                "multi-track":    audio_path,
            }
            
            # Process and store each audio feature.
            for tag, a_path in audio_mapping.items():
                if os.path.exists(a_path):
                    add_audio_features(a_path, 0, hf, f"{midi_path}.audio.{tag}")
                else:
                    print(f"Audio file not found: {a_path} for {midi_path}")
            
            print(f"Finished processing {midi_path}")
            
        print("Processing complete for all songs.")
        

if __name__ == "__main__":
    tokens_folder = "data/formatted/slakh2100/feature"
    file_lst_path = "data/formatted/slakh2100/text/pair_validation_files.lst"
    os.makedirs(tokens_folder, exist_ok=True)
    output_path = os.path.join(tokens_folder, "validation.h5")
    data_folder = "data/formatted/slakh2100/"
    process_data(file_lst_path, data_folder, output_path)
