#!/usr/bin/env python3
import os
import sys
import numpy as np
import re
import torch
import json
import tempfile
import librosa
import h5py
import pretty_midi
from midi2audio import FluidSynth

from shoelace.datasets.preprocess_midi import load_midi
from shoelace.utils.encodec_utils import extract_rvq

device = "cuda"

### Utility functions (unchanged from your previous code)

def encode_unicode_string(normal_string):
    encoded_string = ""
    for char in normal_string:
        if ord(char) > 127:
            encoded_string += f"#U{ord(char):04X}"
        else:
            encoded_string += char
    return encoded_string

def decode_unicode_string(unicode_string):
    matches = re.findall(r'#U([0-9A-Fa-f]{4})', unicode_string)
    for match in matches:
        unicode_char = chr(int(match, 16))
        unicode_string = unicode_string.replace(f'#U{match}', unicode_char)
    return unicode_string

def add_key(hf, dataset_name, data):
    if dataset_name in hf:
        del hf[dataset_name]
    hf[dataset_name] = data

def add_audio(path, st, hf, key):
    wav, sr = librosa.load(path, sr=32000)
    x = torch.from_numpy(wav[None, None, ...])
    rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
    rvq_codes = rvq_codes[st:]
    if key in hf:
        del hf[key]
    hf.create_dataset(key, data=rvq_codes.astype(np.int16))

### Conversion functions for beat/chord TXT files to MIDI
# (Assume these functions are defined as in your previous code.)

def convert_beat_txt_to_midi(beat_txt, output_midi):
    """
    Reads a beat TXT file and creates a MIDI file with drum events.
    Downbeats (beat value 1.0) are mapped to Bass Drum (MIDI 36) and others to Snare (MIDI 38).
    """
    pm = pretty_midi.PrettyMIDI()
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    with open(beat_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        onset = float(parts[0])
        beat_value = float(parts[1])
        note_pitch = 36 if beat_value == 1.0 else 38
        note = pretty_midi.Note(velocity=100, pitch=note_pitch, start=onset, end=onset + 0.1)
        drum_instrument.notes.append(note)
    pm.instruments.append(drum_instrument)
    pm.write(output_midi)
    print(f"Beat MIDI written to {output_midi}")

def convert_chord_txt_to_midi(chord_txt, output_midi, release_epsilon=0.2):
    """
    Reads a chord TXT file and creates a MIDI file with chord events.
    Uses an electric piano (Electric Piano 1, GM program 5 / zero-indexed 4) for a clearer sustain.
    Note durations are extended slightly (release_epsilon) while a sustain pedal on/off event
    is inserted at the chord start/end.
    """
    pm = pretty_midi.PrettyMIDI()
    # Use Electric Piano 1 (GM program 5; zero-indexed 4)
    epiano = pretty_midi.Instrument(program=4)
    with open(chord_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        start_time = float(parts[0])
        end_time = float(parts[1])
        chord_label = parts[2]
        # Insert sustain pedal on at chord start.
        epiano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=start_time)
        )
        midi_notes = parse_chord_to_midi(chord_label)
        for pitch in midi_notes:
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=start_time, end=end_time + release_epsilon)
            epiano.notes.append(note)
        # Insert sustain pedal off at chord end.
        epiano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=0, time=end_time)
        )
    pm.instruments.append(epiano)
    pm.write(output_midi)
    print(f"Chord MIDI written to {output_midi}")

### Revised chord parser (from your previous revision)
def parse_chord_to_midi(chord_str, base_octave=4):
    chord_str = chord_str.strip()
    if chord_str.upper() == "N":
        return []
    m = re.match(r'^([A-G](?:#|b)?)(?::)?(.*)$', chord_str)
    if not m:
        raise ValueError(f"Could not parse chord: {chord_str}")
    root = m.group(1)
    quality = m.group(2).strip().lower()
    quality = quality.replace("(", "").replace(")", "")
    NOTE_OFFSETS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    if root not in NOTE_OFFSETS:
        raise ValueError(f"Unknown root note: {root}")
    root_offset = NOTE_OFFSETS[root]
    root_midi = 12 * (base_octave + 1) + root_offset

    if "minmaj7" in quality:
        intervals = [0, 3, 7, 11]
    elif "min" in quality:
        intervals = [0, 3, 7]
    elif "dim" in quality:
        if "7" in quality:
            if "hdim" in quality or "ø" in quality:
                intervals = [0, 3, 6, 10]
            else:
                intervals = [0, 3, 6, 9]
        else:
            intervals = [0, 3, 6]
    elif "aug" in quality:
        intervals = [0, 4, 8]
    elif "sus2" in quality:
        intervals = [0, 2, 7]
    elif "sus4" in quality:
        intervals = [0, 5, 7]
    else:
        intervals = [0, 4, 7]
    
    if ("maj7" in quality) or (("maj" in quality or quality.startswith("maj/")) and "/7" in quality):
        if 11 not in intervals:
            intervals.append(11)
    if ("min7" in quality) or (("min" in quality or quality.startswith("min/")) and "/7" in quality):
        if 10 not in intervals:
            intervals.append(10)
    if "7" in quality and not any(x in quality for x in ["maj7", "min7", "minmaj7", "dim7"]):
        if 10 not in intervals and 11 not in intervals:
            intervals.append(10)
    if "maj6" in quality or "min6" in quality:
        if 9 not in intervals:
            intervals.append(9)
    if re.search(r'\b9\b', quality):
        if 14 not in intervals:
            intervals.append(14)
    if re.search(r'\b11\b', quality):
        if 17 not in intervals:
            intervals.append(17)
    if re.search(r'\b13\b', quality):
        if 21 not in intervals:
            intervals.append(21)
    if "maj2" in quality or "maj(2)" in quality:
        if 2 not in intervals:
            intervals.append(2)
    if "min2" in quality or "min(2)" in quality:
        if 2 not in intervals:
            intervals.append(2)
    if "#4" in quality:
        if 6 not in intervals:
            intervals.append(6)
    if "b5" in quality:
        if 7 in intervals:
            intervals.remove(7)
        if 6 not in intervals:
            intervals.append(6)
    if "b7" in quality:
        if 10 not in intervals:
            intervals.append(10)
    
    intervals = sorted(set(intervals))
    midi_notes = [root_midi + i for i in intervals]
    return midi_notes

### New function: Extract accompaniment (acc) MIDI
def extract_acc_midi(midi_path, output_path):
    """
    Load the full MIDI file and remove any instrument whose name contains "MELODY"
    (case-insensitive) to obtain an accompaniment MIDI. The resulting MIDI is saved to output_path.
    """
    full_pm = pretty_midi.PrettyMIDI(midi_path)
    # Remove any instrument labeled as melody.
    melody_instr = [instr for instr in full_pm.instruments if instr.name and "MELODY" in instr.name.upper()]
    for instr in melody_instr:
        try:
            full_pm.instruments.remove(instr)
        except ValueError:
            pass
    full_pm.write(output_path)
    print(f"Accompaniment MIDI written to {output_path}")

### Main processing function
def process_data(file_lst_path, output_path):
    with open(file_lst_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.rstrip().split("\t") for line in lines]

    with h5py.File(output_path, "a") as hf:
        for midi_path, audio_path in lines:
            # Skip certain paths if needed.
            if midi_path in ["data/POP909/196/196.mid"]:
                continue

            print("Processing", midi_path)

            # 1. Full MIDI: store the original MIDI file as raw bytes.
            with open(midi_path, "rb") as f:
                full_midi_bytes = f.read()
            add_key(hf, midi_path + ".midi.full", np.void(full_midi_bytes))
            
            # 2. Melody MIDI: extract using load_midi.
            results = load_midi(midi_path, extract_melody=True, return_onset=True, remove_sil=False)
            if results is None:
                continue
            add_key(hf, midi_path + ".midi.melody.index", results["index"].astype(np.int32))
            add_key(hf, midi_path + ".midi.melody.events", results["events"].astype(np.int16))
            add_key(hf, midi_path + ".midi.melody.sos", results["sos"].astype(np.int32))
            add_key(hf, midi_path + ".midi.melody.res_events", results["res_events"].astype(np.int16))
            add_key(hf, midi_path + ".midi.melody.res_sos", results["res_sos"].astype(np.int32))
            add_key(hf, midi_path + ".midi.melody.valid_melody_seg", results["valid_melody_seg"].astype(np.bool))
            onset = results["onset"]

            # 3. Accompaniment (Acc) MIDI: subtract melody instruments.
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_acc:
                acc_midi_path = tmp_acc.name
            extract_acc_midi(midi_path, acc_midi_path)
            with open(acc_midi_path, "rb") as f:
                acc_midi_bytes = f.read()
            add_key(hf, midi_path + ".midi.acc", np.void(acc_midi_bytes))
            os.remove(acc_midi_path)

            # 4. Chord MIDI (if chord TXT exists).
            sample_dir = os.path.dirname(midi_path)
            chord_txt = os.path.join(sample_dir, "chord_audio.txt")
            if os.path.exists(chord_txt):
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_chord:
                    chord_midi_path = tmp_chord.name
                convert_chord_txt_to_midi(chord_txt, chord_midi_path)
                with open(chord_midi_path, "rb") as f:
                    chord_midi_bytes = f.read()
                add_key(hf, midi_path + ".midi.chords", np.void(chord_midi_bytes))
                os.remove(chord_midi_path)
            else:
                print("No chord TXT found for", midi_path)

            # 5. Beat MIDI (if beat TXT exists).
            beat_txt = os.path.join(sample_dir, "beat_audio.txt")
            if os.path.exists(beat_txt):
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_beat:
                    beat_midi_path = tmp_beat.name
                convert_beat_txt_to_midi(beat_txt, beat_midi_path)
                with open(beat_midi_path, "rb") as f:
                    beat_midi_bytes = f.read()
                add_key(hf, midi_path + ".midi.beats", np.void(beat_midi_bytes))
                os.remove(beat_midi_path)
            else:
                print("No beat TXT found for", midi_path)

            # 6. Original audio: extract features from the original audio.
            wav, sr = librosa.load(audio_path, sr=32000)
            x = torch.from_numpy(wav[None, None, ...])
            rvq_codes = extract_rvq(x, sr).transpose(0, 1).cpu().numpy()
            rvq_codes = rvq_codes[onset:]
            add_key(hf, midi_path + ".audio.original", rvq_codes.astype(np.int16))

            # 7. Vocals audio: from vocals.wav in the same folder.
            vocals_path = os.path.join(os.path.dirname(audio_path), "vocals.wav")
            add_audio(path=vocals_path, st=onset, hf=hf, key=midi_path + ".audio.vocals")

            # 8. Non-vocal audio: from no_vocals.wav in the same folder.
            nonvocal_path = os.path.join(os.path.dirname(audio_path), "no_vocals.wav")
            add_audio(path=nonvocal_path, st=onset, hf=hf, key=midi_path + ".audio.nonvocal")

            # 9. Chord audio: synthesize from chord MIDI (if chord TXT exists).
            if os.path.exists(chord_txt):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chord_audio:
                    chord_audio_path = tmp_chord_audio.name
                # Synthesize using FluidSynth.
                fs = FluidSynth(sound_font="data/sf/FluidR3_GM.sf2")
                fs.midi_to_audio(chord_midi_path, chord_audio_path)
                add_audio(path=chord_audio_path, st=onset, hf=hf, key=midi_path + ".audio.chords")
                os.remove(chord_audio_path)

            # 10. Beat audio: synthesize from beat MIDI (if beat TXT exists).
            if os.path.exists(beat_txt):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_beat_audio:
                    beat_audio_path = tmp_beat_audio.name
                fs = FluidSynth(sound_font="data/sf/FluidR3_GM.sf2")
                fs.midi_to_audio(beat_midi_path, beat_audio_path)
                add_audio(path=beat_audio_path, st=onset, hf=hf, key=midi_path + ".audio.beats")
                os.remove(beat_audio_path)

            print("Finished processing", midi_path)
        print("Processing complete for all songs.")

if __name__ == "__main__":
    fid = sys.argv[1]  # e.g., a folder id or grouping identifier
    tokens_folder = "data/formatted/pop909/feature"
    file_lst_path = f"data/formatted/pop909/text/{fid}.lst"
    os.makedirs(tokens_folder, exist_ok=True)
    output_path = os.path.join(tokens_folder, f"{fid}.h5")
    process_data(file_lst_path, output_path)
