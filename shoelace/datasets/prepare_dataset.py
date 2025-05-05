import os
import glob
import shutil
import pandas as pd
import numpy as np
import pretty_midi
import bisect

def prepare_asap_dataset(path, output_path):
    df = pd.read_csv(os.path.join(path, 'metadata.csv'))

    score_dir = os.path.join(output_path, 'Score')
    perf_dir = os.path.join(output_path, 'Performance')
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    for idx, row in df.iterrows():
        idx = f"{idx + 1:03d}"
        composer = row['composer'].replace(" ", "_")
        title = row['title'].replace(" ", "_")

        # Copy Score MIDI
        if pd.notna(row['midi_score']):
            score_src = os.path.join(path, row['midi_score'])
            score_dst = os.path.join(score_dir, f"{idx}.mid")
            shutil.copy(score_src, score_dst)

        # Copy Performance MIDIs
        if pd.notna(row['midi_performance']):
            perf_src = os.path.join(path, row['midi_performance'])
            perf_dst = os.path.join(perf_dir, f"{idx}.mid")
            shutil.copy(perf_src, perf_dst)

    print(f"Dataset prepared under {output_path}/")



def prepare_musicnet(path, output_path):
    metadata = pd.read_csv(os.path.join(path, 'musicnet_metadata.csv'))

    score_dir = os.path.join(output_path, 'Score')
    perf_dir = os.path.join(output_path, 'Performance')
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    for _, row in metadata.iterrows():
        id = row['id']
        composer = row['composer']
        seconds = row['seconds']
        print(id, composer, seconds)

        # Annotation
        annotation = pd.read_csv(os.path.join(path, 'annotations', f"{id}.csv"))
        
        # Score MIDI
        score_midi = build_score_midi_with_annotation(annotation)
        score_midi.write(os.path.join(score_dir, f"{id}.mid"))

        # Performance MIDI
        midi_path = glob.glob(os.path.join(path, composer, f"{id}*.mid"))[0]
        shutil.copy(midi_path, os.path.join(perf_dir, f"{id}.mid"))


def build_score_midi_with_annotation(annotation):
    """Load a detailed annotation DataFrame, use start_beat and end_beat as note onsets and offsets
    (given fixed BPM) and output a new Score MIDI with fixed tempo (no tempo changes)."""

    bpm = 120
    beat_duration = 60.0 / bpm  # in seconds
    score_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    
    instrument = {} # program -> pretty_midi.Instrument

    for _, row in annotation.iterrows():
        start_beat = row['start_beat']
        end_beat = row['end_beat']
        pitch = row['note']
        velocity = 80
        program = row['instrument'] - 1
        start_time = start_beat * beat_duration
        end_time = end_beat * beat_duration + start_time
        
        if program not in instrument:
            instrument[program] = pretty_midi.Instrument(program=program)

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        instrument[program].notes.append(note)
    
    for _, inst in instrument.items():
        score_midi.instruments.append(inst)

    # print(score_midi.get_beats())
    # print(score_midi.get_tempo_changes())
    
    return score_midi


def prepare_rwc(path, output_path):
    score_dir = os.path.join(output_path, 'Score')
    perf_dir = os.path.join(output_path, 'Performance')
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    midi_path = os.path.join(path, 'AIST.RWC-MDB-P-2001.SMF_SYNC')
    beat_path = os.path.join(path, 'AIST.RWC-MDB-P-2001.BEAT')

    midi_list = sorted(glob.glob(os.path.join(midi_path, '*.MID')))
    beat_list = sorted(glob.glob(os.path.join(beat_path, '*.BEAT.TXT')))
    assert len(midi_list) == len(beat_list), "Mismatch between MIDI and BEAT files"
    
    for i in range(len(midi_list)):
        idx = i + 1
        idx = f"{idx:03d}"
        print(idx)

        # Performance MIDI
        shutil.copy(midi_list[i], os.path.join(perf_dir, f"{idx}.mid"))

        # Score MIDI
        try:
            score_midi = build_score_midi_with_beat(midi_list[i], beat_list[i])
        except Exception as e:
            print(f"Error processing {midi_list[i]}: {e}")
            continue
        score_midi.write(os.path.join(score_dir, f"{idx}.mid"))


# def find_nearest_beat_index(beat_times, time):
#     """Find the nearest beat index to the given time."""
#     index = bisect.bisect_right(beat_times, time) - 1
#     if index < 0:
#         return 0
#     elif index >= len(beat_times):
#         return len(beat_times) - 1
#     else:
#         left = beat_times[index]
#         right = beat_times[index + 1] if index + 1 < len(beat_times) else left
#         return index if abs(time - left) < abs(time - right) else index + 1
    
def find_lower_beat_index(beat_times, time):
    """Find the largest beat index whose time is less than or equal to the given time."""
    index = bisect.bisect_right(beat_times, time) - 1
    if index < 0:
        return 0
    elif index >= len(beat_times) - 1:
        return len(beat_times) - 2
    return index



def build_score_midi_with_beat(midi_file, beat_file):
    """Load a Performance MIDI file, quantize the note onsets and offsets to the nearest beat
    (given fixed BPM) and output a new Score MIDI with fixed tempo (no tempo changes)."""

    bpm = 120
    beat_duration = 60.0 / bpm  # in seconds
    
    perform_midi = pretty_midi.PrettyMIDI(midi_file)
    score_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    beats = []
    with open(beat_file, 'r') as f:
        for line in f.readlines():
            beat = float(line.strip().split()[0]) / 100.0
            beats.append(beat)
    

    for inst in perform_midi.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)

        for note in inst.notes:
            # Quantize note start and end times to the nearest 1/4 beat
            i_start = find_lower_beat_index(beats, note.start)
            i_end = find_lower_beat_index(beats, note.end)

            frac_start = (note.start - beats[i_start]) / (beats[i_start + 1] - beats[i_start])
            frac_end = (note.end - beats[i_end]) / (beats[i_end + 1] - beats[i_end])

            frac_start = round(frac_start * 4) / 4.0
            frac_end = round(frac_end * 4) / 4.0

            start_time = (i_start + frac_start) * beat_duration
            end_time = (i_end + frac_end) * beat_duration

            if end_time <= start_time:
                end_time = start_time + beat_duration / 16.0  # Ensure at least 1/16 beat duration

            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=start_time, end=end_time)
            new_inst.notes.append(new_note)
        score_midi.instruments.append(new_inst)

    
    # score_num_notes = [len(inst.notes) for inst in score_midi.instruments]
    # perform_num_notes = [len(inst.notes) for inst in perform_midi.instruments]
    # print(f"Score MIDI: {len(score_midi.instruments)} instruments, notes per instrument: {score_num_notes}")
    # print(f"Perform MIDI: {len(perform_midi.instruments)} instruments, notes per instrument: {perform_num_notes}")

    return score_midi



if __name__ == "__main__":
    # prepare_asap_dataset('raw/asap-dataset', 'data/ASAP')
    # prepare_musicnet('raw/musicnet', 'data/MusicNet')
    prepare_rwc('raw/rwc', 'data/RWC')