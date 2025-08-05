import os
import glob
import shutil
import pandas as pd
import numpy as np
import pretty_midi
import bisect

def prepare_asap_dataset(path, output_path, segment_length=32):
    df = pd.read_csv(os.path.join(path, 'metadata.csv'))

    score_dir = os.path.join(output_path, 'Score')
    perf_dir = os.path.join(output_path, 'Performance')
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)
    cnt = 0

    for idx, row in df.iterrows():
        idx = f"{idx + 1:03d}"
        composer = row['composer'].replace(" ", "_")
        title = row['title'].replace(" ", "_")
        score_annotation = os.path.join(path, row['midi_score_annotations'])
        perf_annotation = os.path.join(path, row['performance_annotations'])
        score_path = os.path.join(path, row['midi_score'])
        perf_path = os.path.join(path, row['midi_performance'])
        print(f"Processing {idx} - {composer} - {title}")

        score_beats = get_beats(score_annotation)
        perf_beats = get_beats(perf_annotation)
        
        # Segment the beats into n-beat segments
        score_segments = find_segments(score_beats, segment_length)
        perf_segments = find_segments(perf_beats, segment_length)

        # if len(score_segments) < 5 or len(perf_segments) < 5:
        #     print(idx, composer, title)
        #     print(f"Score beats: {len(score_beats)}, Performance beats: {len(perf_beats)}")
        #     print(f"Score segments: {len(score_segments)}, Performance segments: {len(perf_segments)}")
        #     cnt += 1
        
        score_midi_list = segment_midi(score_path, score_segments, mode='score')
        perf_midi_list = segment_midi(perf_path, perf_segments, mode='performance')
        # print(len(score_midi_list), len(perf_midi_list))

        for i, (score_midi, perf_midi) in enumerate(zip(score_midi_list, perf_midi_list)):
            score_midi.write(os.path.join(score_dir, f"{idx}_{i+1:03d}.mid"))
            perf_midi.write(os.path.join(perf_dir, f"{idx}_{i+1:03d}.mid"))
    # print(f"Total files less than 5 segments: {cnt}")


def get_beats(annotation_path):
    beats = [0]
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            beat = float(line.strip().split()[0])
            beats.append(beat)
    return beats


def find_segments(beats, segment_length):
    """
    Find segments of a given length in the list of beats.
    """
    segments = []
    i = 0
    for i in range(0, len(beats) - segment_length, segment_length):
        start = beats[i]
        end = beats[i + segment_length]
        segments.append((i, start, end))

    # Ensure the last segment is included if it has enough beats
    i += segment_length
    if len(beats) - i >= segment_length // 4:
        segments.append((i, beats[i], beats[-1]))
    return segments


def segment_midi(midi_path, segments, mode):
    """
    Segment a MIDI file based on the provided segments.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    seg_midi_list = []
    default_velocity = 100

    for i, start, end in segments:
        seg_midi = pretty_midi.PrettyMIDI()
        
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(inst.program, is_drum=inst.is_drum, name=inst.name)
            for note in inst.notes:
                if note.end > start and note.start < end:
                    new_start = max(note.start, start) - start
                    new_end = min(note.end, end) - start
                    if mode == 'performance':
                        new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=new_start, end=new_end)
                    elif mode == 'score':
                        new_note = pretty_midi.Note(velocity=default_velocity, pitch=note.pitch, start=new_start, end=new_end)
                    else:
                        raise ValueError("Invalid mode. Use 'performance' or 'score'.")
                    if new_start < new_end:
                        new_inst.notes.append(new_note)
            seg_midi.instruments.append(new_inst)
        
        seg_midi_list.append(seg_midi)
    return seg_midi_list


            


if __name__ == "__main__":
    prepare_asap_dataset('raw/asap-dataset', 'data/ASAP', segment_length=32)