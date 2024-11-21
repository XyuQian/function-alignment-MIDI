import torch
import numpy as np
import pretty_midi

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"



def data2midi(onsets, velocity, output_path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    fs = 50  # Sampling rate (frames per second)

    # Ensure that both matrices have the same shape
    assert onsets.shape == velocity.shape, "Onsets and velocity matrices must have the same shape."

    # Transpose matrices so that time is along the second axis (rows -> pitches, cols -> time)
    onsets = np.transpose(onsets, (1, 0))
    velocity = np.transpose(velocity, (1, 0))

    for pitch in range(onsets.shape[0]):  # Iterate through each pitch
        for start_idx in range(onsets.shape[1]):
            if onsets[pitch, start_idx] == 0:
                continue
            # Find the end of the note (sustain) by looking for the next onset or the end of the sequence
            end_idx = start_idx + 1
            while end_idx < velocity.shape[1] and velocity[pitch, end_idx] > 0 and onsets[pitch, end_idx] == 0:
                end_idx += 1

            if end_idx - start_idx < 3:
                continue

            # Convert frame indices to time
            vel = int(sum(velocity[pitch, start_idx + 1: end_idx]) / (end_idx - start_idx - 1))
            start_time = start_idx / fs
            end_time = (end_idx + 1) / fs

            # Create and append the MIDI note

            note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start_time, end=end_time)
            piano.notes.append(note)

    # print(len(piano.notes))
    midi.instruments.append(piano)
    midi.write(output_path)



def data2midi_without_onsets(velocity, output_path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    fs = 50  # Sampling rate (frames per second)

    # Transpose the velocity matrix so that time is along the second axis (rows -> pitches, cols -> time)
    velocity = np.transpose(velocity, (1, 0))

    for pitch in range(velocity.shape[0]):  # Iterate through each pitch
        start_idx = None
        for idx in range(velocity.shape[1]):
            if velocity[pitch, idx] > 0:
                # If this is the start of a new note
                if start_idx is None:
                    start_idx = idx
            elif start_idx is not None:
                # If we've reached the end of the note
                end_idx = idx
                if end_idx - start_idx >= 2:  # Ignore short notes
                    # Calculate average velocity and convert frame indices to time
                    vel = int(np.mean(velocity[pitch, start_idx:end_idx]))
                    start_time = start_idx / fs
                    end_time = end_idx / fs

                    # Create and append the MIDI note
                    note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start_time, end=end_time)
                    piano.notes.append(note)

                # Reset start_idx for the next note
                start_idx = None

    # Add instrument and write to MIDI file
    midi.instruments.append(piano)
    midi.write(output_path)
