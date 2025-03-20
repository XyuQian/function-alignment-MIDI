#!/usr/bin/env python3
"""
This script overlays a synthesized MIDI melody track onto a non-vocal audio file.
It:
  - Reads the input WAV file to determine its duration.
  - Extracts the melody track from the provided MIDI file (based on the tag "MELODY").
  - Uses FluidSynth (via midi2audio) to render the MIDI to audio.
  - Overlays the synthesized melody audio onto the original audio using pydub.
  - Exports the combined audio as a new WAV file.
  
Requirements:
  - FluidSynth installed on your system.
  - A General MIDI soundfont (e.g., FluidR3_GM.sf2).
  - Python packages: pretty_midi, midi2audio, pydub.
  - ffmpeg installed (for pydub).
  
Usage:
  python add_melody.py <input_wav> <input_midi> [--soundfont SOUND_FONT] [--output OUTPUT_WAV]
"""

import argparse
import os
import tempfile

import pretty_midi
from midi2audio import FluidSynth
from pydub import AudioSegment
from shoelace.datasets.preprocess_midi import analyze_melody

def extract_melody_tracks(midi_file_path):
    """
    Load a MIDI file and return a list containing a new instrument 
    built from the first instrument with 'MELODY' in its name.
    
    Args:
        midi_file_path (str): Path to the MIDI file.
    
    Returns:
        list of pretty_midi.Instrument: A list with one instrument representing the melody.
    """
    # Load the MIDI file using PrettyMIDI
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Filter instruments that have a name containing "MELODY"
    melody_instruments = [
        inst for inst in pm.instruments
        if inst.name and 'MELODY' in inst.name.upper()
    ]
    
    if not melody_instruments:
        raise ValueError("No instrument with 'MELODY' found in the MIDI file.")
    
    # Assuming analyze_melody is defined elsewhere and returns a valid program number.
    program_number = analyze_melody(melody_instruments[0])
    instrument = pretty_midi.Instrument(program=72)
    
    # Copy notes from the first melody instrument into the new instrument.
    for note in melody_instruments[0].notes:
        new_note = pretty_midi.Note(start=note.start,
                                    end=note.end,
                                    velocity=note.velocity,
                                    pitch=note.pitch)
        instrument.notes.append(new_note)
    
    return [instrument]



def synthesize_midi_to_audio(midi_file, output_wav, soundfont):
    """
    Converts a MIDI file to audio using FluidSynth.
    
    Args:
      midi_file (str): Path to the input MIDI file.
      output_wav (str): Path to the output WAV file.
      soundfont (str): Path to the GM soundfont.
    """
    fs = FluidSynth(sound_font=soundfont)
    fs.midi_to_audio(midi_file, output_wav)


def mix_audio_with_vocal(original_wav, vocal_wav, midi_wav, output_wav):
    """
    Overlays the MIDI-generated audio (melody) on the original audio,
    adjusting the loudness of the MIDI track to match that of the vocal audio.
    
    Args:
      original_wav (str): Path to the original audio file.
      midi_wav (str): Path to the synthesized MIDI audio file (WAV format).
      vocal_wav (str): Path to the vocal audio file used as a loudness reference.
      output_wav (str): Path for the output combined audio file.
    """
    # Load the original audio and the vocal reference.
    original = AudioSegment.from_file(original_wav)
    vocal = AudioSegment.from_file(vocal_wav)
    melody_audio = AudioSegment.from_wav(midi_wav)
    
    # Use the vocal file's dBFS for gain adjustment.
    gain_difference = vocal.dBFS - melody_audio.dBFS
    adjusted_melody = melody_audio.apply_gain(gain_difference)
    
    # Ensure the melody track is at least as long as the original.
    if len(adjusted_melody) < len(original):
        adjusted_melody = adjusted_melody * (len(original) // len(adjusted_melody) + 1)
    adjusted_melody = adjusted_melody[:len(original)]
    
    # Overlay the adjusted melody onto the original audio.
    combined = original.overlay(adjusted_melody)
    combined.export(output_wav, format="wav")

def main():
    parser = argparse.ArgumentParser(
        description="Overlay a synthesized MIDI melody track onto non-vocal audio."
    )
    parser.add_argument("input_wav", help="Path to the input non-vocal WAV file.")
    parser.add_argument("input_vocal", help="Path to the input vocal WAV file.")
    parser.add_argument("input_midi", help="Path to the input MIDI file containing the melody track(s).")
    parser.add_argument("--soundfont", default="FluidR3_GM.sf2", 
                        help="Path to a GM soundfont file (default: FluidR3_GM.sf2).")
    parser.add_argument("--output", help="Path to the output WAV file. Defaults to <input_wav>_with_melody.wav")
    args = parser.parse_args()

    # Determine output file name.
    output_wav = args.output if args.output else os.path.splitext(args.input_wav)[0] + "_with_melody.wav"

    # Extract melody instruments from the MIDI file.
    melody_instruments = extract_melody_tracks(args.input_midi)
    if not melody_instruments:
        print("No 'MELODY' track found in the MIDI file.")
        return

    # Create a new PrettyMIDI object containing only the extracted melody instruments.
    melody_pm = pretty_midi.PrettyMIDI()
    melody_pm.instruments.extend(melody_instruments)
    
    # Write the new MIDI to a temporary file in the current directory.
    with tempfile.NamedTemporaryFile(dir=".", suffix=".mid", delete=False) as midi_temp:
        midi_file = midi_temp.name
    melody_pm.write(midi_file)

    # Create a temporary file for the synthesized MIDI audio in the current directory.
    with tempfile.NamedTemporaryFile(dir=".", suffix=".wav", delete=False) as wav_temp:
        midi_wav = wav_temp.name

    # Render the MIDI to audio using FluidSynth.
    synthesize_midi_to_audio(midi_file, midi_wav, args.soundfont)

    # Overlay the melody track with the original audio.
    mix_audio_with_vocal(args.input_wav, args.input_vocal, midi_wav, output_wav)

    print(f"Output saved to: {output_wav}")

    # Clean up temporary files.
    os.remove(midi_file)
    os.remove(midi_wav)


if __name__ == "__main__":
    main()
