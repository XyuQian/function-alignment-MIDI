#!/usr/bin/env python3
import argparse
import re
import pretty_midi
from midi2audio import FluidSynth

# -------------------------
# Beat-to-MIDI Conversion
# -------------------------
def convert_beat_txt_to_midi(beat_txt, output_midi):
    """
    Reads a beat TXT file and creates a MIDI file with drum events.
    
    Each line in the TXT file should contain:
        onset_time  beat_value
        
    - If beat_value is 1.0, it is treated as a downbeat and mapped to a bass drum (MIDI note 36).
    - Otherwise, it is mapped to a snare drum (MIDI note 38).
    
    A short drum note (0.1 sec duration) is created at each onset.
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
        # Map downbeat to Bass Drum (MIDI 36) and other beats to Snare (MIDI 38).
        note_pitch = 36 if beat_value == 1.0 else 38
        note = pretty_midi.Note(velocity=100, pitch=note_pitch, start=onset, end=onset + 0.1)
        drum_instrument.notes.append(note)
    
    pm.instruments.append(drum_instrument)
    pm.write(output_midi)
    print(f"Beat MIDI written to {output_midi}")

# -------------------------
# Chord-to-MIDI Conversion
# -------------------------
# Mapping of note names to semitone offsets relative to C.
NOTE_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11
}

def parse_chord_to_midi(chord_str, base_octave=4):
    """
    Parse a chord string (e.g. "F#:maj(9)" or "C:min7/b3") and return a list of MIDI note numbers.
    
    This function attempts to handle many of the chord quality variants found in POP909 by:
      - Extracting the root note (with accidental).
      - Normalizing the quality string (removing parentheses and extra inversions).
      - Using heuristic rules to determine a base triad and then add extensions.
    
    If the chord is "N" (no chord), returns an empty list.
    """
    chord_str = chord_str.strip()
    if chord_str.upper() == "N":
        return []
    
    # Extract root and quality using regex.
    m = re.match(r'^([A-G](?:#|b)?)(?::)?(.*)$', chord_str)
    if not m:
        raise ValueError(f"Could not parse chord: {chord_str}")
    root = m.group(1)
    quality = m.group(2).strip().lower()
    quality = quality.replace("(", "").replace(")", "")
    
    if root not in NOTE_OFFSETS:
        raise ValueError(f"Unknown root note: {root}")
    root_offset = NOTE_OFFSETS[root]
    root_midi = 12 * (base_octave + 1) + root_offset

    # Determine base triad based on keywords.
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
        intervals = [0, 4, 7]  # Default to major triad.
    
    # Handle Extensions and Alterations.
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

def convert_chord_txt_to_midi(chord_txt, output_midi, release_epsilon=0.2):
    """
    Reads a chord TXT file and creates a MIDI file with chord events.
    
    Each line in the chord TXT file should contain:
        start_time  end_time  chord_label
        
    For each chord event (if chord_label is not "N"), sustained notes are added
    (one for each note in the chord) to an electric piano instrument. Additionally, 
    sustain pedal control changes (MIDI CC 64) are added to ensure that the chord
    is held for its full intended duration.
    
    To ensure the audible chord stops exactly at end_time, each note's end time is
    extended by a small epsilon (default: 0.2 sec). The sustain pedal is turned off
    at the specified end_time, so when FluidSynth processes the MIDI, the chord will 
    be held (via the pedal) only until end_time.
    """
    pm = pretty_midi.PrettyMIDI()
    # Create an electric piano instrument (Electric Piano 1, GM program 5, zero-indexed is 4).
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
        
        # Add sustain pedal "on" event at chord start.
        epiano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=start_time)
        )
        
        midi_notes = parse_chord_to_midi(chord_label)
        for pitch in midi_notes:
            # Extend the note's end time slightly to allow for proper sustain behavior.
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time + release_epsilon)
            epiano.notes.append(note)
        
        # Add sustain pedal "off" event exactly at chord end time.
        epiano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=0, time=end_time)
        )
    
    pm.instruments.append(epiano)
    pm.write(output_midi)
    print(f"Chord MIDI written to {output_midi}")





# -------------------------
# Main Command-Line Interface
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert beat/chord TXT file to a MIDI file and optionally synthesize audio using FluidSynth."
    )
    parser.add_argument("mode", choices=["beat", "chord"],
                        help="Select conversion mode: 'beat' or 'chord'")
    parser.add_argument("input_txt", help="Path to the input TXT file")
    parser.add_argument("output_midi", help="Path to the output MIDI file")
    parser.add_argument("--audio", help="Optional: Path to save the synthesized audio (WAV format)")
    parser.add_argument("--soundfont", default="data/sf/FluidR3_GM.sf2",
                        help="Path to the GM soundfont (default: data/sf/FluidR3_GM.sf2)")
    args = parser.parse_args()

    if args.mode == "beat":
        convert_beat_txt_to_midi(args.input_txt, args.output_midi)
    elif args.mode == "chord":
        convert_chord_txt_to_midi(args.input_txt, args.output_midi)

    # If an audio output is requested, synthesize audio from the generated MIDI.
    if args.audio:
        fs = FluidSynth(sound_font=args.soundfont)
        fs.midi_to_audio(args.output_midi, args.audio)
        print(f"Audio synthesized and written to {args.audio}")
