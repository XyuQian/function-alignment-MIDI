#!/bin/bash
# This script processes the entire POP909 dataset by converting both chord and beat TXT files 
# into MIDI and synthesized audio (WAV) files using the synthesize_chord_beat module.
#
# Expected dataset structure:
#   data/POP909/<sample_id>/beat_audio.txt
#   data/POP909/<sample_id>/chord_audio.txt
#
# Outputs are saved to:
#   Chords: data/formatted/pop909/chords/<sample_id>.mid and <sample_id>.wav
#   Beats:  data/formatted/pop909/beats/<sample_id>.mid and <sample_id>.wav
#
# Usage:
#   ./process_dataset.sh

# Directories
DATA_DIR="data/POP909"
CHORD_OUTPUT_DIR="data/formatted/pop909/chords"
BEAT_OUTPUT_DIR="data/formatted/pop909/beats"
SOUNDFONT="data/sf/FluidR3_GM.sf2"

# Create output directories if they don't exist.
mkdir -p "$CHORD_OUTPUT_DIR"
mkdir -p "$BEAT_OUTPUT_DIR"

# Loop over each sample directory in DATA_DIR
for sample_dir in "$DATA_DIR"/*; do
    sample_id=$(basename "$sample_dir")
    echo "Processing sample: $sample_id"
    
    # Process chord file (if available)
    chord_txt="$sample_dir/chord_audio.txt"
    if [ -f "$chord_txt" ]; then
        chord_midi="$CHORD_OUTPUT_DIR/${sample_id}.mid"
        chord_audio="$CHORD_OUTPUT_DIR/${sample_id}.wav"
        echo "  Processing chords..."
        python -m shoelace.datasets.synthesize_chord_beat chord "$chord_txt" "$chord_midi" --audio "$chord_audio" --soundfont "$SOUNDFONT"
    else
        echo "  No chord file found for sample $sample_id"
    fi

    # Process beat file (if available)
    beat_txt="$sample_dir/beat_audio.txt"
    if [ -f "$beat_txt" ]; then
        beat_midi="$BEAT_OUTPUT_DIR/${sample_id}.mid"
        beat_audio="$BEAT_OUTPUT_DIR/${sample_id}.wav"
        echo "  Processing beats..."
        python -m shoelace.datasets.synthesize_chord_beat beat "$beat_txt" "$beat_midi" --audio "$beat_audio" --soundfont "$SOUNDFONT"
    else
        echo "  No beat file found for sample $sample_id"
    fi

    echo "--------------------------------------"
done

echo "Dataset processing complete."
