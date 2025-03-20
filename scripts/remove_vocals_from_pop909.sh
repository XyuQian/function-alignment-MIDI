#!/bin/bash
# This script processes the entire POP909 dataset by invoking the original Python script for each sample.
# Dataset structure:
#   Audio: data/pop909_audio/{id}-{songname}/original/no_vocals.wav
#   MIDI:  data/POP909/{id}/{id}.mid
#
# Usage:
#   ./process_dataset.sh
#
# Make sure to update the paths if needed.

# Directories and file paths
AUDIO_DIR="data/pop909_audio"
MIDI_DIR="data/POP909"
SOUNDFONT="data/sf/FluidR3_GM.sf2"
OUTPUT_DIR="data/formatted/pop909_audio_processed"

# Create the output directory if it doesn't exist.
mkdir -p "$OUTPUT_DIR"

# Loop over each audio file in the dataset.
for audio_file in "$AUDIO_DIR"/*/original/no_vocals.wav; do
    # The parent directory is in the format: {id}-{songname}
    parent_dir=$(basename "$(dirname "$(dirname "$audio_file")")")
    # Extract the id (the part before the first hyphen).
    id="${parent_dir%%-*}"
    midi_file="$MIDI_DIR/$id/$id.mid"
    vocal_file="$(dirname "$audio_file")/vocals.wav"
    output_file="$OUTPUT_DIR/${id}_with_melody.wav"
    
    echo "Processing:"
    echo "  Audio:  $audio_file"
    echo "  MIDI:   $midi_file"
    echo "  Vocal:  $vocal_file"

    
    # Invoke the original Python script.
    python -m shoelace.datasets.remove_vocals "$audio_file" "$vocal_file" "$midi_file" --soundfont "$SOUNDFONT"
    
    # Check if the Python script succeeded.
    if [ $? -ne 0 ]; then
        echo "Error processing sample with id $id"
    else
        echo "Successfully processed sample with id $id"
    fi
    
    echo "--------------------------------------"
done

echo "Dataset processing complete."
