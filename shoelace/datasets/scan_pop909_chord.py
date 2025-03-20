#!/usr/bin/env python3
"""
Script to scan all chord qualities in the dataset and save them.

It searches for chord TXT files in the specified input directory (e.g., data/POP909),
extracts the chord quality (the part after ':' in chord labels, ignoring "N" for no chord),
counts each occurrence, and then writes the sorted results (by frequency) to an output file.

Usage:
  python scan_chord_qualities.py --input-dir data/POP909 --output chord_qualities.txt
"""

import argparse
import glob
import os
from collections import Counter

def scan_chord_qualities(input_dir):
    """
    Scan chord TXT files in the given directory (non-recursive within one level)
    and extract chord qualities.
    
    Args:
        input_dir (str): Root directory where chord TXT files are located.
        
    Returns:
        Counter: A Counter object mapping chord quality to its occurrence count.
    """
    # Assumes chord TXT files are in subdirectories: <input_dir>/*/chord_audio.txt
    chord_files = glob.glob(os.path.join(input_dir, "*", "chord_audio.txt"))
    quality_counter = Counter()
    
    for chord_file in chord_files:
        with open(chord_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                chord_label = parts[2].strip()
                # Skip "N" (no chord) events.
                if chord_label.upper() == "N":
                    continue
                # Expect chord label to be of the form "Root:Quality" (e.g., "F#:maj")
                if ":" in chord_label:
                    try:
                        root, quality = chord_label.split(":", 1)
                        quality = quality.strip()
                    except ValueError:
                        quality = "maj"  # default if parsing fails
                else:
                    # Default to "maj" if no quality information is present.
                    quality = "maj"
                quality_counter[quality] += 1
    return quality_counter

def save_chord_qualities(counter, output_file):
    """
    Save the chord qualities and their counts to a text file.
    Each line in the file will be in the format:
    
        quality <tab> count
    
    Args:
        counter (Counter): A Counter mapping chord quality to count.
        output_file (str): Path to the output text file.
    """
    with open(output_file, "w") as f:
        # Sort by frequency in descending order
        for quality, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{quality}\t{count}\n")
    print(f"Chord qualities saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan chord TXT files for chord qualities and save them."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Root directory containing chord TXT files (e.g., data/POP909)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output text file (e.g., chord_qualities.txt)")
    args = parser.parse_args()
    
    quality_counter = scan_chord_qualities(args.input_dir)
    save_chord_qualities(quality_counter, args.output)
