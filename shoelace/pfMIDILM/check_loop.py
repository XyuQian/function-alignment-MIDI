import os
import shutil
import sys
from tqdm import tqdm

import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pypianoroll
from joblib import Parallel, delayed
RES = 50

# Step 1: Load MIDI file
def load_midi(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    return midi_data


# Step 2: Remove silent segments at the beginning and ending
def trim_silence(pianoroll_matrix):
    """
    Remove silent segments at the beginning and ending of the pianoroll matrix.

    Args:
        pianoroll_matrix (np.ndarray): The pianoroll matrix.

    Returns:
        np.ndarray: Trimmed pianoroll matrix.
    """
    non_silent_indices = np.where(np.sum(pianoroll_matrix, axis=1) > 0)[0]
    if len(non_silent_indices) == 0:
        return pianoroll_matrix  # Return original if all silent
    start_idx = non_silent_indices[0]
    end_idx = non_silent_indices[-1] + 1
    return pianoroll_matrix[start_idx:end_idx]


# Step 3: Convert MIDI to multi-track pianoroll
def midi_to_pianoroll(midi_data, resolution=RES):
    multitrack = []
    sec = midi_data.get_end_time()
    max_len = int(sec * resolution + 1)
    for instrument in midi_data.instruments:
        m = np.zeros([max_len, 128])
        for note in instrument.notes:
            st = int(note.start * resolution)
            ed = int(note.end * resolution)
            m[st : ed, note.pitch] = 1
        multitrack.append(m)

    multitrack = np.concatenate(multitrack, -1)
    return multitrack


# Step 4: Compute self-similarity matrix using sliding window
def compute_self_similarity_with_window(pianoroll_matrix, window_size, similarity_threshold):
    """
    Compute self-similarity matrix using patterns extracted with a sliding window.

    Args:
        pianoroll_matrix (np.ndarray): The pianoroll matrix.
        window_size (int): Size of the sliding window.

    Returns:
        np.ndarray: Self-similarity matrix between the extracted patterns.
    """
    num_timesteps = pianoroll_matrix.shape[0]
    num_windows = num_timesteps - window_size + 1

    # Extract sliding window patterns
    patterns = np.array([
        pianoroll_matrix[i:i + window_size].flatten()
        for i in range(0, num_windows, window_size // 10)
    ])


    # Compute similarity between patterns
    similarity_matrix = cosine_similarity(patterns)
    # similarity_matrix[similarity_matrix < similarity_threshold * 2] = 0.
    return similarity_matrix


# Step 5: Check if the similarity matrix is acceptable
def is_similarity_acceptable(similarity_matrix, similarity_threshold, midi_path):
    """
    Evaluate if the similarity matrix is acceptable.

    Args:
        similarity_matrix (np.ndarray): The self-similarity matrix.
        similarity_threshold (float): Threshold for the mean similarity to be considered acceptable.
        midi_path (str): Path to the MIDI file being analyzed.

    Returns:
        bool: True if the mean similarity is below the threshold, False otherwise.
    """
    num_windows = similarity_matrix.shape[0]

    mean_similarity = (
                              np.sum(similarity_matrix) - np.trace(similarity_matrix)
                      ) / (num_windows * (num_windows - 1))
    print(f"MIDI file '{midi_path}': Mean similarity = {mean_similarity:.4f}")
    return mean_similarity < similarity_threshold


# Step 6: Visualize self-similarity matrix
def plot_self_similarity(similarity_matrix, output_path, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(similarity_matrix, origin='lower', cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Similarity')
    plt.title(title)
    plt.xlabel('Pattern Index')
    plt.ylabel('Pattern Index')
    plt.savefig(output_path)
    plt.close()  # Close the figure to prevent memory issues


# Main function
def analyze_and_filter_midi(midi_path, similarity_threshold, window_size, output_folder):
    # Load MIDI
    midi_data = load_midi(midi_path)

    # Convert to multi-track pianoroll
    merged_pianoroll = midi_to_pianoroll(midi_data)
    # Merge tracks into a single matrix (summing tracks)


    # Trim silent segments
    merged_pianoroll = trim_silence(merged_pianoroll)
    # Compute self-similarity matrix using sliding window
    similarity_matrix = compute_self_similarity_with_window(merged_pianoroll, window_size, similarity_threshold)
    similarity_output_path = os.path.join(output_folder, f"{os.path.basename(midi_path)}_similarity_matrix.png")
    plot_self_similarity(similarity_matrix, similarity_output_path, 'Sliding Window Self-Similarity Matrix')

    # Check if the similarity matrix is acceptable
    is_acceptable = is_similarity_acceptable(similarity_matrix, similarity_threshold, midi_path)

    if not is_acceptable:
        print(f"MIDI file '{midi_path}' flagged as low quality.")

    return is_acceptable


if __name__ == "__main__":
    # file_path = "data/formatted/groups/pop909_text/0.lst"
    file_path = "data/formatted/las/short_dur_text_beta/0.lst"
    output_folder = "filtered_midi"
    os.makedirs(output_folder, exist_ok=True)

    similarity_threshold = 0.35  # Adjust this threshold based on your dataset
    window_size = RES * 5  # Sliding window size (e.g., 24 frames)

    # Read file paths from the list file
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = lines[100:120]
        lines = [line.split("\t")[0] for line in lines]


    def process_line(i, line):
        line = line.rstrip()
        output_path = os.path.join(output_folder, str(i) + ".mid")
        shutil.copyfile(src=line, dst=output_path)

        # Analyze and filter MIDI
        analyze_and_filter_midi(output_path, similarity_threshold, window_size, output_folder)


    # process_line(0, lines[0])
    # # Process each MIDI file and filter low-quality ones in parallel
    Parallel(n_jobs=-1)(
        delayed(process_line)(i, line) for i, line in enumerate(tqdm(lines, desc="Processing MIDI files", unit="file")))
