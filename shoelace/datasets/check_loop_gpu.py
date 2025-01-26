import os
import shutil
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch

RES = 50

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
            m[st: ed, note.pitch] = 1
        multitrack.append(m)

    multitrack = np.concatenate(multitrack, -1)
    return multitrack


# Step 4: Compute self-similarity matrix using sliding window
def compute_self_similarity_with_window(pianoroll_matrix, window_size, similarity_threshold, use_cuda=False):
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
    patterns = []
    for i in range(0, num_windows, window_size // 10):
        patterns.append(pianoroll_matrix[i:i + window_size].flatten())
    patterns = np.array(patterns)


    if use_cuda:
        # Move patterns to GPU for computation
        patterns = torch.tensor(patterns, device='cuda', dtype=torch.float32)
        similarity_matrix = torch.matmul(patterns, patterns.T)
        norms = torch.norm(patterns, dim=1, keepdim=True)
        similarity_matrix /= (norms @ norms.T + 1e-8)  # Normalize cosine similarity
        similarity_matrix = similarity_matrix.cpu().numpy()  # Move back to CPU
    else:
        # Compute similarity on CPU
        similarity_matrix = cosine_similarity(patterns)

    return similarity_matrix


def compute_mean_similarity(midi_data, similarity_threshold, window_size, use_cuda=False):
    merged_pianoroll = midi_to_pianoroll(midi_data)
    merged_pianoroll = trim_silence(merged_pianoroll)
    similarity_matrix = compute_self_similarity_with_window(merged_pianoroll, window_size, similarity_threshold, use_cuda)
    num_windows = similarity_matrix.shape[0]

    mean_similarity = (
        np.sum(similarity_matrix) - np.trace(similarity_matrix)
    ) / (num_windows * (num_windows - 1))

    return mean_similarity
