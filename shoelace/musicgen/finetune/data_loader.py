import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .config import MAX_DUR, FRAME_RATE


def load_data_lst(path_folder: str, validation: bool):
    """
    Load file lists and corresponding hdf5 feature paths from a specified folder.

    Args:
        path_folder (str): Root folder containing 'pop909/text' and 'pop909/feature' subdirs.

    Returns:
        (list_of_lists, list_of_paths):
            files: A list of lists, each sub-list containing .lst entries for a subset.
            feature_paths: A list of .h5 file paths (one per .lst file).
    """
    text_dir = os.path.join(path_folder, "pop909", "text")
    feature_dir = os.path.join(path_folder, "pop909", "feature")

    text_dir = text_dir + "_eval" if validation else text_dir

    all_files = []
    all_feature_paths = []

    for f in os.listdir(text_dir):
        path_lst = os.path.join(text_dir, f)
        path_h5 = os.path.join(feature_dir, f.replace(".lst", ".h5"))

        with open(path_lst, "r") as pf:
            lines = [ln.strip().split("\t")[0] for ln in pf]

        all_files.append(lines)
        all_feature_paths.append(path_h5)

    return all_files, all_feature_paths


class AudioDataset(Dataset):
    """
    Loads audio segments from .h5 feature files, using a simple sliding window approach.
    Each .h5 file stores multiple audio items (key = <fname> + '.audio').

    The dataset splits each audio item into segments of length MAX_DUR, stepping by 50 frames.
    """

    def __init__(self,
                 path_folder: str,
                 rid: int,
                 is_mono: bool = True,
                 num_workers: int = 1,
                 use_loader: bool = True,
                 validation: bool = False,
                 vocals_only: bool = False):
        """
        Args:
            path_folder (str): Root folder containing 'pop909' data (text & feature).
            rid (int): Unique rank ID or worker ID for seeding.
            is_mono (bool): Whether audio is single-channel (mono).
            num_workers (int): Number of worker processes for data loading.
            use_loader (bool): If True, indicates usage with a DataLoader.
        """
        super().__init__()
        self.rid = rid
        self.use_loader = use_loader
        self.is_mono = is_mono

        # Load the .lst file references & .h5 paths
        files_list, feature_paths = load_data_lst(path_folder, validation=validation)
        self.files = files_list  # list of lists
        self.feature_paths = feature_paths

        # Prepare indexing
        if num_workers < 1:
            num_workers = 1
        self.index_map = {str(i): [] for i in range(num_workers)}

        # Build index of valid segments for each file
        for i, path_h5 in enumerate(self.feature_paths):
            with h5py.File(path_h5, "r") as hf:
                # For each .lst line in this subset
                for j, fname in tqdm(enumerate(self.files[i]), total=len(self.files),
                                     desc=f"prepare dataset {i} / {len(self.feature_paths)}"):
                    audio_key = fname + ".audio"
                    if audio_key not in hf:
                        continue
                    audio_len = hf[audio_key].shape[0]

                    total_len = audio_len - MAX_DUR
                    # Step in increments of 50
                    for start_idx in range(0, total_len, FRAME_RATE*2):
                        # Worker assignment
                        worker_slot = str(i % num_workers)
                        self.index_map[worker_slot].append([i, j, start_idx])

        # Flatten count
        self.total_segments = sum(len(self.index_map[k]) for k in self.index_map)
        self.cache_data = {}

        # Informational prints
        print("AudioDataset initialized.")
        print("  > is_mono:", self.is_mono)
        print("  > # of .lst groups:", len(self.files))
        print("  > # of total files:", sum(len(x) for x in self.files))
        print("  > # of total segments:", self.total_segments)

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx: int):
        """
        Retrieve a single audio segment.
        """
        # For demonstration, we always use '0' as the worker key in single-process usage.
        index_list = self.index_map[str(0)]
        i, j, start_pos = index_list[idx % len(index_list)]

        # The .lst reference
        fname = self.files[i][j]

        # If not cached, load from HDF5
        if fname not in self.cache_data:
            with h5py.File(self.feature_paths[i], "r") as hf:
                self.cache_data[fname] = {
                    "audio": hf[fname + ".audio"][:]  # full array
                }

        audio_full = self.cache_data[fname]["audio"]
        audio_segment = audio_full[start_pos: start_pos + MAX_DUR]
        return audio_segment

    def reset_random_seed(self, seed_base: int, epoch: int):
        """
        Shuffle the indexing for each worker shard, typically called at epoch start.
        """
        np.random.seed(seed_base + self.rid * 100)
        for k in self.index_map:
            np.random.shuffle(self.index_map[k])


def worker_init_fn(worker_id: int):
    """
    Worker init function. Could set np.random.seed(...) if needed for each worker.
    """
    pass


def collate_fn(batch):
    """
    Collate function that stacks audio segments into a single tensor.
    """
    # Each 'b' in batch is a numpy array or a torch array
    arrays = [torch.from_numpy(b) if isinstance(b, np.ndarray) else b for b in batch]
    audio_data = torch.stack(arrays, dim=0).long()

    return {"x": audio_data}
