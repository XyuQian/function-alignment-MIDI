import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from shoelace.musicgen.finetune.config import FRAME_RATE
from shoelace.midi_lm.models.config import SEG_RES, PAD
from shoelace.actual_shoelace.config import IDX_PAD
from shoelace.actual_shoelace.task_config import TASKS
from shoelace.utils.network_utils import transform_inputs

TOL_WIN = 0

TAIL_STEP = 64
MAX_SEQ_LEN = 512


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

    feature_dir = os.path.join(path_folder, "feature")

    text_dir = os.path.join(path_folder, "text_eval") if validation else os.path.join(path_folder, "text_train")

    all_files = []
    all_feature_paths = []

    for f in os.listdir(text_dir):
        
        path_lst = os.path.join(text_dir, f)
        path_h5 = os.path.join(feature_dir, "train.h5" if not validation else "validation.h5")

        with open(path_lst, "r") as pf:
            lines = [ln.strip().split("\t")[0] for ln in pf]
        # lines = lines[:10]
        
        all_files.append(lines)
        all_feature_paths.append(path_h5)
        
    

    return all_files, all_feature_paths


class ShoelaceDataset(Dataset):
    """
    Loads audio segments from .h5 feature files, using a simple sliding window approach.
    Each .h5 file stores multiple audio items (key = <fname> + '.audio').

    The dataset splits each audio item into segments of length MAX_DUR, stepping by 50 frames.
    """

    def __init__(self,
                 duration: float,
                 path_folder: str,
                 rid: int,
                 task_type: str,
                 num_workers: int = 1,
                 
                 use_loader: bool = True,
                 validation: bool = False,
                 vocals_only: bool = False, ):
        """
        Args:
            path_folder (str): Root folder containing 'pop909' data (text & feature).
            rid (int): Unique rank ID or worker ID for seeding.
            num_workers (int): Number of worker processes for data loading.
            use_loader (bool): If True, indicates usage with a DataLoader.
        """
        super().__init__()
        self.rid = rid
        self.use_loader = use_loader
        
        tasks = TASKS[task_type]
        
        
        max_frame = int(FRAME_RATE * duration)

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
                for j, fid in tqdm(enumerate(self.files[i]), total=len(self.files),
                                     desc=f"prepare dataset {i} / {len(self.feature_paths)}"):
                    
                    full_audio = f"{fid}.audio.multi-track"
                    if full_audio not in hf:
                        continue
                    audio_len = min(hf[f"{fid}.audio.{tag}"].shape[0] for tag in tasks["audio"])
                    total_len = audio_len - max_frame

                    for start_idx in range(0, total_len, SEG_RES):
                        # Worker assignment
                        worker_slot = str(i % num_workers)
                        sample = {
                            "audio": [i, j, start_idx],
                            "midi":{}
                        }
                        for tag in tasks["midi"]:
                            event_indices = hf[f"{fid}.midi.{tag}.sos"][:]
                            res_event_indices = hf[f"{fid}.midi.{tag}.res_sos"][:]
             
                            if start_idx//SEG_RES >= len(event_indices) - 1:
                                continue
                        
                            start_pos = start_idx//SEG_RES
                            end_pos = (start_idx + max_frame) // SEG_RES + TOL_WIN
                            end_pos = len(event_indices) - 1 if end_pos >= len(event_indices) else end_pos
                            midi_st = event_indices[start_pos]
                            midi_ed = event_indices[end_pos]

                            if start_pos + 1 < len(res_event_indices):
                                midi_prefix_st = res_event_indices[start_pos]
                                midi_prefix_ed = res_event_indices[start_pos + 1]
                            else:
                                midi_prefix_st = midi_prefix_ed = -1

                            if midi_ed - midi_st < 2:
                                continue
                            sample["midi"][tag] = [midi_st, midi_ed, midi_prefix_st, midi_prefix_ed]
                            
                        if len(sample["midi"]) > 0:
                            self.index_map[worker_slot].append(sample)
                    
        # Flatten count
        self.total_segments = sum(len(self.index_map[k]) for k in self.index_map)
        self.cache_data = {}
        self.max_frame = max_frame
        self.tasks = tasks

        # Informational prints
        print("AudioDataset initialized.")
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
        sample = index_list[idx % len(index_list)]

        # The .lst reference
        i, j, audio_index = sample["audio"]
        fid = self.files[i][j]
        tasks = self.tasks

        # If not cached, load from HDF5
        if fid not in self.cache_data:
            with h5py.File(self.feature_paths[i], "r") as hf:
                for modality in tasks:
                    for tag in tasks[modality]:
                        if modality == "midi":
                            for sub_tag in ["events", "sos", "res_events", "res_sos"]:
                                target_tag = f"{fid}.{modality}.{tag}.{sub_tag}"
                                self.cache_data[target_tag] = hf[target_tag][:]
                        else:
                            target_tag = f"{fid}.{modality}.{tag}"
                            self.cache_data[target_tag] = hf[target_tag][:]

        audio_tag = tasks["audio"][np.random.randint(len(tasks["audio"]))]
        audio_segment = self.cache_data[f"{fid}.audio.{audio_tag}"][audio_index: audio_index + self.max_frame]

        midi_tags = [tag for tag in sample["midi"]]
        midi_tag = midi_tags[np.random.randint(len(midi_tags))]
        midi_st, midi_ed, midi_prefix_st, midi_prefix_ed = sample["midi"][midi_tag]

        events_len = len(self.cache_data[f"{fid}.midi.{midi_tag}.events"])
        midi_ed = midi_ed + 1 if midi_ed + 1 < events_len else events_len
        midi_segment = self.cache_data[f"{fid}.midi.{midi_tag}.events"][midi_st : midi_ed]

        if midi_prefix_ed > midi_prefix_st:
            prefix = self.cache_data[f"{fid}.midi.{midi_tag}.res_events"][midi_prefix_st : midi_prefix_ed]
            midi_segment = np.concatenate([midi_segment[:1], prefix, midi_segment[1:]], axis=0)

        midi_segment[midi_segment < 0] = PAD
        return audio_segment, midi_segment, audio_tag, midi_tag

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
    arrays = [torch.from_numpy(b[0]) if isinstance(b[0], np.ndarray) else b[0] for b in batch]
    audio_data = torch.stack(arrays, dim=0).long()


    midi_arrays = [torch.from_numpy(b[1]) if isinstance(b[1], np.ndarray) else b[1] for b in batch]

    max_len = max([len(x) for x in midi_arrays])
    midi_seq = [
        F.pad(x, (0, 0, 0, max_len - len(x)), "constant", PAD) for x in midi_arrays
    ]
    if max_len > MAX_SEQ_LEN:
        midi_seq = [x[:MAX_SEQ_LEN] for x in midi_seq]
    midi_data = torch.stack(midi_seq, 0).long()
    
    midi_index = transform_inputs(midi_data[..., 0], SEG_RES).long()
    midi_index[midi_data[..., 0] == PAD] = IDX_PAD
    midi_index = F.pad(midi_index[:, :-1], (1, 0), "constant", 0)
    
    x = torch.arange(len(audio_data[0]) + 3).long().unsqueeze(0)
    
    # audio_index = torch.stack([F.pad(x, (i + 1, 3 - i), "constant", 0) for i in range(4)], -1)
    audio_index = F.pad(x, (1, 0), "constant", 0)
    
    audio_tasks = [b[2] for b in batch]
    midi_tasks = [b[3] for b in batch]