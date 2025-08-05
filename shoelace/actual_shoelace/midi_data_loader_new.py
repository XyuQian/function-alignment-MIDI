import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from tqdm import tqdm
import torch.nn.functional as F
from shoelace.midi_lm.models.config import SEG_RES, PAD
from shoelace.actual_shoelace.config import IDX_PAD
from shoelace.actual_shoelace.midi_config import TASKS
from shoelace.utils.network_utils import transform_inputs
from shoelace.datasets.utils import decode


MAX_SEQ_LEN = 1024
MIN_SEQ_LEN = 128


def load_data_lst(path_folder, modality, validation):
    """
    Load dataset lists and corresponding feature paths.
    """
    list_folder = os.path.join(path_folder, modality, "text")
    feature_folder = os.path.join(path_folder, modality, "feature")
    if validation:
        list_folder = os.path.join(list_folder, "val")
        feature_folder = os.path.join(feature_folder, "val")
    else:
        list_folder = os.path.join(list_folder, "train")
        feature_folder = os.path.join(feature_folder, "train")
    files, feature_paths = [], []

    for f in os.listdir(list_folder):
        path_lst = os.path.join(list_folder, f)
        f_path = os.path.join(feature_folder, f.replace(".lst", ".h5"))
        print(f_path)

        try:
            with open(path_lst, "r") as pf:
                fs = pf.readlines()

            files.append([s.rstrip().split("\t")[0] for s in fs])
            feature_paths.append(f_path)
        except FileNotFoundError:
            print(f"Error: List file not found {path_lst}")
        except Exception as e:
            print(f"Error processing list file {path_lst}: {e}")

    return files, feature_paths


class MIDIDataset(Dataset):
    """
    A helper dataset class to load individual MIDI file segments.
    It pre-loads metadata (tlen, sos_indices, res_sos_indices) for efficient access
    by the PairedMIDIDataset, which handles the actual indexing and alignment.
    """
    def __init__(self, path_folder: str, modality: str, rid: int, num_workers: int = 1, validation: bool = False):
        super().__init__()
        self.rid = rid
        self.use_loader = num_workers > 1 # Not directly used here, but kept for consistency
        self.files, self.feature_paths = load_data_lst(path_folder, modality=modality, validation=validation)
        self.data_handles = [None for _ in self.feature_paths] # To hold h5py file handles

        # Pre-load all tlen, sos, res_sos for all files to enable paired indexing
        self.all_file_metadata = [[] for _ in range(len(self.feature_paths))]
        print(f"Loading metadata for {modality} MIDI files...")
        for i, data_path in enumerate(self.feature_paths):
            try:
                with h5py.File(data_path, "r") as hf:
                    for j, f in tqdm(enumerate(self.files[i]), total=len(self.files[i]),
                                     desc=f"Processing {modality} file {i+1}/{len(self.feature_paths)}"):
                        if f + ".sos" not in hf:
                            self.all_file_metadata[i].append(None) # Mark as invalid
                            continue
                        
                        sos_indices = hf[f + ".sos"][:]
                        res_sos_indices = hf[f + ".res_sos"][:]
                        tlen = sos_indices[-1] if len(sos_indices) > 0 else 0

                        if tlen <= 0 or len(sos_indices) <= 1: # Need at least start and end of one segment
                            self.all_file_metadata[i].append(None)
                            continue
                        
                        self.all_file_metadata[i].append({
                            "tlen": tlen,
                            "sos_indices": sos_indices,
                            "res_sos_indices": res_sos_indices
                        })
            except Exception as e:
                print(f"Error loading H5 file {data_path}: {e}")
                # Append None for all files in this path if the file itself is problematic
                self.all_file_metadata[i] = [None] * len(self.files[i])
        
        self.total_valid_files = sum(1 for f_list in self.all_file_metadata for f_meta in f_list if f_meta is not None)
        print(f"Number of {modality} MIDI files with valid metadata: {self.total_valid_files}")

    def _get_file_metadata(self, tid, fid):
        """Retrieves pre-loaded metadata for a specific file."""
        if tid >= len(self.all_file_metadata) or fid >= len(self.all_file_metadata[tid]):
            return None
        return self.all_file_metadata[tid][fid]

    def _load_cache(self, tid, fid, sid, res_st, res_ed):
        """Load cached MIDI data segment."""
        if self.data_handles[tid] is None:
            self.data_handles[tid] = h5py.File(self.feature_paths[tid], "r")

        fname = self.files[tid][fid]
        
        # Get the total length of the file's events from pre-loaded metadata
        file_meta = self._get_file_metadata(tid, fid)
        if file_meta is None:
            # This should ideally not happen if indexing is done correctly by PairedMIDIDataset
            raise ValueError(f"Metadata not found for {fname} in {self.feature_paths[tid]}")
        
        tlen = file_meta["tlen"]

        # Ensure the end index does not exceed MAX_SEQ_LEN or the file's total length
        eid = min(tlen, sid + MAX_SEQ_LEN)
        data = self.data_handles[tid][fname + ".events"][sid:eid][:]

        if res_ed > res_st:
            prefix = self.data_handles[tid][fname + ".res_events"][res_st:res_ed][:]
            data = np.concatenate([data[:1], prefix, data[1:]], axis=0)
        
        data[data < 0] = PAD # Replace negative values with padding token

        # The collate_fn will handle final padding to batch max length
        return data
    
    # __len__ and __getitem__ are removed from MIDIDataset as it's now a helper.


class PairedMIDIDataset(Dataset):
    """
    Dataset yielding tuples of (score_sequence, performance_sequence) with similar length,
    ensuring content alignment by jointly indexing segments.
    """
    def __init__(self, path_folder: str, rid: int, task_type: str, 
                 num_workers: int = 1, 
                 use_loader: bool = True, 
                 validation: bool = False):
        super().__init__()
        # Initialize individual MIDIDataset instances to load their respective metadata
        self.score_ds = MIDIDataset(path_folder, 'Score', rid, num_workers, validation)
        self.perf_ds  = MIDIDataset(path_folder, 'Performance', rid, num_workers, validation)
        
        self.rid = rid
        self.use_loader = use_loader
        self.task = TASKS[task_type]

        # This index will store aligned (score_tid, score_fid, score_sid, score_res_st, score_res_ed,
        #                                perf_tid, perf_fid, perf_sid, perf_res_st, perf_res_ed) tuples
        self.index = {str(i): [] for i in range(max(1, num_workers))} 
        self._prepare_paired_dataset(num_workers)
        
        self.length = sum(len(v) for v in self.index.values())
        print(f"Number of aligned paired segments: {self.length}")

    def _prepare_paired_dataset(self, num_workers):
        """
        Prepare dataset by jointly indexing sequences for score and performance,
        ensuring content alignment based on MAX_SEQ_LEN.
        """
        # Iterate through feature_paths (assuming score and perf have same structure and order)
        for i, score_data_path in enumerate(self.score_ds.feature_paths):
            # Assuming corresponding performance data path exists at the same index
            perf_data_path = self.perf_ds.feature_paths[i] 

            for j, score_fname in tqdm(enumerate(self.score_ds.files[i]), total=len(self.score_ds.files[i]),
                                     desc=f"Aligning paired dataset {i+1}/{len(self.score_ds.feature_paths)}"):
                # Assuming corresponding performance filename exists at the same index
                perf_fname = self.perf_ds.files[i][j] 

                score_meta = self.score_ds._get_file_metadata(i, j)
                perf_meta = self.perf_ds._get_file_metadata(i, j)

                if score_meta is None or perf_meta is None:
                    continue # Skip if metadata for either modality is invalid for this file

                score_sos_indices = score_meta["sos_indices"]
                score_res_sos_indices = score_meta["res_sos_indices"]
                score_tlen = score_meta["tlen"]

                perf_sos_indices = perf_meta["sos_indices"]
                perf_res_sos_indices = perf_meta["res_sos_indices"]
                perf_tlen = perf_meta["tlen"]

                # Iterate through segments, ensuring both score and performance can provide MAX_SEQ_LEN
                # This assumes that sos_indices are inherently aligned between score and performance
                # for the same musical piece, which is typical for paired datasets.
                min_sos_len = min(len(score_sos_indices), len(perf_sos_indices))

                for s in range(min_sos_len - 1): # Iterate up to the second to last segment start

                    score_sid = score_sos_indices[s]
                    perf_sid = perf_sos_indices[s]

                    # Determine corresponding residual segment indices safely
                    score_res_st = score_res_sos_indices[s] if s < len(score_res_sos_indices) else -1
                    score_res_ed = score_res_sos_indices[s + 1] if s + 1 < len(score_res_sos_indices) else -1

                    perf_res_st = perf_res_sos_indices[s] if s < len(perf_res_sos_indices) else -1
                    perf_res_ed = perf_res_sos_indices[s + 1] if s + 1 < len(perf_res_sos_indices) else -1

                    # CRITICAL ALIGNMENT LOGIC:
                    # Only index this segment if BOTH score and performance have enough remaining length
                    # to provide a MIN_SEQ_LEN segment from their respective starting points.
                    # This prevents indexing a score segment that has enough length but its performance
                    # counterpart doesn't, or vice-versa, ensuring content alignment at the boundaries.
                    if (score_tlen - score_sid < MIN_SEQ_LEN) or \
                       (perf_tlen - perf_sid < MIN_SEQ_LEN):
                        break # Stop indexing further segments for this file if either runs short

                    # If both are long enough, add to the shared index
                    # The worker ID is determined by the file index 'j' to distribute files across workers
                    self.index[str(j % num_workers)].append([
                        i, j, # tid (feature_path index), fid (file index within feature_path)
                        score_sid, score_res_st, score_res_ed,
                        perf_sid, perf_res_st, perf_res_ed
                    ])
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        worker_id = get_worker_info().id if self.use_loader else 0
        # Retrieve all indices for both score and performance from the shared index
        (tid, fid, score_sid, score_res_st, score_res_ed,
         perf_sid, perf_res_st, perf_res_ed) = self.index[str(worker_id)][idx % len(self.index[str(worker_id)])]
        
        # Load the actual data using the individual MIDIDataset instances
        score_seq = self.score_ds._load_cache(tid, fid, score_sid, score_res_st, score_res_ed)
        perf_seq  = self.perf_ds._load_cache(tid, fid, perf_sid, perf_res_st, perf_res_ed)
        
        score_task = self.task["perf_2_score"]
        perf_task  = self.task["score_2_perf"]
        return score_seq, perf_seq, score_task, perf_task


class PairedMIDIDatasetSanity(Dataset):
    def __init__(self, path_folder: str, rid: int, task_type: str, modality: str, validation: bool = False):
        super().__init__()
        self.path_folder = path_folder
        self.rid = rid
        self.task_type = task_type
        self.modality = modality
        self.validation = validation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(worker_id: int):
    """
    Worker init function. Could set np.random.seed(...) if needed for each worker.
    """
    pass


def collate_fn(batch):
    """Collate function to pad sequences."""
    score_array = [torch.from_numpy(x[0]) for x in batch]
    perf_array  = [torch.from_numpy(x[1]) for x in batch]
    max_score_len = max(len(x) for x in score_array)
    max_perf_len  = max(len(x) for x in perf_array)

    score_array = [
        F.pad(x, (0, 0, 0, max_score_len - len(x)), "constant", PAD)
        if len(x) < max_score_len else x
        for x in score_array
    ]
    score_data = torch.stack(score_array, dim=0).long()

    perf_array = [
        F.pad(x, (0, 0, 0, max_perf_len - len(x)), "constant", PAD)
        if len(x) < max_perf_len else x
        for x in perf_array
    ]
    perf_data = torch.stack(perf_array, dim=0).long()
    
    score_index = transform_inputs(score_data[..., 0], SEG_RES).long()
    score_index[score_index > IDX_PAD] = IDX_PAD
    # batch_size, seq_len = score_data.shape[0], score_data.shape[1]
    # score_index = torch.arange(seq_len, device=score_data.device).unsqueeze(0).repeat(batch_size, 1)
    score_index[score_data[..., 0] == PAD] = IDX_PAD
    score_index = F.pad(score_index[:, :-1], (1, 0), "constant", 0)

    perf_index = transform_inputs(perf_data[..., 0], SEG_RES).long()
    perf_index[perf_index > IDX_PAD] = IDX_PAD
    # batch_size, seq_len = perf_data.shape[0], perf_data.shape[1]
    # perf_index = torch.arange(seq_len, device=perf_data.device).unsqueeze(0).repeat(batch_size, 1)
    perf_index[perf_data[..., 0] == PAD] = IDX_PAD
    perf_index = F.pad(perf_index[:, :-1], (1, 0), "constant", 0)

    score_tasks = [x[2][0] for x in batch]
    perf_tasks  = [x[3][0] for x in batch]

    return {
        "ScoreLM": {
                "args": {
                    "input_ids": score_data,  
                },
                "tasks": score_tasks,
                "indices": score_index
            },
        "PerformanceLM": {
                "args": {
                    "input_ids": perf_data,
                },
                "tasks": perf_tasks,
                "indices": perf_index
            }
        }


def test_samples():
    """
    Test the dataset by loading a few samples.
    """
    # dataset = PairedMIDIDatasetSanity(path_folder="data/formatted/ASAP", rid=0, 
    #                                 task_type="midi_conversion", num_workers=0, modality="Score")
    # dataset = PairedMIDIDatasetSanity(path_folder="data/formatted/ASAP", rid=0, 
    #                                 task_type="midi_conversion", num_workers=0, modality="Performance")
    dataset = PairedMIDIDataset(path_folder="data/formatted/ASAP", rid=0, 
                                 task_type="midi_conversion", num_workers=4, validation=True)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn,
                            shuffle=True, num_workers=4, drop_last=True)

    # batch = next(iter(dataloader))
    for i, batch in enumerate(dataloader):
        if i > 20:
            break
        score_data = batch["ScoreLM"]["args"]["input_ids"]
        perf_data = batch["PerformanceLM"]["args"]["input_ids"]
        print(f"Score Data Shape: {score_data.shape}, Performance Data Shape: {perf_data.shape}")
        score_indices = batch["ScoreLM"]["indices"]
        perf_indices = batch["PerformanceLM"]["indices"]
        # print(f"Score Indices Shape: {score_indices.shape}, Performance Indices Shape: {perf_indices.shape}")

        score_tasks = batch["ScoreLM"]["tasks"]
        perf_tasks = batch["PerformanceLM"]["tasks"]

        # score_seq = decode(os.path.join("test_results", f"score_{i}.mid"), score_data[0].numpy())
        # perf_seq = decode(os.path.join("test_results", f"perf_{i}.mid"), perf_data[0].numpy())

        # assert torch.equal(score_data, perf_data), \
        #     "Score and Performance data should be equal in the Sanity dataset."
        # assert torch.equal(score_indices, perf_indices), \
        #     "Score and Performance indices should be equal in the Sanity dataset."
        # assert score_tasks == perf_tasks, \
        #     f"Score and Performance tasks should be equal in the Sanity dataset, {score_tasks}!= {perf_tasks}"
        # print(f"Max Score Indices: {score_indices.max()}, Max Performance Indices: {perf_indices.max()}")
        assert score_indices.max() < 10000
        assert perf_indices.max() < 10000
    


if __name__ == "__main__":
    test_samples()