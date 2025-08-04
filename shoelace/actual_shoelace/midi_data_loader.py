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



def load_data_lst(path_folder, modality, validation):
    """
    Load dataset lists and corresponding feature paths.
    """
    list_folder = os.path.join(path_folder, modality, "text")
    feature_folder = os.path.join(path_folder, modality, "feature")
    if validation:
        list_folder = list_folder + "_eval"
        feature_folder = feature_folder + "_eval"
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
    def __init__(self, path_folder: str, modality: str, rid: int, num_workers: int = 1, validation: bool = False):
        super().__init__()

        self.rid = rid
        self.use_loader = num_workers > 1
        self.files, self.feature_paths = load_data_lst(path_folder, modality=modality, validation=validation)
        num_workers = max(1, num_workers)  # Ensure at least one worker

        self.index = {str(i): [] for i in range(num_workers)}
        self.tlen = [[] for _ in range(len(self.feature_paths))]

        self._prepare_dataset(num_workers, step=1 if not validation else 2)
        self.data = [None for _ in self.feature_paths]

        self.total_files = sum(len(f) for f in self.files)
        print(f"Number of {modality} MIDI files:", self.total_files)
        print("Number of segments:", len(self.index[str(0)]))

    
    def _prepare_dataset(self, num_workers, step):
        """Prepare dataset by reading feature files and indexing sequences."""
        for i, data_path in enumerate(self.feature_paths):
            with h5py.File(data_path, "r") as hf:
                self.tlen[i] = [0] * len(self.files[i])
                
                for j, f in tqdm(enumerate(self.files[i]), total=len(self.files[i]),
                                 desc=f"Preparing dataset {i}/{len(self.feature_paths)}"):
                    if f + ".sos" not in hf:
                        continue

                    sos_indices = hf[f + ".sos"][:]
                    res_sos_indices = hf[f + ".res_sos"][:]

                    if len(sos_indices) <= 1: # Need at least start and end of one segment
                        continue

                    self.tlen[i][j] = sos_indices[-1]
                    assert sos_indices[-1] > 0

                    for s, sid in enumerate(sos_indices[:-1]):
                        # print(s, sid)
                        if s % step != 0:
                            continue

                        # Determine corresponding rest segment indices safely
                        res_st = res_sos_indices[s] if s < len(res_sos_indices) else -1
                        res_ed = res_sos_indices[s + 1] if s + 1 < len(res_sos_indices) else -1

                        self.index[str(i % num_workers)].append([i, j, sid, res_st, res_ed])
                        if self.tlen[i][j] - sid < MAX_SEQ_LEN // 2:
                            break

        self.f_len = sum(len(f) for f in self.files)
    
    def __len__(self):
        return self.f_len
    
    def _load_cache(self, tid, fid, sid, res_st, res_ed):
        """Load cached MIDI data."""
        if self.data[tid] is None:
            self.data[tid] = h5py.File(self.feature_paths[tid], "r")

        fname = self.files[tid][fid]
        eid = min(self.tlen[tid][fid], sid + MAX_SEQ_LEN)
        data = self.data[tid][fname + ".events"][sid:eid][:]

        if res_ed > res_st:
            prefix = self.data[tid][fname + ".res_events"][res_st:res_ed][:]
            data = np.concatenate([data[:1], prefix, data[1:]], axis=0)
        
        data[data < 0] = PAD

        # # Truncation
        # if data.shape[0] > MAX_SEQ_LEN:
        #     data = data[:MAX_SEQ_LEN]
        # # Padding
        # elif data.shape[0] < MAX_SEQ_LEN:
        #     pad_len = MAX_SEQ_LEN - data.shape[0]
        #     pad = np.full((pad_len, data.shape[1]), PAD)
        #     data = np.concatenate([data, pad], axis=0)
        
        return data
    
    def __getitem__(self, idx):
        worker_id = get_worker_info().id if self.use_loader else 0
        tid, fid, sid, res_st, res_ed = self.index[str(worker_id)][idx % len(self.index[str(worker_id)])]
        return self._load_cache(tid, fid, sid, res_st, res_ed)


class PairedMIDIDataset(Dataset):
    """
    Dataset yielding tuples of (score_sequence, performance_sequence) with equal length.
    """
    def __init__(self, path_folder: str, rid: int, task_type: str, 
                 num_workers: int = 1, 
                 use_loader: bool = True, 
                 validation: bool = False):
        """
        Args:
            path_folder (str): Path to the folder containing the dataset (text & feature).
            rid (int): Unique rank ID or worker ID for seeding.
            task_type (str): Type of task ("midi_conversion" only).
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.score_ds = MIDIDataset(path_folder, 'Score', rid, num_workers, validation)
        self.perf_ds  = MIDIDataset(path_folder, 'Performance', rid, num_workers, validation)
        assert len(self.score_ds) == len(self.perf_ds), \
            "Score and Performance datasets have different lengths"
        self.length = len(self.score_ds)

        self.rid = rid
        self.use_loader = use_loader
        self.task = TASKS[task_type]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        score_seq = self.score_ds[idx]
        perf_seq  = self.perf_ds[idx]
        score_task = self.task["perf_2_score"]
        perf_task  = self.task["score_2_perf"]
        return score_seq, perf_seq, score_task, perf_task


# Paired dataset with the same data for sanity check
class PairedMIDIDatasetSanity(Dataset):
    """
    Dataset yielding tuples of the same sequence (x, x).
    """
    def __init__(self, path_folder: str, rid: int, task_type: str, 
                 num_workers: int = 1, 
                 use_loader: bool = True, 
                 validation: bool = False,
                 modality: str = "Score"):
        """
        Args:
            path_folder (str): Path to the folder containing the dataset (text & feature).
            rid (int): Unique rank ID or worker ID for seeding.
            task_type (str): Type of task ("midi_conversion" only).
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.score_ds = MIDIDataset(path_folder, modality, rid, num_workers, validation)
        self.perf_ds  = MIDIDataset(path_folder, modality, rid, num_workers, validation)
        self.length = len(self.score_ds)

        self.rid = rid
        self.use_loader = use_loader
        self.task = TASKS[task_type]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        score_seq = self.score_ds[idx]
        perf_seq  = self.perf_ds[idx]
        score_task = self.task["perf_2_score"]
        perf_task  = self.task["score_2_perf"]
        return score_seq, perf_seq, score_task, perf_task



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
    dataset = PairedMIDIDatasetSanity(path_folder="data/formatted/ASAP", rid=0, 
                                task_type="midi_conversion", num_workers=1)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn,
                            shuffle=True, num_workers=1, drop_last=True)

    # batch = next(iter(dataloader))
    for i, batch in enumerate(dataloader):
        score_data = batch["ScoreLM"]["args"]["input_ids"]
        perf_data = batch["PerformanceLM"]["args"]["input_ids"]
        print(f"Score Data Shape: {score_data.shape}, Performance Data Shape: {perf_data.shape}")
        score_indices = batch["ScoreLM"]["indices"]
        perf_indices = batch["PerformanceLM"]["indices"]
        # print(f"Score Indices Shape: {score_indices.shape}, Performance Indices Shape: {perf_indices.shape}")

        score_tasks = batch["ScoreLM"]["tasks"]
        perf_tasks = batch["PerformanceLM"]["tasks"]

        assert torch.equal(score_data, perf_data), \
            "Score and Performance data should be equal in the Sanity dataset."
        assert torch.equal(score_indices, perf_indices), \
            "Score and Performance indices should be equal in the Sanity dataset."
        assert score_tasks == perf_tasks, \
            f"Score and Performance tasks should be equal in the Sanity dataset, {score_tasks} != {perf_tasks}"
        # print(f"Max Score Indices: {score_indices.max()}, Max Performance Indices: {perf_indices.max()}")
        assert score_indices.max() < 10000
        assert perf_indices.max() < 10000
    

def test():
    dataset = PairedMIDIDataset(path_folder="data/formatted/ASAP", rid=0, 
                                task_type="midi_conversion", num_workers=1)
    # score_ds = MIDIDataset(path_folder="data/formatted/samples", modality='', rid=0, num_workers=1)
    score_ds = dataset.score_ds
    perf_ds = dataset.perf_ds

    for i in range(20):
        tid, fid, sid, res_st, res_ed = score_ds.index[str(0)][i]
        t_length = score_ds.tlen[tid][fid]
        eid = min(t_length, sid + MAX_SEQ_LEN)
        fname = score_ds.files[tid][fid]
        if score_ds.data[tid] is None:
            score_ds.data[tid] = h5py.File(score_ds.feature_paths[tid], "r")
        
        data = score_ds.data[tid][fname + ".events"][sid:eid][:]
        sos = score_ds.data[tid][fname + ".sos"][:]
        res_sos = score_ds.data[tid][fname + ".res_sos"][:]

        file_length = score_ds.data[tid][fname + ".events"][:][:].shape[0]
        seg_length = data.shape[0]
        print(f"Score event {i}: {tid}, {fid}, File len: {file_length}, Segment len:{seg_length}, sid: {sid}, eid: {eid}, {res_st}, {res_ed}")


if __name__ == "__main__":
    test_samples()
    # test()