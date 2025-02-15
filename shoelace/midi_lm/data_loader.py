import os
import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset, get_worker_info
from .config import PAD, MAX_SEQ_LEN


def load_data_lst(path_folder):
    """
    Load dataset lists and corresponding feature paths.
    """
    list_folder = os.path.join(path_folder, "las", "dur_lt_30_text_beta")
    feature_folder = os.path.join(path_folder, "las", "dur_lt_30_text_beta_midis")

    files, feature_paths = [], []

    for f in os.listdir(list_folder):
        path_lst = os.path.join(list_folder, f)
        f_path = os.path.join(feature_folder, f.replace(".lst", ".h5"))
        print(f_path)
        # if not f_path == "data/formatted/las/dur_lt_30_text_beta_midis/1.h5":
        #     continue
        with open(path_lst, "r") as pf:
            fs = pf.readlines()

        files.append([s.rstrip().split("\t")[0] for s in fs])
        feature_paths.append(f_path)

    return files, feature_paths


class MIDIDataset(BaseDataset):
    def __init__(self, path_folder: str, rid: int, num_workers: int = 1):
        super().__init__()

        self.rid = rid
        self.use_loader = num_workers > 1
        self.files, self.feature_paths = load_data_lst(path_folder)
        num_workers = max(1, num_workers)  # Ensure at least one worker

        self.index = {str(i): [] for i in range(num_workers)}
        self.tlen = [[] for _ in range(len(self.feature_paths))]

        self._prepare_dataset(num_workers)
        self.data = [None for _ in self.feature_paths]

        print("Number of files:", sum(len(f) for f in self.files))
        print("Number of segments:", self.f_len)

    def _prepare_dataset(self, num_workers):
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

                    if len(sos_indices) <= 1:
                        continue

                    self.tlen[i][j] = sos_indices[-1]
                    assert sos_indices[-1] > 0

                    for s, sid in enumerate(sos_indices[:-1]):
                        if s % 5 != 0:
                            continue

                        res_st = res_sos_indices[s] if s < len(res_sos_indices) else -1
                        res_ed = res_sos_indices[s + 1] if s + 1 < len(res_sos_indices) else -1

                        self.index[str(i % num_workers)].append([i, j, sid, res_st, res_ed])
                        if self.tlen[i][j] - sid < MAX_SEQ_LEN:
                            break

        self.f_len = sum(len(self.index[i]) for i in self.index)

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
        return data[:MAX_SEQ_LEN]

    def __getitem__(self, idx):
        worker_id = get_worker_info().id if self.use_loader else 0
        tid, fid, sid, res_st, res_ed = self.index[str(worker_id)][idx % len(self.index[str(worker_id)])]
        return self._load_cache(tid, fid, sid, res_st, res_ed)

    def reset_random_seed(self, seed: int, epoch: int):
        """Reset dataset random seed."""
        self.rng = np.random.RandomState(seed + self.rid * 100)
        self.epoch = epoch
        for key in self.index:
            self.rng.shuffle(self.index[key])


def worker_init_fn(worker_id):
    pass


def collate_fn(batch):
    """Collate function to pad sequences."""
    max_len = max(len(x) for x in batch)

    seq = [
        np.pad(x, ((0, max_len - len(x)), (0, 0)), "constant", constant_values=(PAD, PAD))
        if len(x) < max_len else x
        for x in batch
    ]

    return {"x": torch.from_numpy(np.stack(seq, axis=0)).long()}


if __name__ == "__main__":
    dataset = MIDIDataset(path_folder="data/formatted/", rid=0, num_workers=1)

    for i in range(10):
        event = dataset[i]
        print("=================================")
        print(event[:20])
