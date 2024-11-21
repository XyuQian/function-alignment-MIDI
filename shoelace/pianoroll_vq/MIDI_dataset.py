import os

import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset, get_worker_info

DIGITS = [2 ** i for i in range(7)]


def decimal2binary(x):
    total_d = len(DIGITS)
    outs = []
    for i in range(total_d):
        m = x >= DIGITS[total_d - 1 - i]
        outs.append(m)
        x = x - m * DIGITS[total_d - 1 - i]
    return outs


def load_data_lst(path_folder):
    list_folder = os.path.join(path_folder, "pop909_text")
    feature_folder = os.path.join(path_folder, "pop909_feature")
    files = []
    feature_path = []
    for f in os.listdir(list_folder):
        path_lst = os.path.join(list_folder, f)
        f_path = os.path.join(feature_folder, str.replace(f, ".lst", ".h5"))
        with open(path_lst, "r") as rf:
            fs = rf.readlines()
        files.append([s.rstrip().split("\t")[0] for s in fs])
        feature_path.append(f_path)

    return files, feature_path


class MIDIDataset(BaseDataset):
    def __init__(self, path_folder, rid, sec, res=50, num_workers=1, use_loader=True):
        super(BaseDataset, self).__init__()
        self.rid = rid
        self.sec = sec
        self.res = res
        self.seg_len = int(res * sec) // 128 * 128
        self.use_loader = use_loader
        files, feature_path = load_data_lst(path_folder)

        index = {str(i): [] for i in range(num_workers)}

        for i, data_path in enumerate(feature_path):
            with h5py.File(data_path, "r") as hf:
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):
                    # print(data_path, f)
                    shape = hf[f].shape[0]
                    for k in range(0, shape - self.seg_len, 3):
                        index[str(i % num_workers)].append([i, j, k])

        self.f_len = sum([len(index[i]) for i in index])
        self.index = index
        self.files = files
        self.feature_path = feature_path
        self.data = [None for _ in feature_path]

        print("num of files", sum([len(f) for f in self.files]))
        print("num of segs", self.f_len)

    def __len__(self):
        return self.f_len

    def __load_cache__(self, tid, fid, seg_id):
        if self.data[tid] is None:
            self.data[tid] = h5py.File(self.feature_path[tid], "r")
        fname = self.files[tid][fid]
        st = seg_id
        ed = st + self.seg_len
        data = self.data[tid][fname][st: ed]
        melody = data >= 2000
        data = data - melody*2000
        onsets = data >= 1000
        data = data - onsets*1000
        activations = data / 8.
        return np.stack([activations, onsets, melody], 0)

    def __getitem__(self, idx):

        worker_id = get_worker_info().id if self.use_loader else 0
        index = self.index[str(worker_id)]
        tid, fid, seg_id = index[int(idx % len(index))]
        return self.__load_cache__(tid, fid, seg_id)

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e

        for i in self.index:
            self.rng.shuffle(self.index[i])


def worker_init_fn(worker_id):
    pass


def collate_fn(batch):
    x = torch.from_numpy(np.stack([b for b in batch], 0)).float()
    return {
        "piano_roll": x,
    }
