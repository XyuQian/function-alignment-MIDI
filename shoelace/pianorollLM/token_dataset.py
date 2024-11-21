import os

import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset, get_worker_info

DIGITS = [2 ** i for i in range(7)]


def load_data_lst(path_folder):
    list_folder = os.path.join(path_folder, "pop909_text")
    feature_folder = os.path.join(path_folder, "pop909_tokens")
    files = []
    feature_path = []
    for f in os.listdir(list_folder):
        if f == "6.lst":
            print(list_folder, f)
            continue
        path_lst = os.path.join(list_folder, f)
        f_path = os.path.join(feature_folder, str.replace(f, ".lst", ".h5"))
        with open(path_lst, "r") as pf:
            fs = pf.readlines()

        files.append([s.rstrip().split("\t")[0] for s in fs])
        feature_path.append(f_path)

    return files, feature_path


class TokenDataset(BaseDataset):
    def __init__(self, path_folder, rid, seg_len, num_workers=1, use_loader=True):
        super(TokenDataset, self).__init__()
        self.rid = rid
        self.seg_len = seg_len
        self.use_loader = use_loader
        files, feature_path = load_data_lst(path_folder)
        num_workers = 1 if num_workers == 0 else num_workers
        index = {str(i): [] for i in range(num_workers)}
        tlen = [[] for _ in range(len(feature_path))]

        for i, data_path in enumerate(feature_path):
            add_files = []
            with h5py.File(data_path, "r") as hf:
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):
                    if f not in hf:
                        for k in range(20):
                            new_f = f + f".{k}"
                            if new_f in hf:
                                add_files.append(new_f)
                            else:
                                break
                files[i] += add_files

        for i, data_path in enumerate(feature_path):
            with h5py.File(data_path, "r") as hf:
                tlen[i] = [0 for _ in range(len(files[i]))]
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):
                    f = f + ".acc"
                    if f not in hf:
                        continue
                    total_len = hf[f].shape[0]
                    tlen[i][j] = total_len
                    if total_len > seg_len // 8:
                        for k in range(0,  total_len - seg_len // 8, 1):
                            index[str(i % num_workers)].append([i, j, k])
                    else:
                        index[str(i % num_workers)].append([i, j, 0])
        self.tlen = tlen
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
        if ed > self.tlen[tid][fid]:
            ed = self.tlen[tid][fid]

        acc_data = np.reshape(self.data[tid][fname + ".acc"][st: ed][:], [-1, 3, 4])
        mel_data = np.reshape(self.data[tid][fname + ".mel"][st: ed][:], [-1, 3, 4])
        melody = self.data[tid][fname + ".melody"][st: ed][:]
        data = np.reshape(np.stack([mel_data, acc_data], 2), [-1, 24])
        return data, melody[:, 0]

    def __getitem__(self, idx):
        # worker_id = get_worker_info().id if self.use_loader else 0
        index = self.index[str(0)]
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
    max_t = max([b[0].shape[0] for b in batch])
    outs = []
    for b in batch:
        piano = np.pad(b[0], ((0, max_t - b[0].shape[0]), (0, 0)), 'constant', constant_values=(512, 512))
        melody = np.pad(b[1], (0, max_t - b[1].shape[0]), 'constant', constant_values=(128, 128))
        outs.append([piano, melody])
    x = torch.from_numpy(np.stack([out[0] for out in outs], 0)).long()
    melody = torch.from_numpy(np.stack([out[1] for out in outs], 0)).long()
    return {
        "x": x,
        "melody": melody
    }
