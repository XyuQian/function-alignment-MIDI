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


class ShoelaceDataset(BaseDataset):
    def __init__(self, path_folder, rid, seg_sec=16, num_workers=1, use_loader=True):
        super(ShoelaceDataset, self).__init__()
        self.rid = rid
        self.use_loader = use_loader
        self.midi_seg_len = int(seg_sec * 50 // 16)
        files, feature_path = load_data_lst(path_folder)
        num_workers = 1 if num_workers == 0 else num_workers
        index = {str(i): [] for i in range(num_workers)}
        tlen = [[] for _ in range(len(feature_path))]

        for i, data_path in enumerate(feature_path):
            with h5py.File(data_path, "r") as hf:
                tlen[i] = [0 for _ in range(len(files[i]))]
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):

                    total_len = min(hf[f + ".acc"].shape[0], hf[f + ".audio"].shape[0] // 16)
                    for k in range(0, total_len - self.midi_seg_len, 2):
                        index[str(i % num_workers)].append([i, j, k])
        self.f_len = sum([len(index[i]) for i in index])
        self.index = index
        self.files = files
        self.feature_path = feature_path
        self.data = {}

        print("num of files", sum([len(f) for f in self.files]))
        print("num of segs", self.f_len)

    def __len__(self):
        return self.f_len

    def __load_cache__(self, tid, fid, seg_id):
        fname = self.files[tid][fid]
        if fname not in self.data:
            with h5py.File(self.feature_path[tid], "r") as hf:
                self.data[fname] = {
                    "acc": hf[fname + ".acc"][:],
                    "audio": hf[fname + ".audio"][:],
                    "audio_acc": hf[fname + ".audio.acc"][:],
                    "audio_mel": hf[fname + ".audio.mel"][:],
                    "mel": hf[fname + ".mel"][:],
                    "melody": hf[fname + ".melody"][:],
                }

        midi_st = seg_id
        midi_ed = midi_st + self.midi_seg_len

        audio_st = midi_st * 16
        audio_ed = midi_ed * 16

        # acc_data = np.reshape(self.data[fname]["acc"][midi_st: midi_ed], [-1, 3, 4])
        mel_data = self.data[fname]["mel"][midi_st: midi_ed]


        audio_data = self.data[fname]["audio_mel"][audio_st: audio_ed]
        # melody_data = self.data[fname]["melody"][midi_st: midi_ed, 0]

        return mel_data, audio_data

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
    mel_data = torch.from_numpy(np.stack([b[0] for b in batch], 0)).long()
    audio_data = torch.from_numpy(np.stack([b[1] for b in batch], 0)).long()
    # melody_data = torch.from_numpy(np.stack([b[2] for b in batch], 0)).long()
    audio_data = audio_data.transpose(1, 2)
    return {
        "midi_seq": mel_data,
        "audio_seq": audio_data
    }
