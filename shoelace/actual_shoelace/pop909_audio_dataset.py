import os

import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset
from shoelace.pfMIDILM.MIDILM import PAD, SEG_RES


TOL_WIN = 2
DIGITS = [2 ** i for i in range(7)]
MAX_DUR = int(15.36 * 50)
SEG_LEN = MAX_DUR // SEG_RES + TOL_WIN
MAX_SEQ_LEN = 120



def load_data_lst(path_folder):
    list_folder = os.path.join(path_folder, "pop909", "text")
    feature_folder = os.path.join(path_folder, "pop909", "feature")
    files = []
    feature_path = []
    for f in os.listdir(list_folder):
        if f in ["6.lst", "7.lst"]:
            print(list_folder, f)
            continue
        path_lst = os.path.join(list_folder, f)
        f_path = os.path.join(feature_folder, str.replace(f, ".lst", ".h5"))

        with open(path_lst, "r") as pf:
            fs = pf.readlines()

        files.append([s.rstrip().split("\t")[0] for s in fs])
        feature_path.append(f_path)

    return files, feature_path


class AudioDataset(BaseDataset):
    def __init__(self, path_folder, rid, is_mono=True, num_workers=1, use_loader=True):
        super(AudioDataset, self).__init__()
        self.rid = rid
        self.use_loader = use_loader
        files, feature_path = load_data_lst(path_folder)
        num_workers = 1 if num_workers == 0 else num_workers
        index = {str(i): [] for i in range(num_workers)}
        tlen = [[] for _ in range(len(feature_path))]

        for i, data_path in enumerate(feature_path):
            with h5py.File(data_path, "r") as hf:
                tlen[i] = [0 for _ in range(len(files[i]))]
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):
                    if f + ".audio" not in hf:
                        continue
                    total_len = (hf[f + ".audio"].shape[0] - MAX_DUR)


                    for k in range(0, total_len, 50):
                        index[str(i % num_workers)].append([i, j, k])
        self.f_len = sum([len(index[i]) for i in index])
        self.index = index
        self.files = files
        self.feature_path = feature_path
        self.data = {}
        self.is_mono = is_mono

        print("is_mono", self.is_mono)
        print("num of files", sum([len(f) for f in self.files]))
        print("num of segs", self.f_len)

    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        # worker_id = get_worker_info().id if self.use_loader else 0
        index = self.index[str(0)]
        tid, fid, a_id = index[int(idx % len(index))]

        fname = self.files[tid][fid]
        if fname not in self.data:
            with h5py.File(self.feature_path[tid], "r") as hf:

                self.data[fname] = {
                    "audio": hf[fname + ".audio"][:],

                }


        audio_data = self.data[fname]["audio"][a_id: a_id + MAX_DUR]
        return audio_data

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e

        for i in self.index:
            self.rng.shuffle(self.index[i])


def worker_init_fn(worker_id):
    pass


def collate_fn(batch):
    audio_data = torch.from_numpy(np.stack([b for b in batch], 0)).long()

    return {
        "audio_seq": audio_data,
    }
