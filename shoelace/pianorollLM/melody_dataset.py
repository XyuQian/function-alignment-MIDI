import os

import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset, get_worker_info

PAD_VAL = -1
EOS = 384
PAD = 385
STRIDE = 32
SEQ_LEN = 24

def load_data_lst(path_folder):
    list_folder = os.path.join(path_folder, "las_melody", "text")
    feature_folder = os.path.join(path_folder, "las_melody", "feature")
    files = []
    feature_path = []
    for f in os.listdir(list_folder):
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
            with h5py.File(data_path, "r") as hf:
                tlen[i] = [0 for _ in range(len(files[i]))]
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):
                    if f not in hf:
                        continue
                    total_len = hf[f].shape[0]
                    tlen[i][j] = total_len

                    if total_len > seg_len:
                        for k in range(0,  total_len - seg_len, 50):
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
        mel_data = self.data[tid][fname][st: ed][:]
        if len(mel_data) < self.seg_len:
            mel_data = np.pad(mel_data, (0, self.seg_len - ed + st), "constant", constant_values=-1)
        return mel_data

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
    seq = torch.from_numpy(np.stack(batch, 0))
    bs = len(seq)
    x = seq.view(-1, STRIDE)
    n = len(x)
    eos = torch.zeros([n, 1]) + EOS
    x = torch.concat([x, eos], -1).long()
    idx = torch.arange(STRIDE + 1)[None, ...].repeat(n, 1)
    mask = x > 128
    mask[:, 0] = True
    mask[x < 0] = False

    melody = torch.zeros([n, SEQ_LEN, 2]).long()
    melody_id = torch.arange(SEQ_LEN)[None, ...].repeat(n, 1)
    melody[..., 0] = PAD
    melody[..., 1] = STRIDE
    target_mask = mask.sum(-1)[..., None].repeat(1, SEQ_LEN)
    target_mask = melody_id < target_mask
    melody[..., 0][target_mask] = x[mask]
    melody[..., 1][target_mask] = idx[mask]
    melody = melody.view(bs, -1, SEQ_LEN, 2)
    seq = seq.view(bs, -1, STRIDE)
    seq[seq == -1] = PAD

    # print("==========================================")
    # print(melody[0, 0, :, 0])
    # print(melody[0, 0, :, 1])
    # print("------------------------------------------")
    # print(x[0])
    # print("==========================================")

    return {
        "melody": melody,
        "seq": seq,
    }
