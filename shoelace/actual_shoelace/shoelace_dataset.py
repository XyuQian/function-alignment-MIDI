import os

import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset
from shoelace.pfMIDILM.MIDILM import PAD, SEG_RES

TOL_WIN = 2  # vocals2mel
# TOL_WIN = 2 #mel2vocals
DIGITS = [2 ** i for i in range(7)]
MAX_DUR = int(128 * 6)
SEG_LEN = MAX_DUR // SEG_RES + TOL_WIN
MAX_SEQ_LEN = 120


def load_data_lst(path_folder):
    list_folder = os.path.join(path_folder, "pop909", "text")
    feature_folder = os.path.join(path_folder, "pop909", "feature")
    files = []
    feature_path = []
    for f in os.listdir(list_folder):
        if f in ["6.lst"]:
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
    def __init__(self, path_folder, rid, is_mono=True, num_workers=1, use_loader=True):
        super(ShoelaceDataset, self).__init__()
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
                    total_len = (hf[f + ".audio"].shape[0] - MAX_DUR) // SEG_RES
                    sos_indices = hf[f + ".sos"][:]
                    res_sos_indices = hf[f + ".res_sos"][:]
                    if is_mono:
                        valid_melody_seg = hf[f + ".valid_melody_seg"][:]

                    for k in range(0, total_len):
                        if k >= len(sos_indices) - 1:
                            break

                        k_ed = k + SEG_LEN if k + SEG_LEN < len(sos_indices) else len(sos_indices) - 1
                        skip = False
                        if is_mono:
                            start_ed = k_ed
                            while valid_melody_seg[k: k_ed].sum() < k_ed - k:
                                k_ed -= 1
                                if k >= k_ed or start_ed - k_ed > TOL_WIN:
                                    skip = True
                                    break
                        if skip:
                            continue
                        res_st = res_sos_indices[k] if k < len(res_sos_indices) else -1
                        res_ed = res_sos_indices[k + 1] if k + 1 < len(res_sos_indices) else -1
                        e_st = sos_indices[k]
                        e_ed = sos_indices[k_ed]
                        if e_ed - e_st < 2:
                            continue
                        index[str(i % num_workers)].append([i, j, k * SEG_RES, e_st, e_ed, res_st, res_ed])
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
        tid, fid, a_id, e_st, e_ed, res_st, res_ed = index[int(idx % len(index))]


        fname = self.files[tid][fid]
        if fname not in self.data:
            with h5py.File(self.feature_path[tid], "r") as hf:
                events = hf[fname + ".events"][:]
                res_events = hf[fname + ".res_events"][:]

                self.data[fname] = {
                    "audio": hf[fname + ".audio"][:],
                    "vocals": hf[fname + ".audio.vocals"][:],
                    "acc": hf[fname + ".audio.acc"][:],
                    "events": events,
                    "res_events": res_events,
                    "index": hf[fname + ".index"][:],
                }

        tag = "vocals" if self.is_mono else "audio"
        r = self.rng.randint(0, 60) - 30
        na_id = a_id + r
        if na_id < 0:
            na_id = 0
        if na_id + MAX_DUR > len(self.data[fname][tag]):
            na_id = len(self.data[fname][tag]) - MAX_DUR

        shift_r = na_id - a_id
        a_id = na_id


        audio_data = self.data[fname][tag][a_id: a_id + MAX_DUR]
        midi_data = self.data[fname]["events"][e_st: e_ed]

        index = self.data[fname]["index"][e_st: e_ed]

        index = index - index[0]

        if self.is_mono:
            ind = midi_data[:, 1] > 0
            if ind.sum() < 2:
                return self.__getitem__((idx + 1) % len(self.index[str(0)]))

        # if res_ed > res_st:
        #     prefix = self.data[fname]["res_events"][res_st: res_ed][:]
        #     midi_data = np.concatenate([midi_data[:1], prefix, midi_data[1:]], 0)
        #     prefix_pos = np.zeros([len(prefix), index.shape[-1]])
        #     prefix_pos[:, 1] = prefix[:, 3] * SEG_RES + prefix[:, 4]
        #     index = np.concatenate([index[:1], prefix_pos, index[1:]], 0)

        midi_data[midi_data < 0] = PAD

        if self.is_mono:
            ind = midi_data[:, 1] > 0
            midi_data = midi_data[ind]
            index = index[ind]
            if len(index) < 2:
                return self.__getitem__((idx + 1) % len(self.index[str(0)]))

        if len(midi_data) > MAX_SEQ_LEN:
            midi_data = midi_data[:MAX_SEQ_LEN]
            index = index[:MAX_SEQ_LEN]
        index = np.pad(index[:-1], ((1, 0), (0, 0)), "constant", constant_values=0)
        return midi_data, audio_data, index

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e

        for i in self.index:
            self.rng.shuffle(self.index[i])


def worker_init_fn(worker_id):
    pass


def collate_fn(batch):
    audio_data = torch.from_numpy(np.stack([b[1] for b in batch], 0)).long()

    min_len = min([len(x[0]) for x in batch])
    seq = []
    index = []
    for b in batch:
        seq.append(b[0][:min_len])
        index.append(b[2][:min_len])
    midi_data = torch.from_numpy(np.stack(seq, 0)).long()
    midi_index = torch.from_numpy(np.stack(index, 0)).long()
    # print(midi_data.shape)

    return {
        "midi_seq": midi_data,
        "audio_seq": audio_data,
        "midi_index": midi_index,
        "audio_index": None
    }
