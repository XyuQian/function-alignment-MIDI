import os

import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset, get_worker_info
from .MIDILM import PAD, MAX_SEQ_LEN


def load_data_lst(path_folder):
    # list_folder = os.path.join(path_folder, "las", "dur_lt_30_text_beta")
    # feature_folder = os.path.join(path_folder, "las", "dur_lt_30_text_beta_midis")
    list_folder = os.path.join(path_folder, "las", "dur_lt_30_text_beta")
    feature_folder = os.path.join(path_folder, "las", "dur_lt_30_text_beta_midis")
    files = []
    feature_path = []
    for f in os.listdir(list_folder):
        # if f == "0.lst":
        #     continue
        path_lst = os.path.join(list_folder, f)
        f_path = os.path.join(feature_folder, str.replace(f, ".lst", ".h5"))
        with open(path_lst, "r") as pf:
            fs = pf.readlines()
        files.append([s.rstrip().split("\t")[0] for s in fs])
        feature_path.append(f_path)

    return files, feature_path


class MIDIDataset(BaseDataset):
    def __init__(self, path_folder, rid, num_workers=1):
        super(MIDIDataset, self).__init__()
        self.rid = rid
        files, feature_path = load_data_lst(path_folder)
        num_workers = 1 if num_workers == 0 else num_workers
        self.use_loader = num_workers > 1
        index = {str(i): [] for i in range(num_workers)}
        tlen = [[] for _ in range(len(feature_path))]

        for i, data_path in enumerate(feature_path):
            with h5py.File(data_path, "r") as hf:
                tlen[i] = [0 for _ in range(len(files[i]))]
                for j, f in tqdm(enumerate(files[i]), total=len(files),
                                 desc=f"prepare dataset {i} / {len(feature_path)}"):
                    if f + ".sos" not in hf:
                        continue
                    sos_indices = hf[f + ".sos"][:]
                    res_sos_indices = hf[f + ".res_sos"][:]
                    if len(sos_indices) == 1:
                        continue
                    # print("-------------")
                    # print(hf[f + ".events"][:10][:])
                    # print(hf[f + ".res_events"][:10][:])
                    # print(hf[f + ".sos"][:10][:])
                    # print(hf[f + ".res_sos"][:10][:])
                    # print("-------------")

                    tlen[i][j] = sos_indices[-1]
                    assert sos_indices[-1] > 0
                    pre_sid = -2333
                    for s, sid in enumerate(sos_indices[:-1]):
                        # if sid - pre_sid < 512:
                        #     continue
                        if not s % 5 == 0:
                            continue
                        # pre_sid = sid
                        res_st = res_sos_indices[s] if s < len(res_sos_indices) else -1
                        res_ed = res_sos_indices[s + 1] if s + 1 < len(res_sos_indices) else -1
                        index[str(i % num_workers)].append([i, j, sid, res_st, res_ed])
                        if tlen[i][j] - sid < MAX_SEQ_LEN:
                            break

        self.f_len = sum([len(index[i]) for i in index])
        self.index = index
        self.files = files
        self.feature_path = feature_path
        self.data = [None for _ in feature_path]
        self.tlen = tlen

        print("num of files", sum([len(f) for f in self.files]))
        print("num of segs", self.f_len)

    def __len__(self):
        return self.f_len

    def __load_cache__(self, tid, fid, sid, res_st, res_ed):
        if self.data[tid] is None:
            self.data[tid] = h5py.File(self.feature_path[tid], "r")
        fname = self.files[tid][fid]
        eid = sid + MAX_SEQ_LEN
        eid = self.tlen[tid][fid] if self.tlen[tid][fid] < eid else eid
        data = self.data[tid][fname + ".events"][sid: eid][:]
        # if res_ed > res_st:
        #     prefix = self.data[tid][fname + ".res_events"][res_st: res_ed][:]
        #     data = np.concatenate([data[:1], prefix, data[1:]], 0)
        data[data < 0] = PAD
        if len(data) > MAX_SEQ_LEN:
            data = data[:MAX_SEQ_LEN]

        return data

    def __getitem__(self, idx):
        worker_id = get_worker_info().id if self.use_loader else 0
        index = self.index[str(worker_id)]
        tid, fid, sid, res_st, res_ed = index[int(idx % len(index))]
        return self.__load_cache__(tid, fid, sid, res_st, res_ed)

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e

        for i in self.index:
            self.rng.shuffle(self.index[i])


def worker_init_fn(worker_id):
    pass


def collate_fn(batch):
    max_len = max([len(x) for x in batch])
    seq = []
    for x in batch:
        if len(x) < max_len:
            x = np.pad(x, ((0, max_len - len(x)), (0, 0)), "constant", constant_values=(PAD, PAD))
        seq.append(x)
    midi_seq = torch.from_numpy(np.stack(seq, 0)).long()
    return {
        "x": midi_seq
    }


if __name__ == "__main__":
    midi_dataset = MIDIDataset(path_folder="data/formatted/", rid=0, num_workers=1)
    for i in range(10):
        event = midi_dataset.__getitem__(i)
        print("=================================")
        for j in range(20):
            print(event[j])
