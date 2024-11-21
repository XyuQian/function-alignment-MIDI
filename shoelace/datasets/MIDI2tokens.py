import os
import sys
import numpy as np
import h5py
import torch
from shoelace.pianoroll_vq.base_vq import MIDIRVQ as Model

device = "cuda"


def remove_sil(data):
    i = 0
    while data[i].sum() <= 0:
        i += 1
    j = len(data) - 1
    while data[j].sum() <= 0:
        j -= 1

    return data[i: j + 1], i, j + 1

def midi2tokens(model, x):
    seg_len = x.shape[1]
    seg_len = int(seg_len // 16) * 16
    x = x[:, :seg_len]
    x = torch.from_numpy(x)
    x = x.to(device).float()
    with torch.no_grad():
        tokens = model.get_indices(x[None, ...])
    tokens = tokens.squeeze(0).cpu().numpy()
    return tokens


def get_model():
    model = Model(modes=["chords",
                         "cond_onset",
                         "cond_pitch"], main_mode="cond_pitch").to(device)
    model.set_config(path_dict={"chords": "save_models/chords.pth",
                                "cond_onset": "save_models/cond_onset.pth",
                                "cond_pitch": "save_models/cond_pitch.pth",
                                }, device=device)
    model.eval()
    return model


def read_data(h5_path, output_path, file_lst_path):
    model = get_model()

    with open(file_lst_path, "r") as f:
        files = f.readlines()
    files = [f.rstrip() for f in files]

    with h5py.File(h5_path, "r") as hf:
        with h5py.File(output_path, "a") as whf:
            for i, line in enumerate(files):
                if line in whf:
                    continue
                data = hf[line][:]
                data, _, _ = remove_sil(data)
                onsets = data >= 1000
                end = data < 0
                activations = (data - onsets * 1000 + end) / 8.
                x = np.stack([activations, onsets, end], 0)
                t = x.shape[1]
                seg = int(32 * 50 * 60)
                if t > seg:
                    n = t // seg + 1
                    for j in range(n):
                        st = int(j * seg)
                        ed = min(st + seg, t)
                        if ed - st < 30 * 50:
                            break
                        xt = x[:, st:ed]
                        tokens = midi2tokens(model, xt)
                        whf.create_dataset(line + f".{j}", data=tokens.astype(np.int16))
                else:
                    tokens = midi2tokens(model, x)
                    whf.create_dataset(line, data=tokens.astype(np.int16))


if __name__ == "__main__":
    fid = sys.argv[1]
    token_folder = "data/formatted/groups/tokens"
    os.makedirs(
        token_folder, exist_ok=True
    )
    output_path = os.path.join(token_folder, f"{fid}.h5")
    h5_path = f"data/formatted/groups/feature/{fid}.h5"
    file_lst_path = f"data/formatted/groups/text/{fid}.lst"
    read_data(h5_path, output_path, file_lst_path)
