import os

import torch

device = "cuda"

names = [
    "909",
    "803",
    "860",
    "757",
    "765",
    "873",
    "119",
    "164",
    "686"
    #
    # "737",
    # "778",
    # "772",
    # "434",
    # "856"
]
fid = "6"


def test_shoelace(mode, m_id):
    from shoelace.actual_shoelace.shoelace_2 import Yingyang
    # from shoelace.utils.encodec_utils import save_rvq
    import numpy as np
    import h5py
    is_mono = True
    model = Yingyang(is_mono=is_mono)

    model.load_weights(f"exp/mono_shoelace/latest_{m_id}.pth")
    model = model.to(device)
    model.set_config(device)

    midi_seq = []
    audio_seq = []
    mono_str = ".mel" if is_mono else ""
    with h5py.File(f"data/formatted/groups/pop909_tokens/{fid}.h5", "r") as hf:
        for n in names:
            print(f"data/POP909/{n}/{n}.mid")
            mel = hf[f"data/POP909/{n}/{n}.mid.mel"][:][50 * 5: 50 * 6]
            acc = hf[f"data/POP909/{n}/{n}.mid.acc"][:][50 * 5: 50 * 6]
            mel = np.reshape(mel, [-1, 3, 4])
            acc = np.reshape(acc, [-1, 3, 4])
            if is_mono:
                acc = np.zeros_like(acc) + 512
            midi = np.stack([mel, acc], 2)
            midi = np.reshape(midi, [-1, 24])
            midi_seq.append(midi)
            audio_seq.append(hf[f"data/POP909/{n}/{n}.mid.audio" + mono_str][:][5 * 16 * 50:6 * 16 * 50])

    audio_seq = torch.from_numpy(np.stack(audio_seq, 0)).to(device)
    ref_seq = audio_seq.transpose(1, 2)
    midi_seq = torch.from_numpy(np.stack(midi_seq, 0)).to(device).long()
    desc = ["a melodic pop song"] * len(names)
    model.eval()

    results = []
    print(midi_seq.shape)
    for i in range(5):
        print("sample#", i + 1)
        if i == 0:
            midi_top_k = 1
            audio_top_k = 50
        else:
            midi_top_k = 2
            audio_top_k = 100
        results.append(model.inference(midi_seq=midi_seq,
                                       audio_seq=audio_seq,
                                       desc=desc,
                                       mode=mode,
                                       midi_top_k=midi_top_k,
                                       audio_top_k=audio_top_k))

    print(audio_seq.shape)
    audio_seq = torch.stack(results, 1).flatten(0, 1)
    folder = f"yy_test_results/{m_id}"
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f"test_{mode}.npy"), audio_seq.cpu().numpy())
    if mode == "a2b":
        np.save(os.path.join(folder, f"ref_{mode}.npy"), ref_seq[:, :, :audio_seq.shape[-1]].cpu().numpy())
    else:
        np.save(os.path.join(folder, f"ref_{mode}.npy"), midi_seq.cpu().numpy())


def pred_2_audio(m_id):
    import numpy as np
    from shoelace.utils.encodec_utils import save_rvq
    folder = f"yy_test_results/{m_id}"
    pred_data = np.load(os.path.join(folder, "test_a2b.npy"))
    ref_data = np.load(os.path.join(folder, "ref_a2b.npy"))
    n = len(ref_data)
    m = len(pred_data) // n
    print(pred_data.shape, ref_data.shape)
    audio_codes = np.concatenate([pred_data,
                                  ref_data], 0)

    audio_codes = torch.from_numpy(audio_codes).to(device).long()
    filename_list = [

    ]
    audio_folder = os.path.join(folder, "audio")
    os.makedirs(audio_folder, exist_ok=True)
    for i in range(n * m):
        n_i = i // m
        m_i = i % m
        filename_list.append(os.path.join(audio_folder, f"pred_{n_i}_{m_i}"))

    for i in range(n):
        filename_list.append(os.path.join(audio_folder, f"ref_{i}"))
    print(audio_codes.shape)
    print(torch.max(audio_codes), torch.min(audio_codes))

    n_blocks = 20
    for i in range(0, len(filename_list), n_blocks):
        ed = i + n_blocks
        if ed > len(filename_list):
            ed = len(filename_list)
        print("converting", i, ed)
        save_rvq(output_list=filename_list[i: ed],
                 tokens=audio_codes[i: ed].long())


def pred_2_midi(m_id):
    from shoelace.pianoroll_vq.base_vq import MIDIRVQ, predict
    from shoelace.utils.utils import data2midi
    import numpy as np
    folder = f"yy_test_results/{m_id}"
    midi_folder = os.path.join(folder, "midis")
    os.makedirs(midi_folder, exist_ok=True)

    model = MIDIRVQ(modes=["chords", "cond_onset", "cond_pitch"], main_mode="cond_pitch").to(device)
    model.set_config(path_dict={"chords": "save_models/chords.pth",
                                "cond_onset": "save_models/cond_onset.pth",
                                "cond_pitch": "save_models/cond_pitch.pth",
                                }, device=device)

    res = []
    for f in ["test_b2a.npy", "ref_b2a.npy"]:
        pred_data = np.load(os.path.join(folder, f))
        pred_data = torch.from_numpy(pred_data).to(device)
        total_tokens = pred_data.view(len(pred_data), -1, 3, 2, 4)
        melody_tokens = total_tokens[:, :, :, 0].flatten(2, 3)
        acc_tokens = total_tokens[:, :, :, 1].flatten(2, 3)
        res.append([melody_tokens, acc_tokens])

    cls = ["pred", "ref"]
    tags = ["mel", "acc"]
    with torch.no_grad():
        for i in range(len(cls)):
            for j, tokens in enumerate(res[i]):
                if j > 0:
                    continue
                pred = model.decode_from_indices(tokens.long())
                pred = predict(pred, len(tokens))
                for p in pred:
                    print(p.shape)
                n = len(tokens) // len(names)
                for k in range(len(names)):
                    for m in range(n):
                        path = os.path.join(midi_folder, f"{k}_{names[k]}_{m}_{tags[j]}_{cls[i]}.mid") if n > 1 else \
                            os.path.join(midi_folder, f"{k}_{names[k]}_{tags[j]}_{cls[i]}.mid")
                        data2midi(pred[1][n * k + m].cpu().numpy() > .5,
                                  (pred[2][n * k + m].cpu().numpy() > .5) * 100,
                                  path)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1]
    m_id = sys.argv[2]
    if mode == "a2b":
        test_shoelace("a2b", m_id)
    elif mode == "b2a":
        test_shoelace("b2a", m_id)
    elif mode == "a2b_decode":
        print("pred_2_audio")
        pred_2_audio(m_id)
    elif mode == "b2a_decode":
        print("pred_2_midi")
        pred_2_midi(m_id)
