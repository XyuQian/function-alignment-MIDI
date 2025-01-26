import os

import torch

device = "cuda"

names = [
    ["909", 30],
    ["803", 91],
    ["860", 36],
    ["757", 96],
    ["765", 75],
    ["873", 30],
    ["164", 18],
    ["686", 58],
]
fid = "6"


def test_shoelace(mode):
    from shoelace.actual_shoelace.inference_helper import InferenceHelper
    import numpy as np
    import h5py


    inference_helper = InferenceHelper(models=[
        ["vocals2mel", "4_end"]
    ], device=device)

    midi_seq = []
    audio_seq = []
    with h5py.File(f"data/formatted/groups/pop909_tokens/{fid}.h5", "r") as hf:
        for n, nid in names:
            print(f"data/POP909/{n}/{n}.mid")
            st = int(nid * 50 / 16)
            mel = hf[f"data/POP909/{n}/{n}.mid.mel"][:][st: st + 50]
            # acc = hf[f"data/POP909/{n}/{n}.mid.acc"][:][st: st + 50]
            mel = np.reshape(mel, [-1, 3, 4])
            # acc = np.reshape(acc, [-1, 3, 4])
            # midi = np.stack([mel, acc], 2)
            # midi = np.reshape(midi, [-1, 24])
            midi = np.reshape(mel, [-1, 24])
            midi_seq.append(midi)
            audio_seq.append(hf[f"data/POP909/{n}/{n}.mid.audio"][:][int(nid * 50):int(nid * 50) + 16 * 50])

    audio_seq = torch.from_numpy(np.stack(audio_seq, 0)).to(device)
    ref_seq = audio_seq.transpose(1, 2)
    midi_seq = torch.from_numpy(np.stack(midi_seq, 0)).to(device).long()
    desc = ["a melodic pop song"] * len(names)

    results = []
    modes = {
        "a2s": "vocals2mel",
        "s2a": "mel2vocals",
        "walk": "walk"
    }
    m = modes[mode]
    print(midi_seq.shape)
    for i in range(3):
        print("sample#", i + 1)
        results.append(inference_helper.inference(midi_seq=midi_seq,
                                                  audio_seq=audio_seq,
                                                  mode=m))

    if results[0]["audio_pred"] is not None:
        audio_pred = torch.stack([r["audio_pred"] for r in results], 1).flatten(0, 1)
    else:
        audio_pred = None

    if results[0]["midi_pred"] is not None:
        midi_pred = torch.stack([r["midi_pred"] for r in results], 1).flatten(0, 1)
    else:
        midi_pred = None

    folder = f"yy_test_results/{m_id}"
    os.makedirs(folder, exist_ok=True)

    if audio_pred is not None:
        np.save(os.path.join(folder, f"test_{mode}_audio.npy"), audio_pred.cpu().numpy())
        np.save(os.path.join(folder, f"ref_audio.npy"), ref_seq[:, :, :audio_pred.shape[-1]].cpu().numpy())

    if midi_pred is not None:
        np.save(os.path.join(folder, f"test_{mode}_midi.npy"), midi_pred.cpu().numpy())
        np.save(os.path.join(folder, f"ref_midi.npy"), midi_seq.cpu().numpy())






def pred_2_audio(m_id, mode):
    import numpy as np
    from shoelace.utils.encodec_utils import save_rvq
    folder = f"yy_test_results/{m_id}"
    pred_data = np.load(os.path.join(folder, f"test_{mode}_audio.npy"))
    ref_data = np.load(os.path.join(folder, "ref_audio.npy"))
    print(pred_data.shape, ref_data.shape)
    n = len(ref_data)
    m = len(pred_data) // n
    ref_data = ref_data[:n]
    audio_codes = np.concatenate([pred_data,
                                  ref_data], 0)

    audio_codes = audio_codes[:, :, :-5]

    audio_codes = torch.from_numpy(audio_codes).to(device).long()
    filename_list = [

    ]
    audio_folder = os.path.join(folder, f"audio_{mode}")
    os.makedirs(audio_folder, exist_ok=True)
    ref_audio_folder = os.path.join(folder, "audio_ref")
    os.makedirs(ref_audio_folder, exist_ok=True)

    for i in range(n * m):
        n_i = i // m
        m_i = i % m
        filename_list.append(os.path.join(audio_folder, f"pred_{n_i}_{m_i}"))

    for i in range(n):
        filename_list.append(os.path.join(ref_audio_folder, f"ref_{i}"))
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


def pred_2_midi(m_id, mode):
    from shoelace.pianoroll_vq.base_vq import MIDIRVQ, predict
    from shoelace.utils.utils import data2midi
    import numpy as np
    folder = f"yy_test_results/{m_id}"
    midi_folder = os.path.join(folder, f"midis_{mode}")
    os.makedirs(midi_folder, exist_ok=True)
    ref_folder = os.path.join(folder, "midis_ref")
    os.makedirs(ref_folder, exist_ok=True)

    model = MIDIRVQ(modes=["chords", "cond_onset", "cond_pitch"], main_mode="cond_pitch").to(device)
    model.set_config(path_dict={"chords": "save_models/chords.pth",
                                "cond_onset": "save_models/cond_onset.pth",
                                "cond_pitch": "save_models/cond_pitch.pth",
                                }, device=device)

    res = []
    for f in [f"test_{mode}_midi.npy", "ref_midi.npy"]:
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
                        if cls[i] == "ref":
                            path = os.path.join(ref_folder, f"{k}_{names[k][0]}_{tags[j]}_{cls[i]}.mid")
                        else:
                            path = os.path.join(midi_folder, f"{k}_{names[k][0]}_{m}_{tags[j]}_{cls[i]}.mid")
                        data2midi(pred[1][n * k + m].cpu().numpy() > .5,
                                  (pred[2][n * k + m].cpu().numpy() > .5) * 100,
                                  path)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1]
    if mode in ["a2s", "s2a", "walk"]:
        test_shoelace(mode)
    elif mode == "s2a_decode":
        print("pred_2_audio")
        pred_2_audio(sys.argv[2], mode="s2a")
    elif mode == "a2s_decode":
        print("pred_2_midi")
        pred_2_midi(sys.argv[2], mode="a2s")
    elif mode == "walk_decode":
        print("pred_2_audio")
        pred_2_audio(sys.argv[2], mode="walk")
        print("pred_2_midi")
        pred_2_midi(sys.argv[2], mode="walk")
