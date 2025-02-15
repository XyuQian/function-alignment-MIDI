import torch
from shoelace.pfMIDILM.MIDILM import PAD, SEG_RES
import torch.nn.functional as F


def make_index(midi_seq, chunk_len):
    pos = torch.zeros_like(midi_seq[:, 0, 0])
    index = torch.zeros_like(midi_seq[:, :, 0])
    stop_iter = True if chunk_len is not None else False
    for i in range(len(midi_seq)):
        for k in range(1, len(midi_seq[0])):
            if midi_seq[i, k, 0] == PAD:
                break
            if midi_seq[i, k, 0] == SEG_RES:
                pos[i] += SEG_RES
                index[i, k] = pos[i]
            elif midi_seq[i, k, 0] == SEG_RES + 1:
                index[i, k] = pos[i]
            else:
                index[i, k] = pos[i] + midi_seq[i, k, 0]
        if chunk_len is not None and pos[i] / SEG_RES < chunk_len:
            stop_iter = False
    index = torch.stack([index, index], -1).long()
    return F.pad(index, (0, 0, 1, 0), "constant", 0), stop_iter


def crop_midi(seq, chunk_len, hop_len, align_tail=False):
    prefix_idx = []
    suffix_idx = []
    prefix = []


    for i in range(len(seq)):
        pos = 0
        prefix.append([])
        for j in range(len(seq[i])):
            if seq[i, j, 0] == -1:
                break
            if seq[i, j, 0] == SEG_RES + 1:
                continue

            if seq[i, j, 0] == SEG_RES:
                pos += 1
                if pos == hop_len + 1:
                    prefix_idx.append(j)
                elif pos == chunk_len + 1:
                    suffix_idx.append(j)

            if pos < hop_len + 1 <= seq[i, j, 3] + pos and 0 < seq[i, j, 3] < SEG_RES:
                s = [SEG_RES + 1,
                     seq[i, j, 1],
                     seq[i, j, 2],
                     seq[i, j, 3] + pos - hop_len - 1,
                     seq[i, j, 4],
                     seq[i, j, 5]]
                prefix[i].append(torch.asarray(s).long().to(seq[i].device))

    gen_seq = []
    prompts = []
    min_len = max([suffix_idx[i] - prefix_idx[i] for i in range(len(suffix_idx))]) + len(prefix[i])
    for i in range(len(seq)):
        s = seq[i, :prefix_idx[i]]
        s = s[s[:, 0] < SEG_RES + 1]
        gen_seq.append(s)
        if chunk_len == hop_len:
            continue
        s = seq[i, prefix_idx[i]:suffix_idx[i]] if not align_tail else seq[i, prefix_idx[i]:]

        if len(prefix[i]) > 0:
            prefix[i] = torch.stack(prefix[i], 0)
            s = torch.concat([s[:1], prefix[i], s[1:]], 0)

        if align_tail:
            prompts.append(s[:min_len])
        else:
            prompts.append(s)

    return gen_seq, prompts


def crop_midi_and_align(midi_seq, start_idx, min_chunk_len):
    _, prompts = crop_midi(midi_seq, chunk_len=start_idx + min_chunk_len,
                           hop_len=start_idx, align_tail=True)
    prompts = torch.stack(prompts, 0)
    index, _ = make_index(prompts, chunk_len=0)
    return prompts, index
