import torch
from .shoelace import Yinyang
from shoelace.pfMIDILM.MIDILM import PAD, SEG_RES
import torch.nn.functional as F
from .utils import crop_midi, crop_midi_and_align


class InferenceHelper():
    def __init__(self, models, device):
        net = {}
        for path, mode in models:
            model = Yinyang(sec=16, mode=mode)
            model.load_weights(path)
            model = model.to(device)
            model.set_config(device)
            model.eval()
            net[mode] = model

        self.models = net

    def inference(self, midi_seq, audio_seq, mode):
        if mode == "mel2vocals":
            results = self.midi2audio(midi_seq, max_audio_len=audio_seq.shape[1])
        elif mode == "vocals2mel":
            results = self.audio2midi(audio_seq)
        return results

    def audio2midi(self, audio_seq):
        n_steps = 1024
        midi_prompt = torch.zeros([len(audio_seq), 1, 6]).to(audio_seq.device)
        midi_prompt[:, 0, 1:] = PAD
        midi_prompt[:, 0, 0] = SEG_RES
        chunk_len = 6
        hop_len = 1
        ch_offset = 3

        midi_seq = None

        for audio_idx in range(0, audio_seq.shape[1] // 128, hop_len):

            st = audio_idx * SEG_RES
            ed = st + chunk_len * SEG_RES
            if ed > audio_seq.shape[1]:
                ed = audio_seq.shape[1]
            print(f"[Total len {audio_seq.shape[1]}] decode chunk {st} to {ed}...")
            audio_prompt = audio_seq[:, st: ed]
            results = self.models["vocals2mel"].inference(
                seqs={"audio_prompt": audio_prompt,
                      "midi_prompt": midi_prompt,
                      "audio_index": None,
                      "midi_index": None,
                      "audio_top_k": 100,
                      "midi_top_k": 16,
                      "audio_max_gen_len": audio_prompt.shape[1] + 3,
                      "midi_max_gen_len": chunk_len - ch_offset,
                      "num_samples": len(audio_seq),
                      "audio_walk_steps": [0] * n_steps,
                      "midi_walk_steps": [1] * n_steps,
                      "device": "cuda"},
                n_steps=n_steps)

            midi_pred = results["midi_pred"]
            # print(midi_pred[0])

            gen_seq, prompts = crop_midi(midi_pred, chunk_len - ch_offset, hop_len)
            if midi_seq is None:
                midi_seq = gen_seq
            else:
                midi_seq = [torch.concat([midi_seq[j], gen_seq[j]], 0) for j in range(len(gen_seq))]

            max_len = max([len(p) for p in prompts])
            prompts = [F.pad(p, (0, 0, 0, max_len - len(p)), "constant", -1) for p in prompts]

            midi_prompt = torch.stack(prompts, 0)



        prompts = [p[p[:, 0] < SEG_RES + 1] for p in prompts]
        prompts = [p[p[:, 0] > 0] for p in prompts]
        midi_seq = [torch.concat([midi_seq[j], prompts[j]], 0) for j in range(len(gen_seq))]

        return {
            "midi_pred": midi_seq
        }

    def midi2audio(self, midi_seq, max_audio_len):

        chunk_len = 2
        n_steps = chunk_len * SEG_RES
        hop_len = 1

        audio_prompt = None
        audio_seq = None

        for audio_idx in range(0, max_audio_len, hop_len):
            midi_prompt, midi_index = crop_midi_and_align(midi_seq, start_idx=audio_idx, min_chunk_len=chunk_len)
            print(f"[Total len {max_audio_len * SEG_RES}] decode chunk {audio_idx} to {audio_idx + n_steps}...")

            results = self.models["mel2vocals"].inference(
                seqs={"audio_prompt": audio_prompt,
                      "midi_prompt": midi_seq,
                      "audio_index": None,
                      "midi_index": midi_index,
                      "audio_top_k": 100,
                      "midi_top_k": 2,
                      "audio_max_gen_len": n_steps,
                      "midi_max_gen_len": None,
                      "num_samples": len(midi_seq),
                      "audio_walk_steps": [1] * n_steps,
                      "midi_walk_steps": [0] * n_steps,
                      "device": "cuda"},
                n_steps=n_steps)

            audio_pred = results["audio_pred"]

            audio_seq = torch.concat([audio_seq, audio_pred[:, :hop_len * SEG_RES]],
                                     1) if audio_seq is not None else audio_pred[:, :hop_len * SEG_RES]
            audio_prompt = audio_pred[:, hop_len * SEG_RES:]
            break

        audio_seq = torch.concat([audio_seq, audio_prompt], 1)

        return {
            "audio_seq": audio_seq
        }
