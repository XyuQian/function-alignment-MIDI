import torch
from shoelace.pianoroll_vq.base_vq import MIDIRVQ as rvq_model
from shoelace.pianoroll_vq.base_vq import predict
from shoelace.pianorollLM.pianoroll_lm_with_linear import PianoRollLM as lm_model


class InferenceHelper():
    def __init__(self, device):
        rvq = rvq_model(modes=[
            "chords",
            "cond_onset", "cond_pitch"

        ], main_mode="melody").to(device)
        rvq.set_config(path_dict={"chords": "save_models/chords.pth",
                                  "cond_onset": "save_models/cond_onset.pth",
                                  "cond_pitch": "save_models/cond_pitch.pth",
                                  # "cond_pitch": "save_models/cond_pitch.pth"
                                  }, device=device)
        rvq.eval()
        lm = lm_model()
        lm.load_state_dict(torch.load("save_models/llm_pop_vocals_2.pth", map_location="cpu"))
        lm = lm.to(device)
        lm.set_config(device)
        lm.eval()

        self.rvq = rvq
        self.lm = lm

    def refine_fn(self, tokens, activation):
        return tokens, activation
        rvq = self.rvq
        pred = rvq.decode_from_indices(tokens)
        pred = predict(pred, len(activation[0]))
        onset = pred[1] > .5
        pitch = pred[2] > .5
        data = []
        for i in range(16):
            cur = pitch[:, i] * ((activation[0] + activation[1] + onset[:, i]) > 0.)
            data.append(
                torch.stack([cur, cur * onset[:, i]], 1)
            )
            activation[0] = cur
            activation[1] = activation[0]

        data = torch.stack(data, 2)
        data = torch.concat([data, data[:, :1]], 1)
        indices = rvq.get_indices(data)
        return indices, activation

    def refine_fn_fn(self, decoded_sequence, activation):
        decoded_sequence = activation
        return decoded_sequence
        rvq = self.rvq
        if mode == 1:
            chords, x_hat = rvq.rvqs["chords"].decode_from_indices(decoded_sequence[:, None, :4], cond=None)
            chords = (torch.sigmoid(chords) > 0.5).float()

            onset, _ = rvq.rvqs["cond_onset"].decode_from_indices(decoded_sequence[:, None, 4:8], cond=x_hat)
            onset = (torch.sigmoid(onset) > 0.5).float()

            indices, _ = rvq.rvqs["cond_onset"].get_indices(onset, cond=x_hat, prepare_input=False)
            return torch.concat([decoded_sequence[:, :4], indices.squeeze(1)], -1)

        if mode == 2:
            chords, x_hat = rvq.rvqs["chords"].decode_from_indices(decoded_sequence[:, None, :4], cond=None)
            chords = (torch.sigmoid(chords) > 0.5).float()

            onset, x_hat = rvq.rvqs["cond_onset"].decode_from_indices(decoded_sequence[:, None, 4:8], cond=x_hat)
            onset = (torch.sigmoid(onset) > 0.5).float()

            pitch, _ = rvq.rvqs["cond_pitch"].decode_from_indices(decoded_sequence[:, None, 8:], cond=x_hat)
            pitch = (torch.sigmoid(pitch) > 0.5).float()

            indices, _ = rvq.rvqs["cond_pitch"].get_indices(pitch, cond=x_hat, prepare_input=False)
            return torch.concat([decoded_sequence[:, :8], indices.squeeze(1)], -1)

        return decoded_sequence

    def reconstruct(self, data):
        with torch.no_grad():
            indices = self.rvq.get_indices(data)
            pred = self.rvq.decode_from_indices(indices)
            out = predict(pred, n=len(data))
        return out

    def inference(self, data, melody, chords, max_len=100):
        rvq = self.rvq
        lm = self.lm
        activation = [data[:, 0, 15 * 16] > 0, data[:, 0, 15 * 16] > 0]

        with torch.no_grad():
            indices = rvq.get_indices(data)
            prompt = indices[:, :15]
            melody = melody[:, :15]
            chords = chords[:, :15]
            predicted_indices, pred_melody = lm.inference(prompt,
                                                          melody=melody,
                                                          chords=chords,
                                                          top_k=64, temperature=1.,
                                                          activation=activation,
                                                          refine_fn=self.refine_fn,
                                                          max_len=max_len)
            total_tokens = torch.concat([prompt, predicted_indices], 1)
            pred_melody = torch.concat([melody, pred_melody], 1)
            pred = rvq.decode_from_indices(total_tokens)
            out = predict(pred, n=len(data))
        return out, pred_melody
