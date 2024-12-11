import torch
from .shoelace import Yingyang


class InferenceHelper():
    def __init__(self, s2a_id, a2s_id, is_mono, device):
        s2a_model = Yingyang(is_mono=is_mono)
        s2a_model.load_weights(f"exp/mono_shoelace_real/latest_{s2a_id}.pth")
        s2a_model = s2a_model.to(device)
        s2a_model.set_config(device)
        s2a_model.eval()

        # a2s_model = Yingyang(is_mono=is_mono, mode="audio2symbolic")
        # a2s_model.load_weights(f"exp/mono_shoelace_a2s/latest_{a2s_id}.pth")
        # a2s_model = a2s_model.to(device)
        # a2s_model.set_config(device)
        # a2s_model.eval()
        a2s_model = s2a_model

        self.s2a_model = s2a_model
        self.a2s_model = a2s_model

    def inference(self, midi_seq, audio_seq, desc, mode):
        if mode == "audio2symbolic":
            midi_top_k = 2
            midi_pred = self.a2s_model.sholace.a2s(audio_seq=audio_seq,
                                                   desc=desc,
                                                   midi_top_k=midi_top_k)
            return {
                "audio_pred": None,
                "midi_pred": midi_pred,
            }

        elif mode == "symbolic2audio":
            audio_top_k = 25
            audio_pred = self.s2a_model.sholace.s2a(midi_seq=midi_seq,
                                                    desc=desc,
                                                    audio_top_k=audio_top_k)
            return {
                "audio_pred": audio_pred,
                "midi_pred": None,
            }

        elif mode == "walk":
            midi_top_k = 2
            audio_top_k = 25
            print("here.............................")
            audio_pred, midi_pred = self.s2a_model.sholace.walk(
                midi_model=self.a2s_model.sholace,
                midi_seq=midi_seq, audio_seq=audio_seq,
                midi_top_k=midi_top_k, audio_top_k=audio_top_k,
                desc=desc, prompt_len=20)
            return {
                "audio_pred": audio_pred,
                "midi_pred": midi_pred,
            }
