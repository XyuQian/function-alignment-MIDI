import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP
from ..utils.autocast import TorchAutocast
from ..data.audio import audio_write

PAD = 2048


def preprocess(x):
    outputs = []
    for i in range(4):
        outputs.append(F.pad(x[..., i], (4 - i, i), "constant", PAD))
    return torch.stack(outputs, -1)


def postprocess(x):
    outputs = []
    x_len = x.shape[1]
    for i in range(4):
        outputs.append(x[..., 4 - i: x_len - i, i, :])
    return torch.stack(outputs, -2)


class MusicGen(nn.Module):
    def __init__(self, name: str, device):
        super().__init__()
        self.compression_model = None
        self.name = name
        cache_dir = os.environ.get('MUSICGEN_ROOT', None)

        self.lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        if device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=device.type, dtype=torch.float16
            )

        self.compression_model_config = {
            "file_or_url_or_id": name,
            "device": device,
            "cache_dir": cache_dir
        }

    def prepare_for_lora(self):
        self.lm.init_qkv()

    def postprocess(self, audio_seq):
        pass

    def forward(self, input_ids, with_preprocess=True,
                with_postprocess=True, return_loss=True):
        lm = self.lm
        x = preprocess(input_ids) if with_preprocess else input_ids

        with self.autocast:
            pred = lm(sequence=x.transpose(1, 2),
                      conditions=None)

            if return_loss:
                return nn.CrossEntropyLoss(ignore_index=PAD)(pred.flatten(0, 2), x.flatten())
        if with_postprocess:
            pred = postprocess(pred)
        return pred

    def load_compression_model(self):
        if self.compression_model is None:
            self.compression_model = load_compression_model(**self.compression_model_config)
        return self.compression_model_config["device"]

    @torch.no_grad()
    def load_from_audio(self, path: str):
        device = self.load_compression_model()
        x, _ = librosa.load(path, sr=32000)
        x = torch.from_numpy(x)[None, None, ...]
        seq, _ = self.compression_model.encode(x.to(device))
        return seq.transpose(1, 2)

    @torch.no_grad()
    def decode_audio(self, seq, path_list):
        device = self.load_compression_model()
        assert len(seq) == len(path_list)
        gen_audio = self.compression_model.decode(seq.to(device), None)
        for i, path in enumerate(path_list):
            pred_wav = gen_audio[i].cpu()
            if pred_wav.dtype == torch.float16:
                pred_wav = pred_wav.to(torch.float32)
            audio_write(path, pred_wav, 32000, strategy="loudness", loudness_compressor=True)


if __name__ == "__main__":
    audio_path = "data/pop909_audio/004-Dear Friend/original.mp3"
    model = MusicGen(name="large", device="cuda")
    model.prepare_for_lora()
    model.eval()
    seq = model.load_from_audio(audio_path)
    seq = seq[:, 50*10:20*50]
    print(seq.shape)
    with torch.no_grad():
        y = model(seq, return_loss=False)
        codes = torch.argmax(y, -1)
        print(codes.shape)
        codes = codes.transpose(1, 2)
    model.decode_audio(codes, ["test_audio"])
