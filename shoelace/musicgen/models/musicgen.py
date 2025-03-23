import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from shoelace.utils.network_utils import sample
from .loaders import load_compression_model, load_lm_model
from ..utils.autocast import TorchAutocast
from ..data.audio import audio_write
from tqdm import tqdm

PAD = 2048


def preprocess(x, batch_size=None, device=None):
    if x is None:
        return torch.zeros(batch_size, 4, 4).to(device).long() + PAD
    outputs = []
    for i in range(4):
        outputs.append(F.pad(x[..., i], (i + 1, 3 - i), "constant", PAD))
    return torch.stack(outputs, -1)


def postprocess(x):
    outputs = []
    x_len = x.shape[1]

    for i in range(4):
        outputs.append(x[:, i + 1: x_len - (3 - i), i])
    return torch.stack(outputs, -1)


class MusicGen(nn.Module):
    def __init__(self, name: str, device, use_generator: bool = False):
        super().__init__()
        self.compression_model = None
        self.name = name
        cache_dir = os.environ.get('MUSICGEN_ROOT', None)

        self.lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        self.set_use_generator(use_generator)
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

        self.prepare_for_lora()
        self.reset_cache()


    def reset_cache(self, reset_sos=True):
        self.lm.reset_cache()
        if reset_sos:
            self.cache = False

    def set_use_generator(self, flag: bool):
        self.use_generator = flag
        self.lm.set_use_generator(flag)

    def get_cache(self):
        return self.lm.get_cache()

    def prepare_for_lora(self):
        self.lm.init_qkv()


    def yield_forward(self, input_ids, return_loss=True, return_val=True,
                        with_preprocess=True,
                      with_postprocess=True, **kwargs):
        lm = self.lm
        x = preprocess(input_ids) if with_preprocess else input_ids

        with self.autocast:
            pred = yield from lm(sequence=x.transpose(1, 2),
                                 conditions=None)

            if return_loss:
                x = x[:, 1:]
                pred = pred[:, :-1]
                yield nn.CrossEntropyLoss(ignore_index=PAD)(pred.flatten(0, 2), x.flatten())

        if with_postprocess:
            pred = postprocess(pred)
        if return_val:
            return pred
        yield pred


    @torch.no_grad()
    def yield_inference(self, input_ids, max_len, batch_size=None, device=None, 
        last_chunk=True, top_k=250, temperature=1.0):
        """
        Performs inference by generating a sequence step-by-step.
        """
        
        prompt = preprocess(input_ids, batch_size=batch_size, device=device)
        codes = prompt[:, :-3] if max_len > -1 else prompt
        prompt_len = codes.shape[1]
        input_codes = codes
        index = torch.arange(input_codes.shape[1]).to(device).unsqueeze(0)
        
        if input_ids is None:
            initial = True
        else:
            index = F.pad(index[:, :-1], (1, 0), "constant", 0)
            initial = False
            
        max_len = 1 if max_len == -1 else max_len  

        for i in tqdm(range(max_len), initial=prompt_len, desc="Musicgen Inference", total=max_len + prompt_len):
            if self.use_generator:
                yield {
                    "index": index
                }
                logits = yield from self(input_codes, with_preprocess=False, 
                return_loss=False, with_postprocess=False, return_val=True)
            else:
                logits = self(input_codes, with_preprocess=False, return_loss=False, 
                    with_postprocess=False, return_val=False)
            
            next_token = sample(logits[:, -1], top_k_val=top_k)
            index = index[:, -1:] if initial else index[:, -1:] + 1
            initial = False
            if i + prompt_len - 1 < 3 and (prompt[:, prompt_len + i] == PAD).sum() > 0:
                prompt[:, prompt_len + i , :i + 1] = next_token[:, : i + 1]
                codes = prompt[:, :prompt_len + i + 1]
            else:
                codes = torch.concat([codes, next_token[:, None]], 1)
            input_codes = codes[:, -1:]
            if last_chunk:
                break
            
        yield {"output": postprocess(codes)}
        

    def decode(self, input_ids):
        return postprocess(input_ids)
            
    def forward(self, input_ids, return_val=True, **kwargs):
        generator = self.yield_forward(input_ids, return_val=return_val, **kwargs)
        if self.use_generator or return_val:
            return generator
        return next(generator)
        

    @torch.no_grad()
    def inference(self, input_ids, **kwargs):
        generator = self.yield_inference(input_ids=input_ids, **kwargs)
        if self.use_generator:
            return generator
        else:
            return next(generator)["output"]

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
    
    import numpy as np
    # audio_path = "data/pop909_audio/004-Dear Friend/original.mp3"
    model = MusicGen(name="small", device=torch.device("cuda"))
    model.eval()
    # seq = model.load_from_audio(audio_path)
    seq = torch.from_numpy(np.load("encodes.npy")).cuda()
    print(seq.shape)
    # np.save("encodes.npy", seq.cpu().numpy())
    codes = model.inference(seq)
    print(codes.shape)
    print(codes[codes == 2048].sum())
    from shoelace.utils.encodec_utils import save_rvq
    save_rvq(["test"], codes)


