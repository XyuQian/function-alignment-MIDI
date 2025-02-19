# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import math
import random
import re
import typing as tp
import warnings

from einops import rearrange
from num2words import num2words
import spacy
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore
import torchaudio
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


from .transformer import create_sin_embedding
from ..data.audio_dataset import SegmentInfo
from ..utils.autocast import TorchAutocast
from ..utils.utils import hash_trick, length_to_mask, collate

logger = logging.getLogger(__name__)

TextCondition = tp.Optional[str]            # A text condition can be a string or None
ConditionType = tp.Tuple[Tensor, Tensor]    # (condition, mask)


class WavCondition(tp.NamedTuple):
    """Simple container for wave conditioning."""
    wav: Tensor
    length: Tensor
    path: tp.List[tp.Optional[str]] = []


def nullify_condition(condition: ConditionType, dim: int = 1) -> ConditionType:
    """
    Transform an input condition into a null condition by truncating to length=1 along `dim`.
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert type(condition) == tuple and isinstance(condition[0], Tensor) and isinstance(condition[1], Tensor), \
        "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0.0 * out[..., :1]
    out = out.transpose(dim, last_dim)
    new_mask = torch.zeros((B, 1), device=out.device).int()
    return (out, new_mask)


def nullify_wav(wav: Tensor) -> WavCondition:
    """
    Create a nullified WavCondition from a wav tensor of shape [B, T].
    """
    null_wav, _ = nullify_condition((wav, torch.zeros_like(wav)), dim=wav.dim() - 1)
    B = wav.shape[0]
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * B, device=wav.device),
        path=['null_wav'] * B
    )


@dataclass
class ConditioningAttributes:
    """
    Stores text and wav conditions in dictionaries.
    """
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def attributes(self):
        return {"text": self.text_attributes, "wav": self.wav_attributes}

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


class SegmentWithAttributes(SegmentInfo):
    """
    Base class for all dataclasses that are used for conditioning.
    Child classes should implement `to_condition_attributes`.
    """
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()


class Tokenizer:
    """Base tokenizer class."""
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """
    Tokenizer splitting text on whitespace, ignoring punctuation, etc.
    """
    PUNCTUATIONS = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    def __call__(
        self,
        texts: tp.List[tp.Optional[str]],
        return_text: bool = False
    ) -> tp.Union[tp.Tuple[Tensor, Tensor], tp.Tuple[Tensor, Tensor, tp.List[str]]]:
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            if text is None:
                output.append(torch.tensor([self.pad_idx]))
                lengths.append(0)
                continue
            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)
            text_sp = self.nlp(text)
            if self.stopwords:
                text_sp = [w for w in text_sp if not w.is_stop]
            text_sp = [w for w in text_sp if w.text not in self.PUNCTUATIONS]
            text_sp = [getattr(t, "lemma_" if self.lemma else "text") for t in text_sp]
            texts[i] = " ".join(text_sp)
            lengths.append(len(text_sp))
            tokens = torch.tensor([hash_trick(w, self.n_bins) for w in text_sp])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask


class NoopTokenizer(Tokenizer):
    """
    Simple tokenizer that assigns exactly one token per input string (no splitting).
    """
    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[Tensor, Tensor]:
        output, lengths = [], []
        for text in texts:
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)
        tokens = torch.LongTensor(output).unsqueeze(1)
        mask = length_to_mask(torch.IntTensor(lengths)).int()
        return tokens, mask


class BaseConditioner(nn.Module):
    """
    Base model for all conditioner modules. Must implement:
      - tokenize()
      - forward() returning a ConditionType (tensor + mask).
    """

    def __init__(self, dim, output_dim):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        raise NotImplementedError()


class TextConditioner(BaseConditioner):
    """Marker class for text-based conditioners."""
    ...


class LUTConditioner(TextConditioner):
    """
    Lookup table TextConditioner with `n_bins`, embedding dimension, etc.
    """
    def __init__(self, n_bins: int, dim: int, output_dim: int, tokenizer: str, pad_idx: int = 0):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_bins, dim)
        if tokenizer == "whitespace":
            self.tokenizer = WhiteSpaceTokenizer(n_bins, pad_idx=pad_idx)
        elif tokenizer == "noop":
            self.tokenizer = NoopTokenizer(n_bins, pad_idx=pad_idx)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        return tokens.to(device), mask.to(device)

    def forward(self, inputs: tp.Tuple[torch.Tensor, torch.Tensor]) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        embeds = self.output_proj(embeds)
        return (embeds * mask.unsqueeze(-1)), mask


class T5Conditioner(TextConditioner):
    """
    T5-based text conditioner. Uses T5EncoderModel for token -> embedding.
    """

    MODELS = [
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
        "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
        "google/flan-t5-xl", "google/flan-t5-xxl"
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-xl": 1024,
        "google/flan-t5-xxl": 1024,
    }

    def __init__(
        self,
        name: str,
        output_dim: int,
        finetune: bool,
        device: str,
        autocast_dtype: tp.Optional[str] = 'float32',
        word_dropout: float = 0.0,
        normalize_text: bool = False
    ):
        assert name in self.MODELS, f"unrecognized t5 model name, expected one of {self.MODELS}"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout

        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)

        if finetune:
            self.t5 = t5
        else:
            # not saved to checkpoint
            self.__dict__["t5"] = t5.to(device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])
        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"]
        mask[empty_idx, :] = 0
        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class WaveformConditioner(BaseConditioner):
    """
    Base class for waveform-based conditioners. Subclasses must implement:
      - _get_wav_embedding(wav)
      - _downsampling_factor()
    """

    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device

    def tokenize(self, wav_length: WavCondition) -> WavCondition:
        wav, length, path = wav_length
        return WavCondition(wav.to(self.device), length.to(self.device), path)

    def _get_wav_embedding(self, wav: Tensor) -> Tensor:
        raise NotImplementedError()

    def _downsampling_factor(self) -> int:
        raise NotImplementedError()

    def forward(self, inputs: WavCondition) -> ConditionType:
        wav, lengths, path = inputs
        with torch.no_grad():
            embeds = self._get_wav_embedding(wav)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()
        else:
            mask = torch.ones(embeds.shape[:2], dtype=torch.int, device=embeds.device)
        embeds = (embeds * mask.unsqueeze(2).to(self.device))
        return embeds, mask


class ChromaStemConditioner(WaveformConditioner):
    """
    An example WaveformConditioner that extracts a 'chroma' feature from a DEMUCS-filtered waveform.
    """

    def __init__(
        self,
        output_dim: int,
        sample_rate: int,
        n_chroma: int,
        radix2_exp: int,
        duration: float,
        match_len_on_eval: bool = True,
        eval_wavs: tp.Optional[str] = None,
        n_eval_wavs: int = 0,
        device: tp.Union[torch.device, str] = "cpu",
        **kwargs
    ):
        from demucs import pretrained
        super().__init__(dim=n_chroma, output_dim=output_dim, device=device)
        self.autocast = TorchAutocast(enabled=(device != "cpu"), device_type=device, dtype=torch.float32)
        self.sample_rate = sample_rate
        self.match_len_on_eval = match_len_on_eval
        self.duration = duration

        self.__dict__["demucs"] = pretrained.get_model('htdemucs').to(device)
        self.stem2idx = {'drums': 0, 'bass': 1, 'other': 2, 'vocal': 3}
        self.stem_idx = torch.LongTensor([self.stem2idx['vocal'], self.stem2idx['other']]).to(device)
        self.chroma = ChromaExtractor(sample_rate=sample_rate, n_chroma=n_chroma, radix2_exp=radix2_exp,
                                      device=device, **kwargs)
        self.chroma_len = self._get_chroma_len()

    def _downsampling_factor(self):
        return self.chroma.winhop

    def _get_chroma_len(self):
        dummy_wav = torch.zeros((1, self.sample_rate * self.duration), device=self.device)
        dummy_chr = self.chroma(dummy_wav)
        return dummy_chr.shape[1]

    @torch.no_grad()
    def _get_filtered_wav(self, wav):
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = convert_audio(wav, self.sample_rate, self.demucs.samplerate, self.demucs.audio_channels)
            stems = apply_model(self.demucs, wav, device=self.device)
            stems = stems[:, self.stem_idx]
            stems = stems.sum(1)
            stems = stems.mean(1, keepdim=True)
            stems = convert_audio(stems, self.demucs.samplerate, self.sample_rate, 1)
            return stems

    @torch.no_grad()
    def _get_wav_embedding(self, wav):
        if wav.shape[-1] == 1:
            return self.chroma(wav)

        stems = self._get_filtered_wav(wav)
        chroma = self.chroma(stems)

        if self.match_len_on_eval:
            b, t, c = chroma.shape
            if t > self.chroma_len:
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f'chroma truncated from {t} to {chroma.shape[1]}')
            elif t < self.chroma_len:
                n_repeat = int(math.ceil(self.chroma_len / t))
                chroma = chroma.repeat(1, n_repeat, 1)
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f'chroma extended from {t} to {chroma.shape[1]}')
        return chroma


class ChromaExtractor(nn.Module):
    """
    A helper class for computing chroma from audio using librosa-based filters.
    """

    def __init__(
        self,
        sample_rate: int,
        n_chroma: int = 12,
        radix2_exp: int = 12,
        nfft: tp.Optional[int] = None,
        winlen: tp.Optional[int] = None,
        winhop: tp.Optional[int] = None,
        argmax: bool = False,
        norm: float = torch.inf,
        device: tp.Union[torch.device, str] = "cpu"
    ):
        super().__init__()
        from librosa import filters
        self.device = device
        self.autocast = TorchAutocast(enabled=(device != "cpu"), device_type=device, dtype=torch.float32)
        self.winlen = winlen or 2 ** radix2_exp
        self.nfft = nfft or self.winlen
        self.winhop = winhop or (self.winlen // 4)
        self.sr = sample_rate
        self.n_chroma = n_chroma
        self.norm = norm
        self.argmax = argmax
        self.window = torch.hann_window(self.winlen).to(device)
        self.fbanks = torch.from_numpy(
            filters.chroma(sr=sample_rate, n_fft=self.nfft, tuning=0, n_chroma=self.n_chroma)
        ).to(device)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft, win_length=self.winlen, hop_length=self.winhop,
            power=2, center=True, pad=0, normalized=True
        ).to(device)

    def forward(self, wav):
        with self.autocast:
            T = wav.shape[-1]
            if T < self.nfft:
                pad = self.nfft - T
                r = 0 if pad % 2 == 0 else 1
                wav = F.pad(wav, (pad // 2, pad // 2 + r), 'constant', 0)

            spec = self.spec(wav).squeeze(1)
            raw_chroma = torch.einsum("cf,...ft->...ct", self.fbanks, spec)
            norm_chroma = F.normalize(raw_chroma, p=self.norm, dim=-2, eps=1e-6)
            norm_chroma = rearrange(norm_chroma, "b d t -> b t d")

            if self.argmax:
                idx = norm_chroma.argmax(-1, keepdim=True)
                norm_chroma[:] = 0
                norm_chroma.scatter_(dim=-1, index=idx, value=1)
            return norm_chroma


def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str):
    """
    Nullify an attribute named `condition` in the `sample` object, either text or wav.
    Works in-place.
    """
    if condition_type not in ["text", "wav"]:
        raise ValueError(f"Expected condition type 'wav' or 'text', got '{condition_type}'")

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            f"dropout_condition: not found '{condition}' in {condition_type}."
            f"\nAvailable: {getattr(sample, condition_type).keys()}"
        )

    if condition_type == "wav":
        wav, length, path = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav)
    else:
        sample.text[condition] = None
    return sample


class DropoutModule(nn.Module):
    """
    Base class for dropout modules that require a local RNG.
    """
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """
    Applies dropout with a given probability per attribute independently.
    E.g. `p={"wav": {"self_wav": 0.2}, "text": {"genre": 0.5}}` means
    there's a 20% chance to drop 'self_wav' and a 50% chance to drop 'genre'.
    """

    def __init__(
        self,
        p: tp.Dict[str, tp.Dict[str, float]],
        active_on_eval: bool = False,
        seed: int = 1234
    ):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        self.p = {}
        for cond_type, probs in p.items():
            self.p[cond_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():
            for condition, drop_prob in ps.items():
                if torch.rand(1, generator=self.rng).item() < drop_prob:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


class ClassifierFreeGuidanceDropout(DropoutModule):
    """
    Applies CFG dropout: either drop all attributes or none, with probability p.
    """

    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        if not self.training:
            return samples

        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        samples = deepcopy(samples)
        for condition_type in ["wav", "text"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class ConditioningProvider(nn.Module):
    """
    Gathers multiple BaseConditioners, tokenizes & forwards them to produce final ConditionTensors.
    """

    def __init__(
        self,
        conditioners: tp.Dict[str, BaseConditioner],
        merge_text_conditions_p: float = 0.0,
        drop_desc_p: float = 0.0,
        device: tp.Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device
        self.merge_text_conditions_p = merge_text_conditions_p
        self.drop_desc_p = drop_desc_p
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def wav_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, WaveformConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        """
        Convert the raw text/wav fields in each ConditioningAttributes into
        tokenized data for each relevant BaseConditioner.
        """
        assert all(isinstance(x, ConditioningAttributes) for x in inputs), \
            "Expected a list of ConditioningAttributes!"

        out = {}
        text = self._collate_text(inputs)
        wavs = self._collate_wavs(inputs)
        all_keys = set(text.keys()) | set(wavs.keys())
        assert all_keys.issubset(self.conditioners.keys()), \
            f"Unexpected keys {all_keys} not in {self.conditioners.keys()}"

        for attribute, batch in chain(text.items(), wavs.items()):
            out[attribute] = self.conditioners[attribute].tokenize(batch)
        return out

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        out = {}
        for attribute, inputs in tokenized.items():
            c, m = self.conditioners[attribute](inputs)
            out[attribute] = (c, m)
        return out

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        def _merge_conds(cond: ConditioningAttributes, merge_p=0, drop_desc_p=0):
            def is_valid(k, v):
                valid_keys = ['key', 'bpm', 'genre', 'moods', 'instrument']
                return (k in valid_keys) and (v is not None) and isinstance(v, (int, float, str, list))

            def process_value(v):
                if isinstance(v, (int, float, str)):
                    return str(v)
                if isinstance(v, list):
                    return ", ".join(map(str, v))
                raise RuntimeError(f"unknown type for text value! ({type(v), v})")

            desc = cond.text.get('description', None)
            if random.uniform(0, 1) < merge_p:
                meta_pairs = [f'{k}: {process_value(v)}' for k, v in cond.text.items() if is_valid(k, v)]
                random.shuffle(meta_pairs)
                meta_data = ". ".join(meta_pairs)
                if random.uniform(0, 1) < drop_desc_p:
                    desc = None
                if desc is None:
                    desc = meta_data if meta_data else None
                else:
                    desc = desc.rstrip('.') + ". " + meta_data
                cond.text['description'] = desc.strip() if desc else None

        if self.training and self.merge_text_conditions_p > 0:
            for sample in samples:
                _merge_conds(sample, self.merge_text_conditions_p, self.drop_desc_p)

        batch_per_attribute: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        for sample in samples:
            for condition in self.text_conditions:
                batch_per_attribute[condition].append(sample.text.get(condition, None))
        return batch_per_attribute

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]):
        wavs = defaultdict(list)
        lens = defaultdict(list)
        paths = defaultdict(list)
        out = {}

        for sample in samples:
            for attr in self.wav_conditions:
                wv, length, path = sample.wav[attr]
                wavs[attr].append(wv.flatten())
                lens[attr].append(length)
                paths[attr].append(path)

        for attr in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attr], dim=0)
            out[attr] = WavCondition(
                wav=stacked_wav.unsqueeze(1),
                length=torch.cat(lens[attr]),
                path=paths[attr]
            )
        return out


class ConditionFuser(nn.Module):
    """
    Condition fuser that merges ConditionTensors into a model's input.
    We no longer do any streaming or caching logic.
    """

    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(
        self,
        fuse2cond: tp.Dict[str, tp.List[str]],
        cross_attention_pos_emb: bool = False,
        cross_attention_pos_emb_scale: float = 1.0
    ):
        super().__init__()
        # e.g. fuse2cond = {"sum": ["genre"], "cross": ["description"], "prepend": ["artist"]}
        # cond2fuse = {"genre":"sum", "description":"cross", "artist":"prepend"}
        assert all(k in self.FUSING_METHODS for k in fuse2cond.keys()), \
            f"Invalid fuse method, must be in {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition_name in conditions:
                self.cond2fuse[condition_name] = fuse_method

    def forward(
        self,
        model_input: Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        """
        Apply each condition to `model_input` using the specified fuse method.
        Returns (updated_input, cross_attention_input).
        """
        cross_attention_output = None

        for cond_name, (cond_tensor, cond_mask) in conditions.items():
            fuse_method = self.cond2fuse[cond_name]
            if fuse_method == "sum":
                model_input += cond_tensor
            elif fuse_method == "input_interpolate":
                # Interpolate cond_tensor to match model_input time steps
                cond_tensor = rearrange(cond_tensor, "b t d -> b d t")
                cond_tensor = F.interpolate(cond_tensor, size=model_input.shape[1])
                cond_tensor = rearrange(cond_tensor, "b d t -> b t d")
                model_input += cond_tensor
            elif fuse_method == "prepend":
                # Insert condition at the beginning (only once, if multiple calls)
                model_input = torch.cat([cond_tensor, model_input], dim=1)
            elif fuse_method == "cross":
                if cross_attention_output is None:
                    cross_attention_output = cond_tensor
                else:
                    cross_attention_output = torch.cat([cross_attention_output, cond_tensor], dim=1)
            else:
                raise ValueError(f"Unknown fuse method: {fuse_method}")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1], device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        return model_input, cross_attention_output
