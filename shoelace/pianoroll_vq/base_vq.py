import torch
import torch.nn as nn
import torch.nn.functional as F
from .vector_quantize_pytorch.residual_vq import ResidualVQ
from shoelace.utils.network_utils import freeze, init_weights, print_params


def preprare_chords(x, res):
    n, _, t, _ = x.shape
    seg_len, scale = res
    input_x = (x[:, 0] > 0).view(n, t // seg_len, -1, scale, 128).flatten(0, 1).long()
    input_x, _ = torch.max(input_x, dim=2)
    input_x = input_x.unsqueeze(2).transpose(1, 3).float()
    return input_x


def preprare_melody(x, res):
    n, _, t, _ = x.shape
    seg_len, scale = res
    input_x = (x[:, -1] > 0).view(n, t // seg_len, seg_len, 128).flatten(0, 1).long()
    input_x = torch.argmax(input_x, dim=-1)
    input_x, _ = torch.max(input_x, dim=1)
    return input_x


def preprare_onsets(x, res):
    n, _, t, _ = x.shape
    seg_len, scale = res
    input_x = (x[:, 1] > 0).view(n, t // seg_len, -1, scale, 128).flatten(0, 1).long()
    input_x, _ = torch.max(input_x, dim=2)
    input_x = input_x.unsqueeze(2).transpose(1, 3).float()
    return input_x


def preprare_pitch(x, res):
    n, _, t, _ = x.shape
    seg_len, scale = res
    input_x = (x[:, 0] > 0).view(n, t // seg_len, -1, scale, 128).flatten(0, 1).long()
    input_x, _ = torch.max(input_x, dim=2)
    input_x = input_x.unsqueeze(2).transpose(1, 3).float()
    return input_x


def preprare_velocity(x, res):
    n, _, t, _ = x.shape
    seg_len, scale = res
    input_x = x[:, 0].view(n, t // seg_len, -1, scale, 128).flatten(0, 1).long()
    input_x, _ = torch.max(input_x, dim=2)
    input_x = F.one_hot(input_x, 17)[:, :, :, 1:]
    input_x = input_x.transpose(1, 3).transpose(1, 2).float()
    return input_x


def preprare_end(x, scale):
    n, _, t, _ = x.shape
    input_x = x[:, 2].view(n, t // scale, scale, 128).flatten(0, 1).long()
    input_x = input_x.unsqueeze(2).transpose(1, 3).float()
    return input_x


def preprare_border(x, scale):
    n, _, t, _ = x.shape
    input_x = x[:, 0].view(n, t // scale, scale, 128).flatten(0, 1).long()
    pitch = input_x.unsqueeze(2).transpose(1, 3).float()
    border = torch.stack(
        [pitch[..., 0] > 0,
         pitch[..., -1] > 0], -1
    ).float()
    return border


# def preprare_pitch(x, scale):
#     n, t, _ = x.shape
#     input_x = x.view(n, 3, t // scale, scale, 128).transpose(0, 1).flatten(1, 2).long()
#     input_x = input_x.unsqueeze(3).transpose(2, 4)
#     pitch = (input_x[0] > 0).float()
#     onset = input_x[1].float()
#     end = input_x[2].float()
#     border = torch.zeros_like(onset)
#     border[..., 0][pitch[..., 0] > 0] = 1
#     border[..., -1][pitch[..., -1] > 0] = 1
#
#     input_x = torch.concat([pitch, end, border], 2)
#     return input_x


def prepare(mode):
    if mode in ["chords_1_128", "chords_4_64", "chords_16_64", "chords_64_64", "chords"]:
        return preprare_chords
    if mode in ["onset_1_64", "onset_4_64", "onset_16_16", "onset_64_64", "onset", "cond_onset"]:
        return preprare_onsets
    if mode == "border":
        return preprare_border
    if mode in ["cond_pitch", "pitch"]:
        return preprare_pitch
    if mode in ["cond_velocity"]:
        return preprare_velocity


def predict(pred, n):
    outs = []
    for i, p in enumerate(pred):
        p = p.contiguous().view(n, len(p) // n, 128, -1, p.shape[-1])
        p = p.permute(3, 0, 1, 4, 2).flatten(2, 3)
        if i > 0:
            p = torch.sigmoid(p[0])
        else:
            p = torch.argmax(p[0], -1)
            p = F.one_hot(p, 128)
            p[..., 0] = 0
        outs.append(p)
    return outs


class RegularBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), pad=(0, 0),
                 stride=(1, 1), dropout_prob=0., is_transeposed=False, bias=False):
        super(RegularBlock, self).__init__()
        conv = nn.ConvTranspose2d if is_transeposed else nn.Conv2d
        # pad = ((kernel_size[0] - stride[0] + 1) // 2, (kernel_size[1] - stride[1] + 1) // 2)
        self.conv_layer = conv(
            in_channels,
            out_channels,
            kernel_size,
            padding=pad,
            stride=stride,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        self.stride = stride

    def forward(self, x):
        y = self.conv_layer(x)
        y = self.bn(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return F.relu(y)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(LinearBlock, self).__init__()
        self.linear_layer = nn.Linear(in_channels, out_channel, bias=False)
        self.norm_layer = nn.LayerNorm(out_channel)

    def forward(self, x):
        y = self.linear_layer(x)
        y = F.relu(y)
        return self.norm_layer(y)


def add_noise(x):
    noisy_x = x + torch.rand_like(x) * .5
    noisy_x = noisy_x / torch.max(noisy_x)
    r = torch.rand_like(noisy_x)
    noisy_x = torch.where(r > .7, noisy_x, x)
    return noisy_x


def sample_neg(x):
    r = torch.rand_like(x)
    aug_x = torch.where(r > .7, torch.ones_like(x), torch.zeros_like(x))
    return aug_x


def make_encoder_unit(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=4096,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=4096, out_channels=2048,
                     kernel_size=(1, 3)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_decoder_unit(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=4096, is_transeposed=True,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=4096, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_velocity_encoder_unit(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(2, 2)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(3, 3), stride=(2, 2)),
        RegularBlock(in_channels=2048, out_channels=4096,
                     kernel_size=(3, 3), stride=(2, 2)),
        RegularBlock(in_channels=4096, out_channels=2048,
                     kernel_size=(3, 3)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_velocity_decoder_unit(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=4096, is_transeposed=True,
                     kernel_size=(3, 3)),
        RegularBlock(in_channels=4096, out_channels=2048, is_transeposed=True,
                     kernel_size=(3, 3), stride=(2, 2)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(3, 3), stride=(2, 2)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(2, 2)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_chords_encoder(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=2048, out_channels=4096,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=4096, out_channels=2048,
                     kernel_size=(1, 1)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_chords_decoder(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=4096, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=4096, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 1), stride=(1, 1)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(1, 1), stride=(1, 1)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_chords_1_encoder(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=2048, out_channels=4096,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=4096, out_channels=2048,
                     kernel_size=(1, 1)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_chords_1_decoder(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=4096, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=4096, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_chords_4_encoder(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=4096,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=4096, out_channels=2048,
                     kernel_size=(1, 1)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_chords_4_decoder(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=4096, is_transeposed=True,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=4096, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_chords_16_encoder(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=4096,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=4096, out_channels=2048,
                     kernel_size=(1, 3)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_chords_16_decoder(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=4096, is_transeposed=True,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=4096, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_chords_64_encoder(hidden_size, in_channels=128):
    return nn.Sequential(
        RegularBlock(in_channels=in_channels, out_channels=512,
                     kernel_size=(1, 1)),
        RegularBlock(in_channels=512, out_channels=1024,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=1024, out_channels=2048,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=2048, out_channels=2048,
                     kernel_size=(1, 2)),
        nn.Conv2d(2048, hidden_size, kernel_size=(1, 1), bias=False)

    )


def make_chords_64_decoder(hidden_size, out_channels=128):
    return nn.Sequential(
        nn.Conv2d(hidden_size, 2048, kernel_size=(1, 1), bias=False),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=2048, is_transeposed=True,
                     kernel_size=(1, 3), stride=(1, 2)),
        RegularBlock(in_channels=2048, out_channels=1024, is_transeposed=True,
                     kernel_size=(1, 3)),
        RegularBlock(in_channels=1024, out_channels=512, is_transeposed=True,
                     kernel_size=(1, 2)),
        RegularBlock(in_channels=512, out_channels=512,
                     kernel_size=(1, 1)),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(1, 1), bias=False),
    )


def make_encoder(mode, hidden_size):
    if mode in ["cond_velocity"]:
        return make_velocity_encoder_unit(hidden_size)
    elif mode in ["onset", "cond_onset", "cond_pitch"]:
        return make_encoder_unit(hidden_size)
    elif mode in ["pitch"]:
        return make_encoder_unit(hidden_size)
    elif mode in ["chords", "chords_1_128"]:
        return make_chords_encoder(hidden_size)
    elif mode == "chords_1_128":
        return make_chords_1_encoder(hidden_size)


def make_decoder(mode, hidden_size):
    if mode in ["cond_velocity"]:
        return make_velocity_decoder_unit(hidden_size)
    elif mode in ["onset",  "cond_onset", "cond_pitch"]:
        return make_decoder_unit(hidden_size)
    elif mode in ["pitch"]:
        return make_decoder_unit(hidden_size)
    elif mode in ["chords", "chords_1_128"]:
        return make_chords_decoder(hidden_size)


def make_film_unit(channels, hidden_size, bias):
    model = nn.ModuleList(
        [nn.Sequential(nn.Conv2d(hidden_size, 16, kernel_size=(1, 1), bias=False),
                       nn.Conv2d(16, ch, kernel_size=(1, 1), bias=bias)) for ch in channels]
    )
    if bias:
        for layer in model:
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.constant_(sublayer.weight, 0)
                    if sublayer.bias is not None:
                        nn.init.constant_(sublayer.bias, 1)
    return model


def make_encoder_film(hidden_size, bias=False):
    return make_film_unit(hidden_size=hidden_size, channels=[512, 1024, 2048, 4096, 2048], bias=bias)


def make_film(mode, hidden_size):
    if mode in ["cond_onset", "cond_pitch", "cond_velocity"]:
        gamma = make_encoder_film(hidden_size=hidden_size, bias=True)
        beta = make_encoder_film(hidden_size=hidden_size, bias=False)
        latent_project = nn.Linear(hidden_size, hidden_size, bias=False)
        return nn.ModuleList([
            gamma,
            beta,
            latent_project
        ])
    elif mode in ["chords"]:
        gamma = make_film_unit(hidden_size=hidden_size, channels=[512, 1024, 2048, 4096, 2048], bias=True)
        beta = make_film_unit(hidden_size=hidden_size, channels=[512, 1024, 2048, 4096, 2048], bias=False)
        latent_project = nn.Linear(hidden_size, hidden_size, bias=False)
        return nn.ModuleList([
            gamma,
            beta,
            latent_project
        ])
    return None


class BaseRVQ(nn.Module):
    def __init__(self, enc, dec, hidden_size, codebook_size=512, n_codebooks=4,
                 with_cond=False, film=None):
        super().__init__()
        self.rvq = ResidualVQ(dim=hidden_size,
                              num_quantizers=n_codebooks,
                              codebook_dim=hidden_size,
                              codebook_size=codebook_size)

        assert not with_cond or film is not None

        self.encoder = enc
        self.decoder = dec
        self.film_layer = film if with_cond else None

    def encode(self, x, cond=None):
        if self.film_layer is None:
            return self.encoder(x)

        n = len(self.encoder)
        cond = cond[..., None, None]
        gamma = self.film_layer[0]
        beta = self.film_layer[1]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < n - 1:
                x = x * gamma[i](cond) + beta[i](cond)

        return x

    def quantize(self, encoded_x, cond=None):
        x_in = encoded_x.transpose(1, 3).flatten(0, 2).contiguous()
        x_hat, indices, loss = self.rvq(x=x_in)
        if self.film_layer is not None:
            x_hat = x_hat + self.film_layer[2](cond)
        return x_hat, indices, loss

    def decode(self, x, n):
        x = x.view(n, 1, 1, -1).transpose(1, 3).contiguous()

        return self.decoder(x)


class PitchVQ(nn.Module):
    def __init__(self, mode="onset", hidden_size=256, codebook_size=512, n_codebooks=4):
        super().__init__()
        res = {
            "chords_1_128": [128, 128],
            "chords_4_64": [64, 16],
            "chords_16_64": [64, 4],
            "onset_16_16": [16, 1],
            "chords_64_64": [64, 1],
            "cond_onset": [16, 1],
            "chords": [16, 16],
            "cond_pitch": [16, 1],
            "cond_velocity": [16, 1],
        }
        with_cond = {
            "chords_1_128": False,
            "chords_4_64": True,
            "chords_16_64": True,
            "onset_16_16": False,
            "chords_64_64": False,
            "onset": False,
            "pitch": False,
            "cond_onset": True,
            "cond_pitch": True,
            "cond_velocity": True,
            "chords": False,
        }
        self.res = res[mode]
        self.pitch_vq = BaseRVQ(hidden_size=hidden_size,
                                n_codebooks=n_codebooks,
                                codebook_size=codebook_size,
                                enc=make_encoder(mode, hidden_size),
                                dec=make_decoder(mode, hidden_size),
                                with_cond=with_cond[mode],
                                film=make_film(mode, hidden_size))
        # if with_cond[mode]:
        #     self.melody = nn.Embedding(128, 256)
        self.prepare_fn = prepare(mode)
        loss_mul = {
            "chords_1_64": 1.,
            "chords_4_64": 1.,
            "chords_16_64": 1.,
            "onset_16_16": 1.,
            "chords_64_64": 1.,
            "chords": 1.,
            "onset": 1.,
            "pitch": 1.,
            "cond_onset": 1.,
            "cond_pitch": 1,
            "cond_velocity": 16,
            "chords_1_128": 1,
        }
        loss_func = {
            "chords": "bce",
            "cond_onset": "bce",
            "cond_pitch": "bce"
        }
        self.loss_func = loss_func[mode]
        self.loss_mul = loss_mul[mode]

    def set_config(self, device, path):
        # init_weights(self.vq)
        print_params(self)
        pass

    def encode(self, x, cond=None, prepare_input=True):
        if prepare_input:
            input_x = self.prepare_fn(x, self.res)
        else:
            input_x = x
        # if cond is not None:
        #     melody_cond = self.melody(preprare_melody(x, res=self.res))
        #     cond = melody_cond + cond
        encoded_x = self.pitch_vq.encode(input_x, cond=cond)
        x_hat, indices, loss = self.pitch_vq.quantize(encoded_x, cond=cond)
        return x_hat, indices, loss, input_x

    def get_indices(self, piano_roll, cond, prepare_input=True):
        x_hat, indices, loss, input_x = self.encode(piano_roll, cond, prepare_input=prepare_input)
        n = len(piano_roll)
        return indices.view(n, -1, indices.shape[-1]).contiguous(), x_hat

    def decode_from_indices(self, indices, cond):
        n, b, _ = indices.shape
        indices = indices.flatten(0, 1)
        x_hat = self.pitch_vq.rvq.get_output_from_indices(indices)
        if self.pitch_vq.film_layer is not None:
            x_hat = x_hat + self.pitch_vq.film_layer[2](cond)
        pianoroll_pred = self.pitch_vq.decode(x_hat, n * b)
        return pianoroll_pred, x_hat

    def forward(self, piano_roll, cond=None):
        x_hat, _, loss, input_x = self.encode(piano_roll, cond=cond)
        n, _, c, t = input_x.shape
        pred = self.pitch_vq.decode(x_hat, n)
        if self.loss_func == "bce":
            bc_fn = nn.BCEWithLogitsLoss()
            return bc_fn(pred, input_x) * self.loss_mul, \
               loss[0]
        elif self.loss_func == "ce":
            ce_fn = nn.CrossEntropyLoss()
            pred = pred.squeeze(-1).squeeze(-1)
            input_x = input_x.squeeze(-1).squeeze(-1)
            # print(pred.shape, input_x.shape)
            return ce_fn(pred, torch.argmax(input_x, 1)) * self.loss_mul, \
                   loss[0]

    def save_weights(self, model_path):
        torch.save(self.pitch_vq.state_dict(), model_path)


class MIDIRVQ(nn.Module):
    def __init__(self, modes, main_mode):
        super(MIDIRVQ, self).__init__()
        hidden_size = {"chords": 256,
                       "onset": 256, "pitch": 256,
                       "cond_onset": 256,
                       "cond_pitch": 256,
                       "cond_velocity": 256}
        n_codebooks = {
            "chords": 4, "onset": 4, "pitch": 4, "cond_onset": 4,
            "cond_pitch": 4, "cond_velocity": 4}
        codebook_size = {
                         "chords": 512, "onset": 512, "pitch": 512, "cond_onset": 512,
                         "cond_pitch": 512, "cond_velocity": 512}
        self.rvqs = {}
        self.pre_control = None
        self.controls = nn.ModuleList()
        self.n_codebooks = n_codebooks

        for mode in modes:
            model = PitchVQ(mode=mode,
                            hidden_size=hidden_size[mode],
                            n_codebooks=n_codebooks[mode],
                            codebook_size=codebook_size[mode])
            self.rvqs[mode] = model
            if main_mode == mode:
                self.net = model
            else:
                self.controls.append(model)
        freeze(self.controls)
        self.modes = modes
        self.main_mode = main_mode

    def set_config(self, path_dict, device):
        if "all" in path_dict:
            self.load_state_dict(torch.load(path_dict["all"], map_location="cpu"))
            return
        for mode in path_dict:
            self.rvqs[mode].pitch_vq.load_state_dict(torch.load(path_dict[mode], map_location="cpu"))
            self.rvqs[mode].eval()

    def set_training(self):
        self.net.train()
        for model in self.controls:
            model.eval()

    def get_indices(self, x):
        modes = self.modes
        indices = []
        cond = None
        for mode in modes:
            print("get indices", mode)
            ind, x_hat = self.rvqs[mode].get_indices(x, cond)
            cond = x_hat
            indices.append(ind)
        return torch.concat(indices, -1)

    def decode_from_indices(self, indices):
        modes = self.modes
        predict = []
        cond = None
        n = 0
        for i in range(len(modes)):
            print(modes[i])
            ind = indices[..., n:n + self.n_codebooks[modes[i]]]
            n += self.n_codebooks[modes[i]]
            pred, x_hat = self.rvqs[modes[i]].decode_from_indices(ind, cond=cond)
            cond = x_hat
            predict.append(pred)
        return predict

    def get_controls(self, x, cond=None):
        for i, model in enumerate(self.controls):
            x_hat, _, _, _ = model.encode(x, cond)
            cond = x_hat
        return cond

    def forward(self, piano_roll):
        if len(self.controls) > 0:
            with torch.no_grad():
                cond = self.get_controls(piano_roll).detach()
        else:
            cond = None
        return self.net(piano_roll, cond)

    def save_weights(self, model_path):
        torch.save(self.net.state_dict(), model_path + ".net.pth")
