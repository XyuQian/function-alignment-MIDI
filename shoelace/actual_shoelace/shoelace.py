import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from shoelace.utils.network_utils import freeze, print_params

from tqdm import tqdm
from .models import SholaceParam, STOP_ITER
from .config import MODEL_FACTORY, SKIP_LAYERS, RECIPE


def create_mask(a_len, b_len, device, mask_ratio=.7):
    mask_a = torch.zeros([a_len, b_len])
    mask = torch.rand_like(mask_a)
    mask_a[mask < mask_ratio] = float(-torch.inf)
    mask_b = torch.zeros_like(mask_a).transpose(0, 1) + float(-torch.inf)
    mask_a = F.pad(mask_a, (1, 0), "constant", 0)
    mask_b = F.pad(mask_b, (1, 0), "constant", 0)
    return mask_a.to(device), mask_b.to(device)


class Yinyang(nn.Module):
    def __init__(self, mode="vocals2mel", sec=15):
        super().__init__()

        target_recipe = RECIPE[mode]
        models = nn.ModuleList()
        adapters = nn.ModuleList()
        n_skip_layer_pairs = []
        names = []
        for i, m in enumerate(target_recipe["models"]):
            params = MODEL_FACTORY[m]
            model_params = params["model_params"]
            model_params["is_tuned"] = i == 0
            model = params["model"](**model_params)
            if target_recipe["model_weights_path"][i] is not None:
                model.load_weights(target_recipe["model_weights_path"][i])
            models.append(model)
            names.append(m)
            if i == len(target_recipe["models"]) - 1:
                n_skip_layer_pairs.append([-1, -1])
                break
            next_target = target_recipe["models"][i + 1]
            next_param = MODEL_FACTORY[next_target]
            n_skip_layer_pairs.append(SKIP_LAYERS[m + "-" + next_target])
            shoelace = nn.ModuleList()
            multi_factor = params["steps"] / next_param["steps"] if params["steps"] >= next_param["steps"] else \
                next_param["steps"] / params["steps"]
            long_first = params["steps"] >= next_param["steps"]
            shoelace.append(
                SholaceParam(
                    n_layers=model_params["n_layers"],
                    a_embed_dim=params["hidden_size"],
                    b_embed_dim=next_param["hidden_size"],
                    low_rank_dim=params["low_rank_dim"],
                    num_heads=params["n_heads"],
                    multi_factor=multi_factor,
                    long_first=long_first,
                )
            )
            if target_recipe["bi"]:
                shoelace.append(
                    SholaceParam(
                        n_layers=next_param["n_layers"],
                        a_embed_dim=next_param["hidden_size"],
                        b_embed_dim=params["hidden_size"],
                        low_rank_dim=next_param["low_rank_dim"],
                        num_heads=next_param["n_heads"],
                        multi_factor=multi_factor,
                        long_first=not long_first,
                    )
                )
            adapters.append(shoelace)

        self.bi_di = target_recipe["bi"]
        self.models = models
        self.adapters = adapters
        self.names = names
        self.n_skip_layer_pairs = n_skip_layer_pairs
        self.n_layers = MODEL_FACTORY[self.names[0]]["n_layers"]
        self.seq_len = [MODEL_FACTORY[self.names[i]]["seq_len"] for i in range(len(self.models))]
        self.param_list = [MODEL_FACTORY[self.names[i]]["param_list"] for i in range(len(self.models))]
        self.infer_param_list = [MODEL_FACTORY[self.names[i]]["inference_param_list"] for i in range(len(self.models))]
        self.out_params = [MODEL_FACTORY[self.names[i]]["out"] for i in range(len(self.models))]
        self.loss_weight = target_recipe["loss_weight"]
        # print_params(self)

    def set_config(self, device):
        for adapter in self.adapters:
            for ad in adapter:
                ad.set_config(device)
        self.cur_device = device

    def save_weights(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.adapters.state_dict(), os.path.join(folder, "adapters.pth"))
        for i, model in enumerate(self.models):
            model.save_weights(os.path.join(folder, self.names[i]))
            break

    def load_weights(self, folder, device="cpu"):
        self.adapters.load_state_dict(torch.load(os.path.join(folder, "adapters.pth"), map_location=device))
        for i, model in enumerate(self.models):
            model.load_weights(os.path.join(folder, self.names[i]))
            break

    def stitch(self, model_gen, adapters, n_layers, n_skip_layer_pairs,
               masks, auto_cast, index, bi_di=True):
        a_skip_layers, b_skip_layers = n_skip_layer_pairs[0]
        out, query, q, kv_x = None, None, None, None
        if len(adapters) > 0:
            adapter = adapters[0]
            mask_a, mask_b = masks[0]
        else:
            adapter = None
            mask_a, mask_b = None, None

        for i in range(n_layers):
            out, query, q, idx = next(model_gen[0])
            if idx == STOP_ITER:
                return STOP_ITER, None, None
            if idx is not None and len(index) > 1:
                index[1] = idx
            if a_skip_layers > 0 and i % a_skip_layers == 0:
                kv_out, kv_x, kv_q = self.stitch(model_gen[1:],
                                                 adapters[1:],
                                                 n_layers=b_skip_layers,
                                                 n_skip_layer_pairs=n_skip_layer_pairs[1:],
                                                 masks=masks[1:],
                                                 auto_cast=auto_cast[1:],
                                                 index=index[1:],
                                                 bi_di=bi_di)

                if kv_out == STOP_ITER:
                    return STOP_ITER, None, None

                if bi_di:
                    with auto_cast[1]:
                        kv_out[0] = kv_out[0] + next(adapter[1])(q=kv_q,
                                                                 kv_x=query,
                                                                 mask=mask_b,
                                                                 pos_a=index[1],
                                                                 pos_b=index[0])
                    kv_out[1] = i
            if adapter is not None:
                with auto_cast[0]:
                    out[0] = out[0] + next(adapter[0])(q=q,
                                                       kv_x=kv_x,
                                                       mask=mask_a,
                                                       pos_a=index[1],
                                                       pos_b=index[2])

        return out, query, q

    def compute(self, seqs, model_func, param_list, use_mask=True):
        model_gen = []
        unpack_params = []
        seq_len = []
        index = []
        for i, model in enumerate(model_func):
            param_names = param_list[i]
            params = {}
            for n in param_names:
                if str.endswith(n, "index"):
                    index.append(seqs[n])
                else:
                    if str.endswith(n, "seq"):
                        seq_len.append(seqs[n].shape[1])
                    params[n] = seqs[n]
            model_gen.append(model(**params))
            unpack_params.append(params)
        masks = []
        if use_mask:
            for i in range(len(seq_len) - 1):
                mask_a, mask_b = create_mask(seq_len[i] + self.seq_len[i],
                                             seq_len[i + 1] + self.seq_len[i + 1],
                                             device=self.cur_device)
                masks.append([mask_a, mask_b])
        else:
            masks = [[None, None] for i in range(len(self.seq_len))]
        auto_cast = [self.models[i].autocast for i in range(len(self.models))]
        return unpack_params, model_gen, index, masks, auto_cast

    def forward(self, seqs):
        unpack_params, model_gen, index, \
        masks, auto_cast = self.compute(seqs,
                                        model_func=self.models,
                                        param_list=self.param_list)

        adapters = []
        for i in range(len(self.adapters)):
            adapter_fn = []
            for ad in self.adapters[i]:
                adapter_fn.append(ad())
            adapters.append(adapter_fn)

        self.stitch(model_gen=model_gen,
                    adapters=adapters,
                    n_layers=self.n_layers,
                    n_skip_layer_pairs=self.n_skip_layer_pairs,
                    masks=masks,
                    auto_cast=auto_cast,
                    index=[index[0]] + index,
                    bi_di=self.bi_di)

        loss = {}
        for i, model in enumerate(model_gen):
            out = next(model)
            unpack_params[i]["pred"] = out
            loss[self.names[i]] = [self.models[i].loss_func(**unpack_params[i]),
                                   self.loss_weight[i]]
            break
        return loss

    @torch.no_grad()
    def inference(self, seqs, n_steps):
        unpack_params, model_gen, index, \
        masks, auto_cast = self.compute(seqs,
                                        model_func=[model.inference for model in self.models],
                                        param_list=self.infer_param_list,
                                        use_mask=False)

        for i in tqdm(range(n_steps), total=n_steps,
                      desc=f"inference..."):

            adapters = []
            for j in range(len(self.adapters)):
                adapter_fn = []
                for ad in self.adapters[j]:
                    adapter_fn.append(ad())
                adapters.append(adapter_fn)

            sig, _, _ = self.stitch(model_gen=model_gen,
                                    adapters=adapters,
                                    n_layers=self.n_layers,
                                    n_skip_layer_pairs=self.n_skip_layer_pairs,
                                    masks=masks,
                                    index=[index[0]] + index,
                                    auto_cast=auto_cast,
                                    bi_di=self.bi_di)
            if sig == STOP_ITER:
                break

        results = {}
        for i, model in enumerate(model_gen):
            out = next(model)
            results[self.out_params[i]] = out
            break
        return results
