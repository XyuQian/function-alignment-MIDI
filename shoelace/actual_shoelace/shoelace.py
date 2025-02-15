import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from shoelace.utils.network_utils import freeze, print_params
from .cross_attention import SholaceParam, STOP_ITER
from .config import MODEL_FACTORY, SKIP_LAYERS, RECIPE


def create_mask(a_len: int, b_len: int, device: torch.device, mask_ratio: float = 0.7):
    mask = torch.rand(a_len, b_len, device=device)
    mask_a = torch.where(mask < mask_ratio, float('-inf'), 0.0)
    mask_b = torch.full_like(mask_a.T, float('-inf'))
    return F.pad(mask_a, (1, 0)), F.pad(mask_b, (1, 0))


class Yinyang(nn.Module):
    def __init__(self, mode: str = "vocals2mel", sec: int = 15):
        super().__init__()
        target_recipe = RECIPE[mode]
        self.bi_di = target_recipe["bi"]
        self.loss_weight = target_recipe["loss_weight"]
        self.names = target_recipe["models"]

        self.models, self.adapters, self.n_skip_layer_pairs = self._initialize_models(target_recipe)
        self.n_layers = MODEL_FACTORY[self.names[0]]["model_params"]["n_layers"]
        self.seq_len = [MODEL_FACTORY[name]["seq_len"] for name in self.names]
        self.param_list = [MODEL_FACTORY[name]["param_list"] for name in self.names]
        self.infer_param_list = [MODEL_FACTORY[name]["inference_param_list"] for name in self.names]
        self.out_params = [MODEL_FACTORY[name]["out"] for name in self.names]

    def _initialize_models(self, target_recipe):
        models, adapters, n_skip_layer_pairs = nn.ModuleList(), nn.ModuleList(), []
        for i, model_name in enumerate(self.names):
            params = MODEL_FACTORY[model_name]
            model = params["model"](**params["model_params"], is_tuned=(i == 0))
            if target_recipe["model_weights_path"][i]:
                model.load_weights(target_recipe["model_weights_path"][i])
            models.append(model)
            if i < len(self.names) - 1:
                next_params = MODEL_FACTORY[self.names[i + 1]]
                n_skip_layer_pairs.append(SKIP_LAYERS[f"{model_name}-{self.names[i + 1]}"])
                adapters.append(self._create_adapter(params, next_params))
            else:
                n_skip_layer_pairs.append([-1, -1])
        return models, adapters, n_skip_layer_pairs

    def _create_adapter(self, params, next_params):
        multi_factor = max(params["steps"], next_params["steps"]) / min(params["steps"], next_params["steps"])
        long_first = params["steps"] >= next_params["steps"]
        shoelace = [SholaceParam(params["model_params"]["n_layers"], params["hidden_size"],
                                 next_params["hidden_size"], params["low_rank_dim"], params["n_heads"],
                                 multi_factor, long_first)]
        if self.bi_di:
            shoelace.append(SholaceParam(next_params["n_layers"], next_params["hidden_size"],
                                         params["hidden_size"], next_params["low_rank_dim"],
                                         next_params["n_heads"], multi_factor, not long_first))
        return nn.ModuleList(shoelace)

    def save_weights(self, folder):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.adapters.state_dict(), os.path.join(folder, "adapters.pth"))
        self.models[0].save_weights(os.path.join(folder, self.names[0]))

    def load_weights(self, folder, device="cpu"):
        self.adapters.load_state_dict(torch.load(os.path.join(folder, "adapters.pth"), map_location=device))
        self.models[0].load_weights(os.path.join(folder, self.names[0]))

    def forward(self, seqs):
        unpack_params, model_gen, index, masks, auto_cast = self._prepare_computation(seqs)
        adapters = [[ad() for ad in adapter] for adapter in self.adapters]
        self._stitch(model_gen, adapters, index, masks, auto_cast)
        return {self.names[i]: [self.models[i].loss_func(**unpack_params[i]), self.loss_weight[i]] for i in range(1)}

    @torch.no_grad()
    def inference(self, seqs, n_steps):
        unpack_params, model_gen, index, masks, auto_cast = self._prepare_computation(seqs, inference=True)
        for _ in tqdm(range(n_steps), desc="inference..."):
            adapters = [[ad() for ad in adapter] for adapter in self.adapters]
            sig, _, _ = self._stitch(model_gen, adapters, index, masks, auto_cast)
            if sig == STOP_ITER:
                break
        return {self.out_params[i]: next(model_gen[i]) for i in range(1)}

    def _prepare_computation(self, seqs, inference=False):
        model_func = [model.inference if inference else model for model in self.models]
        model_gen, unpack_params, seq_len, index = [], [], [], []
        for i, model in enumerate(model_func):
            params = {n: seqs[n] for n in self.param_list[i]}
            model_gen.append(model(**params))
            unpack_params.append(params)
            seq_len.append(seqs.get(self.param_list[i][0], torch.zeros(1)).shape[1])
        masks = [create_mask(seq_len[i], seq_len[i + 1], self.cur_device) for i in range(len(seq_len) - 1)]
        auto_cast = [model.autocast for model in self.models]
        return unpack_params, model_gen, index, masks, auto_cast

    def _stitch(self, model_gen, adapters, index, masks, auto_cast):
        for i in range(self.n_layers):
            out, query, q, idx = next(model_gen[0])
            if idx == STOP_ITER:
                return STOP_ITER, None, None
            if len(adapters) > 0:
                kv_out, kv_x, kv_q = self._stitch(model_gen[1:], adapters[1:], index[1:], masks[1:], auto_cast[1:])
                if kv_out == STOP_ITER:
                    return STOP_ITER, None, None
                if self.bi_di:
                    kv_out[0] += next(adapters[0][1])(q=kv_q, kv_x=query, mask=masks[0][1], pos_a=index[1], pos_b=index[0])
            out[0] += next(adapters[0][0])(q=q, kv_x=kv_x, mask=masks[0][0], pos_a=index[1], pos_b=index[2])
        return out, query, q
